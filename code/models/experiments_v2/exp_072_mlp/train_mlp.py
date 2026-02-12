"""
exp_072: Simple MLP with MAE Loss
- Tabular data에 적합한 간단한 MLP
- 강한 정규화로 과적합 방지
- CatBoost 앙상블 재료로 활용
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"

# GPU 사용 가능하면 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MLP(nn.Module):
    """Simple MLP for regression"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=2, dropout=0.3):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_features(df):
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    ema_span = 2
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)

    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    return df


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, weight_decay=0.01, patience=20):
    """Train MLP with early stopping"""
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_errors = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)

                # Euclidean distance
                errors = torch.sqrt((pred[:, 0] - y_batch[:, 0])**2 + (pred[:, 1] - y_batch[:, 1])**2)
                val_errors.extend(errors.cpu().numpy())

        val_euclidean = np.mean(val_errors)

        scheduler.step(val_euclidean)

        if val_euclidean < best_val_loss:
            best_val_loss = val_euclidean
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    return model, best_val_loss


def main():
    print("=" * 70)
    print("exp_072: Simple MLP with MAE Loss")
    print("=" * 70)

    # 데이터 로드
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    TOP_12 = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession', 'velocity'
    ]

    X = last_passes[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    print(f"\nData shape: X={X.shape}, y={y.shape}")

    # 하이퍼파라미터
    hidden_dims = [128, 64, 32]
    dropout = 0.3
    lr = 0.001
    weight_decay = 0.01
    epochs = 300
    batch_size = 256
    patience = 30

    print(f"\nHyperparameters:")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Epochs: {epochs} (early stopping patience={patience})")
    print(f"  Batch size: {batch_size}")

    print("\n[1] CV (5-Fold)...")
    gkf = GroupKFold(n_splits=5)
    fold_scores = []
    models = []
    scalers = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        # To tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y[train_idx])
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y[val_idx])

        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model
        model = MLP(
            input_dim=len(TOP_12),
            hidden_dims=hidden_dims,
            output_dim=2,
            dropout=dropout
        ).to(device)

        # Train
        model, val_loss = train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, weight_decay=weight_decay, patience=patience
        )

        fold_scores.append(val_loss)
        print(f"  Fold {fold}: {val_loss:.4f}")

        models.append(model)
        scalers.append(scaler)

    cv = np.mean(fold_scores)
    print(f"  CV: {cv:.4f}")

    # Test 예측
    print("\n[2] Test 예측...")
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    test_episodes = []
    for _, row in test_df.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        if 'dx' not in ep_df.columns:
            ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
            ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']
        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)

    test_all = pd.concat(test_episodes, ignore_index=True)
    test_all = create_features(test_all)
    test_last = test_all.groupby('game_episode').last().reset_index()

    X_test = test_last[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values

    pred_x = np.zeros(len(X_test))
    pred_y = np.zeros(len(X_test))

    for model, scaler in zip(models, scalers):
        model.eval()
        X_test_scaled = scaler.transform(X_test)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)

        with torch.no_grad():
            pred = model(X_test_t).cpu().numpy()
            pred_x += pred[:, 0] / len(models)
            pred_y += pred[:, 1] / len(models)

    pred_y = np.clip(pred_y, 0, 68)

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })

    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_mlp_cv{cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  저장: {output_path}")

    print("\n" + "=" * 70)
    print(f"MLP CV: {cv:.4f}")
    print(f"vs CatBoost MAE (13.66): {'+' if cv > 13.66 else ''}{cv - 13.66:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
