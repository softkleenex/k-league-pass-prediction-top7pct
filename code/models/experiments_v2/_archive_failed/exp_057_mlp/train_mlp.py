"""
exp_057: MLP (Neural Network) Model

CatBoost가 잡지 못하는 비선형 관계를 MLP로 학습
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 2))  # output: end_x, end_y
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_all_features(df):
    """모든 피처 생성"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)

    df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

    def calc_streak(group):
        values = group['is_final_team'].values
        streaks = []
        current_streak = 0
        for val in values:
            if val == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)
        return pd.Series(streaks, index=group.index)

    df['final_poss_len'] = df.groupby('game_episode', group_keys=False).apply(calc_streak)

    ema_span = 2
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['move_distance'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)

    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])

    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    df['ema_goal_distance'] = df.groupby('game_episode')['goal_distance'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['goal_distance'])

    df['ema_distance'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0)

    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['diff_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['diff_goal_dist'] = df.groupby('game_episode')['goal_distance'].diff().fillna(0)

    df['rolling_std_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['rolling_std_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['rolling_std_dist'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['velocity'] = df['move_distance']
    df['acceleration'] = df.groupby('game_episode')['velocity'].diff().fillna(0)

    df['rel_x_from_ema'] = df['start_x'] - df['ema_start_x']
    df['rel_y_from_ema'] = df['start_y'] - df['ema_start_y']

    df['episode_start_x'] = df.groupby('game_episode')['start_x'].transform('first')
    df['episode_start_y'] = df.groupby('game_episode')['start_y'].transform('first')
    df['progress_from_start_x'] = df['start_x'] - df['episode_start_x']
    df['progress_from_start_y'] = df['start_y'] - df['episode_start_y']

    df['x_times_direction'] = df['start_x'] * df['direction']
    df['goal_dist_times_velocity'] = df['goal_distance'] * df['velocity']
    df['success_times_possession'] = df['ema_success_rate'] * df['ema_possession']

    df['is_forward'] = (df['diff_x'] > 0).astype(int)
    df['forward_streak'] = df.groupby('game_episode')['is_forward'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df['avg_forward_x'] = df.groupby('game_episode')['diff_x'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).fillna(0)

    # 추가 피처: 현재 위치 그대로
    df['start_x_norm'] = df['start_x'] / 105
    df['start_y_norm'] = df['start_y'] / 68

    return df


def train_mlp(X_train, y_train, X_val, y_val, epochs=100, batch_size=256, lr=0.001):
    """MLP 학습"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
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

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 15:
            break

    model.load_state_dict(best_model_state)
    return model, scaler


def main():
    print("=" * 70)
    print("exp_057: MLP (Neural Network) Model")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_all_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    features = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct',
        'ema_start_x', 'ema_start_y', 'ema_goal_distance',
        'ema_distance', 'ema_success_rate', 'ema_possession',
        'dist_to_goal_line', 'dist_to_sideline', 'dist_to_center_y', 'position_zone',
        'diff_x', 'diff_y', 'diff_goal_dist',
        'rolling_std_x', 'rolling_std_y', 'rolling_std_dist',
        'velocity', 'acceleration',
        'rel_x_from_ema', 'rel_y_from_ema', 'progress_from_start_x', 'progress_from_start_y',
        'x_times_direction', 'goal_dist_times_velocity', 'success_times_possession',
        'forward_streak', 'avg_forward_x',
        'start_x_norm', 'start_y_norm'
    ]

    X = last_passes[features].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # CV
    print("\n[2] 3-Fold CV...")
    gkf = GroupKFold(n_splits=3)
    fold_scores = []
    models = []
    scalers = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n  Fold {fold}...")
        model, scaler = train_mlp(X[train_idx], y[train_idx], X[val_idx], y[val_idx])

        model.eval()
        X_val_scaled = scaler.transform(X[val_idx])
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()

        errors = np.sqrt((pred[:, 0] - y[val_idx, 0])**2 + (pred[:, 1] - y[val_idx, 1])**2)
        fold_scores.append(errors.mean())
        print(f"    Fold {fold}: {fold_scores[-1]:.4f}")

        models.append(model)
        scalers.append(scaler)

    cv = np.mean(fold_scores)
    print(f"\n  MLP CV: {cv:.4f}")

    # Test 예측
    print("\n[3] Test 예측...")
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
    test_all = create_all_features(test_all)
    test_last = test_all.groupby('game_episode').last().reset_index()

    X_test = test_last[features].fillna(0).replace([np.inf, -np.inf], 0).values

    pred_x = np.zeros(len(X_test))
    pred_y = np.zeros(len(X_test))

    for model, scaler in zip(models, scalers):
        model.eval()
        X_test_scaled = scaler.transform(X_test)
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
        pred_x += pred[:, 0] / len(models)
        pred_y += pred[:, 1] / len(models)

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
    print("=" * 70)


if __name__ == "__main__":
    main()
