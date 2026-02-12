"""
exp_131: Entity Embeddings for Categorical Features
- Learn embeddings for team_id, player_id, action_id, type_name
- Use neural network to learn embeddings, then feed to CatBoost
"""
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"

def create_features(df):
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))
    df['dx'] = df['end_x'] - df['start_x'] if 'end_x' in df.columns else 0
    df['dy'] = df['end_y'] - df['start_y'] if 'end_y' in df.columns else 0
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0) if 'dx' in df.columns else 0
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0) if 'dy' in df.columns else 0
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)
    ema_span = 2
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_y'])
    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt((df['start_x'] - df['prev_start_x'])**2 + (df['start_y'] - df['prev_start_y'])**2)
    df['ema_momentum_y'] = df['ema_start_y'] - df['start_y']
    return df

def load_test_data():
    test_index = pd.read_csv(DATA_DIR / 'test.csv')
    dfs = []
    for _, row in test_index.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

class EntityEmbeddingNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, n_cont, output_dim=2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb) for dim, emb in zip(cat_dims, emb_dims)
        ])
        total_emb = sum(emb_dims)
        self.fc = nn.Sequential(
            nn.Linear(total_emb + n_cont, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x_cat, x_cont):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_cont], dim=1)
        return self.fc(x)

    def get_embeddings(self, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embs, dim=1)

FEATURES = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
            'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
            'ema_start_y', 'ema_success_rate', 'ema_possession',
            'zone_x', 'result_encoded', 'diff_x', 'velocity', 'ema_momentum_y']

CAT_COLS = ['team_id', 'type_name']

def main():
    print("=" * 60)
    print("exp_131: Entity Embeddings")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = load_test_data()
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Create features
    train_df = create_features(train_df)
    test_df['dx'] = 0
    test_df['dy'] = 0
    test_df = create_features(test_df)

    # Last event per episode
    train_last = train_df.groupby('game_episode').last().reset_index()
    test_last = test_df.groupby('game_episode').last().reset_index()
    print(f"Train episodes: {len(train_last)}, Test episodes: {len(test_last)}")

    # Encode categorical features
    label_encoders = {}
    cat_dims = []
    for col in CAT_COLS:
        le = LabelEncoder()
        combined = pd.concat([train_last[col], test_last[col]]).astype(str)
        le.fit(combined)
        train_last[f'{col}_enc'] = le.transform(train_last[col].astype(str))
        test_last[f'{col}_enc'] = le.transform(test_last[col].astype(str))
        label_encoders[col] = le
        cat_dims.append(len(le.classes_))
        print(f"  {col}: {len(le.classes_)} unique values")

    # Embedding dimensions (rule of thumb: min(50, (n+1)//2))
    emb_dims = [min(50, (d+1)//2) for d in cat_dims]
    print(f"Embedding dims: {emb_dims}")

    # Prepare data
    X_cont = train_last[FEATURES].values.astype(np.float32)
    X_cat = train_last[[f'{c}_enc' for c in CAT_COLS]].values.astype(np.int64)
    y = train_last[['dx', 'dy']].values.astype(np.float32)
    groups = train_last['game_id'].values

    X_cont_test = test_last[FEATURES].values.astype(np.float32)
    X_cat_test = test_last[[f'{c}_enc' for c in CAT_COLS]].values.astype(np.int64)

    # Train embedding network
    print("\n[Phase 1] Training Entity Embedding Network...")

    X_cat_tensor = torch.LongTensor(X_cat).to(device)
    X_cont_tensor = torch.FloatTensor(X_cont).to(device)
    y_tensor = torch.FloatTensor(y).to(device)

    model = EntityEmbeddingNet(cat_dims, emb_dims, len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    dataset = TensorDataset(X_cat_tensor, X_cont_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for x_cat, x_cont, y_batch in loader:
            optimizer.zero_grad()
            pred = model(x_cat, x_cont)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    # Extract embeddings
    print("\n[Phase 2] Extracting Embeddings...")
    model.eval()
    with torch.no_grad():
        train_emb = model.get_embeddings(X_cat_tensor).cpu().numpy()
        X_cat_test_tensor = torch.LongTensor(X_cat_test).to(device)
        test_emb = model.get_embeddings(X_cat_test_tensor).cpu().numpy()

    # Add embeddings to features
    emb_cols = [f'emb_{i}' for i in range(train_emb.shape[1])]
    X_train_final = np.hstack([X_cont, train_emb])
    X_test_final = np.hstack([X_cont_test, test_emb])

    print(f"Final feature shape: {X_train_final.shape}")

    # Train CatBoost with embeddings
    print("\n[Phase 3] Training CatBoost with Embeddings...")

    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)

    base_params = {'iterations': 4000, 'depth': 9, 'learning_rate': 0.008,
                   'l2_leaf_reg': 600.0, 'verbose': 0, 'early_stopping_rounds': 100,
                   'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_train_final, y_dx, groups))
    seeds = [42, 123, 456]

    oof_dx = np.zeros(len(X_train_final))
    oof_dy = np.zeros(len(X_train_final))
    pred_dx = np.zeros(len(X_test_final))
    pred_dy = np.zeros(len(X_test_final))

    for seed in seeds:
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            params = {**base_params, 'random_seed': seed}

            model_dx = CatBoostRegressor(**params)
            model_dx.fit(X_train_final[tr_idx], y_dx[tr_idx],
                        eval_set=(X_train_final[val_idx], y_dx[val_idx]))
            oof_dx[val_idx] += model_dx.predict(X_train_final[val_idx])
            pred_dx += model_dx.predict(X_test_final)

            model_dy = CatBoostRegressor(**params)
            model_dy.fit(X_train_final[tr_idx], y_dy[tr_idx],
                        eval_set=(X_train_final[val_idx], y_dy[val_idx]))
            oof_dy[val_idx] += model_dy.predict(X_train_final[val_idx])
            pred_dy += model_dy.predict(X_test_final)
        print(f"  Seed {seed} done")

    oof_dx /= len(seeds)
    oof_dy /= len(seeds)
    pred_dx /= (len(seeds) * len(folds))
    pred_dy /= (len(seeds) * len(folds))

    # Calculate CV
    cv = np.mean(np.sqrt((oof_dx - y_dx)**2 + (oof_dy - y_dy)**2))
    print(f"\nCV Score: {cv:.4f}")

    # Create submission
    end_x = test_last['start_x'].values + pred_dx
    end_y = test_last['start_y'].values + pred_dy

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': end_x,
        'end_y': end_y
    })

    sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = sample[['game_episode']].merge(submission, on='game_episode', how='left')

    out_path = BASE / 'submissions' / f'submission_entity_emb_cv{cv:.2f}.csv'
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
