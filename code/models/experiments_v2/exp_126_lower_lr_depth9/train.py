"""exp_126: Lower LR with Depth=9, L2=600"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"

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
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_y'])
    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt((df['start_x'] - df['prev_start_x'])**2 + (df['start_y'] - df['prev_start_y'])**2)
    df['ema_momentum_y'] = df['ema_start_y'] - df['start_y']
    return df

FEATURES = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x', 'ema_start_y', 'ema_success_rate', 'ema_possession', 'zone_x', 'result_encoded', 'diff_x', 'velocity', 'ema_momentum_y']

def main():
    print("="*60)
    print("exp_126: Lower LR with Depth=9, L2=600")
    print("="*60)
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    train_last = train_df.groupby('game_episode').last().reset_index()
    print(f"Episodes: {len(train_last)}")
    X = train_last[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values
    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X, y_dx, groups))

    configs = [
        ('LR=0.01, i3000 (baseline)', {'learning_rate': 0.01, 'iterations': 3000}),
        ('LR=0.008, i4000', {'learning_rate': 0.008, 'iterations': 4000}),
        ('LR=0.005, i5000', {'learning_rate': 0.005, 'iterations': 5000}),
        ('LR=0.005, i7000', {'learning_rate': 0.005, 'iterations': 7000}),
    ]

    results = {}
    for name, cfg in configs:
        print(f"\n{name}...")
        scores = []
        for seed in [42, 123, 456]:
            params = {'depth': 9, 'l2_leaf_reg': 600.0, 'random_state': seed, 'verbose': 0, 'early_stopping_rounds': 100, 'loss_function': 'MAE', **cfg}
            fold_scores = []
            for train_idx, val_idx in folds:
                m_dx = CatBoostRegressor(**params)
                m_dy = CatBoostRegressor(**params)
                m_dx.fit(X[train_idx], y_dx[train_idx], eval_set=(X[val_idx], y_dx[val_idx]), use_best_model=True)
                m_dy.fit(X[train_idx], y_dy[train_idx], eval_set=(X[val_idx], y_dy[val_idx]), use_best_model=True)
                pred_dx = m_dx.predict(X[val_idx])
                pred_dy = m_dy.predict(X[val_idx])
                dist = np.sqrt((pred_dx - y_dx[val_idx])**2 + (pred_dy - y_dy[val_idx])**2)
                fold_scores.append(dist.mean())
            scores.append(np.mean(fold_scores))
        cv = np.mean(scores)
        std = np.std(scores)
        results[name] = (cv, std)
        print(f"  {name}: CV {cv:.4f} (+/- {std:.4f})")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    best = min(results, key=lambda k: results[k][0])
    for k in sorted(results.keys(), key=lambda k: results[k][0]):
        cv, std = results[k]
        m = " <-- BEST" if k == best else ""
        print(f"  {k}: CV {cv:.4f} (+/- {std:.4f}){m}")

if __name__ == "__main__":
    main()
