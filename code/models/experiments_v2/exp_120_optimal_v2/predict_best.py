"""
exp_120: Best Settings Prediction
- LR=0.01, iterations=3000
- L2=300
- +ema_momentum_y feature
- 11-fold, 7-seeds
"""
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

FEATURES_16 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
               'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
               'ema_start_y', 'ema_success_rate', 'ema_possession',
               'zone_x', 'result_encoded', 'diff_x', 'velocity', 'ema_momentum_y']

def main():
    print("="*60)
    print("Best Prediction: LR=0.01, L2=300")
    print("11-fold, 7-seeds")
    print("="*60)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = load_test_data()
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

    train_df = create_features(train_df)
    test_df = create_features(test_df)

    train_last = train_df.groupby('game_episode').last().reset_index()
    test_last = test_df.groupby('game_episode').last().reset_index()

    print(f"Train episodes: {len(train_last)}")
    print(f"Test episodes: {len(test_last)}")

    X_train = train_last[FEATURES_16].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_test = test_last[FEATURES_16].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_train, y_dx, groups))

    seeds = [42, 123, 456, 789, 1000, 2024, 9999]

    all_preds_dx = []
    all_preds_dy = []
    oof_scores = []

    for seed in seeds:
        print(f"\nSeed {seed}...")
        params = {'iterations': 3000, 'depth': 8, 'learning_rate': 0.01,
                  'l2_leaf_reg': 300.0, 'random_state': seed, 'verbose': 0,
                  'early_stopping_rounds': 100, 'loss_function': 'MAE'}

        seed_preds_dx = []
        seed_preds_dy = []
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            m_dx = CatBoostRegressor(**params)
            m_dy = CatBoostRegressor(**params)
            m_dx.fit(X_train[train_idx], y_dx[train_idx], eval_set=(X_train[val_idx], y_dx[val_idx]), use_best_model=True)
            m_dy.fit(X_train[train_idx], y_dy[train_idx], eval_set=(X_train[val_idx], y_dy[val_idx]), use_best_model=True)

            pred_dx_val = m_dx.predict(X_train[val_idx])
            pred_dy_val = m_dy.predict(X_train[val_idx])
            dist = np.sqrt((pred_dx_val - y_dx[val_idx])**2 + (pred_dy_val - y_dy[val_idx])**2)
            fold_scores.append(dist.mean())

            pred_dx = m_dx.predict(X_test)
            pred_dy = m_dy.predict(X_test)
            seed_preds_dx.append(pred_dx)
            seed_preds_dy.append(pred_dy)

        cv = np.mean(fold_scores)
        oof_scores.append(cv)
        print(f"  Seed {seed} CV: {cv:.4f}")

        all_preds_dx.append(np.mean(seed_preds_dx, axis=0))
        all_preds_dy.append(np.mean(seed_preds_dy, axis=0))

    final_cv = np.mean(oof_scores)
    print(f"\nFinal CV: {final_cv:.4f} (+/- {np.std(oof_scores):.4f})")

    pred_dx = np.mean(all_preds_dx, axis=0)
    pred_dy = np.mean(all_preds_dy, axis=0)

    test_start_x = test_last['start_x'].values
    test_start_y = test_last['start_y'].values
    pred_end_x = test_start_x + pred_dx
    pred_end_y = test_start_y + pred_dy

    pred_end_x = np.clip(pred_end_x, 0, 105)
    pred_end_y = np.clip(pred_end_y, 0, 68)

    submission = sample_sub.copy()
    submission['end_x'] = pred_end_x
    submission['end_y'] = pred_end_y

    out_path = BASE / f'submissions/submission_best_cv{final_cv:.2f}.csv'
    submission.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print(f"\nVerification:")
    print(f"  Shape: {submission.shape}")
    print(f"  end_x: [{submission['end_x'].min():.2f}, {submission['end_x'].max():.2f}]")
    print(f"  end_y: [{submission['end_y'].min():.2f}, {submission['end_y'].max():.2f}]")
    print(f"  NaN: {submission.isna().sum().sum()}")

if __name__ == "__main__":
    main()
