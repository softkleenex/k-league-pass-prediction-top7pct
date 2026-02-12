"""
exp_116: Bootstrap Aggregation
- Test bootstrap sampling (bagging)
- Multiple bootstrap samples, average predictions
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
    return df

TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
          'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
          'ema_start_y', 'ema_success_rate', 'ema_possession',
          'zone_x', 'result_encoded', 'diff_x', 'velocity']

def main():
    print("="*60)
    print("exp_116: Bootstrap Aggregation")
    print("="*60)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    train_last = train_df.groupby('game_episode').last().reset_index()
    print(f"Episodes: {len(train_last)}")

    X = train_last[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X, y_dx, groups))

    # Baseline without bootstrap (single model)
    print("\n[1] Baseline (no bootstrap)...")
    all_scores = []
    for seed in [42, 123, 456]:
        params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                  'l2_leaf_reg': 150.0, 'random_state': seed, 'verbose': 0,
                  'early_stopping_rounds': 50, 'loss_function': 'MAE'}

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

        all_scores.append(np.mean(fold_scores))
    cv_baseline = np.mean(all_scores)
    print(f"  Baseline: CV {cv_baseline:.4f}")

    # Bootstrap with different sample fractions
    bootstrap_configs = [
        {'n_bags': 5, 'sample_frac': 0.8},
        {'n_bags': 10, 'sample_frac': 0.8},
        {'n_bags': 5, 'sample_frac': 0.7},
        {'n_bags': 10, 'sample_frac': 0.7},
    ]

    results = {'Baseline': cv_baseline}

    for config in bootstrap_configs:
        n_bags = config['n_bags']
        sample_frac = config['sample_frac']
        key = f"bags{n_bags}_frac{sample_frac}"

        print(f"\n[2] Testing {key}...")

        all_scores = []
        for seed in [42]:  # Single seed for speed
            fold_scores = []
            for train_idx, val_idx in folds:
                X_train, X_val = X[train_idx], X[val_idx]
                y_dx_train, y_dx_val = y_dx[train_idx], y_dx[val_idx]
                y_dy_train, y_dy_val = y_dy[train_idx], y_dy[val_idx]

                n_samples = len(X_train)
                n_bootstrap = int(n_samples * sample_frac)

                preds_dx = []
                preds_dy = []

                for bag_idx in range(n_bags):
                    np.random.seed(seed + bag_idx)
                    boot_idx = np.random.choice(n_samples, size=n_bootstrap, replace=True)

                    params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                              'l2_leaf_reg': 150.0, 'random_state': seed + bag_idx, 'verbose': 0,
                              'early_stopping_rounds': 50, 'loss_function': 'MAE'}

                    m_dx = CatBoostRegressor(**params)
                    m_dy = CatBoostRegressor(**params)
                    m_dx.fit(X_train[boot_idx], y_dx_train[boot_idx], eval_set=(X_val, y_dx_val), use_best_model=True)
                    m_dy.fit(X_train[boot_idx], y_dy_train[boot_idx], eval_set=(X_val, y_dy_val), use_best_model=True)

                    preds_dx.append(m_dx.predict(X_val))
                    preds_dy.append(m_dy.predict(X_val))

                # Average predictions
                pred_dx = np.mean(preds_dx, axis=0)
                pred_dy = np.mean(preds_dy, axis=0)

                dist = np.sqrt((pred_dx - y_dx_val)**2 + (pred_dy - y_dy_val)**2)
                fold_scores.append(dist.mean())

            all_scores.append(np.mean(fold_scores))

        cv = np.mean(all_scores)
        results[key] = cv
        print(f"  {key}: CV {cv:.4f}")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    best_key = min(results, key=results.get)
    for key, cv in sorted(results.items(), key=lambda x: x[1]):
        marker = " <-- BEST" if key == best_key else ""
        print(f"  {key}: CV {cv:.4f}{marker}")
    print("="*60)

if __name__ == "__main__":
    main()
