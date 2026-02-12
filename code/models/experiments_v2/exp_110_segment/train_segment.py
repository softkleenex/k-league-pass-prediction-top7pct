"""
exp_110: Pass Distance Segmentation
- Split data by pass distance (short vs long)
- Train separate models for each segment
- Compare with unified model
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
    df['pass_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
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
    print("exp_110: Pass Distance Segmentation")
    print("="*60)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    train_last = train_df.groupby('game_episode').last().reset_index()
    print(f"Episodes: {len(train_last)}")

    # Analyze pass distance distribution
    print("\n[1] Pass distance analysis...")
    print(f"Pass distance: min={train_last['pass_distance'].min():.1f}, max={train_last['pass_distance'].max():.1f}")
    print(f"Mean: {train_last['pass_distance'].mean():.1f}, Median: {train_last['pass_distance'].median():.1f}")

    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(train_last['pass_distance'], p)
        print(f"  {p}th percentile: {val:.1f}")

    # Try different thresholds
    thresholds = [10, 15, 20, 25]

    X = train_last[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    pass_dist = train_last['pass_distance'].values
    groups = train_last['game_id'].values

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X, y_dx, groups))

    print("\n[2] Testing segmentation thresholds...")

    # Baseline (unified model)
    def evaluate_unified():
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
        return np.mean(all_scores)

    cv_unified = evaluate_unified()
    print(f"  Unified model: CV {cv_unified:.4f}")

    # Segmented model
    for threshold in thresholds:
        short_mask = pass_dist < threshold
        long_mask = pass_dist >= threshold

        print(f"\n  Threshold={threshold}m (short: {short_mask.sum()}, long: {long_mask.sum()})...")

        all_scores = []
        for seed in [42, 123, 456]:
            params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                      'l2_leaf_reg': 150.0, 'random_state': seed, 'verbose': 0,
                      'early_stopping_rounds': 50, 'loss_function': 'MAE'}

            fold_scores = []
            for train_idx, val_idx in folds:
                # Predictions array
                pred_dx = np.zeros(len(val_idx))
                pred_dy = np.zeros(len(val_idx))

                # Train and predict for short passes
                train_short = train_idx[short_mask[train_idx]]
                val_short_mask = short_mask[val_idx]

                if len(train_short) > 100:
                    m_dx_s = CatBoostRegressor(**params)
                    m_dy_s = CatBoostRegressor(**params)
                    m_dx_s.fit(X[train_short], y_dx[train_short], verbose=0)
                    m_dy_s.fit(X[train_short], y_dy[train_short], verbose=0)
                    if val_short_mask.sum() > 0:
                        pred_dx[val_short_mask] = m_dx_s.predict(X[val_idx][val_short_mask])
                        pred_dy[val_short_mask] = m_dy_s.predict(X[val_idx][val_short_mask])

                # Train and predict for long passes
                train_long = train_idx[long_mask[train_idx]]
                val_long_mask = long_mask[val_idx]

                if len(train_long) > 100:
                    m_dx_l = CatBoostRegressor(**params)
                    m_dy_l = CatBoostRegressor(**params)
                    m_dx_l.fit(X[train_long], y_dx[train_long], verbose=0)
                    m_dy_l.fit(X[train_long], y_dy[train_long], verbose=0)
                    if val_long_mask.sum() > 0:
                        pred_dx[val_long_mask] = m_dx_l.predict(X[val_idx][val_long_mask])
                        pred_dy[val_long_mask] = m_dy_l.predict(X[val_idx][val_long_mask])

                dist = np.sqrt((pred_dx - y_dx[val_idx])**2 + (pred_dy - y_dy[val_idx])**2)
                fold_scores.append(dist.mean())

            all_scores.append(np.mean(fold_scores))

        cv_seg = np.mean(all_scores)
        improvement = cv_unified - cv_seg
        print(f"    Segmented CV: {cv_seg:.4f} (diff: {improvement:+.4f})")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Unified model: CV {cv_unified:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
