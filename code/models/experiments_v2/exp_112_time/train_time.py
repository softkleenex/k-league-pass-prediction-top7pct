"""
exp_112: Time-based Features
- Time in episode (sequence position)
- Time in match
- Episode length
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

def create_features(df, add_time_features=False):
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

    if add_time_features:
        # Episode length (number of actions in episode)
        df['episode_length'] = df.groupby('game_episode')['game_episode'].transform('count')

        # Position in episode (1, 2, 3, ...)
        df['action_position'] = df.groupby('game_episode').cumcount() + 1

        # Relative position in episode (0 to 1)
        df['relative_position'] = df['action_position'] / df['episode_length']

        # Time in match (normalized to 0-1 for each period)
        df['time_normalized'] = df.groupby(['game_id', 'period_id'])['time_seconds'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        )

        # Period (1 or 2)
        df['period'] = df['period_id']

    return df

TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
          'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
          'ema_start_y', 'ema_success_rate', 'ema_possession',
          'zone_x', 'result_encoded', 'diff_x', 'velocity']

TIME_FEATURES = ['episode_length', 'action_position', 'relative_position', 'time_normalized', 'period']

def main():
    print("="*60)
    print("exp_112: Time-based Features")
    print("="*60)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')

    # Create features with time
    train_df = create_features(train_df, add_time_features=True)
    train_last = train_df.groupby('game_episode').last().reset_index()
    print(f"Episodes: {len(train_last)}")

    # Check time features
    print("\nTime features stats:")
    for feat in TIME_FEATURES:
        print(f"  {feat}: min={train_last[feat].min():.2f}, max={train_last[feat].max():.2f}, mean={train_last[feat].mean():.2f}")

    X_base = train_last[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_time = train_last[TIME_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_combined = np.hstack([X_base, X_time])

    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_base, y_dx, groups))

    def evaluate(X, name, seeds=[42, 123, 456]):
        all_scores = []
        for seed in seeds:
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

        cv = np.mean(all_scores)
        print(f"  {name}: CV {cv:.4f}")
        return cv

    print("\nTesting feature sets...")
    cv_base = evaluate(X_base, "Baseline (15 features)")
    cv_combined = evaluate(X_combined, "With Time (20 features)")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Baseline:   CV {cv_base:.4f}")
    print(f"  With Time:  CV {cv_combined:.4f}")
    print(f"  Difference: {cv_base - cv_combined:+.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
