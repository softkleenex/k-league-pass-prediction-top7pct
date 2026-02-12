"""
exp_107: Zone Transition Features
- Learn zone-to-zone transition probabilities
- Add expected dx/dy based on zone statistics
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

# Zone parameters
GRID_X = 6
GRID_Y = 6
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

def coord_to_zone(x, y):
    zone_x = np.clip(np.floor(x / (PITCH_LENGTH / GRID_X)).astype(int), 0, GRID_X - 1)
    zone_y = np.clip(np.floor(y / (PITCH_WIDTH / GRID_Y)).astype(int), 0, GRID_Y - 1)
    return zone_x, zone_y

def calculate_zone_stats(df):
    """Calculate statistics for each zone"""
    n_zones = GRID_X * GRID_Y

    # Mean dx, dy from each zone
    zone_mean_dx = np.zeros(n_zones)
    zone_mean_dy = np.zeros(n_zones)
    zone_std_dx = np.zeros(n_zones)
    zone_std_dy = np.zeros(n_zones)
    zone_counts = np.zeros(n_zones)

    # Calculate dx, dy
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']

    for _, row in df.iterrows():
        sx, sy = coord_to_zone(row['start_x'], row['start_y'])
        zone = sy * GRID_X + sx

        zone_mean_dx[zone] += row['dx']
        zone_mean_dy[zone] += row['dy']
        zone_counts[zone] += 1

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        zone_mean_dx = zone_mean_dx / zone_counts
        zone_mean_dy = zone_mean_dy / zone_counts
        zone_mean_dx = np.nan_to_num(zone_mean_dx, 0)
        zone_mean_dy = np.nan_to_num(zone_mean_dy, 0)

    # Calculate std (second pass)
    for _, row in df.iterrows():
        sx, sy = coord_to_zone(row['start_x'], row['start_y'])
        zone = sy * GRID_X + sx

        zone_std_dx[zone] += (row['dx'] - zone_mean_dx[zone])**2
        zone_std_dy[zone] += (row['dy'] - zone_mean_dy[zone])**2

    with np.errstate(divide='ignore', invalid='ignore'):
        zone_std_dx = np.sqrt(zone_std_dx / zone_counts)
        zone_std_dy = np.sqrt(zone_std_dy / zone_counts)
        zone_std_dx = np.nan_to_num(zone_std_dx, 1)
        zone_std_dy = np.nan_to_num(zone_std_dy, 1)

    return {
        'mean_dx': zone_mean_dx.reshape((GRID_Y, GRID_X)),
        'mean_dy': zone_mean_dy.reshape((GRID_Y, GRID_X)),
        'std_dx': zone_std_dx.reshape((GRID_Y, GRID_X)),
        'std_dy': zone_std_dy.reshape((GRID_Y, GRID_X)),
        'counts': zone_counts.reshape((GRID_Y, GRID_X))
    }

def create_features(df, zone_stats=None):
    """Create features including zone statistics"""
    # Basic features
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

    # Zone statistics features
    if zone_stats is not None:
        def get_zone_stat(row, stat_name):
            zx, zy = int(row['zone_x']), int(row['zone_y'])
            return zone_stats[stat_name][zy, zx]

        df['zone_mean_dx'] = df.apply(lambda r: get_zone_stat(r, 'mean_dx'), axis=1)
        df['zone_mean_dy'] = df.apply(lambda r: get_zone_stat(r, 'mean_dy'), axis=1)
        df['zone_std_dx'] = df.apply(lambda r: get_zone_stat(r, 'std_dx'), axis=1)
        df['zone_std_dy'] = df.apply(lambda r: get_zone_stat(r, 'std_dy'), axis=1)

        # Deviation from zone mean
        df['dev_from_zone_dx'] = df['prev_dx'] - df['zone_mean_dx']
        df['dev_from_zone_dy'] = df['prev_dy'] - df['zone_mean_dy']

    return df

def main():
    print("="*60)
    print("exp_107: Zone Transition Features")
    print("="*60)

    # Load train data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"Loaded {len(train_df)} rows")

    # Calculate zone statistics
    print("\n[1] Calculating zone statistics...")
    zone_stats = calculate_zone_stats(train_df)

    print("Zone Mean dx (rows=y, cols=x):")
    print(np.round(zone_stats['mean_dx'], 1))
    print("\nZone Mean dy:")
    print(np.round(zone_stats['mean_dy'], 1))

    # Create features
    print("\n[2] Creating features...")
    train_df = create_features(train_df, zone_stats)
    train_last = train_df.groupby('game_episode').last().reset_index()
    print(f"Episodes: {len(train_last)}")

    # Feature sets
    TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
              'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
              'ema_start_y', 'ema_success_rate', 'ema_possession',
              'zone_x', 'result_encoded', 'diff_x', 'velocity']

    ZONE_FEATURES = ['zone_mean_dx', 'zone_mean_dy', 'zone_std_dx', 'zone_std_dy',
                     'dev_from_zone_dx', 'dev_from_zone_dy']

    print(f"\n[3] Testing feature sets...")

    X_base = train_last[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_zone = train_last[ZONE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_combined = np.hstack([X_base, X_zone])

    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_base, y_dx, groups))

    def evaluate(X, name, seeds=[42, 123, 456]):
        all_scores = []
        for seed in seeds:
            params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                      'l2_leaf_reg': 30.0, 'random_state': seed, 'verbose': 0,
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

    cv_base = evaluate(X_base, "Baseline (15 features)")
    cv_combined = evaluate(X_combined, "With Zone Stats (21 features)")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Baseline:        CV {cv_base:.4f}")
    print(f"  With Zone Stats: CV {cv_combined:.4f}")
    print(f"  Improvement:     {cv_base - cv_combined:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
