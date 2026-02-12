"""
exp_106: xT (Expected Threat) Feature
Based on Karun Singh's xT model
- 12x8 grid for pitch zones
- Calculate xT values from training data
- Add xT as feature for pass prediction
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

# xT Grid parameters
GRID_X = 12  # zones along x-axis
GRID_Y = 8   # zones along y-axis
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

def coord_to_zone(x, y):
    """Convert coordinates to zone indices"""
    zone_x = np.clip(np.floor(x / (PITCH_LENGTH / GRID_X)).astype(int), 0, GRID_X - 1)
    zone_y = np.clip(np.floor(y / (PITCH_WIDTH / GRID_Y)).astype(int), 0, GRID_Y - 1)
    return zone_x, zone_y

def calculate_xt_grid(df, iterations=20):
    """
    Calculate xT grid from pass data
    Returns 12x8 grid of xT values
    """
    n_zones = GRID_X * GRID_Y

    # Initialize transition matrix
    transition_counts = np.zeros((n_zones, n_zones))
    zone_action_counts = np.zeros(n_zones)
    zone_shot_counts = np.zeros(n_zones)
    zone_goal_counts = np.zeros(n_zones)

    # Shot types
    shot_types = ['On Target', 'Off Target', 'Blocked', 'Low Quality Shot']
    goal_types = ['On Target']  # Approximation: On Target as goal

    for _, row in df.iterrows():
        sx, sy = coord_to_zone(row['start_x'], row['start_y'])
        ex, ey = coord_to_zone(row['end_x'], row['end_y'])

        start_zone = sy * GRID_X + sx
        end_zone = ey * GRID_X + ex

        zone_action_counts[start_zone] += 1

        # Is it a shot?
        if row['result_name'] in shot_types:
            zone_shot_counts[start_zone] += 1
            if row['result_name'] in goal_types:
                zone_goal_counts[start_zone] += 1
        else:
            # Successful pass
            if row['result_name'] == 'Successful':
                transition_counts[start_zone, end_zone] += 1

    # Normalize transition probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_probs = transition_counts / zone_action_counts[:, np.newaxis]
        transition_probs = np.nan_to_num(transition_probs, 0)

        # Shot probability per zone
        shot_prob = zone_shot_counts / zone_action_counts
        shot_prob = np.nan_to_num(shot_prob, 0)

        # Goal probability given shot (xG approximation)
        goal_given_shot = zone_goal_counts / zone_shot_counts
        goal_given_shot = np.nan_to_num(goal_given_shot, 0)

    # Initial xT = shot_prob * goal_given_shot
    xt = shot_prob * goal_given_shot

    # Iteratively update: xT[i] = shot_prob[i] * xG[i] + sum_j(trans_prob[i,j] * xT[j])
    for _ in range(iterations):
        xt_new = shot_prob * goal_given_shot + (1 - shot_prob) * (transition_probs @ xt)
        if np.max(np.abs(xt_new - xt)) < 1e-8:
            break
        xt = xt_new

    return xt.reshape((GRID_Y, GRID_X))

def get_xt_value(xt_grid, x, y):
    """Get xT value for a coordinate"""
    zone_x, zone_y = coord_to_zone(x, y)
    return xt_grid[zone_y, zone_x]

def create_features(df, xt_grid=None):
    """Create features including xT"""
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

    # xT features (if grid provided)
    if xt_grid is not None:
        df['xt_start'] = df.apply(lambda r: get_xt_value(xt_grid, r['start_x'], r['start_y']), axis=1)
        # Previous action's xT
        df['prev_xt'] = df.groupby('game_episode')['xt_start'].shift(1).fillna(0)
        # xT momentum (change in xT)
        df['xt_momentum'] = df['xt_start'] - df['prev_xt']

    return df

def main():
    print("="*60)
    print("exp_106: xT (Expected Threat) Feature")
    print("="*60)

    # Load train data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"Loaded {len(train_df)} rows")

    # Calculate xT grid
    print("\n[1] Calculating xT grid...")
    xt_grid = calculate_xt_grid(train_df)
    print(f"xT grid shape: {xt_grid.shape}")
    print(f"xT range: {xt_grid.min():.4f} ~ {xt_grid.max():.4f}")
    print("\nxT Grid (rows=y, cols=x, goal on right):")
    print(np.round(xt_grid, 3))

    # Create features
    print("\n[2] Creating features with xT...")
    train_df = create_features(train_df, xt_grid)
    train_last = train_df.groupby('game_episode').last().reset_index()
    print(f"Episodes: {len(train_last)}")

    # Feature sets
    TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
              'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
              'ema_start_y', 'ema_success_rate', 'ema_possession',
              'zone_x', 'result_encoded', 'diff_x', 'velocity']

    # Add xT features
    XT_FEATURES = ['xt_start', 'prev_xt', 'xt_momentum']

    print(f"\n[3] Testing feature sets...")

    # Prepare data
    X_base = train_last[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_xt = train_last[XT_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_combined = np.hstack([X_base, X_xt])

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

                # Euclidean distance
                dist = np.sqrt((pred_dx - y_dx[val_idx])**2 + (pred_dy - y_dy[val_idx])**2)
                fold_scores.append(dist.mean())

            all_scores.append(np.mean(fold_scores))

        cv = np.mean(all_scores)
        print(f"  {name}: CV {cv:.4f}")
        return cv

    # Compare
    cv_base = evaluate(X_base, "Baseline (15 features)")
    cv_combined = evaluate(X_combined, "With xT (18 features)")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Baseline:    CV {cv_base:.4f}")
    print(f"  With xT:     CV {cv_combined:.4f}")
    print(f"  Improvement: {cv_base - cv_combined:.4f}")
    print("="*60)

    # Save xT grid for later use
    np.save(DATA_DIR / 'xt_grid.npy', xt_grid)
    print(f"\nSaved xT grid to {DATA_DIR / 'xt_grid.npy'}")

if __name__ == "__main__":
    main()
