"""
exp_094: Zone Statistics from Wyscout
Pre-training이 안 되었으니, Wyscout에서 zone별 통계를 계산하여 피처로 사용
- 각 zone에서 typical한 패스 방향/거리 정보를 prior로 활용
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import gc
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"

def load_wyscout_passes():
    """Load and combine all Wyscout pass data"""
    files = [
        '/tmp/events_England.csv',
        '/tmp/events_France.csv',
        '/tmp/events_Germany.csv',
        '/tmp/events_Italy.csv',
        '/tmp/events_Spain.csv',
    ]

    all_passes = []
    for f in files:
        df = pd.read_csv(f)
        passes = df[df['eventName'] == 'Pass'].copy()
        all_passes.append(passes)

    combined = pd.concat(all_passes, ignore_index=True)

    # Scale coordinates: Wyscout (0-100) → K-League (0-105 x 0-68)
    combined['start_x'] = combined['pos_orig_x'] * 1.05
    combined['start_y'] = combined['pos_orig_y'] * 0.68
    combined['end_x'] = combined['pos_dest_x'] * 1.05
    combined['end_y'] = combined['pos_dest_y'] * 0.68

    # Calculate dx, dy
    combined['dx'] = combined['end_x'] - combined['start_x']
    combined['dy'] = combined['end_y'] - combined['start_y']

    # Create zone features (6x6 grid)
    combined['zone_x'] = (combined['start_x'] / (105/6)).astype(int).clip(0, 5)
    combined['zone_y'] = (combined['start_y'] / (68/6)).astype(int).clip(0, 5)

    # Remove invalid data
    combined = combined.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])
    combined = combined[(combined['end_x'] >= 0) & (combined['end_x'] <= 105)]
    combined = combined[(combined['end_y'] >= 0) & (combined['end_y'] <= 68)]

    return combined

def compute_zone_stats(wyscout_df):
    """Compute zone-level statistics from Wyscout"""
    stats = wyscout_df.groupby(['zone_x', 'zone_y']).agg({
        'dx': ['mean', 'std', 'median'],
        'dy': ['mean', 'std', 'median'],
    }).reset_index()

    # Flatten column names
    stats.columns = ['zone_x', 'zone_y',
                     'wyscout_dx_mean', 'wyscout_dx_std', 'wyscout_dx_median',
                     'wyscout_dy_mean', 'wyscout_dy_std', 'wyscout_dy_median']

    # Fill NaN std with 0
    stats['wyscout_dx_std'] = stats['wyscout_dx_std'].fillna(0)
    stats['wyscout_dy_std'] = stats['wyscout_dy_std'].fillna(0)

    return stats

def create_features(df, zone_stats=None):
    """Create features for K-League data"""
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

    # Merge zone stats if provided
    if zone_stats is not None:
        df = df.merge(zone_stats, on=['zone_x', 'zone_y'], how='left')
        # Fill NaN with global mean
        for col in zone_stats.columns:
            if col not in ['zone_x', 'zone_y']:
                df[col] = df[col].fillna(df[col].mean())

    return df

# Base features (15)
BASE_FEATURES = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
                 'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
                 'ema_start_y', 'ema_success_rate', 'ema_possession',
                 'zone_x', 'result_encoded', 'diff_x', 'velocity']

# Wyscout zone stats features (6)
WYSCOUT_FEATURES = ['wyscout_dx_mean', 'wyscout_dx_std', 'wyscout_dx_median',
                    'wyscout_dy_mean', 'wyscout_dy_std', 'wyscout_dy_median']

def main():
    print("=" * 70)
    print("exp_094: Zone Statistics from Wyscout")
    print("=" * 70)

    # Load Wyscout and compute zone stats
    print("\n[1] Loading Wyscout and computing zone stats...")
    wyscout = load_wyscout_passes()
    print(f"  Wyscout passes: {len(wyscout):,}")

    zone_stats = compute_zone_stats(wyscout)
    print(f"  Zone stats computed: {len(zone_stats)} zones")
    print(zone_stats.head())

    del wyscout; gc.collect()

    # Load K-League data
    print("\n[2] Loading K-League data...")
    kleague = pd.read_csv(DATA_DIR / 'train.csv')
    kleague = create_features(kleague, zone_stats)
    last_passes = kleague.groupby('game_episode').last().reset_index()
    del kleague; gc.collect()

    print(f"  K-League episodes: {len(last_passes):,}")

    # Prepare data
    X_base = last_passes[BASE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_augmented = last_passes[BASE_FEATURES + WYSCOUT_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    y_dx = last_passes['dx'].values.astype(np.float32)
    y_dy = last_passes['dy'].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    # Model params
    params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
              'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
              'early_stopping_rounds': 50, 'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=11)

    # Baseline (15 features)
    print("\n[3] Baseline (15 features)...")
    oof_dx = np.zeros(len(X_base))
    oof_dy = np.zeros(len(X_base))

    for train_idx, val_idx in gkf.split(X_base, y_dx, groups):
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)
        model_dx.fit(X_base[train_idx], y_dx[train_idx],
                    eval_set=(X_base[val_idx], y_dx[val_idx]), use_best_model=True)
        model_dy.fit(X_base[train_idx], y_dy[train_idx],
                    eval_set=(X_base[val_idx], y_dy[val_idx]), use_best_model=True)
        oof_dx[val_idx] = model_dx.predict(X_base[val_idx])
        oof_dy[val_idx] = model_dy.predict(X_base[val_idx])

    pred_x = start_xy[:, 0] + oof_dx
    pred_y = start_xy[:, 1] + oof_dy
    baseline_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"  Baseline CV: {baseline_cv:.4f}")

    # With Wyscout zone stats (21 features)
    print("\n[4] With Wyscout zone stats (21 features)...")
    oof_dx = np.zeros(len(X_augmented))
    oof_dy = np.zeros(len(X_augmented))

    for train_idx, val_idx in gkf.split(X_augmented, y_dx, groups):
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)
        model_dx.fit(X_augmented[train_idx], y_dx[train_idx],
                    eval_set=(X_augmented[val_idx], y_dx[val_idx]), use_best_model=True)
        model_dy.fit(X_augmented[train_idx], y_dy[train_idx],
                    eval_set=(X_augmented[val_idx], y_dy[val_idx]), use_best_model=True)
        oof_dx[val_idx] = model_dx.predict(X_augmented[val_idx])
        oof_dy[val_idx] = model_dy.predict(X_augmented[val_idx])

    pred_x = start_xy[:, 0] + oof_dx
    pred_y = start_xy[:, 1] + oof_dy
    augmented_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"  Augmented CV: {augmented_cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Baseline (15 features):      CV {baseline_cv:.4f}")
    print(f"  + Wyscout zone stats (21):   CV {augmented_cv:.4f} ({augmented_cv - baseline_cv:+.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
