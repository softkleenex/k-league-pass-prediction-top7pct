"""
exp_093: Pre-training with Wyscout + Fine-tuning on K-League
1.5M passes from 5 major leagues → Pre-train
K-League 15K passes → Fine-tune
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
SUBMISSION_DIR = BASE / "submissions"

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
    combined['start_x'] = combined['pos_orig_x'] * 1.05  # 0-100 → 0-105
    combined['start_y'] = combined['pos_orig_y'] * 0.68  # 0-100 → 0-68
    combined['end_x'] = combined['pos_dest_x'] * 1.05
    combined['end_y'] = combined['pos_dest_y'] * 0.68

    # Calculate dx, dy
    combined['dx'] = combined['end_x'] - combined['start_x']
    combined['dy'] = combined['end_y'] - combined['start_y']

    # Create basic features
    combined['zone_x'] = (combined['start_x'] / (105/6)).astype(int).clip(0, 5)
    combined['zone_y'] = (combined['start_y'] / (68/6)).astype(int).clip(0, 5)
    combined['goal_distance'] = np.sqrt((105 - combined['start_x'])**2 + (34 - combined['start_y'])**2)
    combined['goal_angle'] = np.degrees(np.arctan2(34 - combined['start_y'], 105 - combined['start_x']))
    combined['dist_to_goal_line'] = 105 - combined['start_x']
    combined['dist_to_center_y'] = np.abs(combined['start_y'] - 34)

    # Remove invalid data
    combined = combined.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])
    combined = combined[(combined['end_x'] >= 0) & (combined['end_x'] <= 105)]
    combined = combined[(combined['end_y'] >= 0) & (combined['end_y'] <= 68)]

    return combined

def create_kleague_features(df):
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
    return df

# Shared features between Wyscout and K-League
SHARED_FEATURES = ['zone_x', 'zone_y', 'goal_distance', 'goal_angle',
                   'dist_to_goal_line', 'dist_to_center_y']

# K-League only features
KLEAGUE_FEATURES = SHARED_FEATURES + ['prev_dx', 'prev_dy', 'ema_start_x', 'ema_start_y',
                                       'ema_success_rate', 'ema_possession', 'result_encoded',
                                       'diff_x', 'velocity']

def main():
    print("=" * 70)
    print("exp_093: Pre-training with Wyscout")
    print("=" * 70)

    # Load Wyscout data
    print("\n[1] Loading Wyscout passes...")
    wyscout = load_wyscout_passes()
    print(f"  Wyscout passes: {len(wyscout):,}")

    # Sample for faster training (use 500K)
    if len(wyscout) > 500000:
        wyscout = wyscout.sample(500000, random_state=42)
        print(f"  Sampled: {len(wyscout):,}")

    X_wyscout = wyscout[SHARED_FEATURES].fillna(0).values.astype(np.float32)
    y_wyscout_dx = wyscout['dx'].values.astype(np.float32)
    y_wyscout_dy = wyscout['dy'].values.astype(np.float32)

    # Load K-League data
    print("\n[2] Loading K-League data...")
    kleague = pd.read_csv(DATA_DIR / 'train.csv')
    kleague = create_kleague_features(kleague)
    last_passes = kleague.groupby('game_episode').last().reset_index()
    del kleague; gc.collect()

    X_kleague = last_passes[KLEAGUE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_kleague_shared = last_passes[SHARED_FEATURES].fillna(0).values.astype(np.float32)
    y_kleague_dx = last_passes['dx'].values.astype(np.float32)
    y_kleague_dy = last_passes['dy'].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    print(f"  K-League episodes: {len(last_passes):,}")

    # Baseline: K-League only
    print("\n[3] Baseline (K-League only, 15 features)...")
    params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
              'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
              'early_stopping_rounds': 50, 'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=5)
    oof_dx = np.zeros(len(X_kleague))
    oof_dy = np.zeros(len(X_kleague))

    for train_idx, val_idx in gkf.split(X_kleague, y_kleague_dx, groups):
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)
        model_dx.fit(X_kleague[train_idx], y_kleague_dx[train_idx],
                    eval_set=(X_kleague[val_idx], y_kleague_dx[val_idx]), use_best_model=True)
        model_dy.fit(X_kleague[train_idx], y_kleague_dy[train_idx],
                    eval_set=(X_kleague[val_idx], y_kleague_dy[val_idx]), use_best_model=True)
        oof_dx[val_idx] = model_dx.predict(X_kleague[val_idx])
        oof_dy[val_idx] = model_dy.predict(X_kleague[val_idx])

    pred_x = start_xy[:, 0] + oof_dx
    pred_y = start_xy[:, 1] + oof_dy
    baseline_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"  Baseline CV: {baseline_cv:.4f}")

    # Pre-train on Wyscout
    print("\n[4] Pre-training on Wyscout (shared features)...")
    pretrain_params = {'iterations': 500, 'depth': 6, 'learning_rate': 0.1,
                       'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 100,
                       'loss_function': 'MAE'}

    pretrain_dx = CatBoostRegressor(**pretrain_params)
    pretrain_dy = CatBoostRegressor(**pretrain_params)
    pretrain_dx.fit(X_wyscout, y_wyscout_dx)
    pretrain_dy.fit(X_wyscout, y_wyscout_dy)

    # Get Wyscout predictions on K-League as features
    print("\n[5] Creating pre-trained features...")
    pretrain_pred_dx = pretrain_dx.predict(X_kleague_shared)
    pretrain_pred_dy = pretrain_dy.predict(X_kleague_shared)

    # Add as features
    X_augmented = np.column_stack([X_kleague, pretrain_pred_dx, pretrain_pred_dy])
    print(f"  Augmented features: {X_augmented.shape[1]}")

    # Train with augmented features
    print("\n[6] Training with pre-trained features...")
    oof_dx = np.zeros(len(X_augmented))
    oof_dy = np.zeros(len(X_augmented))

    for train_idx, val_idx in gkf.split(X_augmented, y_kleague_dx, groups):
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)
        model_dx.fit(X_augmented[train_idx], y_kleague_dx[train_idx],
                    eval_set=(X_augmented[val_idx], y_kleague_dx[val_idx]), use_best_model=True)
        model_dy.fit(X_augmented[train_idx], y_kleague_dy[train_idx],
                    eval_set=(X_augmented[val_idx], y_kleague_dy[val_idx]), use_best_model=True)
        oof_dx[val_idx] = model_dx.predict(X_augmented[val_idx])
        oof_dy[val_idx] = model_dy.predict(X_augmented[val_idx])

    pred_x = start_xy[:, 0] + oof_dx
    pred_y = start_xy[:, 1] + oof_dy
    pretrain_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"  Pre-trained CV: {pretrain_cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Baseline (K-League only): CV {baseline_cv:.4f}")
    print(f"  + Wyscout pre-train:      CV {pretrain_cv:.4f} ({pretrain_cv - baseline_cv:+.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
