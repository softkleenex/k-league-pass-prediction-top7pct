"""
exp_101: Post-processing Optimization
- Test different clipping strategies
- Test blend with simple baseline
- Test prediction smoothing
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

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def main():
    print("=" * 70)
    print("exp_101: Post-processing Optimization")
    print("=" * 70)

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    del train_df; gc.collect()

    X = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = last_passes['dx'].values.astype(np.float32)
    y_dy = last_passes['dy'].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    print(f"Samples: {len(X)}")

    # Get OOF predictions (7 seeds)
    print("\n[1] Getting OOF predictions...")
    params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
              'l2_leaf_reg': 3.0, 'verbose': 0, 'early_stopping_rounds': 50,
              'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=11)
    all_oof_dx = []
    all_oof_dy = []

    for seed in SEED_POOL[:7]:
        p = params.copy()
        p['random_state'] = seed
        oof_dx = np.zeros(len(X))
        oof_dy = np.zeros(len(X))

        for train_idx, val_idx in gkf.split(X, y_dx, groups):
            model_dx = CatBoostRegressor(**p)
            model_dy = CatBoostRegressor(**p)
            model_dx.fit(X[train_idx], y_dx[train_idx],
                        eval_set=(X[val_idx], y_dx[val_idx]), use_best_model=True)
            model_dy.fit(X[train_idx], y_dy[train_idx],
                        eval_set=(X[val_idx], y_dy[val_idx]), use_best_model=True)
            oof_dx[val_idx] = model_dx.predict(X[val_idx])
            oof_dy[val_idx] = model_dy.predict(X[val_idx])
            del model_dx, model_dy

        all_oof_dx.append(oof_dx)
        all_oof_dy.append(oof_dy)
        gc.collect()

    oof_dx = np.mean(all_oof_dx, axis=0)
    oof_dy = np.mean(all_oof_dy, axis=0)

    # Baseline CV
    pred_x = start_xy[:, 0] + oof_dx
    pred_y = start_xy[:, 1] + oof_dy
    baseline_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"  Baseline CV: {baseline_cv:.4f}")

    # Simple baseline (zone mean)
    zone_mean_dx = last_passes.groupby(['zone_x', 'zone_y'])['dx'].transform('mean')
    zone_mean_dy = last_passes.groupby(['zone_x', 'zone_y'])['dy'].transform('mean')

    results = {'baseline': baseline_cv}

    # Test different post-processing
    print("\n[2] Testing post-processing strategies...")

    # 1. Clip end coordinates
    for clip_margin in [0, 2, 5]:
        pred_x_clipped = np.clip(start_xy[:, 0] + oof_dx, clip_margin, 105 - clip_margin)
        pred_y_clipped = np.clip(start_xy[:, 1] + oof_dy, clip_margin, 68 - clip_margin)
        cv = np.sqrt((pred_x_clipped - y_abs[:, 0])**2 + (pred_y_clipped - y_abs[:, 1])**2).mean()
        results[f'clip_{clip_margin}'] = cv
        print(f"  clip_{clip_margin}: CV {cv:.4f}")

    # 2. Blend with zone mean
    for alpha in [0.9, 0.95, 0.98]:
        blended_dx = alpha * oof_dx + (1 - alpha) * zone_mean_dx.values
        blended_dy = alpha * oof_dy + (1 - alpha) * zone_mean_dy.values
        pred_x = start_xy[:, 0] + blended_dx
        pred_y = start_xy[:, 1] + blended_dy
        cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
        results[f'blend_{alpha}'] = cv
        print(f"  blend_{alpha}: CV {cv:.4f}")

    # 3. Shrink predictions toward mean
    mean_dx = oof_dx.mean()
    mean_dy = oof_dy.mean()
    for shrink in [0.95, 0.98, 1.02, 1.05]:
        shrunk_dx = mean_dx + shrink * (oof_dx - mean_dx)
        shrunk_dy = mean_dy + shrink * (oof_dy - mean_dy)
        pred_x = start_xy[:, 0] + shrunk_dx
        pred_y = start_xy[:, 1] + shrunk_dy
        cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
        results[f'shrink_{shrink}'] = cv
        print(f"  shrink_{shrink}: CV {cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS (sorted by CV)")
    print("=" * 70)
    for name, cv in sorted(results.items(), key=lambda x: x[1]):
        diff = cv - baseline_cv
        print(f"  {name:15s}: CV {cv:.4f} ({diff:+.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
