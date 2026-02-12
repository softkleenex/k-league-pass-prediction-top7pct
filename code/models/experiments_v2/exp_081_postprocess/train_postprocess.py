"""
exp_081: Post-Processing Optimization
- Y clipping range optimization
- X clipping test
- Prediction smoothing
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

def main():
    print("=" * 70)
    print("exp_081: Post-Processing Optimization")
    print("=" * 70)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    del train_df; gc.collect()

    TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
              'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
              'ema_start_y', 'ema_success_rate', 'ema_possession',
              'zone_x', 'result_encoded', 'diff_x', 'velocity']

    X = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = last_passes[['dx', 'dy']].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
              'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
              'early_stopping_rounds': 50, 'loss_function': 'MAE'}

    # Get OOF predictions first
    print("\n[1] Getting OOF predictions...")
    gkf = GroupKFold(n_splits=3)
    oof_delta = np.zeros((len(X), 2))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_delta, groups), 1):
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)
        model_dx.fit(X[train_idx], y_delta[train_idx, 0],
                    eval_set=(X[val_idx], y_delta[val_idx, 0]), use_best_model=True)
        model_dy.fit(X[train_idx], y_delta[train_idx, 1],
                    eval_set=(X[val_idx], y_delta[val_idx, 1]), use_best_model=True)
        oof_delta[val_idx, 0] = model_dx.predict(X[val_idx])
        oof_delta[val_idx, 1] = model_dy.predict(X[val_idx])
        del model_dx, model_dy

    raw_pred_x = start_xy[:, 0] + oof_delta[:, 0]
    raw_pred_y = start_xy[:, 1] + oof_delta[:, 1]

    results = {}

    # Baseline (no clipping)
    print("\n[2] Testing clipping options...")
    cv_no_clip = np.sqrt((raw_pred_x - y_abs[:, 0])**2 + (raw_pred_y - y_abs[:, 1])**2).mean()
    results['no_clip'] = cv_no_clip
    print(f"  No clipping: CV {cv_no_clip:.4f}")

    # Y clipping tests
    for y_min, y_max in [(0, 68), (-5, 73), (2, 66), (5, 63)]:
        pred_y_clipped = np.clip(raw_pred_y, y_min, y_max)
        cv = np.sqrt((raw_pred_x - y_abs[:, 0])**2 + (pred_y_clipped - y_abs[:, 1])**2).mean()
        results[f'y_clip_{y_min}_{y_max}'] = cv
        print(f"  Y clip [{y_min}, {y_max}]: CV {cv:.4f}")

    # X clipping tests
    for x_min, x_max in [(0, 105), (-10, 115), (10, 100)]:
        pred_x_clipped = np.clip(raw_pred_x, x_min, x_max)
        pred_y_clipped = np.clip(raw_pred_y, 0, 68)
        cv = np.sqrt((pred_x_clipped - y_abs[:, 0])**2 + (pred_y_clipped - y_abs[:, 1])**2).mean()
        results[f'xy_clip_x{x_min}_{x_max}'] = cv
        print(f"  X clip [{x_min}, {x_max}] + Y [0,68]: CV {cv:.4f}")

    # Quantile-based clipping
    print("\n[3] Quantile-based clipping...")
    for q in [0.01, 0.02, 0.05]:
        x_low, x_high = np.percentile(raw_pred_x, [q*100, (1-q)*100])
        y_low, y_high = np.percentile(raw_pred_y, [q*100, (1-q)*100])
        pred_x_q = np.clip(raw_pred_x, x_low, x_high)
        pred_y_q = np.clip(raw_pred_y, max(0, y_low), min(68, y_high))
        cv = np.sqrt((pred_x_q - y_abs[:, 0])**2 + (pred_y_q - y_abs[:, 1])**2).mean()
        results[f'quantile_{q}'] = cv
        print(f"  Quantile {q}: CV {cv:.4f}")

    # Blending with mean prediction
    print("\n[4] Blending with mean...")
    mean_end_x = y_abs[:, 0].mean()
    mean_end_y = y_abs[:, 1].mean()
    for alpha in [0.95, 0.9, 0.85]:
        blended_x = alpha * raw_pred_x + (1 - alpha) * mean_end_x
        blended_y = alpha * raw_pred_y + (1 - alpha) * mean_end_y
        blended_y = np.clip(blended_y, 0, 68)
        cv = np.sqrt((blended_x - y_abs[:, 0])**2 + (blended_y - y_abs[:, 1])**2).mean()
        results[f'blend_{alpha}'] = cv
        print(f"  Blend alpha={alpha}: CV {cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary (sorted by CV):")
    print("=" * 70)
    baseline = results['y_clip_0_68']
    for name, cv in sorted(results.items(), key=lambda x: x[1])[:10]:
        diff = cv - baseline
        marker = " ★" if cv == min(results.values()) else ""
        print(f"  {name:25s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    best_name = min(results, key=results.get)
    best_cv = results[best_name]
    print(f"\n  Best: {best_name} (CV {best_cv:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
