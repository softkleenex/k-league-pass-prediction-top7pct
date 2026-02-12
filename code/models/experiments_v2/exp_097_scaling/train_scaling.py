"""
exp_097: Prediction Scaling
모델이 variance를 under-estimate하므로 예측을 scaling
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

def main():
    print("=" * 70)
    print("exp_097: Prediction Scaling")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
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

    # Get baseline OOF predictions
    print("\n[2] Training and getting OOF predictions...")
    params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
              'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
              'early_stopping_rounds': 50, 'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X, y_dx, groups))

    oof_dx = np.zeros(len(X))
    oof_dy = np.zeros(len(X))

    for train_idx, val_idx in folds:
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)
        model_dx.fit(X[train_idx], y_dx[train_idx],
                    eval_set=(X[val_idx], y_dx[val_idx]), use_best_model=True)
        model_dy.fit(X[train_idx], y_dy[train_idx],
                    eval_set=(X[val_idx], y_dy[val_idx]), use_best_model=True)
        oof_dx[val_idx] = model_dx.predict(X[val_idx])
        oof_dy[val_idx] = model_dy.predict(X[val_idx])

    # Baseline CV
    pred_x = start_xy[:, 0] + oof_dx
    pred_y = start_xy[:, 1] + oof_dy
    baseline_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"  Baseline CV: {baseline_cv:.4f}")

    # Statistics
    print(f"\n[3] Distribution comparison...")
    print(f"  dx: actual_std={y_dx.std():.2f}, pred_std={oof_dx.std():.2f}")
    print(f"  dy: actual_std={y_dy.std():.2f}, pred_std={oof_dy.std():.2f}")

    # Scale factors
    dx_scale = y_dx.std() / oof_dx.std()
    dy_scale = y_dy.std() / oof_dy.std()
    print(f"\n  Theoretical scale factors: dx={dx_scale:.3f}, dy={dy_scale:.3f}")

    # Try different scaling factors
    print(f"\n[4] Testing scaling factors...")
    best_cv = baseline_cv
    best_scale = (1.0, 1.0)

    for dx_s in np.arange(1.0, 1.6, 0.05):
        for dy_s in np.arange(1.0, 2.5, 0.1):
            scaled_dx = oof_dx * dx_s
            scaled_dy = oof_dy * dy_s
            pred_x = start_xy[:, 0] + scaled_dx
            pred_y = start_xy[:, 1] + scaled_dy
            cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
            if cv < best_cv:
                best_cv = cv
                best_scale = (dx_s, dy_s)

    print(f"  Best scale: dx={best_scale[0]:.2f}, dy={best_scale[1]:.2f}")
    print(f"  Best scaled CV: {best_cv:.4f}")

    # Try with bias correction too
    print(f"\n[5] Testing bias correction...")
    dx_bias = y_dx.mean() - oof_dx.mean()
    dy_bias = y_dy.mean() - oof_dy.mean()
    print(f"  dx bias: {dx_bias:.4f}")
    print(f"  dy bias: {dy_bias:.4f}")

    best_cv2 = best_cv
    best_params = (best_scale[0], best_scale[1], 0, 0)

    for dx_s in np.arange(0.95, 1.55, 0.05):
        for dy_s in np.arange(0.95, 2.5, 0.1):
            for dx_b in np.arange(-2, 3, 0.5):
                for dy_b in np.arange(-1, 2, 0.25):
                    scaled_dx = oof_dx * dx_s + dx_b
                    scaled_dy = oof_dy * dy_s + dy_b
                    pred_x = start_xy[:, 0] + scaled_dx
                    pred_y = start_xy[:, 1] + scaled_dy
                    cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
                    if cv < best_cv2:
                        best_cv2 = cv
                        best_params = (dx_s, dy_s, dx_b, dy_b)

    print(f"  Best params: dx_scale={best_params[0]:.2f}, dy_scale={best_params[1]:.2f}, dx_bias={best_params[2]:.2f}, dy_bias={best_params[3]:.2f}")
    print(f"  Best CV with bias: {best_cv2:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Baseline CV:             {baseline_cv:.4f}")
    print(f"  Best scaling only:       {best_cv:.4f} ({best_cv - baseline_cv:+.4f})")
    print(f"  Best scaling + bias:     {best_cv2:.4f} ({best_cv2 - baseline_cv:+.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
