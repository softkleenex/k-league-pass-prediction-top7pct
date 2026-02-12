"""
exp_080: Hyperparameter Tuning for Delta Prediction
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

def run_cv(X, y_delta, y_abs, start_xy, groups, params, name):
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

    pred_abs = np.zeros((len(X), 2))
    pred_abs[:, 0] = start_xy[:, 0] + oof_delta[:, 0]
    pred_abs[:, 1] = start_xy[:, 1] + oof_delta[:, 1]
    cv = np.sqrt((pred_abs[:, 0] - y_abs[:, 0])**2 + (pred_abs[:, 1] - y_abs[:, 1])**2).mean()
    gc.collect()
    return cv

def main():
    print("=" * 70)
    print("exp_080: Hyperparameter Tuning")
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

    # Baseline params
    baseline_params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                       'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
                       'early_stopping_rounds': 50, 'loss_function': 'MAE'}

    results = {}

    # Baseline
    print("\n[Baseline]")
    cv_base = run_cv(X, y_delta, y_abs, start_xy, groups, baseline_params, "baseline")
    results['baseline'] = cv_base
    print(f"  CV: {cv_base:.4f}")

    # Test different depths
    print("\n[Depth]")
    for depth in [6, 7, 9, 10]:
        params = baseline_params.copy()
        params['depth'] = depth
        cv = run_cv(X, y_delta, y_abs, start_xy, groups, params, f"depth_{depth}")
        results[f'depth_{depth}'] = cv
        print(f"  depth={depth}: CV {cv:.4f}")

    # Test different learning rates
    print("\n[Learning Rate]")
    for lr in [0.01, 0.03, 0.07, 0.1]:
        params = baseline_params.copy()
        params['learning_rate'] = lr
        params['iterations'] = int(1000 * 0.05 / lr)  # Adjust iterations
        cv = run_cv(X, y_delta, y_abs, start_xy, groups, params, f"lr_{lr}")
        results[f'lr_{lr}'] = cv
        print(f"  lr={lr}: CV {cv:.4f}")

    # Test different l2_leaf_reg
    print("\n[L2 Regularization]")
    for l2 in [1.0, 5.0, 7.0, 10.0]:
        params = baseline_params.copy()
        params['l2_leaf_reg'] = l2
        cv = run_cv(X, y_delta, y_abs, start_xy, groups, params, f"l2_{l2}")
        results[f'l2_{l2}'] = cv
        print(f"  l2={l2}: CV {cv:.4f}")

    # Test more iterations
    print("\n[Iterations]")
    for iters in [2000, 3000, 4000]:
        params = baseline_params.copy()
        params['iterations'] = iters
        params['learning_rate'] = 0.03
        cv = run_cv(X, y_delta, y_abs, start_xy, groups, params, f"iter_{iters}")
        results[f'iter_{iters}'] = cv
        print(f"  iterations={iters}: CV {cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary (sorted by CV):")
    print("=" * 70)
    for name, cv in sorted(results.items(), key=lambda x: x[1])[:10]:
        diff = cv - cv_base
        marker = " ★" if cv == min(results.values()) else ""
        print(f"  {name:20s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    best_name = min(results, key=results.get)
    best_cv = results[best_name]
    print(f"\n  Best: {best_name} (CV {best_cv:.4f})")
    print(f"  vs Baseline: {best_cv - cv_base:+.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
