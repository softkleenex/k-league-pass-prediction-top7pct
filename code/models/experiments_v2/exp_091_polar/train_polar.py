"""
exp_091: Polar Coordinate Prediction (distance + angle)
Instead of dx, dy → predict distance and angle
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

def create_base_features(df, zone_size=6):
    df['zone_x'] = (df['start_x'] / (105/zone_size)).astype(int).clip(0, zone_size-1)
    df['zone_y'] = (df['start_y'] / (68/zone_size)).astype(int).clip(0, zone_size-1)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    # Polar coordinates of pass
    df['pass_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['pass_angle'] = np.arctan2(df['dy'], df['dx'])  # radians

    return df

def extract_features(episode_df):
    episode_df = episode_df.sort_values('action_id').reset_index(drop=True)
    last = episode_df.iloc[-1]

    features = {
        'game_episode': last['game_episode'],
        'game_id': last['game_id'],
        'start_x': last['start_x'],
        'start_y': last['start_y'],
        'end_x': last['end_x'],
        'end_y': last['end_y'],
        'dx': last['dx'],
        'dy': last['dy'],
        'pass_distance': last['pass_distance'],
        'pass_angle': last['pass_angle'],
        # Features
        'zone_x': last['zone_x'],
        'zone_y': last['zone_y'],
        'goal_distance': last['goal_distance'],
        'goal_angle': last['goal_angle'],
        'dist_to_goal_line': last['dist_to_goal_line'],
        'dist_to_center_y': last['dist_to_center_y'],
        'result_encoded': last['result_encoded'],
    }

    if len(episode_df) > 1:
        prev = episode_df.iloc[-2]
        features['prev_dx'] = prev['dx']
        features['prev_dy'] = prev['dy']
        features['prev_dist'] = prev['pass_distance']
        features['prev_angle'] = prev['pass_angle']
    else:
        features['prev_dx'] = 0
        features['prev_dy'] = 0
        features['prev_dist'] = 0
        features['prev_angle'] = 0

    return features

def process_data(df):
    all_features = []
    for game_ep, ep_df in df.groupby('game_episode'):
        features = extract_features(ep_df)
        all_features.append(features)
    return pd.DataFrame(all_features)

FEATURE_COLS = [
    'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y',
    'zone_x', 'result_encoded', 'prev_dx', 'prev_dy',
]

FEATURE_COLS_POLAR = FEATURE_COLS + ['prev_dist', 'prev_angle']

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def run_delta_experiment(X, y_delta, y_abs, start_xy, groups, n_splits, lr, n_seeds):
    """Standard delta prediction"""
    seeds = SEED_POOL[:n_seeds]
    all_oof = []

    for seed in seeds:
        params = {
            'iterations': 1000, 'depth': 8, 'learning_rate': lr,
            'l2_leaf_reg': 3.0, 'random_state': seed, 'verbose': 0,
            'early_stopping_rounds': 50, 'loss_function': 'MAE'
        }

        gkf = GroupKFold(n_splits=n_splits)
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
        all_oof.append(pred_abs.copy())
        gc.collect()

    ensemble_pred = np.mean(all_oof, axis=0)
    cv = np.sqrt((ensemble_pred[:, 0] - y_abs[:, 0])**2 + (ensemble_pred[:, 1] - y_abs[:, 1])**2).mean()
    return cv

def run_polar_experiment(X, y_polar, y_abs, start_xy, groups, n_splits, lr, n_seeds):
    """Polar coordinate prediction (distance + angle)"""
    seeds = SEED_POOL[:n_seeds]
    all_oof = []

    for seed in seeds:
        params = {
            'iterations': 1000, 'depth': 8, 'learning_rate': lr,
            'l2_leaf_reg': 3.0, 'random_state': seed, 'verbose': 0,
            'early_stopping_rounds': 50, 'loss_function': 'MAE'
        }

        gkf = GroupKFold(n_splits=n_splits)
        oof_polar = np.zeros((len(X), 2))

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_polar, groups), 1):
            model_dist = CatBoostRegressor(**params)
            model_angle = CatBoostRegressor(**params)
            model_dist.fit(X[train_idx], y_polar[train_idx, 0],
                          eval_set=(X[val_idx], y_polar[val_idx, 0]), use_best_model=True)
            model_angle.fit(X[train_idx], y_polar[train_idx, 1],
                           eval_set=(X[val_idx], y_polar[val_idx, 1]), use_best_model=True)
            oof_polar[val_idx, 0] = model_dist.predict(X[val_idx])
            oof_polar[val_idx, 1] = model_angle.predict(X[val_idx])
            del model_dist, model_angle

        # Convert polar back to cartesian
        pred_dx = oof_polar[:, 0] * np.cos(oof_polar[:, 1])
        pred_dy = oof_polar[:, 0] * np.sin(oof_polar[:, 1])
        pred_abs = np.zeros((len(X), 2))
        pred_abs[:, 0] = start_xy[:, 0] + pred_dx
        pred_abs[:, 1] = start_xy[:, 1] + pred_dy
        all_oof.append(pred_abs.copy())
        gc.collect()

    ensemble_pred = np.mean(all_oof, axis=0)
    cv = np.sqrt((ensemble_pred[:, 0] - y_abs[:, 0])**2 + (ensemble_pred[:, 1] - y_abs[:, 1])**2).mean()
    return cv

def main():
    print("=" * 70)
    print("exp_091: Polar Coordinate Prediction")
    print("=" * 70)

    print("\n[1] Loading data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Raw: {len(train_raw)} rows")

    print("\n[2] Creating features...")
    train_with_features = create_base_features(train_raw)

    print("\n[3] Processing episodes...")
    processed_df = process_data(train_with_features)
    print(f"  Episodes: {len(processed_df)}")
    del train_raw; gc.collect()

    # Prepare data
    X = processed_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_polar = processed_df[FEATURE_COLS_POLAR].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    y_delta = processed_df[['dx', 'dy']].values.astype(np.float32)
    y_polar = processed_df[['pass_distance', 'pass_angle']].values.astype(np.float32)
    y_abs = processed_df[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = processed_df[['start_x', 'start_y']].values.astype(np.float32)
    groups = processed_df['game_id'].values

    results = {}
    baseline_cv = 13.5358

    print("\n[4] Delta prediction (baseline)")
    cv = run_delta_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, 7)
    results['delta'] = cv
    print(f"  CV: {cv:.4f}")

    print("\n[5] Polar prediction (distance + angle)")
    cv = run_polar_experiment(X, y_polar, y_abs, start_xy, groups, 11, 0.05, 7)
    results['polar'] = cv
    print(f"  CV: {cv:.4f}")

    print("\n[6] Polar with polar features")
    cv = run_polar_experiment(X_polar, y_polar, y_abs, start_xy, groups, 11, 0.05, 7)
    results['polar_features'] = cv
    print(f"  CV: {cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Best LB: {baseline_cv:.4f}")
    print("-" * 70)

    for name, cv in sorted(results.items(), key=lambda x: x[1]):
        diff = cv - baseline_cv
        marker = " *** IMPROVED! ***" if diff < 0 else ""
        print(f"  {name:20s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    print("=" * 70)

if __name__ == "__main__":
    main()
