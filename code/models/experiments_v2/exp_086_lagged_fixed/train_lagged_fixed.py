"""
exp_086: Fixed Lagged Features - NO TARGET LEAKAGE
Key fix: history = episode_df.iloc[:-1] (exclude last pass from all statistics)
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

def create_base_features(df):
    """Create base features for each row"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']

    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)

    return df

def create_lagged_features_fixed(episode_df):
    """
    Create lagged features WITHOUT target leakage.
    KEY FIX: history = episode_df.iloc[:-1] (exclude last pass!)
    """
    episode_df = episode_df.sort_values('action_id').reset_index(drop=True)
    n = len(episode_df)

    # Target pass (last row) - what we predict
    last = episode_df.iloc[-1]

    # History (all passes BEFORE target) - SAFE to use for features
    history = episode_df.iloc[:-1]
    n_history = len(history)

    features = {
        'game_episode': last['game_episode'],
        'game_id': last['game_id'],
        # Targets (for training only)
        'dx': last['dx'],
        'dy': last['dy'],
        'end_x': last['end_x'],
        'end_y': last['end_y'],
        # Features available at inference
        'start_x': last['start_x'],
        'start_y': last['start_y'],
        'zone_x': last['zone_x'],
        'zone_y': last['zone_y'],
        'goal_distance': last['goal_distance'],
        'goal_angle': last['goal_angle'],
        'dist_to_goal_line': last['dist_to_goal_line'],
        'dist_to_center_y': last['dist_to_center_y'],
    }

    # Episode length (including target)
    features['episode_length'] = n

    # Lagged features from HISTORY only
    for lag in [1, 2, 3]:
        if n_history >= lag:
            prev = history.iloc[-lag]
            features[f'prev{lag}_dx'] = prev['dx']
            features[f'prev{lag}_dy'] = prev['dy']
            features[f'prev{lag}_start_x'] = prev['start_x']
            features[f'prev{lag}_start_y'] = prev['start_y']
            features[f'prev{lag}_end_x'] = prev['end_x']
            features[f'prev{lag}_end_y'] = prev['end_y']
            features[f'prev{lag}_result'] = prev['result_encoded']
            features[f'prev{lag}_goal_dist'] = prev['goal_distance']
            features[f'prev{lag}_same_team'] = 1 if prev['team_id'] == last['team_id'] else 0
        else:
            features[f'prev{lag}_dx'] = 0
            features[f'prev{lag}_dy'] = 0
            features[f'prev{lag}_start_x'] = last['start_x']
            features[f'prev{lag}_start_y'] = last['start_y']
            features[f'prev{lag}_end_x'] = last['start_x']
            features[f'prev{lag}_end_y'] = last['start_y']
            features[f'prev{lag}_result'] = 2
            features[f'prev{lag}_goal_dist'] = last['goal_distance']
            features[f'prev{lag}_same_team'] = 1

    # Episode statistics from HISTORY only (NO TARGET LEAKAGE!)
    if n_history > 0:
        features['avg_dx'] = history['dx'].mean()
        features['avg_dy'] = history['dy'].mean()
        features['std_dx'] = history['dx'].std() if n_history > 1 else 0
        features['std_dy'] = history['dy'].std() if n_history > 1 else 0
        features['total_distance'] = np.sqrt(history['dx']**2 + history['dy']**2).sum()
        features['avg_goal_dist'] = history['goal_distance'].mean()

        angles = np.arctan2(history['dy'].values, history['dx'].values)
        if len(angles) > 1:
            features['avg_angle_change'] = np.mean(np.abs(np.diff(angles)))
        else:
            features['avg_angle_change'] = 0

        features['episode_success_rate'] = (history['result_encoded'] == 0).mean()
        features['team_possession_ratio'] = (history['team_id'] == last['team_id']).mean()
    else:
        # First pass of episode (no history)
        features['avg_dx'] = 0
        features['avg_dy'] = 0
        features['std_dx'] = 0
        features['std_dy'] = 0
        features['total_distance'] = 0
        features['avg_goal_dist'] = last['goal_distance']
        features['avg_angle_change'] = 0
        features['episode_success_rate'] = 0.5
        features['team_possession_ratio'] = 1.0

    return features

def process_data(df):
    """Process all episodes"""
    df = create_base_features(df)
    all_features = []
    for game_ep, ep_df in df.groupby('game_episode'):
        features = create_lagged_features_fixed(ep_df)
        all_features.append(features)
    return pd.DataFrame(all_features)

# Feature columns
FEATURE_COLS = [
    'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y',
    'zone_x', 'episode_length',
    'prev1_dx', 'prev1_dy', 'prev1_start_x', 'prev1_start_y', 'prev1_end_x', 'prev1_end_y',
    'prev1_result', 'prev1_goal_dist', 'prev1_same_team',
    'prev2_dx', 'prev2_dy', 'prev2_start_x', 'prev2_start_y', 'prev2_result', 'prev2_same_team',
    'prev3_dx', 'prev3_dy', 'prev3_result', 'prev3_same_team',
    'avg_dx', 'avg_dy', 'std_dx', 'std_dy', 'total_distance', 'avg_goal_dist',
    'avg_angle_change', 'episode_success_rate', 'team_possession_ratio',
]

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def run_experiment(X, y_delta, y_abs, start_xy, groups, n_splits, lr, n_seeds):
    """Run experiment with given settings"""
    seeds = SEED_POOL[:n_seeds]
    all_oof = []

    for seed in seeds:
        params = {
            'iterations': 1000,
            'depth': 8,
            'learning_rate': lr,
            'l2_leaf_reg': 3.0,
            'random_state': seed,
            'verbose': 0,
            'early_stopping_rounds': 50,
            'loss_function': 'MAE'
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

def main():
    print("=" * 70)
    print("exp_086: Fixed Lagged Features (NO TARGET LEAKAGE)")
    print("=" * 70)

    print("\n[1] Loading and processing training data...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Raw data: {len(train_df)} rows")

    processed_df = process_data(train_df)
    print(f"  Processed: {len(processed_df)} episodes")
    del train_df; gc.collect()

    X = processed_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = processed_df[['dx', 'dy']].values.astype(np.float32)
    y_abs = processed_df[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = processed_df[['start_x', 'start_y']].values.astype(np.float32)
    groups = processed_df['game_id'].values

    print(f"  Features: {len(FEATURE_COLS)}")

    results = {}
    baseline_cv = 13.5435  # exp_083 best

    # Test with best hyperparameters from exp_083
    print("\n[2] Testing Fixed Lagged Features (fold=11, lr=0.05, 7seeds)")
    cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, 7)
    results['fixed_fold11_7seeds'] = cv
    diff = cv - baseline_cv
    print(f"  CV: {cv:.4f} (vs baseline {baseline_cv:.4f}: {diff:+.4f})")

    # Test with 3 seeds for comparison
    print("\n[3] Testing Fixed Lagged Features (fold=11, lr=0.05, 3seeds)")
    cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, 3)
    results['fixed_fold11_3seeds'] = cv
    print(f"  CV: {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Baseline (exp_083, no lagged): CV {baseline_cv:.4f}")
    print(f"  exp_085 (leaky lagged):        CV 11.7478 -> LB 16.29 (FAILED)")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for name, cv in sorted_results:
        diff = cv - baseline_cv
        print(f"  {name:30s}: CV {cv:.4f} ({diff:+.4f})")

    best_name, best_cv = sorted_results[0]
    print("\n" + "=" * 70)

    if best_cv < baseline_cv:
        print(f"  IMPROVEMENT: {best_name}")
        print(f"  CV: {best_cv:.4f} (better than baseline by {baseline_cv - best_cv:.4f})")
        print("  Creating submission...")
        create_submission(best_name, best_cv, processed_df)
    else:
        print(f"  No improvement over baseline")
        print(f"  But CV should now match LB (no leakage)")
        print("  Creating submission anyway to verify...")
        create_submission(best_name, best_cv, processed_df)
    print("=" * 70)

def create_submission(name, cv, train_processed_df):
    """Create submission"""
    n_splits = 11
    n_seeds = 7 if '7seeds' in name else 3
    seeds = SEED_POOL[:n_seeds]
    lr = 0.05

    X = train_processed_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = train_processed_df[['dx', 'dy']].values.astype(np.float32)
    groups = train_processed_df['game_id'].values

    all_models_dx = []
    all_models_dy = []

    for seed in seeds:
        params = {
            'iterations': 1000, 'depth': 8, 'learning_rate': lr,
            'l2_leaf_reg': 3.0, 'random_state': seed, 'verbose': 0,
            'early_stopping_rounds': 50, 'loss_function': 'MAE'
        }

        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(X, y_delta, groups):
            model_dx = CatBoostRegressor(**params)
            model_dy = CatBoostRegressor(**params)
            model_dx.fit(X[train_idx], y_delta[train_idx, 0],
                        eval_set=(X[val_idx], y_delta[val_idx, 0]), use_best_model=True)
            model_dy.fit(X[train_idx], y_delta[train_idx, 1],
                        eval_set=(X[val_idx], y_delta[val_idx, 1]), use_best_model=True)
            all_models_dx.append(model_dx)
            all_models_dy.append(model_dy)

    # Process test data
    print("  Processing test data...")
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    test_episodes = []
    for _, row in test_df.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)
    test_all = pd.concat(test_episodes, ignore_index=True)
    test_processed = process_data(test_all)

    X_test = test_processed[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    pred_dx = np.mean([m.predict(X_test) for m in all_models_dx], axis=0)
    pred_dy = np.mean([m.predict(X_test) for m in all_models_dy], axis=0)
    pred_x = test_processed['start_x'].values + pred_dx
    pred_y = np.clip(test_processed['start_y'].values + pred_dy, 0, 68)

    submission = pd.DataFrame({
        'game_episode': test_processed['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })
    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_lagged_fixed_cv{cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

if __name__ == "__main__":
    main()
