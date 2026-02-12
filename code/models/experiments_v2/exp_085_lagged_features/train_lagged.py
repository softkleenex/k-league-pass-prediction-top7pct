"""
exp_085: Lagged Features - Use information from previous passes
Key insight from Gemini: Using only last pass = reading only last sentence of a story

New features:
1. Lagged features from pass_i-1, pass_i-2, pass_i-3
2. Episode-level aggregated statistics
3. Team change indicators
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

def create_lagged_features(episode_df):
    """Create lagged features for an episode"""
    episode_df = episode_df.sort_values('action_id').reset_index(drop=True)
    n = len(episode_df)

    # Last row info
    last = episode_df.iloc[-1]
    features = {
        'game_episode': last['game_episode'],
        'game_id': last['game_id'],
        # Target
        'dx': last['dx'],
        'dy': last['dy'],
        'end_x': last['end_x'],
        'end_y': last['end_y'],
        'start_x': last['start_x'],
        'start_y': last['start_y'],
        # Last pass features (existing)
        'zone_x': last['zone_x'],
        'zone_y': last['zone_y'],
        'goal_distance': last['goal_distance'],
        'goal_angle': last['goal_angle'],
        'dist_to_goal_line': last['dist_to_goal_line'],
        'dist_to_center_y': last['dist_to_center_y'],
        'result_encoded': last['result_encoded'],
    }

    # Episode length
    features['episode_length'] = n

    # Lagged features (previous passes)
    for lag in [1, 2, 3]:
        idx = n - 1 - lag
        if idx >= 0:
            prev = episode_df.iloc[idx]
            features[f'prev{lag}_dx'] = prev['dx']
            features[f'prev{lag}_dy'] = prev['dy']
            features[f'prev{lag}_start_x'] = prev['start_x']
            features[f'prev{lag}_start_y'] = prev['start_y']
            features[f'prev{lag}_end_x'] = prev['end_x']
            features[f'prev{lag}_end_y'] = prev['end_y']
            features[f'prev{lag}_result'] = prev['result_encoded']
            features[f'prev{lag}_goal_dist'] = prev['goal_distance']
            # Team change indicator
            features[f'prev{lag}_same_team'] = 1 if prev['team_id'] == last['team_id'] else 0
        else:
            # Pad with zeros/defaults
            features[f'prev{lag}_dx'] = 0
            features[f'prev{lag}_dy'] = 0
            features[f'prev{lag}_start_x'] = last['start_x']
            features[f'prev{lag}_start_y'] = last['start_y']
            features[f'prev{lag}_end_x'] = last['start_x']
            features[f'prev{lag}_end_y'] = last['start_y']
            features[f'prev{lag}_result'] = 2  # Unknown
            features[f'prev{lag}_goal_dist'] = last['goal_distance']
            features[f'prev{lag}_same_team'] = 1

    # Episode-level aggregated statistics
    if n > 1:
        features['avg_dx'] = episode_df['dx'].mean()
        features['avg_dy'] = episode_df['dy'].mean()
        features['std_dx'] = episode_df['dx'].std()
        features['std_dy'] = episode_df['dy'].std()
        features['total_distance'] = np.sqrt(episode_df['dx']**2 + episode_df['dy']**2).sum()
        features['avg_goal_dist'] = episode_df['goal_distance'].mean()

        # Direction change (angle between consecutive passes)
        angles = np.arctan2(episode_df['dy'].values, episode_df['dx'].values)
        if len(angles) > 1:
            angle_diffs = np.abs(np.diff(angles))
            features['avg_angle_change'] = np.mean(angle_diffs)
        else:
            features['avg_angle_change'] = 0

        # Success rate in episode
        features['episode_success_rate'] = (episode_df['result_encoded'] == 0).mean()

        # Team possession ratio (how much current team had the ball)
        last_team = last['team_id']
        features['team_possession_ratio'] = (episode_df['team_id'] == last_team).mean()
    else:
        features['avg_dx'] = last['dx']
        features['avg_dy'] = last['dy']
        features['std_dx'] = 0
        features['std_dy'] = 0
        features['total_distance'] = np.sqrt(last['dx']**2 + last['dy']**2)
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
        features = create_lagged_features(ep_df)
        all_features.append(features)

    return pd.DataFrame(all_features)

# Feature columns (ordered by expected importance)
LAGGED_FEATURES = [
    # Original top features
    'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y',
    'zone_x', 'result_encoded',
    # Lagged pass features (most important new features)
    'prev1_dx', 'prev1_dy', 'prev1_start_x', 'prev1_start_y', 'prev1_result', 'prev1_same_team', 'prev1_goal_dist',
    'prev2_dx', 'prev2_dy', 'prev2_start_x', 'prev2_start_y', 'prev2_result', 'prev2_same_team',
    'prev3_dx', 'prev3_dy', 'prev3_result', 'prev3_same_team',
    # Episode-level statistics
    'episode_length', 'avg_dx', 'avg_dy', 'std_dx', 'std_dy',
    'total_distance', 'avg_goal_dist', 'avg_angle_change',
    'episode_success_rate', 'team_possession_ratio',
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
    print("exp_085: Lagged Features Experiment")
    print("=" * 70)

    print("\n[1] Loading and processing training data...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Raw data: {len(train_df)} rows")

    processed_df = process_data(train_df)
    print(f"  Processed: {len(processed_df)} episodes")
    del train_df; gc.collect()

    X = processed_df[LAGGED_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = processed_df[['dx', 'dy']].values.astype(np.float32)
    y_abs = processed_df[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = processed_df[['start_x', 'start_y']].values.astype(np.float32)
    groups = processed_df['game_id'].values

    print(f"  Features: {len(LAGGED_FEATURES)}")
    print(f"  Feature list: {LAGGED_FEATURES[:10]}... (and {len(LAGGED_FEATURES)-10} more)")

    results = {}
    baseline_cv = 13.5435  # Current best (fold11, lr0.05, 7seeds)

    # Test 1: Lagged features with best hyperparameters
    print("\n[2] Testing Lagged Features (fold=11, lr=0.05, 7seeds)")
    cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, 7)
    results['lagged_fold11_7seeds'] = cv
    improvement = baseline_cv - cv
    print(f"  CV: {cv:.4f}")
    print(f"  vs baseline (13.5435): {-improvement:+.4f}")
    if improvement > 0:
        print(f"  IMPROVEMENT: {improvement:.4f} points!")

    # Test 2: Try with fewer seeds for speed comparison
    print("\n[3] Testing Lagged Features (fold=11, lr=0.05, 3seeds)")
    cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, 3)
    results['lagged_fold11_3seeds'] = cv
    print(f"  CV: {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Test 3: Try with different fold count
    print("\n[4] Testing Lagged Features (fold=10, lr=0.05, 7seeds)")
    cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 10, 0.05, 7)
    results['lagged_fold10_7seeds'] = cv
    print(f"  CV: {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS (sorted by CV)")
    print("=" * 70)
    print(f"  Baseline (no lagged, fold11_7seeds): CV {baseline_cv:.4f}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for name, cv in sorted_results:
        diff = cv - baseline_cv
        marker = " *** NEW BEST! ***" if diff < -0.01 else (" ** IMPROVED **" if diff < 0 else "")
        print(f"  {name:30s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    best_name = sorted_results[0][0]
    best_cv = sorted_results[0][1]

    print("\n" + "=" * 70)
    if best_cv < baseline_cv:
        print(f"  NEW BEST: {best_name}")
        print(f"  CV: {best_cv:.4f} (improvement: {baseline_cv - best_cv:.4f})")
        print("  Creating submission...")
        create_submission(best_name, best_cv, processed_df)
    else:
        print(f"  No improvement over baseline ({baseline_cv:.4f})")
        print(f"  Best tried: {best_name} (CV {best_cv:.4f})")
    print("=" * 70)

def create_submission(name, cv, train_processed_df):
    """Create submission with lagged features"""
    # Parse settings
    if 'fold11' in name:
        n_splits = 11
    elif 'fold10' in name:
        n_splits = 10
    else:
        n_splits = 11

    if '7seeds' in name:
        n_seeds = 7
    elif '3seeds' in name:
        n_seeds = 3
    else:
        n_seeds = 7

    seeds = SEED_POOL[:n_seeds]
    lr = 0.05

    X = train_processed_df[LAGGED_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = train_processed_df[['dx', 'dy']].values.astype(np.float32)
    groups = train_processed_df['game_id'].values

    all_models_dx = []
    all_models_dy = []

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
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_delta, groups), 1):
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

    X_test = test_processed[LAGGED_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    pred_dx = np.mean([m.predict(X_test) for m in all_models_dx], axis=0)
    pred_dy = np.mean([m.predict(X_test) for m in all_models_dy], axis=0)
    pred_x = test_processed['start_x'].values + pred_dx
    pred_y = np.clip(test_processed['start_y'].values + pred_dy, 0, 68)

    submission = pd.DataFrame({'game_episode': test_processed['game_episode'], 'end_x': pred_x, 'end_y': pred_y})
    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_lagged_cv{cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

if __name__ == "__main__":
    main()
