"""
exp_087: Player-specific Features using LOO encoding
Add player tendencies (avg_dx, avg_dy, success_rate) to baseline features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # code/models
sys.path.insert(0, str(Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models")))
from feature_player_stats import add_player_features_train, add_player_features_test, get_player_stats
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

def extract_last_pass_features(episode_df):
    """Extract features from last pass of each episode"""
    episode_df = episode_df.sort_values('action_id').reset_index(drop=True)
    last = episode_df.iloc[-1]

    features = {
        'game_episode': last['game_episode'],
        'game_id': last['game_id'],
        'dx': last['dx'],
        'dy': last['dy'],
        'end_x': last['end_x'],
        'end_y': last['end_y'],
        'start_x': last['start_x'],
        'start_y': last['start_y'],
        # Baseline features
        'zone_x': last['zone_x'],
        'zone_y': last['zone_y'],
        'goal_distance': last['goal_distance'],
        'goal_angle': last['goal_angle'],
        'dist_to_goal_line': last['dist_to_goal_line'],
        'dist_to_center_y': last['dist_to_center_y'],
        'result_encoded': last['result_encoded'],
        # Player features (LOO encoded)
        'player_avg_dx': last['player_avg_dx'],
        'player_avg_dy': last['player_avg_dy'],
        'player_avg_dist': last['player_avg_dist'],
        'player_success_rate': last['player_success_rate'],
        'player_pass_count': last['player_pass_count'],
        'player_preferred_angle': last['player_preferred_angle'],
    }

    # Previous pass features (from second-to-last)
    if len(episode_df) > 1:
        prev = episode_df.iloc[-2]
        features['prev_dx'] = prev['dx']
        features['prev_dy'] = prev['dy']
    else:
        features['prev_dx'] = 0
        features['prev_dy'] = 0

    return features

def process_data(df):
    """Process all episodes"""
    all_features = []
    for game_ep, ep_df in df.groupby('game_episode'):
        features = extract_last_pass_features(ep_df)
        all_features.append(features)
    return pd.DataFrame(all_features)

# Feature columns
FEATURE_COLS = [
    # Baseline features (from exp_083)
    'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y',
    'zone_x', 'result_encoded', 'prev_dx', 'prev_dy',
    # Player features (NEW)
    'player_avg_dx', 'player_avg_dy', 'player_avg_dist',
    'player_success_rate', 'player_pass_count', 'player_preferred_angle',
]

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def run_experiment(X, y_delta, y_abs, start_xy, groups, n_splits, lr, n_seeds):
    """Run experiment with given settings"""
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

def main():
    print("=" * 70)
    print("exp_087: Player Features (LOO Encoding)")
    print("=" * 70)

    print("\n[1] Loading training data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Raw data: {len(train_raw)} rows")

    print("\n[2] Adding player features (LOO method)...")
    train_with_player = add_player_features_train(train_raw)
    train_with_player = create_base_features(train_with_player)

    print("\n[3] Processing episodes...")
    processed_df = process_data(train_with_player)
    print(f"  Processed: {len(processed_df)} episodes")
    del train_raw; gc.collect()

    X = processed_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = processed_df[['dx', 'dy']].values.astype(np.float32)
    y_abs = processed_df[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = processed_df[['start_x', 'start_y']].values.astype(np.float32)
    groups = processed_df['game_id'].values

    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Feature list: {FEATURE_COLS}")

    results = {}
    baseline_cv = 13.5435  # exp_083 best

    print("\n[4] Testing Player Features (fold=11, lr=0.05, 7seeds)")
    cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, 7)
    results['player_fold11_7seeds'] = cv
    diff = cv - baseline_cv
    print(f"  CV: {cv:.4f} (vs baseline {baseline_cv:.4f}: {diff:+.4f})")

    print("\n[5] Testing Player Features (fold=11, lr=0.05, 3seeds)")
    cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, 3)
    results['player_fold11_3seeds'] = cv
    print(f"  CV: {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Baseline (exp_083): CV {baseline_cv:.4f}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for name, cv in sorted_results:
        diff = cv - baseline_cv
        marker = " *** IMPROVED! ***" if diff < 0 else ""
        print(f"  {name:30s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    best_name, best_cv = sorted_results[0]
    print("\n" + "=" * 70)

    if best_cv < baseline_cv:
        print(f"  IMPROVEMENT: {best_name}")
        print(f"  CV: {best_cv:.4f} (better by {baseline_cv - best_cv:.4f})")
        print("  Creating submission...")
        create_submission(best_name, best_cv, train_with_player)
    else:
        print(f"  No improvement over baseline")
    print("=" * 70)

def create_submission(name, cv, train_with_player_df):
    """Create submission with player features"""
    n_splits = 11
    n_seeds = 7 if '7seeds' in name else 3
    seeds = SEED_POOL[:n_seeds]
    lr = 0.05

    processed_train = process_data(train_with_player_df)
    X = processed_train[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = processed_train[['dx', 'dy']].values.astype(np.float32)
    groups = processed_train['game_id'].values

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

    # Process test data with player features
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

    # Add player features using train stats
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    test_with_player = add_player_features_test(test_all, train_raw)
    test_with_player = create_base_features(test_with_player)
    test_processed = process_data(test_with_player)

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

    output_path = SUBMISSION_DIR / f"submission_player_cv{cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

if __name__ == "__main__":
    main()
