"""
exp_089: Zone 8x8 + Player Features
Zone 세분화 (6x6 -> 8x8) + Player Features (exp_087 best)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models")))
from feature_player_stats import add_player_features_train, add_player_features_test
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import gc
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"

def create_base_features(df, zone_size=8):
    """Create base features for each row with configurable zone size"""
    df['zone_x'] = (df['start_x'] / (105/zone_size)).astype(int).clip(0, zone_size-1)
    df['zone_y'] = (df['start_y'] / (68/zone_size)).astype(int).clip(0, zone_size-1)
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
    # Baseline features (with Zone 8x8)
    'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y',
    'zone_x', 'result_encoded', 'prev_dx', 'prev_dy',
    # Player features
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
    print("exp_089: Zone 8x8 + Player Features")
    print("=" * 70)

    print("\n[1] Loading training data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Raw data: {len(train_raw)} rows")

    print("\n[2] Adding player features (LOO method)...")
    train_with_player = add_player_features_train(train_raw)
    train_with_player = create_base_features(train_with_player, zone_size=8)  # Zone 8x8!

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
    print(f"  Zone size: 8x8")

    results = {}
    baseline_cv = 13.4964  # exp_087 (Zone 6x6 + Player) best

    print("\n[4] Testing Zone 8x8 + Player (fold=11, lr=0.05, 7seeds)")
    cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, 7)
    results['zone8x8_player_7seeds'] = cv
    diff = cv - baseline_cv
    print(f"  CV: {cv:.4f} (vs exp_087 {baseline_cv:.4f}: {diff:+.4f})")

    print("\n[5] Testing Zone 8x8 + Player (fold=11, lr=0.05, 3seeds)")
    cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, 3)
    results['zone8x8_player_3seeds'] = cv
    print(f"  CV: {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Baseline exp_087 (Zone 6x6 + Player): CV {baseline_cv:.4f}")
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
    else:
        print(f"  No improvement over exp_087 baseline")
    print("=" * 70)

if __name__ == "__main__":
    main()
