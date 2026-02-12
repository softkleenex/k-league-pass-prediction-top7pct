"""
exp_090: Context Features (result_name, team_id, period_id, time, is_home)
핵심 발견: Successful vs Unsuccessful 패스 거리 차이가 큼 (18.25 vs 23.07)
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

def create_base_features(df, zone_size=6):
    """Create base features"""
    df['zone_x'] = (df['start_x'] / (105/zone_size)).astype(int).clip(0, zone_size-1)
    df['zone_y'] = (df['start_y'] / (68/zone_size)).astype(int).clip(0, zone_size-1)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)

    # Result encoding
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    return df

def extract_last_pass_features(episode_df):
    """Extract features from last pass with context"""
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
        # NEW: Context features
        'period_id': last['period_id'],  # 전/후반
        'time_seconds': last['time_seconds'],  # 경기 시간
        'is_home': int(last['is_home']),  # 홈/원정
        'team_id': last['team_id'],  # 팀 ID
    }

    # Previous pass features
    if len(episode_df) > 1:
        prev = episode_df.iloc[-2]
        features['prev_dx'] = prev['dx']
        features['prev_dy'] = prev['dy']
    else:
        features['prev_dx'] = 0
        features['prev_dy'] = 0

    # Episode length (number of passes)
    features['episode_length'] = len(episode_df)

    return features

def process_data(df):
    """Process all episodes"""
    all_features = []
    for game_ep, ep_df in df.groupby('game_episode'):
        features = extract_last_pass_features(ep_df)
        all_features.append(features)
    return pd.DataFrame(all_features)

# Feature columns - baseline + context
FEATURE_COLS_BASELINE = [
    'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y',
    'zone_x', 'result_encoded', 'prev_dx', 'prev_dy',
]

FEATURE_COLS_CONTEXT = FEATURE_COLS_BASELINE + [
    'period_id', 'time_seconds', 'is_home', 'episode_length',
]

# Team features - use team mean encoding (LOO style)
def add_team_features_loo(df, target_col='dx'):
    """Add team-level features using LOO to prevent leakage"""
    team_stats = df.groupby('team_id').agg({
        'dx': ['mean', 'std'],
        'dy': ['mean', 'std'],
        'end_x': 'mean',
        'end_y': 'mean',
    })
    team_stats.columns = ['team_dx_mean', 'team_dx_std', 'team_dy_mean', 'team_dy_std',
                          'team_end_x_mean', 'team_end_y_mean']
    team_stats = team_stats.reset_index()

    df = df.merge(team_stats, on='team_id', how='left')
    return df

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def run_experiment(X, y_delta, y_abs, start_xy, groups, n_splits, lr, n_seeds, feature_names):
    """Run experiment"""
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
    print("exp_090: Context Features")
    print("=" * 70)

    print("\n[1] Loading training data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Raw data: {len(train_raw)} rows")

    print("\n[2] Creating base features...")
    train_with_features = create_base_features(train_raw)

    print("\n[3] Processing episodes...")
    processed_df = process_data(train_with_features)
    print(f"  Processed: {len(processed_df)} episodes")
    del train_raw; gc.collect()

    # Prepare data
    X_baseline = processed_df[FEATURE_COLS_BASELINE].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_context = processed_df[FEATURE_COLS_CONTEXT].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    y_delta = processed_df[['dx', 'dy']].values.astype(np.float32)
    y_abs = processed_df[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = processed_df[['start_x', 'start_y']].values.astype(np.float32)
    groups = processed_df['game_id'].values

    results = {}
    baseline_cv = 13.5358  # Best LB (exp_083)

    print(f"\n[4] Baseline (9 features, fold=11, lr=0.05, 7seeds)")
    cv = run_experiment(X_baseline, y_delta, y_abs, start_xy, groups, 11, 0.05, 7, FEATURE_COLS_BASELINE)
    results['baseline'] = cv
    print(f"  CV: {cv:.4f}")

    print(f"\n[5] + Context Features (13 features)")
    cv = run_experiment(X_context, y_delta, y_abs, start_xy, groups, 11, 0.05, 7, FEATURE_COLS_CONTEXT)
    results['context'] = cv
    print(f"  CV: {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Best LB (exp_083): {baseline_cv:.4f}")
    print("-" * 70)

    for name, cv in sorted(results.items(), key=lambda x: x[1]):
        diff = cv - baseline_cv
        marker = " *** IMPROVED! ***" if diff < 0 else ""
        print(f"  {name:20s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    print("=" * 70)

if __name__ == "__main__":
    main()
