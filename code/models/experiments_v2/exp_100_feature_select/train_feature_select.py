"""
exp_100: Feature Selection
Test if fewer features generalize better
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

# Different feature sets to test
FEATURE_SETS = {
    'full_15': ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
                'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
                'ema_start_y', 'ema_success_rate', 'ema_possession',
                'zone_x', 'result_encoded', 'diff_x', 'velocity'],
    'top_10': ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
               'prev_dx', 'prev_dy', 'ema_start_x', 'ema_start_y', 'zone_x', 'result_encoded'],
    'top_8': ['goal_angle', 'zone_y', 'goal_distance', 'prev_dx', 'prev_dy',
              'ema_start_x', 'ema_start_y', 'zone_x'],
    'minimal_5': ['zone_x', 'zone_y', 'prev_dx', 'prev_dy', 'goal_distance'],
    'no_ema': ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
               'dist_to_center_y', 'prev_dx', 'prev_dy', 'zone_x', 'result_encoded', 'diff_x', 'velocity'],
    'no_prev': ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
                'dist_to_center_y', 'ema_start_x', 'ema_start_y', 'ema_success_rate',
                'ema_possession', 'zone_x', 'result_encoded', 'diff_x', 'velocity'],
}

SEED_POOL = [42, 123, 456]

def run_feature_set(df, feature_cols, n_seeds=3):
    """Run experiment with given feature set"""
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = df['dx'].values.astype(np.float32)
    y_dy = df['dy'].values.astype(np.float32)
    y_abs = df[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = df[['start_x', 'start_y']].values.astype(np.float32)
    groups = df['game_id'].values

    params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
              'l2_leaf_reg': 3.0, 'verbose': 0, 'early_stopping_rounds': 50,
              'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=11)
    all_oof = []

    for seed in SEED_POOL[:n_seeds]:
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

        pred_abs = np.column_stack([start_xy[:, 0] + oof_dx, start_xy[:, 1] + oof_dy])
        all_oof.append(pred_abs)
        gc.collect()

    ensemble = np.mean(all_oof, axis=0)
    cv = np.sqrt((ensemble[:, 0] - y_abs[:, 0])**2 + (ensemble[:, 1] - y_abs[:, 1])**2).mean()
    return cv

def main():
    print("=" * 70)
    print("exp_100: Feature Selection Test")
    print("=" * 70)

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    del train_df; gc.collect()

    print(f"Samples: {len(last_passes)}")

    results = {}
    for name, features in FEATURE_SETS.items():
        print(f"\nTesting {name} ({len(features)} features)...")
        cv = run_feature_set(last_passes, features)
        results[name] = (cv, len(features))
        print(f"  {name}: CV {cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS (sorted by CV)")
    print("=" * 70)
    baseline_cv = results['full_15'][0]
    for name, (cv, n_feat) in sorted(results.items(), key=lambda x: x[1][0]):
        diff = cv - baseline_cv
        print(f"  {name:15s} ({n_feat:2d} feat): CV {cv:.4f} ({diff:+.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
