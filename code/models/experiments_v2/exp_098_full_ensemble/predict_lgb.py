"""
exp_098: LightGBM Prediction with 21 Features (EMA + Player)
Best CV: 12.6757
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models")))
from feature_player_stats import add_player_features_train, add_player_features_test
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import gc
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"

def create_all_features(df):
    """Create all features including EMA and player features"""
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

FEATURE_COLS = [
    'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y',
    'prev_dx', 'prev_dy', 'ema_start_x', 'ema_start_y', 'ema_success_rate',
    'ema_possession', 'zone_x', 'result_encoded', 'diff_x', 'velocity',
    'player_avg_dx', 'player_avg_dy', 'player_avg_dist',
    'player_success_rate', 'player_pass_count', 'player_preferred_angle',
]

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def main():
    print("=" * 70)
    print("exp_098: LightGBM Prediction (21 Features)")
    print("=" * 70)

    # Load train data
    print("\n[1] Loading train data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    train_with_player = add_player_features_train(train_raw)
    train_with_player = create_all_features(train_with_player)
    train_last = train_with_player.groupby('game_episode').last().reset_index()

    X_train = train_last[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values

    # Load and process test data
    print("\n[2] Loading test data...")
    test_meta = pd.read_csv(DATA_DIR / 'test.csv')
    test_episodes = []
    for _, row in test_meta.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)
    test_all = pd.concat(test_episodes, ignore_index=True)

    test_with_player = add_player_features_test(test_all, train_raw)
    test_with_player = create_all_features(test_with_player)
    test_last = test_with_player.groupby('game_episode').last().reset_index()

    X_test = test_last[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    start_xy = test_last[['start_x', 'start_y']].values

    # Train LightGBM models
    print("\n[3] Training LightGBM (7 seeds)...")
    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_train, y_dx, groups))

    all_pred_dx = []
    all_pred_dy = []

    for seed in SEED_POOL[:7]:
        params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                  'reg_lambda': 3.0, 'random_state': seed, 'verbosity': -1,
                  'objective': 'mae'}

        models_dx = []
        models_dy = []

        for train_idx, val_idx in folds:
            model_dx = lgb.LGBMRegressor(**params)
            model_dy = lgb.LGBMRegressor(**params)
            model_dx.fit(X_train[train_idx], y_dx[train_idx], eval_set=[(X_train[val_idx], y_dx[val_idx])])
            model_dy.fit(X_train[train_idx], y_dy[train_idx], eval_set=[(X_train[val_idx], y_dy[val_idx])])
            models_dx.append(model_dx)
            models_dy.append(model_dy)

        # Predict with all fold models
        pred_dx = np.mean([m.predict(X_test) for m in models_dx], axis=0)
        pred_dy = np.mean([m.predict(X_test) for m in models_dy], axis=0)
        all_pred_dx.append(pred_dx)
        all_pred_dy.append(pred_dy)

        print(f"  Seed {seed} done")
        gc.collect()

    # Average across seeds
    final_dx = np.mean(all_pred_dx, axis=0)
    final_dy = np.mean(all_pred_dy, axis=0)

    pred_x = start_xy[:, 0] + final_dx
    pred_y = np.clip(start_xy[:, 1] + final_dy, 0, 68)

    # Create submission
    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })
    submission = submission.set_index('game_episode').loc[test_meta['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / "submission_lgb_cv12.68.csv"
    submission.to_csv(output_path, index=False)
    print(f"\n[4] Saved: {output_path}")
    print(f"  Shape: {submission.shape}")
    print(submission.head())

if __name__ == "__main__":
    main()
