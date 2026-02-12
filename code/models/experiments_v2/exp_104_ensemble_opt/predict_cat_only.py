"""
exp_104 CatBoost Only: Quick submission
CatBoost L2=30, 3 seeds - fastest version
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

TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
          'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
          'ema_start_y', 'ema_success_rate', 'ema_possession',
          'zone_x', 'result_encoded', 'diff_x', 'velocity']

def main():
    print("CatBoost Only (L2=30, 3 seeds)")

    # Load train
    print("Loading train data...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    train_last = train_df.groupby('game_episode').last().reset_index()
    del train_df
    gc.collect()

    X_train = train_last[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values
    del train_last
    gc.collect()

    # Load test
    print("Loading test data...")
    test_meta = pd.read_csv(DATA_DIR / 'test.csv')
    test_episodes = []
    for _, row in test_meta.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)
    test_all = pd.concat(test_episodes, ignore_index=True)
    test_all = create_features(test_all)
    test_last = test_all.groupby('game_episode').last().reset_index()
    del test_all
    gc.collect()

    X_test = test_last[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    start_xy = test_last[['start_x', 'start_y']].values

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_train, y_dx, groups))

    # CatBoost (3 seeds)
    print("Training CatBoost...")
    all_pred_dx, all_pred_dy = [], []

    for seed in [42, 123, 456]:
        print(f"  Seed {seed}...")
        params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                  'l2_leaf_reg': 30.0, 'random_state': seed, 'verbose': 0,
                  'early_stopping_rounds': 50, 'loss_function': 'MAE'}
        fold_dx, fold_dy = [], []

        for i, (train_idx, val_idx) in enumerate(folds):
            m_dx = CatBoostRegressor(**params)
            m_dy = CatBoostRegressor(**params)
            m_dx.fit(X_train[train_idx], y_dx[train_idx],
                    eval_set=(X_train[val_idx], y_dx[val_idx]), use_best_model=True)
            m_dy.fit(X_train[train_idx], y_dy[train_idx],
                    eval_set=(X_train[val_idx], y_dy[val_idx]), use_best_model=True)
            fold_dx.append(m_dx.predict(X_test))
            fold_dy.append(m_dy.predict(X_test))
            del m_dx, m_dy
            gc.collect()

        all_pred_dx.append(np.mean(fold_dx, axis=0))
        all_pred_dy.append(np.mean(fold_dy, axis=0))
        print(f"    Seed {seed} done!")
        gc.collect()

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

    output_path = SUBMISSION_DIR / "submission_cat_l2_30_cv13.53.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print("DONE!")

if __name__ == "__main__":
    main()
