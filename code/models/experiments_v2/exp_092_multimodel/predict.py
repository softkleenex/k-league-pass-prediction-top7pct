"""
exp_092: Multi-Model Ensemble Prediction
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
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

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def main():
    print("=" * 70)
    print("Multi-Model Ensemble Prediction")
    print("=" * 70)

    # Load and prepare train data
    print("\n[1] Loading train data...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    del train_df; gc.collect()

    X_train = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = last_passes['dx'].values
    y_dy = last_passes['dy'].values

    # Load test data
    print("[2] Loading test data...")
    test_meta = pd.read_csv(DATA_DIR / 'test.csv')
    test_episodes = []
    for _, row in test_meta.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
        ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']
        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)
    test_all = pd.concat(test_episodes, ignore_index=True)
    test_all = create_features(test_all)
    test_last = test_all.groupby('game_episode').last().reset_index()

    X_test = test_last[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    start_xy = test_last[['start_x', 'start_y']].values

    # Train models and predict
    print("[3] Training CatBoost models...")
    cat_dx_preds = []
    cat_dy_preds = []
    for seed in SEED_POOL[:7]:
        params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                  'l2_leaf_reg': 3.0, 'random_state': seed, 'verbose': 0,
                  'loss_function': 'MAE'}
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)
        model_dx.fit(X_train, y_dx)
        model_dy.fit(X_train, y_dy)
        cat_dx_preds.append(model_dx.predict(X_test))
        cat_dy_preds.append(model_dy.predict(X_test))
        del model_dx, model_dy

    cat_dx = np.mean(cat_dx_preds, axis=0)
    cat_dy = np.mean(cat_dy_preds, axis=0)
    gc.collect()

    print("[4] Training XGBoost models...")
    xgb_dx_preds = []
    xgb_dy_preds = []
    for seed in SEED_POOL[:7]:
        params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                  'reg_lambda': 3.0, 'random_state': seed, 'verbosity': 0,
                  'objective': 'reg:absoluteerror'}
        model_dx = xgb.XGBRegressor(**params)
        model_dy = xgb.XGBRegressor(**params)
        model_dx.fit(X_train, y_dx)
        model_dy.fit(X_train, y_dy)
        xgb_dx_preds.append(model_dx.predict(X_test))
        xgb_dy_preds.append(model_dy.predict(X_test))
        del model_dx, model_dy

    xgb_dx = np.mean(xgb_dx_preds, axis=0)
    xgb_dy = np.mean(xgb_dy_preds, axis=0)
    gc.collect()

    print("[5] Training LightGBM models...")
    lgb_dx_preds = []
    lgb_dy_preds = []
    for seed in SEED_POOL[:7]:
        params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                  'reg_lambda': 3.0, 'random_state': seed, 'verbosity': -1,
                  'objective': 'mae'}
        model_dx = lgb.LGBMRegressor(**params)
        model_dy = lgb.LGBMRegressor(**params)
        model_dx.fit(X_train, y_dx)
        model_dy.fit(X_train, y_dy)
        lgb_dx_preds.append(model_dx.predict(X_test))
        lgb_dy_preds.append(model_dy.predict(X_test))
        del model_dx, model_dy

    lgb_dx = np.mean(lgb_dx_preds, axis=0)
    lgb_dy = np.mean(lgb_dy_preds, axis=0)
    gc.collect()

    # Ensemble: 0.6 Cat + 0.2 XGB + 0.2 LGB
    print("[6] Creating ensemble prediction...")
    ens_dx = 0.6 * cat_dx + 0.2 * xgb_dx + 0.2 * lgb_dx
    ens_dy = 0.6 * cat_dy + 0.2 * xgb_dy + 0.2 * lgb_dy

    pred_x = start_xy[:, 0] + ens_dx
    pred_y = np.clip(start_xy[:, 1] + ens_dy, 0, 68)

    # Create submission
    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })
    submission = submission.set_index('game_episode').loc[test_meta['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / "submission_multimodel_cv13.52.csv"
    submission.to_csv(output_path, index=False)
    print(f"\n[7] Saved: {output_path}")
    print(f"  Shape: {submission.shape}")
    print(submission.head())

if __name__ == "__main__":
    main()
