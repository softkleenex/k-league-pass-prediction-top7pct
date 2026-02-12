"""
exp_102: Optimized Model
- L2=30 (very_high regularization from exp_099)
- 11-fold, 7-seeds
- Multi-model ensemble (CatBoost heavy)
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
    print("exp_102: Optimized Model (L2=30, 7-seeds, Multi-model)")
    print("=" * 70)

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    del train_df; gc.collect()

    X = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = last_passes['dx'].values.astype(np.float32)
    y_dy = last_passes['dy'].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    print(f"Samples: {len(X)}, Features: 15")

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X, y_dx, groups))

    # CatBoost with L2=30 (7 seeds)
    print("\n[1] CatBoost (L2=30, 7 seeds)...")
    cat_params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                  'l2_leaf_reg': 30.0, 'verbose': 0, 'early_stopping_rounds': 50,
                  'loss_function': 'MAE'}

    cat_all_dx = []
    cat_all_dy = []
    for seed in SEED_POOL:
        p = cat_params.copy()
        p['random_state'] = seed
        oof_dx = np.zeros(len(X))
        oof_dy = np.zeros(len(X))
        for train_idx, val_idx in folds:
            m_dx = CatBoostRegressor(**p)
            m_dy = CatBoostRegressor(**p)
            m_dx.fit(X[train_idx], y_dx[train_idx], eval_set=(X[val_idx], y_dx[val_idx]), use_best_model=True)
            m_dy.fit(X[train_idx], y_dy[train_idx], eval_set=(X[val_idx], y_dy[val_idx]), use_best_model=True)
            oof_dx[val_idx] = m_dx.predict(X[val_idx])
            oof_dy[val_idx] = m_dy.predict(X[val_idx])
            del m_dx, m_dy
        cat_all_dx.append(oof_dx)
        cat_all_dy.append(oof_dy)
        gc.collect()

    cat_dx = np.mean(cat_all_dx, axis=0)
    cat_dy = np.mean(cat_all_dy, axis=0)
    pred = np.column_stack([start_xy[:, 0] + cat_dx, start_xy[:, 1] + cat_dy])
    cat_cv = np.sqrt((pred[:, 0] - y_abs[:, 0])**2 + (pred[:, 1] - y_abs[:, 1])**2).mean()
    print(f"  CatBoost CV: {cat_cv:.4f}")

    # XGBoost with high reg (3 seeds)
    print("\n[2] XGBoost (reg_lambda=30, 3 seeds)...")
    xgb_params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                  'reg_lambda': 30.0, 'verbosity': 0, 'objective': 'reg:absoluteerror',
                  'early_stopping_rounds': 50}

    xgb_all_dx = []
    xgb_all_dy = []
    for seed in SEED_POOL[:3]:
        p = xgb_params.copy()
        p['random_state'] = seed
        oof_dx = np.zeros(len(X))
        oof_dy = np.zeros(len(X))
        for train_idx, val_idx in folds:
            m_dx = xgb.XGBRegressor(**p)
            m_dy = xgb.XGBRegressor(**p)
            m_dx.fit(X[train_idx], y_dx[train_idx], eval_set=[(X[val_idx], y_dx[val_idx])], verbose=False)
            m_dy.fit(X[train_idx], y_dy[train_idx], eval_set=[(X[val_idx], y_dy[val_idx])], verbose=False)
            oof_dx[val_idx] = m_dx.predict(X[val_idx])
            oof_dy[val_idx] = m_dy.predict(X[val_idx])
            del m_dx, m_dy
        xgb_all_dx.append(oof_dx)
        xgb_all_dy.append(oof_dy)
        gc.collect()

    xgb_dx = np.mean(xgb_all_dx, axis=0)
    xgb_dy = np.mean(xgb_all_dy, axis=0)
    pred = np.column_stack([start_xy[:, 0] + xgb_dx, start_xy[:, 1] + xgb_dy])
    xgb_cv = np.sqrt((pred[:, 0] - y_abs[:, 0])**2 + (pred[:, 1] - y_abs[:, 1])**2).mean()
    print(f"  XGBoost CV: {xgb_cv:.4f}")

    # Ensemble (CatBoost 0.7 + XGBoost 0.3)
    print("\n[3] Ensemble optimization...")
    best_cv = float('inf')
    best_w = 0.6
    for w in np.arange(0.5, 0.9, 0.05):
        ens_dx = w * cat_dx + (1 - w) * xgb_dx
        ens_dy = w * cat_dy + (1 - w) * xgb_dy
        pred = np.column_stack([start_xy[:, 0] + ens_dx, start_xy[:, 1] + ens_dy])
        cv = np.sqrt((pred[:, 0] - y_abs[:, 0])**2 + (pred[:, 1] - y_abs[:, 1])**2).mean()
        if cv < best_cv:
            best_cv = cv
            best_w = w

    print(f"  Best weight: Cat={best_w:.2f}, XGB={1-best_w:.2f}")
    print(f"  Ensemble CV: {best_cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  CatBoost (L2=30):    CV {cat_cv:.4f}")
    print(f"  XGBoost (lambda=30): CV {xgb_cv:.4f}")
    print(f"  Best Ensemble:       CV {best_cv:.4f}")
    print(f"\n  Previous best LB: 13.4924 (Multi-model L2=3)")
    print("=" * 70)

if __name__ == "__main__":
    main()
