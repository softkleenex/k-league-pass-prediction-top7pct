"""
exp_095: Stacking Ensemble
Level 1: CatBoost, XGBoost, LightGBM, Ridge, ElasticNet
Level 2: Ridge on OOF predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, ElasticNet
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
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

TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
          'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
          'ema_start_y', 'ema_success_rate', 'ema_possession',
          'zone_x', 'result_encoded', 'diff_x', 'velocity']

def main():
    print("=" * 70)
    print("exp_095: Stacking Ensemble")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
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

    print(f"  Samples: {len(X)}")

    # Level 1 OOF predictions
    print("\n[2] Level 1: OOF predictions...")
    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X, y_dx, groups))

    # Storage for OOF predictions
    oof_cat_dx = np.zeros(len(X))
    oof_cat_dy = np.zeros(len(X))
    oof_xgb_dx = np.zeros(len(X))
    oof_xgb_dy = np.zeros(len(X))
    oof_lgb_dx = np.zeros(len(X))
    oof_lgb_dy = np.zeros(len(X))
    oof_ridge_dx = np.zeros(len(X))
    oof_ridge_dy = np.zeros(len(X))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/11...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_dx_train, y_dx_val = y_dx[train_idx], y_dx[val_idx]
        y_dy_train, y_dy_val = y_dy[train_idx], y_dy[val_idx]

        # CatBoost
        cat_params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                      'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
                      'early_stopping_rounds': 50, 'loss_function': 'MAE'}
        cat_dx = CatBoostRegressor(**cat_params)
        cat_dy = CatBoostRegressor(**cat_params)
        cat_dx.fit(X_train, y_dx_train, eval_set=(X_val, y_dx_val), use_best_model=True)
        cat_dy.fit(X_train, y_dy_train, eval_set=(X_val, y_dy_val), use_best_model=True)
        oof_cat_dx[val_idx] = cat_dx.predict(X_val)
        oof_cat_dy[val_idx] = cat_dy.predict(X_val)

        # XGBoost
        xgb_params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                      'reg_lambda': 3.0, 'random_state': 42, 'verbosity': 0,
                      'objective': 'reg:absoluteerror', 'early_stopping_rounds': 50}
        xgb_dx = xgb.XGBRegressor(**xgb_params)
        xgb_dy = xgb.XGBRegressor(**xgb_params)
        xgb_dx.fit(X_train, y_dx_train, eval_set=[(X_val, y_dx_val)], verbose=False)
        xgb_dy.fit(X_train, y_dy_train, eval_set=[(X_val, y_dy_val)], verbose=False)
        oof_xgb_dx[val_idx] = xgb_dx.predict(X_val)
        oof_xgb_dy[val_idx] = xgb_dy.predict(X_val)

        # LightGBM
        lgb_params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                      'reg_lambda': 3.0, 'random_state': 42, 'verbosity': -1,
                      'objective': 'mae'}
        lgb_dx = lgb.LGBMRegressor(**lgb_params)
        lgb_dy = lgb.LGBMRegressor(**lgb_params)
        lgb_dx.fit(X_train, y_dx_train, eval_set=[(X_val, y_dx_val)])
        lgb_dy.fit(X_train, y_dy_train, eval_set=[(X_val, y_dy_val)])
        oof_lgb_dx[val_idx] = lgb_dx.predict(X_val)
        oof_lgb_dy[val_idx] = lgb_dy.predict(X_val)

        # Ridge
        ridge_dx = Ridge(alpha=1.0)
        ridge_dy = Ridge(alpha=1.0)
        ridge_dx.fit(X_train, y_dx_train)
        ridge_dy.fit(X_train, y_dy_train)
        oof_ridge_dx[val_idx] = ridge_dx.predict(X_val)
        oof_ridge_dy[val_idx] = ridge_dy.predict(X_val)

        gc.collect()

    # Individual model CV scores
    print("\n[3] Individual model CVs...")
    for name, oof_dx_pred, oof_dy_pred in [
        ('CatBoost', oof_cat_dx, oof_cat_dy),
        ('XGBoost', oof_xgb_dx, oof_xgb_dy),
        ('LightGBM', oof_lgb_dx, oof_lgb_dy),
        ('Ridge', oof_ridge_dx, oof_ridge_dy),
    ]:
        pred_x = start_xy[:, 0] + oof_dx_pred
        pred_y = start_xy[:, 1] + oof_dy_pred
        cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
        print(f"  {name}: CV {cv:.4f}")

    # Level 2: Stack with Ridge
    print("\n[4] Level 2: Stacking with Ridge...")
    # Create stacking features
    stack_X_dx = np.column_stack([oof_cat_dx, oof_xgb_dx, oof_lgb_dx, oof_ridge_dx])
    stack_X_dy = np.column_stack([oof_cat_dy, oof_xgb_dy, oof_lgb_dy, oof_ridge_dy])

    # Train meta-model with CV
    oof_stack_dx = np.zeros(len(X))
    oof_stack_dy = np.zeros(len(X))

    for train_idx, val_idx in folds:
        meta_dx = Ridge(alpha=1.0)
        meta_dy = Ridge(alpha=1.0)
        meta_dx.fit(stack_X_dx[train_idx], y_dx[train_idx])
        meta_dy.fit(stack_X_dy[train_idx], y_dy[train_idx])
        oof_stack_dx[val_idx] = meta_dx.predict(stack_X_dx[val_idx])
        oof_stack_dy[val_idx] = meta_dy.predict(stack_X_dy[val_idx])

    pred_x = start_xy[:, 0] + oof_stack_dx
    pred_y = start_xy[:, 1] + oof_stack_dy
    stack_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"  Stacking CV: {stack_cv:.4f}")

    # Simple average for comparison
    print("\n[5] Simple average for comparison...")
    avg_dx = (oof_cat_dx + oof_xgb_dx + oof_lgb_dx) / 3  # Exclude Ridge (it's worse)
    avg_dy = (oof_cat_dy + oof_xgb_dy + oof_lgb_dy) / 3
    pred_x = start_xy[:, 0] + avg_dx
    pred_y = start_xy[:, 1] + avg_dy
    avg_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"  Average (Cat+XGB+LGB) CV: {avg_cv:.4f}")

    # Weighted average (based on individual performance)
    print("\n[6] Optimized weighted average...")
    best_cv = float('inf')
    best_weights = None
    for w_cat in np.arange(0.3, 0.8, 0.1):
        for w_xgb in np.arange(0.1, 0.4, 0.05):
            w_lgb = 1.0 - w_cat - w_xgb
            if w_lgb < 0.05 or w_lgb > 0.4:
                continue
            weighted_dx = w_cat * oof_cat_dx + w_xgb * oof_xgb_dx + w_lgb * oof_lgb_dx
            weighted_dy = w_cat * oof_cat_dy + w_xgb * oof_xgb_dy + w_lgb * oof_lgb_dy
            pred_x = start_xy[:, 0] + weighted_dx
            pred_y = start_xy[:, 1] + weighted_dy
            cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
            if cv < best_cv:
                best_cv = cv
                best_weights = (w_cat, w_xgb, w_lgb)

    print(f"  Best weights: Cat={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, LGB={best_weights[2]:.2f}")
    print(f"  Best weighted CV: {best_cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  CatBoost only:              CV 13.60xx")
    print(f"  Stacking (Ridge meta):      CV {stack_cv:.4f}")
    print(f"  Simple average (3 models):  CV {avg_cv:.4f}")
    print(f"  Best weighted average:      CV {best_cv:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
