"""
exp_098: Full Ensemble
Combine:
1. All baseline features (including EMA)
2. Player features (LOO encoding)
3. Multi-model ensemble (CatBoost + XGBoost + LightGBM)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models")))
from feature_player_stats import add_player_features_train
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"

def create_all_features(df):
    """Create all features including EMA and player features"""
    # Base zone features
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    # Delta features
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']

    # Previous pass features
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    # Result encoding
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    # EMA features
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

    # Additional features
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)

    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt((df['start_x'] - df['prev_start_x'])**2 + (df['start_y'] - df['prev_start_y'])**2)

    return df

# All features (21 total: 15 baseline + 6 player)
FEATURE_COLS = [
    # Baseline features (15)
    'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y',
    'prev_dx', 'prev_dy', 'ema_start_x', 'ema_start_y', 'ema_success_rate',
    'ema_possession', 'zone_x', 'result_encoded', 'diff_x', 'velocity',
    # Player features (6)
    'player_avg_dx', 'player_avg_dy', 'player_avg_dist',
    'player_success_rate', 'player_pass_count', 'player_preferred_angle',
]

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def main():
    print("=" * 70)
    print("exp_098: Full Ensemble (EMA + Player + Multi-model)")
    print("=" * 70)

    # Load and process data
    print("\n[1] Loading and processing data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Raw data: {len(train_raw)} rows")

    # Add player features
    print("  Adding player features...")
    train_with_player = add_player_features_train(train_raw)

    # Add all other features
    print("  Adding EMA and base features...")
    train_with_player = create_all_features(train_with_player)

    # Get last pass of each episode
    last_passes = train_with_player.groupby('game_episode').last().reset_index()
    print(f"  Episodes: {len(last_passes)}")
    del train_raw; gc.collect()

    X = last_passes[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = last_passes['dx'].values.astype(np.float32)
    y_dy = last_passes['dy'].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    print(f"  Features: {len(FEATURE_COLS)}")

    # Multi-model OOF predictions
    print("\n[2] Training multi-model ensemble (11-fold, 7-seeds)...")
    n_splits = 11
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y_dx, groups))

    # CatBoost
    print("  Training CatBoost...")
    cat_dx_all = []
    cat_dy_all = []
    for seed in SEED_POOL[:7]:
        params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                  'l2_leaf_reg': 3.0, 'random_state': seed, 'verbose': 0,
                  'early_stopping_rounds': 50, 'loss_function': 'MAE'}
        oof_dx = np.zeros(len(X))
        oof_dy = np.zeros(len(X))
        for train_idx, val_idx in folds:
            model_dx = CatBoostRegressor(**params)
            model_dy = CatBoostRegressor(**params)
            model_dx.fit(X[train_idx], y_dx[train_idx],
                        eval_set=(X[val_idx], y_dx[val_idx]), use_best_model=True)
            model_dy.fit(X[train_idx], y_dy[train_idx],
                        eval_set=(X[val_idx], y_dy[val_idx]), use_best_model=True)
            oof_dx[val_idx] = model_dx.predict(X[val_idx])
            oof_dy[val_idx] = model_dy.predict(X[val_idx])
            del model_dx, model_dy
        cat_dx_all.append(oof_dx)
        cat_dy_all.append(oof_dy)
        gc.collect()

    cat_dx = np.mean(cat_dx_all, axis=0)
    cat_dy = np.mean(cat_dy_all, axis=0)
    pred_x = start_xy[:, 0] + cat_dx
    pred_y = start_xy[:, 1] + cat_dy
    cat_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"    CatBoost CV: {cat_cv:.4f}")

    # XGBoost (3 seeds for speed)
    print("  Training XGBoost...")
    xgb_dx_all = []
    xgb_dy_all = []
    for seed in SEED_POOL[:3]:
        params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                  'reg_lambda': 3.0, 'random_state': seed, 'verbosity': 0,
                  'objective': 'reg:absoluteerror', 'early_stopping_rounds': 50}
        oof_dx = np.zeros(len(X))
        oof_dy = np.zeros(len(X))
        for train_idx, val_idx in folds:
            model_dx = xgb.XGBRegressor(**params)
            model_dy = xgb.XGBRegressor(**params)
            model_dx.fit(X[train_idx], y_dx[train_idx], eval_set=[(X[val_idx], y_dx[val_idx])], verbose=False)
            model_dy.fit(X[train_idx], y_dy[train_idx], eval_set=[(X[val_idx], y_dy[val_idx])], verbose=False)
            oof_dx[val_idx] = model_dx.predict(X[val_idx])
            oof_dy[val_idx] = model_dy.predict(X[val_idx])
            del model_dx, model_dy
        xgb_dx_all.append(oof_dx)
        xgb_dy_all.append(oof_dy)
        gc.collect()

    xgb_dx = np.mean(xgb_dx_all, axis=0)
    xgb_dy = np.mean(xgb_dy_all, axis=0)
    pred_x = start_xy[:, 0] + xgb_dx
    pred_y = start_xy[:, 1] + xgb_dy
    xgb_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"    XGBoost CV: {xgb_cv:.4f}")

    # LightGBM (3 seeds for speed)
    print("  Training LightGBM...")
    lgb_dx_all = []
    lgb_dy_all = []
    for seed in SEED_POOL[:3]:
        params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                  'reg_lambda': 3.0, 'random_state': seed, 'verbosity': -1,
                  'objective': 'mae'}
        oof_dx = np.zeros(len(X))
        oof_dy = np.zeros(len(X))
        for train_idx, val_idx in folds:
            model_dx = lgb.LGBMRegressor(**params)
            model_dy = lgb.LGBMRegressor(**params)
            model_dx.fit(X[train_idx], y_dx[train_idx], eval_set=[(X[val_idx], y_dx[val_idx])])
            model_dy.fit(X[train_idx], y_dy[train_idx], eval_set=[(X[val_idx], y_dy[val_idx])])
            oof_dx[val_idx] = model_dx.predict(X[val_idx])
            oof_dy[val_idx] = model_dy.predict(X[val_idx])
            del model_dx, model_dy
        lgb_dx_all.append(oof_dx)
        lgb_dy_all.append(oof_dy)
        gc.collect()

    lgb_dx = np.mean(lgb_dx_all, axis=0)
    lgb_dy = np.mean(lgb_dy_all, axis=0)
    pred_x = start_xy[:, 0] + lgb_dx
    pred_y = start_xy[:, 1] + lgb_dy
    lgb_cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
    print(f"    LightGBM CV: {lgb_cv:.4f}")

    # Ensemble with optimal weights
    print("\n[3] Finding optimal ensemble weights...")
    best_cv = float('inf')
    best_weights = None

    for w_cat in np.arange(0.4, 0.8, 0.05):
        for w_xgb in np.arange(0.05, 0.35, 0.05):
            w_lgb = 1.0 - w_cat - w_xgb
            if w_lgb < 0.05 or w_lgb > 0.35:
                continue
            ens_dx = w_cat * cat_dx + w_xgb * xgb_dx + w_lgb * lgb_dx
            ens_dy = w_cat * cat_dy + w_xgb * xgb_dy + w_lgb * lgb_dy
            pred_x = start_xy[:, 0] + ens_dx
            pred_y = start_xy[:, 1] + ens_dy
            cv = np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()
            if cv < best_cv:
                best_cv = cv
                best_weights = (w_cat, w_xgb, w_lgb)

    print(f"  Best weights: Cat={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, LGB={best_weights[2]:.2f}")
    print(f"  Best ensemble CV: {best_cv:.4f}")

    # Compare with previous best
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Previous Best CV (exp_087 player): 13.4964")
    print(f"  Previous Best LB (exp_092 multi): 13.4924")
    print("-" * 70)
    print(f"  CatBoost only (21 features):  CV {cat_cv:.4f}")
    print(f"  XGBoost only (21 features):   CV {xgb_cv:.4f}")
    print(f"  LightGBM only (21 features):  CV {lgb_cv:.4f}")
    print(f"  Best Ensemble (21 features):  CV {best_cv:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
