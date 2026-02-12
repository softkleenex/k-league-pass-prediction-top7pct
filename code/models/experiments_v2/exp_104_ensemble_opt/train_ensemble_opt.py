"""
exp_104: Ensemble Weight Optimization
Find optimal weights for CatBoost + XGBoost + LightGBM
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
    print("exp_104: Ensemble Weight Optimization")
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

    print(f"Samples: {len(X)}")

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X, y_dx, groups))

    # Get OOF predictions from each model with L2=30
    print("\n[1] CatBoost (L2=30, 7 seeds)...")
    cat_all_dx, cat_all_dy = [], []
    for seed in SEED_POOL:
        params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                  'l2_leaf_reg': 30.0, 'random_state': seed, 'verbose': 0,
                  'early_stopping_rounds': 50, 'loss_function': 'MAE'}
        oof_dx, oof_dy = np.zeros(len(X)), np.zeros(len(X))
        for train_idx, val_idx in folds:
            m_dx = CatBoostRegressor(**params)
            m_dy = CatBoostRegressor(**params)
            m_dx.fit(X[train_idx], y_dx[train_idx], eval_set=(X[val_idx], y_dx[val_idx]), use_best_model=True)
            m_dy.fit(X[train_idx], y_dy[train_idx], eval_set=(X[val_idx], y_dy[val_idx]), use_best_model=True)
            oof_dx[val_idx] = m_dx.predict(X[val_idx])
            oof_dy[val_idx] = m_dy.predict(X[val_idx])
            del m_dx, m_dy
        cat_all_dx.append(oof_dx)
        cat_all_dy.append(oof_dy)
        gc.collect()
    cat_dx, cat_dy = np.mean(cat_all_dx, axis=0), np.mean(cat_all_dy, axis=0)
    pred = np.column_stack([start_xy[:, 0] + cat_dx, start_xy[:, 1] + cat_dy])
    cat_cv = np.sqrt((pred[:, 0] - y_abs[:, 0])**2 + (pred[:, 1] - y_abs[:, 1])**2).mean()
    print(f"  CatBoost CV: {cat_cv:.4f}")

    print("\n[2] XGBoost (lambda=30, 5 seeds)...")
    xgb_all_dx, xgb_all_dy = [], []
    for seed in SEED_POOL[:5]:
        params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                  'reg_lambda': 30.0, 'random_state': seed, 'verbosity': 0,
                  'objective': 'reg:absoluteerror', 'early_stopping_rounds': 50}
        oof_dx, oof_dy = np.zeros(len(X)), np.zeros(len(X))
        for train_idx, val_idx in folds:
            m_dx = xgb.XGBRegressor(**params)
            m_dy = xgb.XGBRegressor(**params)
            m_dx.fit(X[train_idx], y_dx[train_idx], eval_set=[(X[val_idx], y_dx[val_idx])], verbose=False)
            m_dy.fit(X[train_idx], y_dy[train_idx], eval_set=[(X[val_idx], y_dy[val_idx])], verbose=False)
            oof_dx[val_idx] = m_dx.predict(X[val_idx])
            oof_dy[val_idx] = m_dy.predict(X[val_idx])
            del m_dx, m_dy
        xgb_all_dx.append(oof_dx)
        xgb_all_dy.append(oof_dy)
        gc.collect()
    xgb_dx, xgb_dy = np.mean(xgb_all_dx, axis=0), np.mean(xgb_all_dy, axis=0)
    pred = np.column_stack([start_xy[:, 0] + xgb_dx, start_xy[:, 1] + xgb_dy])
    xgb_cv = np.sqrt((pred[:, 0] - y_abs[:, 0])**2 + (pred[:, 1] - y_abs[:, 1])**2).mean()
    print(f"  XGBoost CV: {xgb_cv:.4f}")

    print("\n[3] LightGBM (lambda=30, 5 seeds)...")
    lgb_all_dx, lgb_all_dy = [], []
    for seed in SEED_POOL[:5]:
        params = {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.05,
                  'reg_lambda': 30.0, 'random_state': seed, 'verbosity': -1, 'objective': 'mae'}
        oof_dx, oof_dy = np.zeros(len(X)), np.zeros(len(X))
        for train_idx, val_idx in folds:
            m_dx = lgb.LGBMRegressor(**params)
            m_dy = lgb.LGBMRegressor(**params)
            m_dx.fit(X[train_idx], y_dx[train_idx], eval_set=[(X[val_idx], y_dx[val_idx])])
            m_dy.fit(X[train_idx], y_dy[train_idx], eval_set=[(X[val_idx], y_dy[val_idx])])
            oof_dx[val_idx] = m_dx.predict(X[val_idx])
            oof_dy[val_idx] = m_dy.predict(X[val_idx])
            del m_dx, m_dy
        lgb_all_dx.append(oof_dx)
        lgb_all_dy.append(oof_dy)
        gc.collect()
    lgb_dx, lgb_dy = np.mean(lgb_all_dx, axis=0), np.mean(lgb_all_dy, axis=0)
    pred = np.column_stack([start_xy[:, 0] + lgb_dx, start_xy[:, 1] + lgb_dy])
    lgb_cv = np.sqrt((pred[:, 0] - y_abs[:, 0])**2 + (pred[:, 1] - y_abs[:, 1])**2).mean()
    print(f"  LightGBM CV: {lgb_cv:.4f}")

    # Grid search for optimal weights
    print("\n[4] Grid search for optimal weights...")
    best_cv = float('inf')
    best_weights = (1.0, 0.0, 0.0)
    results = []

    for w_cat in np.arange(0.4, 1.01, 0.05):
        for w_xgb in np.arange(0.0, 0.61 - w_cat + 0.4, 0.05):
            w_lgb = 1.0 - w_cat - w_xgb
            if w_lgb < -0.01 or w_lgb > 0.4:
                continue
            ens_dx = w_cat * cat_dx + w_xgb * xgb_dx + w_lgb * lgb_dx
            ens_dy = w_cat * cat_dy + w_xgb * xgb_dy + w_lgb * lgb_dy
            pred = np.column_stack([start_xy[:, 0] + ens_dx, start_xy[:, 1] + ens_dy])
            cv = np.sqrt((pred[:, 0] - y_abs[:, 0])**2 + (pred[:, 1] - y_abs[:, 1])**2).mean()
            results.append((w_cat, w_xgb, w_lgb, cv))
            if cv < best_cv:
                best_cv = cv
                best_weights = (w_cat, w_xgb, w_lgb)

    print(f"  Best: Cat={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, LGB={best_weights[2]:.2f}")
    print(f"  Best CV: {best_cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  CatBoost only:  CV {cat_cv:.4f}")
    print(f"  XGBoost only:   CV {xgb_cv:.4f}")
    print(f"  LightGBM only:  CV {lgb_cv:.4f}")
    print(f"  Best Ensemble:  CV {best_cv:.4f}")
    print(f"  Best weights:   Cat={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, LGB={best_weights[2]:.2f}")
    print("=" * 70)

    # Save results
    with open(BASE / "experiments" / "exp_104_results.txt", "w") as f:
        f.write("Ensemble Weight Optimization Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"CatBoost CV: {cat_cv:.4f}\n")
        f.write(f"XGBoost CV: {xgb_cv:.4f}\n")
        f.write(f"LightGBM CV: {lgb_cv:.4f}\n")
        f.write(f"Best Ensemble CV: {best_cv:.4f}\n")
        f.write(f"Best weights: Cat={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, LGB={best_weights[2]:.2f}\n")

if __name__ == "__main__":
    main()
