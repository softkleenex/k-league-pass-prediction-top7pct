"""
exp_092: Multi-Model Ensemble
CatBoost + XGBoost + LightGBM 블렌딩
다른 알고리즘의 다른 bias를 활용
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
    """exp_083의 15개 피처 생성"""
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

def run_catboost(X, y, groups, n_splits, n_seeds, lr=0.05):
    """CatBoost OOF predictions"""
    seeds = SEED_POOL[:n_seeds]
    all_oof = []

    for seed in seeds:
        params = {
            'iterations': 1000, 'depth': 8, 'learning_rate': lr,
            'l2_leaf_reg': 3.0, 'random_state': seed, 'verbose': 0,
            'early_stopping_rounds': 50, 'loss_function': 'MAE'
        }
        gkf = GroupKFold(n_splits=n_splits)
        oof = np.zeros(len(X))

        for train_idx, val_idx in gkf.split(X, y, groups):
            model = CatBoostRegressor(**params)
            model.fit(X[train_idx], y[train_idx],
                     eval_set=(X[val_idx], y[val_idx]), use_best_model=True)
            oof[val_idx] = model.predict(X[val_idx])
            del model

        all_oof.append(oof.copy())
        gc.collect()

    return np.mean(all_oof, axis=0)

def run_xgboost(X, y, groups, n_splits, n_seeds, lr=0.05):
    """XGBoost OOF predictions"""
    seeds = SEED_POOL[:n_seeds]
    all_oof = []

    for seed in seeds:
        params = {
            'n_estimators': 1000, 'max_depth': 8, 'learning_rate': lr,
            'reg_lambda': 3.0, 'random_state': seed, 'verbosity': 0,
            'early_stopping_rounds': 50, 'objective': 'reg:absoluteerror'
        }
        gkf = GroupKFold(n_splits=n_splits)
        oof = np.zeros(len(X))

        for train_idx, val_idx in gkf.split(X, y, groups):
            model = xgb.XGBRegressor(**params)
            model.fit(X[train_idx], y[train_idx],
                     eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            oof[val_idx] = model.predict(X[val_idx])
            del model

        all_oof.append(oof.copy())
        gc.collect()

    return np.mean(all_oof, axis=0)

def run_lightgbm(X, y, groups, n_splits, n_seeds, lr=0.05):
    """LightGBM OOF predictions"""
    seeds = SEED_POOL[:n_seeds]
    all_oof = []

    for seed in seeds:
        params = {
            'n_estimators': 1000, 'max_depth': 8, 'learning_rate': lr,
            'reg_lambda': 3.0, 'random_state': seed, 'verbosity': -1,
            'objective': 'mae'
        }
        gkf = GroupKFold(n_splits=n_splits)
        oof = np.zeros(len(X))

        for train_idx, val_idx in gkf.split(X, y, groups):
            model = lgb.LGBMRegressor(**params)
            model.fit(X[train_idx], y[train_idx],
                     eval_set=[(X[val_idx], y[val_idx])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            oof[val_idx] = model.predict(X[val_idx])
            del model

        all_oof.append(oof.copy())
        gc.collect()

    return np.mean(all_oof, axis=0)

def evaluate(pred_dx, pred_dy, start_xy, y_abs):
    """Calculate CV"""
    pred_x = start_xy[:, 0] + pred_dx
    pred_y = start_xy[:, 1] + pred_dy
    return np.sqrt((pred_x - y_abs[:, 0])**2 + (pred_y - y_abs[:, 1])**2).mean()

def main():
    print("=" * 70)
    print("exp_092: Multi-Model Ensemble")
    print("=" * 70)

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

    print(f"  Episodes: {len(X)}")

    results = {}
    baseline_cv = 13.5358

    # CatBoost only (baseline)
    print("\n[2] CatBoost (11-fold, 7-seeds)")
    cat_dx = run_catboost(X, y_dx, groups, 11, 7)
    cat_dy = run_catboost(X, y_dy, groups, 11, 7)
    cv = evaluate(cat_dx, cat_dy, start_xy, y_abs)
    results['catboost'] = cv
    print(f"  CV: {cv:.4f}")

    # XGBoost only
    print("\n[3] XGBoost (11-fold, 7-seeds)")
    xgb_dx = run_xgboost(X, y_dx, groups, 11, 7)
    xgb_dy = run_xgboost(X, y_dy, groups, 11, 7)
    cv = evaluate(xgb_dx, xgb_dy, start_xy, y_abs)
    results['xgboost'] = cv
    print(f"  CV: {cv:.4f}")

    # LightGBM only
    print("\n[4] LightGBM (11-fold, 7-seeds)")
    lgb_dx = run_lightgbm(X, y_dx, groups, 11, 7)
    lgb_dy = run_lightgbm(X, y_dy, groups, 11, 7)
    cv = evaluate(lgb_dx, lgb_dy, start_xy, y_abs)
    results['lightgbm'] = cv
    print(f"  CV: {cv:.4f}")

    # Simple average ensemble
    print("\n[5] Simple Average Ensemble (1:1:1)")
    ens_dx = (cat_dx + xgb_dx + lgb_dx) / 3
    ens_dy = (cat_dy + xgb_dy + lgb_dy) / 3
    cv = evaluate(ens_dx, ens_dy, start_xy, y_abs)
    results['ensemble_avg'] = cv
    print(f"  CV: {cv:.4f}")

    # Weighted ensemble (CatBoost heavy)
    print("\n[6] Weighted Ensemble (Cat:XGB:LGB = 0.6:0.2:0.2)")
    ens_dx = 0.6*cat_dx + 0.2*xgb_dx + 0.2*lgb_dx
    ens_dy = 0.6*cat_dy + 0.2*xgb_dy + 0.2*lgb_dy
    cv = evaluate(ens_dx, ens_dy, start_xy, y_abs)
    results['ensemble_0.6_0.2_0.2'] = cv
    print(f"  CV: {cv:.4f}")

    # CatBoost + XGBoost only
    print("\n[7] Cat + XGB (0.5:0.5)")
    ens_dx = 0.5*cat_dx + 0.5*xgb_dx
    ens_dy = 0.5*cat_dy + 0.5*xgb_dy
    cv = evaluate(ens_dx, ens_dy, start_xy, y_abs)
    results['cat_xgb_0.5'] = cv
    print(f"  CV: {cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Best LB (exp_083): {baseline_cv:.4f}")
    print("-" * 70)

    for name, cv in sorted(results.items(), key=lambda x: x[1]):
        diff = cv - baseline_cv
        marker = " *** IMPROVED! ***" if diff < 0 else ""
        print(f"  {name:25s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    print("=" * 70)

if __name__ == "__main__":
    main()
