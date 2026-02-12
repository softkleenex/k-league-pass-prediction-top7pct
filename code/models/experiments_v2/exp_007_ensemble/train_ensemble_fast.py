"""
exp_007b: 앙상블 테스트 (CatBoost + LightGBM + XGBoost)
iterations를 줄여서 빠르게 효과 확인

작성일: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../../../../data')


def create_all_features(df):
    """exp_006 피처 (ALL Advanced)"""
    # Baseline
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)

    df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

    def calc_streak(group):
        values = group['is_final_team'].values
        streaks = []
        current_streak = 0
        for val in values:
            if val == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)
        return pd.Series(streaks, index=group.index)

    df['final_poss_len'] = df.groupby('game_episode', group_keys=False).apply(calc_streak)

    # EMA (span=2)
    ema_span = 2
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['move_distance'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)

    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])

    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    df['ema_goal_distance'] = df.groupby('game_episode')['goal_distance'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['goal_distance'])

    df['ema_distance'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0)

    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    # Position
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    # Differencing
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['diff_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['diff_goal_dist'] = df.groupby('game_episode')['goal_distance'].diff().fillna(0)

    # Rolling Stats
    df['rolling_std_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)
    df['rolling_std_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)
    df['rolling_std_dist'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    # Velocity/Acceleration
    df['velocity'] = df['move_distance']
    df['acceleration'] = df.groupby('game_episode')['velocity'].diff().fillna(0)

    # Relative Position
    df['rel_x_from_ema'] = df['start_x'] - df['ema_start_x']
    df['rel_y_from_ema'] = df['start_y'] - df['ema_start_y']
    df['episode_start_x'] = df.groupby('game_episode')['start_x'].transform('first')
    df['episode_start_y'] = df.groupby('game_episode')['start_y'].transform('first')
    df['progress_from_start_x'] = df['start_x'] - df['episode_start_x']
    df['progress_from_start_y'] = df['start_y'] - df['episode_start_y']

    # Interactions
    df['x_times_direction'] = df['start_x'] * df['direction']
    df['goal_dist_times_velocity'] = df['goal_distance'] * df['velocity']
    df['success_times_possession'] = df['ema_success_rate'] * df['ema_possession']

    # Momentum
    df['is_forward'] = (df['diff_x'] > 0).astype(int)
    df['forward_streak'] = df.groupby('game_episode')['is_forward'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df['avg_forward_x'] = df.groupby('game_episode')['diff_x'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).fillna(0)

    return df


def run_cv_catboost(X, y, groups, n_iter=500):
    """CatBoost only CV"""
    gkf = GroupKFold(n_splits=3)
    params = {
        'iterations': n_iter,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)
        model_x.fit(X[train_idx], y[train_idx, 0])
        model_y.fit(X[train_idx], y[train_idx, 1])
        pred_x = model_x.predict(X[val_idx])
        pred_y = model_y.predict(X[val_idx])
        errors = np.sqrt((pred_x - y[val_idx, 0])**2 + (pred_y - y[val_idx, 1])**2)
        fold_scores.append(errors.mean())
        print(f"    Fold {fold}: {fold_scores[-1]:.4f}")

    return np.mean(fold_scores)


def run_cv_ensemble(X, y, groups, n_iter=500, weights=None):
    """Ensemble CV (Cat+LGB+XGB)"""
    if weights is None:
        weights = [1/3, 1/3, 1/3]

    gkf = GroupKFold(n_splits=3)

    cat_params = {'iterations': n_iter, 'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0}
    lgb_params = {'n_estimators': n_iter, 'max_depth': 8, 'learning_rate': 0.05, 'reg_lambda': 3.0, 'random_state': 42, 'verbose': -1, 'n_jobs': -1}
    xgb_params = {'n_estimators': n_iter, 'max_depth': 8, 'learning_rate': 0.05, 'reg_lambda': 3.0, 'random_state': 42, 'verbosity': 0, 'n_jobs': -1}

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # CatBoost
        cat_x = CatBoostRegressor(**cat_params)
        cat_y = CatBoostRegressor(**cat_params)
        cat_x.fit(X_tr, y_tr[:, 0])
        cat_y.fit(X_tr, y_tr[:, 1])
        pred_cat_x = cat_x.predict(X_val)
        pred_cat_y = cat_y.predict(X_val)

        # LightGBM
        lgb_x = LGBMRegressor(**lgb_params)
        lgb_y = LGBMRegressor(**lgb_params)
        lgb_x.fit(X_tr, y_tr[:, 0])
        lgb_y.fit(X_tr, y_tr[:, 1])
        pred_lgb_x = lgb_x.predict(X_val)
        pred_lgb_y = lgb_y.predict(X_val)

        # XGBoost
        xgb_x = XGBRegressor(**xgb_params)
        xgb_y = XGBRegressor(**xgb_params)
        xgb_x.fit(X_tr, y_tr[:, 0])
        xgb_y.fit(X_tr, y_tr[:, 1])
        pred_xgb_x = xgb_x.predict(X_val)
        pred_xgb_y = xgb_y.predict(X_val)

        # Weighted Average
        w_cat, w_lgb, w_xgb = weights
        pred_x = w_cat * pred_cat_x + w_lgb * pred_lgb_x + w_xgb * pred_xgb_x
        pred_y = w_cat * pred_cat_y + w_lgb * pred_lgb_y + w_xgb * pred_xgb_y

        errors = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
        fold_scores.append(errors.mean())
        print(f"    Fold {fold}: {fold_scores[-1]:.4f}")

    return np.mean(fold_scores)


def main():
    print("=" * 70)
    print("exp_007b: 앙상블 테스트 (CatBoost + LightGBM + XGBoost)")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드 및 피처 생성...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_all_features(train_df)

    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # 피처 정의 (exp_006 ALL Advanced)
    ALL_ADVANCED = [
        # Baseline
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct',
        # EMA
        'ema_start_x', 'ema_start_y', 'ema_goal_distance',
        'ema_distance', 'ema_success_rate', 'ema_possession',
        # Position
        'dist_to_goal_line', 'dist_to_sideline', 'dist_to_center_y', 'position_zone',
        # Differencing
        'diff_x', 'diff_y', 'diff_goal_dist',
        # Rolling Stats
        'rolling_std_x', 'rolling_std_y', 'rolling_std_dist',
        # Velocity/Accel
        'velocity', 'acceleration',
        # Relative Position
        'rel_x_from_ema', 'rel_y_from_ema', 'progress_from_start_x', 'progress_from_start_y',
        # Interactions
        'x_times_direction', 'goal_dist_times_velocity', 'success_times_possession',
        # Momentum
        'forward_streak', 'avg_forward_x'
    ]

    X = last_passes[ALL_ADVANCED].fillna(0).replace([np.inf, -np.inf], 0).values

    # 실험 1: CatBoost Only (기준)
    print("\n[실험 1] CatBoost Only (500 iter, 기준)")
    cv_cat = run_cv_catboost(X, y, groups, n_iter=500)
    print(f"  CV: {cv_cat:.4f}")

    # 실험 2: Simple Ensemble (1/3 each)
    print("\n[실험 2] Ensemble (Cat+LGB+XGB, equal weight)")
    cv_ensemble = run_cv_ensemble(X, y, groups, n_iter=500, weights=[1/3, 1/3, 1/3])
    print(f"  CV: {cv_ensemble:.4f}  ({cv_cat - cv_ensemble:+.4f})")

    # 실험 3: Cat-weighted Ensemble
    print("\n[실험 3] Ensemble (Cat 0.5, LGB 0.25, XGB 0.25)")
    cv_ensemble2 = run_cv_ensemble(X, y, groups, n_iter=500, weights=[0.5, 0.25, 0.25])
    print(f"  CV: {cv_ensemble2:.4f}  ({cv_cat - cv_ensemble2:+.4f})")

    # 결과
    print("\n" + "=" * 70)
    print("결과")
    print("=" * 70)
    results = {
        "CatBoost Only": cv_cat,
        "Ensemble (equal)": cv_ensemble,
        "Ensemble (cat-heavy)": cv_ensemble2
    }

    for name, cv in results.items():
        diff = cv_cat - cv
        marker = "***" if cv == min(results.values()) else ""
        print(f"  {name:25s}: CV {cv:.4f}  ({diff:+.4f}) {marker}")

    best_name = min(results, key=results.get)
    best_cv = min(results.values())
    print(f"\n  Best: {best_name} (CV {best_cv:.4f})")


if __name__ == '__main__':
    main()
