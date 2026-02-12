"""
exp_008: Target Encoding + Optuna Hyperparameter Tuning

Perplexity 연구 기반 최신 기술:
1. Target Encoding (K-fold로 leakage 방지)
2. Optuna 하이퍼파라미터 최적화
3. Defensive pressure 피처 (상대 플레이어 근접도 - 데이터 없어서 대체)
4. Multi-scale rolling features

작성일: 2025-12-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../../../../data')


def create_target_encoding(df, col, target_cols, n_splits=5, smoothing=10):
    """
    K-Fold Target Encoding (leakage 방지)
    - 각 fold에서 OOF 방식으로 인코딩
    - smoothing으로 희소 카테고리 처리
    """
    df = df.copy()

    for target_col in target_cols:
        new_col = f'{col}_te_{target_col}'
        df[new_col] = np.nan

        global_mean = df[target_col].mean()

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(df):
            train_df = df.iloc[train_idx]

            # 카테고리별 통계
            agg = train_df.groupby(col)[target_col].agg(['mean', 'count'])

            # Smoothing: (n * mean + m * global_mean) / (n + m)
            smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)

            # Validation set에 적용
            df.iloc[val_idx, df.columns.get_loc(new_col)] = df.iloc[val_idx][col].map(smoothed)

        # 결측치는 global mean으로
        df[new_col] = df[new_col].fillna(global_mean)

    return df


def create_all_features(df, is_train=True, te_stats=None):
    """exp_006 피처 + Target Encoding + Multi-scale"""

    # ========================================
    # Baseline
    # ========================================
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['zone_id'] = df['zone_x'] * 6 + df['zone_y']  # 0-35

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

    # ========================================
    # EMA (span=2)
    # ========================================
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

    # ========================================
    # Position
    # ========================================
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    # ========================================
    # Differencing
    # ========================================
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['diff_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['diff_goal_dist'] = df.groupby('game_episode')['goal_distance'].diff().fillna(0)

    # ========================================
    # Multi-scale Rolling Stats (NEW!)
    # ========================================
    for window in [3, 5]:
        df[f'rolling_std_x_{window}'] = df.groupby('game_episode')['start_x'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).fillna(0)
        df[f'rolling_std_y_{window}'] = df.groupby('game_episode')['start_y'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).fillna(0)
        df[f'rolling_mean_dist_{window}'] = df.groupby('game_episode')['move_distance'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        ).fillna(0)

    # ========================================
    # Velocity/Acceleration
    # ========================================
    df['velocity'] = df['move_distance']
    df['acceleration'] = df.groupby('game_episode')['velocity'].diff().fillna(0)

    # ========================================
    # Relative Position
    # ========================================
    df['rel_x_from_ema'] = df['start_x'] - df['ema_start_x']
    df['rel_y_from_ema'] = df['start_y'] - df['ema_start_y']
    df['episode_start_x'] = df.groupby('game_episode')['start_x'].transform('first')
    df['episode_start_y'] = df.groupby('game_episode')['start_y'].transform('first')
    df['progress_from_start_x'] = df['start_x'] - df['episode_start_x']
    df['progress_from_start_y'] = df['start_y'] - df['episode_start_y']

    # ========================================
    # Interactions
    # ========================================
    df['x_times_direction'] = df['start_x'] * df['direction']
    df['goal_dist_times_velocity'] = df['goal_distance'] * df['velocity']
    df['success_times_possession'] = df['ema_success_rate'] * df['ema_possession']

    # NEW: 추가 interaction features
    df['zone_times_direction'] = df['zone_id'] * df['direction']
    df['x_times_velocity'] = df['start_x'] * df['velocity']
    df['y_times_direction'] = df['start_y'] * df['direction']

    # ========================================
    # Momentum
    # ========================================
    df['is_forward'] = (df['diff_x'] > 0).astype(int)
    df['forward_streak'] = df.groupby('game_episode')['is_forward'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df['avg_forward_x'] = df.groupby('game_episode')['diff_x'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).fillna(0)

    return df


def run_cv(X, y, groups, params):
    """CatBoost CV with given params"""
    gkf = GroupKFold(n_splits=3)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model_x = CatBoostRegressor(**params, verbose=0)
        model_y = CatBoostRegressor(**params, verbose=0)
        model_x.fit(X[train_idx], y[train_idx, 0])
        model_y.fit(X[train_idx], y[train_idx, 1])
        pred_x = model_x.predict(X[val_idx])
        pred_y = model_y.predict(X[val_idx])
        errors = np.sqrt((pred_x - y[val_idx, 0])**2 + (pred_y - y[val_idx, 1])**2)
        fold_scores.append(errors.mean())

    return np.mean(fold_scores)


def objective(trial, X, y, groups):
    """Optuna objective function"""
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'depth': trial.suggest_int('depth', 6, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_state': 42,
    }

    cv_score = run_cv(X, y, groups, params)
    return cv_score


def main():
    print("=" * 70)
    print("exp_008: Target Encoding + Optuna Tuning")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드 및 피처 생성...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_all_features(train_df)

    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # ========================================
    # Target Encoding
    # ========================================
    print("\n[2] Target Encoding 적용...")

    # zone_id에 대한 target encoding
    last_passes = create_target_encoding(
        last_passes, 'zone_id', ['end_x', 'end_y'], n_splits=5, smoothing=10
    )

    # direction에 대한 target encoding
    last_passes = create_target_encoding(
        last_passes, 'direction', ['end_x', 'end_y'], n_splits=5, smoothing=10
    )

    print(f"  Target Encoding 컬럼: zone_id_te_end_x, zone_id_te_end_y, direction_te_end_x, direction_te_end_y")

    # 피처 정의
    BASE_FEATURES = [
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
        # Rolling Stats (multi-scale)
        'rolling_std_x_3', 'rolling_std_y_3', 'rolling_mean_dist_3',
        'rolling_std_x_5', 'rolling_std_y_5', 'rolling_mean_dist_5',
        # Velocity/Accel
        'velocity', 'acceleration',
        # Relative Position
        'rel_x_from_ema', 'rel_y_from_ema', 'progress_from_start_x', 'progress_from_start_y',
        # Interactions
        'x_times_direction', 'goal_dist_times_velocity', 'success_times_possession',
        'zone_times_direction', 'x_times_velocity', 'y_times_direction',
        # Momentum
        'forward_streak', 'avg_forward_x'
    ]

    TE_FEATURES = [
        'zone_id_te_end_x', 'zone_id_te_end_y',
        'direction_te_end_x', 'direction_te_end_y'
    ]

    # ========================================
    # 실험 1: Base (exp_006 + multi-scale)
    # ========================================
    print("\n" + "=" * 70)
    print("[실험 1] Base Features (exp_006 + multi-scale)")
    print("=" * 70)

    base_params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
    }

    X_base = last_passes[BASE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    cv_base = run_cv(X_base, y, groups, base_params)
    print(f"  CV: {cv_base:.4f}")

    # ========================================
    # 실험 2: + Target Encoding
    # ========================================
    print("\n" + "=" * 70)
    print("[실험 2] + Target Encoding")
    print("=" * 70)

    ALL_FEATURES = BASE_FEATURES + TE_FEATURES
    X_te = last_passes[ALL_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    cv_te = run_cv(X_te, y, groups, base_params)
    print(f"  CV: {cv_te:.4f}  ({cv_base - cv_te:+.4f})")

    # ========================================
    # 실험 3: Optuna Hyperparameter Tuning
    # ========================================
    print("\n" + "=" * 70)
    print("[실험 3] Optuna Hyperparameter Tuning (20 trials)")
    print("=" * 70)

    # Best features 사용
    if cv_te < cv_base:
        X_best = X_te
        print("  Using: Base + Target Encoding features")
    else:
        X_best = X_base
        print("  Using: Base features only")

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_best, y, groups),
        n_trials=20,
        show_progress_bar=True
    )

    print(f"\n  Best params: {study.best_params}")
    print(f"  Best CV: {study.best_value:.4f}")

    cv_optuna = study.best_value
    best_params = study.best_params
    best_params['random_state'] = 42

    # ========================================
    # 결과 요약
    # ========================================
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)

    results = {
        "Base (exp_006+multi-scale)": cv_base,
        "+ Target Encoding": cv_te,
        "+ Optuna Tuning": cv_optuna,
    }

    baseline = 14.2513  # exp_006 CV

    for name, cv in results.items():
        diff = baseline - cv
        marker = "***" if cv == min(results.values()) else ""
        print(f"  {name:30s}: CV {cv:.4f}  (vs exp_006: {diff:+.4f}) {marker}")

    print(f"\n  Best params: {best_params}")
    print("=" * 70)


if __name__ == '__main__':
    main()
