"""
exp_006: 최신 기술 적용

Kaggle/논문 검색 결과 적용:
1. Differencing (차분) - 변화량 포착
2. Rolling Statistics (std, min, max) - 변동성 포착
3. Relative Position Features - 상대적 위치
4. Feature Interactions - 피처 간 상호작용
5. Velocity/Acceleration - 속도/가속도

참고: arXiv:2401.03410 - Pass Prediction in Soccer
- Kicker 기준 상대 위치가 절대 위치보다 중요
- 최근 패스 방향/속도 중요

작성일: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../../../../data')


def create_baseline_features(df):
    """Baseline 피처"""
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

    return df


def add_ema_features(df, ema_span=2):
    """EMA 피처 (Best span=2)"""
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['move_distance'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    result_map = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map).fillna(0)

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

    return df


def add_position_features(df):
    """Position Continuous 피처 (exp_005에서 효과 검증됨)"""
    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    return df


def add_advanced_features(df):
    """최신 기술 적용 피처"""

    # ========================================
    # 1. Differencing (차분) - 변화량 포착
    # ========================================
    # 위치 변화량
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['diff_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)

    # 골 거리 변화량 (공격 진전도)
    df['diff_goal_dist'] = df.groupby('game_episode')['goal_distance'].diff().fillna(0)

    # ========================================
    # 2. Rolling Statistics - 변동성 포착
    # ========================================
    # 위치 변동성 (최근 3개 패스)
    df['rolling_std_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['rolling_std_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    # 이동 거리 변동성
    df['rolling_std_dist'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    # ========================================
    # 3. Velocity/Acceleration - 속도/가속도
    # ========================================
    # 속도 (이동 거리/패스 간격은 1로 가정)
    df['velocity'] = df['move_distance']

    # 가속도 (속도 변화)
    df['acceleration'] = df.groupby('game_episode')['velocity'].diff().fillna(0)

    # ========================================
    # 4. Relative Position - 상대적 위치
    # ========================================
    # EMA 위치 대비 현재 위치 (트렌드에서 벗어난 정도)
    df['rel_x_from_ema'] = df['start_x'] - df['ema_start_x']
    df['rel_y_from_ema'] = df['start_y'] - df['ema_start_y']

    # 에피소드 시작점 대비 진전도
    df['episode_start_x'] = df.groupby('game_episode')['start_x'].transform('first')
    df['episode_start_y'] = df.groupby('game_episode')['start_y'].transform('first')
    df['progress_from_start_x'] = df['start_x'] - df['episode_start_x']
    df['progress_from_start_y'] = df['start_y'] - df['episode_start_y']

    # ========================================
    # 5. Feature Interactions - 피처 상호작용
    # ========================================
    # 위치 x 방향 상호작용
    df['x_times_direction'] = df['start_x'] * df['direction']

    # 골 거리 x 속도 상호작용
    df['goal_dist_times_velocity'] = df['goal_distance'] * df['velocity']

    # 성공률 x 점유율 상호작용
    df['success_times_possession'] = df['ema_success_rate'] * df['ema_possession']

    # ========================================
    # 6. Momentum Features - 모멘텀
    # ========================================
    # 연속 전진 횟수
    df['is_forward'] = (df['diff_x'] > 0).astype(int)
    df['forward_streak'] = df.groupby('game_episode')['is_forward'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )

    # 평균 전진 거리
    df['avg_forward_x'] = df.groupby('game_episode')['diff_x'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).fillna(0)

    return df


def run_cv(X, y, groups):
    gkf = GroupKFold(n_splits=3)
    params = {
        'iterations': 1000,
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


def main():
    print("=" * 70)
    print("exp_006: 최신 기술 적용")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드 및 피처 생성...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_baseline_features(train_df)
    train_df = add_ema_features(train_df, ema_span=2)
    train_df = add_position_features(train_df)
    train_df = add_advanced_features(train_df)

    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # 피처 정의
    BASELINE = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    EMA = [
        'ema_start_x', 'ema_start_y', 'ema_goal_distance',
        'ema_distance', 'ema_success_rate', 'ema_possession'
    ]

    POSITION = [
        'dist_to_goal_line', 'dist_to_sideline', 'dist_to_center_y', 'position_zone'
    ]

    DIFF = ['diff_x', 'diff_y', 'diff_goal_dist']

    ROLLING = ['rolling_std_x', 'rolling_std_y', 'rolling_std_dist']

    VELOCITY = ['velocity', 'acceleration']

    RELATIVE = ['rel_x_from_ema', 'rel_y_from_ema', 'progress_from_start_x', 'progress_from_start_y']

    INTERACTIONS = ['x_times_direction', 'goal_dist_times_velocity', 'success_times_possession']

    MOMENTUM = ['forward_streak', 'avg_forward_x']

    # 실험
    experiments = [
        ("Baseline + EMA + Position (기준)", BASELINE + EMA + POSITION),
        ("+ Differencing", BASELINE + EMA + POSITION + DIFF),
        ("+ Rolling Stats", BASELINE + EMA + POSITION + ROLLING),
        ("+ Velocity/Accel", BASELINE + EMA + POSITION + VELOCITY),
        ("+ Relative Position", BASELINE + EMA + POSITION + RELATIVE),
        ("+ Interactions", BASELINE + EMA + POSITION + INTERACTIONS),
        ("+ Momentum", BASELINE + EMA + POSITION + MOMENTUM),
        ("+ ALL Advanced", BASELINE + EMA + POSITION + DIFF + ROLLING + VELOCITY + RELATIVE + INTERACTIONS + MOMENTUM),
    ]

    results = {}
    for name, features in experiments:
        print(f"\n[실험] {name} ({len(features)}개 피처)")
        X = last_passes[features].fillna(0).replace([np.inf, -np.inf], 0).values
        cv = run_cv(X, y, groups)
        results[name] = cv

    # 결과 비교
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)

    baseline = results["Baseline + EMA + Position (기준)"]
    for name, cv in results.items():
        diff = baseline - cv
        marker = "***" if cv == min(results.values()) else ""
        print(f"  {name:35s}: CV {cv:.4f}  ({diff:+.4f}) {marker}")

    best_name = min(results, key=results.get)
    best_cv = min(results.values())
    print(f"\n  Best: {best_name} (CV {best_cv:.4f})")
    print("=" * 70)


if __name__ == '__main__':
    main()
