"""
exp_002: 축구 도메인 지식 최대 활용

핵심 인사이트:
1. 전반/후반 공격 방향이 다름 (period_id)
2. 홈팀은 특정 방향으로 공격
3. 에피소드 내 패스 패턴

작성일: 2025-12-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../../../../data')


def create_domain_features(df):
    """축구 도메인 지식 기반 피처"""

    # ========================================
    # 1. 기본 위치 피처
    # ========================================
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)

    # ========================================
    # 2. 공격 방향 정규화
    # ========================================
    # 축구에서 전반/후반에 공격 방향이 바뀜
    # 홈팀이 전반에 오른쪽(x 증가)으로 공격한다고 가정
    # 후반에는 왼쪽(x 감소)으로 공격

    # 공격 방향 결정 (1: 오른쪽, -1: 왼쪽)
    # 홈팀 전반 = 오른쪽, 홈팀 후반 = 왼쪽
    # 어웨이팀 전반 = 왼쪽, 어웨이팀 후반 = 오른쪽
    df['attack_direction'] = np.where(
        (df['is_home'] & (df['period_id'] == 1)) |
        (~df['is_home'] & (df['period_id'] == 2)),
        1,  # 오른쪽 공격
        -1  # 왼쪽 공격
    )

    # 정규화된 좌표 (항상 오른쪽으로 공격하는 것처럼)
    df['norm_start_x'] = np.where(
        df['attack_direction'] == 1,
        df['start_x'],
        105 - df['start_x']  # x 반전
    )
    df['norm_start_y'] = np.where(
        df['attack_direction'] == 1,
        df['start_y'],
        68 - df['start_y']  # y도 반전
    )

    # ========================================
    # 3. 목표 골대까지 거리 (정규화 기준)
    # ========================================
    # 정규화 후에는 항상 (105, 34)가 상대 골대
    df['norm_goal_distance'] = np.sqrt(
        (105 - df['norm_start_x'])**2 +
        (34 - df['norm_start_y'])**2
    )
    df['norm_goal_angle'] = np.degrees(
        np.arctan2(34 - df['norm_start_y'], 105 - df['norm_start_x'])
    )

    # 기존 goal_distance도 유지 (비교용)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    # ========================================
    # 4. 이전 패스 방향 (정규화 적용)
    # ========================================
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']

    # 정규화된 dx, dy
    df['norm_dx'] = df['dx'] * df['attack_direction']
    df['norm_dy'] = df['dy'] * df['attack_direction']

    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
    df['prev_norm_dx'] = df.groupby('game_episode')['norm_dx'].shift(1).fillna(0)
    df['prev_norm_dy'] = df.groupby('game_episode')['norm_dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    # ========================================
    # 5. 에피소드 내 패스 패턴
    # ========================================
    # 패스 수
    df['pass_count'] = df.groupby('game_episode').cumcount() + 1
    df['episode_length'] = df.groupby('game_episode')['pass_count'].transform('max')

    # 진행률
    df['progress'] = df['pass_count'] / df['episode_length']

    # 이동 거리
    df['move_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['cumsum_distance'] = df.groupby('game_episode')['move_distance'].cumsum()

    # 패스 성공률
    result_map = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map).fillna(0)
    df['success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.expanding().mean()
    )

    # 결과 인코딩
    result_map2 = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map2).fillna(2).astype(int)

    # ========================================
    # 6. 팀/소유권 피처
    # ========================================
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
    # 7. 시간 피처
    # ========================================
    df['time_in_period'] = df['time_seconds']
    df['is_late_game'] = (df['time_seconds'] > 2700).astype(int)  # 45분 이후

    # ========================================
    # 8. 위치 영역 피처
    # ========================================
    # 정규화된 위치 기준
    df['norm_zone_x'] = (df['norm_start_x'] / (105/6)).astype(int).clip(0, 5)
    df['norm_zone_y'] = (df['norm_start_y'] / (68/6)).astype(int).clip(0, 5)

    # 공격 3분의 1 (정규화 기준)
    df['in_attacking_third'] = (df['norm_start_x'] > 70).astype(int)
    df['in_middle_third'] = ((df['norm_start_x'] >= 35) & (df['norm_start_x'] <= 70)).astype(int)
    df['in_defensive_third'] = (df['norm_start_x'] < 35).astype(int)

    # 중앙 vs 측면
    df['in_center'] = ((df['norm_start_y'] >= 20) & (df['norm_start_y'] <= 48)).astype(int)

    return df


def main():
    print("=" * 70)
    print("exp_002: 축구 도메인 지식 최대 활용")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Total passes: {len(train_df):,}")

    # 도메인 피처 생성
    print("\n[2] 도메인 피처 생성...")
    train_df = create_domain_features(train_df)

    # 마지막 패스
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    # 피처 정의
    # A: 기존 Top 10
    BASELINE_FEATURES = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    # B: 정규화된 피처
    NORMALIZED_FEATURES = [
        'norm_goal_distance', 'norm_zone_y', 'norm_goal_angle',
        'prev_norm_dx', 'prev_norm_dy', 'norm_zone_x',
        'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    # C: 추가 도메인 피처
    EXTRA_FEATURES = [
        'in_attacking_third', 'in_center', 'progress',
        'success_rate', 'cumsum_distance'
    ]

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # 실험들
    experiments = [
        ("Baseline Top 10", BASELINE_FEATURES),
        ("Normalized Top 10", NORMALIZED_FEATURES),
        ("Baseline + Extra", BASELINE_FEATURES + EXTRA_FEATURES),
        ("Normalized + Extra", NORMALIZED_FEATURES + EXTRA_FEATURES),
    ]

    results = {}
    for name, features in experiments:
        print(f"\n[실험] {name} ({len(features)}개 피처)")
        X = last_passes[features].fillna(0).replace([np.inf, -np.inf], 0).values
        cv = run_cv(X, y, groups)
        results[name] = cv

    # 결과 비교
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)

    baseline = results["Baseline Top 10"]
    for name, cv in results.items():
        diff = baseline - cv
        marker = "***" if cv == min(results.values()) else ""
        print(f"  {name:25s}: CV {cv:.4f}  ({diff:+.4f}) {marker}")

    best_name = min(results, key=results.get)
    print(f"\n  Best: {best_name}")
    print("=" * 70)


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


if __name__ == '__main__':
    main()
