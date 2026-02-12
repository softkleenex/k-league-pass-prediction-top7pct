"""
exp_003 v2: EMA 피처 (타겟 누수 수정)

문제: ema_dx, ema_dy 계산 시 현재 패스의 end_x, end_y가 포함됨
해결: EMA 계산 시 shift(1)을 사용하여 이전 패스까지만 사용

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


def create_features_with_ema_safe(df, ema_span=5):
    """타겟 누수 없는 EMA 피처 생성

    핵심: 모든 EMA는 "이전 패스까지의 정보"만 사용
    - end_x, end_y는 절대 직접 사용하지 않음
    - EMA 계산 후 shift(1)로 현재 패스 정보 제외
    """

    # ========================================
    # 1. 기본 위치 피처 (start_x, start_y만 사용)
    # ========================================
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    # ========================================
    # 2. 이전 패스 위치 기반 이동 (end_x/y 대신 start_x/y 사용)
    # ========================================
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1)
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1)

    # 이동 방향: 현재 start - 이전 start (end 사용 안함!)
    df['move_dx'] = (df['start_x'] - df['prev_start_x']).fillna(0)
    df['move_dy'] = (df['start_y'] - df['prev_start_y']).fillna(0)

    # 방향 인코딩
    angle = np.degrees(np.arctan2(df['move_dy'], df['move_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    # ========================================
    # 3. EMA 피처 (shift로 현재 값 제외!)
    # ========================================
    # 시작 위치 EMA (shift(1)로 현재 제외)
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])  # 첫 패스는 자기 위치

    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    # 이동 방향 EMA (shift(1)로 현재 제외)
    df['ema_move_dx'] = df.groupby('game_episode')['move_dx'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0)

    df['ema_move_dy'] = df.groupby('game_episode')['move_dy'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0)

    # 이동 거리 (start 기반)
    df['move_distance'] = np.sqrt(df['move_dx']**2 + df['move_dy']**2)
    df['ema_distance'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0)

    # 골 거리 EMA (shift(1)로 현재 제외)
    df['ema_goal_distance'] = df.groupby('game_episode')['goal_distance'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['goal_distance'])

    # ========================================
    # 4. 에피소드 패턴 피처
    # ========================================
    df['pass_count'] = df.groupby('game_episode').cumcount() + 1
    df['episode_length'] = df.groupby('game_episode')['pass_count'].transform('max')
    df['progress'] = df['pass_count'] / df['episode_length']

    # 누적 이동 거리
    df['cumsum_distance'] = df.groupby('game_episode')['move_distance'].cumsum()

    # 성공률
    result_map = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map).fillna(0)
    df['success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.expanding().mean()
    )

    # EMA 성공률 (shift(1)로 현재 제외)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    # 결과 인코딩
    result_map2 = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map2).fillna(2).astype(int)

    # ========================================
    # 5. 팀/소유권 피처
    # ========================================
    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)

    df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

    # EMA 소유권 (shift(1)로 현재 제외)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

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
    # 6. 위치 영역 피처
    # ========================================
    df['in_attacking_third'] = (df['start_x'] > 70).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)

    # prev_dx, prev_dy 호환성 (기존 모델용)
    df['prev_dx'] = df['move_dx']
    df['prev_dy'] = df['move_dy']

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
    print("exp_003 v2: EMA 피처 (타겟 누수 수정)")
    print("=" * 70)
    print("\n핵심 수정: 모든 EMA에 shift(1) 적용")
    print("- 이전 패스까지의 정보만 사용")
    print("- end_x, end_y 직접 사용 안함")

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Total passes: {len(train_df):,}")

    # 피처 생성 (EMA span=5)
    print("\n[2] 피처 생성 (EMA span=5)...")
    train_df = create_features_with_ema_safe(train_df, ema_span=5)

    # 마지막 패스
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    # 피처 정의
    BASELINE = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    EXTRA = ['in_attacking_third', 'in_center', 'progress', 'success_rate', 'cumsum_distance']

    EMA_FEATURES = [
        'ema_start_x', 'ema_start_y', 'ema_move_dx', 'ema_move_dy',
        'ema_distance', 'ema_goal_distance', 'ema_success_rate', 'ema_possession'
    ]

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    experiments = [
        ("Baseline", BASELINE),
        ("Baseline + Extra", BASELINE + EXTRA),
        ("Baseline + EMA", BASELINE + EMA_FEATURES),
        ("Baseline + Extra + EMA", BASELINE + EXTRA + EMA_FEATURES),
    ]

    results = {}
    for name, features in experiments:
        print(f"\n[실험] {name} ({len(features)}개 피처)")
        X = last_passes[features].fillna(0).replace([np.inf, -np.inf], 0).values
        cv = run_cv(X, y, groups)
        results[name] = cv

    # EMA span 최적화 (Best 조합으로)
    best_combo_name = min(results, key=results.get)
    if "EMA" in best_combo_name:
        print("\n" + "=" * 70)
        print("EMA span 최적화")
        print("=" * 70)

        best_features = BASELINE + EXTRA + EMA_FEATURES
        span_results = {}

        for span in [3, 5, 7, 10]:
            print(f"\n[EMA span={span}]")
            train_df_temp = pd.read_csv(DATA_DIR / 'train.csv')
            train_df_temp = create_features_with_ema_safe(train_df_temp, ema_span=span)
            last_temp = train_df_temp.groupby('game_episode').last().reset_index()

            X = last_temp[best_features].fillna(0).replace([np.inf, -np.inf], 0).values
            cv = run_cv(X, y, groups)
            span_results[span] = cv
    else:
        span_results = {}

    # 결과 비교
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)

    baseline = results["Baseline"]
    print("\n[피처 조합별]")
    for name, cv in results.items():
        diff = baseline - cv
        marker = "***" if cv == min(results.values()) else ""
        print(f"  {name:25s}: CV {cv:.4f}  ({diff:+.4f}) {marker}")

    if span_results:
        print("\n[EMA span별]")
        for span, cv in span_results.items():
            diff = baseline - cv
            marker = "***" if cv == min(span_results.values()) else ""
            print(f"  span={span:2d}: CV {cv:.4f}  ({diff:+.4f}) {marker}")

    # 최고 결과
    all_results = {**results}
    if span_results:
        all_results.update({f"span_{k}": v for k, v in span_results.items()})

    best_name = min(all_results, key=all_results.get)
    best_cv = min(all_results.values())

    print(f"\n  Best: {best_name} (CV {best_cv:.4f})")
    print(f"  개선: {baseline - best_cv:+.4f}")
    print("=" * 70)

    return best_cv


if __name__ == '__main__':
    main()
