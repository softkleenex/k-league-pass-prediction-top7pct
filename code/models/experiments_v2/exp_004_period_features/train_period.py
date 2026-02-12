"""
exp_004: Period 기반 피처 + EMA 개선

핵심 정보 (대회 Q&A):
- L→R 공격 통일: period_id(전/후반)에 따라 좌표 반전됨
- 같은 하프 내에서는 모든 episode가 동일한 공격 방향
- period_id = 1 (전반), 2 (후반)

실험 내용:
1. period_id 피처 추가
2. EMA span 미세 조정 (2, 3, 4)
3. 다양한 조합 테스트

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


def add_period_features(df):
    """Period (전반/후반) 기반 피처"""
    # period_id: 1 = 전반, 2 = 후반
    df['is_second_half'] = (df['period_id'] == 2).astype(int)

    # 하프 내 시간 진행 (0-1 정규화)
    # 전반: 0-45분, 후반: 45-90분
    df['half_progress'] = df.groupby(['game_id', 'period_id'])['game_episode'].transform(
        lambda x: (x.rank(method='dense') - 1) / max(1, x.nunique() - 1)
    ).fillna(0)

    # 경기 전체 진행률
    df['game_progress'] = df.groupby('game_id')['game_episode'].transform(
        lambda x: (x.rank(method='dense') - 1) / max(1, x.nunique() - 1)
    ).fillna(0)

    # 전반/후반별 에피소드 번호
    df['episode_in_half'] = df.groupby(['game_id', 'period_id'])['game_episode'].transform(
        lambda x: x.rank(method='dense')
    )

    return df


def add_safe_ema_features(df, ema_span=3):
    """EMA 피처 (shift(1)로 안전하게)"""
    # 이동 거리 계산
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['move_distance'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    # 성공률 계산
    result_map = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map).fillna(0)

    # EMA 피처 (shift(1)로 안전하게)
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
    print("exp_004: Period 기반 피처 + EMA 개선")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Total passes: {len(train_df):,}")

    # 피처 정의
    BASELINE = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    EMA_FEATURES = [
        'ema_start_x', 'ema_start_y', 'ema_goal_distance',
        'ema_distance', 'ema_success_rate', 'ema_possession'
    ]

    PERIOD_FEATURES = [
        'is_second_half', 'half_progress', 'game_progress', 'episode_in_half'
    ]

    # 실험 1: EMA span 미세 조정
    print("\n" + "=" * 70)
    print("실험 1: EMA span 미세 조정 (Baseline + EMA)")
    print("=" * 70)

    span_results = {}
    for span in [2, 3, 4]:
        print(f"\n[EMA span={span}]")
        train_df_temp = pd.read_csv(DATA_DIR / 'train.csv')
        train_df_temp = create_baseline_features(train_df_temp)
        train_df_temp = add_safe_ema_features(train_df_temp, ema_span=span)
        last_temp = train_df_temp.groupby('game_episode').last().reset_index()

        features = BASELINE + EMA_FEATURES
        X = last_temp[features].fillna(0).replace([np.inf, -np.inf], 0).values
        y = last_temp[['end_x', 'end_y']].values
        groups = last_temp['game_id'].values

        cv = run_cv(X, y, groups)
        span_results[span] = cv

    # 실험 2: Period 피처 추가
    print("\n" + "=" * 70)
    print("실험 2: Period 피처 추가")
    print("=" * 70)

    best_span = min(span_results, key=span_results.get)
    print(f"\n[Best EMA span={best_span} 사용]")

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_baseline_features(train_df)
    train_df = add_period_features(train_df)
    train_df = add_safe_ema_features(train_df, ema_span=best_span)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    experiments = [
        ("Baseline + EMA", BASELINE + EMA_FEATURES),
        ("Baseline + EMA + Period", BASELINE + EMA_FEATURES + PERIOD_FEATURES),
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

    print("\n[EMA span별]")
    for span, cv in span_results.items():
        marker = "***" if cv == min(span_results.values()) else ""
        print(f"  span={span}: CV {cv:.4f} {marker}")

    print("\n[피처 조합별]")
    baseline_ema = results["Baseline + EMA"]
    for name, cv in results.items():
        diff = baseline_ema - cv
        marker = "***" if cv == min(results.values()) else ""
        print(f"  {name:30s}: CV {cv:.4f}  ({diff:+.4f}) {marker}")

    # Best 결과
    all_results = {**span_results, **results}
    best_name = min(all_results, key=all_results.get)
    best_cv = min(all_results.values())

    print(f"\n  Best: {best_name} (CV {best_cv:.4f})")
    print("=" * 70)

    return best_cv, best_span


if __name__ == '__main__':
    main()
