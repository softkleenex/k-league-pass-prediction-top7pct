"""
에러 분석: 어디서 예측이 틀리는가?

목적:
  1. CV에서 OOF 예측 수집
  2. 에러가 큰 에피소드 분석
  3. 패턴 발견 → 개선 방향 도출

작성일: 2025-12-21
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


def create_phase1a_features(df):
    """Phase1A 피처 생성"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['time_left'] = 5400 - df['time_seconds']
    df['game_clock_min'] = np.where(df['period_id'] == 1, df['time_seconds'] / 60.0, 45.0 + df['time_seconds'] / 60.0)

    df['pass_count'] = df.groupby('game_episode').cumcount() + 1

    df['is_home_encoded'] = df['is_home'].astype(int)
    type_map = {'Pass': 0, 'Carry': 1}
    df['type_encoded'] = df['type_name'].map(type_map).fillna(2).astype(int)
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)

    df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

    df['team_switch_event'] = (df.groupby('game_episode')['is_final_team'].diff() != 0).astype(int)
    df['team_switches'] = df.groupby('game_episode')['team_switch_event'].cumsum()

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
    df = df.drop(columns=['dx', 'dy', 'team_switch_event', 'final_team_id'], errors='ignore')

    return df


def main():
    print("\n" + "=" * 80)
    print("에러 분석: 어디서 예측이 틀리는가?")
    print("=" * 80)

    # Load data
    print("\n[1] 데이터 로딩...")
    train_df = pd.read_csv('../../../../data/train.csv')
    train_df = create_phase1a_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    # Prepare features
    exclude_cols = [
        'game_id', 'game_episode', 'player_id', 'team_id', 'period_id',
        'time_seconds', 'type_name', 'result_name', 'is_home',
        'start_x', 'start_y', 'end_x', 'end_y'
    ]
    feature_cols = [col for col in last_passes.columns if col not in exclude_cols]

    X = last_passes[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # Collect OOF predictions
    print("\n[2] OOF 예측 수집 (3-Fold CV)...")
    gkf = GroupKFold(n_splits=3)
    oof_pred = np.zeros_like(y)

    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"  Fold {fold}...", end='', flush=True)

        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)

        model_x.fit(X[train_idx], y[train_idx, 0])
        model_y.fit(X[train_idx], y[train_idx, 1])

        oof_pred[val_idx, 0] = model_x.predict(X[val_idx])
        oof_pred[val_idx, 1] = model_y.predict(X[val_idx])

        print(" done")

    # Calculate errors
    print("\n[3] 에러 계산...")
    errors = np.sqrt((oof_pred[:, 0] - y[:, 0])**2 + (oof_pred[:, 1] - y[:, 1])**2)
    error_x = np.abs(oof_pred[:, 0] - y[:, 0])
    error_y = np.abs(oof_pred[:, 1] - y[:, 1])

    last_passes['error'] = errors
    last_passes['error_x'] = error_x
    last_passes['error_y'] = error_y
    last_passes['pred_x'] = oof_pred[:, 0]
    last_passes['pred_y'] = oof_pred[:, 1]

    print(f"  전체 평균 에러: {errors.mean():.4f}")
    print(f"  X 평균 에러: {error_x.mean():.4f}")
    print(f"  Y 평균 에러: {error_y.mean():.4f}")

    # Analyze high error cases
    print("\n" + "=" * 80)
    print("[4] 고에러 케이스 분석 (상위 10%)")
    print("=" * 80)

    threshold = np.percentile(errors, 90)
    high_error = last_passes[last_passes['error'] > threshold]
    low_error = last_passes[last_passes['error'] <= np.percentile(errors, 50)]

    print(f"\n  고에러 케이스: {len(high_error)} ({len(high_error)/len(last_passes)*100:.1f}%)")
    print(f"  고에러 평균: {high_error['error'].mean():.2f}")
    print(f"  저에러 평균: {low_error['error'].mean():.2f}")

    # Compare distributions
    print("\n  [특성 비교: 고에러 vs 저에러]")
    print("  " + "-" * 60)

    compare_cols = ['zone_x', 'zone_y', 'goal_distance', 'pass_count',
                    'is_final_team', 'team_possession_pct', 'game_clock_min',
                    'start_x', 'start_y', 'end_x', 'end_y']

    for col in compare_cols:
        if col in high_error.columns:
            high_mean = high_error[col].mean()
            low_mean = low_error[col].mean()
            diff = high_mean - low_mean
            diff_pct = (diff / low_mean * 100) if low_mean != 0 else 0
            marker = "***" if abs(diff_pct) > 20 else ""
            print(f"  {col:20s}: 고={high_mean:8.2f} | 저={low_mean:8.2f} | 차={diff:+8.2f} ({diff_pct:+.1f}%) {marker}")

    # Analyze by zone
    print("\n" + "=" * 80)
    print("[5] Zone별 에러 분석")
    print("=" * 80)

    zone_error = last_passes.groupby(['zone_x', 'zone_y'])['error'].agg(['mean', 'count']).reset_index()
    zone_error = zone_error.sort_values('mean', ascending=False)

    print("\n  [에러가 높은 Zone Top 10]")
    print("  " + "-" * 40)
    for _, row in zone_error.head(10).iterrows():
        print(f"  Zone ({int(row['zone_x'])}, {int(row['zone_y'])}): 에러 {row['mean']:.2f} (n={int(row['count'])})")

    print("\n  [에러가 낮은 Zone Top 5]")
    print("  " + "-" * 40)
    for _, row in zone_error.tail(5).iterrows():
        print(f"  Zone ({int(row['zone_x'])}, {int(row['zone_y'])}): 에러 {row['mean']:.2f} (n={int(row['count'])})")

    # Analyze by target position
    print("\n" + "=" * 80)
    print("[6] 목표 위치별 에러 분석")
    print("=" * 80)

    last_passes['target_zone_x'] = (last_passes['end_x'] / (105/6)).astype(int).clip(0, 5)
    last_passes['target_zone_y'] = (last_passes['end_y'] / (68/6)).astype(int).clip(0, 5)

    target_error = last_passes.groupby(['target_zone_x', 'target_zone_y'])['error'].agg(['mean', 'count']).reset_index()
    target_error = target_error.sort_values('mean', ascending=False)

    print("\n  [목표 Zone별 에러 Top 10]")
    print("  " + "-" * 40)
    for _, row in target_error.head(10).iterrows():
        print(f"  Target ({int(row['target_zone_x'])}, {int(row['target_zone_y'])}): 에러 {row['mean']:.2f} (n={int(row['count'])})")

    # Analyze prediction bias
    print("\n" + "=" * 80)
    print("[7] 예측 편향 분석")
    print("=" * 80)

    bias_x = (oof_pred[:, 0] - y[:, 0]).mean()
    bias_y = (oof_pred[:, 1] - y[:, 1]).mean()

    print(f"\n  X 편향: {bias_x:+.4f} (예측이 {'높음' if bias_x > 0 else '낮음'})")
    print(f"  Y 편향: {bias_y:+.4f} (예측이 {'높음' if bias_y > 0 else '낮음'})")

    # Analyze by game time
    print("\n" + "=" * 80)
    print("[8] 경기 시간별 에러 분석")
    print("=" * 80)

    last_passes['time_bin'] = pd.cut(last_passes['game_clock_min'],
                                      bins=[0, 15, 30, 45, 60, 75, 90, 120],
                                      labels=['0-15', '15-30', '30-45', '45-60', '60-75', '75-90', '90+'])
    time_error = last_passes.groupby('time_bin')['error'].agg(['mean', 'count'])
    print("\n  [시간대별 에러]")
    print(time_error.to_string())

    # Summary
    print("\n" + "=" * 80)
    print("분석 요약")
    print("=" * 80)

    print("""
  핵심 발견:
  1. 고에러 케이스 특성 확인
  2. Zone별 에러 패턴 파악
  3. 시간대별 에러 변화 확인
  4. 예측 편향 존재 여부 확인

  다음 단계:
  - 고에러 Zone에 특화된 피처 추가
  - 시간대별 모델 분리 고려
  - 편향 보정 후처리 적용
""")

    # Save results
    last_passes.to_csv('error_analysis_results.csv', index=False)
    print("  결과 저장: error_analysis_results.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()
