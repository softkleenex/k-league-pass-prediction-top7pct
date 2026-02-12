"""
예측값 분포 분석

목적:
  1. 우리 예측값 vs 실제값 분포 비교
  2. 편향 분석
  3. 아웃라이어 분석

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
    print("예측값 분포 분석")
    print("=" * 80)

    # Load data
    print("\n[1] 데이터 로딩 및 OOF 예측...")
    train_df = pd.read_csv('../../../../data/train.csv')
    train_df = create_phase1a_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    exclude_cols = [
        'game_id', 'game_episode', 'player_id', 'team_id', 'period_id',
        'time_seconds', 'type_name', 'result_name', 'is_home',
        'start_x', 'start_y', 'end_x', 'end_y'
    ]
    feature_cols = [col for col in last_passes.columns if col not in exclude_cols]

    X = last_passes[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # Get OOF predictions
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
        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)

        model_x.fit(X[train_idx], y[train_idx, 0])
        model_y.fit(X[train_idx], y[train_idx, 1])

        oof_pred[val_idx, 0] = model_x.predict(X[val_idx])
        oof_pred[val_idx, 1] = model_y.predict(X[val_idx])

    print("  OOF 예측 완료")

    # Compare distributions
    print("\n" + "=" * 80)
    print("[2] 실제값 vs 예측값 분포 비교")
    print("=" * 80)

    print("\n  [end_x 분포]")
    print("  {:>15} {:>12} {:>12}".format("", "실제값", "예측값"))
    print("  " + "-" * 40)
    print("  {:>15} {:>12.2f} {:>12.2f}".format("Mean", y[:, 0].mean(), oof_pred[:, 0].mean()))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("Std", y[:, 0].std(), oof_pred[:, 0].std()))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("Min", y[:, 0].min(), oof_pred[:, 0].min()))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("Max", y[:, 0].max(), oof_pred[:, 0].max()))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("25%", np.percentile(y[:, 0], 25), np.percentile(oof_pred[:, 0], 25)))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("50%", np.percentile(y[:, 0], 50), np.percentile(oof_pred[:, 0], 50)))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("75%", np.percentile(y[:, 0], 75), np.percentile(oof_pred[:, 0], 75)))

    print("\n  [end_y 분포]")
    print("  {:>15} {:>12} {:>12}".format("", "실제값", "예측값"))
    print("  " + "-" * 40)
    print("  {:>15} {:>12.2f} {:>12.2f}".format("Mean", y[:, 1].mean(), oof_pred[:, 1].mean()))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("Std", y[:, 1].std(), oof_pred[:, 1].std()))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("Min", y[:, 1].min(), oof_pred[:, 1].min()))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("Max", y[:, 1].max(), oof_pred[:, 1].max()))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("25%", np.percentile(y[:, 1], 25), np.percentile(oof_pred[:, 1], 25)))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("50%", np.percentile(y[:, 1], 50), np.percentile(oof_pred[:, 1], 50)))
    print("  {:>15} {:>12.2f} {:>12.2f}".format("75%", np.percentile(y[:, 1], 75), np.percentile(oof_pred[:, 1], 75)))

    # Bias analysis
    print("\n" + "=" * 80)
    print("[3] 편향 분석")
    print("=" * 80)

    bias_x = oof_pred[:, 0] - y[:, 0]
    bias_y = oof_pred[:, 1] - y[:, 1]

    print(f"\n  [X 편향]")
    print(f"    평균 편향: {bias_x.mean():+.4f}")
    print(f"    편향 표준편차: {bias_x.std():.4f}")
    print(f"    최대 과대예측: {bias_x.max():+.2f}")
    print(f"    최대 과소예측: {bias_x.min():+.2f}")

    print(f"\n  [Y 편향]")
    print(f"    평균 편향: {bias_y.mean():+.4f}")
    print(f"    편향 표준편차: {bias_y.std():.4f}")
    print(f"    최대 과대예측: {bias_y.max():+.2f}")
    print(f"    최대 과소예측: {bias_y.min():+.2f}")

    # Error analysis by region
    print("\n" + "=" * 80)
    print("[4] 영역별 에러 분석")
    print("=" * 80)

    errors = np.sqrt(bias_x**2 + bias_y**2)

    # By target x region
    print("\n  [목표 X 영역별 에러]")
    x_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 105)]
    for low, high in x_bins:
        mask = (y[:, 0] >= low) & (y[:, 0] < high)
        if mask.sum() > 0:
            region_error = errors[mask].mean()
            count = mask.sum()
            print(f"    X={low:3d}~{high:3d}: 에러 {region_error:.2f} (n={count})")

    # By target y region
    print("\n  [목표 Y 영역별 에러]")
    y_bins = [(0, 17), (17, 34), (34, 51), (51, 68)]
    for low, high in y_bins:
        mask = (y[:, 1] >= low) & (y[:, 1] < high)
        if mask.sum() > 0:
            region_error = errors[mask].mean()
            count = mask.sum()
            print(f"    Y={low:3d}~{high:3d}: 에러 {region_error:.2f} (n={count})")

    # Outlier analysis
    print("\n" + "=" * 80)
    print("[5] 아웃라이어 분석")
    print("=" * 80)

    threshold_90 = np.percentile(errors, 90)
    threshold_95 = np.percentile(errors, 95)
    threshold_99 = np.percentile(errors, 99)

    print(f"\n  에러 분포:")
    print(f"    90% 이하: {threshold_90:.2f}")
    print(f"    95% 이하: {threshold_95:.2f}")
    print(f"    99% 이하: {threshold_99:.2f}")
    print(f"    최대 에러: {errors.max():.2f}")

    # Extreme outliers
    extreme_mask = errors > threshold_95
    print(f"\n  상위 5% 아웃라이어 ({extreme_mask.sum()}개):")
    print(f"    평균 에러: {errors[extreme_mask].mean():.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("분석 요약")
    print("=" * 80)

    print(f"""
  핵심 발견:
  1. 예측 분산이 실제보다 작음 (mean-regression 경향)
     - 실제 X std: {y[:, 0].std():.2f}, 예측 X std: {oof_pred[:, 0].std():.2f}
     - 실제 Y std: {y[:, 1].std():.2f}, 예측 Y std: {oof_pred[:, 1].std():.2f}

  2. 편향은 거의 없음 (X: {bias_x.mean():+.2f}, Y: {bias_y.mean():+.2f})

  3. 에러 분포
     - 90%: {threshold_90:.2f} 이하
     - 상위 5%가 전체 에러에 큰 영향

  권장 사항:
  - 예측 분산 확대 (scaling) 시도
  - 아웃라이어 케이스 특별 처리
  - 영역별 모델 분리 고려
""")

    print("=" * 80)


if __name__ == '__main__':
    main()
