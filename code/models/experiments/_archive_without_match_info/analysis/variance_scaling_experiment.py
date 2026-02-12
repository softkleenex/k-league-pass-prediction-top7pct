"""
분산 스케일링 실험

발견: 예측 분산이 실제보다 작음 (mean-regression)
  - 실제 X std: 23.85, 예측 X std: 20.22
  - 실제 Y std: 24.35, 예측 Y std: 20.37

실험: 예측값의 분산을 실제값 수준으로 확대

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


def scale_predictions(pred, y_mean, y_std, pred_mean, pred_std, alpha=1.0):
    """예측값 분산 스케일링"""
    # 표준화
    standardized = (pred - pred_mean) / pred_std

    # 목표 분산으로 스케일링 (alpha로 조절)
    target_std = pred_std + alpha * (y_std - pred_std)
    scaled = standardized * target_std + y_mean

    return scaled


def main():
    print("\n" + "=" * 80)
    print("분산 스케일링 실험")
    print("=" * 80)

    # Load data
    print("\n[1] 데이터 로딩...")
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

    # Calculate target statistics
    y_mean_x, y_std_x = y[:, 0].mean(), y[:, 0].std()
    y_mean_y, y_std_y = y[:, 1].mean(), y[:, 1].std()

    print(f"  실제값 통계:")
    print(f"    X: mean={y_mean_x:.2f}, std={y_std_x:.2f}")
    print(f"    Y: mean={y_mean_y:.2f}, std={y_std_y:.2f}")

    # CV with different scaling factors
    print("\n[2] 스케일링 팩터별 CV 테스트...")

    gkf = GroupKFold(n_splits=3)

    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

    # Test different alpha values
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    results = {}

    for alpha in alphas:
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            model_x = CatBoostRegressor(**params)
            model_y = CatBoostRegressor(**params)

            model_x.fit(X[train_idx], y[train_idx, 0])
            model_y.fit(X[train_idx], y[train_idx, 1])

            pred_x = model_x.predict(X[val_idx])
            pred_y = model_y.predict(X[val_idx])

            # Calculate prediction statistics
            pred_mean_x, pred_std_x = pred_x.mean(), pred_x.std()
            pred_mean_y, pred_std_y = pred_y.mean(), pred_y.std()

            if alpha > 0:
                # Apply scaling
                pred_x_scaled = scale_predictions(pred_x, y_mean_x, y_std_x,
                                                   pred_mean_x, pred_std_x, alpha)
                pred_y_scaled = scale_predictions(pred_y, y_mean_y, y_std_y,
                                                   pred_mean_y, pred_std_y, alpha)

                # Clip to valid range
                pred_x_scaled = np.clip(pred_x_scaled, 0, 105)
                pred_y_scaled = np.clip(pred_y_scaled, 0, 68)
            else:
                pred_x_scaled = pred_x
                pred_y_scaled = pred_y

            errors = np.sqrt((pred_x_scaled - y[val_idx, 0])**2 +
                            (pred_y_scaled - y[val_idx, 1])**2)
            fold_scores.append(errors.mean())

        cv_mean = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        results[alpha] = (cv_mean, cv_std)

        print(f"  alpha={alpha:.2f}: CV {cv_mean:.4f} ± {cv_std:.4f}")

    # Find best alpha
    best_alpha = min(results.keys(), key=lambda k: results[k][0])
    best_cv = results[best_alpha][0]
    baseline_cv = results[0.0][0]
    improvement = baseline_cv - best_cv

    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)

    print(f"""
  최적 alpha: {best_alpha}
  최적 CV: {best_cv:.4f}
  Baseline CV: {baseline_cv:.4f}
  개선: {improvement:+.4f}
""")

    if improvement > 0:
        print("  → 분산 스케일링으로 개선!")
    else:
        print("  → 분산 스케일링 효과 없음")

    print("=" * 80)


if __name__ == '__main__':
    main()
