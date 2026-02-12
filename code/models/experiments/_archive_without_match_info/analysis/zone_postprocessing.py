"""
Zone별 후처리 보정 실험

아이디어:
  - 각 Zone별로 예측 편향(bias) 계산
  - Zone별 보정값 적용
  - CV로 효과 검증

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


def calculate_zone_bias(df, oof_pred):
    """각 Zone별 bias 계산"""
    df = df.copy()
    df['pred_x'] = oof_pred[:, 0]
    df['pred_y'] = oof_pred[:, 1]
    df['bias_x'] = df['pred_x'] - df['end_x']
    df['bias_y'] = df['pred_y'] - df['end_y']

    zone_bias = df.groupby(['zone_x', 'zone_y']).agg({
        'bias_x': 'mean',
        'bias_y': 'mean',
        'end_x': 'count'
    }).rename(columns={'end_x': 'count'}).reset_index()

    return zone_bias


def apply_zone_correction(pred, zones_x, zones_y, zone_bias_dict, alpha=1.0):
    """Zone별 보정 적용"""
    corrected = pred.copy()

    for i in range(len(pred)):
        zone_key = (zones_x[i], zones_y[i])
        if zone_key in zone_bias_dict:
            bias_x, bias_y = zone_bias_dict[zone_key]
            corrected[i, 0] -= alpha * bias_x
            corrected[i, 1] -= alpha * bias_y

    # Clip to valid range
    corrected[:, 0] = np.clip(corrected[:, 0], 0, 105)
    corrected[:, 1] = np.clip(corrected[:, 1], 0, 68)

    return corrected


def main():
    print("\n" + "=" * 80)
    print("Zone별 후처리 보정 실험")
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
    zones_x = last_passes['zone_x'].values
    zones_y = last_passes['zone_y'].values

    # Nested CV: outer for evaluation, inner for bias calculation
    print("\n[2] Nested CV로 보정 효과 검증...")

    gkf = GroupKFold(n_splits=3)

    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

    results = {
        'no_correction': [],
        'correction_0.5': [],
        'correction_1.0': [],
        'correction_1.5': []
    }

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n  Fold {fold}/3:")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        zones_x_train = zones_x[train_idx]
        zones_y_train = zones_y[train_idx]
        zones_x_val = zones_x[val_idx]
        zones_y_val = zones_y[val_idx]

        # Train model
        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)
        model_x.fit(X_train, y_train[:, 0])
        model_y.fit(X_train, y_train[:, 1])

        # Predict on train for bias calculation
        train_pred = np.column_stack([
            model_x.predict(X_train),
            model_y.predict(X_train)
        ])

        # Calculate zone bias from training data
        train_df_fold = last_passes.iloc[train_idx].copy()
        zone_bias = calculate_zone_bias(train_df_fold, train_pred)
        zone_bias_dict = {
            (int(row['zone_x']), int(row['zone_y'])): (row['bias_x'], row['bias_y'])
            for _, row in zone_bias.iterrows()
        }

        # Predict on validation
        val_pred = np.column_stack([
            model_x.predict(X_val),
            model_y.predict(X_val)
        ])

        # Calculate errors with different correction strengths
        for alpha, key in [(0, 'no_correction'), (0.5, 'correction_0.5'),
                           (1.0, 'correction_1.0'), (1.5, 'correction_1.5')]:
            if alpha == 0:
                corrected = val_pred.copy()
                corrected[:, 0] = np.clip(corrected[:, 0], 0, 105)
                corrected[:, 1] = np.clip(corrected[:, 1], 0, 68)
            else:
                corrected = apply_zone_correction(val_pred, zones_x_val, zones_y_val,
                                                   zone_bias_dict, alpha)

            errors = np.sqrt((corrected[:, 0] - y_val[:, 0])**2 +
                            (corrected[:, 1] - y_val[:, 1])**2)
            results[key].append(errors.mean())

        print(f"    보정없음: {results['no_correction'][-1]:.4f}")
        print(f"    보정 0.5: {results['correction_0.5'][-1]:.4f}")
        print(f"    보정 1.0: {results['correction_1.0'][-1]:.4f}")
        print(f"    보정 1.5: {results['correction_1.5'][-1]:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)

    print("\n  [CV 결과 비교]")
    print("  " + "-" * 50)

    for key in results:
        mean = np.mean(results[key])
        std = np.std(results[key])
        print(f"  {key:15s}: {mean:.4f} ± {std:.4f}")

    best_key = min(results.keys(), key=lambda k: np.mean(results[k]))
    baseline = np.mean(results['no_correction'])
    best = np.mean(results[best_key])
    improvement = baseline - best

    print(f"\n  최적 설정: {best_key}")
    print(f"  개선량: {improvement:+.4f}")

    if improvement > 0:
        print(f"\n  결론: Zone별 보정이 효과 있음!")
    else:
        print(f"\n  결론: Zone별 보정 효과 없음")

    print("=" * 80)


if __name__ == '__main__':
    main()
