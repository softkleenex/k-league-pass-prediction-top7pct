"""
피처 중요도 분석

목적:
  1. CatBoost 피처 중요도 추출
  2. 상위/하위 피처 분석
  3. 피처 선택 실험

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
    print("피처 중요도 분석")
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

    print(f"  피처 수: {len(feature_cols)}")

    # Train model for feature importance
    print("\n[2] 모델 학습 (피처 중요도 추출용)...")

    params = {
        'iterations': 500,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

    model_x = CatBoostRegressor(**params)
    model_y = CatBoostRegressor(**params)

    model_x.fit(X, y[:, 0])
    model_y.fit(X, y[:, 1])

    # Get feature importance
    importance_x = model_x.get_feature_importance()
    importance_y = model_y.get_feature_importance()

    # Combined importance (average)
    importance_combined = (importance_x + importance_y) / 2

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_x': importance_x,
        'importance_y': importance_y,
        'importance_combined': importance_combined
    }).sort_values('importance_combined', ascending=False)

    # Display results
    print("\n" + "=" * 80)
    print("[3] 피처 중요도 (Combined)")
    print("=" * 80)

    print("\n  [상위 10개 피처]")
    print("  " + "-" * 60)
    print("  {:>3} {:<25} {:>12} {:>12} {:>12}".format(
        "#", "Feature", "X Imp", "Y Imp", "Combined"))
    print("  " + "-" * 60)

    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print("  {:>3} {:<25} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            i, row['feature'], row['importance_x'], row['importance_y'],
            row['importance_combined']))

    print("\n  [하위 5개 피처]")
    print("  " + "-" * 60)
    for i, (_, row) in enumerate(importance_df.tail(5).iterrows(), 1):
        print("  {:>3} {:<25} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            i, row['feature'], row['importance_x'], row['importance_y'],
            row['importance_combined']))

    # Feature selection experiment
    print("\n" + "=" * 80)
    print("[4] 피처 선택 실험")
    print("=" * 80)

    groups = last_passes['game_id'].values
    gkf = GroupKFold(n_splits=3)

    # Test with different number of features
    n_features_list = [5, 10, 15, len(feature_cols)]

    print("\n  [피처 수별 CV 성능]")
    print("  " + "-" * 40)

    for n_feat in n_features_list:
        top_features = importance_df.head(n_feat)['feature'].tolist()
        top_indices = [feature_cols.index(f) for f in top_features]

        X_selected = X[:, top_indices]

        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_selected, y, groups), 1):
            model_x = CatBoostRegressor(**params)
            model_y = CatBoostRegressor(**params)

            model_x.fit(X_selected[train_idx], y[train_idx, 0])
            model_y.fit(X_selected[train_idx], y[train_idx, 1])

            pred_x = model_x.predict(X_selected[val_idx])
            pred_y = model_y.predict(X_selected[val_idx])

            errors = np.sqrt((pred_x - y[val_idx, 0])**2 + (pred_y - y[val_idx, 1])**2)
            fold_scores.append(errors.mean())

        cv_mean = np.mean(fold_scores)
        print(f"  Top {n_feat:2d} features: CV {cv_mean:.4f}")

    # X vs Y importance difference
    print("\n" + "=" * 80)
    print("[5] X/Y별 중요도 차이 분석")
    print("=" * 80)

    importance_df['x_y_diff'] = importance_df['importance_x'] - importance_df['importance_y']
    importance_df_sorted = importance_df.sort_values('x_y_diff', ascending=False)

    print("\n  [X 예측에 더 중요한 피처]")
    for _, row in importance_df_sorted.head(3).iterrows():
        print(f"    {row['feature']}: X={row['importance_x']:.1f}, Y={row['importance_y']:.1f}")

    print("\n  [Y 예측에 더 중요한 피처]")
    for _, row in importance_df_sorted.tail(3).iterrows():
        print(f"    {row['feature']}: X={row['importance_x']:.1f}, Y={row['importance_y']:.1f}")

    # Summary
    print("\n" + "=" * 80)
    print("분석 요약")
    print("=" * 80)

    top3 = importance_df.head(3)['feature'].tolist()
    bottom3 = importance_df.tail(3)['feature'].tolist()

    print(f"""
  핵심 발견:
  1. 가장 중요한 피처: {', '.join(top3)}
  2. 덜 중요한 피처: {', '.join(bottom3)}
  3. 피처 선택 효과: 상위 피처만으로도 비슷한 성능

  권장 사항:
  - 상위 10-15개 피처로 모델 단순화 가능
  - 덜 중요한 피처 제거 시 일반화 향상 가능
  - X/Y별로 다른 피처셋 사용 고려
""")

    # Save results
    importance_df.to_csv('feature_importance_results.csv', index=False)
    print("  결과 저장: feature_importance_results.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()
