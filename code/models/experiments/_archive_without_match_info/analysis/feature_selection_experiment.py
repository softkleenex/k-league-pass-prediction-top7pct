"""
피처 선택 실험: Top 10 피처로 모델 학습

발견: Top 10 피처만으로 CV 15.20 (전체 19개: 15.41)
→ 피처 선택으로 0.2점 개선 가능!

작성일: 2025-12-21
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
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


def load_test_data():
    """Test 데이터 로드"""
    test_meta = pd.read_csv('../../../../data/test.csv')
    test_dfs = []
    for idx, row in test_meta.iterrows():
        ep_path = Path('../../../../data') / row['path']
        if ep_path.exists():
            ep_df = pd.read_csv(ep_path)
            ep_df['game_episode'] = row['game_episode']
            test_dfs.append(ep_df)
    return pd.concat(test_dfs, ignore_index=True)


def main():
    print("\n" + "=" * 80)
    print("피처 선택 실험: Top 10 피처")
    print("=" * 80)

    # Top 10 features (from feature importance analysis)
    TOP_FEATURES = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    print(f"\n  Selected features ({len(TOP_FEATURES)}개):")
    for i, f in enumerate(TOP_FEATURES, 1):
        print(f"    {i}. {f}")

    # Load data
    print("\n[1] 데이터 로딩...")
    train_df = pd.read_csv('../../../../data/train.csv')
    train_df = create_phase1a_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  Train 에피소드: {len(last_passes)}")

    X = last_passes[TOP_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # CV
    print("\n[2] 3-Fold CV...")
    gkf = GroupKFold(n_splits=3)
    fold_scores = []

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

        pred_x = model_x.predict(X[val_idx])
        pred_y = model_y.predict(X[val_idx])

        errors = np.sqrt((pred_x - y[val_idx, 0])**2 + (pred_y - y[val_idx, 1])**2)
        fold_scores.append(errors.mean())
        print(f"    Fold {fold}: {fold_scores[-1]:.4f}")

    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    print(f"\n  CV Mean: {cv_mean:.4f} ± {cv_std:.4f}")

    # Compare with baseline
    baseline_cv = 15.4881  # exp_036
    improvement = baseline_cv - cv_mean

    print(f"\n  Baseline (exp_036, 17 features): {baseline_cv:.4f}")
    print(f"  This (10 features): {cv_mean:.4f}")
    print(f"  Improvement: {improvement:+.4f}")

    if improvement > 0:
        print("  → 피처 선택으로 개선!")
    else:
        print("  → 피처 선택 효과 없음")

    # Train final model
    print("\n[3] Final Model 학습...")
    final_model_x = CatBoostRegressor(**params)
    final_model_y = CatBoostRegressor(**params)
    final_model_x.fit(X, y[:, 0])
    final_model_y.fit(X, y[:, 1])

    # Save models
    with open('model_x_top10.pkl', 'wb') as f:
        pickle.dump(final_model_x, f)
    with open('model_y_top10.pkl', 'wb') as f:
        pickle.dump(final_model_y, f)

    # Predict on test
    print("\n[4] Test 예측...")
    test_df = load_test_data()
    test_df = create_phase1a_features(test_df)
    test_last = test_df.groupby('game_episode').last().reset_index()

    X_test = test_last[TOP_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values

    pred_x = np.clip(final_model_x.predict(X_test), 0, 105)
    pred_y = np.clip(final_model_y.predict(X_test), 0, 68)

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })

    filename = f'submission_top10_cv{cv_mean:.4f}.csv'
    submission.to_csv(filename, index=False)

    # Save results
    results = {
        'experiment': 'feature_selection_top10',
        'cv_mean': float(cv_mean),
        'cv_std': float(cv_std),
        'cv_folds': [float(s) for s in fold_scores],
        'n_features': len(TOP_FEATURES),
        'feature_cols': TOP_FEATURES,
        'baseline_cv': baseline_cv,
        'improvement': float(improvement)
    }

    with open('feature_selection_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)
    print(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  vs Baseline: {improvement:+.4f}")
    print(f"  Submission: {filename}")
    print("=" * 80)


if __name__ == '__main__':
    main()
