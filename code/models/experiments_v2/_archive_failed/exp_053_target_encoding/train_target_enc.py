"""
exp_053: Target Encoding for Zone-level Statistics

핵심 아이디어:
- 각 zone에서 평균적으로 어디로 패스하는지 학습
- Zone별 end_x, end_y 통계 (mean, std, median)
- 데이터 누수 방지를 위해 fold-out encoding 사용

참고: Kaggle tabular competitions에서 널리 사용되는 기법
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"


def create_baseline_features(df):
    """Baseline 피처"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['zone_id'] = df['zone_x'] * 6 + df['zone_y']  # 0-35

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
    """EMA 피처"""
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
    """Position 피처"""
    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    return df


def add_advanced_features(df):
    """Advanced 피처"""
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['diff_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['diff_goal_dist'] = df.groupby('game_episode')['goal_distance'].diff().fillna(0)

    df['rolling_std_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['rolling_std_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['rolling_std_dist'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['velocity'] = df['move_distance']
    df['acceleration'] = df.groupby('game_episode')['velocity'].diff().fillna(0)

    df['rel_x_from_ema'] = df['start_x'] - df['ema_start_x']
    df['rel_y_from_ema'] = df['start_y'] - df['ema_start_y']

    df['episode_start_x'] = df.groupby('game_episode')['start_x'].transform('first')
    df['episode_start_y'] = df.groupby('game_episode')['start_y'].transform('first')
    df['progress_from_start_x'] = df['start_x'] - df['episode_start_x']
    df['progress_from_start_y'] = df['start_y'] - df['episode_start_y']

    df['x_times_direction'] = df['start_x'] * df['direction']
    df['goal_dist_times_velocity'] = df['goal_distance'] * df['velocity']
    df['success_times_possession'] = df['ema_success_rate'] * df['ema_possession']

    df['is_forward'] = (df['diff_x'] > 0).astype(int)
    df['forward_streak'] = df.groupby('game_episode')['is_forward'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df['avg_forward_x'] = df.groupby('game_episode')['diff_x'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).fillna(0)

    return df


def create_target_encoding(train_last, val_last, test_last, smooth=10):
    """
    Zone-level Target Encoding (Fold-out)

    Zone별 end_x, end_y 통계를 피처로 추가
    - 데이터 누수 방지: train에서만 통계 계산 → val/test에 적용
    """
    # 전체 평균
    global_mean_x = train_last['end_x'].mean()
    global_mean_y = train_last['end_y'].mean()

    # Zone별 통계 (train에서만)
    zone_stats = train_last.groupby('zone_id').agg({
        'end_x': ['mean', 'std', 'median', 'count'],
        'end_y': ['mean', 'std', 'median']
    })
    zone_stats.columns = ['_'.join(col) for col in zone_stats.columns]
    zone_stats = zone_stats.reset_index()

    # Smoothing (regularization)
    zone_stats['zone_end_x_mean_smooth'] = (
        zone_stats['end_x_count'] * zone_stats['end_x_mean'] + smooth * global_mean_x
    ) / (zone_stats['end_x_count'] + smooth)

    zone_stats['zone_end_y_mean_smooth'] = (
        zone_stats['end_x_count'] * zone_stats['end_y_mean'] + smooth * global_mean_y
    ) / (zone_stats['end_x_count'] + smooth)

    # 적용
    for df in [train_last, val_last, test_last]:
        if df is None:
            continue
        df_merged = df.merge(zone_stats[['zone_id', 'zone_end_x_mean_smooth', 'zone_end_y_mean_smooth',
                                          'end_x_std', 'end_y_std', 'end_x_median', 'end_y_median']],
                             on='zone_id', how='left')

        df['te_zone_end_x'] = df_merged['zone_end_x_mean_smooth'].fillna(global_mean_x)
        df['te_zone_end_y'] = df_merged['zone_end_y_mean_smooth'].fillna(global_mean_y)
        df['te_zone_x_std'] = df_merged['end_x_std'].fillna(0)
        df['te_zone_y_std'] = df_merged['end_y_std'].fillna(0)
        df['te_zone_x_median'] = df_merged['end_x_median'].fillna(global_mean_x)
        df['te_zone_y_median'] = df_merged['end_y_median'].fillna(global_mean_y)

    return train_last, val_last, test_last


def create_direction_target_encoding(train_last, val_last, test_last, smooth=5):
    """Direction-level Target Encoding"""
    global_mean_x = train_last['end_x'].mean()
    global_mean_y = train_last['end_y'].mean()

    dir_stats = train_last.groupby('direction').agg({
        'end_x': ['mean', 'count'],
        'end_y': ['mean']
    })
    dir_stats.columns = ['_'.join(col) for col in dir_stats.columns]
    dir_stats = dir_stats.reset_index()

    dir_stats['te_dir_end_x'] = (
        dir_stats['end_x_count'] * dir_stats['end_x_mean'] + smooth * global_mean_x
    ) / (dir_stats['end_x_count'] + smooth)

    dir_stats['te_dir_end_y'] = (
        dir_stats['end_x_count'] * dir_stats['end_y_mean'] + smooth * global_mean_y
    ) / (dir_stats['end_x_count'] + smooth)

    for df in [train_last, val_last, test_last]:
        if df is None:
            continue
        df_merged = df.merge(dir_stats[['direction', 'te_dir_end_x', 'te_dir_end_y']],
                             on='direction', how='left')
        df['te_dir_end_x'] = df_merged['te_dir_end_x'].fillna(global_mean_x)
        df['te_dir_end_y'] = df_merged['te_dir_end_y'].fillna(global_mean_y)

    return train_last, val_last, test_last


def run_cv_with_te(last_passes, features_base, groups):
    """Target Encoding을 Fold-out으로 적용한 CV"""
    gkf = GroupKFold(n_splits=3)
    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0,
        'early_stopping_rounds': 50
    }

    TE_FEATURES = [
        'te_zone_end_x', 'te_zone_end_y',
        'te_zone_x_std', 'te_zone_y_std',
        'te_zone_x_median', 'te_zone_y_median',
        'te_dir_end_x', 'te_dir_end_y'
    ]

    y = last_passes[['end_x', 'end_y']].values

    fold_scores = []
    models_x = []
    models_y = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(last_passes, y, groups), 1):
        train_last = last_passes.iloc[train_idx].copy()
        val_last = last_passes.iloc[val_idx].copy()

        # Target encoding (fold-out)
        train_last, val_last, _ = create_target_encoding(train_last, val_last, None)
        train_last, val_last, _ = create_direction_target_encoding(train_last, val_last, None)

        # 피처
        features = features_base + TE_FEATURES

        X_train = train_last[features].fillna(0).replace([np.inf, -np.inf], 0).values
        X_val = val_last[features].fillna(0).replace([np.inf, -np.inf], 0).values
        y_train = train_last[['end_x', 'end_y']].values
        y_val = val_last[['end_x', 'end_y']].values

        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)

        model_x.fit(X_train, y_train[:, 0], eval_set=(X_val, y_val[:, 0]), use_best_model=True)
        model_y.fit(X_train, y_train[:, 1], eval_set=(X_val, y_val[:, 1]), use_best_model=True)

        pred_x = model_x.predict(X_val)
        pred_y = model_y.predict(X_val)
        errors = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
        fold_scores.append(errors.mean())
        print(f"    Fold {fold}: {fold_scores[-1]:.4f}")

        models_x.append(model_x)
        models_y.append(model_y)

    return np.mean(fold_scores), models_x, models_y, features


def run_cv_baseline(last_passes, features, groups):
    """Baseline CV (TE 없음)"""
    gkf = GroupKFold(n_splits=3)
    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0,
        'early_stopping_rounds': 50
    }

    y = last_passes[['end_x', 'end_y']].values

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(last_passes, y, groups), 1):
        X_train = last_passes.iloc[train_idx][features].fillna(0).replace([np.inf, -np.inf], 0).values
        X_val = last_passes.iloc[val_idx][features].fillna(0).replace([np.inf, -np.inf], 0).values
        y_train = y[train_idx]
        y_val = y[val_idx]

        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)

        model_x.fit(X_train, y_train[:, 0], eval_set=(X_val, y_val[:, 0]), use_best_model=True)
        model_y.fit(X_train, y_train[:, 1], eval_set=(X_val, y_val[:, 1]), use_best_model=True)

        pred_x = model_x.predict(X_val)
        pred_y = model_y.predict(X_val)
        errors = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
        fold_scores.append(errors.mean())
        print(f"    Fold {fold}: {fold_scores[-1]:.4f}")

    return np.mean(fold_scores)


def main():
    print("=" * 70)
    print("exp_053: Target Encoding for Zone-level Statistics")
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

    groups = last_passes['game_id'].values

    # 기본 피처
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

    ADVANCED = [
        'diff_x', 'diff_y', 'diff_goal_dist',
        'rolling_std_x', 'rolling_std_y', 'rolling_std_dist',
        'velocity', 'acceleration',
        'rel_x_from_ema', 'rel_y_from_ema', 'progress_from_start_x', 'progress_from_start_y',
        'x_times_direction', 'goal_dist_times_velocity', 'success_times_possession',
        'forward_streak', 'avg_forward_x'
    ]

    features_base = BASELINE + EMA + POSITION + ADVANCED

    # 실험
    print("\n[2] 실험 비교...")

    print(f"\n--- 기존 (exp_047 수준) ({len(features_base)}개 피처) ---")
    cv_baseline = run_cv_baseline(last_passes, features_base, groups)

    print(f"\n--- + Target Encoding ---")
    cv_te, models_x, models_y, features_all = run_cv_with_te(last_passes, features_base, groups)

    # 결과 비교
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)
    diff = cv_baseline - cv_te
    print(f"  기존 (exp_047 수준): CV {cv_baseline:.4f}")
    print(f"  + Target Encoding:  CV {cv_te:.4f} ({diff:+.4f})")

    if cv_te < cv_baseline:
        print("  → Target Encoding 유효!")
        best_cv = cv_te
    else:
        print("  → Target Encoding 무효, 기존 사용")
        best_cv = cv_baseline

    # Test 예측
    print("\n[3] Test 예측 생성...")
    test_df = pd.read_csv(DATA_DIR / 'test.csv')

    test_episodes = []
    for _, row in test_df.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)

        if 'dx' not in ep_df.columns:
            ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
            ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']

        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)

    test_all = pd.concat(test_episodes, ignore_index=True)

    test_all = create_baseline_features(test_all)
    test_all = add_ema_features(test_all, ema_span=2)
    test_all = add_position_features(test_all)
    test_all = add_advanced_features(test_all)

    test_last = test_all.groupby('game_episode').last().reset_index()

    # TE는 전체 train에서 계산
    train_last_all = last_passes.copy()
    _, test_last, _ = create_target_encoding(train_last_all, test_last, None)
    _, test_last, _ = create_direction_target_encoding(train_last_all, test_last, None)

    # 예측
    X_test = test_last[features_all].fillna(0).replace([np.inf, -np.inf], 0).values

    pred_x_all = np.zeros(len(X_test))
    pred_y_all = np.zeros(len(X_test))

    for mx, my in zip(models_x, models_y):
        pred_x_all += mx.predict(X_test) / len(models_x)
        pred_y_all += my.predict(X_test) / len(models_y)

    # 제출 파일
    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x_all,
        'end_y': pred_y_all
    })

    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_te_cv{best_cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  저장: {output_path}")

    print("\n[예측 분포]")
    print(f"  end_x: mean={submission['end_x'].mean():.2f}, std={submission['end_x'].std():.2f}")
    print(f"  end_y: mean={submission['end_y'].mean():.2f}, std={submission['end_y'].std():.2f}")

    # Feature importance
    print("\n[피처 중요도 Top 15]")
    importance_x = models_x[0].get_feature_importance()
    importance_y = models_y[0].get_feature_importance()
    avg_importance = (importance_x + importance_y) / 2

    feat_imp = pd.DataFrame({
        'feature': features_all,
        'importance': avg_importance
    })
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print(feat_imp.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"Best CV: {best_cv:.4f}")
    print(f"제출 파일: {output_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
