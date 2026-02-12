"""
exp_058: Enhanced Sequence Statistics

에피소드 전체 시퀀스에서 더 많은 통계 추출
- 전체 이동 거리
- 평균/최대 패스 길이
- 방향 변화량
- 시퀀스 길이
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
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
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['pass_length'] = np.sqrt(df['dx']**2 + df['dy']**2)
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

    for col in ['start_x', 'start_y', 'goal_distance', 'move_distance', 'is_successful', 'is_final_team']:
        target = col if col not in ['is_successful', 'is_final_team'] else col
        df[f'ema_{col}'] = df.groupby('game_episode')[col].transform(
            lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
        ).fillna(df[col] if col in df.columns else 0.5)

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

    for col in ['start_x', 'start_y', 'move_distance']:
        df[f'rolling_std_{col}'] = df.groupby('game_episode')[col].transform(
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
    df['success_times_possession'] = df['ema_is_successful'] * df['ema_is_final_team']

    df['is_forward'] = (df['diff_x'] > 0).astype(int)
    df['forward_streak'] = df.groupby('game_episode')['is_forward'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df['avg_forward_x'] = df.groupby('game_episode')['diff_x'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).fillna(0)

    return df


def add_sequence_statistics(df):
    """시퀀스 통계 피처 (핵심 신규!)"""

    # 1. 시퀀스 길이
    df['seq_length'] = df.groupby('game_episode')['start_x'].transform('count')
    df['seq_position'] = df.groupby('game_episode').cumcount() + 1
    df['seq_position_pct'] = df['seq_position'] / df['seq_length']

    # 2. 누적 이동 거리
    df['cumsum_distance'] = df.groupby('game_episode')['move_distance'].transform('cumsum')
    df['total_distance'] = df.groupby('game_episode')['move_distance'].transform('sum')
    df['remaining_distance_pct'] = 1 - (df['cumsum_distance'] / (df['total_distance'] + 1e-6))

    # 3. 패스 길이 통계
    df['mean_pass_length'] = df.groupby('game_episode')['pass_length'].transform('mean')
    df['max_pass_length'] = df.groupby('game_episode')['pass_length'].transform('max')
    df['std_pass_length'] = df.groupby('game_episode')['pass_length'].transform('std').fillna(0)
    df['pass_length_ratio'] = df['pass_length'] / (df['mean_pass_length'] + 1e-6)

    # 4. 방향 변화
    df['pass_angle'] = np.degrees(np.arctan2(df['dy'], df['dx']))
    df['angle_change'] = df.groupby('game_episode')['pass_angle'].diff().fillna(0)
    df['abs_angle_change'] = np.abs(df['angle_change'])
    df['cumsum_angle_change'] = df.groupby('game_episode')['abs_angle_change'].transform('cumsum')
    df['mean_angle_change'] = df.groupby('game_episode')['abs_angle_change'].transform('mean')

    # 5. 진전도 통계
    df['total_progress_x'] = df.groupby('game_episode')['diff_x'].transform('sum')
    df['mean_progress_x'] = df.groupby('game_episode')['diff_x'].transform('mean')
    df['progress_efficiency'] = df['total_progress_x'] / (df['total_distance'] + 1e-6)

    # 6. 성공률 통계
    df['episode_success_rate'] = df.groupby('game_episode')['is_successful'].transform('mean')
    df['success_count'] = df.groupby('game_episode')['is_successful'].transform('sum')

    # 7. 골대 접근 통계
    df['min_goal_dist'] = df.groupby('game_episode')['goal_distance'].transform('min')
    df['goal_dist_improvement'] = df['goal_distance'] - df['min_goal_dist']
    df['start_goal_dist'] = df.groupby('game_episode')['goal_distance'].transform('first')
    df['goal_approach_pct'] = (df['start_goal_dist'] - df['goal_distance']) / (df['start_goal_dist'] + 1e-6)

    # 8. 위치 분산
    df['x_variance'] = df.groupby('game_episode')['start_x'].transform('var').fillna(0)
    df['y_variance'] = df.groupby('game_episode')['start_y'].transform('var').fillna(0)
    df['position_spread'] = np.sqrt(df['x_variance'] + df['y_variance'])

    # 9. 최근 N개 패스 통계
    for n in [3, 5]:
        df[f'last{n}_mean_dx'] = df.groupby('game_episode')['dx'].transform(
            lambda x: x.rolling(n, min_periods=1).mean()
        )
        df[f'last{n}_mean_dy'] = df.groupby('game_episode')['dy'].transform(
            lambda x: x.rolling(n, min_periods=1).mean()
        )
        df[f'last{n}_std_x'] = df.groupby('game_episode')['start_x'].transform(
            lambda x: x.rolling(n, min_periods=1).std()
        ).fillna(0)

    return df


def run_cv(X, y, groups):
    """3-Fold CV"""
    gkf = GroupKFold(n_splits=3)
    params = {
        'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
        'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 50
    }

    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)

        model_x.fit(X[train_idx], y[train_idx, 0],
                   eval_set=(X[val_idx], y[val_idx, 0]), use_best_model=True)
        model_y.fit(X[train_idx], y[train_idx, 1],
                   eval_set=(X[val_idx], y[val_idx, 1]), use_best_model=True)

        pred_x = model_x.predict(X[val_idx])
        pred_y = model_y.predict(X[val_idx])
        errors = np.sqrt((pred_x - y[val_idx, 0])**2 + (pred_y - y[val_idx, 1])**2)
        fold_scores.append(errors.mean())
        print(f"    Fold {fold}: {fold_scores[-1]:.4f}")
        models.append((model_x, model_y))

    return np.mean(fold_scores), models


def main():
    print("=" * 70)
    print("exp_058: Enhanced Sequence Statistics")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')

    train_df = create_baseline_features(train_df)
    train_df = add_ema_features(train_df, ema_span=2)
    train_df = add_position_features(train_df)
    train_df = add_advanced_features(train_df)
    train_df = add_sequence_statistics(train_df)

    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    # 피처
    BASELINE = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    EMA = ['ema_start_x', 'ema_start_y', 'ema_goal_distance', 'ema_move_distance',
           'ema_is_successful', 'ema_is_final_team']

    POSITION = ['dist_to_goal_line', 'dist_to_sideline', 'dist_to_center_y', 'position_zone']

    ADVANCED = [
        'diff_x', 'diff_y', 'diff_goal_dist',
        'rolling_std_start_x', 'rolling_std_start_y', 'rolling_std_move_distance',
        'velocity', 'acceleration',
        'rel_x_from_ema', 'rel_y_from_ema', 'progress_from_start_x', 'progress_from_start_y',
        'x_times_direction', 'goal_dist_times_velocity', 'success_times_possession',
        'forward_streak', 'avg_forward_x'
    ]

    SEQ_STATS = [
        'seq_length', 'seq_position', 'seq_position_pct',
        'cumsum_distance', 'total_distance', 'remaining_distance_pct',
        'mean_pass_length', 'max_pass_length', 'std_pass_length', 'pass_length_ratio',
        'cumsum_angle_change', 'mean_angle_change',
        'total_progress_x', 'mean_progress_x', 'progress_efficiency',
        'episode_success_rate', 'success_count',
        'min_goal_dist', 'goal_dist_improvement', 'goal_approach_pct',
        'x_variance', 'y_variance', 'position_spread',
        'last3_mean_dx', 'last3_mean_dy', 'last3_std_x',
        'last5_mean_dx', 'last5_mean_dy', 'last5_std_x'
    ]

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # 실험
    print("\n[2] 실험...")

    features_base = BASELINE + EMA + POSITION + ADVANCED
    features_new = features_base + SEQ_STATS

    print(f"\n--- 기존 ({len(features_base)}개 피처) ---")
    X_base = last_passes[features_base].fillna(0).replace([np.inf, -np.inf], 0).values
    cv_base, _ = run_cv(X_base, y, groups)
    print(f"  CV: {cv_base:.4f}")

    print(f"\n--- + Sequence Stats ({len(features_new)}개 피처) ---")
    X_new = last_passes[features_new].fillna(0).replace([np.inf, -np.inf], 0).values
    cv_new, models = run_cv(X_new, y, groups)
    print(f"  CV: {cv_new:.4f}")

    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)
    print(f"  기존:           CV {cv_base:.4f}")
    print(f"  + Seq Stats:    CV {cv_new:.4f} ({cv_base - cv_new:+.4f})")

    # Test 예측
    print("\n[3] Test 예측...")
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
    test_all = add_sequence_statistics(test_all)

    test_last = test_all.groupby('game_episode').last().reset_index()

    best_features = features_new if cv_new < cv_base else features_base
    best_cv = min(cv_new, cv_base)

    X_test = test_last[best_features].fillna(0).replace([np.inf, -np.inf], 0).values

    pred_x = np.zeros(len(X_test))
    pred_y = np.zeros(len(X_test))

    for mx, my in models:
        pred_x += mx.predict(X_test) / len(models)
        pred_y += my.predict(X_test) / len(models)

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })

    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_seqstats_cv{best_cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  저장: {output_path}")

    # Feature importance
    print("\n[피처 중요도 Top 15]")
    importance = (models[0][0].get_feature_importance() + models[0][1].get_feature_importance()) / 2
    feat_imp = pd.DataFrame({'feature': best_features, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print(feat_imp.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"Best CV: {best_cv:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
