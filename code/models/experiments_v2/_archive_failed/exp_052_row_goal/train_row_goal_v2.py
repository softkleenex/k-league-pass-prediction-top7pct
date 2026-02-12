"""
exp_052 v2: Row-level + Goal-oriented Features (데이터 누수 수정)

v1 문제점:
- dist_to_goal_after, goal_angle_after 등은 end_x, end_y에서 계산
- 마지막 패스의 end_x, end_y = 타겟 변수 = 데이터 누수!

수정:
- 타겟(end_x, end_y)을 사용하는 피처 제거
- start_x, start_y 기반 피처만 사용
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

FIELD_LENGTH = 105
FIELD_WIDTH = 68
GOAL_X = 105
GOAL_Y = 34


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
    # Differencing
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['diff_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['diff_goal_dist'] = df.groupby('game_episode')['goal_distance'].diff().fillna(0)

    # Rolling Statistics
    df['rolling_std_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['rolling_std_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['rolling_std_dist'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    # Velocity/Acceleration
    df['velocity'] = df['move_distance']
    df['acceleration'] = df.groupby('game_episode')['velocity'].diff().fillna(0)

    # Relative Position
    df['rel_x_from_ema'] = df['start_x'] - df['ema_start_x']
    df['rel_y_from_ema'] = df['start_y'] - df['ema_start_y']

    df['episode_start_x'] = df.groupby('game_episode')['start_x'].transform('first')
    df['episode_start_y'] = df.groupby('game_episode')['start_y'].transform('first')
    df['progress_from_start_x'] = df['start_x'] - df['episode_start_x']
    df['progress_from_start_y'] = df['start_y'] - df['episode_start_y']

    # Feature Interactions
    df['x_times_direction'] = df['start_x'] * df['direction']
    df['goal_dist_times_velocity'] = df['goal_distance'] * df['velocity']
    df['success_times_possession'] = df['ema_success_rate'] * df['ema_possession']

    # Momentum
    df['is_forward'] = (df['diff_x'] > 0).astype(int)
    df['forward_streak'] = df.groupby('game_episode')['is_forward'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df['avg_forward_x'] = df.groupby('game_episode')['diff_x'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).fillna(0)

    return df


def add_goal_oriented_features_safe(df):
    """Goal-oriented 피처 (데이터 누수 없는 버전)

    오직 start_x, start_y 기반 피처만 사용!
    end_x, end_y는 마지막 패스에서 타겟이므로 절대 사용 금지!
    """

    # 1. 에피소드 시작 대비 골대 진전도 (start_x 기준)
    df['start_goal_dist'] = df.groupby('game_episode')['goal_distance'].transform('first')
    df['progress_to_goal'] = df['start_goal_dist'] - df['goal_distance']
    df['progress_to_goal_pct'] = df['progress_to_goal'] / (df['start_goal_dist'] + 1e-6)

    # 2. 이전 패스의 골대 방향 이동량 (prev_dx, prev_dy 기준 - 안전)
    # 골대 방향 단위 벡터 (이전 위치 기준)
    goal_dir_x = GOAL_X - df['prev_start_x']
    goal_dir_y = GOAL_Y - df['prev_start_y']
    goal_dir_norm = np.sqrt(goal_dir_x**2 + goal_dir_y**2) + 1e-6
    goal_dir_x_unit = goal_dir_x / goal_dir_norm
    goal_dir_y_unit = goal_dir_y / goal_dir_norm

    # 이전 패스가 골대 방향으로 얼마나 이동했는지
    df['prev_move_toward_goal'] = df['prev_dx'] * goal_dir_x_unit + df['prev_dy'] * goal_dir_y_unit

    # 3. 누적 골대 방향 이동량 (이전 패스들 기준)
    df['cumsum_toward_goal'] = df.groupby('game_episode')['prev_move_toward_goal'].transform('cumsum')

    # 4. EMA 골대 방향 이동
    df['ema_toward_goal'] = df.groupby('game_episode')['prev_move_toward_goal'].transform(
        lambda x: x.ewm(span=2, adjust=False).mean()
    ).fillna(0)

    # 5. 골대 거리 변화 추세 (과거 기준, 안전)
    df['goal_dist_trend'] = df.groupby('game_episode')['goal_distance'].transform(
        lambda x: x.rolling(3, min_periods=1).apply(
            lambda w: np.polyfit(range(len(w)), w, 1)[0] if len(w) > 1 else 0
        )
    ).fillna(0)

    # 6. 박스 관련 피처 (start_x, start_y 기준 - 안전)
    # 페널티 박스: x > 88.5, 13.85 < y < 54.15
    df['in_box'] = ((df['start_x'] > 88.5) & (df['start_y'] > 13.85) & (df['start_y'] < 54.15)).astype(int)
    df['dist_to_box'] = np.maximum(0, 88.5 - df['start_x'])

    # 7. 골대 각도 변화 (start 기준, 안전)
    df['goal_angle_change'] = df.groupby('game_episode')['goal_angle'].diff().fillna(0)

    # 8. 중앙 침투 정도
    df['central_penetration'] = df['progress_from_start_x'] * (1 - np.abs(df['start_y'] - 34) / 34)

    # 9. 슈팅 존 접근 (x > 85, |y - 34| < 20)
    df['in_shooting_zone'] = ((df['start_x'] > 85) & (np.abs(df['start_y'] - 34) < 20)).astype(int)
    df['dist_to_shooting_zone'] = np.sqrt(
        np.maximum(0, 85 - df['start_x'])**2 +
        np.maximum(0, np.abs(df['start_y'] - 34) - 20)**2
    )

    # 10. 평균 진전 속도
    df['action_idx'] = df.groupby('game_episode').cumcount() + 1
    df['avg_progress_rate'] = df['progress_to_goal'] / df['action_idx']

    return df


def run_cv(X, y, groups, feature_names=None):
    """3-Fold CV"""
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

    fold_scores = []
    models_x = []
    models_y = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)

        model_x.fit(X[train_idx], y[train_idx, 0],
                   eval_set=(X[val_idx], y[val_idx, 0]),
                   use_best_model=True)
        model_y.fit(X[train_idx], y[train_idx, 1],
                   eval_set=(X[val_idx], y[val_idx, 1]),
                   use_best_model=True)

        pred_x = model_x.predict(X[val_idx])
        pred_y = model_y.predict(X[val_idx])
        errors = np.sqrt((pred_x - y[val_idx, 0])**2 + (pred_y - y[val_idx, 1])**2)
        fold_scores.append(errors.mean())
        print(f"    Fold {fold}: {fold_scores[-1]:.4f}")

        models_x.append(model_x)
        models_y.append(model_y)

    return np.mean(fold_scores), models_x, models_y


def main():
    print("=" * 70)
    print("exp_052 v2: Row-level + Goal-oriented (데이터 누수 수정)")
    print("핵심: start_x, start_y 기반 피처만 사용 (end_x, end_y 사용 금지)")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드 및 피처 생성...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')

    # 피처 생성
    train_df = create_baseline_features(train_df)
    train_df = add_ema_features(train_df, ema_span=2)
    train_df = add_position_features(train_df)
    train_df = add_advanced_features(train_df)
    train_df = add_goal_oriented_features_safe(train_df)  # 안전한 버전!

    # 마지막 패스만 추출
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # 피처 정의
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

    # 안전한 Goal-oriented 피처
    GOAL_SAFE = [
        'progress_to_goal',           # 에피소드 시작 대비 진전
        'progress_to_goal_pct',       # 진전 비율
        'prev_move_toward_goal',      # 이전 패스 골대 방향 이동
        'cumsum_toward_goal',         # 누적 골대 방향 이동
        'ema_toward_goal',            # EMA 골대 방향 이동
        'goal_dist_trend',            # 골대 거리 변화 추세
        'in_box',                     # 박스 내 여부
        'dist_to_box',                # 박스까지 거리
        'goal_angle_change',          # 골대 각도 변화
        'central_penetration',        # 중앙 침투 정도
        'in_shooting_zone',           # 슈팅 존 여부
        'dist_to_shooting_zone',      # 슈팅 존까지 거리
        'avg_progress_rate',          # 평균 진전 속도
    ]

    # 실험
    print("\n[2] 실험 비교...")

    ALL_BASELINE = BASELINE + EMA + POSITION + ADVANCED
    ALL_NEW = ALL_BASELINE + GOAL_SAFE

    experiments = [
        ("기존 (exp_047 수준)", ALL_BASELINE),
        ("+ Goal-oriented (안전)", ALL_NEW),
    ]

    results = {}
    best_models = None
    best_cv = float('inf')
    best_features = None

    for name, features in experiments:
        print(f"\n--- {name} ({len(features)}개 피처) ---")
        X = last_passes[features].fillna(0).replace([np.inf, -np.inf], 0).values
        cv, models_x, models_y = run_cv(X, y, groups, features)
        results[name] = cv
        print(f"  CV: {cv:.4f}")

        if cv < best_cv:
            best_cv = cv
            best_models = (models_x, models_y)
            best_features = features

    # 결과 비교
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)

    baseline_cv = results["기존 (exp_047 수준)"]
    for name, cv in results.items():
        diff = baseline_cv - cv
        marker = " ***BEST***" if cv == min(results.values()) else ""
        print(f"  {name:30s}: CV {cv:.4f} ({diff:+.4f}){marker}")

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

    # 피처 생성
    test_all = create_baseline_features(test_all)
    test_all = add_ema_features(test_all, ema_span=2)
    test_all = add_position_features(test_all)
    test_all = add_advanced_features(test_all)
    test_all = add_goal_oriented_features_safe(test_all)

    test_last = test_all.groupby('game_episode').last().reset_index()

    # 예측
    X_test = test_last[best_features].fillna(0).replace([np.inf, -np.inf], 0).values

    pred_x_all = np.zeros(len(X_test))
    pred_y_all = np.zeros(len(X_test))

    models_x, models_y = best_models
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

    output_path = SUBMISSION_DIR / f"submission_rowgoal_v2_cv{best_cv:.2f}.csv"
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
        'feature': best_features,
        'importance': avg_importance,
        'imp_x': importance_x,
        'imp_y': importance_y
    })
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print(feat_imp.head(15).to_string(index=False))

    # Goal-oriented 피처 중요도
    print("\n[Goal-oriented 피처 중요도]")
    goal_imp = feat_imp[feat_imp['feature'].isin(GOAL_SAFE)]
    print(goal_imp.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"Best CV: {best_cv:.4f}")
    print(f"제출 파일: {output_path.name}")
    print("=" * 70)

    return best_cv


if __name__ == "__main__":
    main()
