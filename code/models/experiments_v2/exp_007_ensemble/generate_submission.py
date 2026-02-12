"""
exp_007: Cat-heavy Ensemble Submission Generator
CV 14.1746 (+0.0307 from CatBoost only)
Weights: CatBoost 0.5, LightGBM 0.25, XGBoost 0.25

작성일: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../../../../data')
SUBMISSION_DIR = Path('../../../../submissions')
OUTPUT_DIR = Path('.')


def load_test_data():
    """테스트 데이터 로드"""
    test_meta = pd.read_csv(DATA_DIR / 'test.csv')
    all_episodes = []
    for _, row in test_meta.iterrows():
        episode_path = DATA_DIR / row['path']
        episode_df = pd.read_csv(episode_path)
        episode_df['game_episode'] = row['game_episode']
        episode_df['game_id'] = row['game_id']
        all_episodes.append(episode_df)
    return pd.concat(all_episodes, ignore_index=True)


def create_all_features(df):
    """exp_006 피처 (ALL Advanced)"""
    # Baseline
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

    # EMA (span=2)
    ema_span = 2
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['move_distance'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)

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

    # Position
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    # Differencing
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['diff_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['diff_goal_dist'] = df.groupby('game_episode')['goal_distance'].diff().fillna(0)

    # Rolling Stats
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

    # Interactions
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


def main():
    print("=" * 70)
    print("exp_007: Cat-heavy Ensemble Submission")
    print("Weights: Cat=0.5, LGB=0.25, XGB=0.25")
    print("=" * 70)

    # 피처 목록
    FEATURES = [
        # Baseline
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct',
        # EMA
        'ema_start_x', 'ema_start_y', 'ema_goal_distance',
        'ema_distance', 'ema_success_rate', 'ema_possession',
        # Position
        'dist_to_goal_line', 'dist_to_sideline', 'dist_to_center_y', 'position_zone',
        # Differencing
        'diff_x', 'diff_y', 'diff_goal_dist',
        # Rolling Stats
        'rolling_std_x', 'rolling_std_y', 'rolling_std_dist',
        # Velocity/Accel
        'velocity', 'acceleration',
        # Relative Position
        'rel_x_from_ema', 'rel_y_from_ema', 'progress_from_start_x', 'progress_from_start_y',
        # Interactions
        'x_times_direction', 'goal_dist_times_velocity', 'success_times_possession',
        # Momentum
        'forward_streak', 'avg_forward_x'
    ]

    # 데이터 로드
    print("\n[1] Train 데이터 로드...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_all_features(train_df)
    train_last = train_df.groupby('game_episode').last().reset_index()
    X_train = train_last[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_last[['end_x', 'end_y']].values
    print(f"  Train: {len(X_train)}")

    print("\n[1.5] Test 데이터 로드...")
    test_df = load_test_data()
    test_df = create_all_features(test_df)
    test_last = test_df.groupby('game_episode').last().reset_index()
    X_test = test_last[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    print(f"  Test: {len(X_test)}")

    # 모델 파라미터 (1000 iterations for final)
    n_iter = 1000
    cat_params = {'iterations': n_iter, 'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 100}
    lgb_params = {'n_estimators': n_iter, 'max_depth': 8, 'learning_rate': 0.05, 'reg_lambda': 3.0, 'random_state': 42, 'verbose': -1, 'n_jobs': -1}
    xgb_params = {'n_estimators': n_iter, 'max_depth': 8, 'learning_rate': 0.05, 'reg_lambda': 3.0, 'random_state': 42, 'verbosity': 0, 'n_jobs': -1}

    # 가중치
    w_cat, w_lgb, w_xgb = 0.5, 0.25, 0.25

    # 학습
    print("\n[2] CatBoost 학습...")
    cat_x = CatBoostRegressor(**cat_params)
    cat_y = CatBoostRegressor(**cat_params)
    cat_x.fit(X_train, y_train[:, 0])
    cat_y.fit(X_train, y_train[:, 1])

    print("\n[3] LightGBM 학습...")
    lgb_x = LGBMRegressor(**lgb_params)
    lgb_y = LGBMRegressor(**lgb_params)
    lgb_x.fit(X_train, y_train[:, 0])
    lgb_y.fit(X_train, y_train[:, 1])

    print("\n[4] XGBoost 학습...")
    xgb_x = XGBRegressor(**xgb_params)
    xgb_y = XGBRegressor(**xgb_params)
    xgb_x.fit(X_train, y_train[:, 0])
    xgb_y.fit(X_train, y_train[:, 1])

    # 예측
    print("\n[5] 예측...")
    pred_cat_x = cat_x.predict(X_test)
    pred_cat_y = cat_y.predict(X_test)
    pred_lgb_x = lgb_x.predict(X_test)
    pred_lgb_y = lgb_y.predict(X_test)
    pred_xgb_x = xgb_x.predict(X_test)
    pred_xgb_y = xgb_y.predict(X_test)

    # 앙상블 + clip
    pred_x = np.clip(w_cat * pred_cat_x + w_lgb * pred_lgb_x + w_xgb * pred_xgb_x, 0, 105)
    pred_y = np.clip(w_cat * pred_cat_y + w_lgb * pred_lgb_y + w_xgb * pred_xgb_y, 0, 68)

    # Submission 생성
    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    }).sort_values('game_episode').reset_index(drop=True)

    # 제출 파일 저장
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_file = SUBMISSION_DIR / 'submission_ensemble_cv14.17.csv'
    submission.to_csv(output_file, index=False)
    print(f"\n[6] 제출 파일 저장: {output_file}")
    print(f"    Shape: {submission.shape}")
    print(f"\n[예측 분포]")
    print(f"  end_x: mean={pred_x.mean():.2f}, std={pred_x.std():.2f}")
    print(f"  end_y: mean={pred_y.mean():.2f}, std={pred_y.std():.2f}")
    print(submission.head())

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)


if __name__ == '__main__':
    main()
