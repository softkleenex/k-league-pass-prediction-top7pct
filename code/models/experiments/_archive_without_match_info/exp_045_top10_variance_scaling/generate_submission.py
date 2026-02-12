"""
exp_045: Top 10 + 분산 스케일링 제출 파일 생성

Best: alpha=0.25, CV 15.1802

작성일: 2025-12-23
"""

import pandas as pd
import numpy as np
import pickle
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


def load_test_data(test_csv_path, data_dir):
    """테스트 데이터 로드 (path 컬럼에서 개별 CSV 읽기)"""
    test_df = pd.read_csv(test_csv_path)
    test_episodes = []

    for _, row in test_df.iterrows():
        # path는 ./test/xxxxx/xxxxx_x.csv 형식
        ep_path = data_dir / row['path'].replace('./', '')
        ep_df = pd.read_csv(ep_path)
        ep_df['game_episode'] = row['game_episode']
        test_episodes.append(ep_df)

    return pd.concat(test_episodes, ignore_index=True)


def scale_predictions(pred, y_mean, y_std, pred_mean, pred_std, alpha=0.25):
    """예측값 분산 스케일링"""
    standardized = (pred - pred_mean) / (pred_std + 1e-8)
    target_std = pred_std + alpha * (y_std - pred_std)
    scaled = standardized * target_std + y_mean
    return scaled


def main():
    from pathlib import Path

    print("=" * 60)
    print("Top 10 + 분산 스케일링 (alpha=0.25) 제출 파일 생성")
    print("=" * 60)

    # Paths
    data_dir = Path('../../../../data')

    # Load data
    print("\n[1] 데이터 로딩...")
    train_df = pd.read_csv(data_dir / 'train.csv')
    test_df = load_test_data(data_dir / 'test.csv', data_dir)

    train_df = create_phase1a_features(train_df)
    test_df = create_phase1a_features(test_df)

    train_last = train_df.groupby('game_episode').last().reset_index()
    test_last = test_df.groupby('game_episode').last().reset_index()

    print(f"  Train episodes: {len(train_last)}")
    print(f"  Test episodes: {len(test_last)}")

    # Top 10 features
    TOP_10_FEATURES = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    X_train = train_last[TOP_10_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_last[['end_x', 'end_y']].values
    X_test = test_last[TOP_10_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values

    # Target statistics
    y_mean_x, y_std_x = y_train[:, 0].mean(), y_train[:, 0].std()
    y_mean_y, y_std_y = y_train[:, 1].mean(), y_train[:, 1].std()

    # Train models
    print("\n[2] 모델 학습...")
    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

    model_x = CatBoostRegressor(**params)
    model_y = CatBoostRegressor(**params)

    model_x.fit(X_train, y_train[:, 0])
    model_y.fit(X_train, y_train[:, 1])

    # Predict
    print("\n[3] 예측 및 분산 스케일링...")
    pred_x = model_x.predict(X_test)
    pred_y = model_y.predict(X_test)

    # Apply variance scaling with alpha=0.25
    alpha = 0.25
    pred_mean_x, pred_std_x = pred_x.mean(), pred_x.std()
    pred_mean_y, pred_std_y = pred_y.mean(), pred_y.std()

    pred_x_scaled = scale_predictions(pred_x, y_mean_x, y_std_x,
                                       pred_mean_x, pred_std_x, alpha)
    pred_y_scaled = scale_predictions(pred_y, y_mean_y, y_std_y,
                                       pred_mean_y, pred_std_y, alpha)

    # Clip to valid range
    pred_x_scaled = np.clip(pred_x_scaled, 0, 105)
    pred_y_scaled = np.clip(pred_y_scaled, 0, 68)

    print(f"  Prediction stats (before scaling):")
    print(f"    X: mean={pred_x.mean():.2f}, std={pred_x.std():.2f}")
    print(f"    Y: mean={pred_y.mean():.2f}, std={pred_y.std():.2f}")
    print(f"  Prediction stats (after scaling):")
    print(f"    X: mean={pred_x_scaled.mean():.2f}, std={pred_x_scaled.std():.2f}")
    print(f"    Y: mean={pred_y_scaled.mean():.2f}, std={pred_y_scaled.std():.2f}")

    # Create submission
    print("\n[4] 제출 파일 생성...")
    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x_scaled,
        'end_y': pred_y_scaled
    })

    filename = 'submission_top10_varscale_cv15.18.csv'
    submission.to_csv(filename, index=False)
    print(f"  저장: {filename}")

    # Also save models
    with open('model_x_top10_varscale.pkl', 'wb') as f:
        pickle.dump(model_x, f)
    with open('model_y_top10_varscale.pkl', 'wb') as f:
        pickle.dump(model_y, f)
    print("  모델 저장 완료")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
