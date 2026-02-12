"""
exp_005: Position Continuous 피처 제출
- CV: 14.5523 (이전 14.64보다 0.09점 개선)

작성일: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../../../../data')
SUBMISSION_DIR = Path('../../../../submissions')


def create_all_features(df, ema_span=2):
    """모든 피처 생성"""
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

    # EMA 피처
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

    # Position Continuous 피처
    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    return df


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


def main():
    print("=" * 70)
    print("exp_005: Position Continuous 제출")
    print("- CV: 14.5523")
    print("=" * 70)

    BASELINE = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]
    EMA_FEATURES = [
        'ema_start_x', 'ema_start_y', 'ema_goal_distance',
        'ema_distance', 'ema_success_rate', 'ema_possession'
    ]
    POSITION_CONTINUOUS = [
        'dist_to_goal_line', 'dist_to_sideline', 'dist_to_center_y', 'position_zone'
    ]
    FEATURES = BASELINE + EMA_FEATURES + POSITION_CONTINUOUS

    # Train
    print("\n[1] Train...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_all_features(train_df)
    train_last = train_df.groupby('game_episode').last().reset_index()
    X_train = train_last[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_last[['end_x', 'end_y']].values
    print(f"  Train: {len(train_last)}")

    # Test
    print("\n[2] Test...")
    test_df = load_test_data()
    test_df = create_all_features(test_df)
    test_last = test_df.groupby('game_episode').last().reset_index()
    X_test = test_last[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    print(f"  Test: {len(test_last)}")

    # Train
    print("\n[3] 모델 학습...")
    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 100
    }
    model_x = CatBoostRegressor(**params)
    model_y = CatBoostRegressor(**params)
    model_x.fit(X_train, y_train[:, 0])
    model_y.fit(X_train, y_train[:, 1])

    # Predict
    print("\n[4] 예측...")
    pred_x = np.clip(model_x.predict(X_test), 0, 105)
    pred_y = np.clip(model_y.predict(X_test), 0, 68)

    # Save
    print("\n[5] 저장...")
    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    }).sort_values('game_episode').reset_index(drop=True)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SUBMISSION_DIR / "submission_position_cv14.55.csv"
    submission.to_csv(filepath, index=False)
    print(f"  저장: {filepath}")
    print(f"\n[예측 분포]")
    print(f"  end_x: mean={pred_x.mean():.2f}, std={pred_x.std():.2f}")
    print(f"  end_y: mean={pred_y.mean():.2f}, std={pred_y.std():.2f}")
    print("\n완료!")


if __name__ == '__main__':
    main()
