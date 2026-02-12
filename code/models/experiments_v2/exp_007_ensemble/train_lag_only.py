"""
exp_007a: 추가 Lag Features만 테스트 (CatBoost)

작성일: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../../../../data')


def create_all_features(df):
    """exp_006 피처 + 추가 Lag Features"""
    # ========================================
    # Baseline
    # ========================================
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

    # ========================================
    # EMA (span=2)
    # ========================================
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

    # ========================================
    # Position
    # ========================================
    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    # ========================================
    # Differencing
    # ========================================
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['diff_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['diff_goal_dist'] = df.groupby('game_episode')['goal_distance'].diff().fillna(0)

    # ========================================
    # Rolling Stats
    # ========================================
    df['rolling_std_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)
    df['rolling_std_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)
    df['rolling_std_dist'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    # ========================================
    # Velocity/Acceleration
    # ========================================
    df['velocity'] = df['move_distance']
    df['acceleration'] = df.groupby('game_episode')['velocity'].diff().fillna(0)

    # ========================================
    # Relative Position
    # ========================================
    df['rel_x_from_ema'] = df['start_x'] - df['ema_start_x']
    df['rel_y_from_ema'] = df['start_y'] - df['ema_start_y']
    df['episode_start_x'] = df.groupby('game_episode')['start_x'].transform('first')
    df['episode_start_y'] = df.groupby('game_episode')['start_y'].transform('first')
    df['progress_from_start_x'] = df['start_x'] - df['episode_start_x']
    df['progress_from_start_y'] = df['start_y'] - df['episode_start_y']

    # ========================================
    # Interactions
    # ========================================
    df['x_times_direction'] = df['start_x'] * df['direction']
    df['goal_dist_times_velocity'] = df['goal_distance'] * df['velocity']
    df['success_times_possession'] = df['ema_success_rate'] * df['ema_possession']

    # ========================================
    # Momentum
    # ========================================
    df['is_forward'] = (df['diff_x'] > 0).astype(int)
    df['forward_streak'] = df.groupby('game_episode')['is_forward'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df['avg_forward_x'] = df.groupby('game_episode')['diff_x'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).fillna(0)

    # ========================================
    # NEW: 추가 Lag Features
    # ========================================
    # start_x lag
    df['start_x_lag2'] = df.groupby('game_episode')['start_x'].shift(2).fillna(df['start_x'])
    df['start_x_lag3'] = df.groupby('game_episode')['start_x'].shift(3).fillna(df['start_x'])

    # start_y lag
    df['start_y_lag2'] = df.groupby('game_episode')['start_y'].shift(2).fillna(df['start_y'])
    df['start_y_lag3'] = df.groupby('game_episode')['start_y'].shift(3).fillna(df['start_y'])

    # goal_distance lag
    df['goal_dist_lag2'] = df.groupby('game_episode')['goal_distance'].shift(2).fillna(df['goal_distance'])
    df['goal_dist_lag3'] = df.groupby('game_episode')['goal_distance'].shift(3).fillna(df['goal_distance'])

    # Delta from lag
    df['delta_x_from_lag2'] = df['start_x'] - df['start_x_lag2']
    df['delta_x_from_lag3'] = df['start_x'] - df['start_x_lag3']
    df['delta_y_from_lag2'] = df['start_y'] - df['start_y_lag2']
    df['delta_y_from_lag3'] = df['start_y'] - df['start_y_lag3']

    # 2nd order differencing
    df['diff2_x'] = df.groupby('game_episode')['diff_x'].diff().fillna(0)
    df['diff2_y'] = df.groupby('game_episode')['diff_y'].diff().fillna(0)

    return df


def run_cv(X, y, groups):
    gkf = GroupKFold(n_splits=3)
    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

    fold_scores = []
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

    return np.mean(fold_scores)


def main():
    print("=" * 70)
    print("exp_007a: 추가 Lag Features 테스트 (CatBoost)")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드 및 피처 생성...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_all_features(train_df)

    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # 피처 정의 (exp_006 ALL Advanced)
    ALL_ADVANCED = [
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

    LAG_FEATURES = [
        'start_x_lag2', 'start_x_lag3', 'start_y_lag2', 'start_y_lag3',
        'goal_dist_lag2', 'goal_dist_lag3',
        'delta_x_from_lag2', 'delta_x_from_lag3', 'delta_y_from_lag2', 'delta_y_from_lag3',
        'diff2_x', 'diff2_y'
    ]

    # 실험 1: exp_006 기준
    print("\n[실험 1] exp_006 ALL Advanced (기준)")
    X = last_passes[ALL_ADVANCED].fillna(0).replace([np.inf, -np.inf], 0).values
    cv_baseline = run_cv(X, y, groups)
    print(f"  CV: {cv_baseline:.4f}")

    # 실험 2: + Lag Features
    print("\n[실험 2] + Lag Features")
    X = last_passes[ALL_ADVANCED + LAG_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    cv_lag = run_cv(X, y, groups)
    print(f"  CV: {cv_lag:.4f}  ({cv_baseline - cv_lag:+.4f})")

    # 결과
    print("\n" + "=" * 70)
    print("결과")
    print("=" * 70)
    print(f"  exp_006 ALL Advanced: CV {cv_baseline:.4f}")
    print(f"  + Lag Features      : CV {cv_lag:.4f}  ({cv_baseline - cv_lag:+.4f})")

    if cv_lag < cv_baseline:
        print(f"\n  Lag Features 효과 있음!")
    else:
        print(f"\n  Lag Features 효과 없음")


if __name__ == '__main__':
    main()
