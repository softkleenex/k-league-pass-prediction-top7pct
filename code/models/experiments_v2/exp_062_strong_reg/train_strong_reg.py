"""
exp_062: Strong Regularization

핵심 가설: 과소적합이 오히려 좋음 (Gap 음수)
- l2_leaf_reg: 3.0 → 10.0
- depth: 8 → 6
- iterations: 1000 → 500
- learning_rate: 0.05 → 0.03
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


def create_all_features(df):
    """exp_047과 동일한 피처"""
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

    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

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


def run_cv(X, y, groups, params):
    """CV with given params"""
    gkf = GroupKFold(n_splits=3)

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
    print("exp_062: Strong Regularization")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_all_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    features = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct',
        'ema_start_x', 'ema_start_y', 'ema_goal_distance',
        'ema_distance', 'ema_success_rate', 'ema_possession',
        'dist_to_goal_line', 'dist_to_sideline', 'dist_to_center_y', 'position_zone',
        'diff_x', 'diff_y', 'diff_goal_dist',
        'rolling_std_x', 'rolling_std_y', 'rolling_std_dist',
        'velocity', 'acceleration',
        'rel_x_from_ema', 'rel_y_from_ema', 'progress_from_start_x', 'progress_from_start_y',
        'x_times_direction', 'goal_dist_times_velocity', 'success_times_possession',
        'forward_streak', 'avg_forward_x'
    ]

    X = last_passes[features].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # 파라미터 비교
    params_base = {
        'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
        'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 50
    }

    params_strong = {
        'iterations': 500, 'depth': 6, 'learning_rate': 0.03,
        'l2_leaf_reg': 10.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 50
    }

    print("\n[2] 기존 파라미터...")
    cv_base, _ = run_cv(X, y, groups, params_base)
    print(f"  CV (기존): {cv_base:.4f}")

    print("\n[3] 강한 Regularization...")
    cv_strong, models = run_cv(X, y, groups, params_strong)
    print(f"  CV (강한 Reg): {cv_strong:.4f}")

    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)
    print(f"  기존:      CV {cv_base:.4f}")
    print(f"  강한 Reg:  CV {cv_strong:.4f} ({cv_base - cv_strong:+.4f})")

    # Test 예측
    print("\n[4] Test 예측...")
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
    test_all = create_all_features(test_all)
    test_last = test_all.groupby('game_episode').last().reset_index()

    X_test = test_last[features].fillna(0).replace([np.inf, -np.inf], 0).values

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

    output_path = SUBMISSION_DIR / f"submission_strongreg_cv{cv_strong:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  저장: {output_path}")

    print("\n" + "=" * 70)
    print(f"Strong Reg CV: {cv_strong:.4f}")
    print("목표: CV 높아도 Gap -0.3+ → Public 13.9 이하")
    print("=" * 70)


if __name__ == "__main__":
    main()
