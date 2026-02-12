"""
exp_077: Delta Prediction + Feature Engineering
- start_x, start_y를 feature로 추가
- Delta에 최적화된 feature 탐색
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


def create_features(df):
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    ema_span = 2
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)

    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    # Delta에 유용할 수 있는 추가 feature들
    df['start_x_normalized'] = df['start_x'] / 105.0
    df['start_y_normalized'] = df['start_y'] / 68.0
    df['dist_from_center'] = np.sqrt((df['start_x'] - 52.5)**2 + (df['start_y'] - 34)**2)
    df['in_penalty_area'] = ((df['start_x'] > 88.5) & (df['start_y'] > 13.84) & (df['start_y'] < 54.16)).astype(int)

    return df


def run_cv_delta(X, y_delta, y_abs, start_xy, groups, name):
    """Delta prediction CV"""
    params = {
        'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
        'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 50, 'loss_function': 'MAE'
    }

    gkf = GroupKFold(n_splits=3)
    oof_delta = np.zeros((len(X), 2))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_delta, groups), 1):
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)

        model_dx.fit(X[train_idx], y_delta[train_idx, 0],
                    eval_set=(X[val_idx], y_delta[val_idx, 0]), use_best_model=True)
        model_dy.fit(X[train_idx], y_delta[train_idx, 1],
                    eval_set=(X[val_idx], y_delta[val_idx, 1]), use_best_model=True)

        oof_delta[val_idx, 0] = model_dx.predict(X[val_idx])
        oof_delta[val_idx, 1] = model_dy.predict(X[val_idx])

    # Convert to absolute
    pred_abs = np.zeros((len(X), 2))
    pred_abs[:, 0] = start_xy[:, 0] + oof_delta[:, 0]
    pred_abs[:, 1] = start_xy[:, 1] + oof_delta[:, 1]

    cv = np.sqrt((pred_abs[:, 0] - y_abs[:, 0])**2 + (pred_abs[:, 1] - y_abs[:, 1])**2).mean()
    print(f"  {name}: CV {cv:.4f}")
    return cv


def main():
    print("=" * 70)
    print("exp_077: Delta + Feature Engineering")
    print("=" * 70)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    # Feature sets to test
    TOP_15 = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession',
        'zone_x', 'result_encoded', 'diff_x', 'velocity'
    ]

    # Targets
    y_delta = last_passes[['dx', 'dy']].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    print(f"\nData: {len(last_passes)} samples")
    print("\n[1] Feature Set 비교...")

    results = {}

    # Test 1: Baseline (TOP_15)
    X1 = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    cv1 = run_cv_delta(X1, y_delta, y_abs, start_xy, groups, "TOP_15 (baseline)")
    results['TOP_15'] = cv1

    # Test 2: TOP_15 + start_x, start_y
    FEAT_2 = TOP_15 + ['start_x', 'start_y']
    X2 = last_passes[FEAT_2].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    cv2 = run_cv_delta(X2, y_delta, y_abs, start_xy, groups, "TOP_15 + start_x/y")
    results['+ start_x/y'] = cv2

    # Test 3: TOP_15 + normalized start
    FEAT_3 = TOP_15 + ['start_x_normalized', 'start_y_normalized']
    X3 = last_passes[FEAT_3].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    cv3 = run_cv_delta(X3, y_delta, y_abs, start_xy, groups, "TOP_15 + start_norm")
    results['+ start_norm'] = cv3

    # Test 4: TOP_15 + start_x, start_y + dist_from_center
    FEAT_4 = TOP_15 + ['start_x', 'start_y', 'dist_from_center']
    X4 = last_passes[FEAT_4].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    cv4 = run_cv_delta(X4, y_delta, y_abs, start_xy, groups, "TOP_15 + start + dist_center")
    results['+ start + dist_center'] = cv4

    # Test 5: TOP_15 + start_x, start_y + in_penalty_area
    FEAT_5 = TOP_15 + ['start_x', 'start_y', 'in_penalty_area']
    X5 = last_passes[FEAT_5].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    cv5 = run_cv_delta(X5, y_delta, y_abs, start_xy, groups, "TOP_15 + start + penalty")
    results['+ start + penalty'] = cv5

    # Test 6: Minimal - start + goal features only
    FEAT_6 = ['start_x', 'start_y', 'goal_distance', 'goal_angle', 'prev_dx', 'prev_dy']
    X6 = last_passes[FEAT_6].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    cv6 = run_cv_delta(X6, y_delta, y_abs, start_xy, groups, "Minimal (start + goal)")
    results['Minimal'] = cv6

    # Test 7: ALL features
    FEAT_7 = TOP_15 + ['start_x', 'start_y', 'start_x_normalized', 'start_y_normalized',
                       'dist_from_center', 'in_penalty_area']
    X7 = last_passes[FEAT_7].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    cv7 = run_cv_delta(X7, y_delta, y_abs, start_xy, groups, "ALL features")
    results['ALL'] = cv7

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)

    baseline = results['TOP_15']
    for name, cv in sorted(results.items(), key=lambda x: x[1]):
        diff = cv - baseline
        marker = " ★" if cv == min(results.values()) else ""
        print(f"  {name:25s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    best_name = min(results, key=results.get)
    best_cv = results[best_name]
    print(f"\n  Best: {best_name} (CV {best_cv:.4f})")
    print(f"  vs exp_076 (13.72): {best_cv - 13.72:+.4f}")
    print("=" * 70)

    # If improvement found, create submission
    if best_cv < 13.72:
        print(f"\n[2] Best model로 submission 생성...")
        # Determine best feature set
        if best_name == '+ start_x/y':
            best_feat = FEAT_2
        elif best_name == '+ start_norm':
            best_feat = FEAT_3
        elif best_name == '+ start + dist_center':
            best_feat = FEAT_4
        elif best_name == '+ start + penalty':
            best_feat = FEAT_5
        elif best_name == 'Minimal':
            best_feat = FEAT_6
        elif best_name == 'ALL':
            best_feat = FEAT_7
        else:
            best_feat = TOP_15

        create_submission(last_passes, best_feat, best_cv)


def create_submission(last_passes, feature_cols, cv):
    """Create submission with best features"""
    params = {
        'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
        'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 50, 'loss_function': 'MAE'
    }

    X = last_passes[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = last_passes[['dx', 'dy']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    gkf = GroupKFold(n_splits=3)
    models_dx = []
    models_dy = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_delta, groups), 1):
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)

        model_dx.fit(X[train_idx], y_delta[train_idx, 0],
                    eval_set=(X[val_idx], y_delta[val_idx, 0]), use_best_model=True)
        model_dy.fit(X[train_idx], y_delta[train_idx, 1],
                    eval_set=(X[val_idx], y_delta[val_idx, 1]), use_best_model=True)

        models_dx.append(model_dx)
        models_dy.append(model_dy)

    # Test prediction
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

    # Apply same feature engineering
    test_all['zone_x'] = (test_all['start_x'] / (105/6)).astype(int).clip(0, 5)
    test_all['zone_y'] = (test_all['start_y'] / (68/6)).astype(int).clip(0, 5)
    test_all['goal_distance'] = np.sqrt((105 - test_all['start_x'])**2 + (34 - test_all['start_y'])**2)
    test_all['goal_angle'] = np.degrees(np.arctan2(34 - test_all['start_y'], 105 - test_all['start_x']))
    test_all['prev_dx'] = test_all.groupby('game_episode')['dx'].shift(1).fillna(0)
    test_all['prev_dy'] = test_all.groupby('game_episode')['dy'].shift(1).fillna(0)
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    test_all['result_encoded'] = test_all['result_name'].map(result_map).fillna(2).astype(int)

    ema_span = 2
    test_all['ema_start_x'] = test_all.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(test_all['start_x'])
    test_all['ema_start_y'] = test_all.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(test_all['start_y'])

    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    test_all['is_successful'] = test_all['result_name'].map(result_map2).fillna(0)
    test_all['ema_success_rate'] = test_all.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    test_all['final_team_id'] = test_all.groupby('game_episode')['team_id'].transform('last')
    test_all['is_final_team'] = (test_all['team_id'] == test_all['final_team_id']).astype(int)
    test_all['ema_possession'] = test_all.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    test_all['dist_to_goal_line'] = 105 - test_all['start_x']
    test_all['dist_to_center_y'] = np.abs(test_all['start_y'] - 34)
    test_all['diff_x'] = test_all.groupby('game_episode')['start_x'].diff().fillna(0)
    test_all['prev_start_x'] = test_all.groupby('game_episode')['start_x'].shift(1).fillna(test_all['start_x'])
    test_all['prev_start_y'] = test_all.groupby('game_episode')['start_y'].shift(1).fillna(test_all['start_y'])
    test_all['velocity'] = np.sqrt(
        (test_all['start_x'] - test_all['prev_start_x'])**2 +
        (test_all['start_y'] - test_all['prev_start_y'])**2
    )
    test_all['start_x_normalized'] = test_all['start_x'] / 105.0
    test_all['start_y_normalized'] = test_all['start_y'] / 68.0
    test_all['dist_from_center'] = np.sqrt((test_all['start_x'] - 52.5)**2 + (test_all['start_y'] - 34)**2)
    test_all['in_penalty_area'] = ((test_all['start_x'] > 88.5) & (test_all['start_y'] > 13.84) & (test_all['start_y'] < 54.16)).astype(int)

    test_last = test_all.groupby('game_episode').last().reset_index()

    X_test = test_last[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    test_start_x = test_last['start_x'].values
    test_start_y = test_last['start_y'].values

    pred_dx = np.zeros(len(X_test))
    pred_dy = np.zeros(len(X_test))

    for model_dx, model_dy in zip(models_dx, models_dy):
        pred_dx += model_dx.predict(X_test) / len(models_dx)
        pred_dy += model_dy.predict(X_test) / len(models_dy)

    pred_x = test_start_x + pred_dx
    pred_y = np.clip(test_start_y + pred_dy, 0, 68)

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })
    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_delta_feat_cv{cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
