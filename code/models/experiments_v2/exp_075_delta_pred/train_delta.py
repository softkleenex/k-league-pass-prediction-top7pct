"""
exp_075: Delta Prediction (Best Model)
- exp_067 MAE 기반 + Delta (dx, dy) 예측
- 절대좌표 대신 상대이동 예측
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

    return df


def main():
    print("=" * 70)
    print("exp_075: Delta Prediction")
    print("=" * 70)

    # 데이터 로드
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    TOP_12 = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession', 'velocity'
    ]

    X = last_passes[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    # Target: dx, dy (delta)
    y_delta = last_passes[['dx', 'dy']].values.astype(np.float32)
    # For evaluation: end_x, end_y
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    # start positions
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)

    groups = last_passes['game_id'].values

    print(f"\nData shape: X={X.shape}")
    print(f"Target (delta): dx range [{y_delta[:,0].min():.1f}, {y_delta[:,0].max():.1f}], dy range [{y_delta[:,1].min():.1f}, {y_delta[:,1].max():.1f}]")

    # Parameters (exp_067 동일)
    cat_params = {
        'iterations': 4000, 'depth': 8, 'learning_rate': 0.01,
        'l2_leaf_reg': 7.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 100, 'loss_function': 'MAE'
    }

    gkf = GroupKFold(n_splits=5)

    print("\n[1] 5-Fold CV (Delta Prediction)...")

    oof_delta = np.zeros((len(X), 2))
    models_dx = []
    models_dy = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_delta, groups), 1):
        model_dx = CatBoostRegressor(**cat_params)
        model_dy = CatBoostRegressor(**cat_params)

        model_dx.fit(X[train_idx], y_delta[train_idx, 0],
                    eval_set=(X[val_idx], y_delta[val_idx, 0]), use_best_model=True)
        model_dy.fit(X[train_idx], y_delta[train_idx, 1],
                    eval_set=(X[val_idx], y_delta[val_idx, 1]), use_best_model=True)

        oof_delta[val_idx, 0] = model_dx.predict(X[val_idx])
        oof_delta[val_idx, 1] = model_dy.predict(X[val_idx])

        # Convert to absolute coordinates for evaluation
        pred_abs_x = start_xy[val_idx, 0] + oof_delta[val_idx, 0]
        pred_abs_y = start_xy[val_idx, 1] + oof_delta[val_idx, 1]

        fold_err = np.sqrt(
            (pred_abs_x - y_abs[val_idx, 0])**2 +
            (pred_abs_y - y_abs[val_idx, 1])**2
        ).mean()
        print(f"  Fold {fold}: {fold_err:.4f}")

        models_dx.append(model_dx)
        models_dy.append(model_dy)

    # Overall CV
    oof_abs = np.zeros((len(X), 2))
    oof_abs[:, 0] = start_xy[:, 0] + oof_delta[:, 0]
    oof_abs[:, 1] = start_xy[:, 1] + oof_delta[:, 1]

    cv = np.sqrt((oof_abs[:, 0] - y_abs[:, 0])**2 + (oof_abs[:, 1] - y_abs[:, 1])**2).mean()
    print(f"\n  Delta CV: {cv:.4f}")
    print(f"  vs Baseline (13.66): {cv - 13.66:+.4f}")

    # Test prediction
    print("\n[2] Test Prediction...")
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
    test_all = create_features(test_all)
    test_last = test_all.groupby('game_episode').last().reset_index()

    X_test = test_last[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    test_start_x = test_last['start_x'].values
    test_start_y = test_last['start_y'].values

    # Predict delta
    pred_dx = np.zeros(len(X_test))
    pred_dy = np.zeros(len(X_test))

    for model_dx, model_dy in zip(models_dx, models_dy):
        pred_dx += model_dx.predict(X_test) / len(models_dx)
        pred_dy += model_dy.predict(X_test) / len(models_dy)

    # Convert to absolute
    pred_x = test_start_x + pred_dx
    pred_y = test_start_y + pred_dy

    # Clip y
    pred_y = np.clip(pred_y, 0, 68)

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })

    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_delta_cv{cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    print("\n" + "=" * 70)
    print(f"Delta Prediction CV: {cv:.4f}")
    print(f"vs Baseline MAE (13.66): {cv - 13.66:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
