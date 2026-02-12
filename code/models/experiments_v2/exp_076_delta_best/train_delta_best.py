"""
exp_076: Delta Prediction with exp_067 settings
- TOP_15 features + 3-fold + exp_067 params
- Delta (dx, dy) prediction
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
    print("exp_076: Delta Prediction (exp_067 settings)")
    print("=" * 70)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    # exp_067's TOP_15
    TOP_15 = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession',
        'zone_x', 'result_encoded', 'diff_x', 'velocity'
    ]

    X = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    # Delta targets
    y_delta = last_passes[['dx', 'dy']].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    print(f"\nData: X={X.shape}, TOP_15 features")

    # exp_067 params
    params = {
        'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
        'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 50, 'loss_function': 'MAE'
    }

    # 3-Fold CV (like exp_067)
    gkf = GroupKFold(n_splits=3)

    print("\n[1] 3-Fold CV...")
    print("\n--- Absolute Prediction (baseline) ---")

    oof_abs = np.zeros((len(X), 2))
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_abs, groups), 1):
        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)

        model_x.fit(X[train_idx], y_abs[train_idx, 0],
                   eval_set=(X[val_idx], y_abs[val_idx, 0]), use_best_model=True)
        model_y.fit(X[train_idx], y_abs[train_idx, 1],
                   eval_set=(X[val_idx], y_abs[val_idx, 1]), use_best_model=True)

        oof_abs[val_idx, 0] = model_x.predict(X[val_idx])
        oof_abs[val_idx, 1] = model_y.predict(X[val_idx])

        fold_err = np.sqrt((oof_abs[val_idx, 0] - y_abs[val_idx, 0])**2 +
                          (oof_abs[val_idx, 1] - y_abs[val_idx, 1])**2).mean()
        print(f"  Fold {fold}: {fold_err:.4f}")

    cv_abs = np.sqrt((oof_abs[:, 0] - y_abs[:, 0])**2 + (oof_abs[:, 1] - y_abs[:, 1])**2).mean()
    print(f"  Absolute CV: {cv_abs:.4f}")

    print("\n--- Delta Prediction ---")

    oof_delta = np.zeros((len(X), 2))
    models_dx = []
    models_dy = []

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
        pred_x = start_xy[val_idx, 0] + oof_delta[val_idx, 0]
        pred_y = start_xy[val_idx, 1] + oof_delta[val_idx, 1]

        fold_err = np.sqrt((pred_x - y_abs[val_idx, 0])**2 +
                          (pred_y - y_abs[val_idx, 1])**2).mean()
        print(f"  Fold {fold}: {fold_err:.4f}")

        models_dx.append(model_dx)
        models_dy.append(model_dy)

    # Overall CV
    pred_abs = np.zeros((len(X), 2))
    pred_abs[:, 0] = start_xy[:, 0] + oof_delta[:, 0]
    pred_abs[:, 1] = start_xy[:, 1] + oof_delta[:, 1]

    cv_delta = np.sqrt((pred_abs[:, 0] - y_abs[:, 0])**2 + (pred_abs[:, 1] - y_abs[:, 1])**2).mean()
    print(f"  Delta CV: {cv_delta:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Results:")
    print(f"  Absolute: CV {cv_abs:.4f}")
    print(f"  Delta:    CV {cv_delta:.4f} ({cv_delta - cv_abs:+.4f})")
    print(f"\n  vs exp_067 (13.79): Delta {cv_delta - 13.79:+.4f}")
    print("=" * 70)

    # Test prediction if delta is better
    if cv_delta < cv_abs:
        print("\n[2] Test Prediction (Delta)...")
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

        X_test = test_last[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
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

        output_path = SUBMISSION_DIR / f"submission_delta_3fold_cv{cv_delta:.2f}.csv"
        submission.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
