"""
exp_079: Multi-Seed Ensemble
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import gc
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
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_y'])
    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt((df['start_x'] - df['prev_start_x'])**2 + (df['start_y'] - df['prev_start_y'])**2)
    return df

def main():
    print("=" * 70)
    print("exp_079: Multi-Seed Ensemble")
    print("=" * 70)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    del train_df; gc.collect()

    TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
              'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
              'ema_start_y', 'ema_success_rate', 'ema_possession',
              'zone_x', 'result_encoded', 'diff_x', 'velocity']

    X = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = last_passes[['dx', 'dy']].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    SEEDS = [42, 123, 456, 789, 2024]
    seed_results = {}
    all_oof = []

    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                  'l2_leaf_reg': 3.0, 'random_state': seed, 'verbose': 0,
                  'early_stopping_rounds': 50, 'loss_function': 'MAE'}

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
            del model_dx, model_dy

        pred_abs = np.zeros((len(X), 2))
        pred_abs[:, 0] = start_xy[:, 0] + oof_delta[:, 0]
        pred_abs[:, 1] = start_xy[:, 1] + oof_delta[:, 1]
        cv = np.sqrt((pred_abs[:, 0] - y_abs[:, 0])**2 + (pred_abs[:, 1] - y_abs[:, 1])**2).mean()
        seed_results[seed] = cv
        all_oof.append(pred_abs.copy())
        print(f"  Seed {seed} CV: {cv:.4f}")
        gc.collect()

    # Ensemble predictions
    print("\n[Ensemble]")
    ensemble_preds = np.mean(all_oof, axis=0)
    ensemble_cv = np.sqrt((ensemble_preds[:, 0] - y_abs[:, 0])**2 + (ensemble_preds[:, 1] - y_abs[:, 1])**2).mean()
    print(f"  Ensemble CV: {ensemble_cv:.4f}")

    print("\n" + "=" * 70)
    print("Results Summary:")
    for seed, cv in seed_results.items():
        print(f"  Seed {seed}: CV {cv:.4f}")
    print(f"  Ensemble:  CV {ensemble_cv:.4f} ★")
    print(f"\n  vs Single Seed (42): {ensemble_cv - seed_results[42]:+.4f}")
    print("=" * 70)

    # Create ensemble submission
    create_ensemble_submission(SEEDS, TOP_15, last_passes, ensemble_cv)

def create_ensemble_submission(seeds, feature_cols, last_passes, cv):
    print("\n[Creating Ensemble Submission]")

    X = last_passes[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = last_passes[['dx', 'dy']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    all_models_dx = []
    all_models_dy = []

    for seed in seeds:
        params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
                  'l2_leaf_reg': 3.0, 'random_state': seed, 'verbose': 0,
                  'early_stopping_rounds': 50, 'loss_function': 'MAE'}
        gkf = GroupKFold(n_splits=3)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_delta, groups), 1):
            model_dx = CatBoostRegressor(**params)
            model_dy = CatBoostRegressor(**params)
            model_dx.fit(X[train_idx], y_delta[train_idx, 0],
                        eval_set=(X[val_idx], y_delta[val_idx, 0]), use_best_model=True)
            model_dy.fit(X[train_idx], y_delta[train_idx, 1],
                        eval_set=(X[val_idx], y_delta[val_idx, 1]), use_best_model=True)
            all_models_dx.append(model_dx)
            all_models_dy.append(model_dy)

    # Test prediction
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    test_episodes = []
    for _, row in test_df.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
        ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']
        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)
    test_all = pd.concat(test_episodes, ignore_index=True)
    test_all = create_features(test_all)
    test_last = test_all.groupby('game_episode').last().reset_index()

    X_test = test_last[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    pred_dx = np.mean([m.predict(X_test) for m in all_models_dx], axis=0)
    pred_dy = np.mean([m.predict(X_test) for m in all_models_dy], axis=0)
    pred_x = test_last['start_x'].values + pred_dx
    pred_y = np.clip(test_last['start_y'].values + pred_dy, 0, 68)

    submission = pd.DataFrame({'game_episode': test_last['game_episode'], 'end_x': pred_x, 'end_y': pred_y})
    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()
    submission.to_csv(SUBMISSION_DIR / f"submission_multiseed_cv{cv:.2f}.csv", index=False)
    print(f"  Saved: submission_multiseed_cv{cv:.2f}.csv")

if __name__ == "__main__":
    main()
