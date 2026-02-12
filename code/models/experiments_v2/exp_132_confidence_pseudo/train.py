"""
exp_132: Confidence-weighted Pseudo-Labeling
- Only use high-confidence pseudo-labels (low variance across folds)
- Weight pseudo-labels by confidence
"""
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"

def create_features(df):
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))
    df['dx'] = df['end_x'] - df['start_x'] if 'end_x' in df.columns else 0
    df['dy'] = df['end_y'] - df['start_y'] if 'end_y' in df.columns else 0
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0) if 'dx' in df.columns else 0
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0) if 'dy' in df.columns else 0
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
    df['ema_momentum_y'] = df['ema_start_y'] - df['start_y']
    return df

def load_test_data():
    test_index = pd.read_csv(DATA_DIR / 'test.csv')
    dfs = []
    for _, row in test_index.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

FEATURES = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
            'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
            'ema_start_y', 'ema_success_rate', 'ema_possession',
            'zone_x', 'result_encoded', 'diff_x', 'velocity', 'ema_momentum_y']

def main():
    print("=" * 60)
    print("exp_132: Confidence-weighted Pseudo-Labeling")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = load_test_data()
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    train_df = create_features(train_df)
    test_df['dx'] = 0
    test_df['dy'] = 0
    test_df = create_features(test_df)

    train_last = train_df.groupby('game_episode').last().reset_index()
    test_last = test_df.groupby('game_episode').last().reset_index()
    print(f"Train episodes: {len(train_last)}, Test episodes: {len(test_last)}")

    X_train = train_last[FEATURES].values.astype(np.float32)
    X_test = test_last[FEATURES].values.astype(np.float32)
    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values

    base_params = {'iterations': 4000, 'depth': 9, 'learning_rate': 0.008,
                   'l2_leaf_reg': 600.0, 'verbose': 0, 'early_stopping_rounds': 100,
                   'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_train, y_dx, groups))
    seeds = [42, 123, 456]

    # Phase 1: Get pseudo-labels with variance estimates
    print("\n[Phase 1] Generating pseudo-labels with confidence...")

    all_pred_dx = []
    all_pred_dy = []

    for seed in seeds:
        seed_pred_dx = np.zeros(len(X_test))
        seed_pred_dy = np.zeros(len(X_test))

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            params = {**base_params, 'random_seed': seed}

            model_dx = CatBoostRegressor(**params)
            model_dx.fit(X_train[tr_idx], y_dx[tr_idx],
                        eval_set=(X_train[val_idx], y_dx[val_idx]))
            seed_pred_dx += model_dx.predict(X_test)

            model_dy = CatBoostRegressor(**params)
            model_dy.fit(X_train[tr_idx], y_dy[tr_idx],
                        eval_set=(X_train[val_idx], y_dy[val_idx]))
            seed_pred_dy += model_dy.predict(X_test)

        seed_pred_dx /= len(folds)
        seed_pred_dy /= len(folds)
        all_pred_dx.append(seed_pred_dx)
        all_pred_dy.append(seed_pred_dy)
        print(f"  Seed {seed} done")

    # Calculate mean and std
    all_pred_dx = np.array(all_pred_dx)
    all_pred_dy = np.array(all_pred_dy)

    pseudo_dx = all_pred_dx.mean(axis=0)
    pseudo_dy = all_pred_dy.mean(axis=0)
    std_dx = all_pred_dx.std(axis=0)
    std_dy = all_pred_dy.std(axis=0)

    # Confidence = 1 / (1 + std)
    confidence = 1 / (1 + np.sqrt(std_dx**2 + std_dy**2))

    print(f"\nConfidence stats:")
    print(f"  Mean: {confidence.mean():.4f}")
    print(f"  Min: {confidence.min():.4f}")
    print(f"  Max: {confidence.max():.4f}")

    # Phase 2: Train with confidence-weighted pseudo-labels
    print("\n[Phase 2] Training with confidence-weighted pseudo-labels...")

    # Select high-confidence samples (top 50%)
    threshold = np.percentile(confidence, 50)
    high_conf_mask = confidence >= threshold
    n_high_conf = high_conf_mask.sum()
    print(f"  High confidence samples: {n_high_conf}/{len(X_test)} ({100*n_high_conf/len(X_test):.1f}%)")

    X_pseudo = X_test[high_conf_mask]
    y_pseudo_dx = pseudo_dx[high_conf_mask]
    y_pseudo_dy = pseudo_dy[high_conf_mask]

    X_combined = np.vstack([X_train, X_pseudo])
    y_dx_combined = np.concatenate([y_dx, y_pseudo_dx])
    y_dy_combined = np.concatenate([y_dy, y_pseudo_dy])

    # Extend groups for combined data
    groups_combined = np.concatenate([groups, np.full(len(X_pseudo), -1)])

    final_dx = np.zeros(len(X_test))
    final_dy = np.zeros(len(X_test))
    oof_dx = np.zeros(len(X_train))
    oof_dy = np.zeros(len(X_train))
    n_final = 0

    for seed in seeds:
        params = {**base_params, 'random_seed': seed}

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            # Combine train fold with high-conf pseudo samples
            combined_tr_idx = np.concatenate([tr_idx, np.arange(len(X_train), len(X_combined))])

            model_dx = CatBoostRegressor(**params)
            model_dx.fit(X_combined[combined_tr_idx], y_dx_combined[combined_tr_idx],
                        eval_set=(X_train[val_idx], y_dx[val_idx]))
            final_dx += model_dx.predict(X_test)
            oof_dx[val_idx] += model_dx.predict(X_train[val_idx])

            model_dy = CatBoostRegressor(**params)
            model_dy.fit(X_combined[combined_tr_idx], y_dy_combined[combined_tr_idx],
                        eval_set=(X_train[val_idx], y_dy[val_idx]))
            final_dy += model_dy.predict(X_test)
            oof_dy[val_idx] += model_dy.predict(X_train[val_idx])
            n_final += 1
        print(f"  Seed {seed} done")

    final_dx /= n_final
    final_dy /= n_final
    oof_dx /= len(seeds)
    oof_dy /= len(seeds)

    cv = np.mean(np.sqrt((oof_dx - y_dx)**2 + (oof_dy - y_dy)**2))
    print(f"\nCV Score: {cv:.4f}")

    # Create submission
    end_x = test_last['start_x'].values + final_dx
    end_y = test_last['start_y'].values + final_dy

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': end_x,
        'end_y': end_y
    })

    sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = sample[['game_episode']].merge(submission, on='game_episode', how='left')

    out_path = BASE / 'submissions' / f'submission_conf_pseudo_cv{cv:.2f}.csv'
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
