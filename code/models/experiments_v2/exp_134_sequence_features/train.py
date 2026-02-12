"""
exp_134: Sequence-based Features
- Use ALL events in episode, not just the last one
- Aggregate features: mean, std, trend, first/last diff
- Capture full trajectory pattern
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

def create_base_features(df):
    """Basic features for each event"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)

    # Movement within episode
    df['move_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['move_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['move_dist'] = np.sqrt(df['move_x']**2 + df['move_y']**2)

    # Result encoding
    result_map = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map).fillna(0.5)

    return df

def create_sequence_features(df):
    """Aggregate sequence features per episode"""
    agg_funcs = {
        'start_x': ['mean', 'std', 'min', 'max', 'first', 'last'],
        'start_y': ['mean', 'std', 'min', 'max', 'first', 'last'],
        'goal_distance': ['mean', 'std', 'min', 'last'],
        'goal_angle': ['mean', 'std', 'last'],
        'dist_to_goal_line': ['mean', 'min', 'last'],
        'dist_to_center_y': ['mean', 'std', 'last'],
        'move_x': ['sum', 'mean', 'std'],
        'move_y': ['sum', 'mean', 'std'],
        'move_dist': ['sum', 'mean', 'max'],
        'is_successful': ['mean', 'sum'],
        'zone_x': ['first', 'last', 'nunique'],
        'zone_y': ['first', 'last', 'nunique'],
    }

    seq_features = df.groupby('game_episode').agg(agg_funcs)
    seq_features.columns = ['_'.join(col) for col in seq_features.columns]
    seq_features = seq_features.reset_index()

    # Add derived features
    seq_features['x_range'] = seq_features['start_x_max'] - seq_features['start_x_min']
    seq_features['y_range'] = seq_features['start_y_max'] - seq_features['start_y_min']
    seq_features['x_progress'] = seq_features['start_x_last'] - seq_features['start_x_first']
    seq_features['y_progress'] = seq_features['start_y_last'] - seq_features['start_y_first']
    seq_features['total_distance'] = seq_features['move_dist_sum']
    seq_features['avg_step_size'] = seq_features['move_dist_mean']
    seq_features['success_rate'] = seq_features['is_successful_mean']
    seq_features['zone_changes'] = seq_features['zone_x_nunique'] + seq_features['zone_y_nunique']

    # Episode length
    episode_len = df.groupby('game_episode').size().reset_index(name='episode_length')
    seq_features = seq_features.merge(episode_len, on='game_episode')

    return seq_features

def load_test_data():
    test_index = pd.read_csv(DATA_DIR / 'test.csv')
    dfs = []
    for _, row in test_index.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def main():
    print("=" * 60)
    print("exp_134: Sequence-based Features")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = load_test_data()
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Create base features
    train_df = create_base_features(train_df)
    test_df = create_base_features(test_df)

    # Create sequence features
    print("\nCreating sequence features...")
    train_seq = create_sequence_features(train_df)
    test_seq = create_sequence_features(test_df)
    print(f"Train episodes: {len(train_seq)}, Test episodes: {len(test_seq)}")

    # Get target (dx, dy) from last event
    train_last = train_df.groupby('game_episode').last().reset_index()
    train_last['dx'] = train_last['end_x'] - train_last['start_x']
    train_last['dy'] = train_last['end_y'] - train_last['start_y']

    # Merge targets
    train_seq = train_seq.merge(train_last[['game_episode', 'game_id', 'dx', 'dy']], on='game_episode')

    # Get game_id for test (for potential future use)
    test_last = test_df.groupby('game_episode').last().reset_index()
    test_seq = test_seq.merge(test_last[['game_episode', 'start_x', 'start_y']], on='game_episode')

    # Feature columns (all numeric columns except identifiers and targets)
    exclude_cols = ['game_episode', 'game_id', 'dx', 'dy', 'start_x', 'start_y']
    feature_cols = [c for c in train_seq.columns if c not in exclude_cols]
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols[:10]}...")

    X_train = train_seq[feature_cols].values.astype(np.float32)
    X_test = test_seq[feature_cols].values.astype(np.float32)
    y_dx = train_seq['dx'].values.astype(np.float32)
    y_dy = train_seq['dy'].values.astype(np.float32)
    groups = train_seq['game_id'].values

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    base_params = {'iterations': 4000, 'depth': 9, 'learning_rate': 0.008,
                   'l2_leaf_reg': 600.0, 'verbose': 0, 'early_stopping_rounds': 100,
                   'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_train, y_dx, groups))
    seeds = [42, 123, 456]

    # Train
    print("\nTraining...")
    oof_dx = np.zeros(len(X_train))
    oof_dy = np.zeros(len(X_train))
    pred_dx = np.zeros(len(X_test))
    pred_dy = np.zeros(len(X_test))

    for seed in seeds:
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            params = {**base_params, 'random_seed': seed}

            model_dx = CatBoostRegressor(**params)
            model_dx.fit(X_train[tr_idx], y_dx[tr_idx],
                        eval_set=(X_train[val_idx], y_dx[val_idx]))
            oof_dx[val_idx] += model_dx.predict(X_train[val_idx])
            pred_dx += model_dx.predict(X_test)

            model_dy = CatBoostRegressor(**params)
            model_dy.fit(X_train[tr_idx], y_dy[tr_idx],
                        eval_set=(X_train[val_idx], y_dy[val_idx]))
            oof_dy[val_idx] += model_dy.predict(X_train[val_idx])
            pred_dy += model_dy.predict(X_test)
        print(f"  Seed {seed} done")

    oof_dx /= len(seeds)
    oof_dy /= len(seeds)
    pred_dx /= (len(seeds) * len(folds))
    pred_dy /= (len(seeds) * len(folds))

    cv = np.mean(np.sqrt((oof_dx - y_dx)**2 + (oof_dy - y_dy)**2))
    print(f"\nCV Score: {cv:.4f}")

    # Create submission
    end_x = test_seq['start_x'].values + pred_dx
    end_y = test_seq['start_y'].values + pred_dy

    submission = pd.DataFrame({
        'game_episode': test_seq['game_episode'],
        'end_x': end_x,
        'end_y': end_y
    })

    sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = sample[['game_episode']].merge(submission, on='game_episode', how='left')

    out_path = BASE / 'submissions' / f'submission_seq_cv{cv:.2f}.csv'
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
