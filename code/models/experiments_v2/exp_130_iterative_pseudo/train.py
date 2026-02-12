"""
exp_130: Iterative Pseudo-Labeling
- Round 1: Train on train → predict test → pseudo-labels
- Round 2: Train on train+pseudo → predict test → refined pseudo-labels
- Round 3: Train on train+refined → final prediction
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

def train_and_predict(X_train, y_dx, y_dy, X_test, groups, seeds, base_params):
    """Train models and return predictions for test set"""
    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_train, y_dx, groups))

    pred_dx = np.zeros(len(X_test))
    pred_dy = np.zeros(len(X_test))
    oof_dx = np.zeros(len(X_train))
    oof_dy = np.zeros(len(X_train))
    n_models = 0

    for seed in seeds:
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            params = {**base_params, 'random_seed': seed}

            model_dx = CatBoostRegressor(**params)
            model_dx.fit(X_train[tr_idx], y_dx[tr_idx],
                        eval_set=(X_train[val_idx], y_dx[val_idx]))
            pred_dx += model_dx.predict(X_test)
            oof_dx[val_idx] += model_dx.predict(X_train[val_idx])

            model_dy = CatBoostRegressor(**params)
            model_dy.fit(X_train[tr_idx], y_dy[tr_idx],
                        eval_set=(X_train[val_idx], y_dy[val_idx]))
            pred_dy += model_dy.predict(X_test)
            oof_dy[val_idx] += model_dy.predict(X_train[val_idx])
            n_models += 1

    pred_dx /= n_models
    pred_dy /= n_models
    oof_dx /= len(seeds)
    oof_dy /= len(seeds)

    # Calculate CV
    cv = np.mean(np.sqrt((oof_dx - y_dx)**2 + (oof_dy - y_dy)**2))
    return pred_dx, pred_dy, cv

def main():
    print("=" * 60)
    print("exp_130: Iterative Pseudo-Labeling")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = load_test_data()
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Create features
    train_df = create_features(train_df)
    test_df['dx'] = 0
    test_df['dy'] = 0
    test_df = create_features(test_df)

    # Last event per episode
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

    seeds = [42, 123, 456]
    n_rounds = 3

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'='*40}")
        print(f"Round {round_num}/{n_rounds}")
        print(f"{'='*40}")

        if round_num == 1:
            # Train on original data
            print("Training on original train data...")
            pseudo_dx, pseudo_dy, cv = train_and_predict(
                X_train, y_dx, y_dy, X_test, groups, seeds, base_params)
            print(f"  CV: {cv:.4f}")
        else:
            # Train on combined data
            print("Training on train + pseudo-labeled data...")
            X_combined = np.vstack([X_train, X_test])
            y_dx_combined = np.concatenate([y_dx, pseudo_dx])
            y_dy_combined = np.concatenate([y_dy, pseudo_dy])

            # Use only train groups for CV calculation
            groups_combined = np.concatenate([groups, np.full(len(X_test), -1)])

            pseudo_dx, pseudo_dy, cv = train_and_predict(
                X_combined, y_dx_combined, y_dy_combined, X_test,
                groups_combined, seeds, base_params)
            print(f"  CV (on train only): {cv:.4f}")

        print(f"  Pseudo dx range: [{pseudo_dx.min():.2f}, {pseudo_dx.max():.2f}]")
        print(f"  Pseudo dy range: [{pseudo_dy.min():.2f}, {pseudo_dy.max():.2f}]")

    # Final prediction
    print("\n" + "=" * 40)
    print("Final Results")
    print("=" * 40)

    end_x = test_last['start_x'].values + pseudo_dx
    end_y = test_last['start_y'].values + pseudo_dy

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': end_x,
        'end_y': end_y
    })

    # Match sample submission order
    sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = sample[['game_episode']].merge(submission, on='game_episode', how='left')

    out_path = BASE / 'submissions' / 'submission_iterative_pseudo.csv'
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Shape: {submission.shape}")

if __name__ == '__main__':
    main()
