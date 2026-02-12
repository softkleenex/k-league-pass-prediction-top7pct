"""
exp_135: Hybrid - Base Features + Sequence Features
- exp_128 피처 (마지막 이벤트 기반) + 시퀀스 집계 피처 결합
- exp_134 실패 이유: 마지막 이벤트 위치 정보 손실
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
    """기존 exp_128 피처 (마지막 이벤트용)"""
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

def create_sequence_features(df):
    """시퀀스 집계 피처"""
    # 이동 관련
    df['move_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['move_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['move_dist'] = np.sqrt(df['move_x']**2 + df['move_y']**2)

    agg_funcs = {
        'start_x': ['mean', 'std', 'min', 'max'],
        'start_y': ['mean', 'std', 'min', 'max'],
        'goal_distance': ['mean', 'std', 'min'],
        'goal_angle': ['mean', 'std'],
        'move_x': ['sum', 'mean'],
        'move_y': ['sum', 'mean'],
        'move_dist': ['sum', 'mean', 'max'],
        'is_successful': ['mean', 'sum'],
    }

    seq_features = df.groupby('game_episode').agg(agg_funcs)
    seq_features.columns = ['seq_' + '_'.join(col) for col in seq_features.columns]
    seq_features = seq_features.reset_index()

    # 파생 피처
    seq_features['seq_x_range'] = seq_features['seq_start_x_max'] - seq_features['seq_start_x_min']
    seq_features['seq_y_range'] = seq_features['seq_start_y_max'] - seq_features['seq_start_y_min']

    # 에피소드 길이
    episode_len = df.groupby('game_episode').size().reset_index(name='seq_episode_length')
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

BASE_FEATURES = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
                 'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
                 'ema_start_y', 'ema_success_rate', 'ema_possession',
                 'zone_x', 'result_encoded', 'diff_x', 'velocity', 'ema_momentum_y']

def main():
    print("=" * 60)
    print("exp_135: Hybrid - Base Features + Sequence Features")
    print("=" * 60)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = load_test_data()
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Base features
    train_df = create_base_features(train_df)
    test_df['dx'] = 0
    test_df['dy'] = 0
    test_df = create_base_features(test_df)

    # Sequence features
    print("Creating sequence features...")
    train_seq = create_sequence_features(train_df)
    test_seq = create_sequence_features(test_df)

    # 마지막 이벤트 (base features)
    train_last = train_df.groupby('game_episode').last().reset_index()
    test_last = test_df.groupby('game_episode').last().reset_index()

    # Merge base + sequence
    train_data = train_last.merge(train_seq, on='game_episode')
    test_data = test_last.merge(test_seq, on='game_episode')

    print(f"Train episodes: {len(train_data)}, Test episodes: {len(test_data)}")

    # Feature columns
    seq_cols = [c for c in train_seq.columns if c.startswith('seq_')]
    all_features = BASE_FEATURES + seq_cols
    print(f"Features: {len(BASE_FEATURES)} base + {len(seq_cols)} seq = {len(all_features)} total")

    X_train = train_data[all_features].values.astype(np.float32)
    X_test = test_data[all_features].values.astype(np.float32)
    y_dx = train_data['dx'].values.astype(np.float32)
    y_dy = train_data['dy'].values.astype(np.float32)
    groups = train_data['game_id'].values

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
    end_x = test_data['start_x'].values + pred_dx
    end_y = test_data['start_y'].values + pred_dy

    submission = pd.DataFrame({
        'game_episode': test_data['game_episode'],
        'end_x': end_x,
        'end_y': end_y
    })

    sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    submission = sample[['game_episode']].merge(submission, on='game_episode', how='left')

    out_path = BASE / 'submissions' / f'submission_hybrid_cv{cv:.2f}.csv'
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
