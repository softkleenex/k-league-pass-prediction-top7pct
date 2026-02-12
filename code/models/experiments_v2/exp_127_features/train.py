"""exp_127: New Feature Engineering with Depth=9, L2=600"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"

def create_base_features(df):
    """Original 16 features"""
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
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_y'])
    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt((df['start_x'] - df['prev_start_x'])**2 + (df['start_y'] - df['prev_start_y'])**2)
    df['ema_momentum_y'] = df['ema_start_y'] - df['start_y']
    return df

def add_new_features(df):
    """Add new experimental features"""
    # Sequence position
    df['seq_idx'] = df.groupby('game_episode').cumcount()
    df['seq_len'] = df.groupby('game_episode')['seq_idx'].transform('max') + 1
    df['seq_ratio'] = df['seq_idx'] / df['seq_len']

    # Action type encoding (if available)
    if 'action_type' in df.columns:
        action_map = {a: i for i, a in enumerate(df['action_type'].unique())}
        df['action_encoded'] = df['action_type'].map(action_map).fillna(-1).astype(int)

    # More granular zones
    df['zone_x_fine'] = (df['start_x'] / (105/10)).astype(int).clip(0, 9)
    df['zone_y_fine'] = (df['start_y'] / (68/10)).astype(int).clip(0, 9)

    # Direction features
    df['move_direction'] = np.degrees(np.arctan2(df['dy'], df['dx']))
    df['prev_move_direction'] = df.groupby('game_episode')['move_direction'].shift(1).fillna(0)
    df['direction_change'] = np.abs(df['move_direction'] - df['prev_move_direction'])

    # Cumulative progress
    df['cum_dx'] = df.groupby('game_episode')['dx'].cumsum()
    df['cum_dy'] = df.groupby('game_episode')['dy'].cumsum()

    # Distance features
    df['pass_length'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['avg_pass_length'] = df.groupby('game_episode')['pass_length'].transform('mean')
    df['pass_length_ratio'] = df['pass_length'] / (df['avg_pass_length'] + 1e-6)

    return df

BASE_FEATURES = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line', 'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x', 'ema_start_y', 'ema_success_rate', 'ema_possession', 'zone_x', 'result_encoded', 'diff_x', 'velocity', 'ema_momentum_y']

NEW_FEATURES_1 = ['seq_ratio']  # Sequence position
NEW_FEATURES_2 = ['zone_x_fine', 'zone_y_fine']  # Fine zones
NEW_FEATURES_3 = ['direction_change']  # Direction
NEW_FEATURES_4 = ['cum_dx', 'cum_dy']  # Cumulative
NEW_FEATURES_5 = ['pass_length_ratio']  # Pass length

def main():
    print("="*60)
    print("exp_127: New Feature Engineering")
    print("="*60)
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_base_features(train_df)
    train_df = add_new_features(train_df)
    train_last = train_df.groupby('game_episode').last().reset_index()
    print(f"Episodes: {len(train_last)}")

    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values
    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(train_last, y_dx, groups))

    feature_sets = [
        ('Base 16', BASE_FEATURES),
        ('+seq_ratio', BASE_FEATURES + NEW_FEATURES_1),
        ('+fine_zones', BASE_FEATURES + NEW_FEATURES_2),
        ('+direction', BASE_FEATURES + NEW_FEATURES_3),
        ('+cumulative', BASE_FEATURES + NEW_FEATURES_4),
        ('+pass_length', BASE_FEATURES + NEW_FEATURES_5),
        ('+all_new', BASE_FEATURES + NEW_FEATURES_1 + NEW_FEATURES_2 + NEW_FEATURES_3 + NEW_FEATURES_4 + NEW_FEATURES_5),
    ]

    results = {}
    for name, features in feature_sets:
        print(f"\n{name} ({len(features)} features)...")
        X = train_last[features].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        scores = []
        for seed in [42, 123, 456]:
            params = {'iterations': 3000, 'depth': 9, 'learning_rate': 0.01, 'l2_leaf_reg': 600.0, 'random_state': seed, 'verbose': 0, 'early_stopping_rounds': 100, 'loss_function': 'MAE'}
            fold_scores = []
            for train_idx, val_idx in folds:
                m_dx = CatBoostRegressor(**params)
                m_dy = CatBoostRegressor(**params)
                m_dx.fit(X[train_idx], y_dx[train_idx], eval_set=(X[val_idx], y_dx[val_idx]), use_best_model=True)
                m_dy.fit(X[train_idx], y_dy[train_idx], eval_set=(X[val_idx], y_dy[val_idx]), use_best_model=True)
                pred_dx = m_dx.predict(X[val_idx])
                pred_dy = m_dy.predict(X[val_idx])
                dist = np.sqrt((pred_dx - y_dx[val_idx])**2 + (pred_dy - y_dy[val_idx])**2)
                fold_scores.append(dist.mean())
            scores.append(np.mean(fold_scores))
        cv = np.mean(scores)
        std = np.std(scores)
        results[name] = (cv, std, len(features))
        print(f"  {name}: CV {cv:.4f} (+/- {std:.4f})")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    best = min(results, key=lambda k: results[k][0])
    for k in sorted(results.keys(), key=lambda k: results[k][0]):
        cv, std, n = results[k]
        m = " <-- BEST" if k == best else ""
        print(f"  {k} ({n}f): CV {cv:.4f} (+/- {std:.4f}){m}")

if __name__ == "__main__":
    main()
