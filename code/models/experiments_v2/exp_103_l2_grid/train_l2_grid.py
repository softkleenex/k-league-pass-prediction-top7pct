"""
exp_103: L2 Regularization Grid Search
Find optimal L2 value for CatBoost
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

TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
          'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
          'ema_start_y', 'ema_success_rate', 'ema_possession',
          'zone_x', 'result_encoded', 'diff_x', 'velocity']

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def run_l2_test(X, y_dx, y_dy, y_abs, start_xy, groups, l2_value, n_seeds=5):
    """Test specific L2 value"""
    gkf = GroupKFold(n_splits=11)
    all_oof = []

    for seed in SEED_POOL[:n_seeds]:
        params = {
            'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
            'l2_leaf_reg': l2_value, 'random_state': seed, 'verbose': 0,
            'early_stopping_rounds': 50, 'loss_function': 'MAE'
        }
        oof_dx = np.zeros(len(X))
        oof_dy = np.zeros(len(X))

        for train_idx, val_idx in gkf.split(X, y_dx, groups):
            m_dx = CatBoostRegressor(**params)
            m_dy = CatBoostRegressor(**params)
            m_dx.fit(X[train_idx], y_dx[train_idx], eval_set=(X[val_idx], y_dx[val_idx]), use_best_model=True)
            m_dy.fit(X[train_idx], y_dy[train_idx], eval_set=(X[val_idx], y_dy[val_idx]), use_best_model=True)
            oof_dx[val_idx] = m_dx.predict(X[val_idx])
            oof_dy[val_idx] = m_dy.predict(X[val_idx])
            del m_dx, m_dy

        pred = np.column_stack([start_xy[:, 0] + oof_dx, start_xy[:, 1] + oof_dy])
        all_oof.append(pred)
        gc.collect()

    ensemble = np.mean(all_oof, axis=0)
    cv = np.sqrt((ensemble[:, 0] - y_abs[:, 0])**2 + (ensemble[:, 1] - y_abs[:, 1])**2).mean()
    return cv

def main():
    print("=" * 70)
    print("exp_103: L2 Regularization Grid Search")
    print("=" * 70)

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    del train_df; gc.collect()

    X = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = last_passes['dx'].values.astype(np.float32)
    y_dy = last_passes['dy'].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    print(f"Samples: {len(X)}")

    # Test different L2 values
    l2_values = [3, 10, 20, 25, 30, 35, 40, 50, 75, 100]
    results = {}

    for l2 in l2_values:
        print(f"\nTesting L2={l2}...")
        cv = run_l2_test(X, y_dx, y_dy, y_abs, start_xy, groups, l2, n_seeds=5)
        results[l2] = cv
        print(f"  L2={l2}: CV {cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS (sorted by CV)")
    print("=" * 70)
    baseline = results[3]
    for l2, cv in sorted(results.items(), key=lambda x: x[1]):
        diff = cv - baseline
        marker = " <-- BEST" if cv == min(results.values()) else ""
        print(f"  L2={l2:3d}: CV {cv:.4f} ({diff:+.4f}){marker}")

    best_l2 = min(results, key=results.get)
    print(f"\nBest L2: {best_l2} (CV {results[best_l2]:.4f})")
    print("=" * 70)

    # Save results
    with open(BASE / "experiments" / "exp_103_results.txt", "w") as f:
        f.write("L2 Grid Search Results\n")
        f.write("=" * 50 + "\n")
        for l2, cv in sorted(results.items(), key=lambda x: x[1]):
            f.write(f"L2={l2}: CV {cv:.4f}\n")
        f.write(f"\nBest L2: {best_l2}\n")

if __name__ == "__main__":
    main()
