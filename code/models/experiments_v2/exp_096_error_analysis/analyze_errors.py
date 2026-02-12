"""
exp_096: Error Analysis
Analyze where our model makes the biggest errors
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

    # Episode length
    df['episode_length'] = df.groupby('game_episode')['game_episode'].transform('count')

    return df

TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
          'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
          'ema_start_y', 'ema_success_rate', 'ema_possession',
          'zone_x', 'result_encoded', 'diff_x', 'velocity']

def main():
    print("=" * 70)
    print("exp_096: Error Analysis")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data and training model...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    del train_df; gc.collect()

    X = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = last_passes['dx'].values.astype(np.float32)
    y_dy = last_passes['dy'].values.astype(np.float32)
    groups = last_passes['game_id'].values

    # Train and get OOF predictions
    params = {'iterations': 1000, 'depth': 8, 'learning_rate': 0.05,
              'l2_leaf_reg': 3.0, 'random_state': 42, 'verbose': 0,
              'early_stopping_rounds': 50, 'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=11)

    oof_dx = np.zeros(len(X))
    oof_dy = np.zeros(len(X))

    for train_idx, val_idx in gkf.split(X, y_dx, groups):
        model_dx = CatBoostRegressor(**params)
        model_dy = CatBoostRegressor(**params)
        model_dx.fit(X[train_idx], y_dx[train_idx],
                    eval_set=(X[val_idx], y_dx[val_idx]), use_best_model=True)
        model_dy.fit(X[train_idx], y_dy[train_idx],
                    eval_set=(X[val_idx], y_dy[val_idx]), use_best_model=True)
        oof_dx[val_idx] = model_dx.predict(X[val_idx])
        oof_dy[val_idx] = model_dy.predict(X[val_idx])

    # Calculate errors
    last_passes['pred_dx'] = oof_dx
    last_passes['pred_dy'] = oof_dy
    last_passes['pred_end_x'] = last_passes['start_x'] + oof_dx
    last_passes['pred_end_y'] = last_passes['start_y'] + oof_dy
    last_passes['error'] = np.sqrt(
        (last_passes['pred_end_x'] - last_passes['end_x'])**2 +
        (last_passes['pred_end_y'] - last_passes['end_y'])**2
    )
    last_passes['error_dx'] = last_passes['pred_dx'] - last_passes['dx']
    last_passes['error_dy'] = last_passes['pred_dy'] - last_passes['dy']

    print(f"\n[2] Error Statistics...")
    print(f"  Mean error: {last_passes['error'].mean():.4f}")
    print(f"  Std error: {last_passes['error'].std():.4f}")
    print(f"  Median error: {last_passes['error'].median():.4f}")
    print(f"  Min error: {last_passes['error'].min():.4f}")
    print(f"  Max error: {last_passes['error'].max():.4f}")

    # Error distribution
    print(f"\n[3] Error Distribution...")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(last_passes['error'], p)
        print(f"  {p}th percentile: {val:.4f}")

    # Error by zone
    print(f"\n[4] Error by Zone...")
    zone_errors = last_passes.groupby(['zone_x', 'zone_y']).agg({
        'error': ['mean', 'count'],
        'dx': 'std',
        'dy': 'std'
    }).reset_index()
    zone_errors.columns = ['zone_x', 'zone_y', 'mean_error', 'count', 'dx_std', 'dy_std']
    zone_errors = zone_errors.sort_values('mean_error', ascending=False)
    print("  Top 10 zones with highest error:")
    print(zone_errors.head(10).to_string())

    # Error by result
    print(f"\n[5] Error by Result...")
    result_errors = last_passes.groupby('result_encoded')['error'].agg(['mean', 'std', 'count'])
    print(result_errors.to_string())

    # Error by episode length
    print(f"\n[6] Error by Episode Length...")
    last_passes['ep_len_bin'] = pd.cut(last_passes['episode_length'], bins=[0, 2, 5, 10, 20, 100])
    ep_errors = last_passes.groupby('ep_len_bin')['error'].agg(['mean', 'std', 'count'])
    print(ep_errors.to_string())

    # Error by dx magnitude
    print(f"\n[7] Error by dx magnitude (forward/backward)...")
    last_passes['dx_bin'] = pd.cut(last_passes['dx'], bins=[-100, -20, -10, 0, 10, 20, 100])
    dx_errors = last_passes.groupby('dx_bin')['error'].agg(['mean', 'std', 'count'])
    print(dx_errors.to_string())

    # Bias analysis
    print(f"\n[8] Bias Analysis...")
    print(f"  Mean error_dx (pred - actual): {last_passes['error_dx'].mean():.4f}")
    print(f"  Mean error_dy (pred - actual): {last_passes['error_dy'].mean():.4f}")

    # Error by start position
    print(f"\n[9] Error by Start Position...")
    last_passes['start_x_bin'] = pd.cut(last_passes['start_x'], bins=[0, 35, 70, 105])
    start_errors = last_passes.groupby('start_x_bin')['error'].agg(['mean', 'std', 'count'])
    print(start_errors.to_string())

    # High error samples analysis
    print(f"\n[10] High Error Samples Analysis (top 5%)...")
    high_error_threshold = np.percentile(last_passes['error'], 95)
    high_error = last_passes[last_passes['error'] >= high_error_threshold]
    print(f"  High error threshold (95th percentile): {high_error_threshold:.4f}")
    print(f"  Number of high error samples: {len(high_error)}")
    print(f"\n  High error sample characteristics:")
    print(f"    Mean zone_x: {high_error['zone_x'].mean():.2f} vs {last_passes['zone_x'].mean():.2f} (all)")
    print(f"    Mean zone_y: {high_error['zone_y'].mean():.2f} vs {last_passes['zone_y'].mean():.2f} (all)")
    print(f"    Mean episode_length: {high_error['episode_length'].mean():.2f} vs {last_passes['episode_length'].mean():.2f} (all)")
    print(f"    Mean |dx|: {high_error['dx'].abs().mean():.2f} vs {last_passes['dx'].abs().mean():.2f} (all)")
    print(f"    Mean |dy|: {high_error['dy'].abs().mean():.2f} vs {last_passes['dy'].abs().mean():.2f} (all)")

    # Target distribution
    print(f"\n[11] Target (dx, dy) Distribution...")
    print(f"  dx: mean={last_passes['dx'].mean():.2f}, std={last_passes['dx'].std():.2f}")
    print(f"  dy: mean={last_passes['dy'].mean():.2f}, std={last_passes['dy'].std():.2f}")

    # Prediction distribution
    print(f"\n[12] Prediction Distribution...")
    print(f"  pred_dx: mean={last_passes['pred_dx'].mean():.2f}, std={last_passes['pred_dx'].std():.2f}")
    print(f"  pred_dy: mean={last_passes['pred_dy'].mean():.2f}, std={last_passes['pred_dy'].std():.2f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Overall CV: {last_passes['error'].mean():.4f}")
    print(f"  5% worst samples contribute: {high_error['error'].mean():.2f} error on average")
    print("=" * 70)

if __name__ == "__main__":
    main()
