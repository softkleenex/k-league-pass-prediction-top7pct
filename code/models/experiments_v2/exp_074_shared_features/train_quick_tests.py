"""
exp_074b: Quick Tests - Zone variations and Delta prediction
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


def create_features(df, zone_y_split=6):
    """Feature Engineering with configurable zone_y split"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/zone_y_split)).astype(int).clip(0, zone_y_split-1)

    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    ema_span = 2
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    result_map = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map).fillna(0)
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

    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    return df


def run_cv(X, y, groups, name, target_mode='absolute', start_xy=None):
    """5-Fold CV"""
    cat_params = {
        'iterations': 4000, 'depth': 8, 'learning_rate': 0.01,
        'l2_leaf_reg': 7.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 100, 'loss_function': 'MAE'
    }

    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros((len(X), 2))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model_x = CatBoostRegressor(**cat_params)
        model_y = CatBoostRegressor(**cat_params)

        model_x.fit(X[train_idx], y[train_idx, 0],
                   eval_set=(X[val_idx], y[val_idx, 0]), use_best_model=True)
        model_y.fit(X[train_idx], y[train_idx, 1],
                   eval_set=(X[val_idx], y[val_idx, 1]), use_best_model=True)

        oof_pred[val_idx, 0] = model_x.predict(X[val_idx])
        oof_pred[val_idx, 1] = model_y.predict(X[val_idx])

    # Convert delta back to absolute if needed
    if target_mode == 'delta' and start_xy is not None:
        oof_final = np.zeros((len(X), 2))
        oof_final[:, 0] = start_xy[:, 0] + oof_pred[:, 0]
        oof_final[:, 1] = start_xy[:, 1] + oof_pred[:, 1]
        # Compare against absolute end_x, end_y
        y_abs = start_xy[:, 2:4]  # end_x, end_y
        cv = np.sqrt((oof_final[:, 0] - y_abs[:, 0])**2 + (oof_final[:, 1] - y_abs[:, 1])**2).mean()
    else:
        cv = np.sqrt((oof_pred[:, 0] - y[:, 0])**2 + (oof_pred[:, 1] - y[:, 1])**2).mean()

    print(f"  {name}: CV {cv:.4f}")
    return cv


def main():
    print("=" * 70)
    print("exp_074b: Quick Tests")
    print("=" * 70)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')

    TOP_12 = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession', 'velocity'
    ]

    # Test 1: Zone_y = 6 (baseline)
    print("\n[1] Zone_y = 6 (baseline)...")
    train_feat = create_features(train_df.copy(), zone_y_split=6)
    last_passes = train_feat.groupby('game_episode').last().reset_index()

    X = last_passes[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    cv_zone6 = run_cv(X, y, groups, "Zone_y=6")

    # Test 2: Zone_y = 3 (공유 코드)
    print("\n[2] Zone_y = 3 (shared code style)...")
    train_feat3 = create_features(train_df.copy(), zone_y_split=3)
    last_passes3 = train_feat3.groupby('game_episode').last().reset_index()

    X3 = last_passes3[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y3 = last_passes3[['end_x', 'end_y']].values.astype(np.float32)
    groups3 = last_passes3['game_id'].values

    cv_zone3 = run_cv(X3, y3, groups3, "Zone_y=3")

    # Test 3: Zone_y = 4
    print("\n[3] Zone_y = 4...")
    train_feat4 = create_features(train_df.copy(), zone_y_split=4)
    last_passes4 = train_feat4.groupby('game_episode').last().reset_index()

    X4 = last_passes4[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y4 = last_passes4[['end_x', 'end_y']].values.astype(np.float32)
    groups4 = last_passes4['game_id'].values

    cv_zone4 = run_cv(X4, y4, groups4, "Zone_y=4")

    # Test 4: Delta prediction (predict dx, dy instead of end_x, end_y)
    print("\n[4] Delta prediction (dx, dy)...")
    train_feat_d = create_features(train_df.copy(), zone_y_split=6)
    last_passes_d = train_feat_d.groupby('game_episode').last().reset_index()

    # Target: dx, dy
    y_delta = last_passes_d[['dx', 'dy']].values.astype(np.float32)
    # For converting back: need start_x, start_y, end_x, end_y
    start_xy = last_passes_d[['start_x', 'start_y', 'end_x', 'end_y']].values.astype(np.float32)

    X_d = last_passes_d[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    groups_d = last_passes_d['game_id'].values

    cv_delta = run_cv(X_d, y_delta, groups_d, "Delta (dx,dy)", target_mode='delta', start_xy=start_xy)

    # Results
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    print(f"  Zone_y=6 (baseline): CV {cv_zone6:.4f}")
    print(f"  Zone_y=3:            CV {cv_zone3:.4f} ({cv_zone3 - cv_zone6:+.4f})")
    print(f"  Zone_y=4:            CV {cv_zone4:.4f} ({cv_zone4 - cv_zone6:+.4f})")
    print(f"  Delta (dx,dy):       CV {cv_delta:.4f} ({cv_delta - cv_zone6:+.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
