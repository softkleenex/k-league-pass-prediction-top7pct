"""
exp_036: Last Pass Only Training

핵심 아이디어:
  - 마지막 pass만으로 학습 → 분포 일치 → 더 나은 generalization
  - TRAIN 356K passes → 마지막 15K passes만 사용
  - Phase1A 21개 피처
  - 3-fold GroupKFold CV

목표: Public < 15.0

작성일: 2025-12-21
"""

import pandas as pd
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


def create_phase1a_features(df):
    """Phase1A 피처 생성 (21개)"""
    print("  Creating Phase1A features...")

    # Zone
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)

    # Direction
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    # Goal
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    # Time
    df['time_left'] = 5400 - df['time_seconds']
    df['game_clock_min'] = np.where(
        df['period_id'] == 1,
        df['time_seconds'] / 60.0,
        45.0 + df['time_seconds'] / 60.0
    )

    # Pass count
    df['pass_count'] = df.groupby('game_episode').cumcount() + 1

    # Encoding
    df['is_home_encoded'] = df['is_home'].astype(int)
    type_map = {'Pass': 0, 'Carry': 1}
    df['type_encoded'] = df['type_name'].map(type_map).fillna(2).astype(int)
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    # Final team
    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)

    df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

    df['team_switch_event'] = (
        df.groupby('game_episode')['is_final_team'].diff() != 0
    ).astype(int)
    df['team_switches'] = df.groupby('game_episode')['team_switch_event'].cumsum()

    # Possession length
    def calc_streak(group):
        values = group['is_final_team'].values
        streaks = []
        current_streak = 0
        for val in values:
            if val == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)
        return pd.Series(streaks, index=group.index)

    df['final_poss_len'] = df.groupby('game_episode', group_keys=False).apply(calc_streak)

    # Cleanup
    df = df.drop(columns=['dx', 'dy', 'team_switch_event', 'final_team_id'], errors='ignore')

    print("    ✓ Phase1A features created")
    return df


def main():
    """메인 실행"""
    print("\n" + "=" * 80)
    print("exp_036: Last Pass Only Training")
    print("=" * 80)
    print("  Strategy: Train ONLY on last passes (15K vs 356K)")
    print("  Expected: Better distribution match → Lower Public score")
    print("=" * 80)

    start_time = time.time()

    # Step 1: Load data
    print("\n" + "=" * 80)
    print("Step 1: Loading TRAIN data")
    print("=" * 80)

    train_path = '../../../../data/train.csv'
    print(f"  Loading {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"    Total passes: {len(train_df):,}")
    print(f"    Unique episodes: {train_df['game_episode'].nunique()}")

    # Step 2: Create features
    print("\n" + "=" * 80)
    print("Step 2: Creating Phase1A Features")
    print("=" * 80)

    train_df = create_phase1a_features(train_df)

    # Step 3: Select LAST PASS ONLY!
    print("\n" + "=" * 80)
    print("Step 3: 🎯 Selecting LAST PASS ONLY")
    print("=" * 80)

    print(f"  Before: {len(train_df):,} passes")
    last_passes = train_df.groupby('game_episode').last().reset_index(drop=True)
    print(f"  After:  {len(last_passes):,} passes (last only)")
    print(f"  Reduction: {len(train_df) - len(last_passes):,} passes removed")

    # Step 4: Prepare features
    print("\n" + "=" * 80)
    print("Step 4: Preparing Features")
    print("=" * 80)

    exclude_cols = [
        'game_id', 'game_episode', 'player_id', 'team_id', 'period_id',
        'time_seconds', 'type_name', 'result_name', 'is_home',
        'start_x', 'start_y', 'end_x', 'end_y'
    ]

    feature_cols = [col for col in last_passes.columns if col not in exclude_cols]

    X = last_passes[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Unique games: {len(np.unique(groups))}")

    # Step 5: 3-Fold CV
    print("\n" + "=" * 80)
    print("Step 5: Running 3-Fold Cross-Validation")
    print("=" * 80)

    gkf = GroupKFold(n_splits=3)
    fold_scores = []

    cb_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3.0,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'verbose': 0,
        'random_state': 42
    }

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n  Fold {fold}/3")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"    Train: {len(X_train):,} samples")
        print(f"    Val:   {len(X_val):,} samples")

        print(f"    Training models...", end='', flush=True)
        fold_start = time.time()

        model_x = CatBoostRegressor(**cb_params)
        model_y = CatBoostRegressor(**cb_params)

        model_x.fit(X_train, y_train[:, 0])
        model_y.fit(X_train, y_train[:, 1])

        print(f" {time.time() - fold_start:.1f}s")

        pred_x = model_x.predict(X_val)
        pred_y = model_y.predict(X_val)

        distances = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
        fold_score = distances.mean()
        fold_scores.append(fold_score)

        print(f"    Fold {fold} Score: {fold_score:.4f}")

    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    print(f"\n" + "=" * 80)
    print("CV Results")
    print("=" * 80)
    print(f"  CV Mean: {cv_mean:.4f}")
    print(f"  CV Std:  {cv_std:.4f}")
    print(f"  Folds:   {[f'{s:.4f}' for s in fold_scores]}")

    # Step 6: Train final model
    print(f"\n" + "=" * 80)
    print("Step 6: Training Final Model (All Last Passes)")
    print("=" * 80)

    print(f"  Training on {len(X):,} last passes...")

    final_model_x = CatBoostRegressor(**cb_params)
    final_model_y = CatBoostRegressor(**cb_params)

    final_model_x.fit(X, y[:, 0])
    final_model_y.fit(X, y[:, 1])

    print("  ✓ Models trained")

    # Step 7: Save
    print(f"\n" + "=" * 80)
    print("Step 7: Saving Results")
    print("=" * 80)

    with open('model_x_catboost.pkl', 'wb') as f:
        pickle.dump(final_model_x, f)
    print("  ✓ model_x_catboost.pkl")

    with open('model_y_catboost.pkl', 'wb') as f:
        pickle.dump(final_model_y, f)
    print("  ✓ model_y_catboost.pkl")

    results = {
        'experiment': 'exp_036_last_pass_only',
        'cv_mean': float(cv_mean),
        'cv_std': float(cv_std),
        'cv_folds': [float(s) for s in fold_scores],
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'baseline': 'Phase1A (CV 15.45, Public 15.35)'
    }

    with open('cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  ✓ cv_results.json")

    runtime = time.time() - start_time

    print(f"\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"  CV:      {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Runtime: {runtime:.1f}s")
    print(f"\n  Next: python predict_submission.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
