"""
exp_036: Generate Submission

작성일: 2025-12-21
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path


def create_phase1a_features(df):
    """Phase1A 피처 생성 (train과 동일)"""
    # Zone
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)

    # Direction
    df['dx'] = df['end_x'] - df['start_x'] if 'end_x' in df.columns else 0
    df['dy'] = df['end_y'] - df['start_y'] if 'end_y' in df.columns else 0
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0) if 'dx' in df.columns else 0
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0) if 'dy' in df.columns else 0

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

    return df


def main():
    print("\n" + "=" * 80)
    print("exp_036: Generate Submission")
    print("=" * 80)

    # Load models
    print("\nLoading models...")
    with open('model_x_catboost.pkl', 'rb') as f:
        model_x = pickle.load(f)
    with open('model_y_catboost.pkl', 'rb') as f:
        model_y = pickle.load(f)
    print("  ✓ Models loaded")

    # Load CV results for feature list
    with open('cv_results.json', 'r') as f:
        cv_results = json.load(f)
    feature_cols = cv_results['feature_cols']
    print(f"  Features: {len(feature_cols)}")

    # Load test metadata
    print("\nLoading TEST data...")
    test_meta = pd.read_csv('../../../../data/test.csv')
    print(f"  Test episodes: {len(test_meta)}")

    # Load individual episode files
    test_dfs = []
    for idx, row in test_meta.iterrows():
        episode_path = Path('../../../../data') / row['path']
        if episode_path.exists():
            episode_df = pd.read_csv(episode_path)
            episode_df['game_episode'] = row['game_episode']
            test_dfs.append(episode_df)

    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"  Total passes: {len(test_df)}")

    # Create features
    print("\nCreating features...")
    test_df = create_phase1a_features(test_df)

    # Select last pass per episode
    print("\nSelecting last pass per episode...")
    last_passes = test_df.groupby('game_episode').last().reset_index()
    print(f"  Episodes: {len(last_passes)}")

    # Prepare features
    X_test = last_passes[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values

    # Predict
    print("\nGenerating predictions...")
    pred_x = model_x.predict(X_test)
    pred_y = model_y.predict(X_test)

    # Clip
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)

    # Create submission
    submission = pd.DataFrame({
        'game_episode': last_passes['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })

    # Validate
    assert len(submission) == 2414, f"Expected 2414, got {len(submission)}"
    print(f"  ✓ Validation passed")

    # Save
    cv_score = cv_results['cv_mean']
    filename = f'submission_last_pass_only_cv{cv_score:.4f}.csv'
    submission.to_csv(filename, index=False)

    print(f"\n" + "=" * 80)
    print("Submission Complete!")
    print("=" * 80)
    print(f"  File: {filename}")
    print(f"  Rows: {len(submission)}")
    print(f"  CV: {cv_score:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
