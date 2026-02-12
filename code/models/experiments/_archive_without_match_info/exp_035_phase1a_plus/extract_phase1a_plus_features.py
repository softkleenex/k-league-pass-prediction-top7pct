"""
Extract Phase1A + Selected Features for exp_035

================================================================================
Mission: Create feature-engineered datasets for exp_035
================================================================================

Strategy:
- Phase1A baseline (21 features)
- + 5 selected advanced features that work on last passes:
  1. ball_speed
  2. distance_to_goal_line
  3. distance_to_sideline
  4. possession_duration
  5. pressure_level

Excluded advanced features (invalid for last passes):
- prev_dx, prev_dy (always 0)
- space_in_front (always 0)
- is_counter_attack, is_set_piece (rarely meaningful)
- in_penalty_area, in_final_third (rarely meaningful)

Total features: 21 + 5 = 26

Output:
- train_phase1a_plus_features.csv
- test_phase1a_plus_features.csv

Author: exp_035 Team
Date: 2025-12-17
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class Phase1APlusFeatureExtractor:
    """Phase1A + Selected Advanced Features Extractor"""

    def __init__(self,
                 train_path='../../../../data/train.csv',
                 test_path='../../../../data/test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None

    def load_data(self):
        """Load training and test data"""
        print("\n" + "=" * 80)
        print("Step 1: Loading Data")
        print("=" * 80)

        print("  Loading train.csv...")
        self.train_df = pd.read_csv(self.train_path)
        print(f"    Total passes: {len(self.train_df):,}")
        print(f"    Unique players: {self.train_df['player_id'].nunique()}")
        print(f"    Unique games: {self.train_df['game_id'].nunique()}")

        print("\n  Loading test.csv (episode metadata)...")
        test_meta = pd.read_csv(self.test_path)
        print(f"    Test episodes: {len(test_meta):,}")

        # Load individual episode CSV files
        print("  Loading individual episode CSV files...")
        test_dfs = []
        for idx, row in test_meta.iterrows():
            episode_path = Path(self.test_path).parent / row['path']
            if episode_path.exists():
                episode_df = pd.read_csv(episode_path)
                episode_df['game_episode'] = row['game_episode']
                test_dfs.append(episode_df)

        self.test_df = pd.concat(test_dfs, ignore_index=True)
        print(f"    Loaded {len(self.test_df):,} passes from {len(test_dfs)} episodes")
        print(f"    Unique games: {self.test_df['game_id'].nunique()}")

        return self.train_df, self.test_df

    def create_phase1a_features(self, df):
        """Create Phase1A baseline features (21 total)"""
        print("\n  Creating Phase1A baseline features (21)...")

        # 1. Zone (2 features)
        df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
        df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)

        # 2. Direction (3 features: prev_dx, prev_dy, direction)
        df['dx'] = df['end_x'] - df['start_x']
        df['dy'] = df['end_y'] - df['start_y']
        df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
        df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

        angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
        df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

        # 3. Goal distance/angle (2 features)
        df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
        df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

        # 4. Time features (2 features)
        df['time_left'] = 5400 - df['time_seconds']
        df['game_clock_min'] = np.where(
            df['period_id'] == 1,
            df['time_seconds'] / 60.0,
            45.0 + df['time_seconds'] / 60.0
        )

        # 5. Pass count (1 feature)
        df['pass_count'] = df.groupby('game_episode').cumcount() + 1

        # 6. Encoding (3 features)
        df['is_home_encoded'] = df['is_home'].astype(int)
        type_map = {'Pass': 0, 'Carry': 1}
        df['type_encoded'] = df['type_name'].map(type_map).fillna(2).astype(int)
        result_map = {'Successful': 0, 'Unsuccessful': 1}
        df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

        # 7. Final team features (3 features: is_final_team, team_possession_pct, team_switches)
        df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
        df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)

        df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )

        df['team_switch_event'] = (
            df.groupby('game_episode')['is_final_team'].diff() != 0
        ).astype(int)
        df['team_switches'] = df.groupby('game_episode')['team_switch_event'].cumsum()

        # 8. Possession length (1 feature)
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

        # Total: 21 features
        # zone_x, zone_y (2)
        # prev_dx, prev_dy, direction (3)
        # goal_distance, goal_angle (2)
        # time_left, game_clock_min (2)
        # pass_count (1)
        # is_home_encoded, type_encoded, result_encoded (3)
        # is_final_team, team_possession_pct, team_switches (3)
        # final_poss_len (1)
        # episode_id, action_id (2) - from original data
        # dx, dy (2) - intermediate but useful
        # = 21 total

        print("    ✓ Phase1A features created")

        return df

    def create_selected_advanced_features(self, df):
        """Create 5 selected advanced features"""
        print("\n  Creating selected advanced features (5)...")

        # 1. ball_speed (from ball_velocity_x, ball_velocity_y)
        df['ball_velocity_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
        df['ball_velocity_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
        df['ball_speed'] = np.sqrt(df['ball_velocity_x']**2 + df['ball_velocity_y']**2)

        # 2. distance_to_goal_line
        df['distance_to_goal_line'] = 105 - df['start_x']

        # 3. distance_to_sideline
        df['distance_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])

        # 4. possession_duration (simple version: cumulative time in possession)
        df['time_diff'] = df.groupby('game_episode')['time_seconds'].diff().fillna(0)

        def calc_possession_duration(group):
            durations = []
            current_duration = 0

            for is_final, time_diff in zip(group['is_final_team'].values, group['time_diff'].values):
                if is_final == 1:
                    current_duration += abs(time_diff)
                else:
                    current_duration = 0
                durations.append(current_duration)

            return pd.Series(durations, index=group.index)

        df['possession_duration'] = df.groupby('game_episode', group_keys=False).apply(calc_possession_duration)

        # 5. pressure_level (based on pass speed and team switches)
        # High pressure = many team switches + high pass frequency
        df['passes_per_minute'] = df.groupby('game_episode')['pass_count'].transform(
            lambda x: x / (x.index + 1)
        ) * 60

        df['pressure_level'] = (
            df['team_switches'] * 0.5 +
            df['passes_per_minute'].fillna(0) * 0.3 +
            (df['ball_speed'] > df['ball_speed'].median()).astype(int) * 0.2
        )

        print("    ✓ Selected advanced features created")

        return df

    def create_features(self, df, is_train=True):
        """Create all features for exp_035"""
        print("\n" + "=" * 80)
        print(f"Step 2: Creating Features ({'TRAIN' if is_train else 'TEST'})")
        print("=" * 80)

        df = df.copy()

        # Add episode_id if not exists
        if 'episode_id' not in df.columns:
            df['episode_id'] = df['game_episode']

        # Add action_id if not exists (episode내 순서)
        if 'action_id' not in df.columns:
            df['action_id'] = df.groupby('game_episode').cumcount()

        # Phase1A features
        df = self.create_phase1a_features(df)

        # Selected advanced features
        df = self.create_selected_advanced_features(df)

        # Clean up intermediate columns
        df = df.drop(columns=['dx', 'dy', 'team_switch_event', 'final_team_id',
                               'time_diff', 'passes_per_minute',
                               'ball_velocity_x', 'ball_velocity_y'], errors='ignore')

        print(f"\n  Total features: {len(df.columns)}")
        print(f"  Total rows: {len(df):,}")

        return df

    def save_data(self, train_df, test_df):
        """Save feature-engineered datasets"""
        print("\n" + "=" * 80)
        print("Step 3: Saving Feature-Engineered Datasets")
        print("=" * 80)

        train_file = 'train_phase1a_plus_features.csv'
        test_file = 'test_phase1a_plus_features.csv'

        print(f"  Saving {train_file}...")
        train_df.to_csv(train_file, index=False)
        print(f"    ✓ Saved ({len(train_df):,} rows, {len(train_df.columns)} columns)")

        print(f"\n  Saving {test_file}...")
        test_df.to_csv(test_file, index=False)
        print(f"    ✓ Saved ({len(test_df):,} rows, {len(test_df.columns)} columns)")

        return train_file, test_file


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("exp_035: Extract Phase1A + Selected Features")
    print("=" * 80)
    print("\nStrategy:")
    print("  Phase1A baseline: 21 features")
    print("  Selected advanced: 5 features")
    print("  Total: 26 features")
    print("\nSelected features (valid for last passes):")
    print("  1. ball_speed")
    print("  2. distance_to_goal_line")
    print("  3. distance_to_sideline")
    print("  4. possession_duration")
    print("  5. pressure_level")

    # Initialize extractor
    extractor = Phase1APlusFeatureExtractor()

    # Step 1: Load data
    train_df, test_df = extractor.load_data()

    # Step 2: Create features
    train_df = extractor.create_features(train_df, is_train=True)
    test_df = extractor.create_features(test_df, is_train=False)

    # Step 3: Save datasets
    train_file, test_file = extractor.save_data(train_df, test_df)

    print("\n" + "=" * 80)
    print("Feature Extraction Complete!")
    print("=" * 80)
    print(f"  Output files:")
    print(f"    - {train_file}")
    print(f"    - {test_file}")
    print("\n  Next steps:")
    print("    1. python train_phase1a_plus_model.py")
    print("    2. If CV < 15.0, python predict_submission.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
