"""
Train CatBoost Model with Phase1A + Selected Features

================================================================================
Mission: Phase1A + 5 selected advanced features that work on last passes
================================================================================

Background:
- exp_034 failed: advanced features ineffective for last passes (CV 13.40 → Public 35.44)
- Phase1A succeeded: CV 15.45 → Public 15.35
- Strategy: Phase1A baseline + carefully selected features

Features (26 total):
- Phase1A baseline: 21 features
- Selected advanced: 5 features (ball_speed, distance_to_goal_line,
  distance_to_sideline, possession_duration, pressure_level)

Model:
- CatBoost (same as Phase1A for comparison)
- 3-fold GroupKFold CV (by game_id)
- Separate models for end_x and end_y

Target: CV < 15.0

Output:
- cv_results.json: CV scores and feature list
- model_x_catboost.pkl: X predictor
- model_y_catboost.pkl: Y predictor

Author: exp_035 Team
Date: 2025-12-17
================================================================================
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


class Phase1APlusTrainer:
    """Train CatBoost with Phase1A + Selected Features"""

    def __init__(self, train_features_path='train_phase1a_plus_features.csv'):
        """
        Initialize trainer

        Args:
            train_features_path: Path to feature-engineered training data
        """
        self.train_features_path = train_features_path
        self.train_df = None
        self.feature_cols = None
        self.model_x = None
        self.model_y = None

        print("\n" + "=" * 80)
        print("exp_035: Phase1A + Selected Features Trainer")
        print("=" * 80)
        print(f"  Baseline: Phase1A (CV 15.45 → Public 15.35)")
        print(f"  Strategy: Add 5 selected features valid for last passes")
        print(f"  Target: CV < 15.0")
        print("=" * 80)

    def load_data(self):
        """Load feature-engineered data"""
        print("\n" + "=" * 80)
        print("Step 1: Loading Feature-Engineered Data")
        print("=" * 80)

        print(f"  Loading: {self.train_features_path}")
        self.train_df = pd.read_csv(self.train_features_path)

        print(f"    Total passes: {len(self.train_df):,}")
        print(f"    Columns: {len(self.train_df.columns)}")
        print(f"    Unique games: {self.train_df['game_id'].nunique()}")
        print(f"    Unique episodes: {self.train_df['game_episode'].nunique()}")

        return self.train_df

    def prepare_features(self):
        """Prepare feature columns and target variables"""
        print("\n" + "=" * 80)
        print("Step 2: Preparing Features")
        print("=" * 80)

        # Identify feature columns (exclude metadata and targets)
        exclude_cols = [
            'game_id', 'game_episode', 'player_id', 'team_id', 'period_id',
            'time_seconds', 'type_name', 'result_name', 'is_home',
            'start_x', 'start_y', 'end_x', 'end_y'
        ]

        self.feature_cols = [col for col in self.train_df.columns if col not in exclude_cols]

        print(f"  Total features available: {len(self.feature_cols)}")

        # Display feature categories
        print("\n  Feature Categories:")

        phase1a_features = [
            'episode_id', 'action_id',
            'zone_x', 'zone_y', 'direction', 'prev_dx', 'prev_dy',
            'goal_distance', 'goal_angle', 'time_left', 'game_clock_min',
            'pass_count', 'is_home_encoded', 'type_encoded', 'result_encoded',
            'is_final_team', 'team_possession_pct', 'team_switches', 'final_poss_len'
        ]

        selected_advanced_features = [
            'ball_speed',
            'distance_to_goal_line',
            'distance_to_sideline',
            'possession_duration',
            'pressure_level'
        ]

        phase1a_present = [f for f in phase1a_features if f in self.feature_cols]
        advanced_present = [f for f in selected_advanced_features if f in self.feature_cols]

        print(f"    Phase1A: {len(phase1a_present)}/{len(phase1a_features)}")
        for f in phase1a_features:
            if f not in self.feature_cols:
                print(f"      Missing: {f}")

        print(f"    Selected Advanced: {len(advanced_present)}/{len(selected_advanced_features)}")
        for f in selected_advanced_features:
            if f not in self.feature_cols:
                print(f"      Missing: {f}")

        # Use only the planned features (ignore intermediate ones)
        planned_features = phase1a_present + advanced_present
        self.feature_cols = [f for f in self.feature_cols if f in planned_features]

        print(f"\n  Final feature count: {len(self.feature_cols)}")

        # Check for missing values
        print("\n  Checking for missing values...")
        missing_counts = self.train_df[self.feature_cols].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"  WARNING: Found missing values in {(missing_counts > 0).sum()} features")
            print(f"  Filling with 0...")
            self.train_df[self.feature_cols] = self.train_df[self.feature_cols].fillna(0)
        else:
            print(f"  ✓ No missing values")

        # Check for infinite values
        print("\n  Checking for infinite values...")
        inf_counts = np.isinf(self.train_df[self.feature_cols]).sum()
        if inf_counts.sum() > 0:
            print(f"  WARNING: Found infinite values in {(inf_counts > 0).sum()} features")
            print(f"  Replacing with 0...")
            self.train_df[self.feature_cols] = self.train_df[self.feature_cols].replace([np.inf, -np.inf], 0)
        else:
            print(f"  ✓ No infinite values")

        print(f"\n  ✓ Features ready")
        print(f"  Feature list: {self.feature_cols}")

        return self.feature_cols

    def run_cv(self, n_folds=3):
        """Run cross-validation"""
        print("\n" + "=" * 80)
        print(f"Step 3: Running {n_folds}-Fold Cross-Validation")
        print("=" * 80)

        # Prepare data
        X = self.train_df[self.feature_cols].values
        y = self.train_df[['end_x', 'end_y']].values
        groups = self.train_df['game_id'].values

        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Unique games (groups): {len(np.unique(groups))}")

        # GroupKFold split
        gkf = GroupKFold(n_splits=n_folds)

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            print(f"\n  Fold {fold}/{n_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            print(f"    Train: {len(X_train):,} samples")
            print(f"    Val: {len(X_val):,} samples")

            # CatBoost parameters (same as Phase1A for comparison)
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

            # Train models
            print(f"    Training models...", end='', flush=True)
            start_time = time.time()

            model_x = CatBoostRegressor(**cb_params)
            model_y = CatBoostRegressor(**cb_params)

            model_x.fit(X_train, y_train[:, 0])
            model_y.fit(X_train, y_train[:, 1])

            train_time = time.time() - start_time
            print(f" {train_time:.1f}s")

            # Predict
            pred_x = model_x.predict(X_val)
            pred_y = model_y.predict(X_val)

            # Calculate Euclidean distance
            distances = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
            fold_score = distances.mean()

            fold_scores.append(fold_score)

            print(f"    Fold {fold} Score: {fold_score:.4f}")

        # Calculate mean and std
        cv_mean = np.mean(fold_scores)
        cv_std = np.std(fold_scores)

        print(f"\n" + "=" * 80)
        print("Cross-Validation Results")
        print("=" * 80)
        print(f"  CV Mean: {cv_mean:.4f}")
        print(f"  CV Std: {cv_std:.4f}")
        print(f"  Fold scores: {[f'{s:.4f}' for s in fold_scores]}")

        # Compare with baseline
        baseline_cv = 15.45
        improvement = baseline_cv - cv_mean

        print(f"\n  Baseline (Phase1A): {baseline_cv:.4f}")
        print(f"  Improvement: {improvement:+.4f}")

        if cv_mean < 15.0:
            print(f"\n  ✅ Target achieved! (CV < 15.0)")
        else:
            print(f"\n  ⚠️ Target not met (CV >= 15.0)")

        return cv_mean, cv_std, fold_scores

    def train_final_model(self):
        """Train final model on all data"""
        print("\n" + "=" * 80)
        print("Step 4: Training Final Model (All Data)")
        print("=" * 80)

        X = self.train_df[self.feature_cols].values
        y = self.train_df[['end_x', 'end_y']].values

        print(f"  Training on {len(X):,} samples...")

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

        print(f"  Training model_x...", end='', flush=True)
        self.model_x = CatBoostRegressor(**cb_params)
        self.model_x.fit(X, y[:, 0])
        print(" ✓")

        print(f"  Training model_y...", end='', flush=True)
        self.model_y = CatBoostRegressor(**cb_params)
        self.model_y.fit(X, y[:, 1])
        print(" ✓")

        return self.model_x, self.model_y

    def save_results(self, cv_mean, cv_std, fold_scores):
        """Save CV results and models"""
        print("\n" + "=" * 80)
        print("Step 5: Saving Results")
        print("=" * 80)

        # Save CV results
        results = {
            'experiment': 'exp_035_phase1a_plus',
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'cv_folds': [float(x) for x in fold_scores],
            'n_folds': len(fold_scores),
            'n_features': len(self.feature_cols),
            'feature_cols': self.feature_cols,
            'n_samples': len(self.train_df),
            'baseline': {
                'name': 'Phase1A',
                'cv': 15.45,
                'improvement': float(15.45 - cv_mean)
            }
        }

        with open('cv_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  ✓ CV results saved: cv_results.json")

        # Save models
        with open('model_x_catboost.pkl', 'wb') as f:
            pickle.dump(self.model_x, f)
        print(f"  ✓ Model X saved: model_x_catboost.pkl")

        with open('model_y_catboost.pkl', 'wb') as f:
            pickle.dump(self.model_y, f)
        print(f"  ✓ Model Y saved: model_y_catboost.pkl")


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("exp_035: Train Phase1A + Selected Features Model")
    print("=" * 80)

    start_time = time.time()

    # Initialize trainer
    trainer = Phase1APlusTrainer()

    # Step 1: Load data
    trainer.load_data()

    # Step 2: Prepare features
    trainer.prepare_features()

    # Step 3: Run CV
    cv_mean, cv_std, fold_scores = trainer.run_cv(n_folds=3)

    # Step 4: Train final model
    trainer.train_final_model()

    # Step 5: Save results
    trainer.save_results(cv_mean, cv_std, fold_scores)

    runtime = time.time() - start_time

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Runtime: {runtime:.1f}s")
    print("\n  Next steps:")
    if cv_mean < 15.0:
        print("    ✅ CV < 15.0! Generate submission:")
        print("       python predict_submission.py")
    else:
        print("    ⚠️ CV >= 15.0, consider:")
        print("       - Hyperparameter tuning")
        print("       - Additional feature engineering")
    print("=" * 80)


if __name__ == '__main__':
    main()
