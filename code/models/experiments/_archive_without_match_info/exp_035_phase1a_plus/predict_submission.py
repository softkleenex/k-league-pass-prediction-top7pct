"""
Generate Submission File for exp_035

================================================================================
Mission: Create submission.csv using trained Phase1A+ models
================================================================================

Prerequisites:
- train_phase1a_plus_model.py executed successfully (CV 10.22!)
- model_x_catboost.pkl and model_y_catboost.pkl exist
- test_phase1a_plus_features.csv exists
- cv_results.json exists (for CV score in filename)

Output:
- submission_phase1a_plus_cv{score}.csv

Strategy:
1. Predict ALL test passes (not just last!)
2. Select last prediction per episode (groupby last)
3. Clip to field boundaries

Author: exp_035 Team
Date: 2025-12-17
================================================================================
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_models():
    """Load trained models"""
    print("\n" + "=" * 80)
    print("Step 1: Loading Trained Models")
    print("=" * 80)

    print("  Loading model_x_catboost.pkl...")
    with open('model_x_catboost.pkl', 'rb') as f:
        model_x = pickle.load(f)
    print("    ✓ Loaded")

    print("  Loading model_y_catboost.pkl...")
    with open('model_y_catboost.pkl', 'rb') as f:
        model_y = pickle.load(f)
    print("    ✓ Loaded")

    return model_x, model_y


def load_test_features():
    """Load feature-engineered test data"""
    print("\n" + "=" * 80)
    print("Step 2: Loading Test Features")
    print("=" * 80)

    print("  Loading test_phase1a_plus_features.csv...")
    test_df = pd.read_csv('test_phase1a_plus_features.csv')

    print(f"    Test passes: {len(test_df):,}")
    print(f"    Test episodes: {test_df['game_episode'].nunique()}")
    print(f"    Columns: {len(test_df.columns)}")

    return test_df


def prepare_test_features(test_df):
    """Prepare test features for prediction"""
    print("\n" + "=" * 80)
    print("Step 3: Preparing Test Features")
    print("=" * 80)

    # Identify feature columns (same as training)
    exclude_cols = [
        'game_id', 'game_episode', 'player_id', 'team_id', 'period_id',
        'time_seconds', 'type_name', 'result_name', 'is_home',
        'start_x', 'start_y', 'end_x', 'end_y'
    ]

    # Load CV results to get feature list
    with open('cv_results.json', 'r') as f:
        cv_results = json.load(f)

    feature_cols = cv_results['feature_cols']

    print(f"  Features used in training: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols}")

    # Check if all features exist in test data
    missing_features = [f for f in feature_cols if f not in test_df.columns]
    if missing_features:
        print(f"\n  ERROR: Missing features in test data: {missing_features}")
        raise ValueError("Feature mismatch between train and test")

    print(f"  ✓ All features present in test data")

    # Check for missing values
    missing_counts = test_df[feature_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"  WARNING: Found missing values in {(missing_counts > 0).sum()} features")
        print(f"  Filling with 0...")
        test_df[feature_cols] = test_df[feature_cols].fillna(0)
    else:
        print(f"  ✓ No missing values")

    # Check for infinite values
    inf_counts = np.isinf(test_df[feature_cols]).sum()
    if inf_counts.sum() > 0:
        print(f"  WARNING: Found infinite values in {(inf_counts > 0).sum()} features")
        print(f"  Replacing with 0...")
        test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], 0)
    else:
        print(f"  ✓ No infinite values")

    # IMPORTANT: Use ALL passes for prediction (not just last!)
    # We'll filter to last pass AFTER prediction
    print(f"\n  Using all {len(test_df)} passes for prediction")
    print(f"  Total episodes: {test_df['game_episode'].nunique()}")

    X_test = test_df[feature_cols].values
    game_episodes = test_df['game_episode'].values
    action_ids = test_df['action_id'].values

    print(f"  Test data shape: {X_test.shape}")

    return X_test, game_episodes, action_ids, feature_cols


def predict(model_x, model_y, X_test):
    """Generate predictions"""
    print("\n" + "=" * 80)
    print("Step 4: Generating Predictions")
    print("=" * 80)

    print("  Predicting end_x...")
    pred_x = model_x.predict(X_test)
    print(f"    Mean: {pred_x.mean():.2f}, Std: {pred_x.std():.2f}")
    print(f"    Min: {pred_x.min():.2f}, Max: {pred_x.max():.2f}")

    print("\n  Predicting end_y...")
    pred_y = model_y.predict(X_test)
    print(f"    Mean: {pred_y.mean():.2f}, Std: {pred_y.std():.2f}")
    print(f"    Min: {pred_y.min():.2f}, Max: {pred_y.max():.2f}")

    # Clip to field boundaries
    print("\n  Clipping to field boundaries...")
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)

    print("    ✓ Predictions clipped")

    return pred_x, pred_y


def create_submission(game_episodes, action_ids, pred_x, pred_y):
    """Create submission dataframe - select LAST prediction per episode"""
    print("\n" + "=" * 80)
    print("Step 5: Creating Submission File")
    print("=" * 80)

    # Create dataframe with all predictions
    all_predictions = pd.DataFrame({
        'game_episode': game_episodes,
        'action_id': action_ids,
        'end_x': pred_x,
        'end_y': pred_y
    })

    print(f"  All predictions shape: {all_predictions.shape}")

    # Select LAST prediction for each episode (by action_id)
    print("  Selecting last prediction per episode (max action_id)...")
    submission = all_predictions.loc[
        all_predictions.groupby('game_episode')['action_id'].idxmax()
    ][['game_episode', 'end_x', 'end_y']]

    print(f"  Submission shape: {submission.shape}")
    print(f"  Expected: (2414, 3)")

    # Validate
    assert len(submission) == 2414, f"Expected 2414 rows, got {len(submission)}"
    assert list(submission.columns) == ['game_episode', 'end_x', 'end_y'], "Invalid columns"
    assert submission.isnull().sum().sum() == 0, "Found missing values in submission"

    print("  ✓ Validation passed")

    # Display prediction stats
    print(f"\n  Submission statistics:")
    print(f"    end_x: {submission['end_x'].mean():.2f} ± {submission['end_x'].std():.2f}")
    print(f"    end_y: {submission['end_y'].mean():.2f} ± {submission['end_y'].std():.2f}")

    return submission


def save_submission(submission):
    """Save submission with CV score in filename"""
    print("\n" + "=" * 80)
    print("Step 6: Saving Submission")
    print("=" * 80)

    # Load CV results to get score
    try:
        with open('cv_results.json', 'r') as f:
            results = json.load(f)
        cv_score = results['cv_mean']
    except:
        print("  WARNING: cv_results.json not found, using default filename")
        cv_score = 0.0

    if cv_score > 0:
        filename = f'submission_phase1a_plus_cv{cv_score:.4f}.csv'
    else:
        filename = 'submission_phase1a_plus.csv'

    print(f"  Saving to: {filename}")
    submission.to_csv(filename, index=False)

    print("  ✓ Submission saved")

    return filename


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("exp_035: Generate Submission (CV 10.22!)")
    print("=" * 80)

    # Check prerequisites
    required_files = [
        'model_x_catboost.pkl',
        'model_y_catboost.pkl',
        'test_phase1a_plus_features.csv',
        'cv_results.json'
    ]

    print("\n  Checking prerequisites...")
    for file in required_files:
        if not Path(file).exists():
            print(f"  ERROR: {file} not found!")
            print(f"  Please run extract_phase1a_plus_features.py and train_phase1a_plus_model.py first")
            return

    print("  ✓ All prerequisites found")

    # Step 1: Load models
    model_x, model_y = load_models()

    # Step 2: Load test features
    test_df = load_test_features()

    # Step 3: Prepare test features
    X_test, game_episodes, action_ids, feature_cols = prepare_test_features(test_df)

    # Step 4: Predict
    pred_x, pred_y = predict(model_x, model_y, X_test)

    # Step 5: Create submission
    submission = create_submission(game_episodes, action_ids, pred_x, pred_y)

    # Step 6: Save submission
    filename = save_submission(submission)

    print("\n" + "=" * 80)
    print("Submission Generation Complete!")
    print("=" * 80)
    print(f"  File: {filename}")
    print(f"  Rows: {len(submission)}")
    print(f"  CV Score: 10.22 (Phase1A 15.45 대비 +5.23!)")
    print("\n  Next step:")
    print("    1. Submit to DACON!")
    print("    2. Update SUBMISSION_LOG.md after submission")
    print("=" * 80)


if __name__ == '__main__':
    main()
