"""
exp_128: Pseudo-Labeling for OOD Adaptation
- Train initial model on train data
- Predict pseudo-labels for test data
- Retrain on combined data (train + pseudo-labeled test)
- This helps model adapt to test distribution
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

def create_features(df):
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

def load_test_data():
    test_index = pd.read_csv(DATA_DIR / 'test.csv')
    dfs = []
    for _, row in test_index.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

FEATURES = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
            'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
            'ema_start_y', 'ema_success_rate', 'ema_possession',
            'zone_x', 'result_encoded', 'diff_x', 'velocity', 'ema_momentum_y']

def main():
    print("="*60)
    print("exp_128: Pseudo-Labeling for OOD Adaptation")
    print("="*60)

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = load_test_data()

    train_df = create_features(train_df)

    # For test, we need to handle dx/dy differently
    test_df['dx'] = 0  # placeholder
    test_df['dy'] = 0  # placeholder
    test_df = create_features(test_df)

    train_last = train_df.groupby('game_episode').last().reset_index()
    test_last = test_df.groupby('game_episode').last().reset_index()

    print(f"Train episodes: {len(train_last)}")
    print(f"Test episodes: {len(test_last)}")

    X_train = train_last[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_test = test_last[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values

    # Best params from exp_126
    base_params = {'iterations': 4000, 'depth': 9, 'learning_rate': 0.008,
                   'l2_leaf_reg': 600.0, 'verbose': 0, 'early_stopping_rounds': 100,
                   'loss_function': 'MAE'}

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X_train, y_dx, groups))

    # =====================================================
    # Phase 1: Train initial models and get pseudo-labels
    # =====================================================
    print("\n[Phase 1] Training initial models and generating pseudo-labels...")

    seeds = [42, 123, 456]
    all_pseudo_dx = []
    all_pseudo_dy = []
    baseline_scores = []

    for seed in seeds:
        params = {**base_params, 'random_state': seed}
        fold_preds_dx = []
        fold_preds_dy = []
        fold_scores = []

        for train_idx, val_idx in folds:
            m_dx = CatBoostRegressor(**params)
            m_dy = CatBoostRegressor(**params)
            m_dx.fit(X_train[train_idx], y_dx[train_idx],
                    eval_set=(X_train[val_idx], y_dx[val_idx]), use_best_model=True)
            m_dy.fit(X_train[train_idx], y_dy[train_idx],
                    eval_set=(X_train[val_idx], y_dy[val_idx]), use_best_model=True)

            # Validation score
            pred_dx_val = m_dx.predict(X_train[val_idx])
            pred_dy_val = m_dy.predict(X_train[val_idx])
            dist = np.sqrt((pred_dx_val - y_dx[val_idx])**2 + (pred_dy_val - y_dy[val_idx])**2)
            fold_scores.append(dist.mean())

            # Pseudo-labels for test
            fold_preds_dx.append(m_dx.predict(X_test))
            fold_preds_dy.append(m_dy.predict(X_test))

        cv = np.mean(fold_scores)
        baseline_scores.append(cv)
        print(f"  Seed {seed}: CV {cv:.4f}")

        all_pseudo_dx.append(np.mean(fold_preds_dx, axis=0))
        all_pseudo_dy.append(np.mean(fold_preds_dy, axis=0))

    baseline_cv = np.mean(baseline_scores)
    print(f"\nBaseline CV: {baseline_cv:.4f}")

    # Average pseudo-labels across seeds
    pseudo_dx = np.mean(all_pseudo_dx, axis=0)
    pseudo_dy = np.mean(all_pseudo_dy, axis=0)

    # =====================================================
    # Phase 2: Retrain with pseudo-labeled data
    # =====================================================
    print("\n[Phase 2] Retraining with pseudo-labeled test data...")

    # Create combined dataset
    X_combined = np.vstack([X_train, X_test])
    y_dx_combined = np.concatenate([y_dx, pseudo_dx])
    y_dy_combined = np.concatenate([y_dy, pseudo_dy])

    # Create sample weights (lower weight for pseudo-labels)
    weights = np.ones(len(X_combined))
    weights[len(X_train):] = 0.5  # Test samples have 50% weight

    print(f"Combined dataset: {len(X_combined)} samples")
    print(f"  Original train: {len(X_train)}")
    print(f"  Pseudo-labeled test: {len(X_test)}")

    # Test different pseudo-label weights
    weight_configs = [0.3, 0.5, 0.7, 1.0]
    results = {}

    for pseudo_weight in weight_configs:
        print(f"\nPseudo-label weight = {pseudo_weight}...")
        weights = np.ones(len(X_combined))
        weights[len(X_train):] = pseudo_weight

        scores = []
        for seed in [42, 123, 456]:
            params = {**base_params, 'random_state': seed}
            fold_scores = []

            for train_idx, val_idx in folds:
                # For validation, only use original train data
                # But train on combined with weights

                # Map indices to combined dataset
                combined_train_idx = np.concatenate([
                    train_idx,  # Original train indices
                    np.arange(len(X_train), len(X_combined))  # All test indices
                ])

                m_dx = CatBoostRegressor(**params)
                m_dy = CatBoostRegressor(**params)

                # Train with weights
                m_dx.fit(X_combined[combined_train_idx], y_dx_combined[combined_train_idx],
                        sample_weight=weights[combined_train_idx],
                        eval_set=(X_train[val_idx], y_dx[val_idx]), use_best_model=True)
                m_dy.fit(X_combined[combined_train_idx], y_dy_combined[combined_train_idx],
                        sample_weight=weights[combined_train_idx],
                        eval_set=(X_train[val_idx], y_dy[val_idx]), use_best_model=True)

                # Validate on original train val set
                pred_dx_val = m_dx.predict(X_train[val_idx])
                pred_dy_val = m_dy.predict(X_train[val_idx])
                dist = np.sqrt((pred_dx_val - y_dx[val_idx])**2 + (pred_dy_val - y_dy[val_idx])**2)
                fold_scores.append(dist.mean())

            scores.append(np.mean(fold_scores))

        cv = np.mean(scores)
        std = np.std(scores)
        results[pseudo_weight] = (cv, std)
        print(f"  Weight {pseudo_weight}: CV {cv:.4f} (+/- {std:.4f})")

    # =====================================================
    # Results Summary
    # =====================================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline (no pseudo-labels): CV {baseline_cv:.4f}")
    print()

    best_weight = min(results, key=lambda k: results[k][0])
    for w in sorted(results.keys()):
        cv, std = results[w]
        diff = cv - baseline_cv
        marker = " <-- BEST" if w == best_weight else ""
        print(f"  Weight {w}: CV {cv:.4f} ({diff:+.4f}){marker}")

if __name__ == "__main__":
    main()
