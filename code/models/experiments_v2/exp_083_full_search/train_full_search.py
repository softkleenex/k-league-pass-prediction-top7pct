"""
exp_083: Full Hyperparameter Search
- All fold counts: 5, 7, 10, 12, 15
- All seed counts: 1, 3, 5, 7
- Learning rates: 0.01, 0.02, 0.03, 0.05
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import gc
import warnings
import itertools
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"

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

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999]

def run_experiment(X, y_delta, y_abs, start_xy, groups, n_splits, lr, n_seeds):
    """Run experiment with given settings"""
    seeds = SEED_POOL[:n_seeds]
    all_oof = []

    for seed in seeds:
        iterations = max(500, int(1000 * 0.05 / lr))
        params = {
            'iterations': iterations,
            'depth': 8,
            'learning_rate': lr,
            'l2_leaf_reg': 3.0,
            'random_state': seed,
            'verbose': 0,
            'early_stopping_rounds': 50,
            'loss_function': 'MAE'
        }

        gkf = GroupKFold(n_splits=n_splits)
        oof_delta = np.zeros((len(X), 2))

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_delta, groups), 1):
            model_dx = CatBoostRegressor(**params)
            model_dy = CatBoostRegressor(**params)
            model_dx.fit(X[train_idx], y_delta[train_idx, 0],
                        eval_set=(X[val_idx], y_delta[val_idx, 0]), use_best_model=True)
            model_dy.fit(X[train_idx], y_delta[train_idx, 1],
                        eval_set=(X[val_idx], y_delta[val_idx, 1]), use_best_model=True)
            oof_delta[val_idx, 0] = model_dx.predict(X[val_idx])
            oof_delta[val_idx, 1] = model_dy.predict(X[val_idx])
            del model_dx, model_dy

        pred_abs = np.zeros((len(X), 2))
        pred_abs[:, 0] = start_xy[:, 0] + oof_delta[:, 0]
        pred_abs[:, 1] = start_xy[:, 1] + oof_delta[:, 1]
        all_oof.append(pred_abs.copy())
        gc.collect()

    # Ensemble if multiple seeds
    if len(seeds) > 1:
        ensemble_pred = np.mean(all_oof, axis=0)
    else:
        ensemble_pred = all_oof[0]

    cv = np.sqrt((ensemble_pred[:, 0] - y_abs[:, 0])**2 + (ensemble_pred[:, 1] - y_abs[:, 1])**2).mean()
    return cv

def main():
    print("=" * 70)
    print("exp_083: Full Hyperparameter Search")
    print("=" * 70)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    del train_df; gc.collect()

    TOP_15 = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
              'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
              'ema_start_y', 'ema_success_rate', 'ema_possession',
              'zone_x', 'result_encoded', 'diff_x', 'velocity']

    X = last_passes[TOP_15].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = last_passes[['dx', 'dy']].values.astype(np.float32)
    y_abs = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    start_xy = last_passes[['start_x', 'start_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    results = {}

    # Phase 1: Seed count optimization (fix fold=10, lr=0.05)
    print("\n" + "=" * 70)
    print("[Phase 1] Seed Count Optimization (fold=10, lr=0.05)")
    print("=" * 70)
    for n_seeds in [1, 3, 5, 7]:
        name = f"fold10_lr0.05_seed{n_seeds}"
        print(f"\n  Testing {n_seeds} seeds...")
        cv = run_experiment(X, y_delta, y_abs, start_xy, groups,
                           n_splits=10, lr=0.05, n_seeds=n_seeds)
        results[name] = cv
        print(f"  {name}: CV {cv:.4f}")

    # Phase 2: Fold count optimization (fix seeds=best, lr=0.05)
    best_seeds = 3  # We know 3 seeds worked well
    print("\n" + "=" * 70)
    print(f"[Phase 2] Fold Count Optimization (seeds={best_seeds}, lr=0.05)")
    print("=" * 70)
    for n_folds in [5, 7, 10, 12, 15]:
        name = f"fold{n_folds}_lr0.05_seed{best_seeds}"
        if name in results:
            continue
        print(f"\n  Testing {n_folds} folds...")
        cv = run_experiment(X, y_delta, y_abs, start_xy, groups,
                           n_splits=n_folds, lr=0.05, n_seeds=best_seeds)
        results[name] = cv
        print(f"  {name}: CV {cv:.4f}")

    # Phase 3: Learning rate optimization (fix fold=best, seeds=best)
    # Find best fold from phase 2
    fold_results = {k: v for k, v in results.items() if f'seed{best_seeds}' in k}
    best_fold_key = min(fold_results, key=fold_results.get)
    best_fold = int(best_fold_key.split('_')[0].replace('fold', ''))

    print("\n" + "=" * 70)
    print(f"[Phase 3] Learning Rate Optimization (fold={best_fold}, seeds={best_seeds})")
    print("=" * 70)
    for lr in [0.01, 0.02, 0.03, 0.07]:
        name = f"fold{best_fold}_lr{lr}_seed{best_seeds}"
        print(f"\n  Testing lr={lr}...")
        cv = run_experiment(X, y_delta, y_abs, start_xy, groups,
                           n_splits=best_fold, lr=lr, n_seeds=best_seeds)
        results[name] = cv
        print(f"  {name}: CV {cv:.4f}")

    # Phase 4: Fine-tuning around best
    best_key = min(results, key=results.get)
    best_cv = results[best_key]

    # Parse best settings
    parts = best_key.split('_')
    opt_fold = int(parts[0].replace('fold', ''))
    opt_lr = float(parts[1].replace('lr', ''))
    opt_seeds = int(parts[2].replace('seed', ''))

    print("\n" + "=" * 70)
    print(f"[Phase 4] Fine-tuning around best: fold={opt_fold}, lr={opt_lr}, seeds={opt_seeds}")
    print("=" * 70)

    # Try adjacent values
    fine_tune_configs = [
        (opt_fold - 1, opt_lr, opt_seeds) if opt_fold > 3 else None,
        (opt_fold + 1, opt_lr, opt_seeds),
        (opt_fold, opt_lr, opt_seeds + 2) if opt_seeds < 7 else None,
    ]

    for config in fine_tune_configs:
        if config is None:
            continue
        n_fold, lr, n_seeds = config
        if n_seeds > 7:
            continue
        name = f"fold{n_fold}_lr{lr}_seed{n_seeds}"
        if name in results:
            continue
        print(f"\n  Testing fold={n_fold}, lr={lr}, seeds={n_seeds}...")
        cv = run_experiment(X, y_delta, y_abs, start_xy, groups,
                           n_splits=n_fold, lr=lr, n_seeds=n_seeds)
        results[name] = cv
        print(f"  {name}: CV {cv:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Top 15, sorted by CV)")
    print("=" * 70)
    baseline_cv = 13.5694  # Current best
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for i, (name, cv) in enumerate(sorted_results[:15], 1):
        diff = cv - baseline_cv
        marker = " ★★★" if i == 1 else (" ★★" if i <= 3 else (" ★" if i <= 5 else ""))
        print(f"  {i:2d}. {name:35s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    best_name = sorted_results[0][0]
    best_cv = sorted_results[0][1]
    print(f"\n  BEST: {best_name}")
    print(f"  CV: {best_cv:.4f}")
    print(f"  vs previous best (13.5694): {best_cv - 13.5694:+.4f}")
    print("=" * 70)

    # Create submission for best
    if best_cv < 13.5694:
        print(f"\n[Creating submission for {best_name}...]")
        create_best_submission(best_name, best_cv, TOP_15, last_passes)

    # Also create for top 3 if different
    for name, cv in sorted_results[:3]:
        if cv < 13.57:
            print(f"\n[Creating submission for {name}...]")
            create_submission(name, cv, TOP_15, last_passes)

def create_submission(name, cv, feature_cols, last_passes):
    """Create submission with given settings"""
    parts = name.split('_')
    n_splits = int(parts[0].replace('fold', ''))
    lr = float(parts[1].replace('lr', ''))
    n_seeds = int(parts[2].replace('seed', ''))
    seeds = SEED_POOL[:n_seeds]

    X = last_passes[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_delta = last_passes[['dx', 'dy']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    all_models_dx = []
    all_models_dy = []

    for seed in seeds:
        iterations = max(500, int(1000 * 0.05 / lr))
        params = {
            'iterations': iterations,
            'depth': 8,
            'learning_rate': lr,
            'l2_leaf_reg': 3.0,
            'random_state': seed,
            'verbose': 0,
            'early_stopping_rounds': 50,
            'loss_function': 'MAE'
        }

        gkf = GroupKFold(n_splits=n_splits)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_delta, groups), 1):
            model_dx = CatBoostRegressor(**params)
            model_dy = CatBoostRegressor(**params)
            model_dx.fit(X[train_idx], y_delta[train_idx, 0],
                        eval_set=(X[val_idx], y_delta[val_idx, 0]), use_best_model=True)
            model_dy.fit(X[train_idx], y_delta[train_idx, 1],
                        eval_set=(X[val_idx], y_delta[val_idx, 1]), use_best_model=True)
            all_models_dx.append(model_dx)
            all_models_dy.append(model_dy)

    # Test prediction
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    test_episodes = []
    for _, row in test_df.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
        ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']
        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)
    test_all = pd.concat(test_episodes, ignore_index=True)
    test_all = create_features(test_all)
    test_last = test_all.groupby('game_episode').last().reset_index()

    X_test = test_last[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    pred_dx = np.mean([m.predict(X_test) for m in all_models_dx], axis=0)
    pred_dy = np.mean([m.predict(X_test) for m in all_models_dy], axis=0)
    pred_x = test_last['start_x'].values + pred_dx
    pred_y = np.clip(test_last['start_y'].values + pred_dy, 0, 68)

    submission = pd.DataFrame({'game_episode': test_last['game_episode'], 'end_x': pred_x, 'end_y': pred_y})
    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_{name}_cv{cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

def create_best_submission(name, cv, feature_cols, last_passes):
    create_submission(name, cv, feature_cols, last_passes)

if __name__ == "__main__":
    main()
