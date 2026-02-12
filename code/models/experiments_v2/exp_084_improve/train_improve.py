"""
exp_084: Improve beyond 13.54
- More seeds (9, 10)
- Different folds (8, 12, 13)
- LR fine-tuning around 0.05
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

SEED_POOL = [42, 123, 456, 789, 2024, 777, 999, 1234, 5678, 9999]

def run_experiment(X, y_delta, y_abs, start_xy, groups, n_splits, lr, n_seeds):
    """Run experiment with given settings"""
    seeds = SEED_POOL[:n_seeds]
    all_oof = []

    for seed in seeds:
        params = {
            'iterations': 1000,
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

    ensemble_pred = np.mean(all_oof, axis=0)
    cv = np.sqrt((ensemble_pred[:, 0] - y_abs[:, 0])**2 + (ensemble_pred[:, 1] - y_abs[:, 1])**2).mean()
    return cv

def main():
    print("=" * 70)
    print("exp_084: Improve beyond 13.54")
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
    baseline_cv = 13.5435  # Current best (fold11, lr0.05, 7seeds)

    # Test 1: More seeds with fold=11
    print("\n[1] More seeds (fold=11, lr=0.05)")
    for n_seeds in [8, 9, 10]:
        name = f"fold11_lr0.05_seed{n_seeds}"
        print(f"  Testing {n_seeds} seeds...")
        cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, 0.05, n_seeds)
        results[name] = cv
        print(f"  {name}: CV {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Test 2: Different folds with 7 seeds
    print("\n[2] Different folds (7 seeds, lr=0.05)")
    for n_folds in [8, 12, 13, 14]:
        name = f"fold{n_folds}_lr0.05_seed7"
        print(f"  Testing {n_folds} folds...")
        cv = run_experiment(X, y_delta, y_abs, start_xy, groups, n_folds, 0.05, 7)
        results[name] = cv
        print(f"  {name}: CV {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Test 3: LR fine-tuning with best fold/seeds
    print("\n[3] LR fine-tuning (fold=11, 7 seeds)")
    for lr in [0.04, 0.045, 0.055, 0.06]:
        name = f"fold11_lr{lr}_seed7"
        print(f"  Testing lr={lr}...")
        cv = run_experiment(X, y_delta, y_abs, start_xy, groups, 11, lr, 7)
        results[name] = cv
        print(f"  {name}: CV {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Test 4: Best combinations
    print("\n[4] Best combinations")
    # Find best from each category
    best_seeds = min([k for k in results if 'fold11_lr0.05' in k], key=lambda k: results[k])
    best_folds = min([k for k in results if 'seed7' in k and 'lr0.05' in k], key=lambda k: results[k])

    # Try combining best settings
    if 'seed' in best_seeds:
        best_n_seeds = int(best_seeds.split('seed')[-1])
    else:
        best_n_seeds = 7
    if 'fold' in best_folds:
        best_n_folds = int(best_folds.split('_')[0].replace('fold', ''))
    else:
        best_n_folds = 11

    if best_n_folds != 11 or best_n_seeds != 7:
        name = f"fold{best_n_folds}_lr0.05_seed{best_n_seeds}"
        if name not in results:
            print(f"  Testing combination: {name}...")
            cv = run_experiment(X, y_delta, y_abs, start_xy, groups, best_n_folds, 0.05, best_n_seeds)
            results[name] = cv
            print(f"  {name}: CV {cv:.4f} ({cv - baseline_cv:+.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS (sorted by CV)")
    print("=" * 70)
    print(f"  Baseline (fold11_lr0.05_seed7): CV {baseline_cv:.4f}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for name, cv in sorted_results:
        diff = cv - baseline_cv
        marker = " ★★★" if diff < -0.01 else (" ★★" if diff < 0 else (" ★" if diff < 0.005 else ""))
        print(f"  {name:30s}: CV {cv:.4f} ({diff:+.4f}){marker}")

    best_name = sorted_results[0][0]
    best_cv = sorted_results[0][1]

    print("\n" + "=" * 70)
    if best_cv < baseline_cv:
        print(f"  NEW BEST: {best_name}")
        print(f"  CV: {best_cv:.4f} (vs baseline: {best_cv - baseline_cv:+.4f})")
        print("  Creating submission...")
        create_submission(best_name, best_cv, TOP_15, last_passes)
    else:
        print(f"  No improvement over baseline ({baseline_cv:.4f})")
        print(f"  Best tried: {best_name} (CV {best_cv:.4f})")
    print("=" * 70)

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
        params = {
            'iterations': 1000,
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

if __name__ == "__main__":
    main()
