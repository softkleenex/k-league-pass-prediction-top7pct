"""
exp_055: Stacking Ensemble

1st level: CatBoost, LightGBM, XGBoost
2nd level: Ridge Regression

참고: Kaggle 상위권에서 자주 사용되는 기법
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"


def create_all_features(df):
    """모든 피처 생성"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)

    df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

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

    ema_span = 2
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['move_distance'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)

    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])

    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    df['ema_goal_distance'] = df.groupby('game_episode')['goal_distance'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['goal_distance'])

    df['ema_distance'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0)

    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['diff_y'] = df.groupby('game_episode')['start_y'].diff().fillna(0)
    df['diff_goal_dist'] = df.groupby('game_episode')['goal_distance'].diff().fillna(0)

    df['rolling_std_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['rolling_std_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['rolling_std_dist'] = df.groupby('game_episode')['move_distance'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    ).fillna(0)

    df['velocity'] = df['move_distance']
    df['acceleration'] = df.groupby('game_episode')['velocity'].diff().fillna(0)

    df['rel_x_from_ema'] = df['start_x'] - df['ema_start_x']
    df['rel_y_from_ema'] = df['start_y'] - df['ema_start_y']

    df['episode_start_x'] = df.groupby('game_episode')['start_x'].transform('first')
    df['episode_start_y'] = df.groupby('game_episode')['start_y'].transform('first')
    df['progress_from_start_x'] = df['start_x'] - df['episode_start_x']
    df['progress_from_start_y'] = df['start_y'] - df['episode_start_y']

    df['x_times_direction'] = df['start_x'] * df['direction']
    df['goal_dist_times_velocity'] = df['goal_distance'] * df['velocity']
    df['success_times_possession'] = df['ema_success_rate'] * df['ema_possession']

    df['is_forward'] = (df['diff_x'] > 0).astype(int)
    df['forward_streak'] = df.groupby('game_episode')['is_forward'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df['avg_forward_x'] = df.groupby('game_episode')['diff_x'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ).fillna(0)

    return df


def main():
    print("=" * 70)
    print("exp_055: Stacking Ensemble")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_all_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    features = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct',
        'ema_start_x', 'ema_start_y', 'ema_goal_distance',
        'ema_distance', 'ema_success_rate', 'ema_possession',
        'dist_to_goal_line', 'dist_to_sideline', 'dist_to_center_y', 'position_zone',
        'diff_x', 'diff_y', 'diff_goal_dist',
        'rolling_std_x', 'rolling_std_y', 'rolling_std_dist',
        'velocity', 'acceleration',
        'rel_x_from_ema', 'rel_y_from_ema', 'progress_from_start_x', 'progress_from_start_y',
        'x_times_direction', 'goal_dist_times_velocity', 'success_times_possession',
        'forward_streak', 'avg_forward_x'
    ]

    X = last_passes[features].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # Stacking: OOF predictions
    print("\n[2] Level 1: Base models OOF predictions...")

    n_samples = len(X)
    gkf = GroupKFold(n_splits=3)

    # OOF arrays
    oof_cat_x = np.zeros(n_samples)
    oof_cat_y = np.zeros(n_samples)
    oof_lgb_x = np.zeros(n_samples)
    oof_lgb_y = np.zeros(n_samples)
    oof_xgb_x = np.zeros(n_samples)
    oof_xgb_y = np.zeros(n_samples)

    # 저장할 모델
    cat_models = []
    lgb_models = []
    xgb_models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n  Fold {fold}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # CatBoost
        cat_x = CatBoostRegressor(iterations=1000, depth=8, learning_rate=0.05,
                                   l2_leaf_reg=3.0, random_state=42, verbose=0,
                                   early_stopping_rounds=50)
        cat_y = CatBoostRegressor(iterations=1000, depth=8, learning_rate=0.05,
                                   l2_leaf_reg=3.0, random_state=42, verbose=0,
                                   early_stopping_rounds=50)
        cat_x.fit(X_train, y_train[:, 0], eval_set=(X_val, y_val[:, 0]), use_best_model=True)
        cat_y.fit(X_train, y_train[:, 1], eval_set=(X_val, y_val[:, 1]), use_best_model=True)
        oof_cat_x[val_idx] = cat_x.predict(X_val)
        oof_cat_y[val_idx] = cat_y.predict(X_val)
        cat_models.append((cat_x, cat_y))

        # LightGBM
        lgb_x = lgb.LGBMRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05,
                                   reg_lambda=3.0, random_state=42, verbose=-1)
        lgb_y = lgb.LGBMRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05,
                                   reg_lambda=3.0, random_state=42, verbose=-1)
        lgb_x.fit(X_train, y_train[:, 0], eval_set=[(X_val, y_val[:, 0])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_y.fit(X_train, y_train[:, 1], eval_set=[(X_val, y_val[:, 1])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb_x[val_idx] = lgb_x.predict(X_val)
        oof_lgb_y[val_idx] = lgb_y.predict(X_val)
        lgb_models.append((lgb_x, lgb_y))

        # XGBoost
        xgb_x = xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05,
                                  reg_lambda=3.0, random_state=42, verbosity=0,
                                  early_stopping_rounds=50)
        xgb_y = xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05,
                                  reg_lambda=3.0, random_state=42, verbosity=0,
                                  early_stopping_rounds=50)
        xgb_x.fit(X_train, y_train[:, 0], eval_set=[(X_val, y_val[:, 0])], verbose=False)
        xgb_y.fit(X_train, y_train[:, 1], eval_set=[(X_val, y_val[:, 1])], verbose=False)
        oof_xgb_x[val_idx] = xgb_x.predict(X_val)
        oof_xgb_y[val_idx] = xgb_y.predict(X_val)
        xgb_models.append((xgb_x, xgb_y))

    # Level 1 scores
    print("\n[3] Level 1 CV scores...")
    err_cat = np.sqrt((oof_cat_x - y[:, 0])**2 + (oof_cat_y - y[:, 1])**2).mean()
    err_lgb = np.sqrt((oof_lgb_x - y[:, 0])**2 + (oof_lgb_y - y[:, 1])**2).mean()
    err_xgb = np.sqrt((oof_xgb_x - y[:, 0])**2 + (oof_xgb_y - y[:, 1])**2).mean()
    print(f"  CatBoost: {err_cat:.4f}")
    print(f"  LightGBM: {err_lgb:.4f}")
    print(f"  XGBoost:  {err_xgb:.4f}")

    # Level 2: Stacking
    print("\n[4] Level 2: Stacking with Ridge...")

    # Meta features
    meta_X = np.column_stack([oof_cat_x, oof_cat_y, oof_lgb_x, oof_lgb_y, oof_xgb_x, oof_xgb_y])

    # Ridge CV
    oof_stack_x = np.zeros(n_samples)
    oof_stack_y = np.zeros(n_samples)
    ridge_models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(meta_X, y, groups), 1):
        ridge_x = Ridge(alpha=1.0)
        ridge_y = Ridge(alpha=1.0)
        ridge_x.fit(meta_X[train_idx], y[train_idx, 0])
        ridge_y.fit(meta_X[train_idx], y[train_idx, 1])
        oof_stack_x[val_idx] = ridge_x.predict(meta_X[val_idx])
        oof_stack_y[val_idx] = ridge_y.predict(meta_X[val_idx])
        ridge_models.append((ridge_x, ridge_y))
        print(f"    Fold {fold} Ridge fitted")

    err_stack = np.sqrt((oof_stack_x - y[:, 0])**2 + (oof_stack_y - y[:, 1])**2).mean()
    print(f"\n  Stacking CV: {err_stack:.4f}")

    # Simple average
    oof_avg_x = (oof_cat_x + oof_lgb_x + oof_xgb_x) / 3
    oof_avg_y = (oof_cat_y + oof_lgb_y + oof_xgb_y) / 3
    err_avg = np.sqrt((oof_avg_x - y[:, 0])**2 + (oof_avg_y - y[:, 1])**2).mean()
    print(f"  Simple Avg: {err_avg:.4f}")

    # Cat-heavy weighted average
    oof_wt_x = 0.6 * oof_cat_x + 0.2 * oof_lgb_x + 0.2 * oof_xgb_x
    oof_wt_y = 0.6 * oof_cat_y + 0.2 * oof_lgb_y + 0.2 * oof_xgb_y
    err_wt = np.sqrt((oof_wt_x - y[:, 0])**2 + (oof_wt_y - y[:, 1])**2).mean()
    print(f"  Weighted (0.6/0.2/0.2): {err_wt:.4f}")

    # 결과 비교
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)
    results = {
        'CatBoost only': err_cat,
        'LightGBM only': err_lgb,
        'XGBoost only': err_xgb,
        'Simple Avg': err_avg,
        'Weighted (0.6/0.2/0.2)': err_wt,
        'Stacking': err_stack
    }
    best_name = min(results, key=results.get)
    best_cv = min(results.values())

    for name, cv in sorted(results.items(), key=lambda x: x[1]):
        marker = " ***BEST***" if cv == best_cv else ""
        print(f"  {name:25s}: CV {cv:.4f}{marker}")

    # Test 예측
    print(f"\n[5] {best_name}으로 Test 예측...")

    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    test_episodes = []
    for _, row in test_df.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        if 'dx' not in ep_df.columns:
            ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
            ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']
        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)

    test_all = pd.concat(test_episodes, ignore_index=True)
    test_all = create_all_features(test_all)
    test_last = test_all.groupby('game_episode').last().reset_index()

    X_test = test_last[features].fillna(0).replace([np.inf, -np.inf], 0).values

    # Test predictions from all models
    test_cat_x = np.zeros(len(X_test))
    test_cat_y = np.zeros(len(X_test))
    test_lgb_x = np.zeros(len(X_test))
    test_lgb_y = np.zeros(len(X_test))
    test_xgb_x = np.zeros(len(X_test))
    test_xgb_y = np.zeros(len(X_test))

    for mx, my in cat_models:
        test_cat_x += mx.predict(X_test) / len(cat_models)
        test_cat_y += my.predict(X_test) / len(cat_models)

    for mx, my in lgb_models:
        test_lgb_x += mx.predict(X_test) / len(lgb_models)
        test_lgb_y += my.predict(X_test) / len(lgb_models)

    for mx, my in xgb_models:
        test_xgb_x += mx.predict(X_test) / len(xgb_models)
        test_xgb_y += my.predict(X_test) / len(xgb_models)

    # Best prediction
    if best_name == 'CatBoost only':
        pred_x = test_cat_x
        pred_y = test_cat_y
    elif best_name == 'LightGBM only':
        pred_x = test_lgb_x
        pred_y = test_lgb_y
    elif best_name == 'XGBoost only':
        pred_x = test_xgb_x
        pred_y = test_xgb_y
    elif best_name == 'Simple Avg':
        pred_x = (test_cat_x + test_lgb_x + test_xgb_x) / 3
        pred_y = (test_cat_y + test_lgb_y + test_xgb_y) / 3
    elif best_name == 'Weighted (0.6/0.2/0.2)':
        pred_x = 0.6 * test_cat_x + 0.2 * test_lgb_x + 0.2 * test_xgb_x
        pred_y = 0.6 * test_cat_y + 0.2 * test_lgb_y + 0.2 * test_xgb_y
    else:  # Stacking
        meta_test = np.column_stack([test_cat_x, test_cat_y, test_lgb_x, test_lgb_y,
                                      test_xgb_x, test_xgb_y])
        pred_x = np.zeros(len(X_test))
        pred_y = np.zeros(len(X_test))
        for rx, ry in ridge_models:
            pred_x += rx.predict(meta_test) / len(ridge_models)
            pred_y += ry.predict(meta_test) / len(ridge_models)

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })

    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_stacking_cv{best_cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  저장: {output_path}")

    print("\n" + "=" * 70)
    print(f"Best: {best_name}")
    print(f"CV: {best_cv:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
