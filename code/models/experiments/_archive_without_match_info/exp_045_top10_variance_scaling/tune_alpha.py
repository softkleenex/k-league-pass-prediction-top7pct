"""
exp_045: Alpha 미세 조정 실험

기존 최적: alpha=0.25, CV 15.1802, Public 15.2873
더 세밀하게 탐색

작성일: 2025-12-24
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


def create_phase1a_features(df):
    """Phase1A 피처 생성"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)

    df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

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

    return df


def scale_predictions(pred, y_mean, y_std, pred_mean, pred_std, alpha=0.25):
    standardized = (pred - pred_mean) / (pred_std + 1e-8)
    target_std = pred_std + alpha * (y_std - pred_std)
    scaled = standardized * target_std + y_mean
    return scaled


def main():
    print("=" * 60)
    print("Alpha 미세 조정 실험")
    print("=" * 60)

    # Load data
    print("\n[1] 데이터 로딩...")
    train_df = pd.read_csv('../../../../data/train.csv')
    train_df = create_phase1a_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    # Top 10 features
    TOP_10_FEATURES = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    X = last_passes[TOP_10_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    y_mean_x, y_std_x = y[:, 0].mean(), y[:, 0].std()
    y_mean_y, y_std_y = y[:, 1].mean(), y[:, 1].std()

    # Fine-grained alpha search
    print("\n[2] Alpha 미세 탐색...")
    alphas = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]

    gkf = GroupKFold(n_splits=3)
    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

    results = {}

    for alpha in alphas:
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            model_x = CatBoostRegressor(**params)
            model_y = CatBoostRegressor(**params)

            model_x.fit(X[train_idx], y[train_idx, 0])
            model_y.fit(X[train_idx], y[train_idx, 1])

            pred_x = model_x.predict(X[val_idx])
            pred_y = model_y.predict(X[val_idx])

            if alpha > 0:
                pred_mean_x, pred_std_x = pred_x.mean(), pred_x.std()
                pred_mean_y, pred_std_y = pred_y.mean(), pred_y.std()

                pred_x_scaled = scale_predictions(pred_x, y_mean_x, y_std_x,
                                                   pred_mean_x, pred_std_x, alpha)
                pred_y_scaled = scale_predictions(pred_y, y_mean_y, y_std_y,
                                                   pred_mean_y, pred_std_y, alpha)

                pred_x_scaled = np.clip(pred_x_scaled, 0, 105)
                pred_y_scaled = np.clip(pred_y_scaled, 0, 68)
            else:
                pred_x_scaled = pred_x
                pred_y_scaled = pred_y

            errors = np.sqrt((pred_x_scaled - y[val_idx, 0])**2 +
                            (pred_y_scaled - y[val_idx, 1])**2)
            fold_scores.append(errors.mean())

        cv_mean = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        results[alpha] = {'mean': cv_mean, 'std': cv_std}

        marker = " <-- BEST" if alpha == 0.25 else ""
        print(f"  alpha={alpha:.2f}: CV {cv_mean:.4f} +/- {cv_std:.4f}{marker}")

    # Find best
    best_alpha = min(results.keys(), key=lambda k: results[k]['mean'])
    best_cv = results[best_alpha]['mean']

    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"  Best alpha: {best_alpha}")
    print(f"  Best CV: {best_cv:.4f}")
    print(f"  기존 alpha=0.25 CV: {results[0.25]['mean']:.4f}")

    if best_alpha != 0.25:
        improvement = results[0.25]['mean'] - best_cv
        print(f"  개선: {improvement:+.4f}")
        print(f"\n  -> alpha={best_alpha}로 제출 파일 생성 권장!")
    else:
        print(f"\n  -> alpha=0.25가 여전히 최적!")

    print("=" * 60)

    return best_alpha, results


if __name__ == '__main__':
    main()
