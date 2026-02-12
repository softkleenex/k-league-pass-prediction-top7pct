"""
exp_071: MAE + RMSE Ensemble
- MAE와 RMSE 모델의 앙상블
- 서로 다른 관점 (outlier sensitivity)
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
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)

    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    return df


def main():
    print("=" * 70)
    print("exp_071: MAE + RMSE Ensemble")
    print("=" * 70)

    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    TOP_12 = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession', 'velocity'
    ]

    X = last_passes[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # MAE 파라미터
    params_mae = {
        'iterations': 4000, 'depth': 8, 'learning_rate': 0.01,
        'l2_leaf_reg': 7.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 100, 'loss_function': 'MAE'
    }

    # RMSE 파라미터
    params_rmse = {
        'iterations': 4000, 'depth': 8, 'learning_rate': 0.01,
        'l2_leaf_reg': 7.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 100, 'loss_function': 'RMSE'
    }

    gkf = GroupKFold(n_splits=5)

    # 다양한 앙상블 비율 테스트
    ratios = [0.0, 0.3, 0.5, 0.7, 1.0]  # MAE 비율
    results = {}

    for mae_ratio in ratios:
        print(f"\n[MAE:{mae_ratio:.1f}, RMSE:{1-mae_ratio:.1f}]")
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            # MAE 모델
            mae_x = CatBoostRegressor(**params_mae)
            mae_y = CatBoostRegressor(**params_mae)
            mae_x.fit(X[train_idx], y[train_idx, 0],
                     eval_set=(X[val_idx], y[val_idx, 0]), use_best_model=True)
            mae_y.fit(X[train_idx], y[train_idx, 1],
                     eval_set=(X[val_idx], y[val_idx, 1]), use_best_model=True)

            # RMSE 모델
            rmse_x = CatBoostRegressor(**params_rmse)
            rmse_y = CatBoostRegressor(**params_rmse)
            rmse_x.fit(X[train_idx], y[train_idx, 0],
                      eval_set=(X[val_idx], y[val_idx, 0]), use_best_model=True)
            rmse_y.fit(X[train_idx], y[train_idx, 1],
                      eval_set=(X[val_idx], y[val_idx, 1]), use_best_model=True)

            # 앙상블 예측
            pred_x = mae_ratio * mae_x.predict(X[val_idx]) + (1 - mae_ratio) * rmse_x.predict(X[val_idx])
            pred_y = mae_ratio * mae_y.predict(X[val_idx]) + (1 - mae_ratio) * rmse_y.predict(X[val_idx])

            errors = np.sqrt((pred_x - y[val_idx, 0])**2 + (pred_y - y[val_idx, 1])**2)
            fold_scores.append(errors.mean())
            print(f"  Fold {fold}: {fold_scores[-1]:.4f}")

        cv = np.mean(fold_scores)
        results[mae_ratio] = cv
        print(f"  CV: {cv:.4f}")

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    best_ratio = min(results, key=results.get)
    for ratio, cv in sorted(results.items()):
        marker = " ★ BEST" if ratio == best_ratio else ""
        print(f"  MAE:{ratio:.1f}/RMSE:{1-ratio:.1f} → CV {cv:.4f}{marker}")

    print(f"\n최적 비율: MAE {best_ratio:.1f} / RMSE {1-best_ratio:.1f}")
    print(f"최적 CV: {results[best_ratio]:.4f}")


if __name__ == "__main__":
    main()
