"""
exp_073: Diverse Model Ensemble
- CatBoost MAE (Best) + LightGBM + TabNet
- 다양한 모델 관점 조합
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetRegressor
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
    print("exp_073: Diverse Model Ensemble")
    print("=" * 70)

    # 데이터 로드
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    TOP_12 = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession', 'velocity'
    ]

    X = last_passes[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    # 파라미터
    cat_params = {
        'iterations': 4000, 'depth': 8, 'learning_rate': 0.01,
        'l2_leaf_reg': 7.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 100, 'loss_function': 'MAE'
    }

    lgb_params = {
        'objective': 'mae', 'metric': 'mae', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.01, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l2': 7.0,
        'n_estimators': 4000, 'early_stopping_rounds': 100, 'verbose': -1, 'random_state': 42
    }

    gkf = GroupKFold(n_splits=5)

    # 각 모델 OOF 예측 저장
    oof_cat = np.zeros((len(X), 2))
    oof_lgb = np.zeros((len(X), 2))
    oof_tab = np.zeros((len(X), 2))

    print("\n[1] 각 모델 CV 계산...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n  Fold {fold}:")

        # CatBoost
        cat_x = CatBoostRegressor(**cat_params)
        cat_y = CatBoostRegressor(**cat_params)
        cat_x.fit(X[train_idx], y[train_idx, 0], eval_set=(X[val_idx], y[val_idx, 0]), use_best_model=True)
        cat_y.fit(X[train_idx], y[train_idx, 1], eval_set=(X[val_idx], y[val_idx, 1]), use_best_model=True)
        oof_cat[val_idx, 0] = cat_x.predict(X[val_idx])
        oof_cat[val_idx, 1] = cat_y.predict(X[val_idx])
        cat_err = np.sqrt((oof_cat[val_idx, 0] - y[val_idx, 0])**2 + (oof_cat[val_idx, 1] - y[val_idx, 1])**2).mean()
        print(f"    CatBoost: {cat_err:.4f}")

        # LightGBM
        lgb_x = lgb.LGBMRegressor(**lgb_params)
        lgb_y = lgb.LGBMRegressor(**lgb_params)
        lgb_x.fit(X[train_idx], y[train_idx, 0], eval_set=[(X[val_idx], y[val_idx, 0])])
        lgb_y.fit(X[train_idx], y[train_idx, 1], eval_set=[(X[val_idx], y[val_idx, 1])])
        oof_lgb[val_idx, 0] = lgb_x.predict(X[val_idx])
        oof_lgb[val_idx, 1] = lgb_y.predict(X[val_idx])
        lgb_err = np.sqrt((oof_lgb[val_idx, 0] - y[val_idx, 0])**2 + (oof_lgb[val_idx, 1] - y[val_idx, 1])**2).mean()
        print(f"    LightGBM: {lgb_err:.4f}")

        # TabNet
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_idx])
        X_val_s = scaler.transform(X[val_idx])

        tab = TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.5, lambda_sparse=1e-4,
                              optimizer_params={'lr': 0.02}, verbose=0)
        tab.fit(X_train_s, y[train_idx], eval_set=[(X_val_s, y[val_idx])],
                eval_metric=['mae'], max_epochs=200, patience=30, batch_size=256)
        pred_tab = tab.predict(X_val_s)
        oof_tab[val_idx, 0] = pred_tab[:, 0]
        oof_tab[val_idx, 1] = pred_tab[:, 1]
        tab_err = np.sqrt((oof_tab[val_idx, 0] - y[val_idx, 0])**2 + (oof_tab[val_idx, 1] - y[val_idx, 1])**2).mean()
        print(f"    TabNet: {tab_err:.4f}")

    # 개별 모델 전체 CV
    cv_cat = np.sqrt((oof_cat[:, 0] - y[:, 0])**2 + (oof_cat[:, 1] - y[:, 1])**2).mean()
    cv_lgb = np.sqrt((oof_lgb[:, 0] - y[:, 0])**2 + (oof_lgb[:, 1] - y[:, 1])**2).mean()
    cv_tab = np.sqrt((oof_tab[:, 0] - y[:, 0])**2 + (oof_tab[:, 1] - y[:, 1])**2).mean()

    print("\n" + "=" * 70)
    print("개별 모델 CV:")
    print(f"  CatBoost: {cv_cat:.4f}")
    print(f"  LightGBM: {cv_lgb:.4f}")
    print(f"  TabNet: {cv_tab:.4f}")

    # 앙상블 비율 테스트
    print("\n[2] 앙상블 비율 테스트...")
    results = {}

    # CatBoost 위주 앙상블
    for cat_w in [1.0, 0.9, 0.8, 0.7, 0.6]:
        lgb_w = (1 - cat_w) / 2
        tab_w = (1 - cat_w) / 2

        oof_ens = cat_w * oof_cat + lgb_w * oof_lgb + tab_w * oof_tab
        cv_ens = np.sqrt((oof_ens[:, 0] - y[:, 0])**2 + (oof_ens[:, 1] - y[:, 1])**2).mean()
        results[(cat_w, lgb_w, tab_w)] = cv_ens

    best_weights = min(results, key=results.get)
    best_cv = results[best_weights]

    print("\n앙상블 결과:")
    for weights, cv in sorted(results.items(), key=lambda x: x[1]):
        marker = " ★" if weights == best_weights else ""
        print(f"  Cat:{weights[0]:.1f}/LGB:{weights[1]:.2f}/Tab:{weights[2]:.2f} → CV {cv:.4f}{marker}")

    print("\n" + "=" * 70)
    print(f"Best: CatBoost:{best_weights[0]:.1f} / LightGBM:{best_weights[1]:.2f} / TabNet:{best_weights[2]:.2f}")
    print(f"Best CV: {best_cv:.4f}")
    print(f"vs CatBoost alone (13.66): {'+' if best_cv > cv_cat else ''}{best_cv - cv_cat:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
