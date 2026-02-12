"""
exp_074: Shared Code Features
- Data Augmentation (Y-flip)
- goal_open_angle (골대 열린 각도)
- is_final_third (x > 70)
- match_info context (rest_days, match_hour)
- Comparing with our best CV 13.66
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


def augment_data(df):
    """Y좌표 반전으로 데이터 증강"""
    aug_df = df.copy()
    aug_df['start_y'] = 68.0 - aug_df['start_y']
    aug_df['end_y'] = 68.0 - aug_df['end_y']
    aug_df['game_episode'] = aug_df['game_episode'].astype(str) + '_aug'
    return pd.concat([df, aug_df], ignore_index=True)


def process_match_info(matches):
    """Match context 추출 (휴식일, 시간대 등)"""
    df = matches.copy()
    df['match_date_kst'] = pd.to_datetime(df['game_date']) + pd.Timedelta(hours=9)
    df['match_hour'] = df['match_date_kst'].dt.hour
    df['is_weekend'] = (df['match_date_kst'].dt.weekday >= 5).astype(int)

    # 팀별 휴식일 계산
    home = df[['match_date_kst', 'home_team_id']].rename(columns={'home_team_id': 'team_id'})
    away = df[['match_date_kst', 'away_team_id']].rename(columns={'away_team_id': 'team_id'})
    full_schedule = pd.concat([home, away])
    full_schedule['date_only'] = full_schedule['match_date_kst'].dt.normalize()
    full_schedule = full_schedule.drop_duplicates().sort_values(['team_id', 'date_only'])
    full_schedule['prev_date'] = full_schedule.groupby('team_id')['date_only'].shift(1)
    full_schedule['rest_days'] = (full_schedule['date_only'] - full_schedule['prev_date']).dt.days
    full_schedule['rest_days'] = full_schedule['rest_days'].fillna(7).clip(0, 14)

    rest_map = dict(zip(
        zip(full_schedule['team_id'], full_schedule['date_only'].dt.date.astype(str)),
        full_schedule['rest_days']
    ))
    df['date_str'] = df['match_date_kst'].dt.date.astype(str)

    df['home_rest'] = df.apply(lambda x: rest_map.get((x['home_team_id'], x['date_str']), 7), axis=1)
    df['away_rest'] = df.apply(lambda x: rest_map.get((x['away_team_id'], x['date_str']), 7), axis=1)

    return df[['game_id', 'match_hour', 'is_weekend', 'home_rest', 'away_rest']]


def create_features(df, match_context=None):
    """Feature Engineering - 공유 코드 참고"""

    # 기존 zone features
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)

    # 기존 goal features
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    # [NEW] goal_open_angle - 골대 열린 각도 (공유 코드에서 가져옴)
    Y_NEAR = 30.34  # 골대 가까운 포스트
    Y_FAR = 37.66   # 골대 먼 포스트
    X_GOAL = 105
    angle_near = np.arctan2(Y_NEAR - df['start_y'], X_GOAL - df['start_x'])
    angle_far = np.arctan2(Y_FAR - df['start_y'], X_GOAL - df['start_x'])
    df['goal_open_angle'] = np.abs(angle_far - angle_near)

    # [NEW] is_final_third - 공격 1/3 지역
    df['is_final_third'] = (df['start_x'] > 70).astype(int)

    # [NEW] min_dist_to_touchline - 터치라인까지 거리
    df['min_dist_to_touchline'] = np.minimum(df['start_y'], 68 - df['start_y'])

    # dx, dy
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    # EMA features
    ema_span = 2
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    result_map = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map).fillna(0)
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

    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    # Match context merge
    if match_context is not None:
        df = df.merge(match_context, on='game_id', how='left')
        df['match_hour'] = df['match_hour'].fillna(19)
        df['is_weekend'] = df['is_weekend'].fillna(0)
        df['home_rest'] = df['home_rest'].fillna(7)
        df['away_rest'] = df['away_rest'].fillna(7)

    return df


def main():
    print("=" * 70)
    print("exp_074: Shared Code Features Analysis")
    print("=" * 70)

    # 데이터 로드
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    match_info = pd.read_csv(DATA_DIR / 'match_info.csv')

    # Match context 처리
    match_context = process_match_info(match_info)

    print(f"\n원본 데이터: {len(train_df)} rows")

    # ============================================================
    # Test 1: 기존 TOP_12 (baseline)
    # ============================================================
    print("\n[1] Baseline (TOP_12, no augmentation)...")

    train_feat = create_features(train_df.copy(), match_context)
    last_passes = train_feat.groupby('game_episode').last().reset_index()

    TOP_12 = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession', 'velocity'
    ]

    X = last_passes[TOP_12].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y = last_passes[['end_x', 'end_y']].values.astype(np.float32)
    groups = last_passes['game_id'].values

    cv_baseline = run_cv(X, y, groups, "Baseline")

    # ============================================================
    # Test 2: TOP_12 + New Features (no augmentation)
    # ============================================================
    print("\n[2] TOP_12 + New Features (goal_open_angle, is_final_third, etc.)...")

    NEW_FEATURES = TOP_12 + [
        'goal_open_angle', 'is_final_third', 'min_dist_to_touchline'
    ]

    # match context가 있으면 추가
    if 'match_hour' in last_passes.columns:
        NEW_FEATURES += ['match_hour', 'is_weekend', 'home_rest', 'away_rest']

    X_new = last_passes[NEW_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    cv_new_features = run_cv(X_new, y, groups, "New Features")

    # ============================================================
    # Test 3: Data Augmentation (Y-flip)
    # ============================================================
    print("\n[3] Data Augmentation (Y-flip)...")

    train_aug = augment_data(train_df)
    print(f"  Augmented data: {len(train_aug)} rows (2x)")

    train_aug_feat = create_features(train_aug.copy(), match_context)
    last_passes_aug = train_aug_feat.groupby('game_episode').last().reset_index()

    # augmented 데이터의 game_id 처리 (원본과 동일하게)
    last_passes_aug['game_id_orig'] = last_passes_aug['game_id']

    X_aug = last_passes_aug[NEW_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_aug = last_passes_aug[['end_x', 'end_y']].values.astype(np.float32)
    groups_aug = last_passes_aug['game_id_orig'].values

    cv_aug = run_cv(X_aug, y_aug, groups_aug, "Augmented")

    # ============================================================
    # 결과 요약
    # ============================================================
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    print(f"  Baseline (TOP_12):      CV {cv_baseline:.4f}")
    print(f"  + New Features:         CV {cv_new_features:.4f} ({cv_new_features - cv_baseline:+.4f})")
    print(f"  + Augmentation:         CV {cv_aug:.4f} ({cv_aug - cv_baseline:+.4f})")
    print(f"\n  vs Best (13.66):        {cv_aug - 13.66:+.4f}")
    print("=" * 70)


def run_cv(X, y, groups, name):
    """5-Fold CV 실행"""
    cat_params = {
        'iterations': 4000, 'depth': 8, 'learning_rate': 0.01,
        'l2_leaf_reg': 7.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 100, 'loss_function': 'MAE'
    }

    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros((len(X), 2))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model_x = CatBoostRegressor(**cat_params)
        model_y = CatBoostRegressor(**cat_params)

        model_x.fit(X[train_idx], y[train_idx, 0],
                   eval_set=(X[val_idx], y[val_idx, 0]), use_best_model=True)
        model_y.fit(X[train_idx], y[train_idx, 1],
                   eval_set=(X[val_idx], y[val_idx, 1]), use_best_model=True)

        oof_pred[val_idx, 0] = model_x.predict(X[val_idx])
        oof_pred[val_idx, 1] = model_y.predict(X[val_idx])

        fold_err = np.sqrt((oof_pred[val_idx, 0] - y[val_idx, 0])**2 +
                          (oof_pred[val_idx, 1] - y[val_idx, 1])**2).mean()
        print(f"    Fold {fold}: {fold_err:.4f}")

    cv = np.sqrt((oof_pred[:, 0] - y[:, 0])**2 + (oof_pred[:, 1] - y[:, 1])**2).mean()
    print(f"  {name} CV: {cv:.4f}")
    return cv


if __name__ == "__main__":
    main()
