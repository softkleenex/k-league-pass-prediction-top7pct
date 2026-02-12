"""
exp_005: 위치 기반 피처 실험

새로운 피처:
1. 공격 진영 (x > 52.5) / 수비 진영
2. 측면 (y < 20 or y > 48) / 중앙
3. 페널티 박스 근처 (x > 88.5, 13.84 < y < 54.16)
4. 골라인까지 거리 (105 - x)
5. 측면까지 거리 (min(y, 68-y))

작성일: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../../../../data')


def create_baseline_features(df):
    """Baseline 피처"""
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

    return df


def add_safe_ema_features(df, ema_span=2):
    """EMA 피처 (span=2)"""
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['move_distance'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    result_map = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map).fillna(0)

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

    return df


def add_position_features(df):
    """위치 기반 피처"""
    # 공격 진영 (x > 52.5) / 수비 진영
    df['in_attacking_half'] = (df['start_x'] > 52.5).astype(int)

    # 공격 1/3 (x > 70)
    df['in_final_third'] = (df['start_x'] > 70).astype(int)

    # 측면 vs 중앙
    df['in_center'] = ((df['start_y'] >= 20) & (df['start_y'] <= 48)).astype(int)
    df['in_wide'] = ((df['start_y'] < 20) | (df['start_y'] > 48)).astype(int)

    # 페널티 박스 근처 (x > 88.5, 13.84 < y < 54.16)
    df['near_penalty_box'] = (
        (df['start_x'] > 88.5) &
        (df['start_y'] > 13.84) &
        (df['start_y'] < 54.16)
    ).astype(int)

    # 연속 피처
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)

    # 위치 조합
    df['position_zone'] = df['in_attacking_half'] * 2 + df['in_center']

    return df


def run_cv(X, y, groups):
    gkf = GroupKFold(n_splits=3)
    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)
        model_x.fit(X[train_idx], y[train_idx, 0])
        model_y.fit(X[train_idx], y[train_idx, 1])
        pred_x = model_x.predict(X[val_idx])
        pred_y = model_y.predict(X[val_idx])
        errors = np.sqrt((pred_x - y[val_idx, 0])**2 + (pred_y - y[val_idx, 1])**2)
        fold_scores.append(errors.mean())
        print(f"    Fold {fold}: {fold_scores[-1]:.4f}")

    return np.mean(fold_scores)


def main():
    print("=" * 70)
    print("exp_005: 위치 기반 피처 실험")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_baseline_features(train_df)
    train_df = add_safe_ema_features(train_df, ema_span=2)
    train_df = add_position_features(train_df)

    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # 피처 정의
    BASELINE = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    EMA_FEATURES = [
        'ema_start_x', 'ema_start_y', 'ema_goal_distance',
        'ema_distance', 'ema_success_rate', 'ema_possession'
    ]

    POSITION_BINARY = [
        'in_attacking_half', 'in_final_third', 'in_center', 'in_wide', 'near_penalty_box'
    ]

    POSITION_CONTINUOUS = [
        'dist_to_goal_line', 'dist_to_sideline', 'dist_to_center_y', 'position_zone'
    ]

    # 실험
    experiments = [
        ("Baseline + EMA (기준)", BASELINE + EMA_FEATURES),
        ("+ Position Binary", BASELINE + EMA_FEATURES + POSITION_BINARY),
        ("+ Position Continuous", BASELINE + EMA_FEATURES + POSITION_CONTINUOUS),
        ("+ Position All", BASELINE + EMA_FEATURES + POSITION_BINARY + POSITION_CONTINUOUS),
    ]

    results = {}
    for name, features in experiments:
        print(f"\n[실험] {name} ({len(features)}개 피처)")
        X = last_passes[features].fillna(0).replace([np.inf, -np.inf], 0).values
        cv = run_cv(X, y, groups)
        results[name] = cv

    # 결과 비교
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)

    baseline = results["Baseline + EMA (기준)"]
    for name, cv in results.items():
        diff = baseline - cv
        marker = "***" if cv == min(results.values()) else ""
        print(f"  {name:30s}: CV {cv:.4f}  ({diff:+.4f}) {marker}")

    best_name = min(results, key=results.get)
    best_cv = min(results.values())
    print(f"\n  Best: {best_name} (CV {best_cv:.4f})")
    print("=" * 70)


if __name__ == '__main__':
    main()
