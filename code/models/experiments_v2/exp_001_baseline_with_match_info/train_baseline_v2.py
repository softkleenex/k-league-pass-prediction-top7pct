"""
exp_001 v2: match_info 활용 (누출 수정)

문제: 팀 통계를 전체 데이터로 계산하면 CV 누출
해결: CV 폴드 내에서만 통계 계산

작성일: 2025-12-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('../../../../data')


def load_data():
    """데이터 로드"""
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    match_info = pd.read_csv(DATA_DIR / 'match_info.csv')

    # 필요한 컬럼만 조인
    train_df = train_df.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id', 'game_day']],
        on='game_id',
        how='left'
    )

    return train_df


def create_base_features(df):
    """기본 피처 (누출 없음)"""

    # 위치 피처
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    # 이전 패스 방향
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    # 결과 인코딩
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    # 최종 팀 피처
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

    # 홈팀 여부
    df['is_home_team'] = (df['team_id'] == df['home_team_id']).astype(int)

    # 시즌 진행도
    df['game_day_norm'] = df['game_day'] / 38.0

    return df


def main():
    print("=" * 70)
    print("exp_001 v2: match_info + 누출 없는 CV")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    train_df = load_data()
    train_df = create_base_features(train_df)

    # 마지막 패스만
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"  에피소드: {len(last_passes)}")

    # 기본 피처 (Top 10)
    BASE_FEATURES = [
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct'
    ]

    # match_info 피처 (누출 없는 것만)
    MATCH_FEATURES = [
        'is_home_team',      # 홈팀 여부 - 단순 이진값
        'game_day_norm',     # 시즌 진행도 - 단순 정규화값
    ]

    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # 실험 1: 기존 Top 10만
    print("\n[2] 실험 1: Top 10 피처만 (베이스라인)")
    X1 = last_passes[BASE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    cv1 = run_cv(X1, y, groups)

    # 실험 2: Top 10 + match_info 피처
    print("\n[3] 실험 2: Top 10 + match_info")
    ALL_FEATURES = BASE_FEATURES + MATCH_FEATURES
    X2 = last_passes[ALL_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    cv2 = run_cv(X2, y, groups)

    # 결과 비교
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)
    print(f"  Top 10 only:        CV {cv1:.4f}")
    print(f"  Top 10 + match_info: CV {cv2:.4f}")
    print(f"  차이: {cv1 - cv2:+.4f}")

    if cv2 < cv1:
        print("  -> match_info 피처 효과 있음!")
    else:
        print("  -> match_info 피처 효과 없음")

    print("=" * 70)


def run_cv(X, y, groups):
    """CV 실행"""
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


if __name__ == '__main__':
    main()
