"""
exp_001: match_info를 활용한 베이스라인

새로운 시작 - match_info.csv 활용
기존 without_match_info 실험들과 완전히 분리

작성일: 2025-12-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
DATA_DIR = Path('../../../../data')


def load_data():
    """데이터 로드 및 match_info 조인"""
    print("[1] 데이터 로드...")

    # Train 데이터
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Train passes: {len(train_df):,}")

    # Match info
    match_info = pd.read_csv(DATA_DIR / 'match_info.csv')
    print(f"  Match info: {len(match_info)} games")

    # 조인
    train_df = train_df.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id',
                    'home_score', 'away_score', 'game_day', 'venue']],
        on='game_id',
        how='left'
    )
    print(f"  After merge: {len(train_df):,}")

    return train_df, match_info


def calculate_team_stats(train_df):
    """팀별 통계 계산 (타겟 누출 주의!)"""
    print("\n[2] 팀별 통계 계산...")

    # 마지막 패스만 (end_x, end_y가 타겟)
    last_passes = train_df.groupby('game_episode').last().reset_index()

    # 팀별 평균 end_x, end_y (이건 타겟 누출!)
    # 대신 팀별 평균 start_x, start_y 사용 (마지막 패스 기준)
    team_stats = last_passes.groupby('team_id').agg({
        'start_x': 'mean',
        'start_y': 'mean',
        'game_episode': 'count'  # 에피소드 수
    }).reset_index()
    team_stats.columns = ['team_id', 'team_avg_start_x', 'team_avg_start_y', 'team_episode_count']

    print(f"  팀 수: {len(team_stats)}")

    return team_stats


def create_features(df, team_stats, match_info):
    """피처 생성 (match_info 포함)"""
    print("\n[3] 피처 생성...")

    # 1. 기본 위치 피처
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    # 2. 이전 패스 방향 (타겟 누출 없음 - shift 사용)
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    # 3. 팀 관련 피처
    df['is_home_team'] = (df['team_id'] == df['home_team_id']).astype(int)

    # 팀 통계 조인
    df = df.merge(team_stats, on='team_id', how='left')
    df['team_avg_start_x'] = df['team_avg_start_x'].fillna(df['start_x'].mean())
    df['team_avg_start_y'] = df['team_avg_start_y'].fillna(df['start_y'].mean())

    # 4. 경기 진행 피처
    df['game_day_normalized'] = df['game_day'] / 38.0  # 시즌 진행도

    # 5. 결과 인코딩
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    # 6. 최종 팀 관련 (기존에 효과 있었던 피처)
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


def main():
    print("=" * 70)
    print("exp_001: match_info를 활용한 베이스라인")
    print("=" * 70)

    # 데이터 로드
    train_df, match_info = load_data()

    # 팀 통계 계산
    team_stats = calculate_team_stats(train_df)

    # 피처 생성
    train_df = create_features(train_df, team_stats, match_info)

    # 마지막 패스만
    last_passes = train_df.groupby('game_episode').last().reset_index()
    print(f"\n  에피소드: {len(last_passes)}")

    # 피처 선택
    FEATURES = [
        # 기존 Top 10 (효과 검증됨)
        'goal_distance', 'zone_y', 'goal_angle', 'prev_dx', 'prev_dy',
        'zone_x', 'final_poss_len', 'direction', 'result_encoded', 'team_possession_pct',
        # 새로운 match_info 피처
        'is_home_team',           # 홈팀 여부
        'team_avg_start_x',       # 팀 평균 시작 위치
        'team_avg_start_y',
        'game_day_normalized',    # 시즌 진행도
    ]

    print(f"\n  피처 수: {len(FEATURES)}")
    print(f"  - 기존 Top 10: 10개")
    print(f"  - 새 match_info: {len(FEATURES) - 10}개")

    X = last_passes[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    # CV
    print("\n[4] 3-Fold CV...")
    gkf = GroupKFold(n_splits=3)
    fold_scores = []

    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42,
        'verbose': 0
    }

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

    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    print(f"\n  CV Mean: {cv_mean:.4f} +/- {cv_std:.4f}")

    # 기존 without_match_info 베이스라인과 비교
    baseline_cv = 15.2569  # 기존 Top 10 without match_info
    improvement = baseline_cv - cv_mean

    print(f"\n  기존 (without match_info): {baseline_cv:.4f}")
    print(f"  현재 (with match_info): {cv_mean:.4f}")
    print(f"  개선: {improvement:+.4f}")

    if improvement > 0:
        print("  -> match_info 효과 있음!")
    else:
        print("  -> match_info 효과 없음 (피처 재검토 필요)")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

    return cv_mean, fold_scores


if __name__ == '__main__':
    main()
