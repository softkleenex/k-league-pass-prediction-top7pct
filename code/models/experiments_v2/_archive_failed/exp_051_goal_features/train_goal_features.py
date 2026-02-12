"""
exp_051: Goal-oriented Features + Advanced CatBoost
목표: 12점대 달성

핵심 전략:
1. Goal-oriented 피처 (골대 기준)
2. Progressive 패스 피처
3. 시퀀스 패턴 피처
4. CatBoost (검증된 모델)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
TRAIN_CSV = BASE / "data/train.csv"
TEST_CSV = BASE / "data/test.csv"
SUBMISSION_DIR = BASE / "submissions"

# 축구장 크기
FIELD_LENGTH = 105
FIELD_WIDTH = 68
GOAL_X = 105  # 공격 골대
GOAL_Y = 34   # 골대 중앙

def extract_features(ep_df, is_train=True):
    """에피소드별 고급 피처 추출"""
    ep_df = ep_df.sort_values('time_seconds').reset_index(drop=True)
    n = len(ep_df)
    last = ep_df.iloc[-1]
    first = ep_df.iloc[0]

    # ===== 1. 기본 피처 =====
    features = {
        'start_x': last['start_x'],
        'start_y': last['start_y'],
        'seq_len': n,
    }

    # is_home 처리
    is_home = last['is_home'] if 'is_home' in last else True

    # dx, dy 계산
    if is_train:
        dx = last['end_x'] - last['start_x']
        dy = last['end_y'] - last['start_y']
    else:
        dx = last.get('dx', last['end_x'] - last['start_x'])
        dy = last.get('dy', last['end_y'] - last['start_y'])

    features['dx'] = dx
    features['dy'] = dy
    features['speed'] = np.sqrt(dx**2 + dy**2)
    features['direction'] = np.arctan2(dy, dx)

    # ===== 2. Goal-oriented 피처 (핵심!) =====
    # 골대까지 거리
    features['dist_to_goal'] = np.sqrt((GOAL_X - last['start_x'])**2 + (GOAL_Y - last['start_y'])**2)
    # 골대 방향 각도
    features['angle_to_goal'] = np.arctan2(GOAL_Y - last['start_y'], GOAL_X - last['start_x'])
    # 골대 방향으로의 이동 성분
    goal_direction = np.array([GOAL_X - last['start_x'], GOAL_Y - last['start_y']])
    goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
    move_vector = np.array([dx, dy])
    features['move_toward_goal'] = np.dot(move_vector, goal_direction)

    # 시작점의 골대 거리
    features['start_dist_to_goal'] = np.sqrt((GOAL_X - first['start_x'])**2 + (GOAL_Y - first['start_y'])**2)
    # 진행도 (얼마나 골대에 가까워졌나)
    features['progress_to_goal'] = features['start_dist_to_goal'] - features['dist_to_goal']

    # ===== 3. Zone 피처 (6x6 + 세부) =====
    zone_x = int(last['start_x'] // (FIELD_LENGTH/6))
    zone_y = int(last['start_y'] // (FIELD_WIDTH/6))
    zone_x = min(max(zone_x, 0), 5)
    zone_y = min(max(zone_y, 0), 5)
    features['zone_id'] = zone_y * 6 + zone_x

    # 10x10 세부 zone
    zone_x_10 = int(last['start_x'] // (FIELD_LENGTH/10))
    zone_y_10 = int(last['start_y'] // (FIELD_WIDTH/10))
    zone_x_10 = min(max(zone_x_10, 0), 9)
    zone_y_10 = min(max(zone_y_10, 0), 9)
    features['zone_id_10x10'] = zone_y_10 * 10 + zone_x_10

    # ===== 4. 시퀀스 진행 피처 =====
    features['progression_x'] = last['start_x'] - first['start_x']
    features['progression_y'] = last['start_y'] - first['start_y']
    features['total_progression'] = np.sqrt(features['progression_x']**2 + features['progression_y']**2)

    # 시간 피처
    features['total_time'] = last['time_seconds'] - first['time_seconds']
    features['avg_time_per_action'] = features['total_time'] / n if n > 0 else 0

    # ===== 5. 패스 체인 피처 =====
    # Pass만 필터링
    passes = ep_df[ep_df['type_name'] == 'Pass']
    features['n_passes'] = len(passes)

    if len(passes) > 0:
        # 패스 성공률
        successful = passes[passes['result_name'] == 'Successful']
        features['pass_success_rate'] = len(successful) / len(passes)

        # 평균 패스 거리
        pass_dists = np.sqrt((passes['end_x'] - passes['start_x'])**2 +
                            (passes['end_y'] - passes['start_y'])**2)
        features['avg_pass_dist'] = pass_dists.mean()
        features['max_pass_dist'] = pass_dists.max()

        # Progressive 패스 (골대 방향으로 진행하는 패스)
        prog_passes = 0
        for _, p in passes.iterrows():
            start_goal_dist = np.sqrt((GOAL_X - p['start_x'])**2 + (GOAL_Y - p['start_y'])**2)
            end_goal_dist = np.sqrt((GOAL_X - p['end_x'])**2 + (GOAL_Y - p['end_y'])**2)
            if end_goal_dist < start_goal_dist:
                prog_passes += 1
        features['progressive_passes'] = prog_passes
        features['progressive_pass_rate'] = prog_passes / len(passes)
    else:
        features['pass_success_rate'] = 0
        features['avg_pass_dist'] = 0
        features['max_pass_dist'] = 0
        features['progressive_passes'] = 0
        features['progressive_pass_rate'] = 0

    # ===== 6. Rolling 통계 =====
    for col in ['start_x', 'start_y']:
        roll3 = ep_df[col].rolling(3, min_periods=1)
        features[f'{col}_roll3_mean'] = roll3.mean().iloc[-1]
        features[f'{col}_roll3_std'] = roll3.std().iloc[-1] if n > 1 else 0

        roll5 = ep_df[col].rolling(5, min_periods=1)
        features[f'{col}_roll5_mean'] = roll5.mean().iloc[-1]
        features[f'{col}_roll5_std'] = roll5.std().iloc[-1] if n > 1 else 0

    # dx, dy rolling
    dx_series = ep_df['end_x'] - ep_df['start_x']
    dy_series = ep_df['end_y'] - ep_df['start_y']

    for name, series in [('dx', dx_series), ('dy', dy_series)]:
        roll3 = series.rolling(3, min_periods=1)
        features[f'{name}_roll3_mean'] = roll3.mean().iloc[-1]
        features[f'{name}_roll3_std'] = roll3.std().iloc[-1] if n > 1 else 0

    # ===== 7. 위치 기반 피처 =====
    # 필드 위치 (3분할: 수비/중앙/공격)
    features['field_third'] = 0 if last['start_x'] < 35 else (2 if last['start_x'] > 70 else 1)
    # 측면 vs 중앙
    features['is_wing'] = 1 if (last['start_y'] < 15 or last['start_y'] > 53) else 0
    # 박스 내부 여부
    features['in_box'] = 1 if (last['start_x'] > 88.5 and 13.85 < last['start_y'] < 54.15) else 0

    # ===== 8. 홈/원정 =====
    features['is_home'] = 1 if is_home else 0

    return features


def load_train_data():
    """Train 데이터 로드"""
    df = pd.read_csv(TRAIN_CSV)
    episodes = []
    targets = []
    game_ids = []

    for game_ep, ep_df in tqdm(df.groupby('game_episode'), desc="Loading train"):
        features = extract_features(ep_df, is_train=True)
        features['game_episode'] = game_ep
        features['game_id'] = game_ep.split('_')[0]

        # Target
        last = ep_df.sort_values('time_seconds').iloc[-1]
        targets.append([last['end_x'], last['end_y']])
        game_ids.append(int(game_ep.split('_')[0]))

        episodes.append(features)

    return pd.DataFrame(episodes), np.array(targets), np.array(game_ids)


def load_test_data():
    """Test 데이터 로드"""
    df = pd.read_csv(TEST_CSV)
    episodes = []

    for _, row in tqdm(df.iterrows(), desc="Loading test", total=len(df)):
        path = BASE / "data" / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)

        features = extract_features(ep_df, is_train=False)
        features['game_episode'] = row['game_episode']
        features['game_id'] = str(row['game_id'])

        episodes.append(features)

    return pd.DataFrame(episodes)


# 피처 리스트
FEATURES = [
    # 기본
    'start_x', 'start_y', 'dx', 'dy', 'speed', 'direction', 'seq_len',
    # Goal-oriented (핵심!)
    'dist_to_goal', 'angle_to_goal', 'move_toward_goal',
    'start_dist_to_goal', 'progress_to_goal',
    # Zone
    'zone_id', 'zone_id_10x10',
    # 시퀀스 진행
    'progression_x', 'progression_y', 'total_progression',
    'total_time', 'avg_time_per_action',
    # 패스 체인
    'n_passes', 'pass_success_rate', 'avg_pass_dist', 'max_pass_dist',
    'progressive_passes', 'progressive_pass_rate',
    # Rolling
    'start_x_roll3_mean', 'start_x_roll3_std',
    'start_y_roll3_mean', 'start_y_roll3_std',
    'start_x_roll5_mean', 'start_x_roll5_std',
    'start_y_roll5_mean', 'start_y_roll5_std',
    'dx_roll3_mean', 'dx_roll3_std',
    # 위치 기반
    'field_third', 'is_wing', 'in_box', 'is_home',
]
TARGETS = ['end_x', 'end_y']


def main():
    print("="*70)
    print("exp_051: Goal-oriented Features + CatBoost")
    print("목표: 12점대 (1위)")
    print("="*70)

    # 데이터 로드
    print("\n[1] Train 데이터 로드...")
    train_df, targets, game_ids = load_train_data()
    train_df = train_df.fillna(0)
    train_df['end_x'] = targets[:, 0]
    train_df['end_y'] = targets[:, 1]
    print(f"  에피소드: {len(train_df)}")
    print(f"  피처 수: {len(FEATURES)}")

    # Cross-validation
    print("\n[2] 3-Fold CV...")
    gkf = GroupKFold(n_splits=3)
    cv_scores = []
    oof_preds = np.zeros((len(train_df), 2))

    params = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
        'random_state': 42,
        'loss_function': 'MultiRMSE',
        'verbose': 100,
        'early_stopping_rounds': 50
    }

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=game_ids)):
        print(f"\n--- Fold {fold+1} ---")

        X_train = train_df.iloc[train_idx][FEATURES]
        y_train = train_df.iloc[train_idx][TARGETS]
        X_val = train_df.iloc[val_idx][FEATURES]
        y_val = train_df.iloc[val_idx][TARGETS]

        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        # Euclidean distance
        dist = np.sqrt(np.sum((preds - y_val.values)**2, axis=1))
        score = np.mean(dist)
        cv_scores.append(score)
        print(f"  Fold {fold+1}: {score:.4f}")

    mean_cv = np.mean(cv_scores)
    print(f"\n" + "="*70)
    print(f"CV Score: {mean_cv:.4f}")
    print(f"Individual: {cv_scores}")
    print("="*70)

    # Test 예측
    print("\n[3] Test 데이터 로드...")
    test_df = load_test_data()
    test_df = test_df.fillna(0)
    print(f"  Test: {len(test_df)}")

    print("\n[4] Full train & 예측...")
    X_train_full = train_df[FEATURES]
    y_train_full = train_df[TARGETS]

    model = CatBoostRegressor(**{**params, 'early_stopping_rounds': None})
    model.fit(X_train_full, y_train_full)

    X_test = test_df[FEATURES]
    preds = model.predict(X_test)

    print("\n[5] 제출 파일 생성...")
    submission = pd.DataFrame({
        'game_episode': test_df['game_episode'],
        'end_x': preds[:, 0],
        'end_y': preds[:, 1]
    })

    output_path = SUBMISSION_DIR / f"submission_goal_cv{mean_cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  저장: {output_path}")

    print("\n[예측 분포]")
    print(f"  end_x: mean={submission['end_x'].mean():.2f}, std={submission['end_x'].std():.2f}")
    print(f"  end_y: mean={submission['end_y'].mean():.2f}, std={submission['end_y'].std():.2f}")

    # Feature importance
    print("\n[피처 중요도 Top 10]")
    importance = model.get_feature_importance()
    feat_imp = pd.DataFrame({'feature': FEATURES, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print(feat_imp.head(10).to_string(index=False))

    print("\n" + "="*70)
    print("완료!")
    print("="*70)

    return mean_cv


if __name__ == "__main__":
    main()
