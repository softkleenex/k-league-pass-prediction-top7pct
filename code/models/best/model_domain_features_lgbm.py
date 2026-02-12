"""
K리그 패스 좌표 예측 - 도메인 피처 (축구 지식 기반)

혁명적 접근:
- Zone 완전 제거! ❌
- 축구 도메인 지식 활용 ✅
- 골대 거리, 필드 구역, 전술적 위치
- Player/Team 통계 (Target Encoding)

왜 이게 게임 체인저인가:
- Zone = 기계적 분할, 의미 없음
- Domain = 축구의 실제 의미 반영
- 1등(12.70)이 사용했을 가능성 높음!

목표:
- CV: < 15.0 (1점 개선!)
- Public: < 15.5
- 순위: 100-150위 (수상권!)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 도메인 피처 (축구 지식 기반)")
print("=" * 80)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"  Train passes: {len(train_df):,}")
print(f"  Test passes: {len(test_all):,}")

# =============================================================================
# 2. 도메인 피처 엔지니어링 (축구 지식!)
# =============================================================================
print("\n[2] 도메인 피처 엔지니어링...")

def create_domain_features(df):
    """축구 도메인 지식 기반 피처 생성"""
    df = df.copy()

    # =========================================================================
    # A. 골대 관련 피처 (가장 중요!)
    # =========================================================================
    # 골대 위치: (105, 34) - 필드 끝, 중앙
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))
    df['is_near_goal'] = (df['goal_distance'] < 20).astype(int)  # 페널티 박스

    # =========================================================================
    # B. 필드 구역 피처 (전술적 위치)
    # =========================================================================
    # X축 구역 (공격/중앙/수비)
    df['zone_attack'] = (df['start_x'] > 70).astype(int)
    df['zone_defense'] = (df['start_x'] < 35).astype(int)
    df['zone_middle'] = ((df['start_x'] >= 35) & (df['start_x'] <= 70)).astype(int)

    # Y축 구역 (좌/중/우)
    df['zone_left'] = (df['start_y'] < 22.67).astype(int)
    df['zone_center'] = ((df['start_y'] >= 22.67) & (df['start_y'] <= 45.33)).astype(int)
    df['zone_right'] = (df['start_y'] > 45.33).astype(int)

    # =========================================================================
    # C. 경계선 거리 (제약 조건)
    # =========================================================================
    df['dist_to_left'] = df['start_y']
    df['dist_to_right'] = 68 - df['start_y']
    df['dist_to_top'] = df['start_x']
    df['dist_to_bottom'] = 105 - df['start_x']

    # =========================================================================
    # D. 패스 히스토리 (이전 패스)
    # =========================================================================
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
    df['prev_distance'] = df.groupby('game_episode')['distance'].shift(1).fillna(0)

    # 방향 (8-way)
    df['direction'] = 0
    mask = (np.abs(df['prev_dx']) >= 1) | (np.abs(df['prev_dy']) >= 1)
    angles = np.degrees(np.arctan2(df.loc[mask, 'prev_dy'], df.loc[mask, 'prev_dx']))

    df.loc[mask & (angles >= -22.5) & (angles < 22.5), 'direction'] = 1
    df.loc[mask & (angles >= 22.5) & (angles < 67.5), 'direction'] = 2
    df.loc[mask & (angles >= 67.5) & (angles < 112.5), 'direction'] = 3
    df.loc[mask & (angles >= 112.5) & (angles < 157.5), 'direction'] = 4
    df.loc[mask & ((angles >= 157.5) | (angles < -157.5)), 'direction'] = 5
    df.loc[mask & (angles >= -157.5) & (angles < -112.5), 'direction'] = 6
    df.loc[mask & (angles >= -112.5) & (angles < -67.5), 'direction'] = 7
    df.loc[mask & (angles >= -67.5) & (angles < -22.5), 'direction'] = 8

    # =========================================================================
    # E. Episode 레벨 피처
    # =========================================================================
    df['pass_number'] = df.groupby('game_episode').cumcount() + 1
    df['total_passes'] = df.groupby('game_episode')['game_episode'].transform('count')
    df['episode_progress'] = df['pass_number'] / df['total_passes']
    df['is_last_pass'] = (df['pass_number'] == df['total_passes']).astype(int)

    # Episode 평균 거리
    df['episode_avg_distance'] = df.groupby('game_episode')['distance'].transform('mean')

    # Episode 전진 비율
    df['is_forward'] = (df['dx'] > 0).astype(int)
    df['episode_forward_ratio'] = df.groupby('game_episode')['is_forward'].transform('mean')

    # =========================================================================
    # F. Target (학습용)
    # =========================================================================
    df['delta_x'] = df['end_x'] - df['start_x']
    df['delta_y'] = df['end_y'] - df['start_y']

    return df

print("  도메인 피처 생성 중...")
train_df = create_domain_features(train_df)
test_all = create_domain_features(test_all)

# =========================================================================
# G. Player/Team Target Encoding (과적합 주의!)
# =========================================================================
print("  Player/Team 통계 계산 중...")

# Player 통계 (Train 전체 기준)
player_stats = train_df.groupby('player_id').agg({
    'delta_x': 'mean',
    'delta_y': 'mean',
    'distance': 'mean',
    'is_forward': 'mean'
}).reset_index()
player_stats.columns = ['player_id', 'player_avg_dx', 'player_avg_dy',
                        'player_avg_distance', 'player_forward_ratio']

# Team 통계
team_stats = train_df.groupby('team_id').agg({
    'delta_x': 'mean',
    'delta_y': 'mean',
    'distance': 'mean'
}).reset_index()
team_stats.columns = ['team_id', 'team_avg_dx', 'team_avg_dy', 'team_avg_distance']

# Merge
train_df = train_df.merge(player_stats, on='player_id', how='left')
train_df = train_df.merge(team_stats, on='team_id', how='left')

test_all = test_all.merge(player_stats, on='player_id', how='left')
test_all = test_all.merge(team_stats, on='team_id', how='left')

# 없는 player/team은 global 평균
for col in ['player_avg_dx', 'player_avg_dy', 'player_avg_distance', 'player_forward_ratio']:
    train_df[col] = train_df[col].fillna(train_df[col].mean())
    test_all[col] = test_all[col].fillna(train_df[col].mean())

for col in ['team_avg_dx', 'team_avg_dy', 'team_avg_distance']:
    train_df[col] = train_df[col].fillna(train_df[col].mean())
    test_all[col] = test_all[col].fillna(train_df[col].mean())

print(f"  도메인 피처 생성 완료!")

# =============================================================================
# 3. 피처 선택
# =============================================================================
print("\n[3] 피처 선택...")

feature_cols = [
    # 기본 위치
    'start_x', 'start_y',

    # 골대 관련 (3개)
    'goal_distance', 'goal_angle', 'is_near_goal',

    # 필드 구역 (6개)
    'zone_attack', 'zone_defense', 'zone_middle',
    'zone_left', 'zone_center', 'zone_right',

    # 경계선 거리 (4개)
    'dist_to_left', 'dist_to_right', 'dist_to_top', 'dist_to_bottom',

    # 이전 패스 (4개)
    'prev_dx', 'prev_dy', 'prev_distance', 'direction',

    # Episode (4개)
    'episode_progress', 'episode_avg_distance', 'episode_forward_ratio', 'is_last_pass',

    # Player 통계 (4개)
    'player_avg_dx', 'player_avg_dy', 'player_avg_distance', 'player_forward_ratio',

    # Team 통계 (3개)
    'team_avg_dx', 'team_avg_dy', 'team_avg_distance',

    # 시간 (2개)
    'period_id', 'time_seconds'
]

categorical_features = ['direction', 'period_id', 'is_last_pass',
                        'zone_attack', 'zone_defense', 'zone_middle',
                        'zone_left', 'zone_center', 'zone_right']

X = train_df[feature_cols].fillna(0)
y_x = train_df['delta_x']
y_y = train_df['delta_y']

sample_weights = np.where(train_df['is_last_pass'] == 1, 10.0, 1.0)

X_test = test_all[feature_cols].fillna(0)

print(f"  총 피처 수: {len(feature_cols)} (Zone 제거!)")
print(f"  Categorical: {len(categorical_features)}개")

# =============================================================================
# 4. GroupKFold 교차 검증
# =============================================================================
print("\n[4] GroupKFold 교차 검증 (도메인 피처)...")

gkf = GroupKFold(n_splits=5)
game_ids = train_df['game_id'].values

fold_scores = []

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}:")

    X_train = X.iloc[train_idx]
    X_val_all = X.iloc[val_idx]
    y_train_x = y_x.iloc[train_idx]
    y_train_y = y_y.iloc[train_idx]
    train_weights = sample_weights[train_idx]

    # X 모델
    train_data_x = lgb.Dataset(X_train, label=y_train_x,
                                categorical_feature=categorical_features,
                                weight=train_weights)
    model_x = lgb.train(params, train_data_x, num_boost_round=300,
                        callbacks=[lgb.log_evaluation(0)])

    # Y 모델
    train_data_y = lgb.Dataset(X_train, label=y_train_y,
                                categorical_feature=categorical_features,
                                weight=train_weights)
    model_y = lgb.train(params, train_data_y, num_boost_round=300,
                        callbacks=[lgb.log_evaluation(0)])

    # 마지막 패스만 평가
    val_last_mask = train_df.iloc[val_idx]['is_last_pass'] == 1
    X_val_last = X_val_all[val_last_mask]
    val_df_last = train_df.iloc[val_idx][val_last_mask]

    pred_delta_x = model_x.predict(X_val_last)
    pred_delta_y = model_y.predict(X_val_last)

    pred_end_x = np.clip(val_df_last['start_x'].values + pred_delta_x, 0, 105)
    pred_end_y = np.clip(val_df_last['start_y'].values + pred_delta_y, 0, 68)

    dist = np.sqrt((pred_end_x - val_df_last['end_x'].values)**2 +
                   (pred_end_y - val_df_last['end_y'].values)**2)
    cv = dist.mean()
    fold_scores.append(cv)

    print(f"    CV: {cv:.4f}")

    # Feature importance (Fold 1만)
    if fold == 0:
        importance = model_x.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print(f"\n    Top 10 중요 피처:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"      {row['feature']:30s}: {row['importance']:.0f}")

# =============================================================================
# 5. CV 요약
# =============================================================================
print("\n" + "=" * 80)
print("CV 요약")
print("=" * 80)

for i, score in enumerate(fold_scores):
    print(f"  Fold {i+1}: {score:.4f}")

fold13_cv = np.mean(fold_scores[:3])
print(f"\n  Fold 1-3 평균: {fold13_cv:.4f} ± {np.std(fold_scores[:3]):.4f}")

# =============================================================================
# 6. Test 예측
# =============================================================================
print("\n[6] Test 예측...")

train_data_x = lgb.Dataset(X, label=y_x, categorical_feature=categorical_features, weight=sample_weights)
train_data_y = lgb.Dataset(X, label=y_y, categorical_feature=categorical_features, weight=sample_weights)

model_x = lgb.train(params, train_data_x, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
model_y = lgb.train(params, train_data_y, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])

test_last_mask = test_all['is_last_pass'] == 1
X_test_last = X_test[test_last_mask]
test_last_df = test_all[test_last_mask]

pred_delta_x = model_x.predict(X_test_last)
pred_delta_y = model_y.predict(X_test_last)

pred_end_x = np.clip(test_last_df['start_x'].values + pred_delta_x, 0, 105)
pred_end_y = np.clip(test_last_df['start_y'].values + pred_delta_y, 0, 68)

# =============================================================================
# 7. 제출 파일
# =============================================================================
print("\n[7] 제출 파일 생성...")

submission = pd.DataFrame({
    'game_episode': test_last_df['game_episode'],
    'end_x': pred_end_x,
    'end_y': pred_end_y
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')

filename = f'submission_domain_features_cv{fold13_cv:.2f}.csv'
submission.to_csv(filename, index=False)

print(f"  {filename} 저장 완료")

# =============================================================================
# 8. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 - 도메인 피처 혁명!")
print("=" * 80)

print(f"\n[혁명적 접근]")
print(f"  ❌ Zone 제거 (기계적 분할)")
print(f"  ✅ 도메인 피처 32개 (축구 지식)")
print(f"  ✅ 골대 거리, 필드 구역, 전술적 위치")

print(f"\n[성능]")
print(f"  Fold 1-3 CV: {fold13_cv:.4f}")

print(f"\n[비교]")
print(f"  Zone 6x6:        16.3356 (Public 16.36, 241위)")
print(f"  전체 패스:        15.8795 (Public 16.30, 235위)")
print(f"  도메인 피처:      {fold13_cv:.4f} (Public ???)")

if fold13_cv < 15.0:
    print(f"\n  🔥 15.0 돌파! 게임 체인저! (개선: {16.3356 - fold13_cv:.4f})")
elif fold13_cv < 15.5:
    print(f"\n  ⭐ 15.5 이하! 우수! (개선: {16.3356 - fold13_cv:.4f})")
elif fold13_cv < 16.0:
    print(f"\n  ✅ 16.0 이하 개선 (개선: {16.3356 - fold13_cv:.4f})")
else:
    print(f"\n  😞 개선 부족 ({fold13_cv - 16.3356:+.4f})")

print(f"\n[제출 파일]")
print(f"  {filename}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
