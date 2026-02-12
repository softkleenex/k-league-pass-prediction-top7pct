"""
K리그 패스 좌표 예측 - 전체 패스 학습 (GAME CHANGER!)

혁명적 아이디어:
- 학습: 전체 패스 356,721개 (23배 증가!)
- 평가: 마지막 패스만 15,435개
- 결과: 과적합 완화 + 복잡한 패턴 학습

핵심:
1. 데이터 23배 증가 → Player ID 사용 가능
2. is_last_pass 플래그 → 마지막 패스 구분
3. Sample weight → 마지막 패스 중요도 ↑

목표:
- CV: 14-15
- Public: 13.5-15.5
- 순위: 50-150위 (수상권 진입!)
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
print("K리그 패스 좌표 예측 - 전체 패스 학습 (GAME CHANGER!)")
print("=" * 80)

# =============================================================================
# 1. 데이터 로드 (전체 패스!)
# =============================================================================
print("\n[1] 데이터 로드 (전체 패스)...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"  Train episodes: {train_df['game_episode'].nunique():,}")
print(f"  Train passes (전체): {len(train_df):,}")
print(f"  Test episodes: {test_all['game_episode'].nunique():,}")
print(f"  Test passes (전체): {len(test_all):,}")

# =============================================================================
# 2. 피처 엔지니어링
# =============================================================================
print("\n[2] 피처 엔지니어링...")

def get_zone_6x6(x, y):
    """6x6 Zone 분류"""
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

def get_direction_8way(prev_dx, prev_dy):
    """8방향 분류"""
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 0

    angle = np.arctan2(prev_dy, prev_dx)
    angle_deg = np.degrees(angle)

    if -22.5 <= angle_deg < 22.5:
        return 1
    elif 22.5 <= angle_deg < 67.5:
        return 2
    elif 67.5 <= angle_deg < 112.5:
        return 3
    elif 112.5 <= angle_deg < 157.5:
        return 4
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return 5
    elif -157.5 <= angle_deg < -112.5:
        return 6
    elif -112.5 <= angle_deg < -67.5:
        return 7
    else:
        return 8

def prepare_features(df):
    """피처 준비 (전체 패스)"""
    df = df.copy()

    # Delta 계산
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    # Zone & Direction
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)

    # 마지막 패스 플래그 (핵심!)
    df['pass_number'] = df.groupby('game_episode').cumcount() + 1
    df['total_passes'] = df.groupby('game_episode')['game_episode'].transform('count')
    df['is_last_pass'] = (df['pass_number'] == df['total_passes']).astype(int)

    # Target
    df['delta_x'] = df['end_x'] - df['start_x']
    df['delta_y'] = df['end_y'] - df['start_y']

    return df

train_df = prepare_features(train_df)
test_all = prepare_features(test_all)

# 통계
train_last = train_df[train_df['is_last_pass'] == 1]
train_middle = train_df[train_df['is_last_pass'] == 0]

print(f"\n  Train 전체 패스: {len(train_df):,}")
print(f"    - 마지막 패스: {len(train_last):,} (평가용)")
print(f"    - 중간 패스: {len(train_middle):,} (학습용)")
print(f"    - 비율: 1:{len(train_middle)/len(train_last):.1f}")

print(f"\n  Test 전체 패스: {len(test_all):,}")
print(f"    - 마지막 패스: {(test_all['is_last_pass']==1).sum():,}")

print(f"\n  Unique players: {train_df['player_id'].nunique():,}")
print(f"  Unique teams: {train_df['team_id'].nunique():,}")

# =============================================================================
# 3. 피처 선택
# =============================================================================
print("\n[3] 피처 선택...")

feature_cols = [
    'zone', 'direction', 'player_id', 'team_id',
    'start_x', 'start_y',
    'period_id', 'time_seconds',
    'is_last_pass'  # 핵심!
]

categorical_features = ['zone', 'direction', 'player_id', 'team_id', 'period_id', 'is_last_pass']

# 전체 패스 사용
X = train_df[feature_cols].fillna(0)
y_x = train_df['delta_x']
y_y = train_df['delta_y']

# Sample weight (마지막 패스 가중치 높임)
sample_weights = np.where(train_df['is_last_pass'] == 1, 10.0, 1.0)

X_test = test_all[feature_cols].fillna(0)

print(f"  총 학습 샘플: {len(X):,}")
print(f"  피처 수: {len(feature_cols)}")
print(f"  Categorical: {len(categorical_features)}개")
print(f"\n  Sample weight:")
print(f"    - 마지막 패스: 10.0")
print(f"    - 중간 패스: 1.0")

# =============================================================================
# 4. GroupKFold 교차 검증 (마지막 패스만 평가!)
# =============================================================================
print("\n[4] GroupKFold 교차 검증 (전체 학습, 마지막만 평가)...")

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

    # 전체 패스로 학습
    X_train = X.iloc[train_idx]
    X_val_all = X.iloc[val_idx]
    y_train_x = y_x.iloc[train_idx]
    y_train_y = y_y.iloc[train_idx]
    train_weights = sample_weights[train_idx]

    # X 모델
    train_data_x = lgb.Dataset(
        X_train, label=y_train_x,
        categorical_feature=categorical_features,
        weight=train_weights
    )

    model_x = lgb.train(
        params,
        train_data_x,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)]
    )

    # Y 모델
    train_data_y = lgb.Dataset(
        X_train, label=y_train_y,
        categorical_feature=categorical_features,
        weight=train_weights
    )

    model_y = lgb.train(
        params,
        train_data_y,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)]
    )

    # 마지막 패스만 평가! (핵심!)
    val_last_mask = train_df.iloc[val_idx]['is_last_pass'] == 1
    X_val_last = X_val_all[val_last_mask]
    val_df_last = train_df.iloc[val_idx][val_last_mask]

    pred_delta_x = model_x.predict(X_val_last)
    pred_delta_y = model_y.predict(X_val_last)

    pred_end_x = np.clip(val_df_last['start_x'].values + pred_delta_x, 0, 105)
    pred_end_y = np.clip(val_df_last['start_y'].values + pred_delta_y, 0, 68)

    # 점수 계산 (마지막 패스만!)
    dist = np.sqrt((pred_end_x - val_df_last['end_x'].values)**2 +
                   (pred_end_y - val_df_last['end_y'].values)**2)
    cv = dist.mean()
    fold_scores.append(cv)

    print(f"    학습 샘플: {len(X_train):,} (전체 패스)")
    print(f"    평가 샘플: {len(X_val_last):,} (마지막 패스만)")
    print(f"    CV: {cv:.4f}")

# =============================================================================
# 5. CV 요약
# =============================================================================
print("\n" + "=" * 80)
print("CV 요약")
print("=" * 80)

print(f"\nFold별 점수 (마지막 패스만 평가):")
for i, score in enumerate(fold_scores):
    print(f"  Fold {i+1}: {score:.4f}")

print(f"\n  Fold 1-3 평균: {np.mean(fold_scores[:3]):.4f} ± {np.std(fold_scores[:3]):.4f}")
print(f"  전체 평균: {np.mean(fold_scores):.4f}")

# =============================================================================
# 6. Test 예측
# =============================================================================
print("\n[6] Test 예측...")

# 전체 Train으로 재학습
train_data_x = lgb.Dataset(X, label=y_x, categorical_feature=categorical_features, weight=sample_weights)
train_data_y = lgb.Dataset(X, label=y_y, categorical_feature=categorical_features, weight=sample_weights)

model_x = lgb.train(params, train_data_x, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
model_y = lgb.train(params, train_data_y, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])

# 마지막 패스만 예측
test_last_mask = test_all['is_last_pass'] == 1
X_test_last = X_test[test_last_mask]
test_last_df = test_all[test_last_mask]

pred_delta_x = model_x.predict(X_test_last)
pred_delta_y = model_y.predict(X_test_last)

pred_end_x = np.clip(test_last_df['start_x'].values + pred_delta_x, 0, 105)
pred_end_y = np.clip(test_last_df['start_y'].values + pred_delta_y, 0, 68)

print(f"  Test 예측 샘플: {len(X_test_last):,} (마지막 패스만)")

# =============================================================================
# 7. 제출 파일 생성
# =============================================================================
print("\n[7] 제출 파일 생성...")

fold13_cv = np.mean(fold_scores[:3])

submission = pd.DataFrame({
    'game_episode': test_last_df['game_episode'],
    'end_x': pred_end_x,
    'end_y': pred_end_y
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')

filename = f'submission_all_passes_cv{fold13_cv:.2f}.csv'
submission.to_csv(filename, index=False)

print(f"  {filename} 저장 완료")

# =============================================================================
# 8. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

print(f"\n[혁명적 접근]")
print(f"  학습: 전체 패스 {len(X):,}개 (23배!)")
print(f"  평가: 마지막 패스 {len(train_last):,}개")
print(f"  Sample weight: 마지막 패스 10배")

print(f"\n[성능]")
print(f"  Fold 1-3 CV: {fold13_cv:.4f} ± {np.std(fold_scores[:3]):.4f}")

print(f"\n[비교]")
print(f"  Zone 6x6:            16.3356 (Public 16.36, 241위)")
print(f"  Zone+Player (마지막): 15.9422 (Public 16.58, 과적합)")
print(f"  Zone+Player (전체):   {fold13_cv:.4f} (Public ???)")

if fold13_cv < 16.3356:
    improve = 16.3356 - fold13_cv
    print(f"\n  ✅ Zone 대비 개선: {improve:.4f}")
    if improve > 1.0:
        print(f"  🔥 1점 이상 개선! 게임 체인저!")
else:
    print(f"\n  ❌ 악화: +{fold13_cv - 16.3356:.4f}")

print(f"\n[예상 순위]")
if fold13_cv < 14.5:
    print(f"  🎉 50-100위 (수상권!)")
elif fold13_cv < 15.5:
    print(f"  ⭐ 100-150위 (우수)")
elif fold13_cv < 16.0:
    print(f"  ✅ 150-200위 (개선)")
else:
    print(f"  😞 200위+ (추가 개선 필요)")

print(f"\n[제출 파일]")
print(f"  {filename}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
