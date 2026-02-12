"""
K리그 패스 좌표 예측 - 시퀀스 피처 기반 LightGBM
Phase 1: 시퀀스 정보 활용 + 강한 정규화
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - 시퀀스 피처 기반 LightGBM")
print("=" * 70)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# 테스트 에피소드 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"Train rows: {len(train_df):,}")
print(f"Test rows: {len(test_all):,}")
print(f"Train episodes: {train_df['game_episode'].nunique():,}")
print(f"Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. 시퀀스 피처 엔지니어링
# =============================================================================
print("\n[2] 시퀀스 피처 엔지니어링...")

def create_sequence_features(df):
    """시퀀스 기반 피처 생성"""
    df = df.copy()

    # 기본 이동량 (현재 액션)
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # 골문까지 거리/각도 (골문: x=105, y=34)
    df['dist_to_goal'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['angle_to_goal'] = np.arctan2(34 - df['start_y'], 105 - df['start_x'])

    # 필드 영역 (0: 수비, 1: 중앙, 2: 공격)
    df['field_zone_x'] = pd.cut(df['start_x'], bins=[0, 35, 70, 105], labels=[0, 1, 2]).astype(float)
    df['field_zone_y'] = pd.cut(df['start_y'], bins=[0, 22.67, 45.33, 68], labels=[0, 1, 2]).astype(float)

    # 시퀀스 피처 (그룹별 shift)
    for col in ['dx', 'dy', 'distance', 'start_x', 'start_y']:
        # 직전 1개
        df[f'prev_1_{col}'] = df.groupby('game_episode')[col].shift(1)
        # 직전 2개
        df[f'prev_2_{col}'] = df.groupby('game_episode')[col].shift(2)
        # 직전 3개
        df[f'prev_3_{col}'] = df.groupby('game_episode')[col].shift(3)

    # 이전 N개 평균 (rolling)
    for col in ['dx', 'dy', 'distance']:
        df[f'prev_avg_3_{col}'] = df.groupby('game_episode')[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df[f'prev_avg_5_{col}'] = df.groupby('game_episode')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )

    # 시간 관련 피처
    df['time_delta'] = df.groupby('game_episode')['time_seconds'].diff()
    df['cumulative_time'] = df.groupby('game_episode')['time_seconds'].transform(
        lambda x: x - x.iloc[0]
    )

    # 액션 순서 관련
    df['action_order'] = df.groupby('game_episode').cumcount()
    df['total_actions'] = df.groupby('game_episode')['action_id'].transform('count')
    df['action_ratio'] = df['action_order'] / df['total_actions']

    # 패스 체인 관련 (연속 Pass 액션 수)
    df['is_pass'] = (df['type_name'] == 'Pass').astype(int)

    # 이동 방향 (각도)
    df['move_angle'] = np.arctan2(df['dy'], df['dx'])
    df['prev_1_move_angle'] = df.groupby('game_episode')['move_angle'].shift(1)

    # 방향 변화량
    df['angle_change'] = df['move_angle'] - df['prev_1_move_angle']

    # NaN 처리 (첫 번째 액션)
    df = df.fillna(0)

    return df

print("  Train 피처 생성 중...")
train_df = create_sequence_features(train_df)

print("  Test 피처 생성 중...")
test_all = create_sequence_features(test_all)

print(f"  총 피처 수: {len(train_df.columns)}")

# =============================================================================
# 3. 학습 데이터 준비 (마지막 액션만)
# =============================================================================
print("\n[3] 학습 데이터 준비...")

# 마지막 액션 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])

test_last = test_all.groupby('game_episode').last().reset_index()

print(f"Train last actions: {len(train_last):,}")
print(f"Test last actions: {len(test_last):,}")

# 피처 선택
feature_cols = [
    # 현재 위치
    'start_x', 'start_y',
    # 골문 관련
    'dist_to_goal', 'angle_to_goal',
    # 필드 영역
    'field_zone_x', 'field_zone_y',
    # 직전 패스 정보
    'prev_1_dx', 'prev_1_dy', 'prev_1_distance',
    'prev_1_start_x', 'prev_1_start_y',
    # 직전 2개 패스
    'prev_2_dx', 'prev_2_dy', 'prev_2_distance',
    # 직전 3개 패스
    'prev_3_dx', 'prev_3_dy', 'prev_3_distance',
    # 이동 평균
    'prev_avg_3_dx', 'prev_avg_3_dy', 'prev_avg_3_distance',
    'prev_avg_5_dx', 'prev_avg_5_dy', 'prev_avg_5_distance',
    # 시간 관련
    'time_delta', 'cumulative_time',
    # 액션 순서
    'action_order', 'total_actions', 'action_ratio',
    # 방향
    'prev_1_move_angle', 'angle_change',
    # 경기 정보
    'period_id',
]

# 존재하는 피처만 선택
feature_cols = [c for c in feature_cols if c in train_last.columns]
print(f"사용할 피처 수: {len(feature_cols)}")

X = train_last[feature_cols].values
y_x = train_last['end_x'].values
y_y = train_last['end_y'].values
game_ids = train_last['game_id'].values

X_test = test_last[feature_cols].values

# =============================================================================
# 4. GroupKFold 교차 검증
# =============================================================================
print("\n[4] GroupKFold 교차 검증 (game_id 기준)...")

# LightGBM 파라미터 (강한 정규화)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.5,      # L1 정규화
    'reg_lambda': 0.5,     # L2 정규화
    'min_child_samples': 50,  # 과적합 방지
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

oof_pred_x = np.zeros(len(X))
oof_pred_y = np.zeros(len(X))
test_pred_x = np.zeros(len(X_test))
test_pred_y = np.zeros(len(X_test))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}/{n_splits}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]

    # X 좌표 모델
    train_data_x = lgb.Dataset(X_train, label=y_train_x)
    val_data_x = lgb.Dataset(X_val, label=y_val_x, reference=train_data_x)

    model_x = lgb.train(
        params,
        train_data_x,
        num_boost_round=1000,
        valid_sets=[val_data_x],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Y 좌표 모델
    train_data_y = lgb.Dataset(X_train, label=y_train_y)
    val_data_y = lgb.Dataset(X_val, label=y_val_y, reference=train_data_y)

    model_y = lgb.train(
        params,
        train_data_y,
        num_boost_round=1000,
        valid_sets=[val_data_y],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # OOF 예측
    oof_pred_x[val_idx] = model_x.predict(X_val)
    oof_pred_y[val_idx] = model_y.predict(X_val)

    # 테스트 예측
    test_pred_x += model_x.predict(X_test) / n_splits
    test_pred_y += model_y.predict(X_test) / n_splits

    # Fold 점수
    fold_dist = np.sqrt((oof_pred_x[val_idx] - y_val_x)**2 + (oof_pred_y[val_idx] - y_val_y)**2)
    fold_score = fold_dist.mean()
    fold_scores.append(fold_score)
    print(f"    Fold {fold+1} Score: {fold_score:.4f}")

# 전체 OOF 점수
oof_dist = np.sqrt((oof_pred_x - y_x)**2 + (oof_pred_y - y_y)**2)
oof_score = oof_dist.mean()

print("\n" + "=" * 70)
print(f"CV Score (GroupKFold): {oof_score:.4f}")
print(f"Fold Scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Std: {np.std(fold_scores):.4f}")
print("=" * 70)

# =============================================================================
# 5. 피처 중요도 분석
# =============================================================================
print("\n[5] 피처 중요도 (마지막 Fold 기준)...")

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance_x': model_x.feature_importance(),
    'importance_y': model_y.feature_importance()
})
importance['importance_total'] = importance['importance_x'] + importance['importance_y']
importance = importance.sort_values('importance_total', ascending=False)

print("\nTop 15 피처:")
for i, row in importance.head(15).iterrows():
    print(f"  {row['feature']:<25} {row['importance_total']:>6}")

# =============================================================================
# 6. 제출 파일 생성
# =============================================================================
print("\n[6] 제출 파일 생성...")

# 클리핑
test_pred_x = np.clip(test_pred_x, 0, 105)
test_pred_y = np.clip(test_pred_y, 0, 68)

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_pred_x,
    'end_y': test_pred_y
})

# sample_submission 순서에 맞추기
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_sequence_lgbm.csv', index=False)

print(f"  submission_sequence_lgbm.csv 저장 완료")
print(f"  예측 X 범위: [{test_pred_x.min():.2f}, {test_pred_x.max():.2f}]")
print(f"  예측 Y 범위: [{test_pred_y.min():.2f}, {test_pred_y.max():.2f}]")

# =============================================================================
# 7. Zone Baseline과 비교
# =============================================================================
print("\n[7] Zone Baseline과 비교...")

# 6x6 Zone Baseline 재현
def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# Zone Baseline CV 점수
zone_pred_x = []
zone_pred_y = []
for _, row in train_last.iterrows():
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    dx = zone_stats['delta_x'].get(zone, 0)
    dy = zone_stats['delta_y'].get(zone, 0)
    zone_pred_x.append(np.clip(row['start_x'] + dx, 0, 105))
    zone_pred_y.append(np.clip(row['start_y'] + dy, 0, 68))

zone_pred_x = np.array(zone_pred_x)
zone_pred_y = np.array(zone_pred_y)

zone_dist = np.sqrt((zone_pred_x - y_x)**2 + (zone_pred_y - y_y)**2)
zone_score = zone_dist.mean()

print(f"\n  Zone Baseline (6x6 median) CV: {zone_score:.4f}")
print(f"  Sequence LightGBM CV:          {oof_score:.4f}")
print(f"  차이:                          {oof_score - zone_score:+.4f}")

if oof_score < zone_score:
    print("\n  ✅ 시퀀스 LightGBM이 더 좋음!")
else:
    print("\n  ❌ Zone Baseline이 여전히 더 좋음")

# =============================================================================
# 8. 하이브리드 앙상블 (Zone + ML)
# =============================================================================
print("\n[8] 하이브리드 앙상블 (Zone + ML)...")

best_ratio = None
best_hybrid_score = float('inf')

for zone_ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    ml_ratio = 1 - zone_ratio
    hybrid_x = zone_ratio * zone_pred_x + ml_ratio * oof_pred_x
    hybrid_y = zone_ratio * zone_pred_y + ml_ratio * oof_pred_y

    hybrid_dist = np.sqrt((hybrid_x - y_x)**2 + (hybrid_y - y_y)**2)
    hybrid_score = hybrid_dist.mean()

    if hybrid_score < best_hybrid_score:
        best_hybrid_score = hybrid_score
        best_ratio = zone_ratio

    print(f"  Zone {zone_ratio:.0%} + ML {ml_ratio:.0%}: CV = {hybrid_score:.4f}")

print(f"\n  Best Hybrid: Zone {best_ratio:.0%} + ML {1-best_ratio:.0%} = {best_hybrid_score:.4f}")

# 최적 하이브리드로 제출 파일 생성
test_last['zone'] = test_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
test_zone_pred_x = []
test_zone_pred_y = []
for _, row in test_last.iterrows():
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    dx = zone_stats['delta_x'].get(zone, 0)
    dy = zone_stats['delta_y'].get(zone, 0)
    test_zone_pred_x.append(np.clip(row['start_x'] + dx, 0, 105))
    test_zone_pred_y.append(np.clip(row['start_y'] + dy, 0, 68))

test_zone_pred_x = np.array(test_zone_pred_x)
test_zone_pred_y = np.array(test_zone_pred_y)

hybrid_test_x = best_ratio * test_zone_pred_x + (1 - best_ratio) * test_pred_x
hybrid_test_y = best_ratio * test_zone_pred_y + (1 - best_ratio) * test_pred_y

hybrid_submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': np.clip(hybrid_test_x, 0, 105),
    'end_y': np.clip(hybrid_test_y, 0, 68)
})
hybrid_submission = sample_sub[['game_episode']].merge(hybrid_submission, on='game_episode', how='left')
hybrid_submission.to_csv('submission_hybrid_zone_ml.csv', index=False)

print(f"\n  submission_hybrid_zone_ml.csv 저장 완료")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)
print(f"\n제출 파일:")
print(f"  1. submission_sequence_lgbm.csv     - CV: {oof_score:.4f}")
print(f"  2. submission_hybrid_zone_ml.csv    - CV: {best_hybrid_score:.4f}")
print(f"\n비교:")
print(f"  Zone Baseline (6x6):  CV: {zone_score:.4f} → Public: 16.85")
print(f"  Sequence LightGBM:    CV: {oof_score:.4f}")
print(f"  Hybrid (Zone+ML):     CV: {best_hybrid_score:.4f}")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
