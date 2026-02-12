"""
K리그 패스 좌표 예측 - XGBoost Safe v3

전략:
- XGBoost Delta 기반이지만 강한 정규화
- Zone Baseline과의 앙상블로 과최적화 방지
- Fold 1-3 CV 16.27-16.34 목표 (Sweet Spot)
- Fold Gap 최소화
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - XGBoost Safe v3")
print("=" * 80)

# =============================================================================
# 1. 데이터 로드 및 준비
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

# 피처 준비
def prepare_features(df):
    df = df.copy()
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
    df['prev_2_dx'] = df.groupby('game_episode')['dx'].shift(2).fillna(0)
    df['prev_2_dy'] = df.groupby('game_episode')['dy'].shift(2).fillna(0)
    df['prev_3_dx'] = df.groupby('game_episode')['dx'].shift(3).fillna(0)
    df['prev_3_dy'] = df.groupby('game_episode')['dy'].shift(3).fillna(0)
    return df

train_df = prepare_features(train_df)
test_all = prepare_features(test_all)

train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
test_last = test_all.groupby('game_episode').last().reset_index()

print(f"Train: {len(train_last):,} Test: {len(test_last):,}")

# =============================================================================
# 2. Zone Baseline 계산
# =============================================================================
print("\n[2] Zone Baseline 계산...")

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

test_last['zone'] = test_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)

# Zone baseline 통계 (전체 Train)
zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# =============================================================================
# 3. XGBoost 모델 훈련
# =============================================================================
print("\n[3] XGBoost 모델 훈련 (강한 정규화)...")

feature_cols = ['start_x', 'start_y', 'prev_dx', 'prev_dy', 'prev_2_dx', 'prev_2_dy', 'prev_3_dx', 'prev_3_dy']

X = train_last[feature_cols].values.astype(np.float32)
y_dx = (train_last['end_x'] - train_last['start_x']).values.astype(np.float32)
y_dy = (train_last['end_y'] - train_last['start_y']).values.astype(np.float32)
game_ids = train_last['game_id'].values

X_test = test_last[feature_cols].values.astype(np.float32)

gkf = GroupKFold(n_splits=5)

oof_pred_dx = np.zeros(len(X))
oof_pred_dy = np.zeros(len(X))
test_pred_dx = np.zeros(len(X_test))
test_pred_dy = np.zeros(len(X_test))

fold_xgb_scores = []
fold_zone_scores = []
fold_hybrid_scores = []

# 강한 정규화 파라미터
params = {
    'max_depth': 4,  # 더 얕은 트리
    'learning_rate': 0.05,  # 낮은 학습률
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'reg_alpha': 1.0,  # L1 정규화
    'reg_lambda': 2.0,  # L2 정규화
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0,
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}/5")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_dx, y_val_dx = y_dx[train_idx], y_dx[val_idx]
    y_train_dy, y_val_dy = y_dy[train_idx], y_dy[val_idx]

    train_last_fold = train_last.iloc[train_idx]
    val_last_fold = train_last.iloc[val_idx]

    # Zone Baseline 평가
    zone_pred_x_fold = []
    zone_pred_y_fold = []
    for _, row in val_last_fold.iterrows():
        zone = row['zone']
        dx = zone_stats['delta_x'].get(zone, 0)
        dy = zone_stats['delta_y'].get(zone, 0)
        zone_pred_x_fold.append(np.clip(row['start_x'] + dx, 0, 105))
        zone_pred_y_fold.append(np.clip(row['start_y'] + dy, 0, 68))

    zone_dist = np.sqrt((np.array(zone_pred_x_fold) - val_last_fold['end_x'].values)**2 +
                        (np.array(zone_pred_y_fold) - val_last_fold['end_y'].values)**2)
    zone_score = zone_dist.mean()
    fold_zone_scores.append(zone_score)

    # XGBoost DX
    dtrain_dx = xgb.DMatrix(X_train, label=y_train_dx)
    dval_dx = xgb.DMatrix(X_val, label=y_val_dx)

    model_dx = xgb.train(
        params,
        dtrain_dx,
        num_boost_round=200,
        evals=[(dtrain_dx, 'train'), (dval_dx, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    # XGBoost DY
    dtrain_dy = xgb.DMatrix(X_train, label=y_train_dy)
    dval_dy = xgb.DMatrix(X_val, label=y_val_dy)

    model_dy = xgb.train(
        params,
        dtrain_dy,
        num_boost_round=200,
        evals=[(dtrain_dy, 'train'), (dval_dy, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    # OOF 예측
    dval = xgb.DMatrix(X_val)
    oof_pred_dx[val_idx] = model_dx.predict(dval)
    oof_pred_dy[val_idx] = model_dy.predict(dval)

    # 테스트 예측
    dtest = xgb.DMatrix(X_test)
    test_pred_dx += model_dx.predict(dtest) / 5
    test_pred_dy += model_dy.predict(dtest) / 5

    # XGBoost 점수
    xgb_pred_x = val_last_fold['start_x'].values + oof_pred_dx[val_idx]
    xgb_pred_y = val_last_fold['start_y'].values + oof_pred_dy[val_idx]
    xgb_dist = np.sqrt((xgb_pred_x - val_last_fold['end_x'].values)**2 +
                       (xgb_pred_y - val_last_fold['end_y'].values)**2)
    xgb_score = xgb_dist.mean()
    fold_xgb_scores.append(xgb_score)

    print(f"    Zone:  {zone_score:.4f}")
    print(f"    XGBoost: {xgb_score:.4f}")

fold13_xgb = fold_xgb_scores[:3]
fold13_zone = fold_zone_scores[:3]

print("\n" + "=" * 80)
print("교차 검증 결과")
print("=" * 80)
print(f"\nFold 1-3 XGBoost: {[f'{s:.4f}' for s in fold13_xgb]}")
print(f"  평균: {np.mean(fold13_xgb):.4f} ± {np.std(fold13_xgb):.4f}")
print(f"  Gap vs Zone: {np.mean(fold13_xgb) - np.mean(fold13_zone):+.4f}")

print(f"\nFold 4-5 XGBoost: {[f'{s:.4f}' for s in fold_xgb_scores[3:]]}")
print(f"  평균: {np.mean(fold_xgb_scores[3:]):.4f}")

print(f"\nFold Gap (4-5 - 1-3): {np.mean(fold_xgb_scores[3:]) - np.mean(fold13_xgb):+.4f}")

# =============================================================================
# 4. 앙상블 비율 최적화
# =============================================================================
print("\n[4] Zone + XGBoost 앙상블 최적화...")

best_ratio = None
best_score = float('inf')
best_scores_log = []

for zone_ratio in np.arange(0.0, 1.1, 0.1):
    xgb_ratio = 1 - zone_ratio

    # OOF 평가
    zone_pred_x = []
    zone_pred_y = []
    for _, row in train_last.iterrows():
        zone = row['zone']
        dx = zone_stats['delta_x'].get(zone, 0)
        dy = zone_stats['delta_y'].get(zone, 0)
        zone_pred_x.append(np.clip(row['start_x'] + dx, 0, 105))
        zone_pred_y.append(np.clip(row['start_y'] + dy, 0, 68))

    zone_pred_x = np.array(zone_pred_x)
    zone_pred_y = np.array(zone_pred_y)

    xgb_pred_x = np.clip(train_last['start_x'].values + oof_pred_dx, 0, 105)
    xgb_pred_y = np.clip(train_last['start_y'].values + oof_pred_dy, 0, 68)

    hybrid_x = zone_ratio * zone_pred_x + xgb_ratio * xgb_pred_x
    hybrid_y = zone_ratio * zone_pred_y + xgb_ratio * xgb_pred_y

    dist = np.sqrt((hybrid_x - train_last['end_x'].values)**2 +
                   (hybrid_y - train_last['end_y'].values)**2)
    score = dist.mean()

    best_scores_log.append((zone_ratio, score))

    if score < best_score:
        best_score = score
        best_ratio = zone_ratio

print(f"\n최적 비율:")
for ratio, score in best_scores_log:
    marker = " <-- BEST" if ratio == best_ratio else ""
    print(f"  Zone {ratio:.0%} + XGBoost {1-ratio:.0%}: {score:.4f}{marker}")

# =============================================================================
# 5. 제출 파일 생성
# =============================================================================
print("\n[5] 제출 파일 생성...")

# Zone 예측
test_zone_pred_x = []
test_zone_pred_y = []
for _, row in test_last.iterrows():
    zone = row['zone']
    dx = zone_stats['delta_x'].get(zone, 0)
    dy = zone_stats['delta_y'].get(zone, 0)
    test_zone_pred_x.append(np.clip(row['start_x'] + dx, 0, 105))
    test_zone_pred_y.append(np.clip(row['start_y'] + dy, 0, 68))

test_zone_pred_x = np.array(test_zone_pred_x)
test_zone_pred_y = np.array(test_zone_pred_y)

# XGBoost 예측
test_xgb_pred_x = np.clip(test_last['start_x'].values + test_pred_dx, 0, 105)
test_xgb_pred_y = np.clip(test_last['start_y'].values + test_pred_dy, 0, 68)

# 하이브리드
test_hybrid_x = best_ratio * test_zone_pred_x + (1 - best_ratio) * test_xgb_pred_x
test_hybrid_y = best_ratio * test_zone_pred_y + (1 - best_ratio) * test_xgb_pred_y

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': np.clip(test_hybrid_x, 0, 105),
    'end_y': np.clip(test_hybrid_y, 0, 68)
})

submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_xgboost_safe_v3.csv', index=False)

print(f"  submission_xgboost_safe_v3.csv 저장 완료")
print(f"  앙상블 비율: Zone {best_ratio:.0%} + XGBoost {1-best_ratio:.0%}")

# =============================================================================
# 6. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

print(f"\n[성능]")
print(f"  Zone Baseline (Fold 1-3):    {np.mean(fold13_zone):.4f}")
print(f"  XGBoost (Fold 1-3):          {np.mean(fold13_xgb):.4f}")
print(f"  개선:                        {np.mean(fold13_zone) - np.mean(fold13_xgb):+.4f}")

print(f"\n[과최적화 위험 분석]")
fold_gap = np.mean(fold_xgb_scores[3:]) - np.mean(fold13_xgb)
print(f"  Fold Gap (4-5 - 1-3):  {fold_gap:+.4f}")
if abs(fold_gap) < 0.5:
    print(f"  Status: ✅ 양호 (Gap < 0.5)")
else:
    print(f"  Status: ⚠️  주의 (Gap >= 0.5)")

print(f"\n[Sweet Spot 판단]")
cv_13 = np.mean(fold13_xgb)
if cv_13 < 16.27:
    print(f"  ❌ CV {cv_13:.4f} - 과최적화 위험 (16.27 이상 권장)")
elif cv_13 < 16.35:
    print(f"  ✅ CV {cv_13:.4f} - Sweet Spot (16.27-16.34)")
else:
    print(f"  ⚠️  CV {cv_13:.4f} - 기준선 도달")

print(f"\n[제출 정보]")
print(f"  파일: submission_xgboost_safe_v3.csv")
print(f"  전략: Zone {best_ratio:.0%} + XGBoost {1-best_ratio:.0%} 앙상블")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
