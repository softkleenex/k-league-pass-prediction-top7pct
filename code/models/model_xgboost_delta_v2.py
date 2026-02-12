"""
K리그 패스 좌표 예측 - XGBoost Delta (개선판)

전략:
- Zone 통계 베이스라인 대비 XGBoost로 delta(dx, dy) 예측
- 시퀀스 피처 활용
- GroupKFold로 Fold 1-3 검증
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
print("K리그 패스 좌표 예측 - XGBoost Delta v2")
print("=" * 80)

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

# =============================================================================
# 2. 피처 준비
# =============================================================================
print("\n[2] 피처 준비...")

def prepare_features(df):
    df = df.copy()
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']

    # 이전 이동량
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    # 이전 2개, 3개
    df['prev_2_dx'] = df.groupby('game_episode')['dx'].shift(2).fillna(0)
    df['prev_2_dy'] = df.groupby('game_episode')['dy'].shift(2).fillna(0)
    df['prev_3_dx'] = df.groupby('game_episode')['dx'].shift(3).fillna(0)
    df['prev_3_dy'] = df.groupby('game_episode')['dy'].shift(3).fillna(0)

    return df

train_df = prepare_features(train_df)
test_all = prepare_features(test_all)

# 마지막 액션 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])

test_last = test_all.groupby('game_episode').last().reset_index()

print(f"Train: {len(train_last):,} episodes")
print(f"Test: {len(test_last):,} episodes")

# =============================================================================
# 3. 피처 준비
# =============================================================================
print("\n[3] 피처 선택...")

feature_cols = [
    'start_x', 'start_y',
    'prev_dx', 'prev_dy',
    'prev_2_dx', 'prev_2_dy',
    'prev_3_dx', 'prev_3_dy',
]

X = train_last[feature_cols].values.astype(np.float32)
y_dx = (train_last['end_x'] - train_last['start_x']).values.astype(np.float32)
y_dy = (train_last['end_y'] - train_last['start_y']).values.astype(np.float32)
game_ids = train_last['game_id'].values

X_test = test_last[feature_cols].values.astype(np.float32)

print(f"Features: {X.shape}")
print(f"Target DX: {y_dx.shape}")
print(f"Target DY: {y_dy.shape}")

# =============================================================================
# 4. GroupKFold 교차 검증
# =============================================================================
print("\n[4] GroupKFold 교차 검증...")

gkf = GroupKFold(n_splits=5)

oof_pred_dx = np.zeros(len(X))
oof_pred_dy = np.zeros(len(X))
test_pred_dx = np.zeros(len(X_test))
test_pred_dy = np.zeros(len(X_test))

fold_scores = []

params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 42,
    'verbosity': 0,
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}/5")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_dx, y_val_dx = y_dx[train_idx], y_dx[val_idx]
    y_train_dy, y_val_dy = y_dy[train_idx], y_dy[val_idx]

    # DX 모델
    dtrain_dx = xgb.DMatrix(X_train, label=y_train_dx)
    dval_dx = xgb.DMatrix(X_val, label=y_val_dx)

    model_dx = xgb.train(
        params,
        dtrain_dx,
        num_boost_round=100,
        evals=[(dtrain_dx, 'train'), (dval_dx, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # DY 모델
    dtrain_dy = xgb.DMatrix(X_train, label=y_train_dy)
    dval_dy = xgb.DMatrix(X_val, label=y_val_dy)

    model_dy = xgb.train(
        params,
        dtrain_dy,
        num_boost_round=100,
        evals=[(dtrain_dy, 'train'), (dval_dy, 'val')],
        early_stopping_rounds=20,
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

    # Fold 점수: start + delta 예측값과 실제 end 비교
    pred_end_x = train_last.iloc[val_idx]['start_x'].values + oof_pred_dx[val_idx]
    pred_end_y = train_last.iloc[val_idx]['start_y'].values + oof_pred_dy[val_idx]

    actual_end_x = train_last.iloc[val_idx]['end_x'].values
    actual_end_y = train_last.iloc[val_idx]['end_y'].values

    fold_dist = np.sqrt((pred_end_x - actual_end_x)**2 + (pred_end_y - actual_end_y)**2)
    fold_score = fold_dist.mean()
    fold_scores.append(fold_score)

    print(f"    CV Score: {fold_score:.4f}")

# 전체 OOF 점수
pred_end_x = train_last['start_x'].values + oof_pred_dx
pred_end_y = train_last['start_y'].values + oof_pred_dy
oof_dist = np.sqrt((pred_end_x - train_last['end_x'].values)**2 +
                   (pred_end_y - train_last['end_y'].values)**2)
oof_score = oof_dist.mean()

fold13_scores = fold_scores[:3]
fold45_scores = fold_scores[3:]

print("\n" + "=" * 80)
print("교차 검증 결과")
print("=" * 80)
print(f"\nFold 1-3: {[f'{s:.4f}' for s in fold13_scores]}")
print(f"  평균: {np.mean(fold13_scores):.4f} ± {np.std(fold13_scores):.4f}")
print(f"\nFold 4-5: {[f'{s:.4f}' for s in fold45_scores]}")
print(f"  평균: {np.mean(fold45_scores):.4f}")
print(f"\nFold Gap: {np.mean(fold45_scores) - np.mean(fold13_scores):+.4f}")
print(f"OOF Score: {oof_score:.4f}")

# =============================================================================
# 5. 제출 파일 생성
# =============================================================================
print("\n[5] 제출 파일 생성...")

# 예측된 delta를 시작점에 더해서 end 좌표 계산
test_pred_end_x = np.clip(test_last['start_x'].values + test_pred_dx, 0, 105)
test_pred_end_y = np.clip(test_last['start_y'].values + test_pred_dy, 0, 68)

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_pred_end_x,
    'end_y': test_pred_end_y
})

submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_xgboost_delta_v2.csv', index=False)

print(f"  submission_xgboost_delta_v2.csv 저장 완료")
print(f"  예측 X 범위: [{test_pred_end_x.min():.2f}, {test_pred_end_x.max():.2f}]")
print(f"  예측 Y 범위: [{test_pred_end_y.min():.2f}, {test_pred_end_y.max():.2f}]")

# =============================================================================
# 6. Zone Baseline과 비교
# =============================================================================
print("\n[6] Zone Baseline과 비교...")

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

# Zone Baseline 평가
zone_pred_x = []
zone_pred_y = []
for _, row in train_last.iterrows():
    zone = row['zone']
    dx = zone_stats['delta_x'].get(zone, 0)
    dy = zone_stats['delta_y'].get(zone, 0)
    zone_pred_x.append(np.clip(row['start_x'] + dx, 0, 105))
    zone_pred_y.append(np.clip(row['start_y'] + dy, 0, 68))

zone_dist = np.sqrt((np.array(zone_pred_x) - train_last['end_x'].values)**2 +
                    (np.array(zone_pred_y) - train_last['end_y'].values)**2)
zone_score = zone_dist.mean()

print(f"\n  Zone 6x6 Baseline:    {zone_score:.4f}")
print(f"  XGBoost Delta v2:     {oof_score:.4f}")
print(f"  차이:                 {oof_score - zone_score:+.4f}")

if oof_score < zone_score:
    print(f"  Status: ✅ XGBoost이 더 나음 ({zone_score - oof_score:.4f})")
else:
    print(f"  Status: Zone이 더 나음 ({zone_score - oof_score:+.4f})")

# =============================================================================
# 7. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

print(f"\n[성능]")
print(f"  Fold 1-3 평균: {np.mean(fold13_scores):.4f} ± {np.std(fold13_scores):.4f}")
print(f"  Fold 4-5 평균: {np.mean(fold45_scores):.4f}")
print(f"  OOF Score:     {oof_score:.4f}")

print(f"\n[제출 파일]")
print(f"  submission_xgboost_delta_v2.csv")

print(f"\n[판단]")
cv_13 = np.mean(fold13_scores)
if cv_13 < 16.27:
    print(f"  경고: CV {cv_13:.4f} - 과최적화 위험")
elif cv_13 < 16.35:
    print(f"  좋음: CV {cv_13:.4f} - Sweet Spot 범위")
else:
    print(f"  기준: CV {cv_13:.4f}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
