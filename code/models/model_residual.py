"""
K리그 패스 좌표 예측 - Residual ML
Phase 3: Zone Baseline + ML Residual 보정
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
print("K리그 패스 좌표 예측 - Residual ML")
print("=" * 70)

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

# 마지막 액션 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

test_last = test_all.groupby('game_episode').last().reset_index()

print(f"Train: {len(train_last):,} rows")
print(f"Test: {len(test_last):,} rows")

# =============================================================================
# 2. Zone Baseline 예측 (5x5 median)
# =============================================================================
print("\n[2] Zone Baseline 예측 (5x5 median)...")

def get_zone_5x5(x, y):
    x_zone = min(4, int(x // 21))
    y_zone = min(4, int(y // 13.6))
    return x_zone * 5 + y_zone

train_last['zone'] = train_last.apply(
    lambda r: get_zone_5x5(r['start_x'], r['start_y']), axis=1
)
test_last['zone'] = test_last.apply(
    lambda r: get_zone_5x5(r['start_x'], r['start_y']), axis=1
)

zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

global_median_dx = train_last['delta_x'].median()
global_median_dy = train_last['delta_y'].median()

# Train baseline 예측
train_last['baseline_dx'] = train_last['zone'].map(
    lambda z: zone_stats['delta_x'].get(z, global_median_dx)
)
train_last['baseline_dy'] = train_last['zone'].map(
    lambda z: zone_stats['delta_y'].get(z, global_median_dy)
)
train_last['baseline_x'] = np.clip(train_last['start_x'] + train_last['baseline_dx'], 0, 105)
train_last['baseline_y'] = np.clip(train_last['start_y'] + train_last['baseline_dy'], 0, 68)

# Test baseline 예측
test_last['baseline_dx'] = test_last['zone'].map(
    lambda z: zone_stats['delta_x'].get(z, global_median_dx)
)
test_last['baseline_dy'] = test_last['zone'].map(
    lambda z: zone_stats['delta_y'].get(z, global_median_dy)
)
test_last['baseline_x'] = np.clip(test_last['start_x'] + test_last['baseline_dx'], 0, 105)
test_last['baseline_y'] = np.clip(test_last['start_y'] + test_last['baseline_dy'], 0, 68)

# Residual (잔차) 계산
train_last['residual_x'] = train_last['end_x'] - train_last['baseline_x']
train_last['residual_y'] = train_last['end_y'] - train_last['baseline_y']

print(f"Baseline CV: {np.sqrt((train_last['baseline_x'] - train_last['end_x'])**2 + (train_last['baseline_y'] - train_last['end_y'])**2).mean():.4f}")
print(f"Residual mean: dx={train_last['residual_x'].mean():.4f}, dy={train_last['residual_y'].mean():.4f}")
print(f"Residual std: dx={train_last['residual_x'].std():.4f}, dy={train_last['residual_y'].std():.4f}")

# =============================================================================
# 3. 최소 피처 추출
# =============================================================================
print("\n[3] 최소 피처 추출 (3개만)...")

# 가장 기본적인 피처만 사용
train_last['dist_to_goal'] = np.sqrt((105 - train_last['start_x'])**2 + (34 - train_last['start_y'])**2)
test_last['dist_to_goal'] = np.sqrt((105 - test_last['start_x'])**2 + (34 - test_last['start_y'])**2)

# 피처 컬럼 (극소)
feature_cols = ['start_x', 'start_y', 'dist_to_goal']

X_train = train_last[feature_cols].values
y_residual_x = train_last['residual_x'].values
y_residual_y = train_last['residual_y'].values

X_test = test_last[feature_cols].values

print(f"Features: {feature_cols}")

# =============================================================================
# 4. Residual ML 학습 (극강 정규화)
# =============================================================================
print("\n[4] Residual ML 학습 (극강 정규화)...")

# 극강 정규화 파라미터
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,       # 매우 낮은 학습률
    'num_leaves': 7,             # 극소
    'max_depth': 2,              # 매우 얕음
    'min_child_samples': 500,    # 매우 큼
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 5.0,            # 매우 강한 L1
    'reg_lambda': 5.0,           # 매우 강한 L2
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)
game_ids = train_last['game_id'].values

oof_residual_x = np.zeros(len(X_train))
oof_residual_y = np.zeros(len(X_train))
test_residual_x = np.zeros(len(X_test))
test_residual_y = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, groups=game_ids)):
    print(f"  Fold {fold+1}/{N_FOLDS}")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr_x, y_val_x = y_residual_x[train_idx], y_residual_x[val_idx]
    y_tr_y, y_val_y = y_residual_y[train_idx], y_residual_y[val_idx]

    # Residual X 모델
    model_x = lgb.train(
        params,
        lgb.Dataset(X_tr, label=y_tr_x),
        num_boost_round=300,
        valid_sets=[lgb.Dataset(X_val, label=y_val_x)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    # Residual Y 모델
    model_y = lgb.train(
        params,
        lgb.Dataset(X_tr, label=y_tr_y),
        num_boost_round=300,
        valid_sets=[lgb.Dataset(X_val, label=y_val_y)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    oof_residual_x[val_idx] = model_x.predict(X_val)
    oof_residual_y[val_idx] = model_y.predict(X_val)

    test_residual_x += model_x.predict(X_test) / N_FOLDS
    test_residual_y += model_y.predict(X_test) / N_FOLDS

print(f"\nResidual ML 예측 범위:")
print(f"  X: [{oof_residual_x.min():.2f}, {oof_residual_x.max():.2f}]")
print(f"  Y: [{oof_residual_y.min():.2f}, {oof_residual_y.max():.2f}]")

# =============================================================================
# 5. 최종 예측 및 평가
# =============================================================================
print("\n[5] 최종 예측 및 평가...")

# OOF 예측
final_oof_x = train_last['baseline_x'].values + oof_residual_x
final_oof_y = train_last['baseline_y'].values + oof_residual_y
final_oof_x = np.clip(final_oof_x, 0, 105)
final_oof_y = np.clip(final_oof_y, 0, 68)

# CV Score 계산
baseline_cv = np.sqrt(
    (train_last['baseline_x'].values - train_last['end_x'].values)**2 +
    (train_last['baseline_y'].values - train_last['end_y'].values)**2
).mean()

hybrid_cv = np.sqrt(
    (final_oof_x - train_last['end_x'].values)**2 +
    (final_oof_y - train_last['end_y'].values)**2
).mean()

print(f"\n결과 비교:")
print(f"  Baseline (5x5 median) CV: {baseline_cv:.4f}")
print(f"  Hybrid (Baseline + ML) CV: {hybrid_cv:.4f}")
print(f"  개선: {baseline_cv - hybrid_cv:.4f}")

# ML 보정이 악화시키는 경우
if hybrid_cv > baseline_cv:
    print(f"\n⚠️ ML 보정이 성능을 악화시킴! Baseline만 사용 권장")
    use_ml = False
else:
    print(f"\n✓ ML 보정이 도움됨")
    use_ml = True

# =============================================================================
# 6. 다양한 혼합 비율 테스트
# =============================================================================
print("\n[6] Baseline + ML 혼합 비율 테스트...")

best_ratio = 0
best_cv = baseline_cv

for ml_ratio in np.arange(0, 1.1, 0.1):
    baseline_ratio = 1 - ml_ratio
    mixed_x = baseline_ratio * train_last['baseline_x'].values + ml_ratio * final_oof_x
    mixed_y = baseline_ratio * train_last['baseline_y'].values + ml_ratio * final_oof_y
    mixed_cv = np.sqrt((mixed_x - train_last['end_x'].values)**2 + (mixed_y - train_last['end_y'].values)**2).mean()

    if mixed_cv < best_cv:
        best_cv = mixed_cv
        best_ratio = ml_ratio

    print(f"  Baseline {baseline_ratio*100:.0f}% + ML {ml_ratio*100:.0f}%: CV = {mixed_cv:.4f}")

print(f"\n최적 혼합: Baseline {(1-best_ratio)*100:.0f}% + ML {best_ratio*100:.0f}%")
print(f"최적 CV: {best_cv:.4f}")

# =============================================================================
# 7. 제출 파일 생성
# =============================================================================
print("\n[7] 제출 파일 생성...")

# 1. Baseline만
baseline_sub = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['baseline_x'],
    'end_y': test_last['baseline_y']
})
baseline_sub = sample_sub[['game_episode']].merge(baseline_sub, on='game_episode', how='left')
baseline_sub.to_csv('submission_5x5_baseline_only.csv', index=False)

# 2. Hybrid (최적 비율)
test_final_x = (1 - best_ratio) * test_last['baseline_x'].values + best_ratio * (test_last['baseline_x'].values + test_residual_x)
test_final_y = (1 - best_ratio) * test_last['baseline_y'].values + best_ratio * (test_last['baseline_y'].values + test_residual_y)
test_final_x = np.clip(test_final_x, 0, 105)
test_final_y = np.clip(test_final_y, 0, 68)

hybrid_sub = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_final_x,
    'end_y': test_final_y
})
hybrid_sub = sample_sub[['game_episode']].merge(hybrid_sub, on='game_episode', how='left')
hybrid_sub.to_csv('submission_hybrid_optimal.csv', index=False)

# 3. 아주 보수적인 Hybrid (ML 20%)
conservative_x = 0.8 * test_last['baseline_x'].values + 0.2 * (test_last['baseline_x'].values + test_residual_x)
conservative_y = 0.8 * test_last['baseline_y'].values + 0.2 * (test_last['baseline_y'].values + test_residual_y)
conservative_x = np.clip(conservative_x, 0, 105)
conservative_y = np.clip(conservative_y, 0, 68)

conservative_sub = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': conservative_x,
    'end_y': conservative_y
})
conservative_sub = sample_sub[['game_episode']].merge(conservative_sub, on='game_episode', how='left')
conservative_sub.to_csv('submission_hybrid_conservative.csv', index=False)

print(f"  submission_5x5_baseline_only.csv: CV={baseline_cv:.4f}")
print(f"  submission_hybrid_optimal.csv: CV={best_cv:.4f} (ML {best_ratio*100:.0f}%)")
print(f"  submission_hybrid_conservative.csv: (ML 20%)")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
