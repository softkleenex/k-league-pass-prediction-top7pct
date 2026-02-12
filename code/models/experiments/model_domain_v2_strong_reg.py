"""
Domain v2 Model - Strong Regularization + Reduced Features

목적:
- Domain v1 (CV 16.12, Gap 0.60)의 Gap 축소
- 피처 단순화: 10개 → 6개
- 정규화 강화로 일반화 개선

개선점 (vs v1):
- 피처 축소: zone_6x6, direction_8way, is_near_goal, field_zone 제거
- 정규화 대폭 강화: reg_alpha 0.5→2.0, reg_lambda 0.5→3.0
- 트리 단순화: max_depth 5→4, min_child_samples 50→100
- 학습률 감소: 0.05→0.03

예상:
- CV: 16.2-16.3 (v1 대비 약간 상승 허용)
- Gap: 0.30-0.40 (v1 0.60 대비 절반)
- Public: 16.5-16.7

제출 조건: CV < 16.3 AND Gap < 0.5

날짜: 2025-12-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
DATA_DIR = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
SUBMISSION_DIR = DATA_DIR / "submissions" / "experiments"
SUBMISSION_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("Domain v2 Model - Strong Regularization + Reduced Features")
print("=" * 80)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_meta = pd.read_csv(DATA_DIR / "test.csv")

# Test 데이터는 개별 CSV 파일로 저장됨
test_episodes = []
for _, row in test_meta.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_df = pd.concat(test_episodes, ignore_index=True)

print(f"Train: {len(train_df):,} passes")
print(f"Test:  {len(test_df):,} passes (from {len(test_meta)} episodes)")

# =============================================================================
# 2. 피처 엔지니어링 (6개만)
# =============================================================================
print("\n[2] 피처 엔지니어링 (6개 피처)...")

def create_features(df):
    """6개 피처 생성 (No Target Encoding, No Zone/Direction)"""
    df = df.copy()

    # 기본 좌표 (2)
    # start_x, start_y (이미 존재)

    # 이전 패스 벡터 (2)
    df['dx'] = df['end_x'] - df['start_x'] if 'end_x' in df.columns else 0
    df['dy'] = df['end_y'] - df['start_y'] if 'end_y' in df.columns else 0
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    # 골대 관련 (2)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.arctan2(34 - df['start_y'], 105 - df['start_x'])

    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

feature_cols = [
    'start_x', 'start_y',
    'prev_dx', 'prev_dy',
    'goal_distance', 'goal_angle'
]

print(f"피처 수: {len(feature_cols)}")
print(f"피처 목록: {feature_cols}")

# =============================================================================
# 3. ALL PASSES 학습 데이터 준비
# =============================================================================
print("\n[3] ALL PASSES 학습 데이터 준비...")

# Train: 모든 패스를 학습에 사용 (356,721개)
X_all = train_df[feature_cols].fillna(0)
y_x_all = train_df['end_x'] - train_df['start_x']
y_y_all = train_df['end_y'] - train_df['start_y']
game_ids_all = train_df['game_id'].values

print(f"학습 샘플 수: {len(X_all):,} (ALL passes)")

# 마지막 패스만 추출 (CV 평가용)
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])

print(f"평가 샘플 수: {len(train_last):,} (LAST passes only)")

# =============================================================================
# 4. GroupKFold CV (Fold 1-3만)
# =============================================================================
print("\n[4] GroupKFold CV (Fold 1-3, ALL passes 학습)...")

gkf = GroupKFold(n_splits=5)
folds_to_use = [1, 2, 3]

cv_scores = []
models_x = []
models_y = []

# LightGBM 파라미터 (강한 정규화)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 300,
    'max_depth': 4,              # v1: 5 → v2: 4
    'num_leaves': 15,            # 2^4 - 1
    'min_child_samples': 100,    # v1: 50 → v2: 100
    'learning_rate': 0.03,       # v1: 0.05 → v2: 0.03
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 2.0,            # v1: 0.5 → v2: 2.0
    'reg_lambda': 3.0,           # v1: 0.5 → v2: 3.0
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1
}

print("\n정규화 설정:")
print(f"  max_depth: {lgb_params['max_depth']}")
print(f"  min_child_samples: {lgb_params['min_child_samples']}")
print(f"  learning_rate: {lgb_params['learning_rate']}")
print(f"  reg_alpha: {lgb_params['reg_alpha']}")
print(f"  reg_lambda: {lgb_params['reg_lambda']}")

# Last pass의 인덱스를 미리 추출
last_pass_indices = train_df.groupby('game_episode').tail(1).index

for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_all, groups=game_ids_all)):
    if fold_idx + 1 not in folds_to_use:
        continue

    print(f"\n  Fold {fold_idx + 1}:")

    # ALL PASSES로 학습
    X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_x_train, y_x_val = y_x_all.iloc[train_idx], y_x_all.iloc[val_idx]
    y_y_train, y_y_val = y_y_all.iloc[train_idx], y_y_all.iloc[val_idx]

    print(f"    Train: {len(X_train):,} passes")
    print(f"    Val:   {len(X_val):,} passes")

    # X 모델
    model_x = lgb.LGBMRegressor(**lgb_params)
    model_x.fit(X_train, y_x_train,
                eval_set=[(X_val, y_x_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

    # Y 모델
    model_y = lgb.LGBMRegressor(**lgb_params)
    model_y.fit(X_train, y_y_train,
                eval_set=[(X_val, y_y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

    # 평가: LAST PASS만 평가 (CV 측정)
    val_last_pass_mask = last_pass_indices.isin(val_idx)
    val_last_pass_indices = last_pass_indices[val_last_pass_mask]

    # 원본 train_df 인덱스를 X_all 인덱스로 매핑
    val_last_in_X_all = X_all.index.isin(val_last_pass_indices)

    if val_last_in_X_all.sum() == 0:
        print("    ⚠️ Warning: No last passes in validation set")
        continue

    X_val_last = X_all[val_last_in_X_all]

    pred_x = model_x.predict(X_val_last)
    pred_y = model_y.predict(X_val_last)

    # 실제 좌표 (원본 데이터에서 가져오기)
    val_last_df = train_df.loc[val_last_pass_indices].copy()
    val_last_df['pred_end_x'] = np.clip(val_last_df['start_x'].values + pred_x, 0, 105)
    val_last_df['pred_end_y'] = np.clip(val_last_df['start_y'].values + pred_y, 0, 68)

    # 평가
    distances = np.sqrt((val_last_df['pred_end_x'] - val_last_df['end_x'])**2 +
                        (val_last_df['pred_end_y'] - val_last_df['end_y'])**2)
    fold_cv = distances.mean()
    cv_scores.append(fold_cv)

    print(f"    Last pass 수: {len(val_last_df):,}")
    print(f"    CV: {fold_cv:.4f}")

    models_x.append(model_x)
    models_y.append(model_y)

avg_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)
print(f"\n  Fold 1-3 평균 CV: {avg_cv:.4f} ± {std_cv:.4f}")

# =============================================================================
# 5. Pipeline v2 Gap 예측
# =============================================================================
print("\n[5] Pipeline v2 Gap 예측...")

# Gap 예측 모델 (피처 수 기반)
def predict_gap(cv, feature_count, has_target_encoding):
    """Pipeline v2 Gap 예측 로직"""
    if feature_count <= 4:
        base_gap = 0.02
    elif feature_count <= 15:
        base_gap = 0.75
    else:
        base_gap = 1.25

    # Target Encoding 패널티
    if has_target_encoding:
        base_gap += 0.4

    # CV 기반 조정 (낮을수록 과적합 위험)
    if cv < 15.5:
        base_gap *= 1.2
    elif cv < 16.0:
        base_gap *= 1.0
    else:
        base_gap *= 0.8

    return base_gap

expected_gap = predict_gap(avg_cv, len(feature_cols), has_target_encoding=False)
expected_public = avg_cv + expected_gap

print(f"  Feature 수: {len(feature_cols)}")
print(f"  Target Encoding: False")
print(f"  예상 Gap: {expected_gap:.4f}")
print(f"  예상 Public: {expected_public:.4f}")

# =============================================================================
# 6. 제출 조건 확인
# =============================================================================
print("\n[6] 제출 조건 확인...")

SUBMIT_THRESHOLD_CV = 16.3
SUBMIT_THRESHOLD_GAP = 0.5

print(f"  CV < {SUBMIT_THRESHOLD_CV}: {avg_cv:.4f} < {SUBMIT_THRESHOLD_CV} → {'✅' if avg_cv < SUBMIT_THRESHOLD_CV else '❌'}")
print(f"  Gap < {SUBMIT_THRESHOLD_GAP}: {expected_gap:.4f} < {SUBMIT_THRESHOLD_GAP} → {'✅' if expected_gap < SUBMIT_THRESHOLD_GAP else '❌'}")

should_submit = (avg_cv < SUBMIT_THRESHOLD_CV) and (expected_gap < SUBMIT_THRESHOLD_GAP)

if should_submit:
    print("\n  ✅ 제출 조건 충족! 테스트 예측 진행...")
else:
    print("\n  ❌ 제출 조건 미충족. 모델 개선 필요.")

    # v1과 비교
    print("\n[Domain v1 대비]")
    print(f"  v1 CV: 16.12, v1 Gap: 0.60")
    print(f"  v2 CV: {avg_cv:.4f}, v2 Gap: {expected_gap:.4f}")

    if avg_cv < 16.12:
        print(f"  CV 개선: {16.12 - avg_cv:.4f} ✅")
    else:
        print(f"  CV 악화: {avg_cv - 16.12:.4f} ❌")

    if expected_gap < 0.60:
        print(f"  Gap 개선: {0.60 - expected_gap:.4f} ✅")
    else:
        print(f"  Gap 악화: {expected_gap - 0.60:.4f} ❌")

    print(f"\n[종료] 제출하지 않음.")
    import sys
    sys.exit(0)

# =============================================================================
# 7. 테스트 예측 (Fold 1-3 평균)
# =============================================================================
print("\n[7] 테스트 예측 (Fold 1-3 평균)...")

test_last = test_df.groupby('game_episode').last().reset_index()
X_test = test_last[feature_cols].fillna(0)

# Fold 1-3 모델 평균
pred_x_list = []
pred_y_list = []

for model_x, model_y in zip(models_x, models_y):
    pred_x = model_x.predict(X_test)
    pred_y = model_y.predict(X_test)
    pred_x_list.append(pred_x)
    pred_y_list.append(pred_y)

# 평균 예측
pred_x_avg = np.mean(pred_x_list, axis=0)
pred_y_avg = np.mean(pred_y_list, axis=0)

test_last['pred_end_x'] = np.clip(test_last['start_x'] + pred_x_avg, 0, 105)
test_last['pred_end_y'] = np.clip(test_last['start_y'] + pred_y_avg, 0, 68)

print(f"  테스트 에피소드 수: {len(test_last):,}")

# =============================================================================
# 8. 제출 파일 생성
# =============================================================================
print("\n[8] 제출 파일 생성...")

submission = pd.DataFrame({
    'index': test_last['game_episode'].values,
    'x': test_last['pred_end_x'].values,
    'y': test_last['pred_end_y'].values
})

submission_filename = f"submission_domain_v2_strong_reg_cv{avg_cv:.4f}.csv"
submission_path = SUBMISSION_DIR / submission_filename
submission.to_csv(submission_path, index=False)

print(f"  파일 저장: {submission_path}")

# =============================================================================
# 9. 요약
# =============================================================================
print("\n" + "=" * 80)
print("Domain v2 완료!")
print("=" * 80)

print(f"\n[모델 정보]")
print(f"  피처 수: {len(feature_cols)} (v1: 10개 → v2: 6개)")
print(f"  Target Encoding: False")
print(f"  학습 샘플: {len(X_all):,} (ALL passes)")
print(f"  평가 샘플: {len(train_last):,} (LAST passes)")
print(f"  정규화: 매우 강함 (reg_alpha=2.0, reg_lambda=3.0)")

print(f"\n[성능]")
print(f"  Fold 1-3 평균 CV: {avg_cv:.4f} ± {std_cv:.4f}")
print(f"  예상 Gap: {expected_gap:.4f}")
print(f"  예상 Public: {expected_public:.4f}")

print(f"\n[Domain v1 대비]")
print(f"  v1: CV 16.12, Gap 0.60, Public 16.72")
print(f"  v2: CV {avg_cv:.4f}, Gap {expected_gap:.4f}, Public {expected_public:.4f}")

if avg_cv < 16.12 and expected_gap < 0.60:
    print(f"  상태: ✅ 양쪽 모두 개선!")
elif avg_cv < 16.12 or expected_gap < 0.60:
    print(f"  상태: ⚠️ 부분 개선")
else:
    print(f"  상태: ❌ 개선 없음")

print(f"\n[제출]")
print(f"  파일: {submission_filename}")
print(f"  제출 권장: {'✅ Yes' if should_submit else '❌ No'}")

if expected_public < 16.5:
    print(f"\n  🎉 목표 달성! (Public < 16.5)")
elif expected_public < 16.7:
    print(f"\n  ✅ 양호 (Public < 16.7)")
else:
    print(f"\n  ⚠️ 목표 미달 (Public >= 16.7)")

print("\n" + "=" * 80)
