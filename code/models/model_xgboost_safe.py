"""
K리그 패스 좌표 예측 - 극도로 정규화된 XGBoost 모델

배경:
- 과거 LightGBM 실패: CV 8.55 → Public 20.57 (Gap +12.02)
- 원인: 과적합 (깊은 트리, 많은 피처)
- 목표: 보수적 XGBoost로 CV 15.90-16.00, 통계 모델과 앙상블

전략:
1. 극도로 보수적인 하이퍼파라미터
2. 핵심 피처만 사용 (10-12개)
3. GroupKFold로 안전하게 검증
4. 통계 모델(75%)과 앙상블(25%)
5. CV > 16.0이면 경고
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")

print("=" * 80)
print("K리그 패스 좌표 예측 - 극도로 정규화된 XGBoost 모델")
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

print(f"  Train episodes: {train_df['game_episode'].nunique():,}")
print(f"  Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. 피처 엔지니어링 (핵심 피처만)
# =============================================================================
print("\n[2] 피처 엔지니어링 (보수적: 핵심 피처만)...")

def prepare_features(df):
    """핵심 피처만 생성"""
    df = df.copy()

    # 기본 델타
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']

    # 이전 패스 방향
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    # Zone 분류 (5x5, 6x6)
    df['zone_5x5'] = (np.clip(df['start_x'] // 21, 0, 4).astype(int) * 5 +
                      np.clip(df['start_y'] // 13.6, 0, 4).astype(int))
    df['zone_6x6'] = (np.clip(df['start_x'] // 17.5, 0, 5).astype(int) * 6 +
                      np.clip(df['start_y'] // 11.33, 0, 5).astype(int))

    # 방향 분류 (8방향)
    def get_direction_8way(prev_dx, prev_dy):
        if abs(prev_dx) < 1 and abs(prev_dy) < 1:
            return 0  # none

        angle = np.arctan2(prev_dy, prev_dx)
        angle_deg = np.degrees(angle)

        if -22.5 <= angle_deg < 22.5:
            return 1  # forward
        elif 22.5 <= angle_deg < 67.5:
            return 2  # forward_up
        elif 67.5 <= angle_deg < 112.5:
            return 3  # up
        elif 112.5 <= angle_deg < 157.5:
            return 4  # back_up
        elif angle_deg >= 157.5 or angle_deg < -157.5:
            return 5  # backward
        elif -157.5 <= angle_deg < -112.5:
            return 6  # back_down
        elif -112.5 <= angle_deg < -67.5:
            return 7  # down
        else:
            return 8  # forward_down

    df['direction_8way'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)

    # 패스 거리 및 각도
    df['pass_distance'] = np.sqrt(df['prev_dx']**2 + df['prev_dy']**2)
    df['dist_to_goal'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['angle_to_goal'] = np.arctan2(34 - df['start_y'], 105 - df['start_x'])

    # 필드 영역 (공격/중간/수비)
    df['field_region'] = pd.cut(df['start_x'], bins=[0, 35, 70, 105], labels=[0, 1, 2])
    df['field_region'] = df['field_region'].cat.codes

    return df

train_df = prepare_features(train_df)
test_all = prepare_features(test_all)

train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

test_last = test_all.groupby('game_episode').last().reset_index()

print(f"  Train samples: {len(train_last):,}")
print(f"  Test samples: {len(test_last):,}")

# =============================================================================
# 3. 피처 선택 (핵심만)
# =============================================================================
print("\n[3] 피처 선택...")

FEATURES = [
    'start_x', 'start_y',
    'prev_dx', 'prev_dy',
    'zone_5x5', 'zone_6x6',
    'direction_8way',
    'pass_distance',
    'dist_to_goal',
    'angle_to_goal',
    'field_region',
]

print(f"  선택된 피처: {len(FEATURES)}개")
for feat in FEATURES:
    print(f"    - {feat}")

# =============================================================================
# 4. XGBoost 하이퍼파라미터 (극도 정규화)
# =============================================================================
print("\n[4] XGBoost 하이퍼파라미터 설정 (극도 정규화)...")

PARAMS = {
    # 트리 구조 제약 (매우 보수적)
    'max_depth': 3,
    'min_child_weight': 50,
    'gamma': 1.0,

    # 서브샘플링 (다양성 증가)
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'colsample_bylevel': 0.6,

    # 정규화 (강하게)
    'reg_alpha': 2.0,  # L1
    'reg_lambda': 2.0,  # L2

    # 학습률 (느리게)
    'learning_rate': 0.01,
    'n_estimators': 100,

    # 기타
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}

print("  하이퍼파라미터:")
for key, value in PARAMS.items():
    print(f"    {key}: {value}")

# =============================================================================
# 5. GroupKFold 교차 검증 (XGBoost 단독)
# =============================================================================
print("\n[5] GroupKFold 교차 검증 (XGBoost 단독)...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
game_ids = train_last['game_id'].values

X = train_last[FEATURES]
y_x = train_last['delta_x']
y_y = train_last['delta_y']

cv_scores_x = []
cv_scores_y = []
cv_scores_combined = []

xgb_models_x = []
xgb_models_y = []

print("\n5-Fold Cross Validation (XGBoost 단독):")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}:")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_x, y_val_x = y_x.iloc[train_idx], y_x.iloc[val_idx]
    y_train_y, y_val_y = y_y.iloc[train_idx], y_y.iloc[val_idx]

    val_start_x = train_last.iloc[val_idx]['start_x'].values
    val_start_y = train_last.iloc[val_idx]['start_y'].values
    val_end_x = train_last.iloc[val_idx]['end_x'].values
    val_end_y = train_last.iloc[val_idx]['end_y'].values

    # X 좌표 모델
    model_x = xgb.XGBRegressor(**PARAMS)
    model_x.fit(X_train, y_train_x,
                eval_set=[(X_val, y_val_x)],
                verbose=False)

    pred_delta_x = model_x.predict(X_val)
    pred_x = np.clip(val_start_x + pred_delta_x, 0, 105)

    # Y 좌표 모델
    model_y = xgb.XGBRegressor(**PARAMS)
    model_y.fit(X_train, y_train_y,
                eval_set=[(X_val, y_val_y)],
                verbose=False)

    pred_delta_y = model_y.predict(X_val)
    pred_y = np.clip(val_start_y + pred_delta_y, 0, 68)

    # 거리 계산
    dist = np.sqrt((pred_x - val_end_x)**2 + (pred_y - val_end_y)**2)
    score = dist.mean()

    cv_scores_x.append(np.sqrt(np.mean((pred_x - val_end_x)**2)))
    cv_scores_y.append(np.sqrt(np.mean((pred_y - val_end_y)**2)))
    cv_scores_combined.append(score)

    xgb_models_x.append(model_x)
    xgb_models_y.append(model_y)

    print(f"    Delta X RMSE: {cv_scores_x[-1]:.4f}")
    print(f"    Delta Y RMSE: {cv_scores_y[-1]:.4f}")
    print(f"    Combined Euclidean: {score:.4f}")

xgb_cv_mean = np.mean(cv_scores_combined)
xgb_cv_std = np.std(cv_scores_combined)

print(f"\n  XGBoost 평균 CV Score: {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}")

# =============================================================================
# 6. 과적합 위험 평가
# =============================================================================
print("\n[6] XGBoost 과적합 위험 평가...")

if xgb_cv_mean >= 16.0:
    risk = "SAFE"
    gap_estimate = 0.30
    print(f"  ✅ SAFE: CV {xgb_cv_mean:.4f} >= 16.0")
    print(f"  예상 Gap: +{gap_estimate:.2f}")
    print(f"  예상 Public: {xgb_cv_mean + gap_estimate:.2f}")
elif xgb_cv_mean >= 15.5:
    risk = "WARNING"
    gap_estimate = 0.50
    print(f"  ⚠️  WARNING: CV {xgb_cv_mean:.4f} in 15.5-16.0")
    print(f"  예상 Gap: +{gap_estimate:.2f}")
    print(f"  예상 Public: {xgb_cv_mean + gap_estimate:.2f}")
else:
    risk = "DANGER"
    gap_estimate = 1.00
    print(f"  ❌ DANGER: CV {xgb_cv_mean:.4f} < 15.5")
    print(f"  예상 Gap: +{gap_estimate:.2f}")
    print(f"  예상 Public: {xgb_cv_mean + gap_estimate:.2f}")

# =============================================================================
# 7. 통계 모델 (6x6 8방향) 구축
# =============================================================================
print("\n[7] 통계 모델 (6x6 8방향) 구축...")

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / 17.5))
    y_zone = min(5, int(y / 11.33))
    return x_zone * 6 + y_zone

def get_direction_8way(prev_dx, prev_dy):
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'

    angle = np.arctan2(prev_dy, prev_dx)
    angle_deg = np.degrees(angle)

    if -22.5 <= angle_deg < 22.5:
        return 'forward'
    elif 22.5 <= angle_deg < 67.5:
        return 'forward_up'
    elif 67.5 <= angle_deg < 112.5:
        return 'up'
    elif 112.5 <= angle_deg < 157.5:
        return 'back_up'
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return 'backward'
    elif -157.5 <= angle_deg < -112.5:
        return 'back_down'
    elif -112.5 <= angle_deg < -67.5:
        return 'down'
    else:
        return 'forward_down'

def build_statistical_model(df):
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)
    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']

    # Zone 기본 통계
    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # Zone + 방향 통계
    zone_dir_stats = df.groupby('zone_dir').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    return {
        'zone_stats': zone_stats,
        'zone_dir_x': zone_dir_stats['delta_x'].to_dict(),
        'zone_dir_y': zone_dir_stats['delta_y'].to_dict(),
        'zone_dir_count': zone_dir_stats['count'].to_dict(),
    }

def predict_statistical(row, model, min_samples=20):
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    direction = get_direction_8way(row['prev_dx'], row['prev_dy'])
    key = f"{zone}_{direction}"

    if key in model['zone_dir_x'] and model['zone_dir_count'].get(key, 0) >= min_samples:
        dx = model['zone_dir_x'][key]
        dy = model['zone_dir_y'][key]
    else:
        dx = model['zone_stats']['delta_x'].get(zone, 0)
        dy = model['zone_stats']['delta_y'].get(zone, 0)

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)

    return pred_x, pred_y

# 통계 모델 학습
stat_model = build_statistical_model(train_last)

# 통계 모델 CV
stat_cv_scores = []

print("\n통계 모델 CV 검증:")
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    model = build_statistical_model(train_fold)

    predictions = val_fold.apply(lambda r: predict_statistical(r, model), axis=1)
    val_fold['pred_x_stat'] = predictions.apply(lambda x: x[0])
    val_fold['pred_y_stat'] = predictions.apply(lambda x: x[1])

    dist = np.sqrt((val_fold['pred_x_stat'] - val_fold['end_x'])**2 +
                   (val_fold['pred_y_stat'] - val_fold['end_y'])**2)
    stat_cv_scores.append(dist.mean())

    print(f"  Fold {fold+1}: {dist.mean():.4f}")

stat_cv_mean = np.mean(stat_cv_scores)
stat_cv_std = np.std(stat_cv_scores)

print(f"\n  통계 모델 평균 CV Score: {stat_cv_mean:.4f} ± {stat_cv_std:.4f}")

# =============================================================================
# 8. 앙상블 (통계 75% + XGBoost 25%)
# =============================================================================
print("\n[8] 앙상블 (통계 75% + XGBoost 25%)...")

WEIGHT_STAT = 0.75
WEIGHT_XGB = 0.25

ensemble_cv_scores = []

print("\n앙상블 CV 검증:")
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    X_val = val_fold[FEATURES]
    val_start_x = val_fold['start_x'].values
    val_start_y = val_fold['start_y'].values
    val_end_x = val_fold['end_x'].values
    val_end_y = val_fold['end_y'].values

    # 통계 모델 예측
    stat_model_fold = build_statistical_model(train_fold)
    predictions = val_fold.apply(lambda r: predict_statistical(r, stat_model_fold), axis=1)
    pred_x_stat = predictions.apply(lambda x: x[0]).values
    pred_y_stat = predictions.apply(lambda x: x[1]).values

    # XGBoost 예측
    pred_delta_x = xgb_models_x[fold].predict(X_val)
    pred_delta_y = xgb_models_y[fold].predict(X_val)
    pred_x_xgb = np.clip(val_start_x + pred_delta_x, 0, 105)
    pred_y_xgb = np.clip(val_start_y + pred_delta_y, 0, 68)

    # 앙상블
    pred_x_ensemble = WEIGHT_STAT * pred_x_stat + WEIGHT_XGB * pred_x_xgb
    pred_y_ensemble = WEIGHT_STAT * pred_y_stat + WEIGHT_XGB * pred_y_xgb

    dist = np.sqrt((pred_x_ensemble - val_end_x)**2 + (pred_y_ensemble - val_end_y)**2)
    ensemble_cv_scores.append(dist.mean())

    print(f"  Fold {fold+1}: {dist.mean():.4f}")

ensemble_cv_mean = np.mean(ensemble_cv_scores)
ensemble_cv_std = np.std(ensemble_cv_scores)

print(f"\n  앙상블 평균 CV Score: {ensemble_cv_mean:.4f} ± {ensemble_cv_std:.4f}")

# =============================================================================
# 9. 결과 비교 및 권장사항
# =============================================================================
print("\n" + "=" * 80)
print("[9] 결과 비교 및 권장사항")
print("=" * 80)

print(f"\n모델 성능 비교:")
print(f"  통계 모델 (6x6 8방향):    {stat_cv_mean:.4f} ± {stat_cv_std:.4f}")
print(f"  XGBoost (극도 정규화):    {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}")
print(f"  앙상블 (75% 통계 + 25% XGBoost): {ensemble_cv_mean:.4f} ± {ensemble_cv_std:.4f}")

print(f"\n개선 분석:")
print(f"  통계 → 앙상블: {stat_cv_mean - ensemble_cv_mean:+.4f}")
print(f"  XGBoost → 앙상블: {xgb_cv_mean - ensemble_cv_mean:+.4f}")

print(f"\n과적합 위험 평가:")
if ensemble_cv_mean >= 16.0:
    print(f"  ✅ SAFE: 앙상블 CV {ensemble_cv_mean:.4f} >= 16.0")
    print(f"  예상 Gap: +0.25-0.30")
    print(f"  예상 Public: {ensemble_cv_mean + 0.25:.2f} - {ensemble_cv_mean + 0.30:.2f}")
    recommend = "제출 권장"
elif ensemble_cv_mean >= 15.5:
    print(f"  ⚠️  WARNING: 앙상블 CV {ensemble_cv_mean:.4f} in 15.5-16.0")
    print(f"  예상 Gap: +0.30-0.50")
    print(f"  예상 Public: {ensemble_cv_mean + 0.30:.2f} - {ensemble_cv_mean + 0.50:.2f}")
    recommend = "신중한 제출 권장"
else:
    print(f"  ❌ DANGER: 앙상블 CV {ensemble_cv_mean:.4f} < 15.5")
    print(f"  예상 Gap: +0.50+")
    print(f"  예상 Public: > {ensemble_cv_mean + 0.50:.2f}")
    recommend = "제출 보류 권장"

print(f"\n제출 권장사항: {recommend}")

if ensemble_cv_mean < 16.0:
    print(f"\n⚠️  경고: CV가 16.0 이하입니다!")
    print(f"  현재 Best 제출(CV 16.03 → Public 16.3574)보다")
    print(f"  Gap이 더 클 수 있습니다.")
    print(f"  제출 시 주의가 필요합니다.")

# =============================================================================
# 10. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[10] Test 예측 및 제출 파일 생성...")

# 전체 데이터로 최종 모델 학습
print("  전체 데이터로 최종 모델 학습 중...")

# XGBoost 모델
X_full = train_last[FEATURES]
y_full_x = train_last['delta_x']
y_full_y = train_last['delta_y']

final_model_x = xgb.XGBRegressor(**PARAMS)
final_model_x.fit(X_full, y_full_x, verbose=False)

final_model_y = xgb.XGBRegressor(**PARAMS)
final_model_y.fit(X_full, y_full_y, verbose=False)

# 통계 모델 (이미 학습됨)
# stat_model = build_statistical_model(train_last)

# Test 예측
X_test = test_last[FEATURES]
test_start_x = test_last['start_x'].values
test_start_y = test_last['start_y'].values

# XGBoost 예측
pred_delta_x_test = final_model_x.predict(X_test)
pred_delta_y_test = final_model_y.predict(X_test)
pred_x_xgb_test = np.clip(test_start_x + pred_delta_x_test, 0, 105)
pred_y_xgb_test = np.clip(test_start_y + pred_delta_y_test, 0, 68)

# 통계 모델 예측
predictions_stat = test_last.apply(lambda r: predict_statistical(r, stat_model), axis=1)
pred_x_stat_test = predictions_stat.apply(lambda x: x[0]).values
pred_y_stat_test = predictions_stat.apply(lambda x: x[1]).values

# 앙상블
pred_x_final = WEIGHT_STAT * pred_x_stat_test + WEIGHT_XGB * pred_x_xgb_test
pred_y_final = WEIGHT_STAT * pred_y_stat_test + WEIGHT_XGB * pred_y_xgb_test

# 제출 파일 생성
submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_x_final,
    'end_y': pred_y_final
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')

submission_path = DATA_DIR / 'submission_xgboost_safe.csv'
submission.to_csv(submission_path, index=False)

print(f"  제출 파일 저장: {submission_path}")

# XGBoost 단독 제출 파일도 생성 (비교용)
submission_xgb_only = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_x_xgb_test,
    'end_y': pred_y_xgb_test
})
submission_xgb_only = sample_sub[['game_episode']].merge(submission_xgb_only, on='game_episode', how='left')

submission_xgb_path = DATA_DIR / 'submission_xgboost_only.csv'
submission_xgb_only.to_csv(submission_xgb_path, index=False)

print(f"  XGBoost 단독 파일: {submission_xgb_path}")

# =============================================================================
# 11. Feature Importance 분석
# =============================================================================
print("\n[11] Feature Importance 분석...")

importance_x = final_model_x.feature_importances_
importance_y = final_model_y.feature_importances_

print("\nTop 5 중요 피처 (X 좌표):")
for idx in np.argsort(importance_x)[-5:][::-1]:
    print(f"  {FEATURES[idx]:20s}: {importance_x[idx]:.4f}")

print("\nTop 5 중요 피처 (Y 좌표):")
for idx in np.argsort(importance_y)[-5:][::-1]:
    print(f"  {FEATURES[idx]:20s}: {importance_y[idx]:.4f}")

# =============================================================================
# 12. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

print(f"\n[CV 성능]")
print(f"  통계 모델:     {stat_cv_mean:.4f}")
print(f"  XGBoost:       {xgb_cv_mean:.4f}")
print(f"  앙상블:        {ensemble_cv_mean:.4f}")

print(f"\n[현재 Best와 비교]")
print(f"  현재 Best:     CV 16.04 → Public 16.3502")
print(f"  신규 앙상블:   CV {ensemble_cv_mean:.4f} → Public ???")
print(f"  CV 차이:       {16.04 - ensemble_cv_mean:+.4f}")

print(f"\n[제출 파일]")
print(f"  1. {submission_path}")
print(f"     (앙상블: 75% 통계 + 25% XGBoost)")
print(f"     CV: {ensemble_cv_mean:.4f}")
print(f"  2. {submission_xgb_path}")
print(f"     (XGBoost 단독)")
print(f"     CV: {xgb_cv_mean:.4f}")

print(f"\n[제출 권장]")
if ensemble_cv_mean < 16.0 and ensemble_cv_mean < stat_cv_mean:
    print(f"  ⚠️  주의: 앙상블 CV({ensemble_cv_mean:.4f})가 통계 모델({stat_cv_mean:.4f})보다 낮습니다.")
    print(f"  XGBoost가 과적합을 유발할 수 있습니다.")
    print(f"  통계 모델 단독 제출을 고려하세요.")
elif ensemble_cv_mean >= 16.0:
    print(f"  ✅ 앙상블 제출 권장")
    print(f"  CV가 안전 구간(16.0+)에 있습니다.")
else:
    print(f"  ⚠️  신중한 제출 필요")
    print(f"  Gap이 커질 수 있습니다.")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
