"""
Ensemble Model: Zone 6x6 + Domain v1

전략:
1. Zone 6x6 (안정적): Public 16.36, Gap 0.02 → stability = 0.98
2. Domain v1 (예측력): CV 16.12, Gap 0.60 → stability = 0.625
3. 가중 평균: w_zone = 0.6, w_domain = 0.4 (Gap 기반)

목표:
- 안정성(Zone) + 예측력(Domain) 결합
- CV < 16.3 AND Gap < 0.4
- 예상 Public: 16.2-16.3

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
print("Ensemble Model: Zone 6x6 + Domain v1")
print("=" * 80)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_meta = pd.read_csv(DATA_DIR / "test.csv")

# Test 데이터 로드
test_episodes = []
for _, row in test_meta.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_df = pd.concat(test_episodes, ignore_index=True)

print(f"Train: {len(train_df):,} passes")
print(f"Test:  {len(test_df):,} passes")

# =============================================================================
# 2. 피처 준비 (Zone + Domain 모두)
# =============================================================================
print("\n[2] 피처 준비...")

def get_zone_6x6(x, y):
    """6x6 Zone 분류"""
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

def get_direction_8way(dx, dy):
    """8방향 분류 (45도 간격)"""
    if abs(dx) < 1 and abs(dy) < 1:
        return 0  # none

    angle = np.arctan2(dy, dx)
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

def prepare_features(df):
    """Zone + Domain 피처 준비"""
    df = df.copy()

    # 기본 델타 계산
    if 'end_x' in df.columns:
        df['dx'] = df['end_x'] - df['start_x']
        df['dy'] = df['end_y'] - df['start_y']
        df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
        df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
    else:
        # Test 데이터 - prev_dx, prev_dy 계산
        df['prev_dx'] = 0.0
        df['prev_dy'] = 0.0

    # Domain 피처
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.arctan2(34 - df['start_y'], 105 - df['start_x'])
    df['zone_6x6'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction_8way'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)
    df['is_near_goal'] = (df['goal_distance'] < 20).astype(int)
    df['field_zone'] = pd.cut(df['start_x'], bins=[0, 35, 70, 106], labels=[0, 1, 2], include_lowest=True)
    df['field_zone'] = df['field_zone'].cat.codes.astype(int)

    return df

train_df = prepare_features(train_df)
test_df = prepare_features(test_df)

# 마지막 패스만 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

test_last = test_df.groupby('game_episode').last().reset_index()

print(f"Train 마지막 패스: {len(train_last):,}")
print(f"Test 마지막 패스: {len(test_last):,}")

# =============================================================================
# 3. GroupKFold 설정
# =============================================================================
print("\n[3] GroupKFold 설정...")

gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

zone_predictions_cv = np.zeros((len(train_last), 2))
domain_predictions_cv = np.zeros((len(train_last), 2))

zone_models = []  # Test용 모델 저장
domain_models_x = []
domain_models_y = []

# =============================================================================
# 4. Model 1: Zone 6x6 (안정적, Gap 0.02)
# =============================================================================
print("\n[4] Model 1: Zone 6x6 통계 모델...")

zone_fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    if fold_idx >= 3:  # Fold 1-3만
        continue

    train_fold = train_last.iloc[train_idx].copy()
    val_fold = train_last.iloc[val_idx].copy()

    # Zone + Direction 키 생성
    train_fold['key'] = train_fold['zone_6x6'].astype(str) + '_' + train_fold['direction_8way'].astype(str)
    val_fold['key'] = val_fold['zone_6x6'].astype(str) + '_' + val_fold['direction_8way'].astype(str)

    # 통계 계산
    stats = train_fold.groupby('key').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    zone_fallback = train_fold.groupby('zone_6x6').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    global_dx = train_fold['delta_x'].median()
    global_dy = train_fold['delta_y'].median()

    # 예측 함수
    def predict_zone(row):
        key = row['key']
        min_samples = 25

        if key in stats.index and stats.loc[key, 'count'] >= min_samples:
            dx = stats.loc[key, 'delta_x']
            dy = stats.loc[key, 'delta_y']
        elif row['zone_6x6'] in zone_fallback['delta_x']:
            dx = zone_fallback['delta_x'][row['zone_6x6']]
            dy = zone_fallback['delta_y'][row['zone_6x6']]
        else:
            dx = global_dx
            dy = global_dy

        pred_x = np.clip(row['start_x'] + dx, 0, 105)
        pred_y = np.clip(row['start_y'] + dy, 0, 68)
        return pd.Series({'pred_x': pred_x, 'pred_y': pred_y})

    predictions = val_fold.apply(predict_zone, axis=1)
    zone_predictions_cv[val_idx, 0] = predictions['pred_x'].values
    zone_predictions_cv[val_idx, 1] = predictions['pred_y'].values

    # CV 계산
    dist = np.sqrt((predictions['pred_x'] - val_fold['end_x'])**2 +
                   (predictions['pred_y'] - val_fold['end_y'])**2)
    fold_cv = dist.mean()
    zone_fold_scores.append(fold_cv)

    print(f"  Fold {fold_idx + 1}: {fold_cv:.4f}")

    # Test용 모델 저장
    zone_models.append({
        'stats': stats,
        'zone_fallback': zone_fallback,
        'global_dx': global_dx,
        'global_dy': global_dy
    })

zone_cv = np.mean(zone_fold_scores)
zone_std = np.std(zone_fold_scores)
print(f"  Zone 6x6 CV: {zone_cv:.4f} ± {zone_std:.4f}")

# =============================================================================
# 5. Model 2: Domain v1 (예측력, Gap 0.60)
# =============================================================================
print("\n[5] Model 2: Domain v1 LightGBM...")

feature_cols = [
    'start_x', 'start_y',
    'prev_dx', 'prev_dy',
    'goal_distance', 'goal_angle',
    'zone_6x6', 'direction_8way',
    'is_near_goal', 'field_zone'
]

X = train_last[feature_cols].fillna(0)
y_x = train_last['delta_x']
y_y = train_last['delta_y']

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 300,
    'max_depth': 5,
    'num_leaves': 31,
    'min_child_samples': 50,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1
}

domain_fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    if fold_idx >= 3:  # Fold 1-3만
        continue

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_x_train, y_x_val = y_x.iloc[train_idx], y_x.iloc[val_idx]
    y_y_train, y_y_val = y_y.iloc[train_idx], y_y.iloc[val_idx]

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

    # 예측
    pred_dx = model_x.predict(X_val)
    pred_dy = model_y.predict(X_val)

    val_df = train_last.iloc[val_idx].copy()
    pred_x = np.clip(val_df['start_x'] + pred_dx, 0, 105)
    pred_y = np.clip(val_df['start_y'] + pred_dy, 0, 68)

    domain_predictions_cv[val_idx, 0] = pred_x
    domain_predictions_cv[val_idx, 1] = pred_y

    # CV 계산
    dist = np.sqrt((pred_x - val_df['end_x'])**2 +
                   (pred_y - val_df['end_y'])**2)
    fold_cv = dist.mean()
    domain_fold_scores.append(fold_cv)

    print(f"  Fold {fold_idx + 1}: {fold_cv:.4f}")

    domain_models_x.append(model_x)
    domain_models_y.append(model_y)

domain_cv = np.mean(domain_fold_scores)
domain_std = np.std(domain_fold_scores)
print(f"  Domain v1 CV: {domain_cv:.4f} ± {domain_std:.4f}")

# =============================================================================
# 6. Ensemble: Weighted Average (Gap 기반)
# =============================================================================
print("\n[6] Ensemble: Weighted Average...")

# Gap 기반 가중치 계산
zone_gap = 0.02
domain_gap = 0.60

zone_stability = 1 / (1 + zone_gap)  # 0.98
domain_stability = 1 / (1 + domain_gap)  # 0.625

total_stability = zone_stability + domain_stability
w_zone = zone_stability / total_stability  # 0.61 → 0.6
w_domain = domain_stability / total_stability  # 0.39 → 0.4

# 반올림
w_zone = 0.6
w_domain = 0.4

print(f"  Zone 6x6 가중치: {w_zone:.1f} (Gap: {zone_gap:.2f})")
print(f"  Domain v1 가중치: {w_domain:.1f} (Gap: {domain_gap:.2f})")

# 앙상블 예측
ensemble_predictions_cv = w_zone * zone_predictions_cv + w_domain * domain_predictions_cv

# Fold별 CV 계산
ensemble_fold_scores = []
for fold_idx, (_, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    if fold_idx >= 3:
        continue

    val_fold = train_last.iloc[val_idx]
    ensemble_pred = ensemble_predictions_cv[val_idx]

    dist = np.sqrt((ensemble_pred[:, 0] - val_fold['end_x'].values)**2 +
                   (ensemble_pred[:, 1] - val_fold['end_y'].values)**2)
    fold_cv = dist.mean()
    ensemble_fold_scores.append(fold_cv)

    print(f"  Fold {fold_idx + 1}: {fold_cv:.4f}")

ensemble_cv = np.mean(ensemble_fold_scores)
ensemble_std = np.std(ensemble_fold_scores)

print(f"\n  Ensemble CV: {ensemble_cv:.4f} ± {ensemble_std:.4f}")

# =============================================================================
# 7. Pipeline v2 Gap 예측
# =============================================================================
print("\n[7] Pipeline v2 Gap 예측...")

# Ensemble은 Zone 6x6 기반이므로 Gap이 작을 것으로 예상
# 단, Domain의 불안정성이 섞여서 약간 증가
expected_gap = w_zone * zone_gap + w_domain * domain_gap  # 0.6*0.02 + 0.4*0.6 = 0.252
expected_public = ensemble_cv + expected_gap

print(f"  Zone 6x6 기여: {w_zone} * {zone_gap:.2f} = {w_zone * zone_gap:.3f}")
print(f"  Domain v1 기여: {w_domain} * {domain_gap:.2f} = {w_domain * domain_gap:.3f}")
print(f"  예상 Gap: {expected_gap:.4f}")
print(f"  예상 Public: {expected_public:.4f}")

# =============================================================================
# 8. 제출 조건 확인
# =============================================================================
print("\n[8] 제출 조건 확인...")

SUBMIT_THRESHOLD_CV = 16.3
SUBMIT_THRESHOLD_GAP = 0.4

print(f"  CV < {SUBMIT_THRESHOLD_CV}: {ensemble_cv:.4f} < {SUBMIT_THRESHOLD_CV} → {'✅' if ensemble_cv < SUBMIT_THRESHOLD_CV else '❌'}")
print(f"  Gap < {SUBMIT_THRESHOLD_GAP}: {expected_gap:.4f} < {SUBMIT_THRESHOLD_GAP} → {'✅' if expected_gap < SUBMIT_THRESHOLD_GAP else '❌'}")

should_submit = (ensemble_cv < SUBMIT_THRESHOLD_CV) and (expected_gap < SUBMIT_THRESHOLD_GAP)

if not should_submit:
    print("\n  ❌ 제출 조건 미충족. 제출하지 않음.")
    print(f"\n[종료]")
    import sys
    sys.exit(0)

print("\n  ✅ 제출 조건 충족! Test 예측 진행...")

# =============================================================================
# 9. Test 예측
# =============================================================================
print("\n[9] Test 예측...")

# Zone 6x6 예측
zone_test_predictions = np.zeros((len(test_last), 2))

for model_dict in zone_models:
    stats = model_dict['stats']
    zone_fallback = model_dict['zone_fallback']
    global_dx = model_dict['global_dx']
    global_dy = model_dict['global_dy']

    test_temp = test_last.copy()
    test_temp['key'] = test_temp['zone_6x6'].astype(str) + '_' + test_temp['direction_8way'].astype(str)

    def predict_zone_test(row):
        key = row['key']
        min_samples = 25

        if key in stats.index and stats.loc[key, 'count'] >= min_samples:
            dx = stats.loc[key, 'delta_x']
            dy = stats.loc[key, 'delta_y']
        elif row['zone_6x6'] in zone_fallback['delta_x']:
            dx = zone_fallback['delta_x'][row['zone_6x6']]
            dy = zone_fallback['delta_y'][row['zone_6x6']]
        else:
            dx = global_dx
            dy = global_dy

        pred_x = np.clip(row['start_x'] + dx, 0, 105)
        pred_y = np.clip(row['start_y'] + dy, 0, 68)
        return pd.Series({'pred_x': pred_x, 'pred_y': pred_y})

    preds = test_temp.apply(predict_zone_test, axis=1)
    zone_test_predictions[:, 0] += preds['pred_x'].values / len(zone_models)
    zone_test_predictions[:, 1] += preds['pred_y'].values / len(zone_models)

print(f"  Zone 6x6 예측 완료")

# Domain v1 예측
X_test = test_last[feature_cols].fillna(0)

pred_dx_list = []
pred_dy_list = []

for model_x, model_y in zip(domain_models_x, domain_models_y):
    pred_dx = model_x.predict(X_test)
    pred_dy = model_y.predict(X_test)
    pred_dx_list.append(pred_dx)
    pred_dy_list.append(pred_dy)

pred_dx_avg = np.mean(pred_dx_list, axis=0)
pred_dy_avg = np.mean(pred_dy_list, axis=0)

domain_test_predictions = np.zeros((len(test_last), 2))
domain_test_predictions[:, 0] = np.clip(test_last['start_x'] + pred_dx_avg, 0, 105)
domain_test_predictions[:, 1] = np.clip(test_last['start_y'] + pred_dy_avg, 0, 68)

print(f"  Domain v1 예측 완료")

# Ensemble
ensemble_test_predictions = w_zone * zone_test_predictions + w_domain * domain_test_predictions

print(f"  Ensemble 완료 (0.6*Zone + 0.4*Domain)")

# =============================================================================
# 10. 제출 파일 생성
# =============================================================================
print("\n[10] 제출 파일 생성...")

submission = pd.DataFrame({
    'index': test_last['game_episode'].values,
    'x': ensemble_test_predictions[:, 0],
    'y': ensemble_test_predictions[:, 1]
})

submission_filename = f"submission_ensemble_zone_domain_v1_cv{ensemble_cv:.4f}.csv"
submission_path = SUBMISSION_DIR / submission_filename
submission.to_csv(submission_path, index=False)

print(f"  파일 저장: {submission_path}")

# =============================================================================
# 11. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("Ensemble Model 완료!")
print("=" * 80)

print(f"\n[개별 모델 성능]")
print(f"  Zone 6x6 CV: {zone_cv:.4f} ± {zone_std:.4f} (Gap: {zone_gap:.2f})")
print(f"  Domain v1 CV: {domain_cv:.4f} ± {domain_std:.4f} (Gap: {domain_gap:.2f})")

print(f"\n[Ensemble 성능]")
print(f"  Fold 1-3 CV: {ensemble_cv:.4f} ± {ensemble_std:.4f}")
print(f"  예상 Gap: {expected_gap:.4f}")
print(f"  예상 Public: {expected_public:.4f}")

print(f"\n[가중치]")
print(f"  Zone 6x6: {w_zone:.1f}")
print(f"  Domain v1: {w_domain:.1f}")

print(f"\n[제출]")
print(f"  파일: {submission_filename}")
print(f"  제출 권장: {'✅ Yes' if should_submit else '❌ No'}")

if ensemble_cv < 16.2:
    print(f"\n  🎉 목표 달성! (CV < 16.2)")
elif ensemble_cv < 16.3:
    print(f"\n  ✅ 양호 (CV < 16.3)")
else:
    print(f"\n  ⚠️ 목표 미달 (CV >= 16.3)")

print("\n" + "=" * 80)
