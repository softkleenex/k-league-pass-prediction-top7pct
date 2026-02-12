"""
K리그 패스 좌표 예측 - Residual Correction 모델
Zone Baseline의 오차를 ML로 보정하는 안전한 접근법

핵심 아이디어:
- final_pred = baseline_pred + ml_error_pred
- ML이 실패해도 baseline으로 graceful degradation
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
print("K리그 패스 좌표 예측 - Residual Correction 모델")
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

print(f"Train episodes: {train_df['game_episode'].nunique():,}")
print(f"Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. Zone Baseline (6x6 median) - 검증된 안정적 모델
# =============================================================================
print("\n[2] Zone Baseline (6x6 median) 준비...")

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

# 마지막 액션 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])

test_last = test_all.groupby('game_episode').last().reset_index()

# Zone 계산
train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

# Zone별 통계
zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# Baseline 예측 생성 (Train)
train_last['baseline_pred_x'] = train_last.apply(
    lambda r: np.clip(r['start_x'] + zone_stats['delta_x'].get(r['zone'], 0), 0, 105), axis=1
)
train_last['baseline_pred_y'] = train_last.apply(
    lambda r: np.clip(r['start_y'] + zone_stats['delta_y'].get(r['zone'], 0), 0, 68), axis=1
)

# Baseline 오차 계산 (이것이 ML의 타겟!)
train_last['error_x'] = train_last['end_x'] - train_last['baseline_pred_x']
train_last['error_y'] = train_last['end_y'] - train_last['baseline_pred_y']

# Baseline CV Score
baseline_dist = np.sqrt(
    (train_last['baseline_pred_x'] - train_last['end_x'])**2 +
    (train_last['baseline_pred_y'] - train_last['end_y'])**2
)
baseline_cv = baseline_dist.mean()
print(f"Baseline CV Score: {baseline_cv:.4f}")

# Test Baseline 예측
test_last['zone'] = test_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
test_last['baseline_pred_x'] = test_last.apply(
    lambda r: np.clip(r['start_x'] + zone_stats['delta_x'].get(r['zone'], 0), 0, 105), axis=1
)
test_last['baseline_pred_y'] = test_last.apply(
    lambda r: np.clip(r['start_y'] + zone_stats['delta_y'].get(r['zone'], 0), 0, 68), axis=1
)

# =============================================================================
# 3. 피처 엔지니어링 (Residual 예측용)
# =============================================================================
print("\n[3] Residual 피처 엔지니어링...")

def create_residual_features(df, is_train=True):
    """Baseline 오차를 예측하기 위한 피처"""
    df = df.copy()

    # 기본 위치 피처
    df['dist_to_goal'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['angle_to_goal'] = np.arctan2(34 - df['start_y'], 105 - df['start_x'])

    # 필드 위치 (정규화)
    df['start_x_norm'] = df['start_x'] / 105
    df['start_y_norm'] = df['start_y'] / 68

    # Zone 중심으로부터의 거리 (Zone 내 위치)
    df['x_in_zone'] = (df['start_x'] % (105/6)) / (105/6)
    df['y_in_zone'] = (df['start_y'] % (68/6)) / (68/6)

    # Zone 경계 근접도
    df['near_x_boundary'] = np.minimum(df['x_in_zone'], 1 - df['x_in_zone'])
    df['near_y_boundary'] = np.minimum(df['y_in_zone'], 1 - df['y_in_zone'])

    return df

# 에피소드 레벨 피처
def create_episode_features(main_df, last_df):
    """시퀀스 정보 추출"""
    episode_features = []

    for ep_id in last_df['game_episode'].unique():
        ep_df = main_df[main_df['game_episode'] == ep_id].sort_values('action_id')

        if len(ep_df) == 0:
            continue

        features = {'game_episode': ep_id}

        # 시퀀스 길이
        features['seq_length'] = len(ep_df)

        # 이동량 계산
        ep_df = ep_df.copy()
        ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
        ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']

        # 마지막 N개 액션 통계
        last_n = ep_df.tail(5)
        features['last5_dx_mean'] = last_n['dx'].mean() if len(last_n) > 0 else 0
        features['last5_dy_mean'] = last_n['dy'].mean() if len(last_n) > 0 else 0
        features['last5_dx_std'] = last_n['dx'].std() if len(last_n) > 1 else 0
        features['last5_dy_std'] = last_n['dy'].std() if len(last_n) > 1 else 0

        # 직전 액션
        if len(ep_df) >= 2:
            prev_row = ep_df.iloc[-2]
            features['prev_dx'] = prev_row['dx'] if not np.isnan(prev_row['dx']) else 0
            features['prev_dy'] = prev_row['dy'] if not np.isnan(prev_row['dy']) else 0
        else:
            features['prev_dx'] = 0
            features['prev_dy'] = 0

        # 진행 방향
        features['x_progression'] = ep_df['start_x'].iloc[-1] - ep_df['start_x'].iloc[0] if len(ep_df) > 1 else 0
        features['y_progression'] = ep_df['start_y'].iloc[-1] - ep_df['start_y'].iloc[0] if len(ep_df) > 1 else 0

        episode_features.append(features)

    return pd.DataFrame(episode_features)

print("  Train 에피소드 피처 생성...")
train_ep_features = create_episode_features(train_df, train_last)
train_last = train_last.merge(train_ep_features, on='game_episode', how='left')
train_last = create_residual_features(train_last)

print("  Test 에피소드 피처 생성...")
test_ep_features = create_episode_features(test_all, test_last)
test_last = test_last.merge(test_ep_features, on='game_episode', how='left')
test_last = create_residual_features(test_last)

# =============================================================================
# 4. Residual Correction 모델 학습
# =============================================================================
print("\n[4] Residual Correction 모델 학습...")

# 피처 선택 (최소한으로!)
feature_cols = [
    # 위치 기반 (Zone 보정용)
    'x_in_zone', 'y_in_zone',
    'near_x_boundary', 'near_y_boundary',
    'dist_to_goal', 'angle_to_goal',
    # 시퀀스 기반 (방향 보정용)
    'prev_dx', 'prev_dy',
    'last5_dx_mean', 'last5_dy_mean',
    'x_progression', 'y_progression',
    'seq_length',
]

feature_cols = [c for c in feature_cols if c in train_last.columns]
print(f"피처 수: {len(feature_cols)}")

X = train_last[feature_cols].fillna(0).values
y_error_x = train_last['error_x'].values
y_error_y = train_last['error_y'].values
game_ids = train_last['game_id'].values

X_test = test_last[feature_cols].fillna(0).values

# 극도로 정규화된 LightGBM (오차 예측용)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 7,
    'max_depth': 3,
    'learning_rate': 0.01,  # 매우 낮게
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'reg_alpha': 2.0,  # 강한 정규화
    'reg_lambda': 2.0,
    'min_child_samples': 200,  # 매우 높게
    'min_child_weight': 20,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

oof_error_x = np.zeros(len(X))
oof_error_y = np.zeros(len(X))
test_error_x = np.zeros(len(X_test))
test_error_y = np.zeros(len(X_test))

fold_improvements = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}/{n_splits}")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr_ex, y_val_ex = y_error_x[train_idx], y_error_x[val_idx]
    y_tr_ey, y_val_ey = y_error_y[train_idx], y_error_y[val_idx]

    # Error X 모델
    train_data_x = lgb.Dataset(X_tr, label=y_tr_ex)
    val_data_x = lgb.Dataset(X_val, label=y_val_ex, reference=train_data_x)

    model_x = lgb.train(
        params, train_data_x,
        num_boost_round=300,
        valid_sets=[val_data_x],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    # Error Y 모델
    train_data_y = lgb.Dataset(X_tr, label=y_tr_ey)
    val_data_y = lgb.Dataset(X_val, label=y_val_ey, reference=train_data_y)

    model_y = lgb.train(
        params, train_data_y,
        num_boost_round=300,
        valid_sets=[val_data_y],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    # OOF 예측
    oof_error_x[val_idx] = model_x.predict(X_val)
    oof_error_y[val_idx] = model_y.predict(X_val)

    # 테스트 예측
    test_error_x += model_x.predict(X_test) / n_splits
    test_error_y += model_y.predict(X_test) / n_splits

    # Fold별 개선 확인
    baseline_x = train_last.iloc[val_idx]['baseline_pred_x'].values
    baseline_y = train_last.iloc[val_idx]['baseline_pred_y'].values
    true_x = train_last.iloc[val_idx]['end_x'].values
    true_y = train_last.iloc[val_idx]['end_y'].values

    # Baseline 점수
    baseline_dist = np.sqrt((baseline_x - true_x)**2 + (baseline_y - true_y)**2)
    baseline_score = baseline_dist.mean()

    # Corrected 점수
    corrected_x = baseline_x + oof_error_x[val_idx]
    corrected_y = baseline_y + oof_error_y[val_idx]
    corrected_dist = np.sqrt((corrected_x - true_x)**2 + (corrected_y - true_y)**2)
    corrected_score = corrected_dist.mean()

    improvement = baseline_score - corrected_score
    fold_improvements.append(improvement)

    print(f"    Baseline: {baseline_score:.4f}, Corrected: {corrected_score:.4f}, 개선: {improvement:+.4f}")

# =============================================================================
# 5. 전체 결과 분석
# =============================================================================
print("\n" + "=" * 70)
print("[5] 전체 결과 분석")
print("=" * 70)

# 전체 OOF 결과
baseline_pred_x = train_last['baseline_pred_x'].values
baseline_pred_y = train_last['baseline_pred_y'].values
true_x = train_last['end_x'].values
true_y = train_last['end_y'].values

# Baseline
baseline_dist_all = np.sqrt((baseline_pred_x - true_x)**2 + (baseline_pred_y - true_y)**2)
baseline_cv_all = baseline_dist_all.mean()

# Corrected
corrected_x_all = baseline_pred_x + oof_error_x
corrected_y_all = baseline_pred_y + oof_error_y
corrected_dist_all = np.sqrt((corrected_x_all - true_x)**2 + (corrected_y_all - true_y)**2)
corrected_cv_all = corrected_dist_all.mean()

improvement_all = baseline_cv_all - corrected_cv_all

print(f"\nBaseline CV:   {baseline_cv_all:.4f}")
print(f"Corrected CV:  {corrected_cv_all:.4f}")
print(f"개선:          {improvement_all:+.4f}")
print(f"Fold별 개선:   {[f'{x:+.4f}' for x in fold_improvements]}")
print(f"개선 Std:      {np.std(fold_improvements):.4f}")

# =============================================================================
# 6. 안전한 보정 적용 (Shrinkage)
# =============================================================================
print("\n[6] 안전한 보정 적용 (Shrinkage)...")

# 보정량을 축소하여 과적합 위험 감소
shrinkage_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
best_shrinkage = 0
best_shrink_cv = baseline_cv_all

print("\nShrinkage별 CV Score:")
for shrink in shrinkage_factors:
    shrink_x = baseline_pred_x + shrink * oof_error_x
    shrink_y = baseline_pred_y + shrink * oof_error_y
    shrink_dist = np.sqrt((shrink_x - true_x)**2 + (shrink_y - true_y)**2)
    shrink_cv = shrink_dist.mean()

    marker = ""
    if shrink_cv < best_shrink_cv:
        best_shrink_cv = shrink_cv
        best_shrinkage = shrink
        marker = " *** BEST"

    print(f"  Shrinkage {shrink:.1f}: CV = {shrink_cv:.4f}{marker}")

print(f"\n최적 Shrinkage: {best_shrinkage}")
print(f"최적 CV: {best_shrink_cv:.4f}")

# =============================================================================
# 7. 제출 파일 생성
# =============================================================================
print("\n[7] 제출 파일 생성...")

# 1. Pure Baseline (기존 Best)
sub_baseline = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['baseline_pred_x'],
    'end_y': test_last['baseline_pred_y']
})
sub_baseline = sample_sub[['game_episode']].merge(sub_baseline, on='game_episode', how='left')
sub_baseline.to_csv('submission_baseline_6x6.csv', index=False)
print(f"  1. submission_baseline_6x6.csv (CV: {baseline_cv_all:.4f})")

# 2. 최적 Shrinkage 적용
optimal_x = test_last['baseline_pred_x'].values + best_shrinkage * test_error_x
optimal_y = test_last['baseline_pred_y'].values + best_shrinkage * test_error_y
optimal_x = np.clip(optimal_x, 0, 105)
optimal_y = np.clip(optimal_y, 0, 68)

sub_optimal = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': optimal_x,
    'end_y': optimal_y
})
sub_optimal = sample_sub[['game_episode']].merge(sub_optimal, on='game_episode', how='left')
sub_optimal.to_csv('submission_residual_optimal.csv', index=False)
print(f"  2. submission_residual_optimal.csv (CV: {best_shrink_cv:.4f}, shrink={best_shrinkage})")

# 3. 안전한 버전들 (다양한 shrinkage)
for shrink in [0.2, 0.3, 0.5]:
    safe_x = test_last['baseline_pred_x'].values + shrink * test_error_x
    safe_y = test_last['baseline_pred_y'].values + shrink * test_error_y
    safe_x = np.clip(safe_x, 0, 105)
    safe_y = np.clip(safe_y, 0, 68)

    # Train CV 계산
    train_safe_x = baseline_pred_x + shrink * oof_error_x
    train_safe_y = baseline_pred_y + shrink * oof_error_y
    safe_dist = np.sqrt((train_safe_x - true_x)**2 + (train_safe_y - true_y)**2)
    safe_cv = safe_dist.mean()

    sub_safe = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': safe_x,
        'end_y': safe_y
    })
    sub_safe = sample_sub[['game_episode']].merge(sub_safe, on='game_episode', how='left')
    filename = f'submission_residual_shrink{int(shrink*10)}.csv'
    sub_safe.to_csv(filename, index=False)
    print(f"  3. {filename} (CV: {safe_cv:.4f})")

# =============================================================================
# 8. 피처 중요도 분석
# =============================================================================
print("\n[8] 피처 중요도 (마지막 Fold 기준)...")

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance_x': model_x.feature_importance(),
    'importance_y': model_y.feature_importance()
})
importance['importance_total'] = importance['importance_x'] + importance['importance_y']
importance = importance.sort_values('importance_total', ascending=False)

print("\nTop 피처:")
for _, row in importance.iterrows():
    print(f"  {row['feature']:<20} {row['importance_total']:>6}")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"\n[Residual Correction 효과]")
print(f"  Baseline CV:     {baseline_cv_all:.4f}")
print(f"  Corrected CV:    {corrected_cv_all:.4f}")
print(f"  개선:            {improvement_all:+.4f}")

if improvement_all > 0:
    print(f"\n  ✅ ML 보정이 효과 있음!")
    print(f"  → 최적 shrinkage {best_shrinkage}로 CV {best_shrink_cv:.4f} 달성")
else:
    print(f"\n  ❌ ML 보정이 효과 없음 (과적합)")
    print(f"  → Baseline(6x6 median) 유지 권장")

print(f"\n[권장 제출 순서]")
if improvement_all > 0 and best_shrinkage > 0:
    print(f"  1. submission_residual_shrink2.csv (안전한 보정)")
    print(f"  2. submission_residual_optimal.csv (최적 보정)")
    print(f"  3. submission_baseline_6x6.csv (fallback)")
else:
    print(f"  1. submission_baseline_6x6.csv (가장 안전)")
    print(f"  2. 다른 전략 시도 필요")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
