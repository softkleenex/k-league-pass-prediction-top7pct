"""
K리그 패스 좌표 예측 - Optimized Ensemble Fold 1-3 재가중치

전략:
1. optimized_ensemble의 모델 구성 유지
2. Fold 4-5 제외 (쉬운 데이터, 과적합 위험)
3. Fold 1-3 기준 Inverse Variance Weighting
4. CV 16.2+ 안전 구간 유지

발견:
- Fold 4-5는 실제로 예측하기 쉬운 데이터 (거리 -0.63m, 분산 작음)
- 모든 모델이 Fold 4-5에 과적합 경향
- Public은 Fold 1-3과 유사한 분포일 가능성 높음

예상 결과:
- Fold 1-3 CV: 16.20-16.27
- Gap: +0.10-0.15
- Public: 16.30-16.42 (Best 유지 또는 개선!)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 최적화 앙상블 모델")
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
# 2. 피처 준비
# =============================================================================
print("\n[2] 피처 준비...")

def prepare_features(df):
    df = df.copy()
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
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
# 3. Zone 및 방향 분류 함수
# =============================================================================
print("\n[3] Zone 및 방향 분류 함수 정의...")

def get_zone(x, y, n_x, n_y):
    """NxN Zone 분류"""
    x_zone = min(n_x - 1, int(x / (105 / n_x)))
    y_zone = min(n_y - 1, int(y / (68 / n_y)))
    return x_zone * n_y + y_zone

def get_direction_5way(prev_dx, prev_dy):
    """5방향 분류 (기존 안정적 방식)"""
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'

    angle = np.arctan2(prev_dy, prev_dx)

    if angle > np.pi * 2/3:
        return 'back_up'
    elif angle > np.pi / 3:
        return 'up'
    elif angle > -np.pi / 3:
        if prev_dx > 0:
            return 'forward'
        else:
            return 'backward'
    elif angle > -np.pi * 2/3:
        return 'down'
    else:
        return 'back_down'

def get_direction_8way(prev_dx, prev_dy):
    """8방향 분류 (45도 간격)"""
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
    else:  # -67.5 <= angle_deg < -22.5
        return 'forward_down'

# =============================================================================
# 4. 다양한 모델 구축 함수
# =============================================================================
print("\n[4] 다양한 모델 구축 함수...")

def build_directional_model(df, n_x, n_y, direction_type='5way', min_samples=20):
    """방향 조건부 Zone 통계 구축"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1)

    if direction_type == '5way':
        df['direction'] = df.apply(lambda r: get_direction_5way(r['prev_dx'], r['prev_dy']), axis=1)
    else:
        df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)

    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']

    # 기본 Zone 통계
    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # 방향 조건부 통계
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
        'n_x': n_x,
        'n_y': n_y,
        'direction_type': direction_type,
        'min_samples': min_samples
    }

def predict_directional_zone(row, model):
    """방향 조건부 예측"""
    zone = get_zone(row['start_x'], row['start_y'], model['n_x'], model['n_y'])

    if model['direction_type'] == '5way':
        direction = get_direction_5way(row['prev_dx'], row['prev_dy'])
    else:
        direction = get_direction_8way(row['prev_dx'], row['prev_dy'])

    key = f"{zone}_{direction}"

    # 조건부 통계 사용 가능한지 확인
    if key in model['zone_dir_x'] and model['zone_dir_count'].get(key, 0) >= model['min_samples']:
        dx = model['zone_dir_x'][key]
        dy = model['zone_dir_y'][key]
    else:
        # Fallback to zone-only statistics
        dx = model['zone_stats']['delta_x'].get(zone, 0)
        dy = model['zone_stats']['delta_y'].get(zone, 0)

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)

    return pred_x, pred_y

# =============================================================================
# 5. 다양한 모델 앙상블 구성
# =============================================================================
print("\n[5] 다양한 모델 앙상블 구성...")

# 모델 구성: (zone_size, direction_type, min_samples)
MODEL_CONFIGS = [
    # 5방향 모델들 (안정적)
    ((5, 5), '5way', 20),
    ((6, 6), '5way', 20),
    ((7, 7), '5way', 20),
    ((8, 8), '5way', 25),

    # 8방향 모델들 (세밀함)
    ((5, 5), '8way', 20),
    ((6, 6), '8way', 20),
    ((7, 7), '8way', 20),

    # 보수적 모델들 (높은 min_samples)
    ((6, 6), '5way', 30),
    ((6, 6), '8way', 25),
]

# =============================================================================
# 6. GroupKFold 교차 검증
# =============================================================================
print("\n[6] GroupKFold 교차 검증...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
game_ids = train_last['game_id'].values

cv_scores = {i: [] for i in range(len(MODEL_CONFIGS))}

print("\n개별 모델 CV 점수:")
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    if fold == 0:
        print(f"\n  Fold {fold+1}:")

    # 각 모델 구성별 예측
    for idx, ((n_x, n_y), dir_type, min_samp) in enumerate(MODEL_CONFIGS):
        model = build_directional_model(train_fold, n_x, n_y, dir_type, min_samp)

        predictions = val_fold.apply(
            lambda r: predict_directional_zone(r, model),
            axis=1
        )
        col_name = f'pred_{idx}'
        val_fold[f'{col_name}_x'] = predictions.apply(lambda x: x[0])
        val_fold[f'{col_name}_y'] = predictions.apply(lambda x: x[1])

        dist = np.sqrt(
            (val_fold[f'{col_name}_x'] - val_fold['end_x'])**2 +
            (val_fold[f'{col_name}_y'] - val_fold['end_y'])**2
        )
        score = dist.mean()
        cv_scores[idx].append(score)

        if fold == 0:
            config_str = f"{n_x}x{n_y}_{dir_type}_m{min_samp}"
            print(f"    Model {idx} ({config_str}): {score:.4f}")

print("\n평균 CV 점수:")
for idx, ((n_x, n_y), dir_type, min_samp) in enumerate(MODEL_CONFIGS):
    mean_cv = np.mean(cv_scores[idx])
    std_cv = np.std(cv_scores[idx])
    config_str = f"{n_x}x{n_y}_{dir_type}_m{min_samp}"
    print(f"  Model {idx} ({config_str:20s}): {mean_cv:.4f} ± {std_cv:.4f}")

# =============================================================================
# 7. 앙상블 전략 최적화
# =============================================================================
print("\n[7] 앙상블 전략 최적화...")

# 전체 데이터로 모델 학습
all_models = []
for (n_x, n_y), dir_type, min_samp in MODEL_CONFIGS:
    model = build_directional_model(train_last, n_x, n_y, dir_type, min_samp)
    all_models.append(model)

    predictions = train_last.apply(
        lambda r: predict_directional_zone(r, model),
        axis=1
    )
    idx = len(all_models) - 1
    train_last[f'pred_{idx}_x'] = predictions.apply(lambda x: x[0])
    train_last[f'pred_{idx}_y'] = predictions.apply(lambda x: x[1])

# 앙상블 전략 1: 상위 N개 모델 평균
print("\n전략 1: 상위 성능 모델 선택 및 가중 평균")
model_performances = [(idx, np.mean(cv_scores[idx])) for idx in range(len(MODEL_CONFIGS))]
model_performances.sort(key=lambda x: x[1])

best_ensemble_cv = float('inf')
best_ensemble_config = None

for top_n in [3, 4, 5]:
    top_models = [idx for idx, _ in model_performances[:top_n]]

    # Equal weight
    weights_equal = [1/top_n] * top_n
    pred_x = sum(weights_equal[i] * train_last[f'pred_{top_models[i]}_x'] for i in range(top_n))
    pred_y = sum(weights_equal[i] * train_last[f'pred_{top_models[i]}_y'] for i in range(top_n))
    dist = np.sqrt((pred_x - train_last['end_x'])**2 + (pred_y - train_last['end_y'])**2)
    score_equal = dist.mean()

    # Inverse variance weight (Fold 1-3만 사용!)
    variances = [np.var(cv_scores[idx][:3]) + 1e-8 for idx in top_models]  # Fold 1-3만!
    weights_inv = [1/v for v in variances]
    weights_inv = [w/sum(weights_inv) for w in weights_inv]

    # Fold 1-3 성능 출력
    fold13_scores = [np.mean(cv_scores[idx][:3]) for idx in top_models]
    print(f"    Fold 1-3 CV: {fold13_scores}")
    pred_x = sum(weights_inv[i] * train_last[f'pred_{top_models[i]}_x'] for i in range(top_n))
    pred_y = sum(weights_inv[i] * train_last[f'pred_{top_models[i]}_y'] for i in range(top_n))
    dist = np.sqrt((pred_x - train_last['end_x'])**2 + (pred_y - train_last['end_y'])**2)
    score_inv = dist.mean()

    print(f"  Top {top_n} 모델 평균:")
    print(f"    Equal weights:        {score_equal:.4f}")
    print(f"    Inverse var weights:  {score_inv:.4f}")

    if score_inv < best_ensemble_cv:
        best_ensemble_cv = score_inv
        best_ensemble_config = ('inverse_var', top_models, weights_inv)

# 전략 2: Zone별 최적 모델 선택
print("\n전략 2: 공격/수비 지역별 모델 믹싱")
# 공격 지역 (x > 70): 세밀한 모델 선호
# 수비 지역 (x < 35): 안정적 모델 선호
# 중간 지역: 앙상블

def get_adaptive_ensemble_weights(start_x):
    """위치에 따라 동적으로 모델 가중치 조정"""
    if start_x > 70:  # 공격 지역
        # 8방향 세밀한 모델 선호
        return [0.05, 0.10, 0.10, 0.05, 0.15, 0.25, 0.15, 0.10, 0.05]
    elif start_x < 35:  # 수비 지역
        # 5방향 안정적 모델 선호
        return [0.20, 0.30, 0.20, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02]
    else:  # 중간 지역
        # 균형있게 앙상블
        return [0.10, 0.15, 0.15, 0.10, 0.12, 0.15, 0.12, 0.08, 0.03]

# 전략 3: 메디안 앙상블
print("\n전략 3: 메디안 앙상블 (이상치 제거)")
top_5_models = [idx for idx, _ in model_performances[:5]]
pred_x_median = train_last[[f'pred_{idx}_x' for idx in top_5_models]].median(axis=1)
pred_y_median = train_last[[f'pred_{idx}_y' for idx in top_5_models]].median(axis=1)
dist_median = np.sqrt((pred_x_median - train_last['end_x'])**2 + (pred_y_median - train_last['end_y'])**2)
score_median = dist_median.mean()
print(f"  Top 5 메디안: {score_median:.4f}")

if score_median < best_ensemble_cv:
    best_ensemble_cv = score_median
    best_ensemble_config = ('median', top_5_models, None)

# =============================================================================
# 8. 최종 앙상블 CV 검증
# =============================================================================
print("\n[8] 최종 앙상블 CV 검증...")

ensemble_type, selected_models, weights = best_ensemble_config
print(f"\n최적 앙상블: {ensemble_type}")
print(f"선택된 모델: {selected_models}")
if weights:
    print(f"가중치: {[f'{w:.3f}' for w in weights]}")

final_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    # 선택된 모델들 학습 및 예측
    for model_idx in selected_models:
        (n_x, n_y), dir_type, min_samp = MODEL_CONFIGS[model_idx]
        model = build_directional_model(train_fold, n_x, n_y, dir_type, min_samp)

        predictions = val_fold.apply(
            lambda r: predict_directional_zone(r, model),
            axis=1
        )
        val_fold[f'pred_{model_idx}_x'] = predictions.apply(lambda x: x[0])
        val_fold[f'pred_{model_idx}_y'] = predictions.apply(lambda x: x[1])

    # 앙상블 예측
    if ensemble_type == 'median':
        pred_x = val_fold[[f'pred_{idx}_x' for idx in selected_models]].median(axis=1)
        pred_y = val_fold[[f'pred_{idx}_y' for idx in selected_models]].median(axis=1)
    else:
        pred_x = sum(weights[i] * val_fold[f'pred_{selected_models[i]}_x'] for i in range(len(selected_models)))
        pred_y = sum(weights[i] * val_fold[f'pred_{selected_models[i]}_y'] for i in range(len(selected_models)))

    dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
    final_cv_scores.append(dist.mean())
    print(f"  Fold {fold+1}: {dist.mean():.4f}")

final_cv = np.mean(final_cv_scores)
final_std = np.std(final_cv_scores)
fold13_cv = np.mean(final_cv_scores[:3])
fold13_std = np.std(final_cv_scores[:3])
fold45_cv = np.mean(final_cv_scores[3:])

print(f"\n최종 앙상블 CV Score: {final_cv:.4f} ± {final_std:.4f}")

# =============================================================================
# 9. 과적합 위험 평가
# =============================================================================
print("\n[9] 과적합 위험 평가...")

print(f"\nCV Score (전체): {final_cv:.4f}")
print(f"CV Score (Fold 1-3): {fold13_cv:.4f} ± {fold13_std:.4f}")
print(f"CV Score (Fold 4-5): {fold45_cv:.4f}")
print(f"차이 (Fold 4-5 - Fold 1-3): {fold45_cv - fold13_cv:+.4f}")

# Fold 1-3 기준으로 위험 평가!
if fold13_cv >= 16.2:
    gap_estimate = 0.17
    risk_level = "낮음 (안전)"
    public_estimate = fold13_cv + gap_estimate
    recommend = "즉시 제출 권장"
    color = "GREEN"
elif fold13_cv >= 16.0:
    gap_estimate = 0.25
    risk_level = "중간 (경계)"
    public_estimate = fold13_cv + gap_estimate
    recommend = "신중한 제출 권장"
    color = "YELLOW"
else:
    gap_estimate = 0.35
    risk_level = "높음 (위험)"
    public_estimate = fold13_cv + gap_estimate
    recommend = "제출 보류 권장"
    color = "RED"

print(f"\n[Fold 1-3 기준 평가]")
print(f"과적합 위험: {risk_level}")
print(f"예상 Gap: +{gap_estimate:.2f}")
print(f"예상 Public Score: {public_estimate:.2f}")
print(f"제출 권장: {recommend}")

# =============================================================================
# 10. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[10] Test 예측 및 제출 파일 생성...")

# 선택된 모델들로 Test 예측
for model_idx in selected_models:
    predictions = test_last.apply(
        lambda r: predict_directional_zone(r, all_models[model_idx]),
        axis=1
    )
    test_last[f'pred_{model_idx}_x'] = predictions.apply(lambda x: x[0])
    test_last[f'pred_{model_idx}_y'] = predictions.apply(lambda x: x[1])

# 앙상블 예측
if ensemble_type == 'median':
    test_last['pred_x'] = test_last[[f'pred_{idx}_x' for idx in selected_models]].median(axis=1)
    test_last['pred_y'] = test_last[[f'pred_{idx}_y' for idx in selected_models]].median(axis=1)
else:
    test_last['pred_x'] = sum(weights[i] * test_last[f'pred_{selected_models[i]}_x'] for i in range(len(selected_models)))
    test_last['pred_y'] = sum(weights[i] * test_last[f'pred_{selected_models[i]}_y'] for i in range(len(selected_models)))

# 제출 파일 생성
submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['pred_x'],
    'end_y': test_last['pred_y']
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')

submission_path = '/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_optimized_ensemble_fold13.csv'
submission.to_csv(submission_path, index=False)

print(f"\n제출 파일 저장 완료: {submission_path}")

# =============================================================================
# 11. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 권장사항")
print("=" * 80)

print(f"\n[모델 성능]")
print(f"  전체 CV:       {final_cv:.4f} ± {final_std:.4f}")
print(f"  Fold 1-3 CV:   {fold13_cv:.4f} ± {fold13_std:.4f}")
print(f"  Fold 4-5 CV:   {fold45_cv:.4f}")
print(f"  앙상블 타입:   {ensemble_type}")
print(f"  모델 수:       {len(selected_models)}개")

print(f"\n[기준 모델과 비교 - Fold 1-3 기준]")
print(f"  현재 Best:     Fold 1-3 CV 16.27 → Public 16.3502 (Gap +0.08)")
print(f"  신규 모델:     Fold 1-3 CV {fold13_cv:.4f} → Public {public_estimate:.2f} (예상 Gap +{gap_estimate:.2f})")
print(f"  CV 차이:       {16.27 - fold13_cv:+.4f}")
print(f"  Public 개선:   {16.3502 - public_estimate:+.4f} (예상)")

print(f"\n[과적합 위험 평가]")
print(f"  위험 등급:     {risk_level}")
print(f"  위험 신호:     {color}")
print(f"  제출 권장:     {recommend}")

print(f"\n[제출 파일]")
print(f"  {submission_path}")

if final_cv <= 16.20 and final_cv >= 16.00:
    print(f"\n[전략적 제안]")
    print(f"  이 모델은 CV 안전 구간(16.0-16.2)에 있습니다.")
    print(f"  현재 Best(16.3574)를 개선할 가능성이 있습니다.")
    print(f"  제출을 권장하지만, 오늘 남은 제출 횟수를 고려하세요.")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
