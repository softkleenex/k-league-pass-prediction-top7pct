"""
K리그 패스 좌표 예측 - 8방향 조건부 Zone 모델 (안전한 개선)

과적합 위험 분석을 기반으로 한 신중한 개선:
1. 방향 분류: 5 → 8 방향 (45도 간격)
2. 동적 min_samples: 조건에 따라 15-30 조정
3. 최적 앙상블: 5x5, 6x6, 7x7 가중치 grid search

예상 결과:
- CV: 16.00-16.20
- Gap: +0.20-0.30
- Public: 16.20-16.40
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 8방향 조건부 Zone 모델 (안전한 개선)")
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

def get_direction_8way(prev_dx, prev_dy):
    """
    8방향 분류 (45도 간격)

    방향:
    - none: 움직임 없음
    - forward: 앞으로 (±22.5도)
    - forward_up: 앞쪽 위 (22.5~67.5도)
    - up: 위 (67.5~112.5도)
    - back_up: 뒤쪽 위 (112.5~157.5도)
    - backward: 뒤로 (±157.5도)
    - back_down: 뒤쪽 아래 (-157.5~-112.5도)
    - down: 아래 (-112.5~-67.5도)
    - forward_down: 앞쪽 아래 (-67.5~-22.5도)
    """
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

def get_adaptive_min_samples(zone, n_zones, base=20):
    """
    동적 min_samples 조정

    공격 지역 (중요도 높음): threshold 낮춤
    일반 지역: 기본값 유지
    """
    # 공격 3분의1 영역 식별
    x_zone = zone // int(np.sqrt(n_zones))
    threshold = base

    # 6x6 기준으로 공격 지역 (x >= 4)
    if n_zones == 36 and x_zone >= 4:
        threshold = int(base * 0.8)  # 16
    # 5x5 기준으로 공격 지역 (x >= 3)
    elif n_zones == 25 and x_zone >= 3:
        threshold = int(base * 0.8)  # 16
    # 7x7 기준으로 공격 지역 (x >= 5)
    elif n_zones == 49 and x_zone >= 5:
        threshold = int(base * 0.8)  # 16

    return threshold

# =============================================================================
# 4. 모델 구축 및 예측 함수
# =============================================================================
print("\n[4] 모델 구축 및 예측 함수...")

def build_8direction_model(df, n_x, n_y):
    """8방향 조건부 Zone 통계 구축"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)
    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']

    # 기본 Zone 통계
    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # 8방향 조건부 통계
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
        'n_y': n_y
    }

def predict_8direction_zone(row, model, use_adaptive_threshold=True):
    """8방향 조건부 예측"""
    zone = get_zone(row['start_x'], row['start_y'], model['n_x'], model['n_y'])
    direction = get_direction_8way(row['prev_dx'], row['prev_dy'])
    key = f"{zone}_{direction}"

    # 동적 threshold 사용
    if use_adaptive_threshold:
        min_samples = get_adaptive_min_samples(zone, model['n_x'] * model['n_y'])
    else:
        min_samples = 20

    # 조건부 통계 사용 가능한지 확인
    if key in model['zone_dir_x'] and model['zone_dir_count'].get(key, 0) >= min_samples:
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
# 5. GroupKFold 교차 검증
# =============================================================================
print("\n[5] GroupKFold 교차 검증...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
game_ids = train_last['game_id'].values

ZONE_CONFIGS = [(5, 5), (6, 6), (7, 7)]

# 각 Zone 크기별 CV 점수 저장
cv_scores = {config: [] for config in ZONE_CONFIGS}
cv_scores_adaptive = {config: [] for config in ZONE_CONFIGS}

print("\n  5-Fold Cross Validation:")
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    print(f"\n  Fold {fold+1}:")

    # 각 Zone 크기별 모델 구축 및 예측
    for n_x, n_y in ZONE_CONFIGS:
        model = build_8direction_model(train_fold, n_x, n_y)

        # 기본 threshold
        predictions = val_fold.apply(
            lambda r: predict_8direction_zone(r, model, use_adaptive_threshold=False),
            axis=1
        )
        val_fold[f'pred_x_{n_x}x{n_y}'] = predictions.apply(lambda x: x[0])
        val_fold[f'pred_y_{n_x}x{n_y}'] = predictions.apply(lambda x: x[1])

        dist = np.sqrt(
            (val_fold[f'pred_x_{n_x}x{n_y}'] - val_fold['end_x'])**2 +
            (val_fold[f'pred_y_{n_x}x{n_y}'] - val_fold['end_y'])**2
        )
        score = dist.mean()
        cv_scores[(n_x, n_y)].append(score)
        print(f"    {n_x}x{n_y} (fixed threshold): {score:.4f}")

        # 동적 threshold
        predictions_adaptive = val_fold.apply(
            lambda r: predict_8direction_zone(r, model, use_adaptive_threshold=True),
            axis=1
        )
        val_fold[f'pred_x_{n_x}x{n_y}_adaptive'] = predictions_adaptive.apply(lambda x: x[0])
        val_fold[f'pred_y_{n_x}x{n_y}_adaptive'] = predictions_adaptive.apply(lambda x: x[1])

        dist_adaptive = np.sqrt(
            (val_fold[f'pred_x_{n_x}x{n_y}_adaptive'] - val_fold['end_x'])**2 +
            (val_fold[f'pred_y_{n_x}x{n_y}_adaptive'] - val_fold['end_y'])**2
        )
        score_adaptive = dist_adaptive.mean()
        cv_scores_adaptive[(n_x, n_y)].append(score_adaptive)
        print(f"    {n_x}x{n_y} (adaptive threshold): {score_adaptive:.4f} (diff: {score_adaptive - score:+.4f})")

# =============================================================================
# 6. CV 결과 요약
# =============================================================================
print("\n" + "=" * 80)
print("[6] CV 결과 요약")
print("=" * 80)

print("\n개별 모델 CV 점수:")
for config in ZONE_CONFIGS:
    mean_fixed = np.mean(cv_scores[config])
    std_fixed = np.std(cv_scores[config])
    mean_adaptive = np.mean(cv_scores_adaptive[config])
    std_adaptive = np.std(cv_scores_adaptive[config])

    print(f"  {config[0]}x{config[1]} 8방향:")
    print(f"    Fixed threshold (20):    {mean_fixed:.4f} ± {std_fixed:.4f}")
    print(f"    Adaptive threshold:      {mean_adaptive:.4f} ± {std_adaptive:.4f}")
    print(f"    Difference:              {mean_adaptive - mean_fixed:+.4f}")

# 기준 모델과 비교
print("\n기준 모델과 비교:")
print(f"  6x6 5방향 (기존):    CV 16.35 → Public 16.53 (Gap +0.18)")
print(f"  6x6 8방향 (신규):    CV {np.mean(cv_scores_adaptive[(6,6)]):.4f} → Public ??? (예상 Gap +0.20-0.30)")
print(f"  예상 개선:           {16.35 - np.mean(cv_scores_adaptive[(6,6)]):+.4f} (CV 기준)")

# 과적합 위험 평가
best_cv = np.mean(cv_scores_adaptive[(6, 6)])
print(f"\n과적합 위험 평가:")
if best_cv >= 16.0:
    print(f"  ✅ SAFE: CV {best_cv:.4f} >= 16.0 (안전 구간)")
    print(f"  예상 Gap: +0.15-0.25")
    risk = "낮음"
elif best_cv >= 15.5:
    print(f"  ⚠️  WARNING: CV {best_cv:.4f} in 15.5-16.0 (경계 구간)")
    print(f"  예상 Gap: +0.25-0.40")
    risk = "중간"
else:
    print(f"  ❌ DANGER: CV {best_cv:.4f} < 15.5 (위험 구간)")
    print(f"  예상 Gap: +0.40+")
    risk = "높음"

# =============================================================================
# 7. 앙상블 최적화
# =============================================================================
print("\n[7] 앙상블 가중치 최적화...")

# 전체 데이터로 각 모델 학습
models = {}
for n_x, n_y in ZONE_CONFIGS:
    models[(n_x, n_y)] = build_8direction_model(train_last, n_x, n_y)

# 전체 데이터에 대한 예측 (adaptive threshold 사용)
for n_x, n_y in ZONE_CONFIGS:
    predictions = train_last.apply(
        lambda r: predict_8direction_zone(r, models[(n_x, n_y)], use_adaptive_threshold=True),
        axis=1
    )
    train_last[f'pred_x_{n_x}x{n_y}'] = predictions.apply(lambda x: x[0])
    train_last[f'pred_y_{n_x}x{n_y}'] = predictions.apply(lambda x: x[1])

# 가중치 Grid Search
print("\n가중치 조합 탐색:")
best_score = float('inf')
best_weights = None

for w1 in np.arange(0.15, 0.35, 0.05):
    for w2 in np.arange(0.45, 0.60, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.15 or w3 > 0.35:
            continue

        pred_x = (w1 * train_last['pred_x_5x5'] +
                  w2 * train_last['pred_x_6x6'] +
                  w3 * train_last['pred_x_7x7'])
        pred_y = (w1 * train_last['pred_y_5x5'] +
                  w2 * train_last['pred_y_6x6'] +
                  w3 * train_last['pred_y_7x7'])

        dist = np.sqrt((pred_x - train_last['end_x'])**2 + (pred_y - train_last['end_y'])**2)
        score = dist.mean()

        if score < best_score:
            best_score = score
            best_weights = (w1, w2, w3)
            print(f"  w5x5={w1:.2f}, w6x6={w2:.2f}, w7x7={w3:.2f}: CV = {score:.4f} *")

print(f"\n최적 가중치: 5x5={best_weights[0]:.2f}, 6x6={best_weights[1]:.2f}, 7x7={best_weights[2]:.2f}")
print(f"최적 CV Score: {best_score:.4f}")

# =============================================================================
# 8. CV로 앙상블 검증
# =============================================================================
print("\n[8] CV로 앙상블 검증...")

ensemble_scores = []
w1, w2, w3 = best_weights

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    # 각 Zone 크기별 모델 구축 및 예측
    fold_models = {}
    for n_x, n_y in ZONE_CONFIGS:
        fold_models[(n_x, n_y)] = build_8direction_model(train_fold, n_x, n_y)

    for n_x, n_y in ZONE_CONFIGS:
        predictions = val_fold.apply(
            lambda r: predict_8direction_zone(r, fold_models[(n_x, n_y)], use_adaptive_threshold=True),
            axis=1
        )
        val_fold[f'pred_x_{n_x}x{n_y}'] = predictions.apply(lambda x: x[0])
        val_fold[f'pred_y_{n_x}x{n_y}'] = predictions.apply(lambda x: x[1])

    # 앙상블 예측
    pred_x = (w1 * val_fold['pred_x_5x5'] +
              w2 * val_fold['pred_x_6x6'] +
              w3 * val_fold['pred_x_7x7'])
    pred_y = (w1 * val_fold['pred_y_5x5'] +
              w2 * val_fold['pred_y_6x6'] +
              w3 * val_fold['pred_y_7x7'])

    dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
    ensemble_scores.append(dist.mean())
    print(f"  Fold {fold+1}: {dist.mean():.4f}")

final_cv = np.mean(ensemble_scores)
final_std = np.std(ensemble_scores)
print(f"\n앙상블 평균 CV Score: {final_cv:.4f} ± {final_std:.4f}")

# =============================================================================
# 9. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[9] Test 예측 및 제출 파일 생성...")

# Test 데이터에 대해 각 모델 예측
for n_x, n_y in ZONE_CONFIGS:
    predictions = test_last.apply(
        lambda r: predict_8direction_zone(r, models[(n_x, n_y)], use_adaptive_threshold=True),
        axis=1
    )
    test_last[f'pred_x_{n_x}x{n_y}'] = predictions.apply(lambda x: x[0])
    test_last[f'pred_y_{n_x}x{n_y}'] = predictions.apply(lambda x: x[1])

# 앙상블
test_last['pred_x'] = (w1 * test_last['pred_x_5x5'] +
                       w2 * test_last['pred_x_6x6'] +
                       w3 * test_last['pred_x_7x7'])
test_last['pred_y'] = (w1 * test_last['pred_y_5x5'] +
                       w2 * test_last['pred_y_6x6'] +
                       w3 * test_last['pred_y_7x7'])

# 제출 파일 생성
submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['pred_x'],
    'end_y': test_last['pred_y']
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')

# 제출 디렉토리 확인
submission_dir = Path("submissions/pending")
submission_dir.mkdir(parents=True, exist_ok=True)
submission_path = submission_dir / 'submission_8direction_safe.csv'
submission.to_csv(submission_path, index=False)

print(f"  {submission_path} 저장 완료")
print(f"  CV Score: {final_cv:.4f}")

# 개별 모델 제출 파일도 생성 (6x6만)
sub_6x6 = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['pred_x_6x6'],
    'end_y': test_last['pred_y_6x6']
})
sub_6x6 = sample_sub[['game_episode']].merge(sub_6x6, on='game_episode', how='left')
sub_6x6_path = submission_dir / 'submission_8direction_6x6_only.csv'
sub_6x6.to_csv(sub_6x6_path, index=False)
print(f"  {sub_6x6_path} 저장 완료 (CV: {np.mean(cv_scores_adaptive[(6,6)]):.4f})")

# =============================================================================
# 10. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 권장사항")
print("=" * 80)

print(f"\n[모델 성능]")
print(f"  앙상블 CV:     {final_cv:.4f} ± {final_std:.4f}")
print(f"  6x6 단독 CV:   {np.mean(cv_scores_adaptive[(6,6)]):.4f}")
print(f"  과적합 위험:   {risk}")

print(f"\n[기준 모델과 비교]")
print(f"  5방향 앙상블:  CV 16.19 → Public 16.36 (Gap +0.17)")
print(f"  8방향 앙상블:  CV {final_cv:.4f} → Public ??? (예상 Gap +0.20-0.30)")
print(f"  CV 개선폭:     {16.19 - final_cv:+.4f}")

print(f"\n[예상 Public Score]")
if final_cv >= 16.0:
    lower = final_cv + 0.15
    upper = final_cv + 0.25
    print(f"  예상 범위:     {lower:.2f} - {upper:.2f}")
    print(f"  목표 달성:     {'✅ Yes' if lower <= 16.35 else '⚠️  Marginal'}")
elif final_cv >= 15.5:
    lower = final_cv + 0.25
    upper = final_cv + 0.40
    print(f"  예상 범위:     {lower:.2f} - {upper:.2f}")
    print(f"  목표 달성:     ⚠️  Uncertain")
else:
    print(f"  예상 Public:   > 16.50 (위험)")
    print(f"  목표 달성:     ❌ No")

print(f"\n[제출 권장]")
if final_cv >= 16.0 and final_cv <= 16.20:
    print(f"  ✅ 즉시 제출 권장")
    print(f"  이유: CV 안전 구간, 개선 가능성 높음")
elif final_cv >= 15.8 and final_cv < 16.0:
    print(f"  ⚠️  신중한 제출 권장")
    print(f"  이유: CV 경계 구간, Gap 증가 가능")
else:
    print(f"  ❌ 제출 보류 권장")
    print(f"  이유: 과적합 위험 높음")

print(f"\n[제출 파일]")
print(f"  1. {submission_path}")
print(f"     (앙상블: {best_weights[0]:.0%} × 5x5 + {best_weights[1]:.0%} × 6x6 + {best_weights[2]:.0%} × 7x7)")
print(f"  2. {sub_6x6_path}")
print(f"     (6x6 단독)")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
