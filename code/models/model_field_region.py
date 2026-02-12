"""
K리그 패스 좌표 예측 - 필드 영역별 조건부 Zone 모델

TacticAI 논문 인사이트를 반영한 필드 영역 기반 모델:
1. 필드 3분할: defensive (x<35), midfield (35≤x<70), attacking (x≥70)
2. 영역별 최적 Zone 해상도:
   - defensive: 4x4 (넓은 Zone, 수비는 단순)
   - midfield: 6x6 (중간 Zone, 중앙 밀집)
   - attacking: 8x8 (세밀한 Zone, 공격은 정교)
3. 5방향 분류와 결합
4. CV 16.2+ 유지를 위한 min_samples 조정

예상 결과:
- CV: 16.10-16.30
- Gap: +0.20-0.30
- Public: 16.30-16.60
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 필드 영역별 조건부 Zone 모델")
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
# 3. 필드 영역 및 Zone 분류 함수
# =============================================================================
print("\n[3] 필드 영역 및 Zone 분류 함수 정의...")

def get_field_region(x):
    """
    필드 영역 분류 (start_x 기준)

    - defensive: x < 35 (수비 3분의1)
    - midfield: 35 <= x < 70 (중앙 3분의1)
    - attacking: x >= 70 (공격 3분의1)
    """
    if x < 35:
        return 'defensive'
    elif x < 70:
        return 'midfield'
    else:
        return 'attacking'

def get_zone(x, y, n_x, n_y):
    """NxN Zone 분류"""
    x_zone = min(n_x - 1, int(x / (105 / n_x)))
    y_zone = min(n_y - 1, int(y / (68 / n_y)))
    return x_zone * n_y + y_zone

def get_direction_5way(prev_dx, prev_dy):
    """
    5방향 분류

    방향:
    - none: 움직임 없음
    - forward: 앞으로 (dx > 0 dominant)
    - backward: 뒤로 (dx < 0 dominant)
    - up: 위 (dy > 0 dominant)
    - down: 아래 (dy < 0 dominant)
    """
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'

    if abs(prev_dx) > abs(prev_dy):
        return 'forward' if prev_dx > 0 else 'backward'
    else:
        return 'up' if prev_dy > 0 else 'down'

def get_region_zone_config(region):
    """
    영역별 최적 Zone 해상도 반환

    - defensive: 4x4 (넓은 Zone, 수비는 단순)
    - midfield: 6x6 (중간 Zone, 중앙 밀집)
    - attacking: 8x8 (세밀한 Zone, 공격은 정교)
    """
    config = {
        'defensive': (4, 4),
        'midfield': (6, 6),
        'attacking': (8, 8)
    }
    return config[region]

def get_adaptive_min_samples(region, zone, n_zones):
    """
    영역 및 Zone별 동적 min_samples 조정

    목표: CV 16.2+ 유지
    전략: 샘플이 적은 영역에서는 threshold를 낮춰 과소적합 방지
    """
    base_thresholds = {
        'defensive': 15,  # 수비 영역: 샘플 많음, 낮은 threshold
        'midfield': 20,   # 중앙 영역: 샘플 매우 많음, 기본 threshold
        'attacking': 25   # 공격 영역: 샘플 적음, 높은 threshold (과적합 방지)
    }

    return base_thresholds.get(region, 20)

# =============================================================================
# 4. 모델 구축 및 예측 함수
# =============================================================================
print("\n[4] 모델 구축 및 예측 함수...")

def build_field_region_model(df):
    """필드 영역별 조건부 Zone 통계 구축"""
    df = df.copy()

    # 필드 영역 분류
    df['region'] = df['start_x'].apply(get_field_region)

    # 방향 분류
    df['direction'] = df.apply(lambda r: get_direction_5way(r['prev_dx'], r['prev_dy']), axis=1)

    # 영역별로 다른 Zone 해상도 적용
    def get_row_zone(row):
        n_x, n_y = get_region_zone_config(row['region'])
        return get_zone(row['start_x'], row['start_y'], n_x, n_y)

    df['zone'] = df.apply(get_row_zone, axis=1)
    df['region_zone_dir'] = df['region'] + '_' + df['zone'].astype(str) + '_' + df['direction']
    df['region_zone'] = df['region'] + '_' + df['zone'].astype(str)
    df['region_dir'] = df['region'] + '_' + df['direction']

    # 통계 구축 (계층적 fallback)
    # Level 1: region + zone + direction
    region_zone_dir_stats = df.groupby('region_zone_dir').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # Level 2: region + zone
    region_zone_stats = df.groupby('region_zone').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # Level 3: region + direction
    region_dir_stats = df.groupby('region_dir').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # Level 4: region only
    region_stats = df.groupby('region').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    return {
        'region_zone_dir_x': region_zone_dir_stats['delta_x'].to_dict(),
        'region_zone_dir_y': region_zone_dir_stats['delta_y'].to_dict(),
        'region_zone_dir_count': region_zone_dir_stats['count'].to_dict(),
        'region_zone_x': region_zone_stats['delta_x'].to_dict(),
        'region_zone_y': region_zone_stats['delta_y'].to_dict(),
        'region_zone_count': region_zone_stats['count'].to_dict(),
        'region_dir_x': region_dir_stats['delta_x'].to_dict(),
        'region_dir_y': region_dir_stats['delta_y'].to_dict(),
        'region_dir_count': region_dir_stats['count'].to_dict(),
        'region_stats': region_stats
    }

def predict_field_region(row, model):
    """필드 영역별 조건부 예측 (계층적 fallback)"""
    region = get_field_region(row['start_x'])
    n_x, n_y = get_region_zone_config(region)
    zone = get_zone(row['start_x'], row['start_y'], n_x, n_y)
    direction = get_direction_5way(row['prev_dx'], row['prev_dy'])

    # 동적 threshold
    min_samples = get_adaptive_min_samples(region, zone, n_x * n_y)

    # Level 1: region + zone + direction
    key_rzd = f"{region}_{zone}_{direction}"
    if key_rzd in model['region_zone_dir_x'] and model['region_zone_dir_count'].get(key_rzd, 0) >= min_samples:
        dx = model['region_zone_dir_x'][key_rzd]
        dy = model['region_zone_dir_y'][key_rzd]
    else:
        # Level 2: region + zone
        key_rz = f"{region}_{zone}"
        if key_rz in model['region_zone_x'] and model['region_zone_count'].get(key_rz, 0) >= min_samples:
            dx = model['region_zone_x'][key_rz]
            dy = model['region_zone_y'][key_rz]
        else:
            # Level 3: region + direction
            key_rd = f"{region}_{direction}"
            if key_rd in model['region_dir_x'] and model['region_dir_count'].get(key_rd, 0) >= min_samples * 0.5:
                dx = model['region_dir_x'][key_rd]
                dy = model['region_dir_y'][key_rd]
            else:
                # Level 4: region only
                dx = model['region_stats']['delta_x'].get(region, 0)
                dy = model['region_stats']['delta_y'].get(region, 0)

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

cv_scores = []
region_scores = {region: [] for region in ['defensive', 'midfield', 'attacking']}

print("\n  5-Fold Cross Validation:")
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    # 모델 구축
    model = build_field_region_model(train_fold)

    # 예측
    predictions = val_fold.apply(
        lambda r: predict_field_region(r, model),
        axis=1
    )
    val_fold['pred_x'] = predictions.apply(lambda x: x[0])
    val_fold['pred_y'] = predictions.apply(lambda x: x[1])

    # 전체 성능
    dist = np.sqrt(
        (val_fold['pred_x'] - val_fold['end_x'])**2 +
        (val_fold['pred_y'] - val_fold['end_y'])**2
    )
    score = dist.mean()
    cv_scores.append(score)

    print(f"\n  Fold {fold+1}: {score:.4f}")

    # 영역별 성능 분석
    val_fold['region'] = val_fold['start_x'].apply(get_field_region)
    for region in ['defensive', 'midfield', 'attacking']:
        region_mask = val_fold['region'] == region
        if region_mask.sum() > 0:
            region_dist = dist[region_mask].mean()
            region_scores[region].append(region_dist)
            print(f"    {region:10s}: {region_dist:.4f} (n={region_mask.sum():,})")

# =============================================================================
# 6. CV 결과 요약
# =============================================================================
print("\n" + "=" * 80)
print("[6] CV 결과 요약")
print("=" * 80)

mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)

print(f"\n전체 CV Score: {mean_cv:.4f} ± {std_cv:.4f}")

print("\n영역별 평균 성능:")
for region in ['defensive', 'midfield', 'attacking']:
    if region_scores[region]:
        mean_region = np.mean(region_scores[region])
        std_region = np.std(region_scores[region])
        n_x, n_y = get_region_zone_config(region)
        print(f"  {region:10s}: {mean_region:.4f} ± {std_region:.4f} (Zone: {n_x}x{n_y})")

# 기준 모델과 비교
print("\n기준 모델과 비교:")
print(f"  8방향 앙상블 (기존):  CV 16.03 → Public 16.36 (Gap +0.33)")
print(f"  필드 영역 (신규):     CV {mean_cv:.4f} → Public ??? (예상 Gap +0.20-0.30)")
print(f"  CV 개선폭:            {16.03 - mean_cv:+.4f}")

# 과적합 위험 평가
print(f"\n과적합 위험 평가:")
if mean_cv >= 16.2:
    print(f"  ✅ SAFE: CV {mean_cv:.4f} >= 16.2 (안전 구간)")
    print(f"  예상 Gap: +0.15-0.25")
    print(f"  예상 Public: {mean_cv + 0.15:.2f} - {mean_cv + 0.25:.2f}")
    risk = "낮음"
elif mean_cv >= 16.0:
    print(f"  ⚠️  WARNING: CV {mean_cv:.4f} in 16.0-16.2 (경계 구간)")
    print(f"  예상 Gap: +0.25-0.35")
    print(f"  예상 Public: {mean_cv + 0.25:.2f} - {mean_cv + 0.35:.2f}")
    risk = "중간"
else:
    print(f"  ❌ DANGER: CV {mean_cv:.4f} < 16.0 (위험 구간)")
    print(f"  예상 Gap: +0.35+")
    print(f"  예상 Public: > {mean_cv + 0.35:.2f}")
    risk = "높음"

# =============================================================================
# 7. 전체 데이터로 최종 모델 학습
# =============================================================================
print("\n[7] 전체 데이터로 최종 모델 학습...")

final_model = build_field_region_model(train_last)
print("  모델 학습 완료")

# 학습 데이터에 대한 성능 확인
train_predictions = train_last.apply(
    lambda r: predict_field_region(r, final_model),
    axis=1
)
train_last['pred_x'] = train_predictions.apply(lambda x: x[0])
train_last['pred_y'] = train_predictions.apply(lambda x: x[1])

train_dist = np.sqrt(
    (train_last['pred_x'] - train_last['end_x'])**2 +
    (train_last['pred_y'] - train_last['end_y'])**2
)
train_score = train_dist.mean()
print(f"  Train Score: {train_score:.4f}")

# =============================================================================
# 8. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[8] Test 예측 및 제출 파일 생성...")

# Test 데이터 예측
test_predictions = test_last.apply(
    lambda r: predict_field_region(r, final_model),
    axis=1
)
test_last['pred_x'] = test_predictions.apply(lambda x: x[0])
test_last['pred_y'] = test_predictions.apply(lambda x: x[1])

# 제출 파일 생성
submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['pred_x'],
    'end_y': test_last['pred_y']
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')

# 저장
output_path = DATA_DIR / 'submission_field_region.csv'
submission.to_csv(output_path, index=False)

print(f"  {output_path} 저장 완료")
print(f"  CV Score: {mean_cv:.4f}")

# Test 데이터 영역별 분포
test_last['region'] = test_last['start_x'].apply(get_field_region)
print("\nTest 데이터 영역별 분포:")
for region in ['defensive', 'midfield', 'attacking']:
    count = (test_last['region'] == region).sum()
    pct = count / len(test_last) * 100
    print(f"  {region:10s}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 권장사항")
print("=" * 80)

print(f"\n[모델 성능]")
print(f"  CV Score:      {mean_cv:.4f} ± {std_cv:.4f}")
print(f"  Train Score:   {train_score:.4f}")
print(f"  과적합 위험:   {risk}")

print(f"\n[영역별 성능]")
for region in ['defensive', 'midfield', 'attacking']:
    if region_scores[region]:
        mean_region = np.mean(region_scores[region])
        n_x, n_y = get_region_zone_config(region)
        print(f"  {region:10s}: {mean_region:.4f} (Zone: {n_x}x{n_y})")

print(f"\n[기준 모델과 비교]")
print(f"  8방향 앙상블:  CV 16.03 → Public 16.36 (Gap +0.33)")
print(f"  필드 영역:     CV {mean_cv:.4f} → Public ??? (예상 Gap +0.20-0.30)")
print(f"  CV 개선폭:     {16.03 - mean_cv:+.4f}")

print(f"\n[제출 권장]")
if mean_cv >= 16.2:
    print(f"  ✅ 즉시 제출 권장")
    print(f"  이유: CV 안전 구간 (≥16.2), Gap 안정적")
elif mean_cv >= 16.0:
    print(f"  ⚠️  신중한 제출 권장")
    print(f"  이유: CV 경계 구간 (16.0-16.2), Gap 증가 가능")
else:
    print(f"  ❌ 제출 보류 권장")
    print(f"  이유: CV 위험 구간 (<16.0), 과적합 위험")

print(f"\n[제출 파일]")
print(f"  {output_path}")
print(f"  (필드 3분할 + 영역별 최적 Zone 해상도)")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
