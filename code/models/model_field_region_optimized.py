"""
K리그 패스 좌표 예측 - 필드 영역별 조건부 Zone 모델 (최적화)

필드 영역 기반 모델 개선:
1. 필드 3분할: defensive (x<35), midfield (35≤x<70), attacking (x≥70)
2. 영역별 균등한 Zone 해상도: 모두 6x6 사용 (baseline과 동일)
3. 5방향 분류와 결합
4. min_samples 최적화로 CV 개선

첫 번째 시도 분석:
- CV 16.38 (안전하지만 개선 여지 있음)
- defensive: 20.46 (4x4 너무 넓음)
- midfield: 17.86 (6x6 적절)
- attacking: 11.44 (8x8 너무 세밀, 과소적합)

개선 전략:
- 모든 영역에 6x6 적용 (일관성)
- min_samples를 영역별로 미세 조정
- fallback 로직 강화

예상 결과:
- CV: 16.10-16.25
- Gap: +0.17-0.25
- Public: 16.27-16.50
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 필드 영역별 조건부 Zone 모델 (최적화)")
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

def get_zone(x, y, n_x=6, n_y=6):
    """6x6 Zone 분류 (모든 영역 균등 적용)"""
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

def get_adaptive_min_samples(region):
    """
    영역별 동적 min_samples 조정 (최적화)

    목표: CV 16.2 부근 유지
    전략: 영역 특성에 맞는 threshold 설정
    """
    thresholds = {
        'defensive': 18,   # 수비: 샘플 많음, 약간 높은 threshold
        'midfield': 22,    # 중앙: 샘플 매우 많음, 높은 threshold (과적합 방지)
        'attacking': 16    # 공격: 샘플 적음, 낮은 threshold (과소적합 방지)
    }
    return thresholds.get(region, 20)

# =============================================================================
# 4. 모델 구축 및 예측 함수
# =============================================================================
print("\n[4] 모델 구축 및 예측 함수...")

def build_field_region_model(df, n_x=6, n_y=6):
    """필드 영역별 조건부 Zone 통계 구축 (6x6 균등)"""
    df = df.copy()

    # 필드 영역 분류
    df['region'] = df['start_x'].apply(get_field_region)

    # 방향 분류
    df['direction'] = df.apply(lambda r: get_direction_5way(r['prev_dx'], r['prev_dy']), axis=1)

    # Zone 분류 (모든 영역 6x6)
    df['zone'] = df.apply(lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1)

    # 복합 키 생성
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

    # Level 4: zone + direction (영역 무시)
    zone_dir_key = df['zone'].astype(str) + '_' + df['direction']
    zone_dir_stats = df.groupby(zone_dir_key).agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # Level 5: zone only
    zone_stats = df.groupby('zone').agg({
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
        'zone_dir_x': zone_dir_stats['delta_x'].to_dict(),
        'zone_dir_y': zone_dir_stats['delta_y'].to_dict(),
        'zone_dir_count': zone_dir_stats['count'].to_dict(),
        'zone_stats': zone_stats,
        'n_x': n_x,
        'n_y': n_y
    }

def predict_field_region(row, model):
    """필드 영역별 조건부 예측 (강화된 계층적 fallback)"""
    region = get_field_region(row['start_x'])
    zone = get_zone(row['start_x'], row['start_y'], model['n_x'], model['n_y'])
    direction = get_direction_5way(row['prev_dx'], row['prev_dy'])

    # 영역별 동적 threshold
    min_samples = get_adaptive_min_samples(region)

    # Level 1: region + zone + direction
    key_rzd = f"{region}_{zone}_{direction}"
    if key_rzd in model['region_zone_dir_x'] and model['region_zone_dir_count'].get(key_rzd, 0) >= min_samples:
        dx = model['region_zone_dir_x'][key_rzd]
        dy = model['region_zone_dir_y'][key_rzd]
    else:
        # Level 2: region + zone
        key_rz = f"{region}_{zone}"
        if key_rz in model['region_zone_x'] and model['region_zone_count'].get(key_rz, 0) >= min_samples * 0.7:
            dx = model['region_zone_x'][key_rz]
            dy = model['region_zone_y'][key_rz]
        else:
            # Level 3: region + direction
            key_rd = f"{region}_{direction}"
            if key_rd in model['region_dir_x'] and model['region_dir_count'].get(key_rd, 0) >= min_samples * 0.5:
                dx = model['region_dir_x'][key_rd]
                dy = model['region_dir_y'][key_rd]
            else:
                # Level 4: zone + direction (영역 무시)
                key_zd = f"{zone}_{direction}"
                if key_zd in model['zone_dir_x'] and model['zone_dir_count'].get(key_zd, 0) >= 15:
                    dx = model['zone_dir_x'][key_zd]
                    dy = model['zone_dir_y'][key_zd]
                else:
                    # Level 5: zone only
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

cv_scores = []
region_scores = {region: [] for region in ['defensive', 'midfield', 'attacking']}
fallback_levels = {f'level_{i}': [] for i in range(1, 6)}

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
        print(f"  {region:10s}: {mean_region:.4f} ± {std_region:.4f} (Zone: 6x6)")

# 기준 모델과 비교
print("\n기준 모델과 비교:")
print(f"  8방향 앙상블 (기존):  CV 16.03 → Public 16.36 (Gap +0.33)")
print(f"  필드 영역 v1:         CV 16.38 → Public ??? (Gap +0.20-0.30)")
print(f"  필드 영역 v2 (신규):  CV {mean_cv:.4f} → Public ??? (Gap +0.17-0.25)")
print(f"  v1 대비 개선:         {16.38 - mean_cv:+.4f}")
print(f"  8방향 대비:           {16.03 - mean_cv:+.4f}")

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
output_path = DATA_DIR / 'submission_field_region_v2.csv'
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
        print(f"  {region:10s}: {mean_region:.4f} (Zone: 6x6)")

print(f"\n[기준 모델과 비교]")
print(f"  8방향 앙상블 (Best):  CV 16.03 → Public 16.36")
print(f"  필드 영역 v2:         CV {mean_cv:.4f} → Public {mean_cv + 0.20:.2f} (예상)")

print(f"\n[제출 권장]")
if mean_cv >= 16.2 and mean_cv <= 16.30:
    print(f"  ✅ 즉시 제출 권장")
    print(f"  이유: CV 안전 구간, Gap 안정적, 개선 가능성 있음")
elif mean_cv < 16.03:
    print(f"  ✅ 강력 추천")
    print(f"  이유: CV가 현재 Best보다 낮음, Public 개선 가능")
elif mean_cv >= 16.0:
    print(f"  ⚠️  신중한 제출 권장")
    print(f"  이유: CV 경계 구간, Gap 증가 가능")
else:
    print(f"  ❌ 제출 보류 권장")
    print(f"  이유: 과적합 위험")

print(f"\n[제출 파일]")
print(f"  {output_path}")
print(f"  (필드 3분할 + 6x6 균등 Zone + 5방향 + 강화된 fallback)")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
