"""
K리그 패스 좌표 예측 - Extended Zone Experiments
6x6, 7x7, 8x8 구역 테스트 + Bootstrap 안정성 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - Extended Zone Experiments")
print("=" * 70)

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

# =============================================================================
# 2. 마지막 액션 추출
# =============================================================================
print("\n[2] 마지막 액션 추출...")

train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

test_last = test_all.groupby('game_episode').last().reset_index()

print(f"Train last actions: {len(train_last):,}")
print(f"Test last actions: {len(test_last):,}")

# 전체 통계
global_median_dx = train_last['delta_x'].median()
global_median_dy = train_last['delta_y'].median()

# =============================================================================
# 3. Extended Zone 함수들
# =============================================================================
print("\n[3] Extended Zone 전략 정의...")

def get_zone_nxn(x, y, n):
    """일반화된 NxN 구역 함수"""
    x_zone = min(n-1, int(x / (105 / n)))
    y_zone = min(n-1, int(y / (68 / n)))
    return x_zone * n + y_zone

def get_zone_3x3(x, y): return get_zone_nxn(x, y, 3)
def get_zone_4x4(x, y): return get_zone_nxn(x, y, 4)
def get_zone_5x5(x, y): return get_zone_nxn(x, y, 5)
def get_zone_6x6(x, y): return get_zone_nxn(x, y, 6)
def get_zone_7x7(x, y): return get_zone_nxn(x, y, 7)
def get_zone_8x8(x, y): return get_zone_nxn(x, y, 8)

zone_functions = {
    '3x3': (get_zone_3x3, 9),
    '4x4': (get_zone_4x4, 16),
    '5x5': (get_zone_5x5, 25),
    '6x6': (get_zone_6x6, 36),
    '7x7': (get_zone_7x7, 49),
    '8x8': (get_zone_8x8, 64),
}

# =============================================================================
# 4. CV 평가 함수
# =============================================================================
def evaluate_zone_cv(train_data, zone_func, use_median=True):
    """Zone 전략 CV 평가"""
    train_data = train_data.copy()
    train_data['zone'] = train_data.apply(
        lambda r: zone_func(r['start_x'], r['start_y']), axis=1
    )

    # 구역별 통계 계산
    agg_func = 'median' if use_median else 'mean'
    zone_stats = train_data.groupby('zone').agg({
        'delta_x': agg_func,
        'delta_y': agg_func
    }).to_dict()

    # 구역별 샘플 수
    zone_counts = train_data['zone'].value_counts()
    min_samples = zone_counts.min()
    n_zones = len(zone_counts)

    # 예측 및 평가
    predictions = []
    for _, row in train_data.iterrows():
        zone = row['zone']
        dx = zone_stats['delta_x'].get(zone, global_median_dx)
        dy = zone_stats['delta_y'].get(zone, global_median_dy)
        pred_x = np.clip(row['start_x'] + dx, 0, 105)
        pred_y = np.clip(row['start_y'] + dy, 0, 68)
        predictions.append([pred_x, pred_y])

    predictions = np.array(predictions)
    actuals = train_data[['end_x', 'end_y']].values

    distances = np.sqrt(np.sum((predictions - actuals) ** 2, axis=1))
    cv_score = distances.mean()

    return cv_score, n_zones, min_samples, zone_stats, zone_counts

# =============================================================================
# 5. Bootstrap 안정성 검증
# =============================================================================
def bootstrap_stability(train_data, zone_func, n_iterations=30, sample_ratio=0.8):
    """Bootstrap으로 안정성 검증"""
    scores = []
    n_samples = int(len(train_data) * sample_ratio)

    for i in range(n_iterations):
        # 랜덤 샘플링
        sample_idx = np.random.choice(len(train_data), n_samples, replace=False)
        sample_data = train_data.iloc[sample_idx]

        cv, _, _, _, _ = evaluate_zone_cv(sample_data, zone_func, use_median=True)
        scores.append(cv)

    return np.mean(scores), np.std(scores)

# =============================================================================
# 6. 전체 실험 실행
# =============================================================================
print("\n[4] Zone 전략별 CV 평가 (median)...")
print("=" * 70)
print(f"{'전략':<10} {'CV Score':<12} {'구역수':<8} {'최소샘플':<10} {'Bootstrap Std':<15} {'위험도'}")
print("-" * 70)

results = []

for name, (func, expected_zones) in zone_functions.items():
    # CV 평가
    cv, n_zones, min_samples, zone_stats, zone_counts = evaluate_zone_cv(
        train_last, func, use_median=True
    )

    # Bootstrap 안정성 (6x6 이상만)
    if expected_zones >= 36:
        boot_mean, boot_std = bootstrap_stability(train_last, func, n_iterations=30)
    else:
        boot_mean, boot_std = cv, 0.2  # 이전 결과 참조

    # 위험도 판단
    if min_samples < 100:
        risk = "높음 (샘플 부족)"
    elif min_samples < 200:
        risk = "중간"
    elif boot_std > 0.3:
        risk = "중간 (불안정)"
    else:
        risk = "낮음"

    results.append({
        'name': name,
        'cv': cv,
        'n_zones': n_zones,
        'min_samples': min_samples,
        'boot_std': boot_std,
        'risk': risk,
        'zone_stats': zone_stats,
        'zone_counts': zone_counts
    })

    print(f"{name:<10} {cv:<12.4f} {n_zones:<8} {min_samples:<10} {boot_std:<15.3f} {risk}")

# =============================================================================
# 7. 6x6 세부 분석
# =============================================================================
print("\n" + "=" * 70)
print("[5] 6x6 구역 세부 분석")
print("=" * 70)

result_6x6 = next(r for r in results if r['name'] == '6x6')
zone_counts_6x6 = result_6x6['zone_counts']

print("\n구역별 샘플 수 분포:")
print(f"  최소: {zone_counts_6x6.min()}")
print(f"  최대: {zone_counts_6x6.max()}")
print(f"  평균: {zone_counts_6x6.mean():.1f}")
print(f"  중앙값: {zone_counts_6x6.median():.1f}")

# 샘플이 적은 구역
low_sample_zones = zone_counts_6x6[zone_counts_6x6 < 200]
print(f"\n샘플 200개 미만 구역: {len(low_sample_zones)}개")
if len(low_sample_zones) > 0:
    print(f"  구역 번호: {list(low_sample_zones.index)}")
    print(f"  샘플 수: {list(low_sample_zones.values)}")

# =============================================================================
# 8. 제출 파일 생성 (6x6 median)
# =============================================================================
print("\n" + "=" * 70)
print("[6] 제출 파일 생성")
print("=" * 70)

def generate_submission(test_data, zone_func, zone_stats, sample_sub):
    """제출 파일 생성"""
    predictions = []

    for _, row in test_data.iterrows():
        zone = zone_func(row['start_x'], row['start_y'])
        dx = zone_stats['delta_x'].get(zone, global_median_dx)
        dy = zone_stats['delta_y'].get(zone, global_median_dy)
        pred_x = np.clip(row['start_x'] + dx, 0, 105)
        pred_y = np.clip(row['start_y'] + dy, 0, 68)
        predictions.append({
            'game_episode': row['game_episode'],
            'end_x': pred_x,
            'end_y': pred_y
        })

    pred_df = pd.DataFrame(predictions)
    return sample_sub[['game_episode']].merge(pred_df, on='game_episode', how='left')

# 6x6 median
result_6x6 = next(r for r in results if r['name'] == '6x6')
sub_6x6 = generate_submission(test_last, get_zone_6x6, result_6x6['zone_stats'], sample_sub)
sub_6x6.to_csv('submission_6x6_36구역_median.csv', index=False)
print(f"  submission_6x6_36구역_median.csv: CV={result_6x6['cv']:.4f}")

# 7x7 median
result_7x7 = next(r for r in results if r['name'] == '7x7')
sub_7x7 = generate_submission(test_last, get_zone_7x7, result_7x7['zone_stats'], sample_sub)
sub_7x7.to_csv('submission_7x7_49구역_median.csv', index=False)
print(f"  submission_7x7_49구역_median.csv: CV={result_7x7['cv']:.4f}")

# =============================================================================
# 9. 정교한 앙상블 (5x5 + 6x6)
# =============================================================================
print("\n" + "=" * 70)
print("[7] 정교한 앙상블 생성")
print("=" * 70)

# 기존 5x5 median 로드
sub_5x5 = pd.read_csv('submission_5x5_25구역_median.csv')

# 앙상블 1: 5x5와 6x6 50:50
ensemble_5x5_6x6 = pd.DataFrame({
    'game_episode': sample_sub['game_episode'],
    'end_x': np.clip(0.5 * sub_5x5['end_x'] + 0.5 * sub_6x6['end_x'], 0, 105),
    'end_y': np.clip(0.5 * sub_5x5['end_y'] + 0.5 * sub_6x6['end_y'], 0, 68)
})
ensemble_5x5_6x6.to_csv('submission_ensemble_5x5_6x6.csv', index=False)
print(f"  submission_ensemble_5x5_6x6.csv: 5x5 50% + 6x6 50%")

# 앙상블 2: 5x5 70% + 6x6 30% (보수적)
ensemble_conservative = pd.DataFrame({
    'game_episode': sample_sub['game_episode'],
    'end_x': np.clip(0.7 * sub_5x5['end_x'] + 0.3 * sub_6x6['end_x'], 0, 105),
    'end_y': np.clip(0.7 * sub_5x5['end_y'] + 0.3 * sub_6x6['end_y'], 0, 68)
})
ensemble_conservative.to_csv('submission_ensemble_5x5_70_6x6_30.csv', index=False)
print(f"  submission_ensemble_5x5_70_6x6_30.csv: 5x5 70% + 6x6 30%")

# 앙상블 3: 4x4 + 5x5 + 6x6 (다양성)
sub_4x4 = pd.read_csv('submission_4x4_16구역_median.csv')
ensemble_diverse = pd.DataFrame({
    'game_episode': sample_sub['game_episode'],
    'end_x': np.clip((sub_4x4['end_x'] + sub_5x5['end_x'] + sub_6x6['end_x']) / 3, 0, 105),
    'end_y': np.clip((sub_4x4['end_y'] + sub_5x5['end_y'] + sub_6x6['end_y']) / 3, 0, 68)
})
ensemble_diverse.to_csv('submission_ensemble_4x4_5x5_6x6.csv', index=False)
print(f"  submission_ensemble_4x4_5x5_6x6.csv: 4x4 + 5x5 + 6x6 동일 가중치")

# =============================================================================
# 10. 최종 결과 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 결과 요약")
print("=" * 70)

print("\n[CV Score 순위]")
print("-" * 70)
sorted_results = sorted(results, key=lambda x: x['cv'])
for i, r in enumerate(sorted_results, 1):
    print(f"{i}. {r['name']}: CV={r['cv']:.4f}, 최소샘플={r['min_samples']}, 위험도={r['risk']}")

print("\n[권장 전략]")
print("-" * 70)
best = sorted_results[0]
print(f"Best CV: {best['name']} (CV={best['cv']:.4f})")

# CV 개선량 계산
cv_5x5 = next(r['cv'] for r in results if r['name'] == '5x5')
cv_6x6 = next(r['cv'] for r in results if r['name'] == '6x6')
cv_7x7 = next(r['cv'] for r in results if r['name'] == '7x7')

print(f"\n[CV 변화]")
print(f"  5x5 → 6x6: {cv_5x5:.4f} → {cv_6x6:.4f} (차이: {cv_6x6 - cv_5x5:+.4f})")
print(f"  6x6 → 7x7: {cv_6x6:.4f} → {cv_7x7:.4f} (차이: {cv_7x7 - cv_6x6:+.4f})")

# 위험도 판단
min_6x6 = next(r['min_samples'] for r in results if r['name'] == '6x6')
min_7x7 = next(r['min_samples'] for r in results if r['name'] == '7x7')

print(f"\n[샘플 수 분석]")
print(f"  6x6: 최소 {min_6x6}개 {'(충분)' if min_6x6 >= 100 else '(부족)'}")
print(f"  7x7: 최소 {min_7x7}개 {'(충분)' if min_7x7 >= 100 else '(부족)'}")

print("\n[제출 추천]")
print("-" * 70)
if cv_6x6 < cv_5x5 and min_6x6 >= 100:
    print("1. submission_6x6_36구역_median.csv - CV 개선, 샘플 충분")
    print("2. submission_ensemble_5x5_6x6.csv - 안전한 앙상블")
else:
    print("1. 기존 5x5 유지 - 6x6이 개선 없음 또는 위험")
    print("2. submission_ensemble_5x5_70_6x6_30.csv - 보수적 앙상블")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
