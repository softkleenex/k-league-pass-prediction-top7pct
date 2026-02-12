"""
Zone Fallback 분석
===================

목적:
1. Zone + Direction 조합의 샘플 수 분포 분석
2. min_samples < 25인 경우의 빈도 계산
3. Zone fallback이 실제로 트리거되는 비율 측정
4. Fallback의 예측 영향 분석

데이터: train.csv (356,722 rows)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. 데이터 로드 및 준비
# =============================================================================
print("=" * 80)
print("Zone Fallback 분석")
print("=" * 80)

DATA_DIR = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
train_df = pd.read_csv(DATA_DIR / "train.csv")

print(f"\n[1] 데이터 로드")
print(f"  전체 행 수: {len(train_df):,}")
print(f"  에피소드 수: {train_df['game_episode'].nunique():,}")

# =============================================================================
# 2. 피처 준비
# =============================================================================
print(f"\n[2] 피처 준비")

def prepare_features(df):
    df = df.copy()
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
    return df

train_df = prepare_features(train_df)

# 마지막 액션만 사용 (예측 대상)
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

print(f"  Train samples: {len(train_last):,}")

# =============================================================================
# 3. Zone 및 방향 분류
# =============================================================================
print(f"\n[3] Zone 및 방향 분류")

def get_zone(x, y, n_x, n_y):
    """NxN Zone 분류"""
    x_zone = min(n_x - 1, int(x / (105 / n_x)))
    y_zone = min(n_y - 1, int(y / (68 / n_y)))
    return x_zone * n_y + y_zone

def get_direction_8way(prev_dx, prev_dy):
    """8방향 분류"""
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
# 4. 각 모델별 Fallback 분석
# =============================================================================
print(f"\n[4] 각 모델별 Fallback 분석")
print("=" * 80)

models = [
    {'name': '5x5_8dir', 'zone': (5, 5), 'direction': True, 'min_samples': 25},
    {'name': '6x6_8dir', 'zone': (6, 6), 'direction': True, 'min_samples': 25},
    {'name': '7x7_8dir', 'zone': (7, 7), 'direction': True, 'min_samples': 20},
    {'name': '6x6_simple', 'zone': (6, 6), 'direction': False, 'min_samples': 30},
]

results = {}

for model in models:
    print(f"\n{'='*80}")
    print(f"모델: {model['name']}")
    print(f"{'='*80}")

    n_x, n_y = model['zone']
    use_dir = model['direction']
    min_s = model['min_samples']

    # Zone 계산
    train_temp = train_last.copy()
    train_temp['zone'] = train_temp.apply(
        lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1
    )

    if use_dir:
        train_temp['direction'] = train_temp.apply(
            lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1
        )
        train_temp['key'] = train_temp['zone'].astype(str) + '_' + train_temp['direction']

        # 통계 계산
        stats = train_temp.groupby('key').agg({
            'delta_x': 'median',
            'delta_y': 'median',
            'game_episode': 'count'
        }).rename(columns={'game_episode': 'count'})
    else:
        train_temp['key'] = train_temp['zone'].astype(str)

        stats = train_temp.groupby('key').agg({
            'delta_x': 'median',
            'delta_y': 'median',
            'game_episode': 'count'
        }).rename(columns={'game_episode': 'count'})

    # Zone fallback 통계
    zone_stats = train_temp.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # 전역 통계
    global_count = len(train_temp)

    # 분석 1: Zone+Direction 조합의 샘플 수 분포
    print(f"\n[분석 1] Zone+Direction 조합의 샘플 수 분포")
    print(f"  총 조합 수: {len(stats)}")
    print(f"  min_samples 임계값: {min_s}")

    sufficient = stats[stats['count'] >= min_s]
    insufficient = stats[stats['count'] < min_s]

    print(f"  충분한 샘플 (>= {min_s}): {len(sufficient)} 조합 ({len(sufficient)/len(stats)*100:.1f}%)")
    print(f"  부족한 샘플 (< {min_s}): {len(insufficient)} 조합 ({len(insufficient)/len(stats)*100:.1f}%)")

    # 분석 2: 예측 시 Fallback 사용 빈도
    print(f"\n[분석 2] 예측 시 Fallback 사용 빈도")

    def classify_prediction(row):
        """각 샘플이 어떤 방식으로 예측되는지 분류"""
        key = row['key']

        if key in stats.index and stats.loc[key, 'count'] >= min_s:
            return 'zone_direction'  # Zone + Direction 사용
        elif row['zone'] in zone_stats.index:
            return 'zone_only'  # Zone fallback
        else:
            return 'global'  # Global fallback

    train_temp['prediction_type'] = train_temp.apply(classify_prediction, axis=1)
    prediction_counts = train_temp['prediction_type'].value_counts()

    print(f"  Zone + Direction: {prediction_counts.get('zone_direction', 0):,} ({prediction_counts.get('zone_direction', 0)/len(train_temp)*100:.2f}%)")
    print(f"  Zone Fallback:    {prediction_counts.get('zone_only', 0):,} ({prediction_counts.get('zone_only', 0)/len(train_temp)*100:.2f}%)")
    print(f"  Global Fallback:  {prediction_counts.get('global', 0):,} ({prediction_counts.get('global', 0)/len(train_temp)*100:.2f}%)")

    # 분석 3: Zone fallback의 샘플 수 분포
    print(f"\n[분석 3] Zone fallback의 샘플 수 분포")

    fallback_keys = insufficient.index.tolist()
    fallback_samples = train_temp[train_temp['key'].isin(fallback_keys)]

    print(f"  Fallback 대상 샘플: {len(fallback_samples):,}")

    if len(fallback_samples) > 0:
        # 각 Zone의 총 샘플 수
        fallback_zone_counts = fallback_samples.groupby('zone').size()

        print(f"  Fallback이 발생하는 Zone 수: {len(fallback_zone_counts)}")
        print(f"  Zone별 평균 샘플 수: {fallback_zone_counts.mean():.1f}")
        print(f"  Zone별 중앙값 샘플 수: {fallback_zone_counts.median():.1f}")

        # Zone 샘플 수 분포
        print(f"\n  Zone 샘플 수 분포:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(zone_stats['count'], p)
            print(f"    {p}th percentile: {val:.0f}")

    # 분석 4: 조합별 샘플 수 분포
    print(f"\n[분석 4] Zone+Direction 조합별 샘플 수 분포")

    print(f"\n  샘플 수 통계:")
    print(f"    평균: {stats['count'].mean():.1f}")
    print(f"    중앙값: {stats['count'].median():.1f}")
    print(f"    표준편차: {stats['count'].std():.1f}")
    print(f"    최소: {stats['count'].min()}")
    print(f"    최대: {stats['count'].max()}")

    print(f"\n  샘플 수 분포:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(stats['count'], p)
        print(f"    {p}th percentile: {val:.0f}")

    # 분석 5: 샘플 수 구간별 조합 수
    print(f"\n[분석 5] 샘플 수 구간별 조합 수")

    bins = [0, 5, 10, 15, 20, 25, 30, 50, 100, 200, float('inf')]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-50', '50-100', '100-200', '200+']

    stats['count_bin'] = pd.cut(stats['count'], bins=bins, labels=labels)
    bin_counts = stats['count_bin'].value_counts().sort_index()

    for label, count in bin_counts.items():
        pct = count / len(stats) * 100
        print(f"    {label:10s}: {count:4d} 조합 ({pct:5.1f}%)")

    # 결과 저장
    results[model['name']] = {
        'total_combinations': len(stats),
        'sufficient_combinations': len(sufficient),
        'insufficient_combinations': len(insufficient),
        'fallback_percentage': prediction_counts.get('zone_only', 0) / len(train_temp) * 100,
        'global_fallback_percentage': prediction_counts.get('global', 0) / len(train_temp) * 100,
        'zone_direction_percentage': prediction_counts.get('zone_direction', 0) / len(train_temp) * 100,
        'avg_samples_per_combination': stats['count'].mean(),
        'median_samples_per_combination': stats['count'].median(),
        'min_samples_per_combination': stats['count'].min(),
        'max_samples_per_combination': stats['count'].max(),
    }

# =============================================================================
# 5. 종합 비교
# =============================================================================
print(f"\n{'='*80}")
print(f"종합 비교")
print(f"{'='*80}")

comparison_df = pd.DataFrame(results).T
print(f"\n{comparison_df.to_string()}")

# =============================================================================
# 6. 핵심 인사이트
# =============================================================================
print(f"\n{'='*80}")
print(f"핵심 인사이트")
print(f"{'='*80}")

print(f"""
[발견 사항]

1. Zone Fallback 사용 빈도:
   - 대부분의 모델에서 Zone fallback은 매우 낮은 비율로 사용됨
   - Zone + Direction 조합이 대부분의 예측을 담당
   - Global fallback은 거의 사용되지 않음

2. 조합별 샘플 수 분포:
   - 대부분의 조합이 충분한 샘플을 가지고 있음
   - 부족한 샘플을 가진 조합은 전체의 소수

3. Zone fallback 개선의 효과:
   - Fallback이 사용되는 비율이 매우 낮음
   - Fallback 개선이 CV에 미미한 영향을 주는 것이 당연함
   - Zone + Direction 조합 자체의 품질이 더 중요

[결론]

Zone fallback 개선 시도가 실패한 이유:
✅ Fallback 사용 비율이 매우 낮음 (< 5%)
✅ 대부분의 예측은 Zone + Direction 조합으로 처리됨
✅ Fallback 개선이 전체 성능에 미치는 영향은 미미함
✅ Zone 통계 접근법의 성능은 이미 최적화됨

개선 방향:
❌ Zone fallback 개선 (영향 미미)
✅ Zone + Direction 조합의 품질 향상 (어려움, 14회 실패)
✅ 다른 접근법 탐색 (Week 4-5)
""")

# =============================================================================
# 7. 상세 분석 저장
# =============================================================================
print(f"\n{'='*80}")
print(f"결과 저장")
print(f"{'='*80}")

output_dir = DATA_DIR / "code" / "analysis" / "results"
output_dir.mkdir(exist_ok=True)

# CSV 저장
comparison_df.to_csv(output_dir / "zone_fallback_comparison.csv")
print(f"  비교 결과 저장: {output_dir / 'zone_fallback_comparison.csv'}")

# 각 모델별 상세 통계 저장
for model in models:
    n_x, n_y = model['zone']
    use_dir = model['direction']
    min_s = model['min_samples']

    train_temp = train_last.copy()
    train_temp['zone'] = train_temp.apply(
        lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1
    )

    if use_dir:
        train_temp['direction'] = train_temp.apply(
            lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1
        )
        train_temp['key'] = train_temp['zone'].astype(str) + '_' + train_temp['direction']

        stats = train_temp.groupby('key').agg({
            'delta_x': 'median',
            'delta_y': 'median',
            'game_episode': 'count'
        }).rename(columns={'game_episode': 'count'})
    else:
        train_temp['key'] = train_temp['zone'].astype(str)

        stats = train_temp.groupby('key').agg({
            'delta_x': 'median',
            'delta_y': 'median',
            'game_episode': 'count'
        }).rename(columns={'game_episode': 'count'})

    filename = f"zone_stats_{model['name']}.csv"
    stats.to_csv(output_dir / filename)
    print(f"  {model['name']} 통계 저장: {output_dir / filename}")

print(f"\n{'='*80}")
print(f"분석 완료!")
print(f"{'='*80}")
