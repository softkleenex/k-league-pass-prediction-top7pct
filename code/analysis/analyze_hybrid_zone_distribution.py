"""
Hybrid Zone 데이터 분석

목표:
- 필드 위치별 패스 분포 분석
- 최적 필드 분할 경계 탐색
- Zone 크기별 샘플 커버리지 평가

2025-12-09 Hybrid Zone 구현 Step 1
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(".")

print("=" * 80)
print("Hybrid Zone 데이터 분석")
print("=" * 80)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")

# 피처 준비
def prepare_features(df):
    df = df.copy()
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
    return df

train_df = prepare_features(train_df)
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

print(f"  Train samples: {len(train_last):,}")

# =============================================================================
# 2. 필드 5m 구간별 분포 분석
# =============================================================================
print("\n[2] 필드 5m 구간별 분포 분석...")

bins = np.arange(0, 110, 5)
train_last['x_bin'] = pd.cut(train_last['start_x'], bins=bins, labels=False)

bin_stats = train_last.groupby('x_bin').agg({
    'game_episode': 'count',
    'delta_x': ['mean', 'std'],
    'delta_y': ['mean', 'std'],
    'start_x': 'mean'
}).round(2)

bin_stats.columns = ['count', 'mean_dx', 'std_dx', 'mean_dy', 'std_dy', 'avg_x']
bin_stats['pct'] = (bin_stats['count'] / len(train_last) * 100).round(2)

print("\n필드 5m 구간별 통계:")
print(f"{'구간':>8} {'샘플':>6} {'비율':>6} {'평균Δx':>8} {'σΔx':>7} {'평균Δy':>8} {'σΔy':>7}")
print("-" * 65)
for idx, row in bin_stats.iterrows():
    if pd.notna(idx):
        bin_start = int(idx * 5)
        bin_end = int((idx + 1) * 5)
        print(f"{bin_start:3d}-{bin_end:3d}m {row['count']:6.0f} {row['pct']:5.1f}% "
              f"{row['mean_dx']:8.2f} {row['std_dx']:7.2f} "
              f"{row['mean_dy']:8.2f} {row['std_dy']:7.2f}")

# =============================================================================
# 3. 필드 3분할 분석
# =============================================================================
print("\n[3] 필드 3분할 분석...")

def analyze_field_regions(df, boundaries):
    """필드를 3개 영역으로 분할하여 분석"""
    x1, x2 = boundaries

    defense = df[df['start_x'] < x1]
    midfield = df[(df['start_x'] >= x1) & (df['start_x'] < x2)]
    attack = df[df['start_x'] >= x2]

    regions = {
        f'수비 (0-{x1}m)': defense,
        f'미드 ({x1}-{x2}m)': midfield,
        f'공격 ({x2}-105m)': attack
    }

    print(f"\n경계: ({x1}m, {x2}m)")
    print(f"{'영역':>15} {'샘플':>7} {'비율':>6} {'평균Δx':>8} {'σΔx':>7} {'평균Δy':>8} {'σΔy':>7}")
    print("-" * 70)

    for name, region_df in regions.items():
        count = len(region_df)
        pct = count / len(df) * 100
        mean_dx = region_df['delta_x'].mean()
        std_dx = region_df['delta_x'].std()
        mean_dy = region_df['delta_y'].mean()
        std_dy = region_df['delta_y'].std()

        print(f"{name:>15} {count:7,} {pct:5.1f}% "
              f"{mean_dx:8.2f} {std_dx:7.2f} "
              f"{mean_dy:8.2f} {std_dy:7.2f}")

    return regions

# 여러 경계 시도
for boundaries in [(30, 70), (33, 68), (35, 70), (37, 72), (40, 75)]:
    regions = analyze_field_regions(train_last, boundaries)

# =============================================================================
# 4. Zone 크기별 샘플 커버리지
# =============================================================================
print("\n[4] Zone 크기별 샘플 커버리지...")

def evaluate_zone_coverage(df, boundaries, zone_configs, min_samples=20):
    """Zone 크기별 샘플 커버리지 평가"""
    x1, x2 = boundaries

    # 영역별 데이터 분할
    defense = df[df['start_x'] < x1]
    midfield = df[(df['start_x'] >= x1) & (df['start_x'] < x2)]
    attack = df[df['start_x'] >= x2]

    regions_data = {
        'defense': defense,
        'midfield': midfield,
        'attack': attack
    }

    print(f"\n경계: ({x1}m, {x2}m)")
    print(f"Zone 설정: {zone_configs}")
    print(f"\n{'영역':>10} {'Zone':>6} {'총샘플':>8} {'평균/Zone':>10} {'커버리지':>10}")
    print("-" * 55)

    for region_name, (n_x, n_y) in zone_configs.items():
        region_df = regions_data[region_name]
        total_samples = len(region_df)
        n_zones = n_x * n_y
        avg_per_zone = total_samples / n_zones

        # Zone별 샘플 수 계산
        zone_counts = []
        for i in range(n_x):
            for j in range(n_y):
                # 해당 Zone의 샘플 추출 (영역 내에서만)
                if region_name == 'defense':
                    x_min, x_max = 0, x1
                elif region_name == 'midfield':
                    x_min, x_max = x1, x2
                else:
                    x_min, x_max = x2, 105

                zone_width_x = (x_max - x_min) / n_x
                zone_width_y = 68 / n_y

                zone_x_min = x_min + i * zone_width_x
                zone_x_max = x_min + (i + 1) * zone_width_x
                zone_y_min = j * zone_width_y
                zone_y_max = (j + 1) * zone_width_y

                zone_df = region_df[
                    (region_df['start_x'] >= zone_x_min) &
                    (region_df['start_x'] < zone_x_max) &
                    (region_df['start_y'] >= zone_y_min) &
                    (region_df['start_y'] < zone_y_max)
                ]
                zone_counts.append(len(zone_df))

        # 커버리지 계산 (min_samples 이상인 Zone 비율)
        coverage = np.sum(np.array(zone_counts) >= min_samples) / n_zones * 100

        print(f"{region_name:>10} {n_x}x{n_y:2d} {total_samples:8,} "
              f"{avg_per_zone:10.1f} {coverage:9.1f}%")

# 테스트할 설정들
test_configs = [
    # 설정 1: 보수적 (큰 Zone)
    {'defense': (5, 5), 'midfield': (6, 6), 'attack': (7, 7)},
    # 설정 2: 균형
    {'defense': (6, 6), 'midfield': (6, 6), 'attack': (7, 7)},
    # 설정 3: 공격적 (작은 Zone)
    {'defense': (5, 5), 'midfield': (5, 5), 'attack': (6, 6)},
]

for config in test_configs:
    for boundaries in [(35, 70), (37, 72)]:
        evaluate_zone_coverage(train_last, boundaries, config, min_samples=20)

# =============================================================================
# 5. 최적 설정 추천
# =============================================================================
print("\n" + "=" * 80)
print("최적 설정 추천")
print("=" * 80)

print("\n[근거]")
print("1. 필드 3분할 균형: (35m, 70m) 또는 (37m, 72m)")
print("   - 1/3씩 균등 분할")
print("   - 각 영역의 샘플 비율 유사")

print("\n2. Zone 크기:")
print("   - 수비 (0-35m):     5x5 (샘플 많음 → 세밀)")
print("   - 미드필드 (35-70m): 6x6 (균형)")
print("   - 공격 (70-105m):    7x7 (샘플 적음 → 안정)")

print("\n3. min_samples: 20-25")
print("   - safe_fold13 검증값: 25")
print("   - Hybrid Zone은 더 세밀하므로 20도 고려")

print("\n[권장 설정]")
print("  BOUNDARIES = (35, 70)")
print("  ZONE_CONFIG = {")
print("      'defense': (5, 5),")
print("      'midfield': (6, 6),")
print("      'attack': (7, 7)")
print("  }")
print("  MIN_SAMPLES = 22")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
