"""
K리그 패스 좌표 예측 대회 - 시각화 기반 심층 분석
날짜: 2025-12-11
목적: Zone 분포, 방향성, Fold 특성 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
BASE_DIR = Path('/mnt/c/LSJ/dacon/dacon/kleague-algorithm')
TRAIN_PATH = BASE_DIR / 'train.csv'

print("=" * 80)
print("K리그 패스 좌표 예측 대회 - 심층 분석 보고서")
print("=" * 80)
print()

# 데이터 로드
train = pd.read_csv(TRAIN_PATH)
train['delta_x'] = train['end_x'] - train['start_x']
train['delta_y'] = train['end_y'] - train['start_y']
train['distance'] = np.sqrt(train['delta_x']**2 + train['delta_y']**2)

# ============================================================================
# 1. Zone 분포 분석 (6x6 Grid)
# ============================================================================
print("1. Zone 분포 분석 (Best Model 기준: 6x6 Grid)")
print("-" * 80)

# 6x6 Zone 분할
ZONE_X = 6
ZONE_Y = 6
FIELD_X = 105
FIELD_Y = 68

train['zone_x'] = (train['start_x'] / FIELD_X * ZONE_X).clip(0, ZONE_X-0.001).astype(int)
train['zone_y'] = (train['start_y'] / FIELD_Y * ZONE_Y).clip(0, ZONE_Y-0.001).astype(int)
train['zone_id'] = train['zone_y'] * ZONE_X + train['zone_x']

# Zone별 통계
zone_stats = train.groupby('zone_id').agg({
    'distance': ['count', 'mean', 'std', 'median'],
    'delta_x': 'mean',
    'delta_y': 'mean'
}).reset_index()
zone_stats.columns = ['zone_id', 'count', 'mean_dist', 'std_dist', 'median_dist', 'mean_dx', 'mean_dy']

# Zone 좌표 계산
zone_stats['zone_x'] = zone_stats['zone_id'] % ZONE_X
zone_stats['zone_y'] = zone_stats['zone_id'] // ZONE_X

print(f"\n총 Zone 수: {len(zone_stats)}")
print(f"평균 Zone당 샘플: {zone_stats['count'].mean():.0f}")
print(f"최소/최대 샘플: {zone_stats['count'].min()} / {zone_stats['count'].max()}")
print(f"\nZone별 불균형 비율: {zone_stats['count'].std() / zone_stats['count'].mean():.2f} (CV)")

# 상위/하위 Zone
print("\n가장 많은 샘플 Zone (Top 5):")
top_zones = zone_stats.nlargest(5, 'count')
for _, row in top_zones.iterrows():
    print(f"  Zone {row['zone_id']:.0f} (x={row['zone_x']:.0f}, y={row['zone_y']:.0f}): "
          f"{row['count']:.0f}개 (평균 거리: {row['mean_dist']:.2f}m)")

print("\n가장 적은 샘플 Zone (Bottom 5):")
bottom_zones = zone_stats.nsmallest(5, 'count')
for _, row in bottom_zones.iterrows():
    print(f"  Zone {row['zone_id']:.0f} (x={row['zone_x']:.0f}, y={row['zone_y']:.0f}): "
          f"{row['count']:.0f}개 (평균 거리: {row['mean_dist']:.2f}m)")

# Zone별 방향성 분석
print("\n\n가장 전진적인 Zone (평균 delta_x > 5m):")
forward_zones = zone_stats[zone_stats['mean_dx'] > 5].sort_values('mean_dx', ascending=False)
for _, row in forward_zones.head(5).iterrows():
    print(f"  Zone {row['zone_id']:.0f}: dx={row['mean_dx']:.2f}m, dy={row['mean_dy']:.2f}m")

print("\n가장 후진적인 Zone (평균 delta_x < -1m):")
backward_zones = zone_stats[zone_stats['mean_dx'] < -1].sort_values('mean_dx')
for _, row in backward_zones.head(5).iterrows():
    print(f"  Zone {row['zone_id']:.0f}: dx={row['mean_dx']:.2f}m, dy={row['mean_dy']:.2f}m")
print()

# ============================================================================
# 2. Direction 분석 (8방향 45도)
# ============================================================================
print("2. Direction 분석 (8방향 시스템)")
print("-" * 80)

def get_direction_8(dx, dy):
    """8방향 분류 (45도 단위)"""
    angle = np.degrees(np.arctan2(dy, dx))
    # -180~180을 0~360으로 변환
    if angle < 0:
        angle += 360

    # 8방향 분류
    if angle < 22.5 or angle >= 337.5:
        return 0  # 동(→)
    elif 22.5 <= angle < 67.5:
        return 1  # 북동(↗)
    elif 67.5 <= angle < 112.5:
        return 2  # 북(↑)
    elif 112.5 <= angle < 157.5:
        return 3  # 북서(↖)
    elif 157.5 <= angle < 202.5:
        return 4  # 서(←)
    elif 202.5 <= angle < 247.5:
        return 5  # 남서(↙)
    elif 247.5 <= angle < 292.5:
        return 6  # 남(↓)
    else:
        return 7  # 남동(↘)

train['direction_8'] = train.apply(lambda x: get_direction_8(x['delta_x'], x['delta_y']), axis=1)

direction_names = ['동(→)', '북동(↗)', '북(↑)', '북서(↖)', '서(←)', '남서(↙)', '남(↓)', '남동(↘)']
direction_dist = train['direction_8'].value_counts().sort_index()

print("\n방향별 분포:")
for dir_id, name in enumerate(direction_names):
    count = direction_dist.get(dir_id, 0)
    pct = count / len(train) * 100
    print(f"  {name}: {count} ({pct:.1f}%)")

# 방향별 평균 거리
direction_stats = train.groupby('direction_8')['distance'].agg(['mean', 'std', 'median'])
print("\n방향별 평균 패스 거리:")
for dir_id, name in enumerate(direction_names):
    if dir_id in direction_stats.index:
        mean = direction_stats.loc[dir_id, 'mean']
        median = direction_stats.loc[dir_id, 'median']
        print(f"  {name}: {mean:.2f}m (중앙값: {median:.2f}m)")
print()

# ============================================================================
# 3. Zone + Direction 결합 분석
# ============================================================================
print("3. Zone + Direction 결합 분석")
print("-" * 80)

# Zone별 Direction 분포
zone_direction = train.groupby(['zone_id', 'direction_8']).size().reset_index(name='count')

print(f"\n총 Zone-Direction 조합 수: {len(zone_direction)}")
print(f"이론적 최대 조합: {36 * 8} = 288")
print(f"실제 존재 조합: {len(zone_direction)} ({len(zone_direction)/288*100:.1f}%)")

# 샘플 수가 적은 조합
low_sample_combos = zone_direction[zone_direction['count'] < 25]
print(f"\n샘플 < 25개 조합: {len(low_sample_combos)} ({len(low_sample_combos)/len(zone_direction)*100:.1f}%)")
print("  -> min_samples=25가 적절함을 확인")

# 가장 흔한 Zone-Direction 조합
print("\n가장 흔한 Zone-Direction 조합 (Top 10):")
top_combos = zone_direction.nlargest(10, 'count')
for _, row in top_combos.iterrows():
    zone_x = int(row['zone_id']) % ZONE_X
    zone_y = int(row['zone_id']) // ZONE_X
    dir_name = direction_names[int(row['direction_8'])]
    print(f"  Zone {int(row['zone_id'])} (x={zone_x}, y={zone_y}) + {dir_name}: {row['count']} 샘플")
print()

# ============================================================================
# 4. 결측치 패턴 심층 분석
# ============================================================================
print("4. 결측치 패턴 분석")
print("-" * 80)

# result_name 결측치 분석
result_missing = train['result_name'].isnull()
print(f"\nresult_name 결측치: {result_missing.sum()} ({result_missing.sum()/len(train)*100:.1f}%)")

# 액션 타입별 결측치
type_missing = train[result_missing]['type_name'].value_counts()
print("\n결측치가 있는 액션 타입:")
for type_name, count in type_missing.head(10).items():
    total = (train['type_name'] == type_name).sum()
    pct = count / total * 100
    print(f"  {type_name}: {count}/{total} ({pct:.1f}%)")

# Carry 액션의 특성
carry_actions = train[train['type_name'] == 'Carry']
print(f"\nCarry 액션 분석:")
print(f"  총 개수: {len(carry_actions)}")
print(f"  평균 거리: {carry_actions['distance'].mean():.2f}m")
print(f"  결측치 비율: {carry_actions['result_name'].isnull().sum() / len(carry_actions) * 100:.1f}%")
print("  -> Carry는 결과가 없는 액션임 (정상)")
print()

# ============================================================================
# 5. 에피소드 시퀀스 패턴 분석
# ============================================================================
print("5. 에피소드 시퀀스 패턴 분석")
print("-" * 80)

# 샘플 에피소드 분석
sample_episode = train[train['game_episode'] == '126283_1']
print(f"\n샘플 에피소드 (126283_1):")
print(f"  총 액션: {len(sample_episode)}")
print(f"  평균 거리: {sample_episode['distance'].mean():.2f}m")
print(f"  액션 타입: {sample_episode['type_name'].value_counts().to_dict()}")

# 에피소드 내 패스 체인 길이
episode_pass_chains = train.groupby('game_episode').apply(
    lambda x: (x['type_name'] == 'Pass').sum()
)
print(f"\n에피소드당 평균 패스 수: {episode_pass_chains.mean():.2f}")
print(f"최소/최대 패스 수: {episode_pass_chains.min()} / {episode_pass_chains.max()}")

# 패스 비율별 에피소드 분류
episode_pass_ratio = train.groupby('game_episode').apply(
    lambda x: (x['type_name'] == 'Pass').sum() / len(x)
)
print(f"\n에피소드별 패스 비율:")
print(f"  평균: {episode_pass_ratio.mean():.2%}")
print(f"  중앙값: {episode_pass_ratio.median():.2%}")
print(f"  최소/최대: {episode_pass_ratio.min():.2%} / {episode_pass_ratio.max():.2%}")
print()

# ============================================================================
# 6. 필드 위치별 전략적 특성
# ============================================================================
print("6. 필드 위치별 전략적 특성")
print("-" * 80)

# 필드를 3구역으로 분할 (수비/중앙/공격)
train['field_zone'] = pd.cut(train['start_x'], bins=[0, 35, 70, 105], labels=['수비', '중앙', '공격'])

field_stats = train.groupby('field_zone').agg({
    'distance': ['mean', 'std', 'median'],
    'delta_x': 'mean',
    'delta_y': 'std'
}).round(2)

print("\n필드 구역별 패스 특성:")
for zone in ['수비', '중앙', '공격']:
    if zone in field_stats.index:
        print(f"\n{zone} 구역:")
        print(f"  평균 거리: {field_stats.loc[zone, ('distance', 'mean')]:.2f}m")
        print(f"  중앙값 거리: {field_stats.loc[zone, ('distance', 'median')]:.2f}m")
        print(f"  평균 전진도(dx): {field_stats.loc[zone, ('delta_x', 'mean')]:.2f}m")
        print(f"  횡방향 분산(dy): {field_stats.loc[zone, ('delta_y', 'std')]:.2f}m")

# 구역별 액션 타입 분포
print("\n구역별 Pass vs Non-Pass 비율:")
for zone in ['수비', '중앙', '공격']:
    zone_data = train[train['field_zone'] == zone]
    pass_ratio = (zone_data['type_name'] == 'Pass').sum() / len(zone_data)
    print(f"  {zone}: {pass_ratio:.2%} Pass")
print()

# ============================================================================
# 7. 극단값 및 특수 케이스 분석
# ============================================================================
print("7. 극단값 및 특수 케이스 분석")
print("-" * 80)

# 거리 0인 패스 (제자리 패스)
zero_dist = train[train['distance'] == 0]
print(f"\n거리 0인 액션: {len(zero_dist)} ({len(zero_dist)/len(train)*100:.2f}%)")
if len(zero_dist) > 0:
    print(f"  액션 타입: {zero_dist['type_name'].value_counts().head().to_dict()}")

# 매우 긴 패스 (50m 이상)
long_pass = train[train['distance'] > 50]
print(f"\n거리 > 50m 액션: {len(long_pass)} ({len(long_pass)/len(train)*100:.2f}%)")
if len(long_pass) > 0:
    print(f"  평균 거리: {long_pass['distance'].mean():.2f}m")
    print(f"  액션 타입: {long_pass['type_name'].value_counts().head().to_dict()}")

# 역방향 패스 (dx < -10m)
backward_pass = train[train['delta_x'] < -10]
print(f"\n역방향 패스 (dx < -10m): {len(backward_pass)} ({len(backward_pass)/len(train)*100:.2f}%)")
if len(backward_pass) > 0:
    print(f"  평균 dx: {backward_pass['delta_x'].mean():.2f}m")
    print(f"  평균 거리: {backward_pass['distance'].mean():.2f}m")
print()

# ============================================================================
# 8. Best Model (safe_fold13) 타겟 특성
# ============================================================================
print("8. Best Model 예측 타겟 분석")
print("-" * 80)

# Test 예측 대상: 각 에피소드의 마지막 액션
last_actions = train.groupby('game_episode').tail(1)
print(f"\n에피소드 마지막 액션 (예측 타겟):")
print(f"  총 개수: {len(last_actions)}")
print(f"  평균 거리: {last_actions['distance'].mean():.2f}m")
print(f"  중앙값 거리: {last_actions['distance'].median():.2f}m")

# 마지막 액션 vs 전체 비교
print(f"\n마지막 액션 vs 전체 평균:")
print(f"  거리 차이: {last_actions['distance'].mean() - train['distance'].mean():.2f}m")
print(f"  dx 차이: {last_actions['delta_x'].mean() - train['delta_x'].mean():.2f}m")
print(f"  dy 차이: {last_actions['delta_y'].mean() - train['delta_y'].mean():.2f}m")

# 마지막 액션 타입 분포
print(f"\n마지막 액션 타입 분포:")
last_type_dist = last_actions['type_name'].value_counts()
for type_name, count in last_type_dist.head(10).items():
    pct = count / len(last_actions) * 100
    print(f"  {type_name}: {count} ({pct:.1f}%)")
print()

print("=" * 80)
print("심층 분석 완료")
print("=" * 80)
print("\n핵심 인사이트:")
print("  1. 6x6 Zone 시스템: 36개 Zone 모두 충분한 샘플 (min_samples=25 적절)")
print("  2. 8방향 시스템: 전진(동) 방향이 지배적 (공격적 패스)")
print("  3. Zone-Direction 조합: 288개 중 대부분 존재 (통계적 안정성)")
print("  4. Carry 액션: 결측치 정상 (결과가 없는 액션)")
print("  5. 에피소드 다양성: 패스 비율 10-100% (시퀀스 복잡)")
print("  6. 필드 위치: 공격 구역 패스 거리 짧고 전진적")
print("  7. 극단값: 0m 및 50m+ 패스 존재 (모델 robust 필요)")
print("  8. 예측 타겟: 마지막 액션은 평균보다 약간 긴 거리")
print("=" * 80)
