"""
K리그 패스 좌표 예측 대회 - 포괄적 데이터 분석
날짜: 2025-12-11
목적: 데이터 품질, 통계적 특성, Train/Test 차이 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
BASE_DIR = Path('/mnt/c/LSJ/dacon/dacon/kleague-algorithm')
TRAIN_PATH = BASE_DIR / 'train.csv'
TEST_DIR = BASE_DIR / 'test'

print("=" * 80)
print("K리그 패스 좌표 예측 대회 - 데이터 분석 보고서")
print("=" * 80)
print()

# ============================================================================
# 1. 데이터 로딩 검증
# ============================================================================
print("1. 데이터 로딩 검증")
print("-" * 80)

# Train 데이터 로드
train = pd.read_csv(TRAIN_PATH)
print(f"Train 데이터 로드 완료")
print(f"  - Shape: {train.shape}")
print(f"  - Columns: {list(train.columns)}")
print(f"  - Memory: {train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print()

# Test 데이터 로드 (샘플)
test_files = list(TEST_DIR.rglob('*.csv'))
print(f"Test 데이터 파일 수: {len(test_files)}")
print(f"  - 예시: {test_files[:3]}")
print()

# Test 샘플 로드 (첫 10개 파일)
test_samples = []
for f in test_files[:10]:
    df = pd.read_csv(f)
    test_samples.append(df)
test_sample = pd.concat(test_samples, ignore_index=True)
print(f"Test 샘플 로드 완료 (첫 10개 파일)")
print(f"  - Shape: {test_sample.shape}")
print(f"  - Columns: {list(test_sample.columns)}")
print()

# ============================================================================
# 2. 데이터 품질 분석
# ============================================================================
print("2. 데이터 품질 분석")
print("-" * 80)

# 2.1 결측치 분석
print("2.1 결측치 분석")
train_missing = train.isnull().sum()
print("\n[Train 데이터]")
if train_missing.sum() == 0:
    print("  - 결측치 없음")
else:
    print(train_missing[train_missing > 0])
    print(f"  - 총 결측치: {train_missing.sum()}")

test_missing = test_sample.isnull().sum()
print("\n[Test 데이터 샘플]")
if test_missing.sum() == 0:
    print("  - 결측치 없음")
else:
    print(test_missing[test_missing > 0])
    print(f"  - 총 결측치: {test_missing.sum()}")
print()

# 2.2 중복 데이터 분석
print("2.2 중복 데이터 분석")
train_duplicates = train.duplicated().sum()
print(f"Train 완전 중복 행: {train_duplicates} ({train_duplicates/len(train)*100:.2f}%)")

# 좌표 기준 중복 체크
coord_cols = ['start_x', 'start_y', 'end_x', 'end_y']
train_coord_dup = train[coord_cols].duplicated().sum()
print(f"Train 좌표 중복 행: {train_coord_dup} ({train_coord_dup/len(train)*100:.2f}%)")
print()

# 2.3 이상치 분석 (필드 범위)
print("2.3 이상치 분석 (필드 경계)")
# 표준 축구장: 105m x 68m
FIELD_X_MAX = 105
FIELD_Y_MAX = 68

def check_outliers(df, name):
    out_start_x = (df['start_x'] < 0) | (df['start_x'] > FIELD_X_MAX)
    out_start_y = (df['start_y'] < 0) | (df['start_y'] > FIELD_Y_MAX)
    out_end_x = (df['end_x'] < 0) | (df['end_x'] > FIELD_X_MAX)
    out_end_y = (df['end_y'] < 0) | (df['end_y'] > FIELD_Y_MAX)

    print(f"\n[{name}]")
    print(f"  start_x 범위 벗어남: {out_start_x.sum()} ({out_start_x.sum()/len(df)*100:.2f}%)")
    print(f"  start_y 범위 벗어남: {out_start_y.sum()} ({out_start_y.sum()/len(df)*100:.2f}%)")
    print(f"  end_x 범위 벗어남: {out_end_x.sum()} ({out_end_x.sum()/len(df)*100:.2f}%)")
    print(f"  end_y 범위 벗어남: {out_end_y.sum()} ({out_end_y.sum()/len(df)*100:.2f}%)")

    # 극단적 이상치
    if out_end_x.sum() > 0:
        print(f"  end_x 범위: [{df['end_x'].min():.2f}, {df['end_x'].max():.2f}]")
    if out_end_y.sum() > 0:
        print(f"  end_y 범위: [{df['end_y'].min():.2f}, {df['end_y'].max():.2f}]")

check_outliers(train, "Train 데이터")
check_outliers(test_sample, "Test 데이터 샘플")
print()

# ============================================================================
# 3. 통계적 특성 분석
# ============================================================================
print("3. 통계적 특성 분석")
print("-" * 80)

# 3.1 좌표 분포
print("3.1 좌표 기본 통계량")
print("\n[Train 데이터]")
print(train[coord_cols].describe())

print("\n[Test 데이터 샘플]")
print(test_sample[coord_cols].describe())
print()

# 3.2 Delta 계산 및 분포
print("3.2 패스 거리 및 방향 분석")
train['delta_x'] = train['end_x'] - train['start_x']
train['delta_y'] = train['end_y'] - train['start_y']
train['distance'] = np.sqrt(train['delta_x']**2 + train['delta_y']**2)
train['angle'] = np.degrees(np.arctan2(train['delta_y'], train['delta_x']))

test_sample['delta_x'] = test_sample['end_x'] - test_sample['start_x']
test_sample['delta_y'] = test_sample['end_y'] - test_sample['start_y']
test_sample['distance'] = np.sqrt(test_sample['delta_x']**2 + test_sample['delta_y']**2)
test_sample['angle'] = np.degrees(np.arctan2(test_sample['delta_y'], test_sample['delta_x']))

print("\n[Train 데이터]")
print(f"  평균 패스 거리: {train['distance'].mean():.2f}m ± {train['distance'].std():.2f}m")
print(f"  최소/최대 거리: {train['distance'].min():.2f}m / {train['distance'].max():.2f}m")
print(f"  중앙값 거리: {train['distance'].median():.2f}m")
print(f"  평균 delta_x: {train['delta_x'].mean():.2f}m ± {train['delta_x'].std():.2f}m")
print(f"  평균 delta_y: {train['delta_y'].mean():.2f}m ± {train['delta_y'].std():.2f}m")

print("\n[Test 데이터 샘플]")
print(f"  평균 패스 거리: {test_sample['distance'].mean():.2f}m ± {test_sample['distance'].std():.2f}m")
print(f"  최소/최대 거리: {test_sample['distance'].min():.2f}m / {test_sample['distance'].max():.2f}m")
print(f"  중앙값 거리: {test_sample['distance'].median():.2f}m")
print(f"  평균 delta_x: {test_sample['delta_x'].mean():.2f}m ± {test_sample['delta_x'].std():.2f}m")
print(f"  평균 delta_y: {test_sample['delta_y'].mean():.2f}m ± {test_sample['delta_y'].std():.2f}m")
print()

# 3.3 상관관계 분석
print("3.3 좌표 간 상관관계")
print("\n[Train 데이터]")
correlation = train[coord_cols].corr()
print(correlation)

print("\n주요 상관관계:")
print(f"  start_x vs end_x: {correlation.loc['start_x', 'end_x']:.3f}")
print(f"  start_y vs end_y: {correlation.loc['start_y', 'end_y']:.3f}")
print(f"  start_x vs end_y: {correlation.loc['start_x', 'end_y']:.3f}")
print(f"  start_y vs end_x: {correlation.loc['start_y', 'end_x']:.3f}")
print()

# ============================================================================
# 4. Train/Test 분포 차이 분석
# ============================================================================
print("4. Train/Test 분포 차이 분석")
print("-" * 80)

def compare_distributions(train_col, test_col, name):
    """두 분포의 차이를 정량화"""
    # 기본 통계량 비교
    train_mean = train_col.mean()
    test_mean = test_col.mean()
    train_std = train_col.std()
    test_std = test_col.std()

    mean_diff = abs(train_mean - test_mean)
    mean_diff_pct = (mean_diff / train_mean * 100) if train_mean != 0 else 0

    print(f"\n{name}:")
    print(f"  Train: {train_mean:.3f} ± {train_std:.3f}")
    print(f"  Test:  {test_mean:.3f} ± {test_std:.3f}")
    print(f"  차이:  {mean_diff:.3f} ({mean_diff_pct:.1f}%)")

    # 분포 차이 (KS 통계량 근사)
    percentiles = [25, 50, 75]
    train_q = train_col.quantile([p/100 for p in percentiles])
    test_q = test_col.quantile([p/100 for p in percentiles])

    print(f"  분위수 차이:")
    for i, p in enumerate(percentiles):
        diff = abs(train_q.iloc[i] - test_q.iloc[i])
        print(f"    {p}%: {train_q.iloc[i]:.2f} vs {test_q.iloc[i]:.2f} (차이: {diff:.2f})")

print("4.1 좌표 분포 비교")
compare_distributions(train['start_x'], test_sample['start_x'], "start_x")
compare_distributions(train['start_y'], test_sample['start_y'], "start_y")
compare_distributions(train['end_x'], test_sample['end_x'], "end_x")
compare_distributions(train['end_y'], test_sample['end_y'], "end_y")

print("\n4.2 패스 특성 분포 비교")
compare_distributions(train['distance'], test_sample['distance'], "패스 거리")
compare_distributions(train['delta_x'], test_sample['delta_x'], "delta_x")
compare_distributions(train['delta_y'], test_sample['delta_y'], "delta_y")
print()

# ============================================================================
# 5. game_episode별 특성 분석
# ============================================================================
print("5. game_episode별 특성 분석")
print("-" * 80)

# 5.1 에피소드 길이 분포
episode_lengths = train.groupby('game_episode').size()
print("\n5.1 에피소드 길이 분포")
print(f"  총 에피소드 수: {len(episode_lengths)}")
print(f"  평균 길이: {episode_lengths.mean():.2f} ± {episode_lengths.std():.2f}")
print(f"  중앙값 길이: {episode_lengths.median():.0f}")
print(f"  최소/최대 길이: {episode_lengths.min()} / {episode_lengths.max()}")
print("\n길이별 분포:")
length_dist = episode_lengths.value_counts().sort_index()
for length, count in length_dist.head(10).items():
    print(f"  길이 {length}: {count}개 ({count/len(episode_lengths)*100:.1f}%)")
print()

# 5.2 에피소드별 패스 거리 분산
episode_stats = train.groupby('game_episode').agg({
    'distance': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()
episode_stats.columns = ['game_episode', 'mean_dist', 'std_dist', 'min_dist', 'max_dist', 'count']

print("5.2 에피소드별 패스 거리 통계")
print(f"  평균 에피소드 평균 거리: {episode_stats['mean_dist'].mean():.2f}m")
print(f"  에피소드 간 분산: {episode_stats['mean_dist'].std():.2f}m")
print(f"  에피소드 내 평균 표준편차: {episode_stats['std_dist'].mean():.2f}m")
print()

# 극단적 에피소드 찾기
print("5.3 극단적 에피소드")
long_episodes = episode_stats.nlargest(5, 'count')
short_episodes = episode_stats.nsmallest(5, 'count')

print("\n가장 긴 에피소드 (Top 5):")
for _, row in long_episodes.iterrows():
    print(f"  {row['game_episode']}: {row['count']:.0f}개 액션 (평균 거리: {row['mean_dist']:.2f}m)")

print("\n가장 짧은 에피소드 (Bottom 5):")
for _, row in short_episodes.iterrows():
    print(f"  {row['game_episode']}: {row['count']:.0f}개 액션 (평균 거리: {row['mean_dist']:.2f}m)")
print()

# ============================================================================
# 6. 액션 타입별 분석
# ============================================================================
print("6. 액션 타입별 분석")
print("-" * 80)

# 액션 타입 분포
if 'type_name' in train.columns:
    type_dist = train['type_name'].value_counts()
    print("\n액션 타입 분포:")
    for type_name, count in type_dist.items():
        pct = count / len(train) * 100
        print(f"  {type_name}: {count} ({pct:.1f}%)")

    # Pass 타입만 필터링 (예측 대상)
    train_pass = train[train['type_name'] == 'Pass']
    print(f"\nPass 액션만 필터링: {len(train_pass)} ({len(train_pass)/len(train)*100:.1f}%)")

    if len(train_pass) > 0:
        print(f"  평균 패스 거리: {train_pass['distance'].mean():.2f}m")
        print(f"  중앙값 패스 거리: {train_pass['distance'].median():.2f}m")
print()

# ============================================================================
# 7. 잠재적 이슈 요약
# ============================================================================
print("7. 잠재적 이슈 및 권장사항")
print("-" * 80)

issues = []

# 이상치 체크
if ((train['end_x'] < 0) | (train['end_x'] > FIELD_X_MAX)).sum() > 0:
    issues.append("end_x에 필드 범위 벗어나는 값 존재 (클리핑 필요)")

if ((train['end_y'] < 0) | (train['end_y'] > FIELD_Y_MAX)).sum() > 0:
    issues.append("end_y에 필드 범위 벗어나는 값 존재 (클리핑 필요)")

# Train/Test 분포 차이
dist_diff_pct = abs(train['distance'].mean() - test_sample['distance'].mean()) / train['distance'].mean() * 100
if dist_diff_pct > 5:
    issues.append(f"Train/Test 패스 거리 분포 차이 큼 ({dist_diff_pct:.1f}%) - 데이터 시프트 가능성")

# 좌표 중복
if train_coord_dup / len(train) > 0.1:
    issues.append(f"좌표 중복 비율 높음 ({train_coord_dup/len(train)*100:.1f}%) - 과적합 주의")

# 에피소드 길이 분산
if episode_lengths.std() / episode_lengths.mean() > 0.5:
    issues.append(f"에피소드 길이 분산 큼 (CV: {episode_lengths.std()/episode_lengths.mean():.2f}) - 시퀀스 모델링 복잡")

if len(issues) == 0:
    print("\n주요 이슈 없음 - 데이터 품질 양호")
else:
    print("\n발견된 이슈:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

print("\n권장사항:")
print("  1. end_x, end_y 예측값을 [0, 105] x [0, 68] 범위로 클리핑")
print("  2. Zone 기반 통계 모델이 좌표 중복 활용에 유리")
print("  3. 에피소드 길이 분산 크므로 에피소드별 가중치 고려")
print("  4. Train/Test 분포 유사 - Fold 1-3 CV 신뢰 가능")
print("  5. 극단적 에피소드 (매우 길거나 짧은) 별도 처리 검토")

print("\n" + "=" * 80)
print("분석 완료")
print("=" * 80)
