"""
Train vs Test 데이터 분포 차이 분석
목표: 과적합의 원인을 찾기 위해 Train과 Test 데이터의 차이를 정량화
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("K리그 패스 좌표 예측 - Train vs Test 분포 분석")
print("=" * 80)

# 1. 데이터 로드
print("\n[1] 데이터 로드 중...")
train = pd.read_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/train.csv')
match_info = pd.read_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/match_info.csv')
sample_submission = pd.read_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/sample_submission.csv')

print(f"  - Train: {len(train):,} rows, {train['game_episode'].nunique():,} episodes")
print(f"  - Test episodes: {len(sample_submission):,}")

# Test 데이터 읽기 (game_id별 폴더 구조)
test_dir = '/mnt/c/LSJ/dacon/dacon/kleague-algorithm/test/'
test_game_folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
print(f"  - Test game folders: {len(test_game_folders)}")

test_data_list = []
for game_folder in test_game_folders:
    game_path = os.path.join(test_dir, game_folder)
    episode_files = [f for f in os.listdir(game_path) if f.endswith('.csv')]
    for file in episode_files:
        df = pd.read_csv(os.path.join(game_path, file))
        test_data_list.append(df)

test = pd.concat(test_data_list, ignore_index=True)
print(f"  - Test total: {len(test):,} rows, {test['game_episode'].nunique():,} episodes")

# 2. Game ID 분석
print("\n" + "=" * 80)
print("[2] Game ID 분포 분석")
print("=" * 80)

train_games = set(train['game_id'].unique())
test_games = set(test['game_id'].unique())

print(f"\nTrain unique games: {len(train_games)}")
print(f"Test unique games: {len(test_games)}")
print(f"Overlap games: {len(train_games & test_games)}")
print(f"Test-only games: {len(test_games - train_games)}")

# Test game_id 분석
test_game_ids = sorted(test_games)
print(f"\nTest game_ids: {test_game_ids[:10]}..." if len(test_game_ids) > 10 else f"\nTest game_ids: {test_game_ids}")

# Match info에서 확인
match_train = match_info[match_info['game_id'].isin(train_games)]
match_test = match_info[match_info['game_id'].isin(test_games)]

print(f"\n[Match Info 분석]")
print(f"Train games in match_info: {len(match_train)}")
print(f"Test games in match_info: {len(match_test)}")

if len(match_test) > 0:
    print(f"\nTest 경기 날짜 범위:")
    print(f"  - 최소: {match_test['game_date'].min()}")
    print(f"  - 최대: {match_test['game_date'].max()}")

    print(f"\nTrain 경기 날짜 범위:")
    print(f"  - 최소: {match_train['game_date'].min()}")
    print(f"  - 최대: {match_train['game_date'].max()}")

    print("\n>>> 시간적 분리 (Temporal Split):")
    if match_test['game_date'].min() > match_train['game_date'].max():
        print("  ✓ Test는 Train 이후 경기 (미래 예측)")
    else:
        print("  ✓ Test와 Train이 시간적으로 겹침")

# 3. 시작 좌표 분포 비교
print("\n" + "=" * 80)
print("[3] 시작 좌표 (start_x, start_y) 분포 비교")
print("=" * 80)

# Pass만 필터링
train_pass = train[train['type_name'] == 'Pass'].copy()
test_pass = test[test['type_name'] == 'Pass'].copy()

print(f"\nTrain Pass: {len(train_pass):,} ({len(train_pass)/len(train)*100:.1f}%)")
print(f"Test Pass: {len(test_pass):,} ({len(test_pass)/len(test)*100:.1f}%)")

def compare_distribution(train_data, test_data, column, label):
    """분포 비교 함수"""
    train_vals = train_data[column].dropna()
    test_vals = test_data[column].dropna()

    # 기본 통계
    print(f"\n{label}:")
    print(f"  Train: mean={train_vals.mean():.2f}, std={train_vals.std():.2f}, "
          f"median={train_vals.median():.2f}")
    print(f"  Test:  mean={test_vals.mean():.2f}, std={test_vals.std():.2f}, "
          f"median={test_vals.median():.2f}")

    # 차이
    mean_diff = abs(test_vals.mean() - train_vals.mean())
    std_diff = abs(test_vals.std() - train_vals.std())
    print(f"  Diff:  mean={mean_diff:.2f}, std={std_diff:.2f}")

    # KS Test (분포 차이 검정)
    ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)
    print(f"  KS Test: statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")
    if ks_pval < 0.05:
        print(f"  >>> WARNING: 분포가 통계적으로 유의미하게 다름! (p<0.05)")
    else:
        print(f"  >>> OK: 분포 차이 없음")

    return ks_stat, ks_pval

# 마지막 패스만 (예측 대상)
train_last = train_pass.groupby('game_episode').tail(1)
test_last = test_pass.groupby('game_episode').tail(1)

print(f"\n[마지막 패스 (예측 대상)]")
print(f"Train last pass: {len(train_last):,}")
print(f"Test last pass: {len(test_last):,}")

compare_distribution(train_last, test_last, 'start_x', 'start_x')
compare_distribution(train_last, test_last, 'start_y', 'start_y')

# 4. 패스 이동량 분포 비교
print("\n" + "=" * 80)
print("[4] 패스 이동량 (delta_x, delta_y) 분포 비교")
print("=" * 80)

train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']
train_last['distance'] = np.sqrt(train_last['delta_x']**2 + train_last['delta_y']**2)

# Test는 end_x, end_y가 없으므로 전체 패스로 비교
train_pass['delta_x'] = train_pass['end_x'] - train_pass['start_x']
train_pass['delta_y'] = train_pass['end_y'] - train_pass['start_y']
train_pass['distance'] = np.sqrt(train_pass['delta_x']**2 + train_pass['delta_y']**2)

print("\n[전체 패스 기준]")
compare_distribution(train_pass, train_pass, 'delta_x', 'delta_x (Train only)')
compare_distribution(train_pass, train_pass, 'delta_y', 'delta_y (Train only)')
compare_distribution(train_pass, train_pass, 'distance', 'distance (Train only)')

# 5. 시퀀스 길이 분포 비교
print("\n" + "=" * 80)
print("[5] 에피소드 시퀀스 길이 분포 비교")
print("=" * 80)

train_seq_len = train.groupby('game_episode').size()
test_seq_len = test.groupby('game_episode').size()

print(f"\nTrain 시퀀스 길이:")
print(f"  mean={train_seq_len.mean():.1f}, std={train_seq_len.std():.1f}, "
      f"median={train_seq_len.median():.0f}")
print(f"  min={train_seq_len.min()}, max={train_seq_len.max()}")

print(f"\nTest 시퀀스 길이:")
print(f"  mean={test_seq_len.mean():.1f}, std={test_seq_len.std():.1f}, "
      f"median={test_seq_len.median():.0f}")
print(f"  min={test_seq_len.min()}, max={test_seq_len.max()}")

ks_stat, ks_pval = stats.ks_2samp(train_seq_len, test_seq_len)
print(f"\nKS Test: statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")
if ks_pval < 0.05:
    print(">>> WARNING: 시퀀스 길이 분포가 다름!")
else:
    print(">>> OK: 시퀀스 길이 분포 유사")

# 6. Action Type 분포 비교
print("\n" + "=" * 80)
print("[6] Action Type 분포 비교")
print("=" * 80)

train_action_dist = train['type_name'].value_counts(normalize=True).sort_index()
test_action_dist = test['type_name'].value_counts(normalize=True).sort_index()

print("\nAction Type 분포:")
print(f"{'Type':<20} {'Train %':>10} {'Test %':>10} {'Diff':>10}")
print("-" * 55)

all_types = sorted(set(train_action_dist.index) | set(test_action_dist.index))
for action_type in all_types:
    train_pct = train_action_dist.get(action_type, 0) * 100
    test_pct = test_action_dist.get(action_type, 0) * 100
    diff = test_pct - train_pct
    flag = "!" if abs(diff) > 5 else ""
    print(f"{action_type:<20} {train_pct:>9.1f}% {test_pct:>9.1f}% {diff:>9.1f}% {flag}")

# 7. Zone 분포 비교 (6x6 기준)
print("\n" + "=" * 80)
print("[7] Zone 분포 비교 (6x6 Grid)")
print("=" * 80)

def get_zone_6x6(x, y):
    """6x6 구역 계산"""
    x_zone = np.minimum(5, (x / (105 / 6)).astype(int))
    y_zone = np.minimum(5, (y / (68 / 6)).astype(int))
    return x_zone * 6 + y_zone

train_last['zone'] = get_zone_6x6(train_last['start_x'].values, train_last['start_y'].values)
test_last['zone'] = get_zone_6x6(test_last['start_x'].values, test_last['start_y'].values)

train_zone_dist = train_last['zone'].value_counts(normalize=True).sort_index()
test_zone_dist = test_last['zone'].value_counts(normalize=True).sort_index()

print("\n구역별 데이터 분포 (Top 10):")
print(f"{'Zone':>6} {'Train %':>10} {'Test %':>10} {'Diff':>10}")
print("-" * 40)

all_zones = sorted(set(train_zone_dist.index) | set(test_zone_dist.index))
zone_diffs = []
for zone in all_zones:
    train_pct = train_zone_dist.get(zone, 0) * 100
    test_pct = test_zone_dist.get(zone, 0) * 100
    diff = test_pct - train_pct
    zone_diffs.append((zone, train_pct, test_pct, abs(diff)))

# 차이가 큰 순으로 정렬
zone_diffs.sort(key=lambda x: x[3], reverse=True)
for zone, train_pct, test_pct, abs_diff in zone_diffs[:10]:
    diff = test_pct - train_pct
    flag = "!" if abs_diff > 2 else ""
    print(f"{zone:>6} {train_pct:>9.1f}% {test_pct:>9.1f}% {diff:>9.1f}% {flag}")

# Chi-square test
from scipy.stats import chi2_contingency
train_zone_counts = train_last['zone'].value_counts()
test_zone_counts = test_last['zone'].value_counts()

all_zones_set = sorted(set(train_zone_counts.index) | set(test_zone_counts.index))
train_counts = [train_zone_counts.get(z, 0) for z in all_zones_set]
test_counts = [test_zone_counts.get(z, 0) for z in all_zones_set]

chi2, pval, dof, expected = chi2_contingency([train_counts, test_counts])
print(f"\nChi-square Test: χ²={chi2:.2f}, p-value={pval:.6f}")
if pval < 0.05:
    print(">>> WARNING: Zone 분포가 통계적으로 다름!")
else:
    print(">>> OK: Zone 분포 유사")

# 8. 팀/선수 분포 비교
print("\n" + "=" * 80)
print("[8] 팀 분포 비교")
print("=" * 80)

train_teams = set(train['team_id'].unique())
test_teams = set(test['team_id'].unique())

print(f"\nTrain unique teams: {len(train_teams)}")
print(f"Test unique teams: {len(test_teams)}")
print(f"Overlap teams: {len(train_teams & test_teams)}")
print(f"Test-only teams: {len(test_teams - train_teams)}")

if len(test_teams - train_teams) > 0:
    print(f">>> WARNING: Test에만 있는 팀: {sorted(test_teams - train_teams)}")

# 9. 종합 요약
print("\n" + "=" * 80)
print("[9] 종합 분석 요약")
print("=" * 80)

print("\n과적합 원인 분석:")
print("-" * 80)

issues = []

# Game ID 중복 체크
if len(train_games & test_games) > 0:
    issues.append("✓ Train/Test에 동일 경기 포함 (Data Leakage 가능성)")
else:
    print("1. Game ID: Train과 Test 완전 분리 (Good)")

# 시간적 분리 체크
if len(match_test) > 0:
    if match_test['game_date'].min() > match_train['game_date'].max():
        issues.append("✓ Test는 미래 경기 (Temporal Shift 가능성)")
        print("2. Temporal: Test는 Train 이후 경기 (시간적 분포 변화 위험)")
    else:
        print("2. Temporal: Train과 Test 시간 중복")

# Zone 분포 체크
if pval < 0.05:
    issues.append(f"✓ Zone 분포 차이 (Chi-square p={pval:.6f})")
    print(f"3. Zone 분포: 통계적으로 유의미한 차이 (p={pval:.6f})")
else:
    print("3. Zone 분포: 유사 (Good)")

# 시퀀스 길이 체크
seq_ks_stat, seq_ks_pval = stats.ks_2samp(train_seq_len, test_seq_len)
if seq_ks_pval < 0.05:
    issues.append(f"✓ 시퀀스 길이 분포 차이 (KS p={seq_ks_pval:.6f})")
    print(f"4. 시퀀스 길이: 분포 차이 있음 (p={seq_ks_pval:.6f})")
else:
    print("4. 시퀀스 길이: 유사 (Good)")

print("\n" + "=" * 80)
print("핵심 발견사항:")
print("=" * 80)

if len(issues) == 0:
    print("✓ Train과 Test 분포가 유사 → 과적합 원인은 모델 자체")
else:
    print(f"\n발견된 문제: {len(issues)}개")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")

print("\n" + "=" * 80)
print("권장 조치:")
print("=" * 80)

print("""
1. 모델 정규화 강화:
   - LightGBM: min_child_samples 증가, max_depth 감소
   - 피처 수 축소: 현재 31개 → 15-20개로 감소
   - Dropout 추가 (딥러닝 모델)

2. 검증 전략 개선:
   - GroupKFold로 게임 단위 분리 (현재 사용 중)
   - TimeSeriesSplit 고려 (시간순 분리)
   - Stratified by Zone (구역별 균형)

3. 앙상블 전략:
   - Zone Baseline (안정적) + ML (성능) 혼합
   - 비중: Zone 60-70%, ML 30-40%
   - Bootstrap aggregating

4. 피처 선택:
   - 일반화 가능한 피처 우선
   - 경기별/팀별 특이한 피처 제거
   - angle_change, distance 등 기하학적 피처 중심
""")

print("\n분석 완료!")
print("=" * 80)
