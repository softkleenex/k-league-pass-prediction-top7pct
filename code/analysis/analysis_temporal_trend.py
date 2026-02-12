"""
시간에 따른 패스 패턴 변화 분석
목표: Temporal Shift가 과적합의 주요 원인인지 검증
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("시간에 따른 패스 패턴 변화 분석")
print("=" * 80)

# 데이터 로드
train = pd.read_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/train.csv')
match_info = pd.read_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/match_info.csv')

# 날짜 정보 병합
match_info['game_date'] = pd.to_datetime(match_info['game_date'])
train = train.merge(match_info[['game_id', 'game_date', 'game_day']], on='game_id', how='left')

# Pass만 필터링
train_pass = train[train['type_name'] == 'Pass'].copy()
train_last = train_pass.groupby('game_episode').tail(1).copy()

# 델타 계산
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']
train_last['distance'] = np.sqrt(train_last['delta_x']**2 + train_last['delta_y']**2)

# Zone 계산
def get_zone_6x6(x, y):
    x_zone = np.minimum(5, (x / (105 / 6)).astype(int))
    y_zone = np.minimum(5, (y / (68 / 6)).astype(int))
    return x_zone * 6 + y_zone

train_last['zone'] = get_zone_6x6(train_last['start_x'].values, train_last['start_y'].values)

print(f"\n총 {len(train_last):,}개 마지막 패스 분석")

# 1. 시간에 따른 기본 통계 추이
print("\n" + "=" * 80)
print("[1] 월별 패스 패턴 변화")
print("=" * 80)

train_last['month'] = train_last['game_date'].dt.to_period('M')

monthly_stats = train_last.groupby('month').agg({
    'delta_x': ['mean', 'std', 'median'],
    'delta_y': ['mean', 'std', 'median'],
    'distance': ['mean', 'std', 'median'],
    'start_x': ['mean', 'std'],
    'start_y': ['mean', 'std'],
    'game_episode': 'count'
}).round(2)

print("\n월별 통계:")
print(monthly_stats)

# 2. Train 초기 vs 후기 비교
print("\n" + "=" * 80)
print("[2] Train 초기 (3-5월) vs 후기 (8-10월) 비교")
print("=" * 80)

train_last['month_num'] = train_last['game_date'].dt.month

early_train = train_last[train_last['month_num'].isin([3, 4, 5])]
late_train = train_last[train_last['month_num'].isin([8, 9, 10])]

print(f"\n초기 데이터: {len(early_train):,}개 ({len(early_train)/len(train_last)*100:.1f}%)")
print(f"후기 데이터: {len(late_train):,}개 ({len(late_train)/len(train_last)*100:.1f}%)")

def compare_periods(early, late, column, label):
    """두 시기 비교"""
    early_vals = early[column].dropna()
    late_vals = late[column].dropna()

    early_mean = early_vals.mean()
    late_mean = late_vals.mean()
    diff = late_mean - early_mean
    pct_change = (diff / early_mean * 100) if early_mean != 0 else 0

    # T-test
    t_stat, p_val = stats.ttest_ind(early_vals, late_vals)

    print(f"\n{label}:")
    print(f"  초기: mean={early_mean:.2f}, std={early_vals.std():.2f}")
    print(f"  후기: mean={late_mean:.2f}, std={late_vals.std():.2f}")
    print(f"  변화: {diff:+.2f} ({pct_change:+.1f}%)")
    print(f"  T-test: t={t_stat:.2f}, p={p_val:.4f}")

    if p_val < 0.05:
        print(f"  >>> 통계적으로 유의미한 변화!")
        return True
    else:
        print(f"  >>> 변화 없음")
        return False

significant_changes = []

if compare_periods(early_train, late_train, 'delta_x', 'delta_x'):
    significant_changes.append('delta_x')

if compare_periods(early_train, late_train, 'delta_y', 'delta_y'):
    significant_changes.append('delta_y')

if compare_periods(early_train, late_train, 'distance', 'distance'):
    significant_changes.append('distance')

if compare_periods(early_train, late_train, 'start_x', 'start_x'):
    significant_changes.append('start_x')

if compare_periods(early_train, late_train, 'start_y', 'start_y'):
    significant_changes.append('start_y')

# 3. Zone 분포 변화
print("\n" + "=" * 80)
print("[3] Zone 분포 시간적 변화")
print("=" * 80)

early_zone_dist = early_train['zone'].value_counts(normalize=True).sort_index()
late_zone_dist = late_train['zone'].value_counts(normalize=True).sort_index()

all_zones = sorted(set(early_zone_dist.index) | set(late_zone_dist.index))

print("\n주요 Zone 분포 변화 (Top 10):")
print(f"{'Zone':>6} {'초기 %':>10} {'후기 %':>10} {'변화':>10}")
print("-" * 40)

zone_changes = []
for zone in all_zones:
    early_pct = early_zone_dist.get(zone, 0) * 100
    late_pct = late_zone_dist.get(zone, 0) * 100
    diff = late_pct - early_pct
    zone_changes.append((zone, early_pct, late_pct, abs(diff)))

zone_changes.sort(key=lambda x: x[3], reverse=True)
for zone, early_pct, late_pct, abs_diff in zone_changes[:10]:
    diff = late_pct - early_pct
    flag = "!" if abs_diff > 1 else ""
    print(f"{zone:>6} {early_pct:>9.1f}% {late_pct:>9.1f}% {diff:>9.1f}% {flag}")

# Chi-square test
from scipy.stats import chi2_contingency
early_zone_counts = early_train['zone'].value_counts()
late_zone_counts = late_train['zone'].value_counts()

all_zones_set = sorted(set(early_zone_counts.index) | set(late_zone_counts.index))
early_counts = [early_zone_counts.get(z, 0) for z in all_zones_set]
late_counts = [late_zone_counts.get(z, 0) for z in all_zones_set]

chi2, pval, dof, expected = chi2_contingency([early_counts, late_counts])
print(f"\nChi-square Test: χ²={chi2:.2f}, p-value={pval:.6f}")

zone_changed = False
if pval < 0.05:
    print(">>> Zone 분포가 시간에 따라 유의미하게 변화!")
    zone_changed = True
else:
    print(">>> Zone 분포 안정적")

# 4. 라운드(Game Day)별 추이
print("\n" + "=" * 80)
print("[4] 라운드별 패턴 추이")
print("=" * 80)

gameday_stats = train_last.groupby('game_day').agg({
    'delta_x': 'mean',
    'delta_y': 'mean',
    'distance': 'mean',
    'game_episode': 'count'
}).round(2)

print(f"\n총 {len(gameday_stats)} 라운드")
print(f"\n라운드별 평균 (처음 5개):")
print(gameday_stats.head())
print(f"\n라운드별 평균 (마지막 5개):")
print(gameday_stats.tail())

# Trend analysis
from scipy.stats import pearsonr

gameday_list = train_last['game_day'].values
delta_x_list = train_last['delta_x'].values

corr_dx, p_dx = pearsonr(gameday_list, delta_x_list)
corr_dy, p_dy = pearsonr(gameday_list, train_last['delta_y'].values)
corr_dist, p_dist = pearsonr(gameday_list, train_last['distance'].values)

print(f"\n라운드-패턴 상관관계:")
print(f"  delta_x: r={corr_dx:.4f}, p={p_dx:.4f}")
print(f"  delta_y: r={corr_dy:.4f}, p={p_dy:.4f}")
print(f"  distance: r={corr_dist:.4f}, p={p_dist:.4f}")

trend_exists = False
if abs(corr_dx) > 0.05 and p_dx < 0.05:
    print(f"  >>> delta_x에 시간적 트렌드 존재!")
    trend_exists = True
if abs(corr_dy) > 0.05 and p_dy < 0.05:
    print(f"  >>> delta_y에 시간적 트렌드 존재!")
    trend_exists = True
if abs(corr_dist) > 0.05 and p_dist < 0.05:
    print(f"  >>> distance에 시간적 트렌드 존재!")
    trend_exists = True

if not trend_exists:
    print("  >>> 시간적 트렌드 미미")

# 5. CV Strategy 시뮬레이션
print("\n" + "=" * 80)
print("[5] 검증 전략 시뮬레이션")
print("=" * 80)

# Random Split vs Time Split 비교
from sklearn.model_selection import KFold, GroupKFold

# 간단한 Zone Baseline 모델로 테스트
def zone_baseline_score(train_data, val_data):
    """Zone 기반 median 예측"""
    zone_stats = train_data.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    })

    val_data = val_data.copy()
    val_data['pred_delta_x'] = val_data['zone'].map(zone_stats['delta_x'])
    val_data['pred_delta_y'] = val_data['zone'].map(zone_stats['delta_y'])

    # 매핑되지 않은 zone은 전체 median 사용
    overall_dx = train_data['delta_x'].median()
    overall_dy = train_data['delta_y'].median()
    val_data['pred_delta_x'].fillna(overall_dx, inplace=True)
    val_data['pred_delta_y'].fillna(overall_dy, inplace=True)

    val_data['pred_end_x'] = val_data['start_x'] + val_data['pred_delta_x']
    val_data['pred_end_y'] = val_data['start_y'] + val_data['pred_delta_y']

    # 유클리드 거리
    errors = np.sqrt(
        (val_data['end_x'] - val_data['pred_end_x'])**2 +
        (val_data['end_y'] - val_data['pred_end_y'])**2
    )

    return errors.mean()

# Time-based split (Train 80% / Val 20%)
sorted_data = train_last.sort_values('game_date')
split_idx = int(len(sorted_data) * 0.8)
time_train = sorted_data.iloc[:split_idx]
time_val = sorted_data.iloc[split_idx:]

time_score = zone_baseline_score(time_train, time_val)

print(f"\nTime-based Split (80/20):")
print(f"  Train: {time_train['game_date'].min()} ~ {time_train['game_date'].max()}")
print(f"  Val:   {time_val['game_date'].min()} ~ {time_val['game_date'].max()}")
print(f"  Score: {time_score:.2f}")

# Random split (5-fold)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

random_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(train_last), 1):
    fold_train = train_last.iloc[train_idx]
    fold_val = train_last.iloc[val_idx]
    score = zone_baseline_score(fold_train, fold_val)
    random_scores.append(score)

print(f"\nRandom 5-Fold CV:")
print(f"  Mean Score: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
print(f"  Individual: {[f'{s:.2f}' for s in random_scores]}")

# Gap 분석
gap = time_score - np.mean(random_scores)
print(f"\nGap (Time - Random): {gap:+.2f}")

if gap > 1.0:
    print("  >>> WARNING: Time split이 더 어려움 (Temporal Shift 존재)")
    temporal_shift = True
else:
    print("  >>> Time split과 Random split 유사")
    temporal_shift = False

# 6. 종합 결론
print("\n" + "=" * 80)
print("[6] 종합 결론")
print("=" * 80)

print("\n과적합 원인 분석:")
print("-" * 80)

issues_found = []

if len(significant_changes) > 0:
    issues_found.append(f"시간에 따른 패턴 변화: {', '.join(significant_changes)}")
    print(f"1. 시간적 패턴 변화: {len(significant_changes)}개 변수 유의미하게 변화")
    for var in significant_changes:
        print(f"   - {var}")
else:
    print("1. 시간적 패턴 변화: 없음 (Good)")

if zone_changed:
    issues_found.append("Zone 분포 시간적 변화")
    print("2. Zone 분포: 시간에 따라 변화 (p<0.05)")
else:
    print("2. Zone 분포: 시간적으로 안정 (Good)")

if trend_exists:
    issues_found.append("라운드별 트렌드 존재")
    print("3. 라운드 트렌드: 존재")
else:
    print("3. 라운드 트렌드: 없음 (Good)")

if temporal_shift:
    issues_found.append(f"Temporal Shift (Gap: {gap:+.2f})")
    print(f"4. Temporal Shift: 존재 (Gap: {gap:+.2f})")
else:
    print("4. Temporal Shift: 미미 (Good)")

print("\n" + "=" * 80)
print("최종 진단:")
print("=" * 80)

if len(issues_found) == 0:
    print("\n✓ 시간적 분포 변화 없음")
    print(">>> 과적합 원인: 모델 복잡도 과다 (피처/파라미터)")
    print(">>> 해결책: 정규화 강화, 피처 선택")
else:
    print(f"\n발견된 시간적 이슈: {len(issues_found)}개")
    for i, issue in enumerate(issues_found, 1):
        print(f"{i}. {issue}")

    print("\n>>> 과적합 원인: Temporal Shift (시간적 분포 변화)")
    print(">>> 해결책:")
    print("    1. Time-based validation 사용")
    print("    2. 최근 데이터에 더 높은 가중치")
    print("    3. 시간 불변 피처 사용 (기하학적 피처)")
    print("    4. 앙상블: Robust baseline + Adaptive model")

print("\n" + "=" * 80)
print("추천 모델 전략:")
print("=" * 80)

if temporal_shift:
    print("""
1. Zone Baseline (시간 불변):
   - 전체 Train 데이터 사용
   - 6x6 median (현재 Best: 16.85)
   - 가중치: 60-70%

2. Adaptive Model (시간 고려):
   - 최근 3개월 데이터만 사용
   - 정규화 강화
   - 가중치: 30-40%

3. 앙상블:
   - Zone 70% + ML 30%
   - Weighted by recency
    """)
else:
    print("""
1. 모델 단순화:
   - 피처 수 감소 (31개 → 15-20개)
   - 정규화 파라미터 강화

2. Cross-validation:
   - GroupKFold by game_id
   - 5-fold sufficient

3. 앙상블:
   - Zone 50% + ML 50%
   - Bootstrap aggregating
    """)

print("분석 완료!")
print("=" * 80)
