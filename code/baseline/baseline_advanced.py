"""
K리그 패스 좌표 예측 - Advanced Baseline (에이전트 합의 전략)
Phase 1: Zone 세분화 실험
Phase 2: Baseline 앙상블
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - Advanced Baseline")
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

print(f"Train episodes: {train_df['game_episode'].nunique():,}")
print(f"Test episodes: {test_all['game_episode'].nunique():,}")

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

# 전체 평균
global_mean_dx = train_last['delta_x'].mean()
global_mean_dy = train_last['delta_y'].mean()
global_median_dx = train_last['delta_x'].median()
global_median_dy = train_last['delta_y'].median()

print(f"\n전체 평균: dx={global_mean_dx:.2f}, dy={global_mean_dy:.2f}")
print(f"전체 중앙값: dx={global_median_dx:.2f}, dy={global_median_dy:.2f}")

# =============================================================================
# 3. Zone 분할 함수들
# =============================================================================
print("\n[3] Zone 분할 전략 정의...")

def get_zone_3x3(x, y):
    """9구역 (3x3) - 기존"""
    x_zone = 0 if x < 35 else (1 if x < 70 else 2)
    y_zone = 0 if y < 22.67 else (1 if y < 45.33 else 2)
    return x_zone * 3 + y_zone

def get_zone_4x3(x, y):
    """12구역 (4x3)"""
    x_zone = min(3, int(x // 26.25))
    y_zone = min(2, int(y // 22.67))
    return x_zone * 3 + y_zone

def get_zone_4x4(x, y):
    """16구역 (4x4)"""
    x_zone = min(3, int(x // 26.25))
    y_zone = min(3, int(y // 17))
    return x_zone * 4 + y_zone

def get_zone_5x5(x, y):
    """25구역 (5x5)"""
    x_zone = min(4, int(x // 21))
    y_zone = min(4, int(y // 13.6))
    return x_zone * 5 + y_zone

def get_zone_x_only(x, y):
    """X축 기준 4구역"""
    return min(3, int(x // 26.25))

def get_zone_attacking(x, y):
    """공격 지역 세분화 (x>70 구역만 4분할)"""
    if x < 35:
        return 0  # 수비
    elif x < 70:
        return 1  # 중앙
    else:
        # 공격 지역 세분화
        sub_x = 2 if x < 87.5 else 3
        sub_y = 0 if y < 34 else 1
        return 2 + sub_x + sub_y * 2

zone_functions = {
    '3x3 (9구역)': get_zone_3x3,
    '4x3 (12구역)': get_zone_4x3,
    '4x4 (16구역)': get_zone_4x4,
    '5x5 (25구역)': get_zone_5x5,
    'X축 (4구역)': get_zone_x_only,
    '공격세분화': get_zone_attacking,
}

# =============================================================================
# 4. 각 Zone 전략별 통계 계산 및 CV 평가
# =============================================================================
print("\n[4] Zone 전략별 CV 평가...")

def evaluate_zone_strategy(train_data, zone_func, use_median=False):
    """Zone 전략 평가 (Leave-One-Game-Out CV 근사)"""
    train_data = train_data.copy()
    train_data['zone'] = train_data.apply(
        lambda r: zone_func(r['start_x'], r['start_y']), axis=1
    )

    # 구역별 통계 계산
    if use_median:
        zone_stats = train_data.groupby('zone').agg({
            'delta_x': 'median',
            'delta_y': 'median'
        }).to_dict()
    else:
        zone_stats = train_data.groupby('zone').agg({
            'delta_x': 'mean',
            'delta_y': 'mean'
        }).to_dict()

    # 구역별 샘플 수
    zone_counts = train_data['zone'].value_counts().to_dict()
    min_samples = min(zone_counts.values())
    n_zones = len(zone_counts)

    # 예측 및 평가
    predictions = []
    for _, row in train_data.iterrows():
        zone = row['zone']
        dx = zone_stats['delta_x'].get(zone, global_mean_dx)
        dy = zone_stats['delta_y'].get(zone, global_mean_dy)
        pred_x = np.clip(row['start_x'] + dx, 0, 105)
        pred_y = np.clip(row['start_y'] + dy, 0, 68)
        predictions.append([pred_x, pred_y])

    predictions = np.array(predictions)
    actuals = train_data[['end_x', 'end_y']].values

    distances = np.sqrt(np.sum((predictions - actuals) ** 2, axis=1))
    cv_score = distances.mean()

    return cv_score, n_zones, min_samples, zone_stats, zone_counts

results = []

for name, func in zone_functions.items():
    # Mean 기반
    cv_mean, n_zones, min_samples, stats_mean, counts = evaluate_zone_strategy(
        train_last, func, use_median=False
    )
    results.append({
        'strategy': name,
        'stat': 'mean',
        'cv_score': cv_mean,
        'n_zones': n_zones,
        'min_samples': min_samples,
        'stats': stats_mean,
        'counts': counts
    })

    # Median 기반
    cv_median, _, _, stats_median, _ = evaluate_zone_strategy(
        train_last, func, use_median=True
    )
    results.append({
        'strategy': name,
        'stat': 'median',
        'cv_score': cv_median,
        'n_zones': n_zones,
        'min_samples': min_samples,
        'stats': stats_median,
        'counts': counts
    })

# 결과 정렬
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('cv_score')

print("\n" + "=" * 70)
print("Zone 전략별 CV Score (낮을수록 좋음)")
print("=" * 70)
print(f"{'전략':<20} {'통계':<8} {'CV Score':<10} {'구역수':<8} {'최소샘플':<10}")
print("-" * 70)
for _, row in results_df.iterrows():
    print(f"{row['strategy']:<20} {row['stat']:<8} {row['cv_score']:<10.4f} {row['n_zones']:<8} {row['min_samples']:<10}")

# =============================================================================
# 5. 상위 전략 선택 및 제출 파일 생성
# =============================================================================
print("\n[5] 상위 전략으로 제출 파일 생성...")

def generate_submission(test_data, zone_func, zone_stats, global_dx, global_dy):
    """제출 파일 생성"""
    predictions = []

    for _, row in test_data.iterrows():
        zone = zone_func(row['start_x'], row['start_y'])
        dx = zone_stats['delta_x'].get(zone, global_dx)
        dy = zone_stats['delta_y'].get(zone, global_dy)
        pred_x = np.clip(row['start_x'] + dx, 0, 105)
        pred_y = np.clip(row['start_y'] + dy, 0, 68)
        predictions.append({
            'game_episode': row['game_episode'],
            'end_x': pred_x,
            'end_y': pred_y
        })

    return pd.DataFrame(predictions)

# 상위 3개 전략으로 제출 파일 생성
top_strategies = results_df.head(6)  # mean/median 포함해서 상위 6개

submissions = {}

for idx, row in top_strategies.iterrows():
    strategy_name = row['strategy']
    stat_type = row['stat']

    # 해당 전략의 zone 함수
    zone_func = zone_functions[strategy_name]

    # 통계 재계산
    train_temp = train_last.copy()
    train_temp['zone'] = train_temp.apply(
        lambda r: zone_func(r['start_x'], r['start_y']), axis=1
    )

    if stat_type == 'median':
        zone_stats = train_temp.groupby('zone').agg({
            'delta_x': 'median',
            'delta_y': 'median'
        }).to_dict()
        global_dx = global_median_dx
        global_dy = global_median_dy
    else:
        zone_stats = train_temp.groupby('zone').agg({
            'delta_x': 'mean',
            'delta_y': 'mean'
        }).to_dict()
        global_dx = global_mean_dx
        global_dy = global_mean_dy

    # 제출 파일 생성
    sub_df = generate_submission(test_last, zone_func, zone_stats, global_dx, global_dy)
    sub_df = sample_sub[['game_episode']].merge(sub_df, on='game_episode', how='left')

    filename = f"submission_{strategy_name.replace(' ', '_').replace('(', '').replace(')', '')}_{stat_type}.csv"
    sub_df.to_csv(filename, index=False)

    submissions[f"{strategy_name}_{stat_type}"] = {
        'filename': filename,
        'cv_score': row['cv_score'],
        'submission': sub_df
    }

    print(f"  {filename}: CV={row['cv_score']:.4f}")

# =============================================================================
# 6. 앙상블 생성
# =============================================================================
print("\n[6] 앙상블 제출 파일 생성...")

# 상위 3개 전략 앙상블 (동일 가중치)
top3 = list(submissions.keys())[:3]
ensemble_preds_x = np.zeros(len(sample_sub))
ensemble_preds_y = np.zeros(len(sample_sub))

for key in top3:
    sub = submissions[key]['submission']
    ensemble_preds_x += sub['end_x'].values / 3
    ensemble_preds_y += sub['end_y'].values / 3

ensemble_df = pd.DataFrame({
    'game_episode': sample_sub['game_episode'],
    'end_x': np.clip(ensemble_preds_x, 0, 105),
    'end_y': np.clip(ensemble_preds_y, 0, 68)
})
ensemble_df.to_csv('submission_ensemble_top3.csv', index=False)

# 상위 전략의 CV 점수로 가중 평균
weights = []
for key in top3:
    cv = submissions[key]['cv_score']
    weights.append(1 / cv)  # CV가 낮을수록 높은 가중치
weights = np.array(weights) / sum(weights)

weighted_preds_x = np.zeros(len(sample_sub))
weighted_preds_y = np.zeros(len(sample_sub))

for i, key in enumerate(top3):
    sub = submissions[key]['submission']
    weighted_preds_x += sub['end_x'].values * weights[i]
    weighted_preds_y += sub['end_y'].values * weights[i]

weighted_df = pd.DataFrame({
    'game_episode': sample_sub['game_episode'],
    'end_x': np.clip(weighted_preds_x, 0, 105),
    'end_y': np.clip(weighted_preds_y, 0, 68)
})
weighted_df.to_csv('submission_ensemble_weighted.csv', index=False)

print(f"  submission_ensemble_top3.csv: 상위 3개 동일 가중치 앙상블")
print(f"  submission_ensemble_weighted.csv: CV 기반 가중 앙상블")
print(f"    가중치: {dict(zip(top3, weights))}")

# =============================================================================
# 7. 전체 평균 + Zone Baseline 혼합
# =============================================================================
print("\n[7] 전체 평균 + Zone Baseline 혼합...")

# 기존 Zone Baseline (9구역)
train_temp = train_last.copy()
train_temp['zone'] = train_temp.apply(
    lambda r: get_zone_3x3(r['start_x'], r['start_y']), axis=1
)
zone_stats_3x3 = train_temp.groupby('zone').agg({
    'delta_x': 'mean',
    'delta_y': 'mean'
}).to_dict()

zone_preds = []
for _, row in test_last.iterrows():
    zone = get_zone_3x3(row['start_x'], row['start_y'])
    dx = zone_stats_3x3['delta_x'].get(zone, global_mean_dx)
    dy = zone_stats_3x3['delta_y'].get(zone, global_mean_dy)
    zone_preds.append([row['start_x'] + dx, row['start_y'] + dy])
zone_preds = np.array(zone_preds)

# 전체 평균 예측
global_preds = np.column_stack([
    test_last['start_x'].values + global_mean_dx,
    test_last['start_y'].values + global_mean_dy
])

# 혼합 (Zone 80% + Global 20%)
mixed_preds = 0.8 * zone_preds + 0.2 * global_preds
mixed_df = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': np.clip(mixed_preds[:, 0], 0, 105),
    'end_y': np.clip(mixed_preds[:, 1], 0, 68)
})
mixed_df = sample_sub[['game_episode']].merge(mixed_df, on='game_episode', how='left')
mixed_df.to_csv('submission_mixed_zone_global.csv', index=False)

print(f"  submission_mixed_zone_global.csv: Zone 80% + Global 20%")

# =============================================================================
# 8. 최종 결과 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 결과 요약")
print("=" * 70)

print("\n[제출 파일 목록]")
print("-" * 70)

all_submissions = [
    ('submission_zone_baseline.csv', 17.57, 17.95, '기존 제출'),
]

for key, data in submissions.items():
    all_submissions.append((data['filename'], data['cv_score'], None, '새로 생성'))

all_submissions.extend([
    ('submission_ensemble_top3.csv', None, None, '앙상블'),
    ('submission_ensemble_weighted.csv', None, None, '가중 앙상블'),
    ('submission_mixed_zone_global.csv', None, None, '혼합'),
])

print(f"{'파일명':<45} {'CV Score':<12} {'Public':<10} {'비고':<15}")
print("-" * 70)
for filename, cv, public, note in all_submissions:
    cv_str = f"{cv:.4f}" if cv else "-"
    public_str = f"{public:.2f}" if public else "미제출"
    print(f"{filename:<45} {cv_str:<12} {public_str:<10} {note:<15}")

print("\n[추천 제출 순서]")
print("-" * 70)
print("1. 가장 낮은 CV의 단일 전략 제출")
print("2. submission_ensemble_weighted.csv (가중 앙상블)")
print("3. submission_ensemble_top3.csv (동일 가중 앙상블)")

# Best CV 전략
best = results_df.iloc[0]
print(f"\n[Best CV 전략]")
print(f"  전략: {best['strategy']} ({best['stat']})")
print(f"  CV Score: {best['cv_score']:.4f}")
print(f"  구역 수: {best['n_zones']}")
print(f"  최소 샘플: {best['min_samples']}")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
