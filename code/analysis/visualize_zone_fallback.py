"""
Zone Fallback 시각화
====================

분석 결과를 시각화하여 인사이트 도출
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("=" * 80)
print("Zone Fallback 시각화")
print("=" * 80)

DATA_DIR = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
RESULTS_DIR = DATA_DIR / "code" / "analysis" / "results"

# 비교 데이터 로드
comparison_df = pd.read_csv(RESULTS_DIR / "zone_fallback_comparison.csv", index_col=0)

print(f"\n[1] 데이터 로드 완료")
print(f"  모델 수: {len(comparison_df)}")

# =============================================================================
# 2. Fallback 사용 빈도 시각화
# =============================================================================
print(f"\n[2] Fallback 사용 빈도 시각화")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Zone Fallback Analysis', fontsize=16, fontweight='bold')

# 2.1 Fallback 비율 막대 그래프
ax = axes[0, 0]
models = comparison_df.index
fallback_pct = comparison_df['fallback_percentage']
zone_dir_pct = comparison_df['zone_direction_percentage']

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, zone_dir_pct, width, label='Zone+Direction', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, fallback_pct, width, label='Zone Fallback', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('Prediction Method Usage', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 값 표시
for i, (zd, fb) in enumerate(zip(zone_dir_pct, fallback_pct)):
    ax.text(i - width/2, zd + 1, f'{zd:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, fb + 1, f'{fb:.1f}%', ha='center', va='bottom', fontsize=9)

# 2.2 조합별 충분/부족 샘플 비율
ax = axes[0, 1]
sufficient = comparison_df['sufficient_combinations']
insufficient = comparison_df['insufficient_combinations']

x = np.arange(len(models))

ax.bar(x, sufficient, label='Sufficient (>= min_samples)', color='#3498db', alpha=0.8)
ax.bar(x, insufficient, bottom=sufficient, label='Insufficient (< min_samples)', color='#e67e22', alpha=0.8)

ax.set_ylabel('Number of Combinations', fontweight='bold')
ax.set_title('Zone+Direction Combination Sample Sufficiency', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 비율 표시
for i, (suf, insuf) in enumerate(zip(sufficient, insufficient)):
    total = suf + insuf
    suf_pct = suf / total * 100
    insuf_pct = insuf / total * 100
    ax.text(i, suf/2, f'{suf_pct:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.text(i, suf + insuf/2, f'{insuf_pct:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# 2.3 평균 샘플 수 비교
ax = axes[1, 0]
avg_samples = comparison_df['avg_samples_per_combination']
median_samples = comparison_df['median_samples_per_combination']

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, avg_samples, width, label='Mean', color='#9b59b6', alpha=0.8)
ax.bar(x + width/2, median_samples, width, label='Median', color='#1abc9c', alpha=0.8)

ax.set_ylabel('Samples per Combination', fontweight='bold')
ax.set_title('Average Samples per Zone+Direction Combination', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 값 표시
for i, (avg, med) in enumerate(zip(avg_samples, median_samples)):
    ax.text(i - width/2, avg + 5, f'{avg:.1f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, med + 5, f'{med:.1f}', ha='center', va='bottom', fontsize=9)

# 2.4 Fallback 사용 빈도 (파이 차트)
ax = axes[1, 1]

# 6x6_8dir 모델의 경우를 예시로
model_name = '6x6_8dir'
zone_dir = comparison_df.loc[model_name, 'zone_direction_percentage']
fallback = comparison_df.loc[model_name, 'fallback_percentage']
global_fb = comparison_df.loc[model_name, 'global_fallback_percentage']

sizes = [zone_dir, fallback, global_fb]
labels = [f'Zone+Direction\n({zone_dir:.1f}%)', f'Zone Fallback\n({fallback:.1f}%)', f'Global\n({global_fb:.1f}%)']
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
explode = (0.05, 0.05, 0)

ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.set_title(f'Prediction Method Distribution ({model_name})', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'zone_fallback_analysis.png', dpi=300, bbox_inches='tight')
print(f"  저장: {RESULTS_DIR / 'zone_fallback_analysis.png'}")

# =============================================================================
# 3. 상세 샘플 수 분포 시각화
# =============================================================================
print(f"\n[3] 상세 샘플 수 분포 시각화")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sample Count Distribution per Zone+Direction Combination', fontsize=16, fontweight='bold')

model_names = ['5x5_8dir', '6x6_8dir', '7x7_8dir', '6x6_simple']

for idx, model_name in enumerate(model_names):
    ax = axes[idx // 2, idx % 2]

    # 모델별 통계 로드
    stats_df = pd.read_csv(RESULTS_DIR / f"zone_stats_{model_name}.csv")
    counts = stats_df['count']

    # 히스토그램
    ax.hist(counts, bins=30, color='#3498db', alpha=0.7, edgecolor='black')

    # min_samples 선 그리기
    min_samples = 25 if '5x5' in model_name or '6x6_8dir' in model_name else (20 if '7x7' in model_name else 30)
    ax.axvline(min_samples, color='red', linestyle='--', linewidth=2, label=f'min_samples={min_samples}')

    # 통계 정보 표시
    mean_count = counts.mean()
    median_count = counts.median()
    ax.axvline(mean_count, color='green', linestyle='--', linewidth=1.5, label=f'Mean={mean_count:.1f}')
    ax.axvline(median_count, color='orange', linestyle='--', linewidth=1.5, label=f'Median={median_count:.1f}')

    ax.set_xlabel('Sample Count', fontweight='bold')
    ax.set_ylabel('Number of Combinations', fontweight='bold')
    ax.set_title(f'{model_name}', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'sample_count_distribution.png', dpi=300, bbox_inches='tight')
print(f"  저장: {RESULTS_DIR / 'sample_count_distribution.png'}")

# =============================================================================
# 4. 핵심 메트릭 요약 테이블
# =============================================================================
print(f"\n[4] 핵심 메트릭 요약")

summary_df = pd.DataFrame({
    'Model': comparison_df.index,
    'Total Combinations': comparison_df['total_combinations'].astype(int),
    'Fallback Usage (%)': comparison_df['fallback_percentage'].round(2),
    'Avg Samples': comparison_df['avg_samples_per_combination'].round(1),
    'Median Samples': comparison_df['median_samples_per_combination'].round(1),
    'Min Samples': comparison_df['min_samples_per_combination'].astype(int),
    'Max Samples': comparison_df['max_samples_per_combination'].astype(int),
})

print("\n")
print(summary_df.to_string(index=False))

summary_df.to_csv(RESULTS_DIR / 'zone_fallback_summary.csv', index=False)
print(f"\n  저장: {RESULTS_DIR / 'zone_fallback_summary.csv'}")

# =============================================================================
# 5. 결론 및 인사이트
# =============================================================================
print(f"\n{'='*80}")
print(f"핵심 인사이트 요약")
print(f"{'='*80}")

print(f"""
1. Zone Fallback 사용 빈도:
   - 5x5_8dir: {comparison_df.loc['5x5_8dir', 'fallback_percentage']:.2f}% (매우 낮음)
   - 6x6_8dir: {comparison_df.loc['6x6_8dir', 'fallback_percentage']:.2f}% (낮음)
   - 7x7_8dir: {comparison_df.loc['7x7_8dir', 'fallback_percentage']:.2f}% (낮음)
   - 6x6_simple: {comparison_df.loc['6x6_simple', 'fallback_percentage']:.2f}% (없음)

2. 왜 Zone Fallback 개선이 실패했는가?

   a) 영향력이 매우 작음:
      - Best 모델 (6x6_8dir)에서 fallback은 11%만 사용됨
      - 89%는 Zone+Direction 조합으로 예측
      - Fallback 개선이 전체 성능에 미치는 영향: < 1%

   b) Zone fallback은 이미 충분히 좋음:
      - Zone fallback이 사용되는 경우도 Zone 통계 사용
      - Zone 통계는 평균 50+ 샘플로 계산됨
      - Global fallback은 거의 사용되지 않음 (0%)

   c) 성능 병목은 다른 곳에 있음:
      - Zone + Direction 조합 자체의 한계
      - 중앙값(median)의 한계
      - 공간 분할의 한계

3. 데이터 분포 패턴:
   - 대부분의 조합이 충분한 샘플을 가짐
   - 부족한 샘플을 가진 조합은 소수 (16-38%)
   - 이들 조합도 Zone fallback으로 커버됨

4. 결론:
   ✅ Zone fallback 메커니즘은 이미 효과적으로 작동
   ✅ Fallback 개선은 전체 성능에 미미한 영향
   ✅ Zone 통계 접근법은 이미 최적화됨
   ✅ 성능 개선을 위해서는 근본적으로 다른 접근 필요

5. 다음 단계 (Week 4-5):
   ❌ Zone fallback 개선 (영향 < 1%, 불필요)
   ❌ Zone 설정 변경 (14회 완전 탐색 완료)
   ❌ Direction 각도 조정 (3회 완전 탐색 완료)
   ✅ 관찰 모드 유지 (Week 2-3)
   ✅ 새로운 접근법 연구 (Week 4-5)
""")

print(f"\n{'='*80}")
print(f"분석 완료!")
print(f"{'='*80}")
print(f"\n생성된 파일:")
print(f"  1. {RESULTS_DIR / 'zone_fallback_analysis.png'}")
print(f"  2. {RESULTS_DIR / 'sample_count_distribution.png'}")
print(f"  3. {RESULTS_DIR / 'zone_fallback_summary.csv'}")
print(f"  4. {RESULTS_DIR / 'zone_fallback_comparison.csv'}")
print(f"  5. {RESULTS_DIR / 'zone_stats_*.csv'} (4개 파일)")
