"""
OOD Impact Visualization

시각화:
1. 게임별 CV 변동성 분포
2. CV-Public Gap 비교 (접근법별)
3. 모델 복잡도 vs Gap 산점도
4. LOGO CV vs GroupKFold CV 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = Path("analysis_results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("OOD Impact Visualization")
print("=" * 80)

# 데이터 로드
game_df = pd.read_csv(RESULTS_DIR / "game_level_cv_scores.csv")
gap_df = pd.read_csv(RESULTS_DIR / "cv_public_gap_analysis.csv")

# =============================================================================
# 1. 게임별 CV 변동성 분포
# =============================================================================
print("\n[1] 게임별 CV 변동성 분포...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Game-Level CV Variability Analysis (Zone 6x6)', fontsize=16, weight='bold')

# 1.1 히스토그램
ax = axes[0, 0]
ax.hist(game_df['cv_score'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(game_df['cv_score'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax.axvline(game_df['cv_score'].median(), color='green', linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('CV Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Game-Level CV Scores', fontsize=13, weight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 1.2 박스플롯 + 바이올린
ax = axes[0, 1]
parts = ax.violinplot([game_df['cv_score']], positions=[0], widths=0.7,
                       showmeans=True, showmedians=True)
ax.boxplot([game_df['cv_score']], positions=[0], widths=0.3)
ax.set_ylabel('CV Score', fontsize=12)
ax.set_title('CV Score Distribution', fontsize=13, weight='bold')
ax.set_xticks([0])
ax.set_xticklabels(['Zone 6x6'])
ax.grid(alpha=0.3)

# 1.3 게임 ID별 CV (상위/하위 강조)
ax = axes[1, 0]
sorted_games = game_df.sort_values('cv_score')
colors = ['green' if i < 5 else 'red' if i >= len(sorted_games) - 5 else 'gray'
          for i in range(len(sorted_games))]
ax.scatter(range(len(sorted_games)), sorted_games['cv_score'], c=colors, alpha=0.6, s=30)
ax.axhline(game_df['cv_score'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
ax.set_xlabel('Game Index (sorted by CV)', fontsize=12)
ax.set_ylabel('CV Score', fontsize=12)
ax.set_title('CV Scores Sorted by Performance', fontsize=13, weight='bold')
ax.legend(['Mean', 'Easy (Top 5)', 'Hard (Bottom 5)', 'Others'])
ax.grid(alpha=0.3)

# 1.4 에피소드 수 vs CV
ax = axes[1, 1]
ax.scatter(game_df['n_episodes'], game_df['cv_score'], alpha=0.6, s=50, color='steelblue')
z = np.polyfit(game_df['n_episodes'], game_df['cv_score'], 1)
p = np.poly1d(z)
ax.plot(game_df['n_episodes'], p(game_df['n_episodes']), "r--", linewidth=2,
        label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
ax.set_xlabel('Number of Episodes', fontsize=12)
ax.set_ylabel('CV Score', fontsize=12)
ax.set_title('Episodes vs CV Score', fontsize=13, weight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '1_game_cv_variability.png', dpi=300, bbox_inches='tight')
print(f"  저장: {PLOTS_DIR / '1_game_cv_variability.png'}")

# =============================================================================
# 2. CV-Public Gap 비교 (접근법별)
# =============================================================================
print("\n[2] CV-Public Gap 비교...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CV-Public Gap Analysis by Approach', fontsize=16, weight='bold')

# 2.1 접근법별 Gap 박스플롯
ax = axes[0, 0]
approach_order = ['Zone', 'GBDT+Features', 'Deep Learning']
gap_df_filtered = gap_df[gap_df['approach'].isin(approach_order)]
sns.boxplot(data=gap_df_filtered, x='approach', y='gap', ax=ax, palette='Set2')
ax.set_xlabel('Approach', fontsize=12)
ax.set_ylabel('CV-Public Gap', fontsize=12)
ax.set_title('Gap Distribution by Approach', fontsize=13, weight='bold')
ax.grid(alpha=0.3)

# 2.2 복잡도별 Gap
ax = axes[0, 1]
complexity_order = ['Low', 'Medium', 'High', 'Very High']
gap_df['complexity'] = pd.Categorical(gap_df['complexity'], categories=complexity_order, ordered=True)
complexity_stats = gap_df.groupby('complexity')['gap'].agg(['mean', 'std', 'count'])
complexity_stats = complexity_stats.reindex(complexity_order)

x_pos = np.arange(len(complexity_stats))
ax.bar(x_pos, complexity_stats['mean'], yerr=complexity_stats['std'],
       alpha=0.7, color=['green', 'yellow', 'orange', 'red'], capsize=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(complexity_stats.index, rotation=45)
ax.set_ylabel('Average Gap', fontsize=12)
ax.set_title('Gap by Model Complexity', fontsize=13, weight='bold')
for i, (idx, row) in enumerate(complexity_stats.iterrows()):
    ax.text(i, row['mean'] + (row['std'] if not pd.isna(row['std']) else 0) + 0.2,
            f"n={int(row['count'])}", ha='center', fontsize=10)
ax.grid(alpha=0.3)

# 2.3 CV vs Public 산점도
ax = axes[1, 0]
for approach in approach_order:
    subset = gap_df[gap_df['approach'] == approach]
    ax.scatter(subset['cv'], subset['public'], label=approach, s=100, alpha=0.7)

# Perfect prediction line
min_val = min(gap_df['cv'].min(), gap_df['public'].min())
max_val = max(gap_df['cv'].max(), gap_df['public'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect (Gap=0)')

ax.set_xlabel('CV Score', fontsize=12)
ax.set_ylabel('Public Score', fontsize=12)
ax.set_title('CV vs Public Score', fontsize=13, weight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2.4 모델별 Gap 막대 그래프 (상위/하위 10개)
ax = axes[1, 1]
gap_df_sorted = gap_df.sort_values('gap')
top_models = pd.concat([gap_df_sorted.head(5), gap_df_sorted.tail(5)])
colors_bar = ['green' if gap < 0.5 else 'orange' if gap < 2 else 'red'
              for gap in top_models['gap']]

y_pos = np.arange(len(top_models))
ax.barh(y_pos, top_models['gap'], color=colors_bar, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_models['name'], fontsize=9)
ax.set_xlabel('CV-Public Gap', fontsize=12)
ax.set_title('Top 5 & Bottom 5 Models by Gap', fontsize=13, weight='bold')
ax.axvline(0.2, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Threshold (0.2)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '2_cv_public_gap_comparison.png', dpi=300, bbox_inches='tight')
print(f"  저장: {PLOTS_DIR / '2_cv_public_gap_comparison.png'}")

# =============================================================================
# 3. 모델 복잡도 vs Gap
# =============================================================================
print("\n[3] 모델 복잡도 vs Gap...")

# 파라미터 수 데이터 (수동)
complexity_data = [
    {'name': 'Zone 6x6', 'n_params': 288, 'gap': 0.0283, 'approach': 'Zone'},
    {'name': 'Zone 5x5', 'n_params': 200, 'gap': 0.0843, 'approach': 'Zone'},
    {'name': 'Zone 7x7', 'n_params': 392, 'gap': 0.1435, 'approach': 'Zone'},
    {'name': 'Zone 8x8', 'n_params': 512, 'gap': 0.2227, 'approach': 'Zone'},
    {'name': 'Phase 2 LGBM', 'n_params': 1000, 'gap': 1.43, 'approach': 'GBDT'},
    {'name': 'LSTM v5', 'n_params': 12700, 'gap': 3.00, 'approach': 'DL'},
    {'name': 'LSTM v3', 'n_params': 50000, 'gap': 2.93, 'approach': 'DL'},
    {'name': 'LSTM v2', 'n_params': 50000, 'gap': 6.90, 'approach': 'DL'},
]
complexity_scatter_df = pd.DataFrame(complexity_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Complexity vs OOD Gap', fontsize=16, weight='bold')

# 3.1 산점도 (로그 스케일)
ax = axes[0]
for approach in ['Zone', 'GBDT', 'DL']:
    subset = complexity_scatter_df[complexity_scatter_df['approach'] == approach]
    ax.scatter(subset['n_params'], subset['gap'], label=approach, s=150, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of Parameters (log scale)', fontsize=12)
ax.set_ylabel('CV-Public Gap (log scale)', fontsize=12)
ax.set_title('Complexity vs Gap (Log-Log)', fontsize=13, weight='bold')
ax.legend()
ax.grid(alpha=0.3, which='both', linestyle='--')

# 주요 모델 라벨
for _, row in complexity_scatter_df.iterrows():
    if row['name'] in ['Zone 6x6', 'LSTM v2', 'Phase 2 LGBM']:
        ax.annotate(row['name'], (row['n_params'], row['gap']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# 3.2 Zone 접근법만 (선형)
ax = axes[1]
zone_data = complexity_scatter_df[complexity_scatter_df['approach'] == 'Zone']
ax.scatter(zone_data['n_params'], zone_data['gap'], s=150, alpha=0.7, color='steelblue')

# 선형 회귀
z = np.polyfit(zone_data['n_params'], zone_data['gap'], 1)
p = np.poly1d(z)
x_fit = np.linspace(zone_data['n_params'].min(), zone_data['n_params'].max(), 100)
ax.plot(x_fit, p(x_fit), "r--", linewidth=2,
        label=f'y={z[0]:.6f}x+{z[1]:.3f}\nR²={np.corrcoef(zone_data["n_params"], zone_data["gap"])[0,1]**2:.3f}')

ax.set_xlabel('Number of Parameters', fontsize=12)
ax.set_ylabel('CV-Public Gap', fontsize=12)
ax.set_title('Zone Models: Complexity vs Gap', fontsize=13, weight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 각 점에 모델명 표시
for _, row in zone_data.iterrows():
    ax.annotate(row['name'], (row['n_params'], row['gap']),
               xytext=(5, 5), textcoords='offset points',
               fontsize=9, alpha=0.7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '3_complexity_vs_gap.png', dpi=300, bbox_inches='tight')
print(f"  저장: {PLOTS_DIR / '3_complexity_vs_gap.png'}")

# =============================================================================
# 4. LOGO CV vs GroupKFold CV 비교
# =============================================================================
print("\n[4] LOGO CV vs GroupKFold CV 비교...")

# GroupKFold CV 결과 (수동)
gkf_scores = [16.5387, 16.4770, 16.4682, 15.8413, 15.9006]
logo_mean = game_df['cv_score'].mean()
logo_std = game_df['cv_score'].std()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('LOGO CV vs GroupKFold CV Comparison', fontsize=16, weight='bold')

# 4.1 분포 비교
ax = axes[0]
ax.hist(game_df['cv_score'], bins=30, alpha=0.5, label='LOGO CV (per game)', color='steelblue')
ax.axvline(logo_mean, color='blue', linestyle='--', linewidth=2, label=f'LOGO Mean: {logo_mean:.4f}')
ax.axvline(np.mean(gkf_scores), color='red', linestyle='--', linewidth=2,
           label=f'GKF Mean: {np.mean(gkf_scores):.4f}')

for i, score in enumerate(gkf_scores):
    ax.axvline(score, color='red', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('CV Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution: LOGO vs GroupKFold', fontsize=13, weight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 4.2 박스플롯 비교
ax = axes[1]
data_to_plot = [game_df['cv_score'], gkf_scores]
bp = ax.boxplot(data_to_plot, labels=['LOGO CV\n(198 games)', 'GroupKFold CV\n(5 folds)'],
                patch_artist=True, widths=0.5)

for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('CV Score', fontsize=12)
ax.set_title('CV Score Comparison', fontsize=13, weight='bold')
ax.grid(alpha=0.3)

# 통계 정보 추가
ax.text(0.5, 0.95, f'LOGO: {logo_mean:.4f} ± {logo_std:.4f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.5, 0.88, f'GKF: {np.mean(gkf_scores):.4f} ± {np.std(gkf_scores):.4f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.5, 0.81, f'Diff: {logo_mean - np.mean(gkf_scores):+.4f} ({(logo_mean/np.mean(gkf_scores)-1)*100:+.2f}%)',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / '4_logo_vs_gkf.png', dpi=300, bbox_inches='tight')
print(f"  저장: {PLOTS_DIR / '4_logo_vs_gkf.png'}")

# =============================================================================
# 5. 종합 요약 플롯
# =============================================================================
print("\n[5] 종합 요약 플롯...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('OOD Impact Quantification - Executive Summary', fontsize=18, weight='bold')

# 5.1 게임별 CV 변동계수
ax1 = fig.add_subplot(gs[0, :])
cv_coefficient = game_df['cv_score'].std() / game_df['cv_score'].mean() * 100
bars = ax1.bar(['Zone 6x6'], [cv_coefficient], color='green', alpha=0.7, width=0.3)
ax1.axhline(10, color='orange', linestyle='--', linewidth=2, label='Stable Threshold (10%)')
ax1.axhline(5, color='green', linestyle='--', linewidth=2, label='Very Stable Threshold (5%)')
ax1.set_ylabel('Coefficient of Variation (%)', fontsize=12)
ax1.set_title('Game-Level CV Stability (Lower = Better)', fontsize=14, weight='bold')
ax1.set_ylim([0, 15])
ax1.legend()
ax1.grid(alpha=0.3)
ax1.text(0, cv_coefficient + 0.5, f'{cv_coefficient:.2f}%', ha='center', fontsize=16, weight='bold')

# 5.2 접근법별 평균 Gap
ax2 = fig.add_subplot(gs[1, 0])
approach_gaps = gap_df.groupby('approach')['gap'].mean().sort_values()
colors_gap = ['green' if gap < 0.5 else 'orange' if gap < 2 else 'red' for gap in approach_gaps]
ax2.barh(range(len(approach_gaps)), approach_gaps.values, color=colors_gap, alpha=0.7)
ax2.set_yticks(range(len(approach_gaps)))
ax2.set_yticklabels(approach_gaps.index)
ax2.set_xlabel('Average Gap', fontsize=12)
ax2.set_title('Approach Stability', fontsize=13, weight='bold')
ax2.axvline(0.2, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Safe Threshold')
ax2.legend()
ax2.grid(alpha=0.3)

# 5.3 복잡도별 Gap
ax3 = fig.add_subplot(gs[1, 1])
complexity_gaps = gap_df.groupby('complexity')['gap'].mean()
complexity_gaps = complexity_gaps.reindex(['Low', 'Medium', 'High', 'Very High'])
colors_complexity = ['green', 'yellow', 'orange', 'red']
ax3.bar(range(len(complexity_gaps)), complexity_gaps.values,
        color=colors_complexity, alpha=0.7)
ax3.set_xticks(range(len(complexity_gaps)))
ax3.set_xticklabels(complexity_gaps.index, rotation=45)
ax3.set_ylabel('Average Gap', fontsize=12)
ax3.set_title('Complexity Impact', fontsize=13, weight='bold')
ax3.grid(alpha=0.3)

# 5.4 OOD 성능 저하
ax4 = fig.add_subplot(gs[1, 2])
ood_degradation = logo_mean - np.mean(gkf_scores)
ood_pct = (logo_mean / np.mean(gkf_scores) - 1) * 100
bar_color = 'green' if abs(ood_degradation) < 0.1 else 'orange' if abs(ood_degradation) < 0.3 else 'red'
ax4.bar(['Zone 6x6'], [ood_degradation], color=bar_color, alpha=0.7, width=0.3)
ax4.axhline(0, color='black', linewidth=2)
ax4.set_ylabel('LOGO - GKF CV', fontsize=12)
ax4.set_title('OOD Performance Degradation', fontsize=13, weight='bold')
ax4.set_ylim([-0.5, 0.5])
ax4.grid(alpha=0.3)
ax4.text(0, ood_degradation + 0.02, f'{ood_degradation:+.4f}\n({ood_pct:+.2f}%)',
        ha='center', fontsize=14, weight='bold')

# 5.5 핵심 메트릭 테이블
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('tight')
ax5.axis('off')

metrics_data = [
    ['Metric', 'Zone 6x6', 'LSTM v3', 'Phase 2', 'Interpretation'],
    ['CV Score', '16.34', '14.36', '15.38', 'Lower = Better prediction'],
    ['Public Score', '16.36', '17.29', '16.81', 'Lower = Better ranking'],
    ['CV-Public Gap', '0.028', '2.93', '1.43', 'Lower = More stable'],
    ['Game CV%', '9.47%', 'N/A', 'N/A', '< 10% = Very stable'],
    ['OOD Degradation', '-0.044', 'N/A', 'N/A', '~ 0 = OOD robust'],
    ['Parameters', '288', '50,000', '~1,000', 'Fewer = Less overfitting'],
    ['Status', 'BEST', 'FAILED', 'FAILED', 'Simplicity wins!'],
]

table = ax5.table(cellText=metrics_data, cellLoc='center', loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 헤더 스타일
for i in range(5):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Zone 6x6 행 강조
for i in range(5):
    table[(1, i)].set_facecolor('#90EE90')
    table[(2, i)].set_facecolor('#90EE90')
    table[(3, i)].set_facecolor('#90EE90')
    table[(4, i)].set_facecolor('#90EE90')
    table[(5, i)].set_facecolor('#90EE90')
    table[(6, i)].set_facecolor('#90EE90')
    table[(7, i)].set_facecolor('#90EE90')

plt.savefig(PLOTS_DIR / '5_executive_summary.png', dpi=300, bbox_inches='tight')
print(f"  저장: {PLOTS_DIR / '5_executive_summary.png'}")

# =============================================================================
# 완료
# =============================================================================
print("\n" + "=" * 80)
print("시각화 완료!")
print("=" * 80)
print(f"\n생성된 플롯:")
print(f"  1. {PLOTS_DIR / '1_game_cv_variability.png'}")
print(f"  2. {PLOTS_DIR / '2_cv_public_gap_comparison.png'}")
print(f"  3. {PLOTS_DIR / '3_complexity_vs_gap.png'}")
print(f"  4. {PLOTS_DIR / '4_logo_vs_gkf.png'}")
print(f"  5. {PLOTS_DIR / '5_executive_summary.png'}")
print("\n" + "=" * 80)
