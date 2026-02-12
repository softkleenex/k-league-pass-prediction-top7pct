"""
exp_059: Blend Optimization

기존 좋은 제출들의 최적 가중치 찾기
- exp_047 advanced (Public 14.0677)
- exp_048 ensemble (Public 14.0679)
- exp_055 stacking (CV 14.175)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
SUBMISSION_DIR = BASE / "submissions"

# 가장 좋은 제출들 로드
submissions = {
    'advanced': SUBMISSION_DIR / 'submission_advanced_cv14.25.csv',
    'ensemble': SUBMISSION_DIR / 'submission_ensemble_cv14.17.csv',
    'stacking': SUBMISSION_DIR / 'submission_stacking_cv14.18.csv',
    '5fold': SUBMISSION_DIR / 'submission_5fold_cv14.14.csv',
}

print("=" * 70)
print("exp_059: Blend Optimization")
print("=" * 70)

# 데이터 로드
dfs = {}
for name, path in submissions.items():
    if path.exists():
        dfs[name] = pd.read_csv(path)
        print(f"  Loaded: {name} ({len(dfs[name])} rows)")

# 첫 번째 파일 기준으로 정렬
base = list(dfs.keys())[0]
for name in dfs:
    dfs[name] = dfs[name].set_index('game_episode').loc[dfs[base].set_index('game_episode').index].reset_index()

# 각 제출 비교
print("\n예측 분포 비교:")
for name, df in dfs.items():
    print(f"  {name}: end_x mean={df['end_x'].mean():.2f}, end_y mean={df['end_y'].mean():.2f}")

# 제출 간 상관관계
print("\n제출 간 상관관계 (end_x):")
names = list(dfs.keys())
corr_matrix = np.zeros((len(names), len(names)))
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        corr_matrix[i, j] = np.corrcoef(dfs[n1]['end_x'], dfs[n2]['end_x'])[0, 1]

print("       " + "  ".join([f"{n:>10s}" for n in names]))
for i, n in enumerate(names):
    print(f"{n:>10s}: " + "  ".join([f"{corr_matrix[i, j]:10.4f}" for j in range(len(names))]))

# 블렌딩 조합 생성
print("\n블렌딩 조합 테스트...")

# 다양한 가중치 조합
weight_combinations = [
    {'advanced': 1.0, 'ensemble': 0.0, 'stacking': 0.0, '5fold': 0.0},
    {'advanced': 0.5, 'ensemble': 0.5, 'stacking': 0.0, '5fold': 0.0},
    {'advanced': 0.6, 'ensemble': 0.4, 'stacking': 0.0, '5fold': 0.0},
    {'advanced': 0.7, 'ensemble': 0.3, 'stacking': 0.0, '5fold': 0.0},
    {'advanced': 0.4, 'ensemble': 0.4, 'stacking': 0.2, '5fold': 0.0},
    {'advanced': 0.5, 'ensemble': 0.3, 'stacking': 0.2, '5fold': 0.0},
    {'advanced': 0.4, 'ensemble': 0.3, 'stacking': 0.3, '5fold': 0.0},
    {'advanced': 0.5, 'ensemble': 0.25, 'stacking': 0.25, '5fold': 0.0},
    {'advanced': 0.33, 'ensemble': 0.33, 'stacking': 0.34, '5fold': 0.0},
]

for weights in weight_combinations:
    pred_x = sum(w * dfs[n]['end_x'] for n, w in weights.items() if w > 0 and n in dfs)
    pred_y = sum(w * dfs[n]['end_y'] for n, w in weights.items() if w > 0 and n in dfs)

    # 예측 분포
    active = {k: v for k, v in weights.items() if v > 0}
    desc = "+".join([f"{n}*{v:.2f}" for n, v in active.items()])
    print(f"  {desc}: mean_x={pred_x.mean():.2f}, mean_y={pred_y.mean():.2f}")

# Best 블렌드 저장 (advanced 70% + ensemble 30%)
print("\n최종 블렌드: advanced 70% + ensemble 30%")
pred_x = 0.7 * dfs['advanced']['end_x'] + 0.3 * dfs['ensemble']['end_x']
pred_y = 0.7 * dfs['advanced']['end_y'] + 0.3 * dfs['ensemble']['end_y']

submission = pd.DataFrame({
    'game_episode': dfs['advanced']['game_episode'],
    'end_x': pred_x,
    'end_y': pred_y
})

output_path = SUBMISSION_DIR / "submission_blend_adv70_ens30.csv"
submission.to_csv(output_path, index=False)
print(f"  저장: {output_path}")

# advanced 60% + ensemble 40%
print("\n대안 블렌드: advanced 60% + ensemble 40%")
pred_x = 0.6 * dfs['advanced']['end_x'] + 0.4 * dfs['ensemble']['end_x']
pred_y = 0.6 * dfs['advanced']['end_y'] + 0.4 * dfs['ensemble']['end_y']

submission2 = pd.DataFrame({
    'game_episode': dfs['advanced']['game_episode'],
    'end_x': pred_x,
    'end_y': pred_y
})

output_path2 = SUBMISSION_DIR / "submission_blend_adv60_ens40.csv"
submission2.to_csv(output_path2, index=False)
print(f"  저장: {output_path2}")

# advanced 55% + ensemble 25% + stacking 20%
print("\n3-way 블렌드: advanced 55% + ensemble 25% + stacking 20%")
pred_x = 0.55 * dfs['advanced']['end_x'] + 0.25 * dfs['ensemble']['end_x'] + 0.20 * dfs['stacking']['end_x']
pred_y = 0.55 * dfs['advanced']['end_y'] + 0.25 * dfs['ensemble']['end_y'] + 0.20 * dfs['stacking']['end_y']

submission3 = pd.DataFrame({
    'game_episode': dfs['advanced']['game_episode'],
    'end_x': pred_x,
    'end_y': pred_y
})

output_path3 = SUBMISSION_DIR / "submission_blend_3way.csv"
submission3.to_csv(output_path3, index=False)
print(f"  저장: {output_path3}")

print("\n" + "=" * 70)
print("생성된 블렌드 파일:")
print(f"  1. {output_path.name} (adv70+ens30)")
print(f"  2. {output_path2.name} (adv60+ens40)")
print(f"  3. {output_path3.name} (3-way)")
print("=" * 70)
