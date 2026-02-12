"""
CatBoost Hyperparameter Tuning (로컬 실험)

Ultrathink 2025-12-16:
- Batch 1: CV 15.79 ± 0.27 (16 features)
- Tuning: iterations, depth, learning_rate
- 목표: CV 15.65-15.70 (△0.1-0.15)

⚠️ 로컬 실험만! 제출 안 함!
"""

import sys
sys.path.append('../../utils')

import pandas as pd
import numpy as np
import catboost as cb
import time
import warnings
warnings.filterwarnings('ignore')

from fast_experiment_v2 import FastExperimentV2

print("=" * 80)
print("CatBoost Hyperparameter Tuning (로컬 실험)")
print("=" * 80)
print("\nBatch 1 Baseline: CV 15.79 ± 0.27")
print("목표: CV 15.65-15.70 (△0.1-0.15)")
print("\n⚠️ 로컬 실험만! 제출 안 함!")

# =============================================================================
# Setup
# =============================================================================
exp = FastExperimentV2(sample_frac=1.0, n_folds=5, random_state=42)

# Load & Features
train_df = exp.load_data(train_path='../../../train.csv', sample=False)
train_df = exp.create_features(train_df)
X, y, groups, feature_cols = exp.prepare_data(train_df)

# =============================================================================
# Grid Search
# =============================================================================
print("\n" + "=" * 80)
print("Grid Search")
print("=" * 80)

# Parameter grid (간단하게)
param_grid = {
    'iterations': [150, 200, 300],
    'depth': [7, 8],
    'learning_rate': [0.05, 0.08]
}

print(f"\n파라미터 조합:")
print(f"  iterations: {param_grid['iterations']}")
print(f"  depth: {param_grid['depth']}")
print(f"  learning_rate: {param_grid['learning_rate']}")
print(f"  총 조합: {len(param_grid['iterations']) * len(param_grid['depth']) * len(param_grid['learning_rate'])}개")

baseline_params = {
    'subsample': 0.8,
    'random_state': 42,
    'verbose': 0
}

results = []
best_cv = float('inf')
best_params = None

total_combinations = (
    len(param_grid['iterations']) *
    len(param_grid['depth']) *
    len(param_grid['learning_rate'])
)

print(f"\n시작...")
start_all = time.time()

combination = 0
for iterations in param_grid['iterations']:
    for depth in param_grid['depth']:
        for lr in param_grid['learning_rate']:
            combination += 1

            params = {
                **baseline_params,
                'iterations': iterations,
                'depth': depth,
                'learning_rate': lr
            }

            print(f"\n[{combination}/{total_combinations}] iter={iterations}, depth={depth}, lr={lr}")

            start = time.time()

            model_x = cb.CatBoostRegressor(**params)
            model_y = cb.CatBoostRegressor(**params)

            cv_mean, cv_std, fold_scores = exp.run_cv(
                model_x, model_y, X, y, groups,
                model_name=f'CatBoost (iter={iterations}, depth={depth}, lr={lr})'
            )

            runtime = time.time() - start

            results.append({
                'iterations': iterations,
                'depth': depth,
                'learning_rate': lr,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'runtime': runtime
            })

            if cv_mean < best_cv:
                best_cv = cv_mean
                best_params = params
                print(f"  ✅ New Best! CV {cv_mean:.4f}")

            print(f"  Runtime: {runtime:.1f}s")

total_time = time.time() - start_all

# =============================================================================
# 결과 요약
# =============================================================================
print("\n" + "=" * 80)
print("튜닝 결과")
print("=" * 80)

# Sort by CV
results_sorted = sorted(results, key=lambda x: x['cv_mean'])

print(f"\n{'Rank':<5} {'Iter':<6} {'Depth':<6} {'LR':<6} {'CV':<18} {'Runtime':<10}")
print("-" * 65)

for i, r in enumerate(results_sorted[:10], 1):  # Top 10
    marker = "⭐" if i == 1 else ""
    print(f"{i:<5} {r['iterations']:<6} {r['depth']:<6} {r['learning_rate']:<6.2f} "
          f"{r['cv_mean']:.4f}±{r['cv_std']:.4f}  {r['runtime']:<10.1f}s {marker}")

# Best result
print(f"\n" + "=" * 80)
print("최적 파라미터")
print("=" * 80)

baseline_cv = 15.79
improvement = baseline_cv - best_cv

print(f"\n  Baseline: {baseline_cv:.4f}")
print(f"  Best:     {best_cv:.4f}")
print(f"  개선:     {improvement:+.4f}점")

print(f"\n  최적 파라미터:")
for k, v in best_params.items():
    print(f"    {k}: {v}")

print(f"\n  총 실행 시간: {total_time:.1f}s")

# 누적 개선
print(f"\n" + "=" * 80)
print("누적 개선 추적")
print("=" * 80)

baseline_original = 16.04
cumulative = baseline_original - best_cv

print(f"\nBaseline (Zone 6x6): {baseline_original:.2f}")
print(f"Batch 1:             {baseline_cv:.2f} (△{baseline_original - baseline_cv:+.2f})")
print(f"Batch 1 + Tuning:    {best_cv:.2f} (△{baseline_original - best_cv:+.2f})")
print(f"\n누적 개선: {cumulative:.2f}점")

if improvement > 0:
    print(f"\n✅ Tuning 성공! {improvement:.2f}점 개선")

    if cumulative >= 0.3:
        print(f"\n🎯 제출 기준 달성! (△0.3 이상)")
        print(f"   Public 예상: {best_cv + 0.35:.2f} (Gap 0.35 가정)")
        print(f"   vs 현재 Best (16.14): {16.14 - (best_cv + 0.35):.2f}점 개선")
        print(f"\n다음 단계:")
        print(f"  1. Best 파라미터로 재학습")
        print(f"  2. Test 예측")
        print(f"  3. 제출!")
    else:
        print(f"\n⏳ 제출 기준 미달 (△0.3 필요, 현재 △{cumulative:.2f})")
        print(f"   추가 필요: {0.3 - cumulative:.2f}점")
        print(f"\n다음 단계:")
        print(f"  1. Ensemble 시도")
        print(f"  2. 또는 더 많은 파라미터 탐색")
else:
    print(f"\n⚠️ Tuning 효과 없음 ({improvement:.2f}점)")

# Save best params
import json
from pathlib import Path

best_result = {
    'params': best_params,
    'cv_mean': float(best_cv),
    'cv_std': float(results_sorted[0]['cv_std']),
    'improvement': float(improvement),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

output_file = Path('../../../logs/best_params.json')
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(best_result, f, indent=2, ensure_ascii=False)

print(f"\n✅ 최적 파라미터 저장: {output_file}")

print("\n" + "=" * 80)
print("✅ Hyperparameter Tuning 완료!")
print("=" * 80)
