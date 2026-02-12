"""
Phase 1-A 실험 실행 스크립트

목표:
  1. FastExperimentPhase1A 초기화
  2. 전체 데이터 로드 및 피처 생성
  3. 3-Fold GroupKFold CV 실행
  4. CatBoost 모델로 평가
  5. 결과 저장 (cv_results.json)

실행 방법:
  python run_experiment.py [--sample 0.1] [--folds 3]

작성일: 2025-12-17
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor

# Add utils to path
utils_path = str(Path(__file__).parent.parent.parent / 'utils')
sys.path.insert(0, utils_path)

# Import (absolute path)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fast_experiment_phase1a",
    Path(utils_path) / "fast_experiment_phase1a.py"
)
phase1a_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase1a_module)
FastExperimentPhase1A = phase1a_module.FastExperimentPhase1A


def main():
    """메인 실행 함수"""

    import argparse

    parser = argparse.ArgumentParser(description='Phase 1-A 실험 실행')
    parser.add_argument('--sample', type=float, default=1.0,
                        help='데이터 샘플링 비율 (0.0-1.0)')
    parser.add_argument('--folds', type=int, default=3,
                        help='Cross-validation folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data-path', type=str,
                        default='../../../../train.csv',
                        help='학습 데이터 경로')

    args = parser.parse_args()

    # 시간 측정 시작
    start_time = time.time()

    print(f"\n{'='*80}")
    print("Phase 1-A 실험 실행")
    print(f"{'='*80}")
    print(f"  설정:")
    print(f"    샘플링: {args.sample*100:.0f}%")
    print(f"    Folds: {args.folds}")
    print(f"    Random seed: {args.seed}")
    print(f"    데이터 경로: {args.data_path}")

    # 1. 초기화
    exp = FastExperimentPhase1A(
        sample_frac=args.sample,
        n_folds=args.folds,
        random_state=args.seed
    )

    # 2. 데이터 로드
    train_df = exp.load_data(train_path=args.data_path, sample=(args.sample < 1.0))

    # 3. 피처 생성
    train_df = exp.create_features(train_df)

    # 4. 데이터 준비
    X, y, groups, feature_cols = exp.prepare_data(train_df)

    # 5. 모델 설정 (CatBoost)
    print(f"\n{'='*80}")
    print("CatBoost 모델 설정")
    print(f"{'='*80}")

    cb_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3.0,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'verbose': 0,
        'random_state': args.seed
    }

    print(f"  하이퍼파라미터:")
    for key, value in cb_params.items():
        print(f"    {key}: {value}")

    # 6. CV 실행
    model_x = CatBoostRegressor(**cb_params)
    model_y = CatBoostRegressor(**cb_params)

    cv_mean, cv_std, fold_scores = exp.run_cv(
        model_x, model_y, X, y, groups,
        model_name='CatBoost (Phase 1-A)'
    )

    runtime = time.time() - start_time

    # 7. 결과 저장
    print(f"\n{'='*80}")
    print("결과 저장")
    print(f"{'='*80}")

    results = {
        'experiment': 'Phase 1-A',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'CatBoost',
        'cv_mean': float(cv_mean),
        'cv_std': float(cv_std),
        'cv_folds': [float(x) for x in fold_scores],
        'features': {
            'total': len(feature_cols),
            'existing': 16,
            'new': 5,
            'names': feature_cols
        },
        'new_features': [
            'is_final_team',
            'team_possession_pct',
            'team_switches',
            'game_clock_min',
            'final_poss_len'
        ],
        'data': {
            'n_episodes': len(X),
            'sample_frac': args.sample,
            'n_folds': args.folds
        },
        'model_params': cb_params,
        'runtime_seconds': float(runtime),
        'baseline_comparison': {
            'baseline_name': 'catboost_tuned (exp_028)',
            'baseline_cv': 15.60,
            'baseline_std': 0.27,
            'improvement': float(15.60 - cv_mean),
            'improvement_pct': float((15.60 - cv_mean) / 15.60 * 100)
        }
    }

    results_file = Path(__file__).parent / 'cv_results.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 결과 저장: {results_file}")

    # 8. 결과 요약
    print(f"\n{'='*80}")
    print("실행 완료!")
    print(f"{'='*80}")
    print(f"  CV Mean: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Fold scores: {fold_scores}")
    print(f"  Runtime: {runtime:.1f}초")
    print(f"\n  기존 모델 대비:")
    print(f"    개선폭: {15.60 - cv_mean:+.4f}")
    print(f"    개선율: {(15.60 - cv_mean) / 15.60 * 100:+.2f}%")

    if 15.60 - cv_mean > 0.10:
        print(f"\n  평가: 🚀 강력 개선!")
    elif 15.60 - cv_mean > 0.0:
        print(f"\n  평가: ✅ 개선 확인!")
    elif 15.60 - cv_mean > -0.05:
        print(f"\n  평가: ⚠️ 중립")
    else:
        print(f"\n  평가: ❌ 악화 확인")

    print(f"\n다음 단계:")
    print(f"  1. 분석 스크립트 실행:")
    print(f"     python analyze.py")
    print(f"  2. 상세 분석 및 권장사항 확인")
    print(f"  3. 결과가 만족스럽면 최종 모델 학습 및 제출")


if __name__ == '__main__':
    main()
