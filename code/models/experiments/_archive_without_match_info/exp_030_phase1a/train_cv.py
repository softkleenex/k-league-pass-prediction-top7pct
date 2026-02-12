#!/usr/bin/env python3
"""
Phase 1-A 학습 스크립트: 전체 데이터 CV 검증

============================================================================
목표:
  - 전체 데이터 (sample_frac=1.0) 로드
  - 21개 피처 생성 (기존 16개 + Phase 1-A 신규 5개)
  - CatBoost 모델 학습 (best_params 로드)
  - 3-Fold GroupKFold CV 수행
  - 결과 저장 (CV, 모델)

예상 결과:
  - CV: 15.3-15.5점 (기존 15.60 대비 0.1-0.3점 개선)
  - Gap: 0.2 이하 (안정성 향상)
  - Runtime: ~50-60분

작성일: 2025-12-17 03:00
============================================================================
"""

import sys
import os
from pathlib import Path

# Add parent paths - 절대 경로 설정
project_root = Path('/mnt/c/LSJ/dacon/dacon/kleague-algorithm')
code_utils = project_root / 'code' / 'utils'

import pandas as pd
import numpy as np
import json
import time
import warnings
from datetime import datetime
import importlib.util

warnings.filterwarnings('ignore')

# Import FastExperimentPhase1A - 절대 경로로 로드
spec = importlib.util.spec_from_file_location(
    "fast_experiment_phase1a",
    str(code_utils / "fast_experiment_phase1a.py")
)
fast_exp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast_exp_module)
FastExperimentPhase1A = fast_exp_module.FastExperimentPhase1A

try:
    from catboost import CatBoostRegressor
except ImportError:
    print("ERROR: CatBoost not installed!")
    print("Install: pip install catboost")
    sys.exit(1)


# ============================================================================
# 상수 정의
# ============================================================================

PROJECT_ROOT = project_root  # 위에서 정의한 project_root 사용
DATA_PATH = PROJECT_ROOT / 'data' / 'train.csv'
PARAMS_PATH = PROJECT_ROOT / 'logs' / 'best_params.json'
EXP_DIR = Path(__file__).parent
RESULTS_DIR = EXP_DIR / 'results'
MODELS_DIR = EXP_DIR / 'models'

# 하이퍼파라미터
DEFAULT_PARAMS = {
    'subsample': 0.8,
    'random_state': 42,
    'verbose': 0,
    'iterations': 300,
    'depth': 8,
    'learning_rate': 0.05
}

CV_FOLDS = 3
SAMPLE_FRAC = 1.0  # 전체 데이터


# ============================================================================
# 유틸리티 함수
# ============================================================================

def load_best_params(params_path: Path) -> dict:
    """best_params.json 로드"""
    try:
        with open(params_path, 'r') as f:
            data = json.load(f)
        params = data.get('params', DEFAULT_PARAMS)
        print(f"\n✅ best_params 로드 성공")
        print(f"   - 파일: {params_path}")
        print(f"   - CV: {data.get('cv_mean', 'N/A'):.4f}")
        return params
    except FileNotFoundError:
        print(f"\n⚠️  best_params 파일 없음: {params_path}")
        print(f"   기본값 사용")
        return DEFAULT_PARAMS


def ensure_directories():
    """필요한 디렉토리 생성"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n✅ 디렉토리 준비 완료")
    print(f"   - Results: {RESULTS_DIR}")
    print(f"   - Models: {MODELS_DIR}")


def print_section(title: str):
    """섹션 제목 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")


def print_progress(step: int, total: int, message: str):
    """진행상황 출력"""
    pct = int((step / total) * 100)
    bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
    print(f"\n[{pct:3d}%] {bar} | {message}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """Phase 1-A CV 실행"""

    start_time = time.time()
    print_section("Phase 1-A 전체 데이터 CV 실행")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: 디렉토리 준비
    print_progress(1, 6, "디렉토리 준비 중...")
    ensure_directories()

    # Step 2: 파라미터 로드
    print_progress(2, 6, "최적 파라미터 로드 중...")
    params = load_best_params(PARAMS_PATH)
    print(f"\n  로드된 파라미터:")
    for k, v in params.items():
        print(f"    - {k}: {v}")

    # Step 3: FastExperimentPhase1A 초기화 및 데이터 로드
    print_progress(3, 6, "데이터 로드 및 피처 생성 중...")
    print_section("1. 데이터 로드")

    exp = FastExperimentPhase1A(sample_frac=SAMPLE_FRAC, n_folds=CV_FOLDS)

    # 데이터 로드
    print(f"\n📂 데이터 경로: {DATA_PATH}")
    train_df = exp.load_data(train_path=str(DATA_PATH), sample=True)

    # 피처 생성
    train_df = exp.create_features(train_df)

    # 데이터 준비
    X, y, groups, feature_cols = exp.prepare_data(train_df)

    # Step 4: CatBoost 모델 준비
    print_progress(4, 6, "CatBoost 모델 준비 중...")
    print_section("2. CatBoost 모델 생성")

    # 별도의 model_x, model_y 생성
    model_x = CatBoostRegressor(**params)
    model_y = CatBoostRegressor(**params)

    print(f"\n✅ 모델 준비 완료")
    print(f"   - Model X (end_x 예측)")
    print(f"   - Model Y (end_y 예측)")
    print(f"   - Parameters: {params}")

    # Step 5: Cross-Validation
    print_progress(5, 6, "Cross-Validation 수행 중...")
    cv_results = {
        'mean': None,
        'std': None,
        'folds': []
    }

    mean_cv, std_cv, fold_scores = exp.run_cv(
        model_x, model_y, X, y, groups, model_name='CatBoost (Phase 1-A)'
    )

    cv_results['mean'] = float(mean_cv)
    cv_results['std'] = float(std_cv)
    cv_results['folds'] = [float(s) for s in fold_scores]

    # Step 6: 전체 데이터로 최종 모델 학습 및 저장
    print_progress(6, 6, "최종 모델 학습 및 저장 중...")
    print_section("3. 최종 모델 학습")

    print(f"\n전체 데이터로 최종 모델 학습 중...")
    print(f"  - 전체 데이터: {len(X):,} episodes")

    start = time.time()
    model_x.fit(X, y[:, 0])
    model_y.fit(X, y[:, 1])
    train_time = time.time() - start

    print(f"✅ 최종 모델 학습 완료 ({train_time:.1f}s)")

    # 모델 저장
    model_x_path = MODELS_DIR / 'model_x.cbm'
    model_y_path = MODELS_DIR / 'model_y.cbm'

    model_x.save_model(str(model_x_path))
    model_y.save_model(str(model_y_path))

    print(f"\n✅ 모델 저장 완료")
    print(f"   - {model_x_path}")
    print(f"   - {model_y_path}")

    # 결과 저장
    print_section("4. 결과 저장")

    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'Phase 1-A Full Data CV',
        'description': '공유 코드 인사이트 5개 통합 (21개 피처)',
        'data': {
            'train_path': str(DATA_PATH),
            'n_episodes': len(X),
            'sample_frac': SAMPLE_FRAC,
            'n_features': len(feature_cols)
        },
        'cv': cv_results,
        'params': params,
        'features': feature_cols,
        'new_features': [
            'is_final_team',
            'team_possession_pct',
            'team_switches',
            'game_clock_min',
            'final_poss_len'
        ],
        'models': {
            'model_x': str(model_x_path),
            'model_y': str(model_y_path)
        }
    }

    # CV 결과 저장
    cv_results_path = RESULTS_DIR / 'cv_results.json'
    with open(cv_results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 결과 저장 완료")
    print(f"   - {cv_results_path}")

    # Step 7: 최종 결과 요약
    total_time = time.time() - start_time

    print_section("5. 최종 결과")
    print(f"\n📊 CV 성과:")
    print(f"   - Mean CV: {mean_cv:.4f}")
    print(f"   - Std Dev: {std_cv:.4f}")
    print(f"   - Fold 1: {fold_scores[0]:.4f}")
    print(f"   - Fold 2: {fold_scores[1]:.4f}")
    print(f"   - Fold 3: {fold_scores[2]:.4f}")

    print(f"\n📈 개선도:")
    prev_cv = 15.60
    improvement = prev_cv - mean_cv
    pct_improvement = (improvement / prev_cv) * 100
    print(f"   - 이전 CV: {prev_cv:.4f}")
    print(f"   - 현재 CV: {mean_cv:.4f}")
    print(f"   - 개선: {improvement:.4f}점 ({pct_improvement:.2f}%)")

    if improvement > 0:
        print(f"   ✅ 개선 달성!")
    else:
        print(f"   ⚠️  개선 실패")

    print(f"\n⏱️  실행 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_section("✅ Phase 1-A CV 완료!")

    return mean_cv, std_cv, fold_scores


# ============================================================================
# 스크립트 진입점
# ============================================================================

if __name__ == '__main__':
    try:
        mean_cv, std_cv, fold_scores = main()

        # 최종 성공 메시지
        print(f"\n{'='*80}")
        print(f"🎯 최종 CV 점수: {mean_cv:.4f} ± {std_cv:.4f}")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n❌ 오류 발생!")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
