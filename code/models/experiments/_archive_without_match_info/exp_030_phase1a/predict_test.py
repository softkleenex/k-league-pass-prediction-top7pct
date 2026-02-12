"""
Phase 1-A Test Prediction Script

================================================================================
목표: 학습된 모델로 test 데이터 예측 및 submission 생성
================================================================================

워크플로우:
1. 학습된 모델 로드 (model_x.cbm, model_y.cbm)
2. Test 데이터 로드 (data/test.csv)
3. 각 episode별 데이터 로드 (data/test/{game_id}/)
4. FastExperimentPhase1A로 피처 생성
5. 마지막 패스만 추출 & 예측
6. 범위 클리핑 (0-105, 0-68)
7. Submission 생성 및 저장

작성일: 2025-12-17
작성자: Phase 1-A Team
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# CatBoost import
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("ERROR: CatBoost not installed. Install with: pip install catboost")
    sys.exit(1)

# Add utils to path
UTILS_PATH = Path(__file__).parent.parent.parent.parent / 'utils'
sys.path.insert(0, str(UTILS_PATH))

from fast_experiment_phase1a import FastExperimentPhase1A


class Phase1APredictor:
    """
    Phase 1-A 모델을 사용한 Test 예측

    주요 기능:
    - 학습된 모델 로드
    - Test 데이터 처리
    - 피처 생성 및 예측
    - Submission 생성
    """

    def __init__(self, exp_dir: Path, data_dir: Path):
        """
        초기화

        Args:
            exp_dir: 모델이 저장된 디렉토리 (exp_030_phase1a)
            data_dir: 데이터 디렉토리 (data)
        """
        self.exp_dir = Path(exp_dir)
        self.data_dir = Path(data_dir)
        self.submissions_dir = self.data_dir.parent / 'submissions'

        # 모델 경로
        self.model_x_path = self.exp_dir / 'model_x.cbm'
        self.model_y_path = self.exp_dir / 'model_y.cbm'

        # 데이터 경로
        self.test_csv_path = self.data_dir / 'test.csv'
        self.test_dir = self.data_dir / 'test'
        self.sample_submission_path = self.data_dir / 'sample_submission.csv'

        # Submission 디렉토리 생성
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print("Phase 1-A Predictor 초기화")
        print(f"{'='*80}")
        print(f"  Experiment dir:  {self.exp_dir}")
        print(f"  Data dir:        {self.data_dir}")
        print(f"  Submissions dir: {self.submissions_dir}")

    def load_models(self) -> tuple:
        """
        학습된 CatBoost 모델 로드

        Returns:
            (model_x, model_y): 로드된 모델 튜플
        """
        print(f"\n{'='*80}")
        print("1. 모델 로드")
        print(f"{'='*80}")

        if not self.model_x_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_x_path}")

        if not self.model_y_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_y_path}")

        print(f"  Loading model_x.cbm...", end='', flush=True)
        model_x = CatBoostRegressor()
        model_x.load_model(str(self.model_x_path))
        print(" ✓")

        print(f"  Loading model_y.cbm...", end='', flush=True)
        model_y = CatBoostRegressor()
        model_y.load_model(str(self.model_y_path))
        print(" ✓")

        print(f"\n  ✅ 모델 로드 완료")

        return model_x, model_y

    def load_test_data(self) -> pd.DataFrame:
        """
        Test 데이터 로드

        Returns:
            test_df: game_episode, path 포함 DataFrame
        """
        print(f"\n{'='*80}")
        print("2. Test 데이터 로드")
        print(f"{'='*80}")

        if not self.test_csv_path.exists():
            raise FileNotFoundError(f"Test CSV not found: {self.test_csv_path}")

        test_df = pd.read_csv(self.test_csv_path)
        print(f"  로드된 데이터: {len(test_df):,}개 episode")
        print(f"  컬럼: {list(test_df.columns)}")

        return test_df

    def load_episode_data(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Test 데이터의 각 episode별 CSV 파일 로드

        Args:
            test_df: game_episode, path 포함 DataFrame

        Returns:
            combined_df: 전체 episode 데이터 결합
        """
        print(f"\n{'='*80}")
        print("3. Episode별 데이터 로드")
        print(f"{'='*80}")

        all_episodes = []
        failed_count = 0

        for idx, row in test_df.iterrows():
            game_id = row['game_id']
            game_episode = row['game_episode']

            # 경로 구성 (test.csv의 path는 상대경로일 수 있음)
            episode_path = self.test_dir / game_id / f'{game_episode}.csv'

            # 파일이 없으면 다른 경로 시도
            if not episode_path.exists():
                # data/test/{game_id}/{game_episode}.csv 형식
                alternative_path = self.data_dir / 'test' / str(game_id) / f'{game_episode}.csv'
                if alternative_path.exists():
                    episode_path = alternative_path
                else:
                    print(f"  WARNING: Episode 파일 없음: {game_episode}")
                    failed_count += 1
                    continue

            try:
                episode_data = pd.read_csv(episode_path)
                episode_data['game_episode'] = game_episode
                all_episodes.append(episode_data)
            except Exception as e:
                print(f"  ERROR 로드 실패: {game_episode} - {str(e)}")
                failed_count += 1

        if not all_episodes:
            raise ValueError("No episode data loaded!")

        combined_df = pd.concat(all_episodes, ignore_index=True)

        print(f"\n  로드 완료:")
        print(f"    - 성공: {len(all_episodes):,}개 episode")
        print(f"    - 실패: {failed_count:,}개 episode")
        print(f"    - 총 패스: {len(combined_df):,}개")
        print(f"    - 컬럼: {list(combined_df.columns)}")

        return combined_df

    def create_features(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        FastExperimentPhase1A를 사용한 피처 생성

        Args:
            test_df: Test 데이터

        Returns:
            test_df: 피처가 추가된 DataFrame
        """
        print(f"\n{'='*80}")
        print("4. 피처 생성 (FastExperimentPhase1A)")
        print(f"{'='*80}")

        # FastExperimentPhase1A 인스턴스 생성
        exp = FastExperimentPhase1A(sample_frac=1.0, n_folds=3, random_state=42)

        # 피처 생성 (상세 로그 출력)
        test_df = exp.create_features(test_df)

        return test_df

    def prepare_test_data(self, test_df: pd.DataFrame) -> tuple:
        """
        마지막 패스 추출 & Feature 분리

        Args:
            test_df: 피처가 생성된 DataFrame

        Returns:
            (X, game_episodes, feature_cols): Feature matrix, episode 리스트, feature 이름
        """
        print(f"\n{'='*80}")
        print("5. Test 데이터 준비")
        print(f"{'='*80}")

        # 각 episode의 마지막 패스만 추출
        test_last = test_df.groupby('game_episode').last().reset_index()
        print(f"  마지막 패스 추출: {len(test_last):,}개 episode")

        # Feature columns (Phase 1-A: 21개)
        feature_cols = [
            # === 기존 피처 (16개) ===
            'start_x', 'start_y',
            'zone_x', 'zone_y',
            'direction', 'prev_dx', 'prev_dy',
            'goal_distance', 'goal_angle',
            'period_id', 'time_seconds', 'time_left',
            'pass_count',
            'is_home_encoded', 'type_encoded', 'result_encoded',

            # === Phase 1-A 신규 피처 (5개) ===
            'is_final_team',
            'team_possession_pct',
            'team_switches',
            'game_clock_min',
            'final_poss_len'
        ]

        # Feature matrix 생성
        X = test_last[feature_cols].values

        # game_episode 보존
        game_episodes = test_last['game_episode'].values

        print(f"\n  데이터 형상:")
        print(f"    X: {X.shape} (episodes, features)")
        print(f"    episodes: {len(game_episodes)}")
        print(f"  피처 수: {len(feature_cols)}개")

        return X, game_episodes, feature_cols

    def predict(self, model_x, model_y, X: np.ndarray) -> np.ndarray:
        """
        Test 데이터 예측

        Args:
            model_x: X 좌표 모델
            model_y: Y 좌표 모델
            X: Feature matrix

        Returns:
            predictions: (n_samples, 2) - [end_x, end_y]
        """
        print(f"\n{'='*80}")
        print("6. 예측 수행")
        print(f"{'='*80}")

        print(f"  X 좌표 예측 중...", end='', flush=True)
        start = time.time()
        pred_x = model_x.predict(X)
        print(f" 완료 ({time.time()-start:.1f}s)")

        print(f"  Y 좌표 예측 중...", end='', flush=True)
        start = time.time()
        pred_y = model_y.predict(X)
        print(f" 완료 ({time.time()-start:.1f}s)")

        # 범위 클리핑 (0-105, 0-68)
        pred_x = np.clip(pred_x, 0, 105)
        pred_y = np.clip(pred_y, 0, 68)

        predictions = np.column_stack([pred_x, pred_y])

        print(f"\n  예측 결과:")
        print(f"    - 총 예측: {len(predictions):,}개")
        print(f"    - X 범위: [{pred_x.min():.2f}, {pred_x.max():.2f}]")
        print(f"    - Y 범위: [{pred_y.min():.2f}, {pred_y.max():.2f}]")

        return predictions

    def create_submission(self, game_episodes: np.ndarray,
                         predictions: np.ndarray, cv_score: float = None) -> str:
        """
        Submission CSV 생성

        Args:
            game_episodes: game_episode 배열
            predictions: (n_samples, 2) 예측 결과
            cv_score: CV 점수 (선택사항, 파일명에 포함)

        Returns:
            submission_path: 저장된 submission 파일 경로
        """
        print(f"\n{'='*80}")
        print("7. Submission 생성")
        print(f"{'='*80}")

        # Submission DataFrame 생성
        submission_df = pd.DataFrame({
            'game_episode': game_episodes,
            'end_x': predictions[:, 0],
            'end_y': predictions[:, 1]
        })

        print(f"  데이터 형상: {submission_df.shape}")
        print(f"  샘플:\n{submission_df.head()}")

        # 파일명 생성
        if cv_score is not None:
            # CV 점수를 파일명에 포함 (예: submission_phase1a_cv15.95.csv)
            cv_str = f"{cv_score:.2f}".replace('.', '_')
            submission_filename = f'submission_phase1a_cv{cv_str}.csv'
        else:
            # CV 점수 없이 타임스탬프로 파일명 생성
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            submission_filename = f'submission_phase1a_{timestamp}.csv'

        submission_path = self.submissions_dir / submission_filename

        # 저장
        submission_df.to_csv(submission_path, index=False)

        print(f"\n  ✅ Submission 저장:")
        print(f"    경로: {submission_path}")
        print(f"    파일명: {submission_filename}")
        print(f"    파일 크기: {submission_path.stat().st_size / 1024:.1f} KB")

        return str(submission_path)

    def run(self, cv_score: float = None) -> dict:
        """
        전체 예측 파이프라인 실행

        Args:
            cv_score: CV 점수 (선택사항)

        Returns:
            results: 예측 결과 요약 dict
        """
        start_time = time.time()

        try:
            # 1. 모델 로드
            model_x, model_y = self.load_models()

            # 2. Test 데이터 로드
            test_csv = self.load_test_data()

            # 3. Episode 데이터 로드
            test_df = self.load_episode_data(test_csv)

            # 4. 피처 생성
            test_df = self.create_features(test_df)

            # 5. Test 데이터 준비
            X, game_episodes, feature_cols = self.prepare_test_data(test_df)

            # 6. 예측
            predictions = self.predict(model_x, model_y, X)

            # 7. Submission 생성
            submission_path = self.create_submission(game_episodes, predictions, cv_score)

            elapsed_time = time.time() - start_time

            # 결과 요약
            results = {
                'status': 'success',
                'submission_path': submission_path,
                'n_predictions': len(predictions),
                'n_features': len(feature_cols),
                'elapsed_time': elapsed_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            print(f"\n{'='*80}")
            print("✅ 예측 완료!")
            print(f"{'='*80}")
            print(f"  총 실행 시간: {elapsed_time:.1f}초")
            print(f"  예측 수: {len(predictions):,}개")
            print(f"  Submission: {submission_path}")

            return results

        except Exception as e:
            elapsed_time = time.time() - start_time

            print(f"\n{'='*80}")
            print("❌ 예측 실패!")
            print(f"{'='*80}")
            print(f"  에러: {str(e)}")

            results = {
                'status': 'error',
                'error': str(e),
                'elapsed_time': elapsed_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            raise


# ================================================================================
# 메인 실행
# ================================================================================

if __name__ == '__main__':
    # 경로 설정
    SCRIPT_DIR = Path(__file__).parent
    EXP_DIR = SCRIPT_DIR
    DATA_DIR = SCRIPT_DIR.parent.parent.parent / 'data'

    print(f"\n{'='*80}")
    print("Phase 1-A Test Prediction")
    print(f"{'='*80}")
    print(f"  Script dir: {SCRIPT_DIR}")
    print(f"  Exp dir:    {EXP_DIR}")
    print(f"  Data dir:   {DATA_DIR}")

    # Predictor 생성 및 실행
    predictor = Phase1APredictor(exp_dir=EXP_DIR, data_dir=DATA_DIR)

    # CV 점수가 있으면 전달 (선택사항)
    # 예: cv_score=15.95
    results = predictor.run(cv_score=None)

    # 결과 출력
    print(f"\n{'='*80}")
    print("결과 요약")
    print(f"{'='*80}")
    for key, value in results.items():
        print(f"  {key}: {value}")

    print(f"\n{'='*80}")
    print("다음 단계:")
    print(f"{'='*80}")
    print(f"  1. DACON에서 제출 파일 확인")
    print(f"  2. 리더보드에서 순위 확인")
    print(f"  3. 결과를 SUBMISSION_LOG.md에 기록")
    print(f"\nSubmission 파일:")
    print(f"  {results['submission_path']}")
