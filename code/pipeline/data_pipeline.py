"""
데이터 파이프라인

데이터 로드, 캐싱, 전처리 관리
"""

import pandas as pd
import pickle
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict
import time


class DataPipeline:
    """데이터 로드 및 캐싱 관리"""

    def __init__(self, data_dir: str = "/mnt/c/LSJ/dacon/dacon/kleague-algorithm"):
        """
        Args:
            data_dir: 프로젝트 루트 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        # 데이터 파일 경로
        self.train_path = self.data_dir / "train.csv"
        self.test_path = self.data_dir / "test.csv"

        self._validate_data_files()

    def _validate_data_files(self):
        """데이터 파일 존재 확인"""
        if not self.train_path.exists():
            raise FileNotFoundError(f"train.csv를 찾을 수 없습니다: {self.train_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"test.csv를 찾을 수 없습니다: {self.test_path}")

    def load_data(
        self,
        use_cache: bool = True,
        sample_rate: float = 1.0,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        학습/테스트 데이터 로드

        Args:
            use_cache: 캐시 사용 여부
            sample_rate: 샘플링 비율 (0.1 = 10% 샘플)
            random_state: 랜덤 시드

        Returns:
            (train_df, test_df)
        """
        start_time = time.time()

        # 캐시 키 생성
        cache_key = self._generate_cache_key(sample_rate, random_state)
        cache_path = self.cache_dir / cache_key

        # 캐시 확인
        if use_cache and cache_path.exists():
            print(f"[DataPipeline] 캐시에서 로드: {cache_path.name}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            train_df, test_df = data['train'], data['test']
            print(f"[DataPipeline] 로드 완료: {time.time() - start_time:.2f}초")
            self._print_data_info(train_df, test_df)
            return train_df, test_df

        # 원본 CSV 로드
        print(f"[DataPipeline] 원본 CSV 로드...")
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # 샘플링
        if sample_rate < 1.0:
            print(f"[DataPipeline] {sample_rate*100:.1f}% 샘플링...")
            train_df = self._sample_episodes(train_df, sample_rate, random_state)

        # 기본 전처리
        train_df = self._preprocess(train_df)
        test_df = self._preprocess(test_df)

        # 캐시 저장
        if use_cache:
            cache_data = {'train': train_df, 'test': test_df}
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[DataPipeline] 캐시 저장: {cache_path.name}")

        print(f"[DataPipeline] 로드 완료: {time.time() - start_time:.2f}초")
        self._print_data_info(train_df, test_df)

        return train_df, test_df

    def _generate_cache_key(self, sample_rate: float, random_state: int) -> str:
        """캐시 키 생성"""
        # 파일 수정 시간 포함 (데이터 변경 감지)
        train_mtime = self.train_path.stat().st_mtime
        test_mtime = self.test_path.stat().st_mtime

        key_str = f"sample{int(sample_rate*100)}_rs{random_state}_train{train_mtime}_test{test_mtime}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:8]

        return f"data_{key_str}_{key_hash}.pkl"

    def _sample_episodes(
        self,
        df: pd.DataFrame,
        sample_rate: float,
        random_state: int
    ) -> pd.DataFrame:
        """Episode 단위 샘플링"""
        episodes = df['episode_id'].unique()
        n_samples = max(1, int(len(episodes) * sample_rate))

        sampled_episodes = pd.Series(episodes).sample(
            n=n_samples,
            random_state=random_state
        ).values

        sampled_df = df[df['episode_id'].isin(sampled_episodes)].copy()
        print(f"  - Episodes: {len(episodes):,} → {len(sampled_episodes):,}")

        return sampled_df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리"""
        # 정렬 (episode_id, pass_number)
        df = df.sort_values(['episode_id', 'pass_number']).reset_index(drop=True)

        # 데이터 타입 최적화
        df['episode_id'] = df['episode_id'].astype('int32')
        df['pass_number'] = df['pass_number'].astype('int16')

        for col in ['start_x', 'start_y', 'end_x', 'end_y']:
            if col in df.columns:
                df[col] = df[col].astype('float32')

        return df

    def _print_data_info(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """데이터 정보 출력"""
        print(f"  - Train: {len(train_df):,} rows, {train_df['episode_id'].nunique():,} episodes")
        print(f"  - Test: {len(test_df):,} rows, {test_df['episode_id'].nunique():,} episodes")

    def prepare_episode_data(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Episode별 데이터 준비

        Args:
            df: 전체 데이터프레임

        Returns:
            {episode_id: episode_df} 딕셔너리
        """
        episodes = {}
        for episode_id, group in df.groupby('episode_id'):
            episodes[episode_id] = group.sort_values('pass_number').reset_index(drop=True)

        print(f"[DataPipeline] Episode 준비 완료: {len(episodes):,}개")
        return episodes

    def get_last_pass_targets(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        각 Episode의 마지막 패스 좌표 추출

        Args:
            df: 전체 데이터프레임

        Returns:
            (last_x, last_y) Series
        """
        last_pass = df.groupby('episode_id').last()
        return last_pass['end_x'], last_pass['end_y']

    def clear_cache(self, sample_rate: Optional[float] = None):
        """
        캐시 삭제

        Args:
            sample_rate: None이면 전체 삭제, 지정하면 해당 샘플링 비율만 삭제
        """
        if sample_rate is None:
            # 전체 삭제
            for cache_file in self.cache_dir.glob("data_*.pkl"):
                cache_file.unlink()
            print(f"[DataPipeline] 전체 캐시 삭제 완료")
        else:
            # 특정 샘플링 비율만 삭제
            pattern = f"data_sample{int(sample_rate*100)}_*.pkl"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
            print(f"[DataPipeline] 캐시 삭제 완료: {pattern}")


def test_data_pipeline():
    """데이터 파이프라인 테스트"""
    print("=" * 80)
    print("DataPipeline 테스트")
    print("=" * 80)

    pipeline = DataPipeline()

    # 1. 전체 데이터 로드 (캐시 생성)
    print("\n[Test 1] 전체 데이터 로드 (캐시 생성)")
    train_df, test_df = pipeline.load_data(use_cache=True, sample_rate=1.0)

    # 2. 10% 샘플 로드 (캐시 생성)
    print("\n[Test 2] 10% 샘플 로드 (캐시 생성)")
    train_sample, test_sample = pipeline.load_data(use_cache=True, sample_rate=0.1)

    # 3. 캐시에서 재로드 (빠름)
    print("\n[Test 3] 캐시에서 재로드")
    train_cached, test_cached = pipeline.load_data(use_cache=True, sample_rate=1.0)

    # 4. Episode 데이터 준비
    print("\n[Test 4] Episode 데이터 준비")
    episodes = pipeline.prepare_episode_data(train_sample)
    print(f"  - Episode 1 샘플:\n{episodes[list(episodes.keys())[0]].head()}")

    # 5. 마지막 패스 타겟 추출
    print("\n[Test 5] 마지막 패스 타겟 추출")
    last_x, last_y = pipeline.get_last_pass_targets(train_sample)
    print(f"  - Shape: {last_x.shape}")
    print(f"  - Sample: X={last_x.iloc[0]:.2f}, Y={last_y.iloc[0]:.2f}")

    print("\n[Test] 완료!")


if __name__ == "__main__":
    test_data_pipeline()
