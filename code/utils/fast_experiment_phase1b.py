"""
Phase 1-B: match_info 피처 추가 실험 시스템

================================================================================
목표: Phase 1-A(21개) + match_info 피처 3개 = 총 24개 피처
================================================================================

배경:
- Phase 1-A: 공유 코드 인사이트 5개 통합 (21개 피처)
- Phase 1-B: match_info 추가 인사이트 활용
- 현재 Best: 15.84점 → 목표: 15.3-15.5점

신규 피처 3가지:
================================================================================
1. is_attacking_home (⭐⭐⭐⭐⭐)
   - 공격권 팀이 홈팀인지 여부
   - 홈/어웨이 심리적 압박 반영
   - is_final_team==1 and team_id==home_team_id

2. team_avg_goals_for (⭐⭐⭐⭐)
   - 공격권 팀의 최근 평균 득점
   - 공격력 지표
   - 강팀 vs 약팀 구분

3. team_avg_goals_against (⭐⭐⭐)
   - 공격권 팀의 최근 평균 실점
   - 수비력 지표
   - 안정성 반영
================================================================================

변경사항:
- Phase 1-A: 21개 피처
- Phase 1-B: 21 + 3 = 24개 피처
- GroupKFold 유지 (game_id 기준)
- n_folds: 3 (시간 단축)

예상 효과:
- CV: 15.50 → 15.2-15.4 (0.1-0.3점 개선)
- Gap: 0.20 → 0.15 이하 (안정성 향상)
- Public: 15.60 → 15.3-15.5

작성일: 2025-12-17 03:30
작성자: Phase 1-B Team
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from fast_experiment_phase1a import FastExperimentPhase1A


class FastExperimentPhase1B(FastExperimentPhase1A):
    """
    Phase 1-B 전용 실험 시스템

    Phase 1-A(21개 피처) + match_info 피처 3개 = 총 24개 피처
    """

    def __init__(
        self,
        match_stats_path: str = None,
        match_info_path: str = None,
        sample_frac: float = 0.1,
        n_folds: int = 3,
        random_state: int = 42
    ):
        """
        초기화

        Args:
            match_stats_path: preprocessed_match_stats.csv 경로
            match_info_path: match_info.csv 경로 (home_team_id 정보)
            sample_frac: Episode 샘플링 비율 (0.1 = 10%, 1.0 = 100%)
            n_folds: Cross-validation fold 수 (권장: 3)
            random_state: Random seed (재현성)
        """
        # 부모 클래스 초기화
        super().__init__(
            sample_frac=sample_frac,
            n_folds=n_folds,
            random_state=random_state
        )

        # match_stats 경로 설정
        self.match_stats_path = match_stats_path
        self.match_info_path = match_info_path

        # 데이터 로드 (lazy loading)
        self.match_stats = None
        self.match_info = None

        print(f"\n{'='*80}")
        print("FastExperiment Phase 1-B 초기화")
        print(f"{'='*80}")
        print(f"  Phase 1-A 피처: 21개")
        print(f"  Phase 1-B 신규: 3개 (match_info)")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  총 피처: 24개")
        print(f"\n  Match stats path: {match_stats_path or 'Not provided'}")
        print(f"  Match info path: {match_info_path or 'Not provided'}")

    def load_match_stats(self):
        """
        preprocessed_match_stats.csv 로드

        컬럼:
        - game_id: 경기 ID
        - team_id: 팀 ID
        - team_avg_goals_for: 해당 경기까지의 평균 득점
        - team_avg_goals_against: 해당 경기까지의 평균 실점

        Returns:
            match_stats: DataFrame
        """
        if self.match_stats is not None:
            return self.match_stats

        print(f"\n{'='*80}")
        print("Match Stats 로드")
        print(f"{'='*80}")

        if self.match_stats_path is None:
            raise ValueError(
                "match_stats_path가 지정되지 않았습니다. "
                "FastExperimentPhase1B(..., match_stats_path='경로')로 지정하세요."
            )

        stats_path = Path(self.match_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Match stats 파일을 찾을 수 없습니다: {stats_path}\n"
                f"preprocessed_match_stats.csv를 먼저 생성하세요."
            )

        self.match_stats = pd.read_csv(stats_path)
        print(f"  파일: {stats_path}")
        print(f"  행 수: {len(self.match_stats):,}")
        print(f"  컬럼: {list(self.match_stats.columns)}")

        # 필수 컬럼 확인
        required_cols = ['game_id', 'team_id', 'team_avg_goals_for', 'team_avg_goals_against']
        missing = [c for c in required_cols if c not in self.match_stats.columns]
        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing}")

        print(f"  ✓ 필수 컬럼 확인 완료")

        return self.match_stats

    def load_match_info(self):
        """
        match_info.csv 로드 (home_team_id 정보)

        컬럼:
        - game_id: 경기 ID
        - home_team_id: 홈팀 ID
        - away_team_id: 어웨이팀 ID
        - home_score: 홈팀 득점
        - away_score: 어웨이팀 득점

        Returns:
            match_info: DataFrame
        """
        if self.match_info is not None:
            return self.match_info

        print(f"\n{'='*80}")
        print("Match Info 로드")
        print(f"{'='*80}")

        if self.match_info_path is None:
            # 기본 경로 시도
            default_path = Path(__file__).parent.parent.parent / 'data' / 'match_info.csv'
            if default_path.exists():
                self.match_info_path = str(default_path)
            else:
                raise ValueError(
                    "match_info_path가 지정되지 않았습니다. "
                    "FastExperimentPhase1B(..., match_info_path='경로')로 지정하세요."
                )

        info_path = Path(self.match_info_path)
        if not info_path.exists():
            raise FileNotFoundError(
                f"Match info 파일을 찾을 수 없습니다: {info_path}"
            )

        self.match_info = pd.read_csv(info_path)
        print(f"  파일: {info_path}")
        print(f"  경기 수: {len(self.match_info):,}")

        # 필수 컬럼 확인
        required_cols = ['game_id', 'home_team_id', 'away_team_id']
        missing = [c for c in required_cols if c not in self.match_info.columns]
        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing}")

        print(f"  ✓ 필수 컬럼 확인 완료")

        return self.match_info

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        피처 생성 - Phase 1-B (24개)

        Phase 1-A 피처 21개 + 신규 3개 = 총 24개

        구성:
        -------------------------
        Phase 1-A 피처 (21개):
          - 기존 v2: 16개
          - 공유 코드 인사이트: 5개

        Phase 1-B 신규 피처 (3개): ⭐
          - is_attacking_home: 공격권 팀 홈/어웨이
          - team_avg_goals_for: 공격권 팀 평균 득점
          - team_avg_goals_against: 공격권 팀 평균 실점
        -------------------------

        Args:
            df: 원본 DataFrame

        Returns:
            df: 피처가 추가된 DataFrame
        """
        # 1. Phase 1-A 피처 생성 (부모 클래스 호출)
        df = super().create_features(df)

        print(f"\n{'='*80}")
        print("Phase 1-B 신규 피처 생성 (3개)")
        print(f"{'='*80}")

        # 2. match_stats & match_info 로드 (아직 안 했으면)
        if self.match_stats is None:
            self.load_match_stats()

        if self.match_info is None:
            self.load_match_info()

        # ========================================================================
        # 신규 피처 1: is_attacking_home (⭐⭐⭐⭐⭐)
        # ========================================================================

        print("\n  [1/3] is_attacking_home 생성 중...")
        print("        의미: 공격권 팀이 홈팀인지 여부")

        # game_id로 match_info와 조인하여 home_team_id 가져오기
        df = df.merge(
            self.match_info[['game_id', 'home_team_id']],
            on='game_id',
            how='left'
        )

        # is_final_team==1 (공격권) and team_id==home_team_id
        df['is_attacking_home'] = (
            (df['is_final_team'] == 1) &
            (df['team_id'] == df['home_team_id'])
        ).astype(int)

        # home_team_id는 더 이상 필요 없음 (중간 컬럼)
        # 하지만 디버깅용으로 유지할 수도 있음 - 일단 제거
        df = df.drop(columns=['home_team_id'])

        # 통계
        last_df = df.groupby('game_episode').last()
        is_home_dist = last_df['is_attacking_home'].value_counts().to_dict()
        print(f"      ✓ is_attacking_home")
        print(f"         분포: {is_home_dist}")
        print(f"         (1=홈팀 공격, 0=어웨이팀 공격)")

        # ========================================================================
        # 신규 피처 2: team_avg_goals_for (⭐⭐⭐⭐)
        # ========================================================================

        print("\n  [2/3] team_avg_goals_for 생성 중...")
        print("        의미: 공격권 팀의 평균 득점 (공격력 지표)")

        # match_stats와 조인 (game_id, team_id)
        df = df.merge(
            self.match_stats[['game_id', 'team_id', 'team_avg_goals_for']],
            on=['game_id', 'team_id'],
            how='left'
        )

        # NaN 처리 (첫 경기 등)
        df['team_avg_goals_for'] = df['team_avg_goals_for'].fillna(0)

        # 통계
        last_df = df.groupby('game_episode').last()
        avg_gf = last_df['team_avg_goals_for']
        print(f"      ✓ team_avg_goals_for")
        print(f"         평균: {avg_gf.mean():.3f}")
        print(f"         범위: {avg_gf.min():.3f} ~ {avg_gf.max():.3f}")

        # ========================================================================
        # 신규 피처 3: team_avg_goals_against (⭐⭐⭐)
        # ========================================================================

        print("\n  [3/3] team_avg_goals_against 생성 중...")
        print("        의미: 공격권 팀의 평균 실점 (수비력 지표)")

        # match_stats와 조인
        df = df.merge(
            self.match_stats[['game_id', 'team_id', 'team_avg_goals_against']],
            on=['game_id', 'team_id'],
            how='left',
            suffixes=('', '_dup')
        )

        # 중복 컬럼 제거 (이미 조인되어 있는 경우)
        dup_cols = [c for c in df.columns if c.endswith('_dup')]
        if dup_cols:
            df = df.drop(columns=dup_cols)

        # NaN 처리
        df['team_avg_goals_against'] = df['team_avg_goals_against'].fillna(0)

        # 통계
        last_df = df.groupby('game_episode').last()
        avg_ga = last_df['team_avg_goals_against']
        print(f"      ✓ team_avg_goals_against")
        print(f"         평균: {avg_ga.mean():.3f}")
        print(f"         범위: {avg_ga.min():.3f} ~ {avg_ga.max():.3f}")

        # ========================================================================
        # 요약
        # ========================================================================

        print(f"\n{'='*80}")
        print("피처 생성 완료 (Phase 1-B)")
        print(f"{'='*80}")
        print(f"  Phase 1-A: 21개")
        print(f"  Phase 1-B: 3개")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  총 피처:   24개")
        print(f"\n  Phase 1-B 신규 피처 상세:")
        print(f"    1. is_attacking_home       ⭐⭐⭐⭐⭐ 공격권 팀 홈/어웨이")
        print(f"    2. team_avg_goals_for      ⭐⭐⭐⭐   평균 득점 (공격력)")
        print(f"    3. team_avg_goals_against  ⭐⭐⭐     평균 실점 (수비력)")

        return df

    def prepare_data(self, df: pd.DataFrame):
        """
        마지막 패스만 추출 & Feature/Target 분리 (24개 피처)

        설명:
        - 각 Episode의 마지막 패스만 사용 (골 직전 상황)
        - Feature(X): 24개 피처 (Phase 1-A 21개 + Phase 1-B 3개)
        - Target(y): end_x, end_y
        - Groups: game_id (GroupKFold용)

        Args:
            df: 피처가 생성된 DataFrame

        Returns:
            X: Feature matrix (n_episodes, 24)
            y: Target matrix (n_episodes, 2)
            groups: game_id array (GroupKFold용)
            feature_cols: Feature 이름 리스트
        """
        print(f"\n{'='*80}")
        print("3. 데이터 준비 (마지막 패스 추출 - Phase 1-B)")
        print(f"{'='*80}")

        # 마지막 패스만 (Episode별!)
        train_last = df.groupby('game_episode').last().reset_index()
        print(f"  Episode 수: {len(train_last):,}개")

        # Feature columns (Phase 1-B: 24개)
        feature_cols = [
            # === Phase 1-A 피처 (21개) ===
            # 기존 v2 피처 (16개)
            'start_x', 'start_y',
            'zone_x', 'zone_y',
            'direction', 'prev_dx', 'prev_dy',
            'goal_distance', 'goal_angle',
            'period_id', 'time_seconds', 'time_left',
            'pass_count',
            'is_home_encoded', 'type_encoded', 'result_encoded',

            # 공유 코드 인사이트 (5개)
            'is_final_team',
            'team_possession_pct',
            'team_switches',
            'game_clock_min',
            'final_poss_len',

            # === Phase 1-B 신규 피처 (3개) ===
            'is_attacking_home',       # 공격권 팀 홈/어웨이
            'team_avg_goals_for',      # 평균 득점
            'team_avg_goals_against'   # 평균 실점
        ]

        # Feature matrix
        X = train_last[feature_cols].values

        # Target (end_x, end_y)
        y = train_last[['end_x', 'end_y']].values

        # Groups (game_id for GroupKFold)
        # game_episode = "{game_id}_{episode_num}" 형식
        groups = train_last['game_episode'].str.split('_').str[0].values

        print(f"\n  데이터 형상:")
        print(f"    X: {X.shape} (episodes, features)")
        print(f"    y: {y.shape} (episodes, targets)")
        print(f"    groups: {len(np.unique(groups))} games")

        print(f"\n  Feature 목록 (24개):")
        for i, col in enumerate(feature_cols):
            if i >= 21:
                marker = " ⭐ NEW (Phase 1-B)"
            elif i >= 16:
                marker = " ⭐ NEW (Phase 1-A)"
            else:
                marker = ""
            print(f"    {i+1:2d}. {col:30s}{marker}")

        return X, y, groups, feature_cols


# ================================================================================
# 테스트 코드
# ================================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("FastExperiment Phase 1-B 테스트")
    print("=" * 80)

    # 경로 설정 (프로젝트 구조에 맞게 조정)
    from pathlib import Path

    # 현재 파일 위치에서 상대 경로 계산
    base_path = Path(__file__).parent.parent.parent

    match_stats_path = base_path / 'code' / 'models' / 'experiments' / 'exp_031_phase1b' / 'preprocessed_match_stats.csv'
    match_info_path = base_path / 'data' / 'match_info.csv'
    train_path = base_path / 'train.csv'

    print(f"\n경로 확인:")
    print(f"  Match stats: {match_stats_path.exists()} - {match_stats_path}")
    print(f"  Match info: {match_info_path.exists()} - {match_info_path}")
    print(f"  Train data: {train_path.exists()} - {train_path}")

    # 1. Initialize
    exp = FastExperimentPhase1B(
        match_stats_path=str(match_stats_path),
        match_info_path=str(match_info_path),
        sample_frac=0.01,  # 1% 샘플 (빠른 테스트)
        n_folds=3,
        random_state=42
    )

    # 2. Load data (1% 샘플)
    train_df = exp.load_data(train_path=str(train_path), sample=True)

    # 3. Create features
    train_df = exp.create_features(train_df)

    # 4. Prepare data
    X, y, groups, feature_cols = exp.prepare_data(train_df)

    # 5. 신규 피처 통계 확인
    print(f"\n{'='*80}")
    print("5. Phase 1-B 신규 피처 통계 확인")
    print(f"{'='*80}")

    last_df = train_df.groupby('game_episode').last()

    for feat in ['is_attacking_home', 'team_avg_goals_for', 'team_avg_goals_against']:
        print(f"\n  {feat}:")
        if feat == 'is_attacking_home':
            print(f"    분포: {last_df[feat].value_counts().to_dict()}")
        else:
            print(f"    평균: {last_df[feat].mean():.3f}")
            print(f"    표준편차: {last_df[feat].std():.3f}")
            print(f"    범위: [{last_df[feat].min():.3f}, {last_df[feat].max():.3f}]")

    # 6. 피처 상관관계 확인 (신규 3개)
    print(f"\n{'='*80}")
    print("6. 신규 피처 간 상관관계")
    print(f"{'='*80}")

    corr_cols = ['is_attacking_home', 'team_avg_goals_for', 'team_avg_goals_against']
    corr_matrix = last_df[corr_cols].corr()
    print(f"\n{corr_matrix}")

    print(f"\n{'='*80}")
    print("✅ FastExperiment Phase 1-B 테스트 완료!")
    print(f"{'='*80}")
    print(f"\n요약:")
    print(f"  총 피처 수: {len(feature_cols)}개")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Games: {len(np.unique(groups))}")

    print(f"\n다음 단계:")
    print(f"  1. 전체 데이터로 CV 검증 (sample_frac=1.0)")
    print(f"  2. Phase 1-A와 성능 비교")
    print(f"  3. 최적 파라미터로 최종 모델 학습")
    print(f"  4. Test 데이터 예측 및 제출")
