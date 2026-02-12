"""
Phase 1-A: 공유 코드 인사이트 통합 실험 시스템

================================================================================
목표: 공유 코드의 핵심 인사이트를 우리 Best 모델에 통합
================================================================================

배경:
- 공유 코드 작성자: "16점 → 13점대 가능" 언급  
- 댓글: "LightGBM으로 12점대 나오나요?"
- 현재 우리 Best: 15.84점 (catboost_tuned)
- 목표: 15.3-15.5점 (0.3-0.5점 개선)

핵심 인사이트 5가지:
================================================================================
1. is_final_team (공격권 플래그) ⭐⭐⭐⭐⭐
   - 가장 중요한 발견!
   - 각 패스가 골 넣은 팀의 것인지 표시
   - 공격 vs 수비 맥락 구분
   
2. team_possession_pct (점유율)
   - 최근 20개 패스 중 우리 팀 비율
   - 조직적 공격 vs 역습 구분
   
3. team_switches (공수 전환)
   - 팀이 바뀐 횟수 누적
   - 공수 전환이 많으면 혼란스러운 상황
   
4. game_clock_min (경기 시간)
   - 0-90분+ 연속 표현
   - 전반/후반 통합
   
5. final_poss_len (연속 소유)
   - 현재 연속으로 우리 팀이 소유한 패스 수
   - 빌드업 vs 단발성 구분
================================================================================

변경사항:
- 기존 v2: 16개 피처
- Phase 1-A: 16 + 5 = 21개 피처
- GroupKFold 유지 (game_id 기준)
- n_folds: 3 (시간 단축)

예상 효과:
- CV: 15.60 → 15.3-15.5 (0.1-0.3점 개선)
- Gap: 0.24 → 0.2 이하 (안정성 향상)
- Public: 15.84 → 15.5-15.7

작성일: 2025-12-17 02:50
작성자: Phase 1-A Team
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FastExperimentPhase1A:
    """
    Phase 1-A 전용 실험 시스템
    
    공유 코드 인사이트 5개를 통합한 버전
    기존 FastExperimentV2 (16개 피처) + 신규 5개 = 21개 피처
    """

    def __init__(self, sample_frac=0.1, n_folds=3, random_state=42):
        """
        초기화
        
        Args:
            sample_frac: Episode 샘플링 비율 (0.1 = 10%, 1.0 = 100%)
            n_folds: Cross-validation fold 수 (권장: 3)
            random_state: Random seed (재현성)
        """
        self.sample_frac = sample_frac
        self.n_folds = n_folds
        self.random_state = random_state
        np.random.seed(random_state)
        
        print(f"\n{'='*80}")
        print("FastExperiment Phase 1-A 초기화")
        print(f"{'='*80}")
        print(f"  Sample fraction: {sample_frac*100:.0f}%")
        print(f"  CV folds: {n_folds}")
        print(f"  Random seed: {random_state}")

    def load_data(self, train_path='../../../train.csv', sample=True):
        """
        데이터 로드 (Episode 단위 샘플링)
        
        IMPORTANT: 
        - Episode 단위로 샘플링 (패스 단위 X)
        - 같은 경기의 episode는 함께 유지
        
        Args:
            train_path: 학습 데이터 경로
            sample: 샘플링 여부 (False=전체 데이터)
            
        Returns:
            train_df: 로드된 DataFrame
        """
        print(f"\n{'='*80}")
        print("1. 데이터 로드")
        print(f"{'='*80}")

        # 전체 데이터 로드
        train_df = pd.read_csv(train_path)
        print(f"  원본 데이터:")
        print(f"    - 패스 수: {len(train_df):,}개")
        print(f"    - 컬럼: {list(train_df.columns)}")

        if sample and self.sample_frac < 1.0:
            # Episode 단위 샘플링 (중요!)
            episodes = train_df['game_episode'].unique()
            n_total_episodes = len(episodes)
            print(f"\n  Episode 샘플링:")
            print(f"    - 전체 Episode: {n_total_episodes:,}개")

            n_sample = int(n_total_episodes * self.sample_frac)
            sampled_episodes = np.random.choice(
                episodes,
                size=n_sample,
                replace=False
            )

            train_df = train_df[train_df['game_episode'].isin(sampled_episodes)]
            print(f"    - 샘플 Episode: {n_sample:,}개 ({self.sample_frac*100:.0f}%)")
            print(f"    - 샘플 패스: {len(train_df):,}개")
        else:
            print(f"\n  전체 데이터 사용 (샘플링 없음)")

        return train_df

    def create_features(self, df):
        """
        피처 생성 - Phase 1-A (21개)
        
        기존 16개 + 신규 5개 = 총 21개
        
        구성:
        -------------------------
        기존 피처 (16개):
          - 공간: start_x/y, zone_x/y
          - 방향: direction, prev_dx/dy
          - 골: goal_distance, goal_angle
          - 시간: period_id, time_seconds, time_left
          - 진행: pass_count
          - 타입: is_home, type, result
          
        신규 피처 (5개): ⭐
          - is_final_team: 공격권 플래그
          - team_possession_pct: 점유율 (최근 20개)
          - team_switches: 공수 전환 횟수
          - game_clock_min: 0-90분 연속
          - final_poss_len: 연속 소유 길이
        -------------------------
        
        CONSTRAINT:
        - 모든 피처는 Episode 내부에서만 계산 (Data Leakage 방지)
        - groupby('game_episode') 필수
        - shift(), cumsum() 등 시간 순서 고려
        
        Args:
            df: 원본 DataFrame
            
        Returns:
            df: 피처가 추가된 DataFrame
        """
        print(f"\n{'='*80}")
        print("2. 피처 생성 (Phase 1-A)")
        print(f"{'='*80}")

        df = df.copy()

        # ========================================================================
        # 기존 피처 (v2와 동일, 16개)
        # ========================================================================
        
        print("\n  [기존 피처 16개 생성 중...]")

        # 1. Zone 6x6 (필드를 36개 구역으로 분할)
        df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
        df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
        print("    ✓ zone_x, zone_y (6x6 grid)")

        # 2. Direction 8-way (이전 패스의 방향, 45도 간격)
        df['dx'] = df['end_x'] - df['start_x']
        df['dy'] = df['end_y'] - df['start_y']

        # Episode별로 shift (중요! Episode 경계 넘지 않게)
        df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
        df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

        angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
        df['direction'] = ((angle + 22.5) // 45).astype(int) % 8
        print("    ✓ direction, prev_dx, prev_dy (8-way)")

        # 3. Goal distance & angle (골문까지 거리 및 각도)
        # 골문 중심: (105, 34)
        df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
        df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))
        print("    ✓ goal_distance, goal_angle")

        # 4. Time features
        # time_left: 경기 종료까지 남은 시간 (전체 5400초 = 90분)
        df['time_left'] = 5400 - df['time_seconds']
        print("    ✓ time_left")

        # 5. Episode features
        # pass_count: Episode 내에서 현재 패스 번호 (1부터 시작)
        df['pass_count'] = df.groupby('game_episode').cumcount() + 1
        print("    ✓ pass_count")

        # 6-8. v2 피처 (홈/어웨이, 타입, 결과)
        df['is_home_encoded'] = df['is_home'].astype(int)

        type_map = {'Pass': 0, 'Carry': 1}
        df['type_encoded'] = df['type_name'].map(type_map).fillna(2).astype(int)

        result_map = {'Successful': 0, 'Unsuccessful': 1}
        df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)
        print("    ✓ is_home_encoded, type_encoded, result_encoded")

        # ========================================================================
        # 신규 피처 (Phase 1-A, 5개) ⭐
        # ========================================================================
        
        print("\n  [Phase 1-A 신규 피처 5개 생성 중...] ⭐")

        # 1. is_final_team (공격권 플래그) ⭐⭐⭐⭐⭐
        # 설명: 각 패스가 "골 넣은 팀"의 패스인지 표시
        # 중요도: 최고! 공격 맥락 vs 수비 맥락 구분
        print("\n    [1/5] is_final_team 생성 중...")
        
        # 각 episode의 마지막 team_id 가져오기
        df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
        
        # 현재 패스의 team_id와 비교
        df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
        
        print(f"      ✓ is_final_team")
        print(f"         - 의미: 골 넣은 팀의 패스 = 1, 상대 팀 = 0")
        print(f"         - 분포: {df.groupby('game_episode').last()['is_final_team'].value_counts().to_dict()}")

        # 2. team_possession_pct (점유율) ⭐⭐⭐⭐
        # 설명: 최근 20개 패스 중 우리 팀(골 넣은 팀) 비율
        # 중요도: 높음! 조직적 공격(높은 점유율) vs 역습(낮은 점유율) 구분
        print("\n    [2/5] team_possession_pct 생성 중...")
        
        # rolling mean (최근 20개, 부족하면 가능한 만큼)
        df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )
        
        print(f"      ✓ team_possession_pct")
        print(f"         - 의미: 최근 20개 중 우리 팀 비율 (0.0~1.0)")
        last_poss = df.groupby('game_episode').last()['team_possession_pct']
        print(f"         - 평균: {last_poss.mean():.3f}")
        print(f"         - 범위: {last_poss.min():.3f} ~ {last_poss.max():.3f}")

        # 3. team_switches (공수 전환 횟수) ⭐⭐⭐
        # 설명: is_final_team이 바뀐 횟수 (0→1 또는 1→0)
        # 중요도: 중간! 공수 전환이 많으면 혼란스러운 상황
        print("\n    [3/5] team_switches 생성 중...")
        
        # diff()로 변화 감지 (Episode별로!)
        df['team_switch_event'] = (
            df.groupby('game_episode')['is_final_team'].diff() != 0
        ).astype(int)
        
        # 누적 합계
        df['team_switches'] = df.groupby('game_episode')['team_switch_event'].cumsum()
        df = df.drop(columns=['team_switch_event'])
        
        print(f"      ✓ team_switches")
        print(f"         - 의미: 공수 전환 누적 횟수")
        last_switches = df.groupby('game_episode').last()['team_switches']
        print(f"         - 평균: {last_switches.mean():.1f}회")
        print(f"         - 범위: {last_switches.min()} ~ {last_switches.max()}회")

        # 4. game_clock_min (경기 시간, 0-90분+) ⭐⭐⭐
        # 설명: 전반/후반 통합한 연속 시간 (분 단위)
        # 중요도: 중간! 경기 후반일수록 급해짐
        print("\n    [4/5] game_clock_min 생성 중...")
        
        # period_id: 1=전반, 2=후반
        # 전반: time_seconds/60 (0-45분)
        # 후반: 45 + time_seconds/60 (45-90분+)
        df['game_clock_min'] = np.where(
            df['period_id'] == 1,
            df['time_seconds'] / 60.0,
            45.0 + df['time_seconds'] / 60.0
        )
        
        print(f"      ✓ game_clock_min")
        print(f"         - 의미: 경기 시작부터 시간 (분)")
        last_clock = df.groupby('game_episode').last()['game_clock_min']
        print(f"         - 평균: {last_clock.mean():.1f}분")
        print(f"         - 범위: {last_clock.min():.1f} ~ {last_clock.max():.1f}분")

        # 5. final_poss_len (연속 소유 길이) ⭐⭐
        # 설명: 현재 연속으로 우리 팀이 소유한 패스 수
        # 중요도: 낮음~중간! 긴 빌드업 vs 단발성 구분
        print("\n    [5/5] final_poss_len 생성 중...")
        
        # 각 행마다 현재까지의 연속 개수 계산
        def calc_streak(group):
            values = group['is_final_team'].values
            result = []
            current_streak = 0
            for val in values:
                if val == 1:
                    current_streak += 1
                else:
                    current_streak = 0
                result.append(current_streak)
            return pd.Series(result, index=group.index)
        
        df['final_poss_len'] = df.groupby('game_episode').apply(
            calc_streak
        ).reset_index(level=0, drop=True)
        
        print(f"      ✓ final_poss_len")
        print(f"         - 의미: 연속 우리 팀 패스 수")
        last_poss_len = df.groupby('game_episode').last()['final_poss_len']
        print(f"         - 평균: {last_poss_len.mean():.1f}개")
        print(f"         - 범위: {last_poss_len.min()} ~ {last_poss_len.max()}개")

        # ========================================================================
        # 요약
        # ========================================================================
        
        print(f"\n{'='*80}")
        print("피처 생성 완료!")
        print(f"{'='*80}")
        print(f"  기존 피처: 16개")
        print(f"  신규 피처: 5개 (Phase 1-A)")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  총 피처:   21개")
        print(f"\n  신규 피처 상세:")
        print(f"    1. is_final_team       ⭐⭐⭐⭐⭐ 공격권 플래그")
        print(f"    2. team_possession_pct ⭐⭐⭐⭐   점유율 (최근 20개)")
        print(f"    3. team_switches       ⭐⭐⭐     공수 전환 횟수")
        print(f"    4. game_clock_min      ⭐⭐⭐     경기 시간 (분)")
        print(f"    5. final_poss_len      ⭐⭐       연속 소유 길이")

        return df

    def prepare_data(self, df):
        """
        마지막 패스만 추출 & Feature/Target 분리
        
        설명:
        - 각 Episode의 마지막 패스만 사용 (골 직전 상황)
        - Feature(X): 21개 피처
        - Target(y): end_x, end_y
        - Groups: game_id (GroupKFold용)
        
        Args:
            df: 피처가 생성된 DataFrame
            
        Returns:
            X: Feature matrix (n_episodes, 21)
            y: Target matrix (n_episodes, 2)
            groups: game_id array (GroupKFold용)
            feature_cols: Feature 이름 리스트
        """
        print(f"\n{'='*80}")
        print("3. 데이터 준비 (마지막 패스 추출)")
        print(f"{'='*80}")
        
        # 마지막 패스만 (Episode별!)
        train_last = df.groupby('game_episode').last().reset_index()
        print(f"  Episode 수: {len(train_last):,}개")

        # Feature columns (Phase 1-A: 21개)
        feature_cols = [
            # === 기존 피처 (16개) ===
            # 위치
            'start_x', 'start_y',
            # Zone
            'zone_x', 'zone_y',
            # 방향
            'direction', 'prev_dx', 'prev_dy',
            # 골
            'goal_distance', 'goal_angle',
            # 시간
            'period_id', 'time_seconds', 'time_left',
            # 진행
            'pass_count',
            # 타입
            'is_home_encoded', 'type_encoded', 'result_encoded',
            
            # === Phase 1-A 신규 피처 (5개) ===
            'is_final_team',        # 공격권
            'team_possession_pct',  # 점유율
            'team_switches',        # 공수 전환
            'game_clock_min',       # 경기 시간
            'final_poss_len'        # 연속 소유
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
        
        print(f"\n  Feature 목록 (21개):")
        for i, col in enumerate(feature_cols):
            marker = " ⭐ NEW" if i >= 16 else ""
            print(f"    {i+1:2d}. {col:25s}{marker}")

        return X, y, groups, feature_cols

    def run_cv(self, model_x, model_y, X, y, groups, model_name='CatBoost'):
        """
        Cross-validation with GroupKFold
        
        설명:
        - GroupKFold: 같은 game의 episode는 train/val 분리
        - 각 fold마다 별도 x, y 모델 학습
        - Euclidean distance로 평가
        
        Args:
            model_x: X 좌표 예측 모델
            model_y: Y 좌표 예측 모델
            X: Feature matrix
            y: Target matrix (end_x, end_y)
            groups: game_id array
            model_name: 모델 이름 (로그용)
            
        Returns:
            mean_cv: 평균 CV score
            std_cv: CV score 표준편차
            fold_scores: 각 fold별 score 리스트
        """
        print(f"\n{'='*80}")
        print(f"4. Cross-Validation ({self.n_folds}-Fold GroupKFold)")
        print(f"{'='*80}")
        print(f"  Model: {model_name}")
        print(f"  Strategy: GroupKFold (game-level split)")
        print(f"  Metric: Euclidean Distance (mean)")

        gkf = GroupKFold(n_splits=self.n_folds)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            print(f"\n  {'─'*60}")
            print(f"  Fold {fold}/{self.n_folds}")
            print(f"  {'─'*60}")
            
            # Split
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"    Train: {len(X_train):,} episodes")
            print(f"    Val:   {len(X_val):,} episodes")

            # Fit separate models for x and y
            print(f"    학습 중...", end='', flush=True)
            start = time.time()
            
            model_x.fit(X_train, y_train[:, 0])
            model_y.fit(X_train, y_train[:, 1])
            
            train_time = time.time() - start
            print(f" 완료 ({train_time:.1f}s)")

            # Predict
            pred_x = np.clip(model_x.predict(X_val), 0, 105)
            pred_y = np.clip(model_y.predict(X_val), 0, 68)

            # Euclidean distance
            dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
            cv = dist.mean()
            fold_scores.append(cv)

            print(f"    Fold {fold} Score: {cv:.4f}")

        # Summary
        mean_cv = np.mean(fold_scores)
        std_cv = np.std(fold_scores)

        print(f"\n  {'='*60}")
        print(f"  CV 결과 요약")
        print(f"  {'='*60}")
        print(f"  Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")
        print(f"  Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
        print(f"  Min: {min(fold_scores):.4f}")
        print(f"  Max: {max(fold_scores):.4f}")

        return mean_cv, std_cv, fold_scores

    def log_experiment(self, name, cv, params, features, runtime, notes=''):
        """
        실험 로그 저장 (JSON)
        
        Args:
            name: 실험 이름
            cv: (mean, std, fold_scores) 튜플
            params: 모델 하이퍼파라미터 dict
            features: Feature 이름 리스트
            runtime: 총 실행 시간 (초)
            notes: 추가 메모
            
        Returns:
            log: 저장된 로그 dict
        """
        log = {
            'name': name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cv_mean': float(cv[0]),
            'cv_std': float(cv[1]),
            'cv_folds': [float(x) for x in cv[2]],
            'params': params,
            'features': features,
            'n_features': len(features),
            'runtime': float(runtime),
            'sample_frac': float(self.sample_frac),
            'n_folds': int(self.n_folds),
            'notes': notes,
            'phase': 'Phase 1-A',
            'new_features': [
                'is_final_team',
                'team_possession_pct',
                'team_switches',
                'game_clock_min',
                'final_poss_len'
            ]
        }

        # Append to log file
        log_file = Path(__file__).parent.parent.parent / 'logs' / 'experiment_log_phase1a.json'
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, 'a') as f:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

        print(f"\n  ✅ 로그 저장: {log_file}")

        return log


# ================================================================================
# 테스트 코드
# ================================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("FastExperiment Phase 1-A 테스트")
    print("=" * 80)
    
    # 1. Initialize
    exp = FastExperimentPhase1A(sample_frac=0.05, n_folds=3, random_state=42)
    
    # 2. Load data (5% 샘플)
    train_df = exp.load_data(sample=True)
    
    # 3. Create features
    train_df = exp.create_features(train_df)
    
    # 4. Prepare data
    X, y, groups, feature_cols = exp.prepare_data(train_df)
    
    # 5. 신규 피처 통계 확인
    print(f"\n{'='*80}")
    print("5. 신규 피처 통계 확인")
    print(f"{'='*80}")
    
    last_df = train_df.groupby('game_episode').last()
    
    for feat in ['is_final_team', 'team_possession_pct', 'team_switches', 
                 'game_clock_min', 'final_poss_len']:
        print(f"\n  {feat}:")
        if feat == 'is_final_team':
            print(f"    분포: {last_df[feat].value_counts().to_dict()}")
        else:
            print(f"    평균: {last_df[feat].mean():.2f}")
            print(f"    표준편차: {last_df[feat].std():.2f}")
            print(f"    범위: [{last_df[feat].min():.2f}, {last_df[feat].max():.2f}]")
    
    print(f"\n{'='*80}")
    print("✅ FastExperiment Phase 1-A 테스트 완료!")
    print(f"{'='*80}")
    print(f"\n다음 단계:")
    print(f"  1. 전체 데이터로 CV 검증 (sample_frac=1.0)")
    print(f"  2. 최적 파라미터로 최종 모델 학습")
    print(f"  3. Test 데이터 예측")
    print(f"  4. Submission 생성")
    print(f"  5. DACON 제출 및 결과 확인")
