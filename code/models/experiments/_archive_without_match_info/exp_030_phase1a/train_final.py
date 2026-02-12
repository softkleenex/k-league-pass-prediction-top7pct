"""
Phase 1-A 최종 모델 학습 및 저장

목표:
  1. 전체 학습 데이터로 최종 모델 학습
  2. Test 데이터 예측
  3. Submission 파일 생성

실행 방법:
  python train_final.py [--data-path path] [--output-dir output]

작성일: 2025-12-17
"""

import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor

# Add utils to path
utils_path = Path(__file__).resolve().parent.parent.parent.parent / 'utils'
sys.path.insert(0, str(utils_path))

from fast_experiment_phase1a import FastExperimentPhase1A


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 1-A 최종 모델 학습')
    parser.add_argument('--data-path', type=str, default='../../../../train.csv',
                        help='학습 데이터 경로')
    parser.add_argument('--test-path', type=str, default='../../../../test.csv',
                        help='테스트 데이터 경로')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='출력 디렉토리')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print(f"\n{'='*80}")
    print("Phase 1-A 최종 모델 학습")
    print(f"{'='*80}")

    # 1. 데이터 로드
    print(f"\n{'='*80}")
    print("1. 학습 데이터 로드")
    print(f"{'='*80}")

    exp = FastExperimentPhase1A(sample_frac=1.0, n_folds=1, random_state=args.seed)
    train_df = exp.load_data(train_path=args.data_path, sample=False)

    # 2. 피처 생성
    print(f"\n{'='*80}")
    print("2. 피처 생성")
    print(f"{'='*80}")

    train_df = exp.create_features(train_df)

    # 3. 데이터 준비
    print(f"\n{'='*80}")
    print("3. 데이터 준비")
    print(f"{'='*80}")

    X, y, groups, feature_cols = exp.prepare_data(train_df)

    print(f"  학습 데이터: {X.shape}")

    # 4. 최종 모델 학습
    print(f"\n{'='*80}")
    print("4. 최종 모델 학습 (전체 데이터)")
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

    print(f"  모델 학습 중...", end='', flush=True)
    model_train_start = time.time()

    model_x = CatBoostRegressor(**cb_params)
    model_y = CatBoostRegressor(**cb_params)

    model_x.fit(X, y[:, 0])
    model_y.fit(X, y[:, 1])

    model_train_time = time.time() - model_train_start
    print(f" 완료 ({model_train_time:.1f}s)")

    # 5. 모델 저장
    print(f"\n{'='*80}")
    print("5. 모델 저장")
    print(f"{'='*80}")

    model_x_path = output_dir / 'model_x_catboost.pkl'
    model_y_path = output_dir / 'model_y_catboost.pkl'

    with open(model_x_path, 'wb') as f:
        pickle.dump(model_x, f)

    with open(model_y_path, 'wb') as f:
        pickle.dump(model_y, f)

    print(f"  ✓ 모델 X 저장: {model_x_path}")
    print(f"  ✓ 모델 Y 저장: {model_y_path}")

    # 6. Test 데이터 로드 및 예측
    print(f"\n{'='*80}")
    print("6. Test 데이터 로드 및 예측")
    print(f"{'='*80}")

    # Test 데이터는 별도 디렉토리에 Episode별로 저장되어 있을 수 있음
    # 예: data/test/episode_xxxx.csv
    # 여기서는 test.csv가 있다고 가정

    test_path = Path(args.test_path)

    if not test_path.exists():
        print(f"  경고: {test_path} 파일이 없습니다.")
        print(f"  Test 데이터 경로를 확인하세요.")
        return False

    test_df = pd.read_csv(test_path)
    print(f"  Test 데이터 로드: {len(test_df)} rows")

    # Test 데이터에 피처 생성 (같은 방식)
    print(f"  Test 피처 생성 중...", end='', flush=True)

    # 수동으로 피처 생성 (FastExperimentPhase1A 내부 로직 재현)
    test_df = test_df.copy()

    # Zone
    test_df['zone_x'] = (test_df['start_x'] / (105/6)).astype(int).clip(0, 5)
    test_df['zone_y'] = (test_df['start_y'] / (68/6)).astype(int).clip(0, 5)

    # Direction
    test_df['dx'] = test_df['end_x'] - test_df['start_x']
    test_df['dy'] = test_df['end_y'] - test_df['start_y']
    test_df['prev_dx'] = test_df.groupby('game_episode')['dx'].shift(1).fillna(0)
    test_df['prev_dy'] = test_df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(test_df['prev_dy'], test_df['prev_dx']))
    test_df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    # Goal
    test_df['goal_distance'] = np.sqrt((105 - test_df['start_x'])**2 + (34 - test_df['start_y'])**2)
    test_df['goal_angle'] = np.degrees(np.arctan2(34 - test_df['start_y'], 105 - test_df['start_x']))

    # Time
    test_df['time_left'] = 5400 - test_df['time_seconds']

    # Pass count
    test_df['pass_count'] = test_df.groupby('game_episode').cumcount() + 1

    # Type and result
    test_df['is_home_encoded'] = test_df['is_home'].astype(int)
    type_map = {'Pass': 0, 'Carry': 1}
    test_df['type_encoded'] = test_df['type_name'].map(type_map).fillna(2).astype(int)
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    test_df['result_encoded'] = test_df['result_name'].map(result_map).fillna(2).astype(int)

    # Phase 1-A 피처
    test_df['final_team_id'] = test_df.groupby('game_episode')['team_id'].transform('last')
    test_df['is_final_team'] = (test_df['team_id'] == test_df['final_team_id']).astype(int)

    test_df['team_possession_pct'] = test_df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

    test_df['team_switch_event'] = (
        test_df.groupby('game_episode')['is_final_team'].diff() != 0
    ).astype(int)
    test_df['team_switches'] = test_df.groupby('game_episode')['team_switch_event'].cumsum()

    test_df['game_clock_min'] = np.where(
        test_df['period_id'] == 1,
        test_df['time_seconds'] / 60.0,
        45.0 + test_df['time_seconds'] / 60.0
    )

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

    test_df['final_poss_len'] = test_df.groupby('game_episode').apply(
        calc_streak
    ).reset_index(level=0, drop=True)

    print(f" 완료")

    # 마지막 패스만 추출
    test_last = test_df.groupby('game_episode').last().reset_index()

    X_test = test_last[feature_cols].values

    print(f"  Test 데이터: {X_test.shape}")

    # 예측
    print(f"  예측 중...", end='', flush=True)
    pred_x = np.clip(model_x.predict(X_test), 0, 105)
    pred_y = np.clip(model_y.predict(X_test), 0, 68)
    print(f" 완료")

    # 7. Submission 파일 생성
    print(f"\n{'='*80}")
    print("7. Submission 파일 생성")
    print(f"{'='*80}")

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'x': pred_x,
        'y': pred_y
    })

    submission_path = output_dir / 'submission_phase1a.csv'
    submission.to_csv(submission_path, index=False)

    print(f"  ✓ Submission 저장: {submission_path}")
    print(f"  크기: {len(submission)} rows")
    print(f"  샘플:")
    print(submission.head(10).to_string(index=False))

    # 8. 메타데이터 저장
    total_time = time.time() - start_time

    metadata = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phase': 'Phase 1-A',
        'model': 'CatBoost',
        'features': {
            'total': len(feature_cols),
            'existing': 16,
            'new': 5,
            'names': feature_cols
        },
        'data': {
            'train_episodes': len(X),
            'test_episodes': len(X_test),
            'n_folds': 1
        },
        'model_params': cb_params,
        'training_time': model_train_time,
        'total_time': total_time,
        'submission_file': str(submission_path),
        'model_x_file': str(model_x_path),
        'model_y_file': str(model_y_path)
    }

    metadata_path = output_dir / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ 메타데이터 저장: {metadata_path}")

    # 9. 완료
    print(f"\n{'='*80}")
    print("✅ 최종 모델 학습 완료!")
    print(f"{'='*80}")
    print(f"  총 시간: {total_time:.1f}s")
    print(f"  모델 X: {model_x_path}")
    print(f"  모델 Y: {model_y_path}")
    print(f"  Submission: {submission_path}")

    print(f"\n다음 단계:")
    print(f"  1. DACON에 {submission_path} 제출")
    print(f"  2. 결과 확인 및 SUBMISSION_LOG.md 업데이트")
    print(f"  3. Public Score와 Gap 분석")


if __name__ == '__main__':
    main()
