"""
exp_037: Phase1A + exp_036 Ensemble

전략:
  - Phase1A (15.35) + exp_036 (15.38) 평균
  - 두 좋은 모델 결합
  - 예상: 더 안정적이고 좋은 성능

작성일: 2025-12-21
"""

import pandas as pd
import numpy as np


def main():
    print("\n" + "=" * 80)
    print("exp_037: Phase1A + exp_036 Ensemble")
    print("=" * 80)

    # Phase1A 제출 파일
    phase1a_path = '../../../../submissions/submission_phase1a_cv15.45.csv'
    # exp_036 제출 파일
    exp036_path = '../exp_036_last_pass_only/submission_last_pass_only_cv15.4881.csv'

    print("\nLoading submissions...")
    print(f"  Phase1A: {phase1a_path}")
    phase1a = pd.read_csv(phase1a_path)
    print(f"    Loaded: {len(phase1a)} rows")

    print(f"  exp_036: {exp036_path}")
    exp036 = pd.read_csv(exp036_path)
    print(f"    Loaded: {len(exp036)} rows")

    # Validate
    assert len(phase1a) == len(exp036) == 2414, "Row count mismatch"

    # Sort both by game_episode for alignment
    phase1a = phase1a.sort_values('game_episode').reset_index(drop=True)
    exp036 = exp036.sort_values('game_episode').reset_index(drop=True)

    assert (phase1a['game_episode'] == exp036['game_episode']).all(), "Episode mismatch after sort"

    print("  ✓ Validation passed")

    # Ensemble (simple average)
    print("\nCreating ensemble (average)...")
    ensemble = pd.DataFrame({
        'game_episode': phase1a['game_episode'],
        'end_x': (phase1a['end_x'] + exp036['end_x']) / 2,
        'end_y': (phase1a['end_y'] + exp036['end_y']) / 2
    })

    # Clip to field boundaries
    ensemble['end_x'] = ensemble['end_x'].clip(0, 105)
    ensemble['end_y'] = ensemble['end_y'].clip(0, 68)

    print(f"  Ensemble shape: {ensemble.shape}")
    print(f"  end_x: {ensemble['end_x'].mean():.2f} ± {ensemble['end_x'].std():.2f}")
    print(f"  end_y: {ensemble['end_y'].mean():.2f} ± {ensemble['end_y'].std():.2f}")

    # Save
    filename = 'submission_ensemble_phase1a_exp036.csv'
    ensemble.to_csv(filename, index=False)

    print(f"\n" + "=" * 80)
    print("Ensemble Complete!")
    print("=" * 80)
    print(f"  File: {filename}")
    print(f"  Rows: {len(ensemble)}")
    print(f"  Components:")
    print(f"    - Phase1A: Public 15.35, CV 15.45")
    print(f"    - exp_036: Public 15.38, CV 15.49")
    print(f"  Expected: ~15.3 (better than both)")
    print("=" * 80)


if __name__ == '__main__':
    main()
