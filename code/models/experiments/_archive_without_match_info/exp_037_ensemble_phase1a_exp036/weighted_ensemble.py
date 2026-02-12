"""
Weighted Ensemble Script for K-League Competition
Phase1A (Public 15.35) + exp_036 (Public 15.38)

Phase1A is better, so we give it higher weight.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def load_submissions(
    phase1a_path: str,
    exp036_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load two submission files and align by game_episode."""
    df1 = pd.read_csv(phase1a_path)
    df2 = pd.read_csv(exp036_path)

    print(f"Phase1A shape: {df1.shape}")
    print(f"exp_036 shape: {df2.shape}")

    # Sort both by game_episode to ensure alignment
    df1 = df1.sort_values('game_episode').reset_index(drop=True)
    df2 = df2.sort_values('game_episode').reset_index(drop=True)

    # Verify same game_episodes after sorting
    assert df1['game_episode'].equals(df2['game_episode']), "game_episode mismatch after sorting!"

    print(f"Sorted and aligned successfully")

    return df1, df2


def weighted_ensemble(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    weight1: float,
    weight2: float
) -> pd.DataFrame:
    """
    Create weighted ensemble of two submissions.

    Args:
        df1: Phase1A submission (better, Public 15.35)
        df2: exp_036 submission (Public 15.38)
        weight1: Weight for Phase1A
        weight2: Weight for exp_036

    Returns:
        Ensembled submission dataframe
    """
    # Normalize weights
    total = weight1 + weight2
    w1 = weight1 / total
    w2 = weight2 / total

    print(f"\nWeights: {w1:.3f} (Phase1A) : {w2:.3f} (exp_036)")

    # Create ensemble
    ensemble = df1[['game_episode']].copy()
    ensemble['end_x'] = w1 * df1['end_x'] + w2 * df2['end_x']
    ensemble['end_y'] = w1 * df1['end_y'] + w2 * df2['end_y']

    return ensemble


def main():
    """Run weighted ensemble with multiple weight combinations."""
    # Input paths
    phase1a_path = '/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submissions/submission_phase1a_cv15.45.csv'
    exp036_path = '/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models/experiments/exp_036_last_pass_only/submission_last_pass_only_cv15.4881.csv'

    # Output directory
    output_dir = Path('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models/experiments/exp_037_ensemble_phase1a_exp036')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Weighted Ensemble: Phase1A (15.35) + exp_036 (15.38)")
    print("="*60)

    # Load submissions
    df_phase1a, df_exp036 = load_submissions(phase1a_path, exp036_path)

    # Weight combinations (Phase1A : exp_036)
    # Phase1A is better, so it gets higher weight
    weight_combinations = [
        (0.7, 0.3),   # Heavy Phase1A
        (0.6, 0.4),   # Moderate Phase1A
        (0.55, 0.45), # Slight Phase1A
    ]

    results = []

    for w1, w2 in weight_combinations:
        print(f"\n{'='*60}")
        print(f"Testing weights: {w1}:{w2}")
        print(f"{'='*60}")

        # Create ensemble
        ensemble = weighted_ensemble(df_phase1a, df_exp036, w1, w2)

        # Save to file
        output_filename = f'submission_weighted_{int(w1*100)}_{int(w2*100)}.csv'
        output_path = output_dir / output_filename
        ensemble.to_csv(output_path, index=False)

        print(f"Saved: {output_filename}")
        print(f"Shape: {ensemble.shape}")
        print(f"Sample:\n{ensemble.head(3)}")

        # Statistics
        stats = {
            'weights': f"{w1}:{w2}",
            'filename': output_filename,
            'mean_x': ensemble['end_x'].mean(),
            'mean_y': ensemble['end_y'].mean(),
            'std_x': ensemble['end_x'].std(),
            'std_y': ensemble['end_y'].std(),
        }
        results.append(stats)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = output_dir / 'ensemble_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Weighted Ensemble Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input 1: Phase1A (Public 15.35, CV 15.45)\n")
        f.write(f"Input 2: exp_036 (Public 15.38, CV 15.4881)\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nNote: Phase1A gets higher weight as it has better Public LB score.\n")

    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "="*60)
    print("DONE! Files created:")
    print("="*60)
    for stats in results:
        print(f"  - {stats['filename']}")
    print(f"  - ensemble_summary.txt")


if __name__ == '__main__':
    main()
