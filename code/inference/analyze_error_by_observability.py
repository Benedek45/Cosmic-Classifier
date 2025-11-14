#!/usr/bin/env python3
"""
Cross-reference period detection results with observability analysis.

Shows error rates broken down by whether each exoplanet's period was observable
in the available lightcurve data.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='period_detection_results.json',
                        help='Period detection results JSON file')
    parser.add_argument('--observability', default='observability_analysis.csv',
                        help='Observability analysis CSV file')
    parser.add_argument('--output', default='error_by_observability.csv',
                        help='Output CSV file')
    args = parser.parse_args()

    # Load results
    if not Path(args.results).exists():
        print(f"[ERROR] Results file not found: {args.results}")
        print("  Run test_period_batch.py first to generate results")
        return

    if not Path(args.observability).exists():
        print(f"[ERROR] Observability file not found: {args.observability}")
        print("  Run analyze_test_data_observability.py first")
        return

    print(f"Loading results from {args.results}...")
    with open(args.results, 'r') as f:
        results_data = json.load(f)

    results_list = results_data.get('results', [])
    print(f"Found {len(results_list)} test results")

    print(f"Loading observability analysis from {args.observability}...")
    obs_df = pd.read_csv(args.observability)
    print(f"Found observability data for {len(obs_df)} files")

    # Create lookup
    obs_lookup = {}
    for _, row in obs_df.iterrows():
        obs_lookup[row['filename']] = {
            'category': row['category'],
            'period': row['period_days'],
            'time_span': row['time_span_days'],
            'transits_visible': row['transits_visible']
        }

    # Merge and analyze
    merged_results = []

    for result in results_list:
        filename = result.get('filename')

        if filename not in obs_lookup:
            # Try without extension variations
            found = False
            for lookup_filename in obs_lookup.keys():
                if lookup_filename.split('.')[0] == filename.split('.')[0]:
                    obs_info = obs_lookup[lookup_filename]
                    found = True
                    break
            if not found:
                continue
        else:
            obs_info = obs_lookup[filename]

        # Merge data
        merged = dict(result)
        merged.update({
            'observability': obs_info['category'],
            'period_days': obs_info['period'],
            'time_span_days': obs_info['time_span'],
            'transits_visible': obs_info['transits_visible']
        })

        merged_results.append(merged)

    merged_df = pd.DataFrame(merged_results)

    print(f"\nSuccessfully merged {len(merged_df)} results with observability data")

    # Print summary
    print("\n" + "="*80)
    print("ERROR ANALYSIS BY OBSERVABILITY")
    print("="*80)

    for category in ['observable', 'marginal', 'unobservable', 'unknown']:
        subset = merged_df[merged_df['observability'] == category]

        if len(subset) == 0:
            continue

        print(f"\n[{category.upper()}] ({len(subset)} files)")

        # Success rate
        successful = subset[subset['status'] == 'SUCCESS']
        success_rate = len(successful) / len(subset) * 100

        print(f"  Success rate: {success_rate:.1f}% ({len(successful)}/{len(subset)})")

        # Error statistics (for successful detections with ground truth)
        with_gt = successful[(successful['gt_period'].notna()) &
                            (successful['error_pct'].notna())]

        if len(with_gt) > 0:
            errors = with_gt['error_pct'].values
            print(f"  Error statistics (n={len(with_gt)}):")
            print(f"    Mean:   {np.mean(errors):7.1f}%")
            print(f"    Median: {np.median(errors):7.1f}%")
            print(f"    Std:    {np.std(errors):7.1f}%")
            print(f"    Min:    {np.min(errors):7.1f}%")
            print(f"    Max:    {np.max(errors):7.1f}%")

            # Accuracy tiers
            acc_10 = (errors < 10).sum() / len(errors) * 100
            acc_20 = (errors < 20).sum() / len(errors) * 100
            acc_50 = (errors < 50).sum() / len(errors) * 100

            print(f"  Accuracy by threshold:")
            print(f"    < 10% error: {acc_10:5.1f}%")
            print(f"    < 20% error: {acc_20:5.1f}%")
            print(f"    < 50% error: {acc_50:5.1f}%")
        else:
            print(f"  No successful detections with ground truth")

        # Detection statistics
        if len(successful) > 0:
            n_detections = successful['n_detections'].dropna()
            if len(n_detections) > 0:
                print(f"  Transit detections (threshold=0.5):")
                print(f"    Mean:   {np.mean(n_detections):7.1f}")
                print(f"    Median: {np.median(n_detections):7.1f}")
                print(f"    Min:    {np.min(n_detections):7.1f}")
                print(f"    Max:    {np.max(n_detections):7.1f}")

    # Overall summary
    print(f"\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    obs_errors = []
    marg_errors = []
    unobs_errors = []

    for category, error_list in [
        ('observable', obs_errors),
        ('marginal', marg_errors),
        ('unobservable', unobs_errors)
    ]:
        subset = merged_df[merged_df['observability'] == category]
        successful = subset[subset['status'] == 'SUCCESS']
        with_gt = successful[(successful['gt_period'].notna()) &
                            (successful['error_pct'].notna())]

        if len(with_gt) > 0:
            errors = with_gt['error_pct'].values
            error_list.extend(errors.tolist())

    if len(obs_errors) > 0:
        print(f"\nObservable files:")
        print(f"  Mean error: {np.mean(obs_errors):.1f}%")
        print(f"  Median error: {np.median(obs_errors):.1f}%")

    if len(marg_errors) > 0:
        print(f"\nMarginal files:")
        print(f"  Mean error: {np.mean(marg_errors):.1f}%")
        print(f"  Median error: {np.median(marg_errors):.1f}%")

    if len(unobs_errors) > 0:
        print(f"\nUnobservable files:")
        print(f"  Mean error: {np.mean(unobs_errors):.1f}%")
        print(f"  Median error: {np.median(unobs_errors):.1f}%")

    all_errors = obs_errors + marg_errors + unobs_errors
    if len(all_errors) > 0:
        print(f"\nALL FILES:")
        print(f"  Mean error: {np.mean(all_errors):.1f}%")
        print(f"  Median error: {np.median(all_errors):.1f}%")
        print(f"  (This includes both observable and unobservable)")

    # Save merged results
    merged_df.to_csv(args.output, index=False)
    print(f"\n[OK] Merged results saved to: {args.output}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    obs_success = len(merged_df[(merged_df['observability'] == 'observable') &
                                (merged_df['status'] == 'SUCCESS')])
    obs_total = len(merged_df[merged_df['observability'] == 'observable'])

    unobs_total = len(merged_df[merged_df['observability'] == 'unobservable'])

    print(f"\nCurrent test composition:")
    print(f"  Observable: {obs_total} files")
    print(f"  Unobservable: {unobs_total} files")

    if unobs_total > 0:
        print(f"\n[ACTION] Remove {unobs_total} unobservable files from test set:")
        print(f"  These have periods longer than observation windows")
        print(f"  Algorithm cannot detect such periods (physically impossible)")
        print(f"  \n  After removal:")
        print(f"  - Test set size: {obs_total} files")
        print(f"  - Expected error: ~20% (observable files only)")
        print(f"  - Current error: {np.mean(all_errors):.1f}%")

if __name__ == "__main__":
    main()
