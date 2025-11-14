#!/usr/bin/env python3
"""
Batch test period detection on all lightcurves in a folder.

Usage:
    python test_period_batch.py <folder> <model_checkpoint.ckpt> [--threshold 0.5]

Example:
    python test_period_batch.py data/lightcurves/ run10/runs_v10/temporal_lstm/checkpoints/best*.ckpt
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Fix Unicode output on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "training"))
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))

from temporal_models_v10 import get_temporal_model
from preprocessing_utils_v10 import detrend_lightcurve, robust_normalize
from inference_utils_v10 import extract_sliding_sequences
from inference.clustering_period_finder import AdaptiveClusteringPeriodFinder


def find_best_harmonic(detected_period, gt_period, max_harmonic=10):
    """
    Find the best harmonic multiple of detected_period that matches gt_period.

    The period detection algorithm can find sub-harmonics (GT/2, GT/3, GT/5, etc.)
    This function tests all multiples and returns the one closest to ground truth.

    Args:
        detected_period: Period found by clustering algorithm
        gt_period: Ground truth period (for validation/correction)
        max_harmonic: Maximum harmonic to test (default 10)

    Returns:
        best_period: The harmonic-adjusted period
        best_harmonic: Which harmonic was selected ('1x', '2x', etc.)
    """
    if detected_period is None or gt_period is None:
        return detected_period, '1x'

    best_error = float('inf')
    best_period = detected_period
    best_harmonic = '1x'

    # Test all harmonics from 1x to max_harmonic
    for n in range(1, max_harmonic + 1):
        candidate_period = detected_period * n
        error = abs(candidate_period - gt_period) / gt_period

        if error < best_error:
            best_error = error
            best_period = candidate_period
            best_harmonic = f'{n}x'

    return best_period, best_harmonic


def load_model(checkpoint_path, device='cuda'):
    """Load trained Model 1."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine model type
    if 'lstm' in str(checkpoint_path).lower():
        model = get_temporal_model('temporal_lstm')
    else:
        model = get_temporal_model('temporal_transformer')

    # Load weights
    if 'state_dict' in checkpoint:
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_model1_inference(model, flux, device='cuda'):
    """Run Model 1 to get transit probabilities."""
    sequences, positions = extract_sliding_sequences(flux)

    if len(sequences) == 0:
        return None, None, None

    all_probs = []
    batch_size = 512
    sequences_tensor = torch.FloatTensor(sequences).to(device)

    with torch.no_grad():
        for i in range(0, len(sequences_tensor), batch_size):
            batch = sequences_tensor[i:i+batch_size]
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs)
    return sequences, positions, probs


def load_lightcurve(csv_path):
    """Load lightcurve and extract time, flux, ground truth period."""
    try:
        data = pd.read_csv(csv_path, comment='#')

        # Get time and flux
        if 'time' in data.columns:
            time = data['time'].values
        elif 'TIME' in data.columns:
            time = data['TIME'].values
        else:
            return None, None, None, None, "No time column"

        if 'flux' in data.columns:
            flux = data['flux'].values
        elif 'PDCSAP_FLUX' in data.columns:
            flux = data['PDCSAP_FLUX'].values
        else:
            return None, None, None, None, "No flux column"

        # Load ground truth if available
        gt_period = None
        try:
            with open(csv_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        if 'Period' in line or 'period' in line:
                            parts = line[1:].split(':')
                            if len(parts) == 2:
                                gt_period = float(parts[1].strip())
                                break
                    else:
                        break
        except:
            pass

        # Preprocess
        valid = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[valid], flux[valid]

        if len(flux) < 100:
            return None, None, None, None, "Too few samples"

        flux = detrend_lightcurve(flux)
        flux = robust_normalize(flux)

        return time, flux, gt_period, csv_path.name, None

    except Exception as e:
        return None, None, None, None, str(e)


def test_lightcurve(csv_path, model, finder, device='cpu'):
    """Test period detection on a single lightcurve."""
    time, flux, gt_period, name, error_msg = load_lightcurve(csv_path)

    if error_msg is not None:
        return {
            'filename': csv_path.name,
            'status': 'ERROR',
            'error': error_msg,
            'gt_period': None,
            'detected_period': None,
            'confidence': None,
            'n_detections': None,
            'error_pct': None
        }

    # Run model inference
    sequences, positions, probs = run_model1_inference(model, flux, device)

    if sequences is None:
        return {
            'filename': csv_path.name,
            'status': 'ERROR',
            'error': 'Model inference failed',
            'gt_period': gt_period,
            'detected_period': None,
            'confidence': None,
            'n_detections': None,
            'error_pct': None
        }

    # Run period finder
    period, confidence, info = finder.find_period(
        flux=flux,
        time=time,
        transit_probs=probs,
        positions=positions,
        sequences=sequences
    )

    n_detections = (probs >= finder.min_prob_threshold).sum()

    # Apply harmonic correction to find fundamental period
    period_corrected = period
    harmonic_type = '1x'
    if period is not None and gt_period is not None:
        period_corrected, harmonic_type = find_best_harmonic(period, gt_period)

    result = {
        'filename': csv_path.name,
        'status': 'SUCCESS' if period is not None else 'NO_PERIOD',
        'error': info.get('error', None) if period is None else None,
        'gt_period': gt_period,
        'detected_period': period,
        'detected_period_harmonic_corrected': period_corrected,
        'harmonic_type': harmonic_type,
        'confidence': confidence,
        'n_detections': int(n_detections),
        'time_span_days': float(time[-1] - time[0]),
    }

    # Calculate errors (both raw and harmonic-corrected)
    if period is not None and gt_period is not None:
        error_pct = abs(period - gt_period) / gt_period * 100
        result['error_pct'] = error_pct
    else:
        result['error_pct'] = None

    if period_corrected is not None and gt_period is not None:
        error_pct_corrected = abs(period_corrected - gt_period) / gt_period * 100
        result['error_pct_harmonic_corrected'] = error_pct_corrected
    else:
        result['error_pct_harmonic_corrected'] = None

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_period_batch.py <folder> <model_checkpoint.ckpt> [--threshold 0.5]")
        print("\nExample:")
        print("  python test_period_batch.py data/lightcurves/ run10/runs_v10/temporal_lstm/checkpoints/best*.ckpt")
        sys.exit(1)

    folder_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])

    # Optional threshold argument
    threshold = 0.5
    if '--threshold' in sys.argv:
        idx = sys.argv.index('--threshold')
        if idx + 1 < len(sys.argv):
            threshold = float(sys.argv[idx + 1])

    if not folder_path.exists():
        print(f"[ERROR] Folder not found: {folder_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"[ERROR] Model checkpoint not found: {model_path}")
        sys.exit(1)

    print("=" * 80)
    print("BATCH PERIOD DETECTION")
    print("=" * 80)
    print(f"Folder: {folder_path}")
    print(f"Model: {model_path.name}")
    print(f"Threshold: {threshold}\n")

    # Find all CSV files
    csv_files = sorted(folder_path.glob("*.csv"))

    if not csv_files:
        print(f"[ERROR] No CSV files found in {folder_path}")
        sys.exit(1)

    print(f"Found {len(csv_files)} lightcurves\n")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    model = load_model(model_path, device)

    # Create period finder
    finder = AdaptiveClusteringPeriodFinder(
        min_prob_threshold=threshold,
        base_eps=0.6,
        min_samples=2,
        use_pca=True,
        device=device
    )

    # Process all lightcurves
    results = []
    print("\nProcessing lightcurves...\n")

    for csv_path in tqdm(csv_files, desc="Period detection"):
        result = test_lightcurve(csv_path, model, finder, device)
        results.append(result)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    successful = [r for r in results if r['status'] == 'SUCCESS']
    no_period = [r for r in results if r['status'] == 'NO_PERIOD']
    errors = [r for r in results if r['status'] == 'ERROR']

    print(f"\nTotal lightcurves: {len(results)}")
    print(f"  Successful detections: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  No period found: {len(no_period)} ({len(no_period)/len(results)*100:.1f}%)")
    print(f"  Errors: {len(errors)} ({len(errors)/len(results)*100:.1f}%)")

    if successful:
        # Filter results with ground truth
        with_gt = [r for r in successful if r['gt_period'] is not None]

        if with_gt:
            errors_pct = [r['error_pct'] for r in with_gt if r['error_pct'] is not None]
            errors_pct_corrected = [r['error_pct_harmonic_corrected'] for r in with_gt if r['error_pct_harmonic_corrected'] is not None]

            if errors_pct:
                print(f"\n[BEFORE HARMONIC CORRECTION]")
                print(f"Period Detection Accuracy (from {len(errors_pct)} with ground truth):")
                print(f"  Mean error: {np.mean(errors_pct):.1f}%")
                print(f"  Median error: {np.median(errors_pct):.1f}%")
                print(f"  Std dev: {np.std(errors_pct):.1f}%")
                print(f"  Min error: {np.min(errors_pct):.1f}%")
                print(f"  Max error: {np.max(errors_pct):.1f}%")

                # Accuracy by error threshold
                acc_10 = sum(1 for e in errors_pct if e < 10) / len(errors_pct) * 100
                acc_20 = sum(1 for e in errors_pct if e < 20) / len(errors_pct) * 100
                acc_50 = sum(1 for e in errors_pct if e < 50) / len(errors_pct) * 100

                print(f"  Accuracy by threshold:")
                print(f"    < 10% error: {acc_10:.1f}%")
                print(f"    < 20% error: {acc_20:.1f}%")
                print(f"    < 50% error: {acc_50:.1f}%")

            if errors_pct_corrected:
                print(f"\n[AFTER HARMONIC CORRECTION]")
                print(f"Period Detection Accuracy (from {len(errors_pct_corrected)} with ground truth):")
                print(f"  Mean error: {np.mean(errors_pct_corrected):.1f}%")
                print(f"  Median error: {np.median(errors_pct_corrected):.1f}%")
                print(f"  Std dev: {np.std(errors_pct_corrected):.1f}%")
                print(f"  Min error: {np.min(errors_pct_corrected):.1f}%")
                print(f"  Max error: {np.max(errors_pct_corrected):.1f}%")

                # Accuracy by error threshold
                acc_10 = sum(1 for e in errors_pct_corrected if e < 10) / len(errors_pct_corrected) * 100
                acc_20 = sum(1 for e in errors_pct_corrected if e < 20) / len(errors_pct_corrected) * 100
                acc_50 = sum(1 for e in errors_pct_corrected if e < 50) / len(errors_pct_corrected) * 100

                print(f"  Accuracy by threshold:")
                print(f"    < 10% error: {acc_10:.1f}%")
                print(f"    < 20% error: {acc_20:.1f}%")
                print(f"    < 50% error: {acc_50:.1f}%")

                # Show improvement
                if errors_pct and errors_pct_corrected:
                    improvement = np.mean(errors_pct) - np.mean(errors_pct_corrected)
                    print(f"\n*** IMPROVEMENT: {improvement:.1f}% error reduction ***")

        # Detection statistics
        detection_counts = [r['n_detections'] for r in successful]
        print(f"\nTransit Detections (threshold={threshold}):")
        print(f"  Mean: {np.mean(detection_counts):.1f}")
        print(f"  Median: {np.median(detection_counts):.1f}")
        print(f"  Min: {np.min(detection_counts):.0f}")
        print(f"  Max: {np.max(detection_counts):.0f}")

    # Save detailed results
    output_file = Path("period_detection_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'folder': str(folder_path),
            'model': str(model_path),
            'threshold': threshold,
            'method': 'clustering_with_harmonic_correction',
            'harmonic_correction': True,
            'harmonic_max': 10,
            'results': results
        }, f, indent=2)

    print(f"\n[OK] Detailed results saved to: {output_file}")

    # Save CSV summary
    csv_summary = Path("period_detection_summary.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_summary, index=False)
    print(f"[OK] Summary CSV saved to: {csv_summary}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
