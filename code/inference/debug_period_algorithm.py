#!/usr/bin/env python3
"""
Comprehensive diagnostic of period detection algorithm.

Traces through the entire pipeline on a single lightcurve to identify where errors occur.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "training"))
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))

from temporal_models_v10 import get_temporal_model
from preprocessing_utils_v10 import detrend_lightcurve, robust_normalize
from inference_utils_v10 import extract_sliding_sequences
from inference.clustering_period_finder import AdaptiveClusteringPeriodFinder


def load_lightcurve(csv_path):
    """Load lightcurve and extract metadata."""
    try:
        data = pd.read_csv(csv_path, comment='#')

        # Get time and flux
        if 'time' in data.columns:
            time = data['time'].values
        elif 'TIME' in data.columns:
            time = data['TIME'].values
        else:
            return None, None, None, "No time column"

        if 'flux' in data.columns:
            flux = data['flux'].values
        elif 'PDCSAP_FLUX' in data.columns:
            flux = data['PDCSAP_FLUX'].values
        else:
            return None, None, None, "No flux column"

        # Load ground truth period
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
            return None, None, None, "Too few samples"

        flux = detrend_lightcurve(flux)
        flux = robust_normalize(flux)

        return time, flux, gt_period, None

    except Exception as e:
        return None, None, None, str(e)


def trace_algorithm(csv_path, model, finder, device='cpu'):
    """Trace through algorithm step by step."""
    time, flux, gt_period, error_msg = load_lightcurve(csv_path)

    if error_msg:
        return {'error': error_msg}

    print(f"\n{'='*80}")
    print(f"TRACING: {csv_path.name}")
    print(f"{'='*80}")
    print(f"Ground truth period: {gt_period} days" if gt_period else "No ground truth")
    print(f"Lightcurve: {len(time)} samples, {time[-1]-time[0]:.1f} days")
    print(f"Cadence: {np.median(np.diff(time)):.4f} days/sample")

    # Step 1: Extract sequences
    print("\n[STEP 1] Extract sliding sequences...")
    sequences, positions = extract_sliding_sequences(flux)
    print(f"  Extracted {len(sequences)} sequences")
    print(f"  Position range: {positions[0]} to {positions[-1]} (stride: {positions[1]-positions[0]})")

    # Map positions to times
    sequence_times = time[positions]
    print(f"  Time range: {sequence_times[0]:.4f} to {sequence_times[-1]:.4f} days")
    print(f"  Time stride: {np.median(np.diff(sequence_times)):.4f} days")

    # Step 2: Run model inference
    print("\n[STEP 2] Run model inference...")
    sequences_tensor = torch.FloatTensor(sequences).to(device)

    all_probs = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(sequences_tensor), batch_size):
            batch = sequences_tensor[i:i+batch_size]
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs)
    print(f"  Probabilities range: {probs.min():.4f} to {probs.max():.4f}")
    print(f"  Mean probability: {probs.mean():.4f}")
    print(f"  Prob > 0.5: {(probs >= 0.5).sum()} sequences")
    print(f"  Prob > 0.7: {(probs >= 0.7).sum()} sequences")
    print(f"  Prob > 0.9: {(probs >= 0.9).sum()} sequences")

    # Step 3: Filter high probability
    print("\n[STEP 3] Filter high-probability detections...")
    threshold = finder.min_prob_threshold
    high_prob_mask = probs >= threshold
    n_detections = high_prob_mask.sum()

    print(f"  Threshold: {threshold}")
    print(f"  High-prob detections: {n_detections}")

    if n_detections < 2:
        return {'error': f'Too few detections: {n_detections}'}

    high_prob_times = sequence_times[high_prob_mask]
    high_prob_probs = probs[high_prob_mask]

    # Show detected times
    print(f"  Detected times: {high_prob_times[:10]}" + (" ..." if len(high_prob_times) > 10 else ""))
    print(f"  Time spacings between detections:")
    if len(high_prob_times) > 1:
        spacings = np.diff(high_prob_times)
        print(f"    Mean: {np.mean(spacings):.4f} days")
        print(f"    Median: {np.median(spacings):.4f} days")
        print(f"    Std: {np.std(spacings):.4f} days")
        print(f"    Min: {np.min(spacings):.4f} days")
        print(f"    Max: {np.max(spacings):.4f} days")

    # Step 4: Run clustering
    print("\n[STEP 4] Run period detection...")
    period, confidence, info = finder.find_period(
        flux=flux,
        time=time,
        transit_probs=probs,
        positions=positions,
        sequences=sequences
    )

    print(f"  Detected period: {period:.4f} days" if period else "  No period detected")
    print(f"  Confidence: {confidence:.4f}" if period else "")
    print(f"  Number of clusters: {info.get('n_clusters', 'N/A')}")
    print(f"  Method info: {info}")

    # Step 5: Calculate error
    if period and gt_period:
        error_pct = abs(period - gt_period) / gt_period * 100
        print(f"\n[RESULT]")
        print(f"  Detected: {period:.4f} days")
        print(f"  Ground truth: {gt_period} days")
        print(f"  Error: {error_pct:.1f}%")
    else:
        print(f"\n[RESULT]")
        print(f"  FAILED - No period detected or no ground truth")
        error_pct = None

    return {
        'filename': csv_path.name,
        'gt_period': gt_period,
        'detected_period': period,
        'confidence': confidence,
        'error_pct': error_pct,
        'n_detections': int(n_detections),
        'detection_times': high_prob_times.tolist(),
        'info': info
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='Folder with lightcurves')
    parser.add_argument('model_checkpoint', help='Model checkpoint path')
    parser.add_argument('--limit', type=int, default=5, help='Number of files to trace')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')
    args = parser.parse_args()

    folder_path = Path(args.folder)
    model_path = Path(args.model_checkpoint)

    if not folder_path.exists():
        print(f"[ERROR] Folder not found: {folder_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"[ERROR] Model checkpoint not found: {model_path}")
        sys.exit(1)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")

    checkpoint = torch.load(model_path, map_location=device)
    if 'lstm' in str(model_path).lower():
        model = get_temporal_model('temporal_lstm')
    else:
        model = get_temporal_model('temporal_transformer')

    if 'state_dict' in checkpoint:
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create period finder
    finder = AdaptiveClusteringPeriodFinder(
        min_prob_threshold=args.threshold,
        base_eps=0.6,
        min_samples=2,
        use_pca=True,
        device=device
    )

    # Find CSV files
    csv_files = sorted(folder_path.glob("*.csv"))[:args.limit]

    if not csv_files:
        print(f"[ERROR] No CSV files found in {folder_path}")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} lightcurves to trace\n")

    # Trace each
    results = []
    for csv_path in csv_files:
        result = trace_algorithm(csv_path, model, finder, device)
        results.append(result)

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in results if 'error' not in r and r['detected_period'] is not None]
    failed = [r for r in results if r.get('detected_period') is None or 'error' in r]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        errors = [r['error_pct'] for r in successful if r['error_pct'] is not None]
        if errors:
            print(f"\nError statistics (successful detections):")
            print(f"  Mean: {np.mean(errors):.1f}%")
            print(f"  Median: {np.median(errors):.1f}%")
            print(f"  Std: {np.std(errors):.1f}%")


if __name__ == "__main__":
    main()
