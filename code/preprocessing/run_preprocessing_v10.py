#!/usr/bin/env python3
"""
CLI for V10 Preprocessing - Temporal Sequences

Usage:
    python run_preprocessing_v10.py \\
        --data-dir data \\
        --output-dir orbital_windows_dataset_v10_preprocessed \\
        --workers 16
"""

import argparse
import sys
from pathlib import Path

# Add preprocessing directory to path
PREPROCESSING_DIR = Path(__file__).parent / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

from orbital_data_preprocessor_v10 import V10Preprocessor, V10Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="V10 Preprocessing - Temporal Sequences for Transit Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_preprocessing_v10.py \\
      --data-dir data \\
      --output-dir orbital_windows_dataset_v10_preprocessed

  # With custom settings
  python run_preprocessing_v10.py \\
      --data-dir data \\
      --output-dir orbital_windows_dataset_v10_preprocessed \\
      --sequence-length 5 \\
      --sequence-stride 128 \\
      --workers 16
        """
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Input directory with CSV lightcurves"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="orbital_windows_dataset_v10_preprocessed",
        help="Output directory for preprocessed sequences"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Window size in samples (default: 256)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Number of windows per sequence (default: 5)"
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=128,
        help="Stride between consecutive windows in sequence (default: 128 = 50%% overlap)"
    )
    parser.add_argument(
        "--samples-per-transit",
        type=int,
        default=6,
        help="Number of window samples per transit event (default: 6)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes (default: 8)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=65536,
        help="Number of sequences per output chunk (default: 65536)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate data directory
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Create config
    config = V10Config(
        data_dir=data_dir,
        output_dir=Path(args.output_dir).resolve(),
        window_size=args.window_size,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        samples_per_transit=args.samples_per_transit,
        workers=args.workers,
        chunk_size=args.chunk_size,
        seed=args.seed
    )

    # Run preprocessing
    preprocessor = V10Preprocessor(config)
    preprocessor.process_all()

    print("\n✅ V10 preprocessing completed successfully!")
    print(f"Output: {config.output_dir}")


if __name__ == "__main__":
    main()
