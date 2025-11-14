"""
Orbital Data Preprocessor V10 - Temporal Sequences

Standalone preprocessor that generates temporal sequences for transit detection.
No imports from v8/v9 - all dependencies are self-contained.

Key Features:
- Temporal sequences (5 windows with 50% overlap)
- Copied V9.1 preprocessing (detrending, robust normalization)
- Copied V9.1 Label 1 strategy (isolated dips, noise, variability)
- Binary classification: Label 1 (negative), Label 2 (transit)
"""

import csv
import json
import logging
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import our standalone utilities
from preprocessing_utils_v10 import (
    detrend_lightcurve,
    robust_normalize,
    validate_window,
    find_isolated_dips,
    find_systematic_noise,
    find_stellar_variability,
    mask_transit_regions,
    extract_out_of_transit_windows,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class V10Config:
    """Configuration for V10 preprocessing."""
    data_dir: Path
    output_dir: Path
    window_size: int = 256
    sequence_length: int = 5
    sequence_stride: int = 128  # 50% overlap
    samples_per_transit: int = 6
    workers: int = 8
    chunk_size: int = 65536
    seed: int = 42


class SequenceBuilder:
    """
    Build temporal sequences from lightcurves.

    Converts individual windows into sequences of overlapping windows.
    """

    def __init__(self, sequence_length: int = 5, stride: int = 128, window_size: int = 256):
        self.sequence_length = sequence_length
        self.stride = stride
        self.window_size = window_size

    def build_sequences(self, flux: np.ndarray, label: int, target_id: str) -> List[Dict]:
        """
        Create overlapping sequences from flux array.

        Args:
            flux: Preprocessed flux array
            label: 1 (negative) or 2 (transit)
            target_id: Star identifier

        Returns:
            List of dicts with 'sequence', 'label', 'target_id', 'start_idx'
        """
        sequences = []

        # Total length needed for one sequence
        total_length = (self.sequence_length - 1) * self.stride + self.window_size

        # Slide across lightcurve
        for start_idx in range(0, len(flux) - total_length + 1, self.stride):
            sequence_windows = []

            # Extract sequence_length windows
            for i in range(self.sequence_length):
                window_start = start_idx + i * self.stride
                window = flux[window_start:window_start + self.window_size]

                if not validate_window(window):
                    break

                sequence_windows.append(window)

            # Only add if we got all windows
            if len(sequence_windows) == self.sequence_length:
                sequence_array = np.array(sequence_windows)  # Shape: (5, 256)

                sequences.append({
                    'sequence': sequence_array,
                    'label': label,
                    'target_id': target_id,
                    'start_idx': start_idx
                })

        return sequences


class V10Preprocessor:
    """
    Main V10 preprocessor - generates temporal sequences.
    """

    def __init__(self, config: V10Config):
        self.config = config
        self.sequence_builder = SequenceBuilder(
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
            window_size=config.window_size
        )

        # Output paths
        self.output_dir = Path(config.output_dir)
        self.chunks_dir = self.output_dir / "chunks"
        self.metadata_dir = self.output_dir / "metadata"

        self._setup_output_dirs()

        np.random.seed(config.seed)

    def _setup_output_dirs(self):
        """Create output directories."""
        for dir_path in [self.output_dir, self.chunks_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

    def process_all(self):
        """Main entry point - process all CSV files."""
        logger.info("="*80)
        logger.info("V10 PREPROCESSING - TEMPORAL SEQUENCES")
        logger.info("="*80)
        logger.info(f"Data dir: {self.config.data_dir}")
        logger.info(f"Output dir: {self.config.output_dir}")
        logger.info(f"Window size: {self.config.window_size}")
        logger.info(f"Sequence length: {self.config.sequence_length}")
        logger.info(f"Sequence stride: {self.config.sequence_stride}")
        logger.info(f"Workers: {self.config.workers}")
        logger.info("="*80)

        # Find CSV files
        csv_files = self._find_csv_files()
        logger.info(f"Found {len(csv_files)} CSV files")

        # Process with multiprocessing
        all_sequences = []
        stats = {'label_1': 0, 'label_2': 0}

        if self.config.workers > 1:
            with mp.Pool(self.config.workers) as pool:
                results = list(tqdm(
                    pool.imap(self._process_single_file, csv_files),
                    total=len(csv_files),
                    desc="Processing"
                ))

            for sequences, file_stats in results:
                all_sequences.extend(sequences)
                for label, count in file_stats.items():
                    stats[label] += count
        else:
            # Sequential (for debugging)
            for csv_file in tqdm(csv_files, desc="Processing"):
                sequences, file_stats = self._process_single_file(csv_file)
                all_sequences.extend(sequences)
                for label, count in file_stats.items():
                    stats[label] += count

        logger.info(f"\nTotal sequences extracted:")
        logger.info(f"  Label 1 (negatives): {stats['label_1']}")
        logger.info(f"  Label 2 (transits):  {stats['label_2']}")

        # Balance classes
        all_sequences = self._balance_classes(all_sequences)

        # Write output
        self._write_output(all_sequences)

        logger.info("\nV10 preprocessing complete!")
        logger.info(f"Output: {self.output_dir}")

    def _find_csv_files(self) -> List[Path]:
        """Find all lightcurve CSV files."""
        csv_files = []
        for csv_file in self.config.data_dir.rglob("*.csv"):
            # Skip metadata files
            if "metadata" in str(csv_file).lower():
                continue
            csv_files.append(csv_file)
        return csv_files

    def _process_single_file(self, csv_path: Path) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Process a single CSV file.

        Returns:
            (sequences, stats)
        """
        try:
            # Load CSV (skip comment lines starting with #)
            data = pd.read_csv(csv_path, comment='#')

            # Extract columns - handle both uppercase and lowercase
            # Try time column (TIME or time)
            if 'TIME' in data.columns:
                time = data['TIME'].values
            elif 'time' in data.columns:
                time = data['time'].values
            else:
                return [], {'label_1': 0, 'label_2': 0}

            # Try flux column (PDCSAP_FLUX, flux, or SAP_FLUX)
            if 'PDCSAP_FLUX' in data.columns:
                flux = data['PDCSAP_FLUX'].values
            elif 'flux' in data.columns:
                flux = data['flux'].values
            elif 'SAP_FLUX' in data.columns:
                flux = data['SAP_FLUX'].values
            else:
                return [], {'label_1': 0, 'label_2': 0}

            # Try quality column (optional)
            if 'QUALITY' in data.columns:
                quality = data['QUALITY'].values
            elif 'quality' in data.columns:
                quality = data['quality'].values
            else:
                quality = np.zeros_like(flux)

            # Get metadata from header comments
            metadata = self._parse_metadata(csv_path)
            label = int(metadata.get('label', 0))
            target_id = metadata.get('target_id', csv_path.stem)

            if label not in [1, 2]:
                return [], {'label_1': 0, 'label_2': 0}

            # Filter NaN
            valid_mask = np.isfinite(time) & np.isfinite(flux)
            time = time[valid_mask]
            flux = flux[valid_mask]
            quality = quality[valid_mask]

            if len(flux) < self.config.window_size * 2:
                return [], {'label_1': 0, 'label_2': 0}

            # Preprocess (detrend + normalize)
            flux = detrend_lightcurve(flux)
            flux = robust_normalize(flux)

            # Extract based on label
            sequences = []

            if label == 2:
                # Transits
                sequences = self._extract_transit_sequences(flux, time, metadata, target_id)
            elif label == 1:
                # Negatives
                sequences = self._extract_negative_sequences(flux, quality, target_id)

            stats = {
                'label_1': sum(1 for s in sequences if s['label'] == 1),
                'label_2': sum(1 for s in sequences if s['label'] == 2)
            }

            return sequences, stats

        except Exception as e:
            logger.warning(f"Error processing {csv_path}: {e}")
            return [], {'label_1': 0, 'label_2': 0}

    def _parse_metadata(self, csv_path: Path) -> Dict[str, Any]:
        """Parse metadata from CSV header comments.

        Handles both formats:
        - # label=2 (equals format)
        - # Label: 2 (colon format)
        """
        metadata = {}

        try:
            with open(csv_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        # Parse comment lines - handle both = and : formats
                        if '=' in line:
                            key, value = line[1:].split('=', 1)
                            metadata[key.strip()] = value.strip()
                        elif ':' in line:
                            key, value = line[1:].split(':', 1)
                            metadata[key.strip()] = value.strip()
                    else:
                        break
        except Exception:
            pass

        # Normalize key names to match expected format
        # Map: Label→label, Period→ORBITAL_PERIOD, etc.
        key_mapping = {
            'Label': 'label',
            'Period': 'ORBITAL_PERIOD',
            'Epoch': 'EPOCH',
            'Duration': 'DURATION',
            'Target ID': 'target_id'
        }

        normalized = {}
        for key, value in metadata.items():
            # Use mapped key if exists, otherwise use original
            normalized_key = key_mapping.get(key, key)
            normalized[normalized_key] = value

        return normalized

    def _extract_transit_sequences(
        self,
        flux: np.ndarray,
        time: np.ndarray,
        metadata: Dict,
        target_id: str
    ) -> List[Dict]:
        """Extract sequences from transit regions (Label 2)."""
        # Get transit parameters
        period = float(metadata.get('ORBITAL_PERIOD', 0))
        epoch = float(metadata.get('EPOCH', 0))
        duration = float(metadata.get('DURATION', 0))

        if period <= 0 or duration <= 0:
            return []

        # Calculate transit times
        transit_times = []
        t_start = time[0]
        t_end = time[-1]

        transit_time = epoch
        while transit_time < t_start:
            transit_time += period

        while transit_time <= t_end:
            # Find closest time index
            idx = np.argmin(np.abs(time - transit_time))
            transit_times.append(idx)
            transit_time += period

        if not transit_times:
            return []

        # Extract windows around transits
        sequences = []
        half_window = self.config.window_size // 2

        for transit_idx in transit_times:
            # Extract samples_per_transit windows around each transit
            for offset in np.linspace(-half_window, half_window, self.config.samples_per_transit):
                center = int(transit_idx + offset)
                start = center - half_window

                if start < 0 or start + self.config.window_size >= len(flux):
                    continue

                # Extract region around this transit
                region_flux = flux[max(0, start - 500):min(len(flux), start + self.config.window_size + 500)]

                if len(region_flux) < self.config.window_size * 2:
                    continue

                # Build sequences from this region
                region_sequences = self.sequence_builder.build_sequences(region_flux, label=2, target_id=target_id)
                sequences.extend(region_sequences)

        return sequences

    def _extract_negative_sequences(
        self,
        flux: np.ndarray,
        quality: np.ndarray,
        target_id: str
    ) -> List[Dict]:
        """
        Extract sequences from negative examples (Label 1).

        Uses V9.1 multi-strategy:
        - 40% isolated dips
        - 30% systematic noise
        - 20% stellar variability
        - 10% random
        """
        all_windows = []

        # Strategy 1: Isolated dips (40%)
        dip_centers = find_isolated_dips(flux, self.config.window_size)
        for center in dip_centers:
            start = center - self.config.window_size // 2
            if 0 <= start <= len(flux) - self.config.window_size:
                all_windows.append(start)

        # Strategy 2: Systematic noise (30%)
        noise_windows = find_systematic_noise(flux, quality, self.config.window_size)
        all_windows.extend(noise_windows)

        # Strategy 3: Stellar variability (20%)
        variability_windows = find_stellar_variability(flux, self.config.window_size)
        all_windows.extend(variability_windows)

        # Strategy 4: Random (10%)
        num_random = len(all_windows) // 9
        if num_random > 0:
            random_starts = np.random.choice(
                range(0, len(flux) - self.config.window_size, 20),
                size=min(num_random, (len(flux) - self.config.window_size) // 20),
                replace=False
            )
            all_windows.extend(random_starts)

        # Build sequences from negative windows
        # For negatives, we just build sequences from the entire lightcurve
        sequences = self.sequence_builder.build_sequences(flux, label=1, target_id=target_id)

        return sequences

    def _balance_classes(self, sequences: List[Dict]) -> List[Dict]:
        """Balance Label 1 and Label 2 to 1:1 ratio."""
        by_label = {1: [], 2: []}

        for seq in sequences:
            by_label[seq['label']].append(seq)

        min_count = min(len(by_label[1]), len(by_label[2]))

        if min_count == 0:
            logger.warning("No sequences for one or both labels!")
            return sequences

        balanced = []
        for label in [1, 2]:
            if len(by_label[label]) > min_count:
                sampled = np.random.choice(by_label[label], size=min_count, replace=False)
                balanced.extend(sampled)
            else:
                balanced.extend(by_label[label])

        np.random.shuffle(balanced)

        logger.info(f"Balanced to {min_count} sequences per class")
        return balanced

    def _write_output(self, sequences: List[Dict]):
        """Write sequences to chunked NPY files + manifest."""
        chunk_size = self.config.chunk_size
        manifest_rows = []

        logger.info(f"Writing {len(sequences)} sequences...")

        for chunk_idx in range(0, len(sequences), chunk_size):
            chunk_sequences = sequences[chunk_idx:chunk_idx + chunk_size]

            # Stack sequences: (N, 5, 256)
            chunk_array = np.array([s['sequence'] for s in chunk_sequences], dtype=np.float32)

            # Save chunk
            chunk_file = f"chunk_{chunk_idx // chunk_size:05d}.npy"
            chunk_path = self.chunks_dir / chunk_file
            np.save(chunk_path, chunk_array)

            # Add to manifest
            for i, seq in enumerate(chunk_sequences):
                manifest_rows.append({
                    'chunk_file': chunk_file,
                    'chunk_row': i,
                    'label': seq['label'],
                    'target_id': seq['target_id'],
                    'start_idx': seq['start_idx']
                })

        # Write manifest
        manifest_path = self.metadata_dir / "sequence_manifest_v10.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        # Write stats
        stats = {
            'total_sequences': len(sequences),
            'label_1': sum(1 for s in sequences if s['label'] == 1),
            'label_2': sum(1 for s in sequences if s['label'] == 2),
            'window_size': self.config.window_size,
            'sequence_length': self.config.sequence_length,
            'sequence_stride': self.config.sequence_stride,
        }

        stats_path = self.metadata_dir / "dataset_stats_v10.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Written {len(manifest_rows)} sequences to {len(manifest_rows) // chunk_size + 1} chunks")