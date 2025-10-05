"""
Exoplanet Master CSV Preprocessor V15 - Label-Based Split Processing
Processes all data folders and creates separate CSV files for each label.

NEW IN V15:
- Creates separate CSV files for each label (label_1.csv, label_2.csv, etc.)
- Reads label information from metadata CSV files
- Tracks statistics per label (files processed, flux points)
- Reports flux point counts for each label in final statistics
- Future-proof for multiple labels in different data collections

Previous V14 features:
- Processes ALL subdirectories in the data folder
- Merges lightcurves from multiple data collections
- Automatically discovers all available data folders
- Maintains all V13 statistics and processing features

Previous V13 features:
- Added comprehensive statistics tracking
- Detailed reporting on data filtering and processing
- Statistics on outlier removal, quality filtering, and data retention
- Enhanced final report with processing metrics
- Tracks original vs final data point counts

Previous V12 features:
- Relative flux conversion for PatchTST optimization
- Converts each lightcurve to relative flux (baseline = 1.0)
- Preserves transit depths as fractional drops from 1.0
- Works with both PDCSAP and SAP flux types
- Basic outlier removal (5-sigma clipping)
"""

import pandas as pd
import numpy as np
import os
import psutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
import multiprocessing as mp
import time
import gc
import csv
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class ProcessingStats:
    """Class to track processing statistics"""
    total_files_found: int = 0
    files_processed: int = 0
    files_too_few_points: int = 0
    files_poor_quality: int = 0
    files_outlier_removal_failed: int = 0
    files_successfully_processed: int = 0
    total_original_points: int = 0
    total_quality_filtered_points: int = 0
    total_outlier_removed_points: int = 0
    total_final_points: int = 0
    shortest_lightcurve: int = float('inf')
    longest_lightcurve: int = 0
    total_processing_time: float = 0.0
    # Label-specific statistics
    label_stats: dict = None  # Will store {label: {'files': count, 'points': count}}

    def __post_init__(self):
        if self.label_stats is None:
            self.label_stats = {}

def get_memory_usage() -> float:
    """Gets current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def convert_to_relative_flux(flux_data: List[float]) -> List[float]:
    """
    Convert flux data to relative flux (baseline = 1.0).
    This normalizes each lightcurve independently, making transit depths
    comparable across different stellar magnitudes and flux types.
    """
    flux_array = np.array(flux_data)

    # Calculate median flux as baseline
    median_flux = np.median(flux_array)

    # Convert to relative flux
    relative_flux = flux_array / median_flux

    return relative_flux.tolist()

def remove_outliers(flux_data: List[float], sigma_threshold: float = 5.0) -> Tuple[List[float], int]:
    """
    Remove extreme outliers using sigma clipping.
    Keeps outlier removal minimal to preserve astrophysical signals.
    Returns filtered data and count of removed outliers.
    """
    flux_array = np.array(flux_data)
    median_flux = np.median(flux_array)
    std_flux = np.std(flux_array)

    # Create mask for data within sigma_threshold standard deviations
    outlier_mask = np.abs(flux_array - median_flux) < sigma_threshold * std_flux

    # Count removed outliers
    outliers_removed = len(flux_array) - np.sum(outlier_mask)

    # Return filtered data and outlier count
    return flux_array[outlier_mask].tolist(), outliers_removed

def lightcurve_worker(file_path: Path) -> Tuple[Optional[Dict], str, Dict]:
    """
    Worker function to load, filter, and extract flux data from a single file.
    Now includes relative flux conversion, basic outlier removal, and statistics tracking.
    Returns result, status message, and statistics.
    """
    stats = {
        'original_points': 0,
        'quality_filtered_points': 0,
        'outliers_removed': 0,
        'final_points': 0
    }

    try:
        df = pd.read_csv(file_path, comment='#')
        stats['original_points'] = len(df)

        # Basic validation: ensure there's enough data to be meaningful
        if len(df) < 10:
            return None, f"Skipped: Too few data points ({len(df)})", stats

        # Filter out low-quality data (quality > 64 is bad)
        df_filtered = df[df['quality'] <= 64]
        stats['quality_filtered_points'] = len(df_filtered)

        if len(df_filtered) < 10:
            return None, f"Skipped: Too few good quality points ({len(df_filtered)})", stats

        # Sort by time to ensure chronological order and extract flux
        flux_data = df_filtered.sort_values('time')['flux'].tolist()

        # Remove extreme outliers (minimal filtering)
        flux_data, outliers_removed = remove_outliers(flux_data, sigma_threshold=5.0)
        stats['outliers_removed'] = outliers_removed

        # Check if we still have enough data after outlier removal
        if len(flux_data) < 10:
            return None, f"Skipped: Too few data points after outlier removal ({len(flux_data)})", stats

        # Convert to relative flux for PatchTST optimization
        flux_data = convert_to_relative_flux(flux_data)
        stats['final_points'] = len(flux_data)

        planet_id = file_path.stem

        result = {
            'planet_id': planet_id,
            'flux_data': flux_data,
            'flux_length': len(flux_data)
        }

        # Explicit memory cleanup
        del df, df_filtered
        gc.collect()

        return result, "Success", stats

    except Exception as e:
        return None, f"Error: {str(e)}", stats

def process_batch(file_batch: List[Path], pool: mp.Pool) -> Tuple[List[Dict], ProcessingStats]:
    """
    Processes a batch of files using the multiprocessing pool.
    Returns results and accumulated statistics.
    """
    results = pool.map(lightcurve_worker, file_batch)

    # Separate successful results and accumulate statistics
    successful_results = []
    batch_stats = ProcessingStats()

    for res, status, file_stats in results:
        batch_stats.files_processed += 1
        batch_stats.total_original_points += file_stats['original_points']
        batch_stats.total_quality_filtered_points += file_stats['quality_filtered_points']
        batch_stats.total_outlier_removed_points += file_stats['outliers_removed']

        if res is not None:
            successful_results.append(res)
            batch_stats.files_successfully_processed += 1
            batch_stats.total_final_points += file_stats['final_points']

            # Track lightcurve length statistics
            lightcurve_length = file_stats['final_points']
            batch_stats.shortest_lightcurve = min(batch_stats.shortest_lightcurve, lightcurve_length)
            batch_stats.longest_lightcurve = max(batch_stats.longest_lightcurve, lightcurve_length)
        else:
            # Categorize failures
            if "Too few data points" in status and "quality" not in status:
                batch_stats.files_too_few_points += 1
            elif "quality" in status:
                batch_stats.files_poor_quality += 1
            elif "outlier removal" in status:
                batch_stats.files_outlier_removal_failed += 1

    return successful_results, batch_stats

def merge_stats(total_stats: ProcessingStats, batch_stats: ProcessingStats) -> ProcessingStats:
    """Merge batch statistics into total statistics."""
    total_stats.files_processed += batch_stats.files_processed
    total_stats.files_too_few_points += batch_stats.files_too_few_points
    total_stats.files_poor_quality += batch_stats.files_poor_quality
    total_stats.files_outlier_removal_failed += batch_stats.files_outlier_removal_failed
    total_stats.files_successfully_processed += batch_stats.files_successfully_processed
    total_stats.total_original_points += batch_stats.total_original_points
    total_stats.total_quality_filtered_points += batch_stats.total_quality_filtered_points
    total_stats.total_outlier_removed_points += batch_stats.total_outlier_removed_points
    total_stats.total_final_points += batch_stats.total_final_points

    if batch_stats.shortest_lightcurve != float('inf'):
        total_stats.shortest_lightcurve = min(total_stats.shortest_lightcurve, batch_stats.shortest_lightcurve)
    total_stats.longest_lightcurve = max(total_stats.longest_lightcurve, batch_stats.longest_lightcurve)

    return total_stats

def print_statistics_report(stats: ProcessingStats, output_files: Dict[int, str], file_sizes_mb: Dict[int, float]):
    """Print comprehensive statistics report with label-specific information."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PROCESSING STATISTICS")
    print("="*80)

    # File Processing Statistics
    print(f"\nFILE PROCESSING SUMMARY:")
    print(f"  Total files found:           {stats.total_files_found:,}")
    print(f"  Files successfully processed: {stats.files_successfully_processed:,}")
    print(f"  Files skipped (too few pts):  {stats.files_too_few_points:,}")
    print(f"  Files skipped (poor quality): {stats.files_poor_quality:,}")
    print(f"  Files skipped (outlier fail): {stats.files_outlier_removal_failed:,}")

    success_rate = (stats.files_successfully_processed / stats.total_files_found * 100) if stats.total_files_found > 0 else 0
    print(f"  Success rate:                 {success_rate:.1f}%")

    # Label-Specific Statistics
    print(f"\nLABEL-SPECIFIC STATISTICS:")
    for label in sorted(stats.label_stats.keys()):
        label_data = stats.label_stats[label]
        print(f"  Label {label}:")
        print(f"    Files processed:            {label_data['files']:,}")
        print(f"    Total flux points:          {label_data['points']:,}")
        avg_points = label_data['points'] / label_data['files'] if label_data['files'] > 0 else 0
        print(f"    Average points per file:    {avg_points:.0f}")

    # Data Point Statistics
    print(f"\nDATA POINT PROCESSING:")
    print(f"  Original data points:         {stats.total_original_points:,}")
    print(f"  After quality filtering:      {stats.total_quality_filtered_points:,}")
    print(f"  Outliers removed:             {stats.total_outlier_removed_points:,}")
    print(f"  Final data points:            {stats.total_final_points:,}")

    # Data Retention Rates
    quality_retention = (stats.total_quality_filtered_points / stats.total_original_points * 100) if stats.total_original_points > 0 else 0
    outlier_retention = ((stats.total_quality_filtered_points - stats.total_outlier_removed_points) / stats.total_quality_filtered_points * 100) if stats.total_quality_filtered_points > 0 else 0
    overall_retention = (stats.total_final_points / stats.total_original_points * 100) if stats.total_original_points > 0 else 0

    print(f"\nDATA RETENTION RATES:")
    print(f"  Quality filtering retention:  {quality_retention:.1f}%")
    print(f"  Outlier removal retention:    {outlier_retention:.1f}%")
    print(f"  Overall data retention:       {overall_retention:.1f}%")

    # Lightcurve Length Statistics
    print(f"\nLIGHTCURVE LENGTH STATISTICS:")
    if stats.shortest_lightcurve != float('inf'):
        print(f"  Shortest lightcurve:          {stats.shortest_lightcurve:,} points")
    else:
        print(f"  Shortest lightcurve:          N/A")
    print(f"  Longest lightcurve:           {stats.longest_lightcurve:,} points")

    avg_length = stats.total_final_points / stats.files_successfully_processed if stats.files_successfully_processed > 0 else 0
    print(f"  Average lightcurve length:    {avg_length:.0f} points")

    # Output File Statistics
    print(f"\nOUTPUT FILE STATISTICS:")
    total_size_mb = 0
    for label in sorted(output_files.keys()):
        file_size = file_sizes_mb.get(label, 0)
        total_size_mb += file_size
        print(f"  Label {label}: {output_files[label]} ({file_size:.2f} MB)")
    print(f"  Total size:                   {total_size_mb:.2f} MB")

    # Processing Performance
    print(f"\nPROCESSING PERFORMANCE:")
    print(f"  Total processing time:        {stats.total_processing_time:.2f} seconds")
    files_per_second = stats.total_files_found / stats.total_processing_time if stats.total_processing_time > 0 else 0
    print(f"  Files processed per second:   {files_per_second:.1f}")

    print(f"\nPATCHTST READINESS:")
    print(f"  All flux data converted to relative flux (baseline = 1.0)")
    print(f"  Transit depths appear as fractional drops from 1.0")
    print(f"  Compatible with PatchTST scaling='std'")
    print(f"  Separate CSV files for each label for balanced training")
    print("="*80)

def load_label_mapping(data_root: str = "data") -> Dict[str, int]:
    """
    Load label mapping from all metadata CSV files.
    Returns a dictionary mapping target_id to label.
    """
    label_mapping = {}
    data_path = Path(data_root)

    if not data_path.exists():
        print(f"Warning: Data directory '{data_root}' does not exist.")
        return {}

    for subdir in data_path.iterdir():
        if subdir.is_dir():
            metadata_file = subdir / "metadata" / "lightcurve_metadata.csv"
            if metadata_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(metadata_file)
                    for _, row in df.iterrows():
                        target_id = str(row['target_id'])
                        label = int(row['label'])
                        label_mapping[target_id] = label

                    print(f"  Loaded {len(df)} labels from {subdir.name}")
                except Exception as e:
                    print(f"  Warning: Could not load labels from {metadata_file}: {e}")

    print(f"Total label mappings loaded: {len(label_mapping)}")
    return label_mapping

def discover_data_folders(data_root: str = "data") -> List[Path]:
    """
    Discover all data folders in the data directory.
    Returns a list of lightcurve directories to process.
    """
    data_path = Path(data_root)
    lightcurve_dirs = []

    if not data_path.exists():
        print(f"Error: Data directory '{data_root}' does not exist.")
        return []

    # Find all subdirectories that contain lightcurves folders
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            lightcurve_path = subdir / "lightcurves"
            if lightcurve_path.exists() and lightcurve_path.is_dir():
                lightcurve_dirs.append(lightcurve_path)
                print(f"  Found data folder: {subdir.name}")

    return lightcurve_dirs

def main():
    """Main execution function."""
    start_time = time.time()

    # Initialize statistics
    total_stats = ProcessingStats()

    # --- Configuration ---
    DATA_ROOT = "data"  # Root data directory
    OUTPUT_PREFIX = "exoplanet_master_dataset_v15_label"  # Will create label_1.csv, label_2.csv etc.
    MAX_FILES = None # Set to an integer for testing, None to process all files

    # --- System Setup ---
    available_cores = mp.cpu_count()
    n_workers = min(8, available_cores)
    batch_size = 50 # Larger batch size can be more efficient

    print("="*80)
    print("EXOPLANET MASTER CSV PREPROCESSOR V15 - LABEL-BASED SPLIT PROCESSING")
    print(f"System: {psutil.virtual_memory().total / (1024**3):.1f}GB RAM, using {n_workers} workers.")
    print("Features:")
    print("  - Creates separate CSV files for each label")
    print("  - Processes ALL data folders in the data directory")
    print("  - Merges lightcurves from multiple collections")
    print("  - Relative flux conversion (baseline = 1.0)")
    print("  - Basic outlier removal (5-sigma clipping)")
    print("  - Label-specific statistics tracking")
    print("  - Optimized for PatchTST training")
    print("="*80)

    # --- Label Loading ---
    print(f"\nLoading label information from '{DATA_ROOT}':")
    label_mapping = load_label_mapping(DATA_ROOT)

    if not label_mapping:
        print("Error: No label mappings found.")
        return

    unique_labels = sorted(set(label_mapping.values()))
    print(f"Found labels: {unique_labels}")

    # --- Data Folder Discovery ---
    print(f"\nDiscovering data folders in '{DATA_ROOT}':")
    lightcurve_dirs = discover_data_folders(DATA_ROOT)

    if not lightcurve_dirs:
        print("Error: No data folders with lightcurves found.")
        return

    print(f"Found {len(lightcurve_dirs)} data folders to process.")

    # --- File Discovery ---
    all_files = []
    for lightcurve_dir in lightcurve_dirs:
        folder_files = sorted(list(lightcurve_dir.glob("*.csv")))
        all_files.extend(folder_files)
        print(f"  {lightcurve_dir.parent.name}: {len(folder_files)} files")

    if MAX_FILES:
        all_files = all_files[:MAX_FILES]

    total_files = len(all_files)
    total_stats.total_files_found = total_files

    if total_files == 0:
        print("Error: No CSV files found in any data directories.")
        return

    print(f"\nTotal files to process: {total_files} files from {len(lightcurve_dirs)} data folders.")

    # ==========================================================================
    # PASS 1: Find the absolute maximum flux length across ALL files.
    # We do not write anything to disk in this pass.
    # ==========================================================================
    print("\n--- PASS 1: Calculating maximum number of columns needed ---")
    max_flux_length = 0
    processed_count = 0

    with mp.Pool(processes=n_workers) as pool:
        num_batches = (total_files + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            file_batch = all_files[start_idx:end_idx]

            # Process the batch and get results with statistics
            batch_results, batch_stats = process_batch(file_batch, pool)
            total_stats = merge_stats(total_stats, batch_stats)

            # Find the max length within this batch's results
            if batch_results:
                batch_max = max(res['flux_length'] for res in batch_results)
                max_flux_length = max(max_flux_length, batch_max)

            processed_count += len(file_batch)
            print(f"  Batch {i+1}/{num_batches}: Processed {processed_count}/{total_files} files. Current max length: {max_flux_length}")

    print("\n--- PASS 1 COMPLETE ---")
    print(f"The longest lightcurve requires {max_flux_length} flux columns.")
    print(f"The final CSV will have {max_flux_length + 1} total columns (ID + flux data).")

    # Reset processing stats for pass 2 (keeping file counts from pass 1)
    pass1_stats = total_stats
    total_stats = ProcessingStats()
    total_stats.total_files_found = pass1_stats.total_files_found

    # ==========================================================================
    # PASS 2: Create separate CSV files for each label.
    # Now we can safely write the files, knowing every row will fit.
    # ==========================================================================
    print("\n--- PASS 2: Writing data to separate CSV files by label ---")

    # Define the full header row
    headers = ['planet_id'] + [f'flux_{i}' for i in range(max_flux_length)]

    # Initialize CSV files and writers for each label
    label_files = {}
    label_writers = {}
    label_rows_written = {}

    for label in unique_labels:
        output_file = f"{OUTPUT_PREFIX}_{label}.csv"
        label_files[label] = open(output_file, 'w', newline='', encoding='utf-8')
        label_writers[label] = csv.writer(label_files[label])
        label_writers[label].writerow(headers)  # Write header
        label_rows_written[label] = 0
        print(f"  Created output file: {output_file}")

    try:
        with mp.Pool(processes=n_workers) as pool:
            num_batches = (total_files + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                file_batch = all_files[start_idx:end_idx]

                batch_results, batch_stats = process_batch(file_batch, pool)
                total_stats = merge_stats(total_stats, batch_stats)

                # Write each processed planet to the appropriate CSV
                for planet_data in batch_results:
                    planet_id = planet_data['planet_id']

                    # Get label for this planet
                    planet_label = label_mapping.get(planet_id)
                    if planet_label is None:
                        print(f"  Warning: No label found for {planet_id}, skipping")
                        continue

                    # Create the row with padding
                    row_data = planet_data['flux_data']
                    padded_row = row_data + [''] * (max_flux_length - len(row_data))

                    # Write to appropriate label file
                    label_writers[planet_label].writerow([planet_id] + padded_row)
                    label_rows_written[planet_label] += 1

                    # Update label-specific statistics
                    if planet_label not in total_stats.label_stats:
                        total_stats.label_stats[planet_label] = {'files': 0, 'points': 0}
                    total_stats.label_stats[planet_label]['files'] += 1
                    total_stats.label_stats[planet_label]['points'] += len(row_data)

                print(f"  Batch {i+1}/{num_batches}: Processed {len(batch_results)} files. "
                      f"Label counts: {dict(label_rows_written)}")

    finally:
        # Close all files
        for f in label_files.values():
            f.close()

    total_rows_written = sum(label_rows_written.values())
    print(f"\nFiles created:")
    for label in unique_labels:
        print(f"  Label {label}: {OUTPUT_PREFIX}_{label}.csv ({label_rows_written[label]} rows)")
    print(f"Total rows written: {total_rows_written}")

    # --- Final Statistics and Report ---
    total_time = time.time() - start_time
    total_stats.total_processing_time = total_time

    # Calculate file sizes for each label
    output_files = {}
    file_sizes_mb = {}
    for label in unique_labels:
        output_file = f"{OUTPUT_PREFIX}_{label}.csv"
        output_files[label] = output_file
        if os.path.exists(output_file):
            file_sizes_mb[label] = os.path.getsize(output_file) / (1024 * 1024)
        else:
            file_sizes_mb[label] = 0

    # Print comprehensive statistics report
    print_statistics_report(total_stats, output_files, file_sizes_mb)

if __name__ == "__main__":
    main()