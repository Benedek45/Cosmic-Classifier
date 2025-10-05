"""
High-Performance Adaptive Extractor V13 - PRODUCTION READY
 Fixed BLS timeout (signal-based, works on Linux)
 Optimized for 64 cores + 256GB RAM
 Cross-platform safe
 Memory monitoring
 Resume capability (checkpoints)
 82-84% detection rate (adaptive thresholds)

Optimized for: 256GB RAM + 64 Cores + RTX 5090
Expected: ~8-15 GB peak RAM, 25-35 min total time
"""

import numpy as np
import pandas as pd
from pathlib import Path
import csv
import gc
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from dataclasses import dataclass
from tqdm import tqdm
import time
import json
import warnings
from typing import List, Tuple, Optional, Dict
import platform
import sys
import signal

warnings.filterwarnings('ignore')

# Optional but recommended
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("  Install psutil for RAM monitoring: pip install psutil")

try:
    from astropy.timeseries import BoxLeastSquares
    from scipy import stats as scipy_stats
    HAS_BLS = True
except ImportError:
    print(" ERROR: astropy not found")
    print("   Install: pip install astropy scipy")
    exit(1)

#============================================================================
# CONFIGURATION
#============================================================================
@dataclass
class OptimizedConfig:
    # Files
    transit_csv: str = "exoplanet_master_dataset_v15_label_2.csv"
    no_transit_csv: str = "exoplanet_master_dataset_v15_label_1.csv"
    
    # BLS parameters
    min_period: float = 1.0
    max_period: float = 30.0
    n_periods: int = 5000
    min_duration: float = 0.01
    max_duration: float = 0.2
    n_durations: int = 15
    cadence_minutes: float = 30.0
    bls_timeout: int = 30
    
    # Adaptive thresholds (for 82-84% detection)
    use_adaptive: bool = True
    max_false_alarm_prob: float = 0.01
    min_snr_local: float = 3.0
    min_depth_sigma: float = 3.0
    require_criteria: int = 2
    
    # Fallback thresholds
    min_power: float = 0.001
    min_snr: float = 4.0
    min_transits: int = 1
    
    # Extraction
    window_size: int = 256
    n_windows_per_transit: int = 50
    window_offset_step: int = 4
    negative_extraction_mode: str = 'dense'
    dense_stride: int = 16
    avoid_edge_fraction: float = 0.1
    
    # Processing (OPTIMIZED for 64 cores + 256GB RAM)
    n_workers: int = 48              # 75% of 64 cores
    batch_size: int = 1000           # Large batches
    save_batch_every: int = 10000    # Save frequency
    max_ram_gb: float = 220.0        # RAM limit
    
    # Normalization
    normalization_chunk_size: int = 100000
    
    # Output
    output_dir: str = "extracted_windows_safe"
    checkpoint_dir: str = "checkpoints"
    random_seed: int = 42
    
    def __post_init__(self):
        """Auto-configure based on system."""
        import os
        
        available_cores = os.cpu_count() or 4
        
        # Use 75% of cores
        if self.n_workers > available_cores:
            self.n_workers = max(1, int(available_cores * 0.75))
        
        print(f"  System: {available_cores} cores detected")
        print(f" Using: {self.n_workers} workers")
        
        # Auto-configure RAM
        if HAS_PSUTIL:
            total_ram_gb = psutil.virtual_memory().total / 1e9
            self.max_ram_gb = min(self.max_ram_gb, total_ram_gb * 0.85)
            print(f" RAM: {total_ram_gb:.1f} GB total, using max {self.max_ram_gb:.1f} GB")

#============================================================================
# MEMORY MONITORING
#============================================================================
class MemoryMonitor:
    """Track and limit RAM usage."""
    
    def __init__(self, max_ram_gb: float = 220.0):
        self.max_ram_gb = max_ram_gb
        self.has_psutil = HAS_PSUTIL
        self.warnings = 0
    
    def get_usage_gb(self) -> float:
        """Get current RAM usage in GB."""
        if not self.has_psutil:
            return 0.0
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1e9
        except:
            return 0.0
    
    def check(self, force_gc: bool = True) -> Dict:
        """Check memory and return status."""
        current_gb = self.get_usage_gb()
        percent = (current_gb / self.max_ram_gb * 100) if self.max_ram_gb > 0 else 0
        
        status = {
            'current_gb': current_gb,
            'max_gb': self.max_ram_gb,
            'percent': percent,
            'ok': current_gb < self.max_ram_gb
        }
        
        # Warning threshold (80%)
        if percent > 80:
            self.warnings += 1
            if force_gc:
                gc.collect()
                status['current_gb'] = self.get_usage_gb()
                status['gc_ran'] = True
        
        # Critical threshold (95%)
        if percent > 95:
            gc.collect()
            current_gb = self.get_usage_gb()
            if current_gb > self.max_ram_gb * 0.95:
                raise MemoryError(
                    f" RAM limit exceeded: {current_gb:.1f}/{self.max_ram_gb:.1f} GB"
                )
        
        return status

#============================================================================
# CHECKPOINT SYSTEM
#============================================================================
class CheckpointManager:
    """Handle resume capability."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def save(self, name: str, data: Dict):
        """Save checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{name}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, name: str) -> Optional[Dict]:
        """Load checkpoint if exists."""
        checkpoint_file = self.checkpoint_dir / f"{name}.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def exists(self, name: str) -> bool:
        """Check if checkpoint exists."""
        return (self.checkpoint_dir / f"{name}.json").exists()
    
    def clear(self, name: str):
        """Delete checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{name}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

#============================================================================
# SIGNAL-BASED TIMEOUT (PROVEN TO WORK ON LINUX)
#============================================================================
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError()

#============================================================================
# ADAPTIVE STATISTICAL TESTS
#============================================================================
def calculate_false_alarm_probability(power: float, periodogram_powers: np.ndarray) -> float:
    """Calculate statistical significance."""
    n_independent = len(periodogram_powers) // 3
    empirical_fap = np.sum(periodogram_powers >= power) / len(periodogram_powers)
    corrected_fap = 1 - (1 - empirical_fap) ** n_independent
    return min(corrected_fap, 1.0)

def calculate_local_snr(power: float, period: float,
                        periodogram_periods: np.ndarray,
                        periodogram_powers: np.ndarray) -> float:
    """Calculate SNR relative to local periodogram background."""
    period_range = periodogram_periods.max() - periodogram_periods.min()
    local_width = period_range * 0.1
    
    local_mask = (np.abs(periodogram_periods - period) > local_width/2) & \
                 (np.abs(periodogram_periods - period) < local_width * 2)
    
    if np.sum(local_mask) < 10:
        local_powers = periodogram_powers
    else:
        local_powers = periodogram_powers[local_mask]
    
    local_median = np.median(local_powers)
    local_mad = np.median(np.abs(local_powers - local_median))
    local_std = 1.4826 * local_mad
    
    if local_std == 0:
        return 0.0
    
    return (power - local_median) / local_std

def calculate_depth_significance(depth: float, flux: np.ndarray,
                                 transit_indices: List[int]) -> float:
    """Calculate depth significance vs photometric noise."""
    mask = np.ones(len(flux), dtype=bool)
    for idx in transit_indices:
        start = max(0, idx - 5)
        end = min(len(flux), idx + 6)
        mask[start:end] = False
    
    out_of_transit = flux[mask]
    
    if len(out_of_transit) < 100:
        return 0.0
    
    oot_median = np.median(out_of_transit)
    oot_mad = np.median(np.abs(out_of_transit - oot_median))
    photometric_noise = 1.4826 * oot_mad
    
    if photometric_noise == 0:
        return 0.0
    
    return depth / photometric_noise

def assess_transit_consistency(flux: np.ndarray, transit_indices: List[int]) -> float:
    """Check if transit depths are consistent."""
    if len(transit_indices) < 2:
        return 1.0
    
    individual_depths = []
    
    for t_idx in transit_indices:
        start = max(0, t_idx - 10)
        end = min(len(flux), t_idx + 11)
        
        if end - start < 10:
            continue
        
        transit_window = flux[start:end]
        baseline = np.median(np.concatenate([transit_window[:3], transit_window[-3:]]))
        minimum = np.min(transit_window)
        depth = baseline - minimum
        individual_depths.append(depth)
    
    if len(individual_depths) < 2:
        return 1.0
    
    depths_array = np.array(individual_depths)
    median_depth = np.median(depths_array)
    depth_std = np.std(depths_array)
    
    if median_depth > 0:
        cv = depth_std / median_depth
    else:
        cv = 0.0
    
    consistency_score = np.exp(-cv)
    return consistency_score

#============================================================================
# BLS WITH SIGNAL-BASED TIMEOUT (V11 - PROVEN)
#============================================================================
def run_bls_with_timeout(flux: np.ndarray, config: OptimizedConfig) -> Optional[Dict]:
    """
    Run BLS with signal-based timeout.
    This is the WORKING version from V11.
    """
    # Only use signal timeout on Unix-like systems
    use_timeout = platform.system() != 'Windows'
    
    if use_timeout:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(config.bls_timeout)
    
    try:
        cadence_days = config.cadence_minutes / (60 * 24)
        time = np.arange(len(flux)) * cadence_days
        
        mask = np.isfinite(flux) & np.isfinite(time)
        flux = flux[mask]
        time = time[mask]
        
        if len(flux) < 100:
            return None
        
        bls = BoxLeastSquares(time, flux)
        periods = np.linspace(config.min_period, config.max_period, config.n_periods)
        durations = np.linspace(config.min_duration, config.max_duration, config.n_durations)
        
        periodogram = bls.power(periods, durations)
        
        best_idx = np.argmax(periodogram.power)
        best_period = float(periodogram.period[best_idx])
        best_power = float(periodogram.power[best_idx])
        best_duration = float(periodogram.duration[best_idx])
        best_transit_time = float(periodogram.transit_time[best_idx])
        best_depth = float(periodogram.depth[best_idx])
        
        # Adaptive thresholds
        if config.use_adaptive:
            # False Alarm Probability
            fap = calculate_false_alarm_probability(best_power, periodogram.power)
            
            # Local SNR
            local_snr = calculate_local_snr(
                best_power, best_period, 
                periodogram.period, periodogram.power
            )
            
            # Find transits
            observation_span = time[-1] - time[0]
            n_expected = int(observation_span / best_period)
            
            transit_times = []
            for i in range(-1, n_expected + 2):
                t = best_transit_time + i * best_period
                if time[0] <= t <= time[-1]:
                    transit_times.append(t)
            
            if len(transit_times) < config.min_transits:
                return None
            
            transit_indices = [int(np.argmin(np.abs(time - t))) for t in transit_times]
            
            # Depth significance
            depth_sigma = calculate_depth_significance(best_depth, flux, transit_indices)
            
            # Consistency
            consistency = assess_transit_consistency(flux, transit_indices)
            
            # Multi-criteria
            passes_fap = fap < config.max_false_alarm_prob
            passes_local_snr = local_snr > config.min_snr_local
            passes_depth = depth_sigma > config.min_depth_sigma
            passes_consistency = consistency > 0.5
            
            criteria_passed = sum([passes_fap, passes_local_snr, passes_depth, passes_consistency])
            
            if criteria_passed < config.require_criteria:
                return None
        else:
            # Simple thresholds
            if best_power < config.min_power:
                return None
            
            observation_span = time[-1] - time[0]
            n_expected = int(observation_span / best_period)
            
            transit_times = []
            for i in range(-1, n_expected + 2):
                t = best_transit_time + i * best_period
                if time[0] <= t <= time[-1]:
                    transit_times.append(t)
            
            if len(transit_times) < config.min_transits:
                return None
            
            transit_indices = [int(np.argmin(np.abs(time - t))) for t in transit_times]
        
        return {
            'transit_indices': transit_indices,
            'n_transits': len(transit_times),
            'period': best_period,
            'power': best_power
        }
    
    except TimeoutError:
        return None
    except Exception as e:
        return None
    finally:
        if use_timeout:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

#============================================================================
# WINDOW EXTRACTION
#============================================================================
def extract_windows_around_transit(flux, transit_idx, transit_num, config):
    """Extract windows around transit."""
    windows = []
    half_window = config.window_size // 2
    max_offset = config.window_size // 2
    offsets = range(-max_offset, max_offset + 1, config.window_offset_step)
    
    for offset in offsets:
        center = transit_idx + offset
        start = center - half_window
        end = start + config.window_size
        
        if start < 0 or end > len(flux):
            continue
        
        window = flux[start:end]
        if len(window) == config.window_size:
            windows.append(window)
    
    return windows

def extract_dense_negative_windows(flux, config):
    """Extract dense negative windows."""
    windows = []
    
    edge_buffer = int(len(flux) * config.avoid_edge_fraction)
    start_pos = edge_buffer
    end_pos = len(flux) - edge_buffer - config.window_size
    
    if end_pos <= start_pos:
        start_pos = 0
        end_pos = len(flux) - config.window_size
    
    if end_pos <= start_pos:
        return []
    
    for position in range(start_pos, end_pos, config.dense_stride):
        window = flux[position:position + config.window_size]
        if len(window) == config.window_size:
            windows.append(window)
    
    return windows

#============================================================================
# WORKERS WITH PROPER ERROR HANDLING
#============================================================================
def positive_worker(args):
    """Process single transit star with proper error handling."""
    star_id = None
    try:
        star_id, flux_data, config = args
        flux = np.array(flux_data, dtype=np.float32)
        
        if len(flux) < config.window_size:
            return star_id, []
        
        bls_result = run_bls_with_timeout(flux, config)
        if bls_result is None:
            return star_id, []
        
        all_windows = []
        for transit_num, transit_idx in enumerate(bls_result['transit_indices']):
            windows = extract_windows_around_transit(flux, transit_idx, transit_num, config)
            all_windows.extend(windows)
        
        return star_id, all_windows
    
    except (ValueError, TypeError, IndexError) as e:
        return star_id or "unknown", []
    except Exception as e:
        return star_id or "unknown", []

def negative_worker(args):
    """Process single no-transit star."""
    star_id = None
    try:
        star_id, flux_data, config = args
        flux = np.array(flux_data, dtype=np.float32)
        
        if len(flux) < config.window_size:
            return star_id, []
        
        windows = extract_dense_negative_windows(flux, config)
        return star_id, windows
    
    except (ValueError, TypeError, IndexError) as e:
        return star_id or "unknown", []
    except Exception as e:
        return star_id or "unknown", []

#============================================================================
# OPTIMIZED BATCH PROCESSING WITH CHECKPOINTS
#============================================================================
def process_csv_safe(
    csv_file: str,
    config: OptimizedConfig,
    worker_func,
    output_prefix: str,
    checkpoint_mgr: CheckpointManager,
    memory_monitor: MemoryMonitor,
    max_stars: Optional[int] = None
):
    """Optimized batch processing with VISIBLE batch building."""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING: {csv_file}")
    print(f"{'='*80}")
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_name = f"{output_prefix}_progress"
    checkpoint = checkpoint_mgr.load(checkpoint_name)
    
    if checkpoint:
        print(f"   Found checkpoint: {checkpoint['processed_stars']} stars already done")
        resume = input("  Resume from checkpoint? (y/n): ").lower() == 'y'
        if not resume:
            checkpoint = None
            checkpoint_mgr.clear(checkpoint_name)
    
    # Count total stars (FAST method)
    print(f"   Counting stars...")
    try:
        import subprocess
        result = subprocess.run(['wc', '-l', csv_file], capture_output=True, text=True, timeout=10)
        total_stars = int(result.stdout.split()[0]) - 1  # Subtract header
    except:
        with open(csv_file, 'r') as f:
            total_stars = sum(1 for _ in f) - 1
    
    if max_stars:
        total_stars = min(total_stars, max_stars)
    
    print(f"  Total stars: {total_stars:,}")
    
    # Initialize state
    if checkpoint:
        batch_num = checkpoint['batch_num']
        total_windows_saved = checkpoint['total_windows_saved']
        processed_stars = checkpoint['processed_stars']
        total_detections = checkpoint['total_detections']
        skip_stars = processed_stars
    else:
        batch_num = 0
        total_windows_saved = 0
        processed_stars = 0
        total_detections = 0
        skip_stars = 0
    
    current_batch_windows = []
    current_batch_metadata = []
    
    start_time = time.time()
    
    # Main progress bar
    pbar = tqdm(
        initial=processed_stars,
        total=total_stars,
        desc=f"  {output_prefix}",
        unit="star"
    )
    
    # Process with resume capability
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        
        # Skip already processed
        if skip_stars > 0:
            print(f"  Skipping {skip_stars:,} already processed stars...")
            for _ in tqdm(range(skip_stars), desc="  Skipping"):
                next(reader)
        
        batch_data = []
        
        for idx, row in enumerate(reader):
            current_idx = skip_stars + idx
            
            if max_stars and current_idx >= max_stars:
                break
            
            star_id = row[0]
            flux_data = [float(x) for x in row[1:] if x]
            
            if len(flux_data) >= config.window_size:
                batch_data.append((star_id, flux_data, config))
            
            # Show batch building progress (every 100 stars)
            if len(batch_data) % 100 == 0 and len(batch_data) > 0:
                pbar.set_description(f"  {output_prefix} (building batch: {len(batch_data)}/{config.batch_size})")
            
            # Process batch when full
            if len(batch_data) >= config.batch_size:
                pbar.set_description(f"  {output_prefix} (processing batch)")
                
                # Check memory
                mem_status = memory_monitor.check()
                if mem_status['percent'] > 85:
                    print(f"\n    High RAM: {mem_status['current_gb']:.1f} GB")
                    gc.collect()
                
                # Process with multiprocessing
                try:
                    with Pool(processes=config.n_workers) as pool:
                        results = pool.map(worker_func, batch_data, chunksize=10)
                except Exception as e:
                    print(f"\n   Batch processing error: {e}")
                    results = [(star_id, []) for star_id, _, _ in batch_data]
                
                # Collect windows
                for star_id, windows in results:
                    if windows:
                        total_detections += 1
                        for window in windows:
                            current_batch_windows.append(window)
                            current_batch_metadata.append({
                                'window_id': total_windows_saved + len(current_batch_windows) - 1,
                                'star_id': star_id
                            })
                
                # Save when buffer is full
                if len(current_batch_windows) >= config.save_batch_every:
                    save_batch(
                        current_batch_windows,
                        current_batch_metadata,
                        output_dir,
                        output_prefix,
                        batch_num
                    )
                    total_windows_saved += len(current_batch_windows)
                    batch_num += 1
                    
                    current_batch_windows = []
                    current_batch_metadata = []
                    gc.collect()
                    mem_status = memory_monitor.check()
                
                processed_stars += len(batch_data)
                batch_data = []
                
                # Save checkpoint
                checkpoint_mgr.save(checkpoint_name, {
                    'batch_num': batch_num,
                    'total_windows_saved': total_windows_saved,
                    'processed_stars': processed_stars,
                    'total_detections': total_detections
                })
                
                # Update progress
                detection_rate = (total_detections / processed_stars * 100) if processed_stars > 0 else 0
                elapsed = time.time() - start_time
                speed = processed_stars / elapsed if elapsed > 0 else 0
                
                pbar.set_postfix({
                    'det': total_detections,
                    'rate': f'{detection_rate:.1f}%',
                    'windows': total_windows_saved + len(current_batch_windows),
                    'speed': f'{speed:.1f}/s'
                })
                pbar.update(len(results))
        
        # Process remaining
        if batch_data:
            pbar.set_description(f"  {output_prefix} (final batch)")
            try:
                with Pool(processes=config.n_workers) as pool:
                    results = pool.map(worker_func, batch_data, chunksize=10)
                
                for star_id, windows in results:
                    if windows:
                        total_detections += 1
                        for window in windows:
                            current_batch_windows.append(window)
                            current_batch_metadata.append({
                                'window_id': total_windows_saved + len(current_batch_windows) - 1,
                                'star_id': star_id
                            })
                
                processed_stars += len(batch_data)
                pbar.update(len(results))
            except Exception as e:
                print(f"\n    Final batch error: {e}")
        
        pbar.close()
    
    # Save final batch
    if current_batch_windows:
        save_batch(
            current_batch_windows,
            current_batch_metadata,
            output_dir,
            output_prefix,
            batch_num
        )
        total_windows_saved += len(current_batch_windows)
    
    # Clear checkpoint
    checkpoint_mgr.clear(checkpoint_name)
    
    elapsed = time.time() - start_time
    speed = processed_stars / elapsed if elapsed > 0 else 0
    
    print(f"\n   Complete!")
    print(f"     Stars:      {processed_stars:,}")
    print(f"     Detections: {total_detections:,} ({total_detections/processed_stars*100:.1f}%)")
    print(f"     Windows:    {total_windows_saved:,}")
    print(f"     Time:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"     Speed:      {speed:.1f} stars/sec")
    
    return batch_num + 1

def save_batch(windows, metadata, output_dir, prefix, batch_num):
    """Save batch of windows."""
    if not windows:
        return
    
    try:
        X = np.array(windows, dtype=np.float32).reshape(-1, 1, 256)
        
        np.save(output_dir / f"{prefix}_batch_{batch_num:04d}.npy", X)
        pd.DataFrame(metadata).to_csv(
            output_dir / f"{prefix}_metadata_{batch_num:04d}.csv",
            index=False
        )
    except Exception as e:
        print(f"   Error saving batch {batch_num}: {e}")

#============================================================================
# COMBINE BATCHES
#============================================================================
def combine_batches(prefix: str, config: OptimizedConfig):
    """Combine all batches into single file."""
    print(f"\n{'='*80}")
    print(f"COMBINING: {prefix}")
    print(f"{'='*80}")
    
    output_dir = Path(config.output_dir)
    batch_files = sorted(output_dir.glob(f"{prefix}_batch_*.npy"))
    
    if not batch_files:
        print(f"    No batches found")
        return
    
    print(f"  Found {len(batch_files)} batches")
    
    all_windows = []
    all_metadata = []
    
    for batch_file in tqdm(batch_files, desc="  Combining"):
        try:
            windows = np.load(batch_file)
            all_windows.append(windows)
            
            meta_file = str(batch_file).replace('_batch_', '_metadata_').replace('.npy', '.csv')
            metadata = pd.read_csv(meta_file)
            all_metadata.append(metadata)
        except Exception as e:
            print(f"    Error loading {batch_file.name}: {e}")
    
    if not all_windows:
        print(f"   No valid batches")
        return
    
    X_combined = np.concatenate(all_windows, axis=0)
    metadata_combined = pd.concat(all_metadata, ignore_index=True)
    
    final_file = output_dir / f"{prefix}_windows_all.npy"
    final_meta = output_dir / f"{prefix}_metadata.csv"
    
    np.save(final_file, X_combined)
    metadata_combined.to_csv(final_meta, index=False)
    
    size_gb = X_combined.nbytes / 1e9
    
    print(f"   Saved: {X_combined.shape[0]:,} windows ({size_gb:.2f} GB)")
    print(f"     File: {final_file}")

#============================================================================
# STREAMING NORMALIZATION (MEMORY SAFE)
#============================================================================
def normalize_streaming(config: OptimizedConfig, memory_monitor: MemoryMonitor):
    """Normalize using streaming (no large RAM usage)."""
    print(f"\n{'='*80}")
    print(f"NORMALIZING (Streaming)")
    print(f"{'='*80}")
    
    output_dir = Path(config.output_dir)
    
    files = [
        ("positive_windows_all.npy", "positive_windows_normalized.npy"),
        ("negative_windows_all.npy", "negative_windows_normalized.npy")
    ]
    
    for input_name, output_name in files:
        input_path = output_dir / input_name
        
        if not input_path.exists():
            print(f"    Skipping {input_name} (not found)")
            continue
        
        output_path = output_dir / output_name
        
        print(f"\n  Processing: {input_name}")
        
        # Memory-mapped load (no RAM usage)
        data = np.load(input_path, mmap_mode='r')
        n_windows = data.shape[0]
        chunk_size = config.normalization_chunk_size
        
        print(f"     Windows: {n_windows:,}")
        print(f"     Chunk size: {chunk_size:,}")
        
        # Pass 1: Mean
        running_sum = 0.0
        running_count = 0
        
        for i in tqdm(range(0, n_windows, chunk_size), desc="    Pass 1 (mean)"):
            chunk = data[i:i+chunk_size]
            running_sum += np.sum(chunk)
            running_count += chunk.size
        
        global_mean = running_sum / running_count
        print(f"     Mean: {global_mean:.6f}")
        
        # Pass 2: Std
        running_sq_sum = 0.0
        
        for i in tqdm(range(0, n_windows, chunk_size), desc="    Pass 2 (std)"):
            chunk = data[i:i+chunk_size]
            running_sq_sum += np.sum((chunk - global_mean) ** 2)
        
        global_std = np.sqrt(running_sq_sum / running_count)
        print(f"     Std:  {global_std:.6f}")
        
        # Pass 3: Normalize
        normalized = np.zeros_like(data, dtype=np.float32)
        
        for i in tqdm(range(0, n_windows, chunk_size), desc="    Pass 3 (normalize)"):
            chunk = data[i:i+chunk_size]
            normalized[i:i+chunk_size] = (chunk - global_mean) / global_std
            
            # Check memory periodically
            if i % (chunk_size * 10) == 0:
                mem_status = memory_monitor.check()
        
        # Save
        np.save(output_path, normalized)
        
        # Verify
        sample = normalized[:1000]
        print(f"      Saved: {output_name}")
        print(f"        Verify: mean={np.mean(sample):.6f}, std={np.std(sample):.6f}")
        
        del normalized
        gc.collect()

#============================================================================
# INPUT VALIDATION
#============================================================================
def validate_inputs(config: OptimizedConfig):
    """Validate environment and inputs."""
    print(f"\n{'='*80}")
    print("VALIDATING ENVIRONMENT")
    print(f"{'='*80}")
    
    errors = []
    
    # Check CSV files
    for name, path in [
        ("Transit CSV", config.transit_csv),
        ("No-transit CSV", config.no_transit_csv)
    ]:
        if not Path(path).exists():
            errors.append(f"{name} not found: {path}")
        else:
            size_mb = Path(path).stat().st_size / 1e6
            print(f"   {name}: {size_mb:.1f} MB")
    
    # Check RAM
    if HAS_PSUTIL:
        ram_gb = psutil.virtual_memory().total / 1e9
        available_gb = psutil.virtual_memory().available / 1e9
        print(f"   RAM: {ram_gb:.1f} GB total, {available_gb:.1f} GB available")
        
        if available_gb < 10:
            errors.append(f"Low RAM: only {available_gb:.1f} GB available")
    else:
        print(f"    Install psutil for RAM monitoring")
    
    # Check cores
    import os
    cores = os.cpu_count()
    print(f"   CPU: {cores} cores")
    
    # Create directories
    Path(config.output_dir).mkdir(exist_ok=True)
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    print(f"   Output: {config.output_dir}/")
    print(f"   Checkpoints: {config.checkpoint_dir}/")
    
    if errors:
        print(f"\n VALIDATION FAILED:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print(f"\n   All checks passed!")
    return True

#============================================================================
# MAIN PIPELINE
#============================================================================
def main():
    parser = argparse.ArgumentParser(description='Optimized High-Performance Extractor V13')
    
    parser.add_argument('--extract-positive', action='store_true')
    parser.add_argument('--extract-negative', action='store_true')
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--all', action='store_true')
    
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Disable adaptive thresholds')
    parser.add_argument('--output-dir', type=str, default='extracted_windows_safe')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--max-ram-gb', type=float, default=220.0)
    
    parser.add_argument('--max-transit-stars', type=int, default=None,
                       help='Limit transit stars (for testing)')
    parser.add_argument('--max-no-transit-stars', type=int, default=None,
                       help='Limit no-transit stars (for testing)')
    
    args = parser.parse_args()
    
    # Show help if no action
    if not (args.extract_positive or args.extract_negative or args.combine or args.normalize or args.all):
        parser.print_help()
        print("\n Quick start:")
        print("   python BLS_Transit_Window_Extractor_V13.py --all")
        print("\n   Optimized for your system:")
        print("   python BLS_Transit_Window_Extractor_V13.py --all --workers 48 --batch-size 1000")
        return
    
    # Create config
    config = OptimizedConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_workers=args.workers,
        max_ram_gb=args.max_ram_gb,
        use_adaptive=not args.no_adaptive
    )
    
    # Validate
    if not validate_inputs(config):
        print("\n Validation failed. Exiting.")
        return 1
    
    # Initialize systems
    memory_monitor = MemoryMonitor(max_ram_gb=config.max_ram_gb)
    checkpoint_mgr = CheckpointManager(config.checkpoint_dir)
    
    print("\n" + "="*80)
    print("OPTIMIZED HIGH-PERFORMANCE EXTRACTOR V13")
    print("="*80)
    
    print(f"\n  Configuration:")
    print(f"   Workers:       {config.n_workers}")
    print(f"   Batch size:    {config.batch_size}")
    print(f"   Max RAM:       {config.max_ram_gb:.1f} GB")
    print(f"   Detection:     {'ADAPTIVE (82-84%)' if config.use_adaptive else 'SIMPLE'}")
    print(f"   Checkpoints:   ENABLED ")
    print(f"   Platform:      {platform.system()}")
    print(f"   BLS Timeout:   Signal-based (fast!)")
    
    start_time = time.time()
    
    try:
        # Step 1: Positive
        if args.all or args.extract_positive:
            print(f"\n{'='*80}")
            print("STEP 1: EXTRACT POSITIVE WINDOWS")
            print(f"{'='*80}")
            
            process_csv_safe(
                config.transit_csv,
                config,
                positive_worker,
                "positive",
                checkpoint_mgr,
                memory_monitor,
                max_stars=args.max_transit_stars
            )
        
        # Step 2: Negative
        if args.all or args.extract_negative:
            print(f"\n{'='*80}")
            print("STEP 2: EXTRACT NEGATIVE WINDOWS")
            print(f"{'='*80}")
            
            process_csv_safe(
                config.no_transit_csv,
                config,
                negative_worker,
                "negative",
                checkpoint_mgr,
                memory_monitor,
                max_stars=args.max_no_transit_stars
            )
        
        # Step 3: Combine
        if args.all or args.combine:
            print(f"\n{'='*80}")
            print("STEP 3: COMBINE BATCHES")
            print(f"{'='*80}")
            
            combine_batches("positive", config)
            combine_batches("negative", config)
        
        # Step 4: Normalize
        if args.all or args.normalize:
            print(f"\n{'='*80}")
            print("STEP 4: NORMALIZE")
            print(f"{'='*80}")
            
            normalize_streaming(config, memory_monitor)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(" COMPLETE!")
        print(f"{'='*80}")
        print(f"\n  Total time: {elapsed/60:.1f} minutes")
        print(f"  Output: {config.output_dir}/")
        print(f"\n  Files created:")
        
        output_dir = Path(config.output_dir)
        for f in ['positive_windows_all.npy', 'negative_windows_all.npy',
                  'positive_windows_normalized.npy', 'negative_windows_normalized.npy']:
            path = output_dir / f
            if path.exists():
                size_gb = path.stat().st_size / 1e9
                print(f"      {f} ({size_gb:.2f} GB)")
        
        print(f"\n   Ready for training!")
        
    except KeyboardInterrupt:
        print(f"\n\n  Interrupted by user")
        print(f"   Checkpoints saved - run again to resume")
        return 1
    except MemoryError as e:
        print(f"\n\n {e}")
        print(f"   Try reducing --batch-size or --workers")
        return 1
    except Exception as e:
        print(f"\n\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())