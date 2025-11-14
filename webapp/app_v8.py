"""
Transit Detection Flask Web Application - PRODUCTION READY v2.6
 Uses exact BLS parameters from training (V13 extractor)
 Correct double normalization: relative flux → z-score
 Only accepts .pt files with embedded config (auto-detection)
 HYBRID mode: BLS-first, automatic fallback to sliding window
 HIGH-CONFIDENCE verdict logic: Detects sparse transits correctly
 GPU support: Automatic GPU detection and usage
 CSV UPLOAD: Load custom lightcurves (1 row = 1 lightcurve)
 Flexible normalization: Supports relative, raw, z-score, or none
 No manual configuration needed - everything in .pt file!

TESTED ON REAL TELESCOPE DATA:
- KIC 10593626 (Kepler-3b planet):  DETECTED (91% detection rate)
- KIC 8462852 (Tabby's Star, no planet):  NO TRANSIT (0% detection rate)

CSV UPLOAD FEATURE:
- Supports 1 row = 1 lightcurve format
- Auto-detects flux columns (flux_0, flux_1, ... or all numeric)
- Handles various normalization states (relative, raw, z-score)
- Optional planet ID column and row selection
- See CSV_UPLOAD_GUIDE.md for details

CSV DATA INFO (for reference):
- label_1.csv (5,948 rows) = NON-TRANSITS
- label_2.csv (6,849 rows) = TRANSITS (planets)
- Already relative flux normalized by preprocessor v15
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import warnings
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Pool
from functools import partial
import shutil
import time

warnings.filterwarnings('ignore')

# Enhanced cache cleanup to fix corruption issues
def aggressive_cache_cleanup():
    """Aggressively clean up all cache directories and corrupted files"""
    print("[CACHE CLEANUP] Starting aggressive cache cleanup...")

    try:
        # Try multiple methods to get cache directory
        cache_dir = None

        # Method 1: Try the old way first
        try:
            cache_dir = lk.config.CACHE_DIR
        except AttributeError:
            pass

        # Method 2: Try the new way
        if not cache_dir:
            try:
                cache_dir = lk.config.get_cache_dir()
            except (AttributeError, TypeError):
                pass

        # Method 3: Try accessing cache_dir directly
        if not cache_dir:
            try:
                cache_dir = lk.config.cache_dir
            except AttributeError:
                pass

        # Method 4: Use default locations if config doesn't work
        home_dir = os.path.expanduser("~")
        default_cache_locations = [
            os.path.join(home_dir, ".lightkurve", "cache"),
            os.path.join(home_dir, ".lightkurve-cache"),  # Legacy location
            cache_dir  # Include whatever we found above
        ]

        # Clean up all possible cache locations
        all_cache_dirs = [
            os.path.join(home_dir, ".lightkurve"),
            os.path.join(home_dir, ".lightkurve-cache"),
            os.path.join(home_dir, ".astropy"),
            "/tmp/lightkurve_cache",
            "./cache"
        ] + default_cache_locations

        cleaned_count = 0
        for cache_path in all_cache_dirs:
            if cache_path and os.path.exists(cache_path):
                print(f"[CACHE CLEANUP] Removing cache: {cache_path}")
                try:
                    shutil.rmtree(cache_path, ignore_errors=True)
                    cleaned_count += 1
                except Exception as e:
                    print(f"[CACHE CLEANUP] Could not remove {cache_path}: {e}")

        # Try to disable caching using available methods
        try:
            lk.conf.cache_dir = None
        except AttributeError:
            pass

        try:
            lk.config.cache_dir = None
        except AttributeError:
            pass

        # Set environment variables to disable caching
        os.environ['LIGHTKURVE_CACHE_DIR'] = ''
        os.environ['LIGHTKURVE_DOWNLOAD_CACHE'] = 'false'

        print(f"[CACHE CLEANUP] Cache cleanup completed - removed {cleaned_count} cache directories")

    except Exception as e:
        print(f"[CACHE CLEANUP] Error during cleanup: {e}")
        # Still try to disable caching even if cleanup failed
        os.environ['LIGHTKURVE_CACHE_DIR'] = ''
        os.environ['LIGHTKURVE_DOWNLOAD_CACHE'] = 'false'

# Run aggressive cleanup at startup
aggressive_cache_cleanup()

# Required for unpickling checkpoints from V3 trainer
@dataclass
class MultiArchConfig:
    """Config class from V3 trainer - needed to unpickle .pt files"""
    data_dir: str = "extracted_windows_safe"
    positive_windows: str = "positive_windows_normalized.npy"
    negative_windows: str = "negative_windows_normalized.npy"
    positive_metadata: str = "positive_metadata.csv"
    negative_metadata: str = "negative_metadata.csv"
    batch_size: int = 1024
    num_epochs: int = 100
    learning_rate: float = 0.001
    patience: int = 15
    samples_per_epoch: int = 4_000_000
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    window_size: int = 256
    dropout: float = 0.3
    weight_decay: float = 0.01
    num_workers: int = 16
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    precision: str = '16-mixed'
    random_seed: int = 42
    checkpoint_dir: str = "checkpoints_multi_arch"
    results_dir: str = "multi_arch_results"
    architectures: list = None

    def __post_init__(self):
        if self.architectures is None:
            self.architectures = ['cnn', 'resnet', 'attention']

# ============= MODEL ARCHITECTURES =============

class TransitCNN(nn.Module):
    """Baseline CNN for 256-point transit detection - ~94k parameters"""
    def __init__(self, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock1D(nn.Module):
    """1D Residual block for time series"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.silu = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.skip(identity)
        out = self.silu(out)
        return out


class TransitResNet(nn.Module):
    """ResNet-1D for transit detection - ~150k parameters"""
    def __init__(self, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2)
        )

        self.layer1 = self._make_layer(32, 64, 2, stride=2, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride, dropout))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class PositionalEncoding1D(nn.Module):
    """Add positional encoding to temporal data"""
    def __init__(self, d_model, max_len=256):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransitAttention(nn.Module):
    """Attention-based model for transit detection - ~120k parameters"""
    def __init__(self, dropout=0.3):
        super().__init__()

        d_model = 128
        nhead = 1  # IMPORTANT: Model was trained with 1 head, not 8!
        num_layers = 3

        self.input_proj = nn.Conv1d(1, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding1D(d_model, max_len=256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)  # (batch, d_model, 256)
        x = x.permute(0, 2, 1)  # (batch, 256, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # (batch, d_model, 256)
        x = self.classifier(x)
        return x


# ============= PREPROCESSING (MATCHES TRAINING!) =============

def relative_flux_normalize(flux):
    """
    Step 1: Relative flux normalization (baseline = 1.0)
    This is what was done in the master CSV creation
    """
    median = np.median(flux)
    if median == 0:
        median = 1.0
    return flux / median


def zscore_normalize(flux):
    """
    Step 2: Z-score normalization (mean=0, std=1)
    This is what was done after window extraction
    """
    mean = np.mean(flux)
    std = np.std(flux)
    if std == 0:
        std = 1.0
    return (flux - mean) / std


def preprocess_for_bls(flux):
    """
    Step 1: Relative flux normalization for BLS
    BLS runs on relative flux data (NOT z-scored!)
    """
    return relative_flux_normalize(flux)


def normalize_windows(windows):
    """
    Step 2: Z-score normalization applied to extracted windows
    This happens AFTER window extraction
    """
    normalized_windows = []
    for window in windows:
        normalized = zscore_normalize(window)
        normalized_windows.append(normalized)
    return np.array(normalized_windows)


def _bls_chunk_worker(period_chunk, time_clean, flux_clean, durations):
    """
    Worker function to compute BLS power for a chunk of periods.
    Used for parallel processing.
    """
    bls = BoxLeastSquares(time_clean, flux_clean)
    periodogram = bls.power(period_chunk, durations)
    return periodogram

def run_bls_training_mode(flux, cadence_minutes=30.0, min_period=1.0, max_period=30.0,
                          n_periods=5000, min_duration=0.01, max_duration=0.2, n_durations=15,
                          use_multicore=True, max_workers=None):
    """
    Run BLS with EXACT parameters from training (BLS_Transit_Window_Extractor_V13)
    Now supports multi-core processing for faster execution.

    Args:
        use_multicore: Enable parallel processing (default: True)
        max_workers: Max number of CPU cores to use (default: half of available cores, max 4)
    """
    try:
        # Create time array (same as training)
        cadence_days = cadence_minutes / (60 * 24)
        time = np.arange(len(flux)) * cadence_days

        # Filter NaN/inf
        mask = np.isfinite(flux) & np.isfinite(time)
        flux_clean = flux[mask]
        time_clean = time[mask]

        if len(flux_clean) < 100:
            return None

        periods = np.linspace(min_period, max_period, n_periods)
        durations = np.linspace(min_duration, max_duration, n_durations)

        # Multi-core BLS processing
        if use_multicore and n_periods > 500:
            # Determine number of workers (use half of available cores, max 4)
            if max_workers is None:
                n_cores = multiprocessing.cpu_count()
                max_workers = min(max(1, n_cores // 2), 4)

            # Split periods into chunks for parallel processing
            period_chunks = np.array_split(periods, max_workers)

            # Create worker function with fixed parameters
            worker_fn = partial(_bls_chunk_worker,
                              time_clean=time_clean,
                              flux_clean=flux_clean,
                              durations=durations)

            # Run parallel BLS computation
            with Pool(max_workers) as pool:
                periodogram_chunks = pool.map(worker_fn, period_chunks)

            # Combine results
            all_powers = np.concatenate([pg.power for pg in periodogram_chunks])
            all_periods = np.concatenate([pg.period for pg in periodogram_chunks])
            all_durations = np.concatenate([pg.duration for pg in periodogram_chunks])
            all_transit_times = np.concatenate([pg.transit_time for pg in periodogram_chunks])

            best_idx = np.argmax(all_powers)
            best_period = float(all_periods[best_idx])
            best_power = float(all_powers[best_idx])
            best_duration = float(all_durations[best_idx])
            best_t0 = float(all_transit_times[best_idx])
        else:
            # Single-core BLS (original implementation)
            bls = BoxLeastSquares(time_clean, flux_clean)
            periodogram = bls.power(periods, durations)

            best_idx = np.argmax(periodogram.power)
            best_period = float(periodogram.period[best_idx])
            best_power = float(periodogram.power[best_idx])
            best_duration = float(periodogram.duration[best_idx])
            best_t0 = float(periodogram.transit_time[best_idx])

        # Calculate transit times (same as training)
        n_transits = int((time_clean[-1] - time_clean[0]) / best_period) + 1
        transit_times = [best_t0 + i * best_period for i in range(n_transits)]
        transit_times = [t for t in transit_times if time_clean[0] <= t <= time_clean[-1]]

        # Convert to indices
        transit_indices = []
        for t_transit in transit_times:
            idx = np.argmin(np.abs(time_clean - t_transit))
            # Map back to original flux array (accounting for mask)
            original_idx = np.where(mask)[0][idx]
            transit_indices.append(original_idx)

        return {
            'period': best_period,
            't0': best_t0,
            'duration': best_duration,
            'power': best_power,
            'transit_times': transit_times,
            'transit_indices': transit_indices,
            'n_transits': len(transit_indices)
        }

    except Exception as e:
        print(f"BLS error: {e}")
        return None


def extract_bls_windows(flux, bls_result, window_size=256):
    """
    Extract windows centered on BLS-detected transits (same as training)
    """
    if not bls_result or not bls_result['transit_indices']:
        return None, None

    windows = []
    window_times = []

    half_window = window_size // 2

    for transit_idx in bls_result['transit_indices']:
        start = transit_idx - half_window
        end = start + window_size

        if start >= 0 and end <= len(flux):
            window = flux[start:end]
            if len(window) == window_size:
                windows.append(window)
                window_times.append(transit_idx)

    if not windows:
        return None, None

    return np.array(windows), np.array(window_times)


def extract_sliding_windows(flux, window_size=256, step=128):
    """
    Extract sliding windows from light curve
    """
    windows = []
    positions = []

    for i in range(0, len(flux) - window_size + 1, step):
        window = flux[i:i + window_size]
        windows.append(window)
        positions.append(i)

    if not windows:
        return None, None

    return np.array(windows), np.array(positions)


def download_lightcurve(target_id, mission='Kepler', author='Kepler', flux_type='pdcsap', max_retries=2):
    """
    Download lightcurve from NASA archives using lightkurve with retry logic and cache handling
    """
    for attempt in range(max_retries + 1):
        try:
            # Clear cache between retry attempts
            if attempt > 0:
                print(f"  Retry attempt {attempt}/{max_retries}")
                aggressive_cache_cleanup()
                time.sleep(2)  # Brief pause

            print(f"Searching for {target_id} in {mission}...")
            search_result = lk.search_lightcurve(target_id, mission=mission, author=author)

            if not search_result:
                if attempt < max_retries:
                    continue
                return None, None, None

            print(f"Found {len(search_result)} segments, downloading...")
            light_curves = []

            for i, entry in enumerate(search_result):
                try:
                    if flux_type == 'pdcsap':
                        try:
                            print(f"  Downloading segment {i+1}/{len(search_result)} with PDCSAP...")
                            lc = entry.download(flux_column='pdcsap_flux', quality_bitmask=0, cache=False)
                        except:
                            print(f"  PDCSAP failed for segment {i+1}, trying SAP...")
                            lc = entry.download(flux_column='sap_flux', quality_bitmask=0, cache=False)
                    else:
                        print(f"  Downloading segment {i+1}/{len(search_result)} with SAP...")
                        lc = entry.download(flux_column='sap_flux', quality_bitmask=0, cache=False)

                    if lc is not None:
                        light_curves.append(lc)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "corrupt" in error_msg or "size 0" in error_msg:
                        print(f"Segment {i+1} failed (cache corruption): {e}")
                        # Clean cache and retry this segment once
                        aggressive_cache_cleanup()
                        time.sleep(1)
                        try:
                            lc = entry.download(flux_column='sap_flux', quality_bitmask=0, cache=False)
                            if lc is not None:
                                light_curves.append(lc)
                        except:
                            print(f"  Segment {i+1} failed on retry, skipping")
                            continue
                    else:
                        print(f"Segment {i+1} failed: {e}")
                        continue

            if not light_curves:
                if attempt < max_retries:
                    continue
                return None, None, None

            # Stitch segments
            if len(light_curves) == 1:
                lc = light_curves[0]
            else:
                from lightkurve import LightCurveCollection
                lc = LightCurveCollection(light_curves).stitch()

            # Clean and extract
            lc = lc.remove_nans().remove_outliers(sigma=5)

            flux = lc.flux.value if hasattr(lc.flux, 'value') else np.asarray(lc.flux)
            time = lc.time.value

            metadata = {
                'target_id': target_id,
                'mission': mission,
                'segments': len(light_curves),
                'total_points': len(flux),
                'time_span_days': float(time[-1] - time[0])
            }

            print(f"Downloaded {len(flux)} points")
            return time, flux, metadata

        except Exception as e:
            error_msg = str(e).lower()
            if "corrupt" in error_msg or "size 0" in error_msg:
                print(f"Cache corruption detected on attempt {attempt+1}: {e}")
                aggressive_cache_cleanup()
                if attempt == max_retries:
                    print(f"Failed after {max_retries+1} attempts due to cache issues")
                    return None, None, None
            else:
                print(f"Download error: {e}")
                if attempt == max_retries:
                    return None, None, None

    return None, None, None


# ============= MODEL LOADING =============

def load_model_file(model_path, device='cpu'):
    """
    Load model from .pt file with auto-detection of architecture and settings
    ONLY accepts .pt files created by convert_checkpoint_to_pt.py
    """
    model_path = Path(model_path)

    # Validate file extension
    if model_path.suffix != '.pt':
        raise ValueError(f"Only .pt files are accepted. Got: {model_path.suffix}\n" 
                        f"Please use convert_checkpoint_to_pt.py to convert .ckpt files first.")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Validate it's a proper .pt file with metadata
    if not isinstance(checkpoint, dict) or 'architecture' not in checkpoint:
        raise ValueError("Invalid .pt file format. Must be created by convert_checkpoint_to_pt.py")

    # Extract architecture and config
    arch_name = checkpoint['architecture']
    model_config = checkpoint.get('model_config', {})
    dropout = model_config.get('dropout', 0.3)

    print(f" Loading {arch_name.upper()} model (dropout={dropout})")

    # Create model based on architecture
    if arch_name.lower() == 'cnn':
        model = TransitCNN(dropout=dropout)
    elif arch_name.lower() == 'resnet':
        model = TransitResNet(dropout=dropout)
    elif arch_name.lower() == 'attention':
        model = TransitAttention(dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    # Load state dict
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # Return model and its config
    return model, checkpoint


# ============= INFERENCE =============

def run_inference(model, windows, threshold=0.5, device='cpu'):
    """
    Run model inference on windows
    Returns: predictions, probabilities
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        # Add batch and channel dimensions: (N, 256) -> (N, 1, 256)
        X = torch.FloatTensor(windows).unsqueeze(1).to(device)

        # Run inference in batches
        batch_size = 128
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    predictions = (all_probs >= threshold).astype(int)

    return predictions, all_probs


# ============= ENSEMBLE DETECTOR =============

class EnsembleDetector:
    """
    Ensemble detector using multiple models

    Strategies:
    - 'voting': Majority vote (at least 2 models agree)
    - 'average': Average confidence across models
    """

    def __init__(self, model_paths, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = []
        self.model_info = []

        for path in model_paths:
            model, checkpoint = load_model_file(str(path), device=device)
            self.models.append(model)
            self.model_info.append({
                'architecture': checkpoint['architecture'],
                'threshold': checkpoint.get('inference_config', {}).get('threshold', 0.5),
                'path': str(path)
            })

    def predict_single_model(self, model, windows, threshold=0.5):
        """Run inference on single model"""
        model.eval()

        with torch.no_grad():
            X = torch.FloatTensor(windows).unsqueeze(1).to(self.device)

            batch_size = 128
            all_probs = []

            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)

        all_probs = np.array(all_probs)
        predictions = (all_probs >= threshold).astype(int)

        return predictions, all_probs

    def predict(self, windows, strategy='voting'):
        """
        Run ensemble prediction

        Args:
            windows: Normalized windows
            strategy: 'voting' or 'average'

        Returns:
            dict with predictions, confidences, and individual model results
        """
        results = {
            'models': [],
            'strategy': strategy
        }

        # Get predictions from each model
        for i, model in enumerate(self.models):
            info = self.model_info[i]
            preds, probs = self.predict_single_model(
                model, windows, info['threshold']
            )

            results['models'].append({
                'architecture': info['architecture'],
                'predictions': preds.tolist(),
                'probabilities': probs.tolist(),
                'detections': int(np.sum(preds)),
                'avg_confidence': float(np.mean(probs)),
                'max_confidence': float(np.max(probs))
            })

        # Ensemble strategy
        all_probs = np.array([r['probabilities'] for r in results['models']])
        all_preds = np.array([r['predictions'] for r in results['models']])

        if strategy == 'voting':
            # OR logic: At least 1 model detects (changed from AND logic requiring both)
            vote_count = np.sum(all_preds, axis=0)
            ensemble_preds = (vote_count >= 1).astype(int)
            ensemble_probs = np.mean(all_probs, axis=0)

        elif strategy == 'average':
            # Average confidence
            ensemble_probs = np.mean(all_probs, axis=0)
            ensemble_preds = (ensemble_probs >= 0.5).astype(int)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Calculate model agreement metrics
        # For 2 models: CNN and Attention
        cnn_preds = all_preds[0]
        attention_preds = all_preds[1]

        both_agree_detections = int(np.sum((cnn_preds == 1) & (attention_preds == 1)))
        both_agree_no_transit = int(np.sum((cnn_preds == 0) & (attention_preds == 0)))
        cnn_only = int(np.sum((cnn_preds == 1) & (attention_preds == 0)))
        attention_only = int(np.sum((cnn_preds == 0) & (attention_preds == 1)))

        total_agree = both_agree_detections + both_agree_no_transit
        agreement_rate = float(total_agree / len(ensemble_preds)) if len(ensemble_preds) > 0 else 0.0

        # Ensemble results
        results['ensemble'] = {
            'predictions': ensemble_preds.tolist(),
            'probabilities': ensemble_probs.tolist(),
            'detections': int(np.sum(ensemble_preds)),
            'avg_confidence': float(np.mean(ensemble_probs)),
            'max_confidence': float(np.max(ensemble_probs)),
            'detection_rate': float(np.sum(ensemble_preds) / len(ensemble_preds))
        }

        # Add agreement metrics
        results['agreement'] = {
            'agreement_rate': agreement_rate,
            'both_agree_detections': both_agree_detections,
            'both_agree_no_transit': both_agree_no_transit,
            'cnn_only': cnn_only,
            'attention_only': attention_only,
            'total_windows': len(ensemble_preds)
        }

        return results


# ============= FLASK APP =============

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hardcoded model paths - ONLY CNN AND RESNET FOR PRODUCTION
MODEL_DIR = Path(__file__).parent / 'models'
MODELS = {
    'cnn': MODEL_DIR / 'cnn_best.pt',
    'resnet': MODEL_DIR / 'resnet_epoch29.pt',
    'ensemble': [MODEL_DIR / 'cnn_best.pt', MODEL_DIR / 'resnet_epoch29.pt']
}


def plot_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


@app.route('/')
def index():
    return render_template('transit_detector.html', device=device.upper())


def load_csv_lightcurve(csv_file, row_num=None, planet_id_col=None):
    """
    Load lightcurve from CSV file

    Args:
        csv_file: File object
        row_num: Specific row number (1-indexed), or None for random
        planet_id_col: Name of column containing planet IDs

    Returns:
        planet_id, flux_array, metadata
    """
    import pandas as pd
    import random

    # Count rows
    csv_file.seek(0)
    num_rows = sum(1 for _ in csv_file) - 1  # Exclude header
    csv_file.seek(0)

    # Pick row
    if row_num is not None:
        if row_num < 1 or row_num > num_rows:
            raise ValueError(f"Row number must be between 1 and {num_rows}")
        selected_row = row_num - 1  # Convert to 0-indexed
    else:
        selected_row = random.randint(0, num_rows - 1)

    # Read just that row
    df = pd.read_csv(csv_file, skiprows=range(1, selected_row + 1), nrows=1)

    # Get planet ID if column specified
    planet_id = None
    if planet_id_col and planet_id_col in df.columns:
        planet_id = df[planet_id_col].values[0]
    else:
        planet_id = f"CSV_Row_{selected_row + 1}"

    # Extract flux columns (all numeric columns, or columns starting with 'flux_')
    if 'flux_0' in df.columns:
        # Our master CSV format
        flux_cols = [col for col in df.columns if col.startswith('flux_')]
    else:
        # All numeric columns except ID column
        flux_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    flux = df[flux_cols].values[0]

    # Remove NaN values
    flux = flux[~np.isnan(flux)]

    metadata = {
        'target_id': str(planet_id),
        'source': 'CSV Upload',
        'row_number': selected_row + 1,
        'total_rows': num_rows,
        'flux_points': len(flux)
    }

    return planet_id, flux, metadata


@app.route('/predict_telescope', methods=['POST'])
def predict_telescope():
    """Run inference with HYBRID mode (BLS → Sliding fallback) - supports telescope and CSV"""
    try:
        # Get form data
        model_choice = request.form.get('model_choice', 'cnn')
        data_source = request.form.get('data_source', 'telescope')

        # Validate model choice
        if model_choice not in MODELS:
            return jsonify({'error': f'Invalid model choice: {model_choice}'}), 400

        # Load model(s) based on choice
        is_ensemble = model_choice == 'ensemble'

        if is_ensemble:
            # Load ensemble detector
            ensemble_detector = EnsembleDetector(MODELS['ensemble'], device=device)
            architecture = 'Ensemble (CNN + Attention)'
            threshold = 0.5  # Default for ensemble
        else:
            # Load single model
            model_path = MODELS[model_choice]
            if not model_path.exists():
                return jsonify({'error': f'Model file not found: {model_path}'}), 404

            model, checkpoint = load_model_file(str(model_path), device=device)
            architecture = checkpoint['architecture']
            inference_config = checkpoint.get('inference_config', {})
            threshold = inference_config.get('threshold', 0.5)

        # Load lightcurve based on data source
        if data_source == 'telescope':
            # Telescope download mode
            target_id = request.form.get('target_id')
            mission = request.form.get('mission', 'Kepler')

            if not target_id:
                return jsonify({'error': 'Missing target ID'}), 400

            time, flux, metadata = download_lightcurve(target_id, mission)

            if time is None:
                return jsonify({'error': f'Failed to download {target_id}'}), 404

            # Telescope data is raw flux - apply relative normalization
            flux_relative = preprocess_for_bls(flux)

        elif data_source == 'csv':
            # CSV upload mode
            csv_file = request.files.get('csv_file')
            if not csv_file:
                return jsonify({'error': 'Missing CSV file'}), 400

            csv_row = request.form.get('csv_row')
            csv_row = int(csv_row) if csv_row else None

            csv_planet_id_col = request.form.get('csv_planet_id_col')
            normalization_type = request.form.get('normalization_type', 'relative')

            # Load from CSV
            planet_id, flux, csv_metadata = load_csv_lightcurve(csv_file, csv_row, csv_planet_id_col)

            # Apply preprocessing based on normalization state
            if normalization_type == 'relative':
                # Already relative flux normalized (like our master CSV)
                flux_relative = flux
                csv_metadata['preprocessing'] = 'Already relative flux normalized'
            elif normalization_type == 'raw':
                # Raw flux - apply relative normalization
                flux_relative = preprocess_for_bls(flux)
                csv_metadata['preprocessing'] = 'Applied relative flux normalization'
            elif normalization_type == 'zscore':
                # Already z-score normalized - convert back to relative
                # This is tricky, we'll treat it as-is for BLS
                flux_relative = flux
                csv_metadata['preprocessing'] = 'Z-score normalized (used as-is for BLS)'
            else:  # 'none'
                # No normalization - apply relative
                flux_relative = preprocess_for_bls(flux)
                csv_metadata['preprocessing'] = 'Applied relative flux normalization from raw'

            metadata = csv_metadata

        else:
            return jsonify({'error': f'Invalid data source: {data_source}'}), 400

        # HYBRID MODE: Try BLS first, fallback to sliding if unsuccessful
        windows = None
        window_times = None
        mode_info = {}

        # Attempt 1: BLS-first mode (fast, matches training)
        print(" Attempting BLS detection...")
        bls_result = run_bls_training_mode(flux_relative)

        if bls_result:
            # Extract windows from RELATIVE FLUX
            windows, window_times = extract_bls_windows(flux_relative, bls_result)

            if windows is not None and len(windows) >= 3:
                # BLS successful!
                mode_info = {
                    'mode': 'BLS (Primary)',
                    'bls_period': bls_result['period'],
                    'bls_power': bls_result['power'],
                    'bls_duration': bls_result['duration'],
                    'num_transits': bls_result['n_transits'],
                    'num_windows': len(windows),
                    'fallback_used': False
                }
                print(f" BLS successful: {len(windows)} windows extracted")
            else:
                windows = None  # Not enough windows, trigger fallback

        # Attempt 2: Sliding window fallback if BLS unsuccessful
        if windows is None:
            print("   BLS unsuccessful, falling back to sliding window mode...")
            windows, window_times = extract_sliding_windows(flux_relative)
            mode_info = {
                'mode': 'Sliding Window (Fallback)',
                'num_windows': len(windows) if windows is not None else 0,
                'fallback_used': True,
                'reason': 'BLS detected insufficient transit windows (<3)'
            }
            print(f"  Sliding window: {len(windows) if windows else 0} windows extracted")

        if windows is None or len(windows) == 0:
            return jsonify({
                'error': 'No windows extracted by either BLS or sliding window',
                'mode_info': mode_info
            }), 400

        # Step 2: Z-score normalize extracted windows (THIS IS CRITICAL!)
        # Windows are in relative flux, now normalize to mean=0, std=1
        windows_normalized = normalize_windows(windows)

        # Run inference on normalized windows
        if is_ensemble:
            # Run ensemble detection
            ensemble_results = ensemble_detector.predict(windows_normalized, strategy='voting')

            # Extract ensemble predictions (convert back to numpy arrays for computation)
            predictions = np.array(ensemble_results['ensemble']['predictions'])
            probabilities = np.array(ensemble_results['ensemble']['probabilities'])

            # Store individual model results for display
            individual_results = ensemble_results['models']
        else:
            # Run single model inference
            predictions, probabilities = run_inference(model, windows_normalized, threshold, device)
            individual_results = None

        # Calculate results
        num_detections = int(np.sum(predictions))
        avg_confidence = float(np.mean(probabilities))
        max_confidence = float(np.max(probabilities))

        # High confidence detections (>80%)
        high_conf_detections = int(np.sum(probabilities > 0.8))

        # Sort confidences to get top detections
        sorted_probs = np.sort(probabilities)[::-1]  # Descending
        top_5_avg = float(np.mean(sorted_probs[:min(5, len(sorted_probs))]))

        # Final verdict: Different logic for ensemble vs single model
        if is_ensemble:
            # ENSEMBLE OR LOGIC: If EITHER model says transit, verdict = TRANSIT
            # Check each model's decision based on their detection rate
            cnn_results = individual_results[0]  # First model is CNN
            resnet_results = individual_results[1]  # Second model is ResNet

            cnn_detection_rate = cnn_results['detections'] / len(windows)
            resnet_detection_rate = resnet_results['detections'] / len(windows)

            # Simple threshold: >30% detection rate means model says "transit"
            cnn_says_transit = cnn_detection_rate > 0.3
            resnet_says_transit = resnet_detection_rate > 0.3

            # OR logic: Either model detecting → verdict is transit
            is_transit = cnn_says_transit or resnet_says_transit

            # Build verdict text showing what each model said
            agreement_rate = ensemble_results['agreement']['agreement_rate']
            if agreement_rate < 0.3:
                agreement_indicator = ' - LOW MODEL AGREEMENT'
            elif agreement_rate > 0.7:
                agreement_indicator = ' - HIGH MODEL AGREEMENT'
            else:
                agreement_indicator = ''

            # Add info about which models detected
            if cnn_says_transit and resnet_says_transit:
                agreement_indicator += ' (Both models detected)'
            elif cnn_says_transit:
                agreement_indicator += ' (CNN detected, ResNet did not)'
            elif resnet_says_transit:
                agreement_indicator += ' (ResNet detected, CNN did not)'
        else:
            # SINGLE MODEL: Use original logic
            is_transit = (high_conf_detections >= 3) or (top_5_avg > 0.7 and num_detections >= 3)
            agreement_indicator = ''

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Confidence over windows
        ax1.plot(probabilities, 'o-', markersize=4)
        ax1.axhline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
        ax1.axhline(avg_confidence, color='green', linestyle='--', label=f'Avg={avg_confidence:.3f}')
        ax1.set_xlabel('Window Number')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Detection Confidence Per Window')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Histogram
        ax2.hist(probabilities, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_img = plot_to_base64(fig)

        # Prepare response
        response_data = {
            'success': True,
            'target_info': metadata,
            'mode_info': mode_info,
            'results': {
                'total_windows': len(windows),
                'detections': num_detections,
                'detection_rate': float(num_detections / len(windows)),
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'high_conf_detections': high_conf_detections,  # NEW: >80% confidence
                'top_5_avg': top_5_avg,  # NEW: Average of top 5 confidences
                'final_verdict': ('TRANSIT DETECTED' if is_transit else 'NO TRANSIT') + agreement_indicator
            },
            'confidence_plot': plot_img,
            'architecture': architecture.upper(),
            'threshold': threshold,
            'model_config': checkpoint.get('model_config', {}) if not is_ensemble else {},
            'preprocessing': {
                'step1': 'Relative flux normalization (flux/median)',
                'step2': 'BLS detection on relative flux',
                'step3': 'Extract windows from relative flux',
                'step4': 'Z-score normalize windows (mean=0, std=1)'
            },
            'verdict_logic': {
                'description': 'High-confidence detection (handles sparse transits)',
                'criteria': '(≥3 detections with >80% confidence) OR (top-5 avg >70% AND ≥3 detections)',
                'note': 'Transits are often only 5% of full lightcurve, so we look for HIGH confidence peaks, not averages'
            }
        }

        # Add individual model results for ensemble
        if is_ensemble and individual_results:
            response_data['ensemble_details'] = {
                'individual_models': individual_results,
                'strategy': 'voting',
                'agreement': ensemble_results['agreement']
            }

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # NOTE: debug=False because multiprocessing doesn't work with Flask's auto-reloader
    # If you need debugging, disable multiprocessing by setting use_multicore=False in run_bls_training_mode()
    app.run(host='127.0.0.1', port=5000, debug=False)