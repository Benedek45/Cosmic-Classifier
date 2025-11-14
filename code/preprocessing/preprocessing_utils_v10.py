"""
Preprocessing Utilities V10 - Standalone

Signal processing functions for V10 preprocessing.
No imports from v8/v9 - all code is self-contained.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, List, Optional


def detrend_lightcurve(flux: np.ndarray, window_length: int = 101, polyorder: int = 3) -> np.ndarray:
    """
    Remove long-term trends using Savitzky-Golay filter.

    Preserves short-duration transit signals while removing:
    - Stellar rotation
    - Pulsations
    - Long-term instrumental drifts

    Args:
        flux: Flux array
        window_length: Filter window (must be odd, default 101)
        polyorder: Polynomial order (default 3)

    Returns:
        Detrended flux array
    """
    if len(flux) < window_length:
        return flux

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    try:
        # Compute trend
        trend = savgol_filter(flux, window_length, polyorder, mode='nearest')

        # Remove trend
        detrended = flux - trend

        return detrended
    except Exception:
        # If savgol fails, return original
        return flux


def robust_normalize(flux: np.ndarray) -> np.ndarray:
    """
    Normalize using median and MAD (robust to outliers).

    Uses Median Absolute Deviation instead of standard deviation
    to prevent bias from transits/outliers.

    Args:
        flux: Flux array

    Returns:
        Normalized flux (median=0, MAD-scaled)
    """
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))

    # Avoid division by zero
    if mad < 1e-10:
        return flux - median

    # Normalize: (flux - median) / MAD
    # MAD ~= 0.6745 * std for normal distribution
    # So we scale by 1.4826 to make it comparable to std
    normalized = (flux - median) / (1.4826 * mad)

    return normalized


def compute_mad(arr: np.ndarray) -> float:
    """
    Compute Median Absolute Deviation.

    Args:
        arr: Input array

    Returns:
        MAD value
    """
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return float(mad)


def sliding_window_indices(
    length: int,
    window_size: int,
    stride: int = 1
) -> List[Tuple[int, int]]:
    """
    Generate sliding window start/end indices.

    Args:
        length: Array length
        window_size: Window size
        stride: Step size between windows

    Returns:
        List of (start, end) tuples
    """
    indices = []
    for start in range(0, length - window_size + 1, stride):
        end = start + window_size
        indices.append((start, end))
    return indices


def validate_window(window: np.ndarray) -> bool:
    """
    Check if window is valid (no NaN/Inf, reasonable variance).

    Args:
        window: Flux window

    Returns:
        True if valid
    """
    # Check for NaN/Inf
    if not np.all(np.isfinite(window)):
        return False

    # Check for reasonable variance
    if np.std(window) < 1e-6:
        return False

    return True


def find_flat_regions(
    flux: np.ndarray,
    window_size: int = 128,
    threshold_sigma: float = 2.0
) -> List[Tuple[int, int]]:
    """
    Find contiguous regions with minimal variability (no dips).

    Used for extracting true negative samples from Label 1 stars.

    Args:
        flux: Flux array
        window_size: Rolling window size
        threshold_sigma: Threshold in units of MAD

    Returns:
        List of (start, end) tuples for flat regions
    """
    if len(flux) < window_size:
        return []

    # Compute rolling standard deviation
    rolling_std = []
    for i in range(len(flux) - window_size + 1):
        window = flux[i:i + window_size]
        rolling_std.append(np.std(window))

    rolling_std = np.array(rolling_std)

    # Compute threshold
    mad = compute_mad(flux)
    threshold = threshold_sigma * mad

    # Find flat regions (low variability)
    is_flat = rolling_std < threshold

    # Convert to contiguous regions
    regions = []
    start = None

    for i, flat in enumerate(is_flat):
        if flat and start is None:
            start = i
        elif not flat and start is not None:
            # Region ended
            if i - start >= window_size:
                regions.append((start, i))
            start = None

    # Handle last region
    if start is not None and len(flux) - start >= window_size:
        regions.append((start, len(flux)))

    return regions


def mask_transit_regions(
    length: int,
    transit_times: List[int],
    duration_samples: int,
    padding_factor: float = 1.5
) -> np.ndarray:
    """
    Create boolean mask for transit times.

    True = in-transit, False = out-of-transit

    Args:
        length: Lightcurve length
        transit_times: List of transit center indices
        duration_samples: Transit duration in samples
        padding_factor: Safety padding (default 1.5x duration)

    Returns:
        Boolean mask array
    """
    mask = np.zeros(length, dtype=bool)

    padded_duration = int(duration_samples * padding_factor)
    half_duration = padded_duration // 2

    for transit_time in transit_times:
        start = max(0, transit_time - half_duration)
        end = min(length, transit_time + half_duration)
        mask[start:end] = True

    return mask


def extract_out_of_transit_windows(
    flux: np.ndarray,
    transit_mask: np.ndarray,
    window_size: int,
    stride: int = None
) -> List[np.ndarray]:
    """
    Extract windows from out-of-transit regions.

    Args:
        flux: Flux array
        transit_mask: Boolean mask (True = in-transit)
        window_size: Window size
        stride: Step size (default = window_size, non-overlapping)

    Returns:
        List of flux windows
    """
    if stride is None:
        stride = window_size

    windows = []

    for start in range(0, len(flux) - window_size + 1, stride):
        end = start + window_size

        # Check if entire window is out-of-transit
        if not np.any(transit_mask[start:end]):
            window = flux[start:end]
            if validate_window(window):
                windows.append(window)

    return windows


def find_isolated_dips(
    flux: np.ndarray,
    window_size: int,
    min_depth_sigma: float = 2.5,
    isolation_distance: int = 150
) -> List[int]:
    """
    Find isolated deep dips (not periodic).

    Used for Label 1 hard negatives that look like transits
    but are isolated events.

    Args:
        flux: Flux array
        window_size: Window size
        min_depth_sigma: Minimum depth in sigma units
        isolation_distance: Minimum distance between dips

    Returns:
        List of dip center indices
    """
    # Compute threshold
    median = np.median(flux)
    mad = compute_mad(flux)
    threshold = median - (min_depth_sigma * 1.4826 * mad)

    # Find points below threshold
    below_threshold = flux < threshold

    # Find local minima
    dip_centers = []
    i = window_size // 2

    while i < len(flux) - window_size // 2:
        if below_threshold[i]:
            # Check if local minimum
            window = flux[max(0, i-10):min(len(flux), i+11)]
            if flux[i] == np.min(window):
                # Check isolation (no other dips nearby)
                is_isolated = True
                for prev_center in dip_centers:
                    if abs(i - prev_center) < isolation_distance:
                        is_isolated = False
                        break

                if is_isolated:
                    dip_centers.append(i)
                    i += isolation_distance  # Skip ahead
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    return dip_centers


def find_systematic_noise(
    flux: np.ndarray,
    quality: np.ndarray,
    window_size: int,
    max_windows: int = 50
) -> List[int]:
    """
    Find regions with systematic noise/artifacts.

    Args:
        flux: Flux array
        quality: Quality flags
        window_size: Window size
        max_windows: Maximum windows to extract

    Returns:
        List of window start indices
    """
    # Find regions with poor quality flags
    bad_quality_mask = quality > 0

    # Find contiguous bad regions
    windows = []
    in_bad_region = False
    region_start = 0

    for i, is_bad in enumerate(bad_quality_mask):
        if is_bad and not in_bad_region:
            region_start = i
            in_bad_region = True
        elif not is_bad and in_bad_region:
            # Region ended
            if i - region_start >= window_size:
                windows.append(region_start)
            in_bad_region = False

            if len(windows) >= max_windows:
                break

    return windows[:max_windows]


def find_stellar_variability(
    flux: np.ndarray,
    window_size: int,
    max_windows: int = 30
) -> List[int]:
    """
    Find regions with high stellar variability (flares, rotation).

    Args:
        flux: Flux array
        window_size: Window size
        max_windows: Maximum windows to extract

    Returns:
        List of window start indices
    """
    # Compute rolling variance
    rolling_var = []
    for i in range(0, len(flux) - window_size + 1, window_size // 2):
        window = flux[i:i + window_size]
        rolling_var.append((i, np.var(window)))

    # Sort by variance (highest first)
    rolling_var.sort(key=lambda x: x[1], reverse=True)

    # Take top N high-variance regions
    windows = [start for start, var in rolling_var[:max_windows]]

    return windows
