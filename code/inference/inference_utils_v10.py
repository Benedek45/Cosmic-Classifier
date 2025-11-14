"""
Inference Utilities for V10 Models

Functions for running inference on full lightcurves:
- Preprocessing
- Sliding window extraction
- Period finding (BLS, autocorrelation, peaks)
- Phase folding validation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from astropy.timeseries import BoxLeastSquares


def preprocess_lightcurve(time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess full lightcurve (same as training).

    Args:
        time: Time array
        flux: Flux array

    Returns:
        time, flux: Preprocessed arrays
    """
    # Import preprocessing functions
    import sys
    from pathlib import Path
    preprocessing_dir = Path(__file__).parent / "preprocessing"
    sys.path.insert(0, str(preprocessing_dir))

    from preprocessing_utils_v10 import detrend_lightcurve, robust_normalize

    # Remove NaN/inf
    valid = np.isfinite(time) & np.isfinite(flux)
    time = time[valid]
    flux = flux[valid]

    if len(flux) < 1000:
        raise ValueError(f"Lightcurve too short: {len(flux)} samples")

    # Preprocess
    flux = detrend_lightcurve(flux)
    flux = robust_normalize(flux)

    return time, flux


def extract_sliding_sequences(
    flux: np.ndarray,
    window_size: int = 256,
    sequence_length: int = 5,
    stride: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract overlapping 5-window sequences from full lightcurve.

    Args:
        flux: Preprocessed flux array
        window_size: Size of each window (default: 256)
        sequence_length: Number of windows per sequence (default: 5)
        stride: Stride between sequences (default: 128)

    Returns:
        sequences: Array of shape (N, 5, 256)
        positions: Starting index of each sequence
    """
    sequences = []
    positions = []

    # Total length needed for one sequence
    total_length = (sequence_length - 1) * stride + window_size

    # Slide across lightcurve
    for start_idx in range(0, len(flux) - total_length + 1, stride):
        sequence_windows = []
        valid = True

        # Extract sequence_length windows
        for i in range(sequence_length):
            window_start = start_idx + i * stride
            window = flux[window_start:window_start + window_size]

            # Validate window
            if len(window) != window_size or not np.all(np.isfinite(window)):
                valid = False
                break

            sequence_windows.append(window)

        if valid and len(sequence_windows) == sequence_length:
            sequences.append(np.array(sequence_windows))
            positions.append(start_idx)

    return np.array(sequences), np.array(positions)


def create_transit_timeline(
    flux_length: int,
    positions: np.ndarray,
    transit_probs: np.ndarray,
    window_size: int = 256,
    stride: int = 128,
    sequence_length: int = 5
) -> np.ndarray:
    """
    Map sequence predictions back to original time axis.

    For overlapping predictions, average them.

    Args:
        flux_length: Length of original flux array
        positions: Starting positions of sequences
        transit_probs: Transit probability for each sequence
        window_size: Window size
        stride: Stride between windows
        sequence_length: Number of windows per sequence

    Returns:
        timeline: Transit probability at each time point
    """
    timeline = np.zeros(flux_length)
    counts = np.zeros(flux_length)

    # Map each sequence prediction to its time range
    total_seq_length = (sequence_length - 1) * stride + window_size

    for pos, prob in zip(positions, transit_probs):
        end = min(pos + total_seq_length, flux_length)
        timeline[pos:end] += prob
        counts[pos:end] += 1

    # Average overlapping predictions
    timeline = timeline / np.maximum(counts, 1)

    return timeline


def find_period_autocorr(
    time: np.ndarray,
    signal: np.ndarray,
    max_period_days: float = 20.0
) -> Tuple[Optional[float], float]:
    """
    Find period using autocorrelation.

    Args:
        time: Time array (days)
        signal: Signal to analyze (transit probability timeline)
        max_period_days: Maximum period to search (days)

    Returns:
        period: Detected period (days), or None
        confidence: Confidence score (0-1)
    """
    # Compute autocorrelation
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    autocorr = np.correlate(signal_norm, signal_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only

    # Convert lag index to time
    cadence = np.median(np.diff(time))
    lags_time = np.arange(len(autocorr)) * cadence

    # Only search up to max_period_days
    max_lag = int(max_period_days / cadence)
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1

    autocorr = autocorr[:max_lag]
    lags_time = lags_time[:max_lag]

    # Find peaks (excluding lag=0)
    peaks, properties = find_peaks(
        autocorr[10:],  # Skip very short lags
        height=0.1,
        distance=20
    )

    if len(peaks) == 0:
        return None, 0.0

    # First major peak = period
    peak_idx = peaks[0] + 10
    period = lags_time[peak_idx]
    confidence = autocorr[peak_idx] / np.max(autocorr)

    return period, confidence


def find_period_peaks(
    time: np.ndarray,
    transit_prob_timeline: np.ndarray,
    threshold: float = 0.6
) -> Tuple[Optional[float], float, List[float]]:
    """
    Find period by detecting transit peaks and computing spacing.

    Args:
        time: Time array (days)
        transit_prob_timeline: Transit probability at each time point
        threshold: Minimum probability to consider as transit

    Returns:
        period: Median spacing between transits (days)
        consistency: How consistent the spacing is (0-1)
        transit_times: Detected transit times (days)
    """
    # Smooth timeline to reduce noise
    smoothed = uniform_filter1d(transit_prob_timeline, size=20)

    # Find peaks
    cadence = np.median(np.diff(time))
    min_distance = int(0.5 / cadence)  # Minimum 0.5 days between transits

    peaks, properties = find_peaks(
        smoothed,
        height=threshold,
        distance=min_distance,
        prominence=0.1
    )

    if len(peaks) < 3:
        return None, 0.0, []

    # Get times of detected transits
    transit_times = time[peaks]

    # Calculate spacing between consecutive transits
    spacings = np.diff(transit_times)

    if len(spacings) == 0:
        return None, 0.0, transit_times.tolist()

    # Period = median spacing
    period = np.median(spacings)

    # Consistency: how uniform are the spacings?
    consistency = 1.0 - np.std(spacings) / (np.mean(spacings) + 1e-8)
    consistency = np.clip(consistency, 0, 1)

    return period, consistency, transit_times.tolist()


def find_period_bls(
    time: np.ndarray,
    flux: np.ndarray,
    timeline: np.ndarray,
    min_period: float = 0.5,
    max_period: float = 50.0
) -> Tuple[Optional[float], float, Dict]:
    """
    Find period using Box Least Squares (BLS) algorithm.

    BLS is the industry standard for detecting box-shaped exoplanet transits.

    Args:
        time: Time array (days)
        flux: Flux array (preprocessed, centered around 0)
        timeline: Transit probability timeline from Model 1 (not used directly, kept for API compatibility)
        min_period: Minimum period to search (days)
        max_period: Maximum period to search (days)

    Returns:
        period: Best period (days), or None
        confidence: BLS power normalized to 0-1
        info: Dictionary with BLS results
    """
    import signal
    import logging
    logger = logging.getLogger(__name__)

    def timeout_handler(signum, frame):
        raise TimeoutError("BLS computation timed out")

    try:
        # For very large files (>50k samples), skip BLS entirely - it's too slow
        # Use fast autocorrelation + peaks instead on the full timeline
        if len(time) > 50000:
            logger.info(f"Large lightcurve ({len(time)} samples, {time[-1]-time[0]:.0f} days) - skipping slow BLS")
            return None, 0.0, {'error': 'skipped_large_for_speed', 'samples': len(time)}

        # BLS expects flux centered around 1 with transits as dips below 1
        # Our flux is normalized around 0 (transits are negative)
        # Transform: flux=-0.05 (5% dip) → flux_bls=0.95 (dip below 1.0)
        flux_bls = 1.0 + flux

        # Initialize BLS model
        from astropy.timeseries import BoxLeastSquares
        model = BoxLeastSquares(time, flux_bls)

        # Expected transit duration (in days)
        # Typical exoplanet transits: 2-6 hours = 0.08-0.25 days
        # Use conservative estimate: 0.1 days (2.4 hours)
        duration = 0.1

        # Set alarm for 15 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)

        try:
            # Adjust BLS parameters based on data size
            if len(time) > 20000:
                # Large file (20k-50k samples) - use aggressive parameters
                freq_factor = 12.0
                oversample_val = 2
            elif len(time) > 10000:
                # Medium file
                freq_factor = 8.0
                oversample_val = 3
            else:
                # Normal file - can afford good resolution
                freq_factor = 5.0
                oversample_val = 4

            # Run BLS autopower with optimized parameters
            periodogram = model.autopower(
                duration,
                minimum_period=min_period,
                maximum_period=max_period,
                frequency_factor=freq_factor,
                oversample=oversample_val
            )
        finally:
            # Cancel alarm
            signal.alarm(0)

        # Find best period
        best_idx = np.argmax(periodogram.power)
        best_period = float(periodogram.period[best_idx])
        best_power = float(periodogram.power[best_idx])

        # Calculate confidence from power
        # Normalize power to 0-1 range
        power_median = np.median(periodogram.power)
        power_std = np.std(periodogram.power)
        sde = (best_power - power_median) / (power_std + 1e-10)
        confidence = min(sde / 10.0, 1.0)

        # Get additional info if available
        info = {
            'power': best_power,
            'sde': float(sde),
            'duration': duration
        }

        # Try to get depth and transit_time if available
        try:
            info['depth'] = float(periodogram.depth[best_idx])
            info['transit_time'] = float(periodogram.transit_time[best_idx])
        except (AttributeError, IndexError):
            pass

        return best_period, float(confidence), info

    except TimeoutError as e:
        # BLS timed out - too complex signal, fall back to faster methods
        logger.warning(f"BLS timed out: {e}")
        return None, 0.0, {'error': 'bls_timeout'}
    except Exception as e:
        # BLS can fail for very short lightcurves or bad data
        logger.warning(f"BLS failed: {e}")
        return None, 0.0, {'error': str(e)}


def test_harmonic_candidates(
    time: np.ndarray,
    flux: np.ndarray,
    timeline: np.ndarray,
    base_period: float
) -> Tuple[float, float, str]:
    """
    Test period and its harmonics to find the true fundamental period.

    This helps resolve the common issue of detecting 2P, 3P, or P/2 instead of P.

    Args:
        time: Time array (days)
        flux: Flux array
        timeline: Transit probability timeline
        base_period: Initial period estimate (days)

    Returns:
        best_period: Best period after harmonic testing (days)
        best_score: Coherence score for best period
        harmonic_type: Which harmonic was selected (e.g., '1x', '2x', '0.5x')
    """
    # Generate harmonic candidates
    candidates = [
        (base_period, '1x'),           # Original
        (base_period * 2, '2x'),       # First harmonic
        (base_period * 3, '3x'),       # Second harmonic
        (base_period / 2, '0.5x'),     # Subharmonic
        (base_period / 3, '0.33x'),    # Second subharmonic
    ]

    best_score = -1
    best_period = base_period
    best_type = '1x'

    for period, harmonic_type in candidates:
        # Skip if period is out of reasonable range
        if period < 0.3 or period > 100:
            continue

        # Phase fold and score
        score = _score_period_coherence(time, flux, timeline, period)

        if score > best_score:
            best_score = score
            best_period = period
            best_type = harmonic_type

    return best_period, best_score, best_type


def _score_period_coherence(
    time: np.ndarray,
    flux: np.ndarray,
    timeline: np.ndarray,
    period: float,
    nbins: int = 50
) -> float:
    """
    Score how well a period organizes the transit detections.

    A good period will phase-fold the transits into a coherent signal.

    Args:
        time: Time array (days)
        flux: Flux array
        timeline: Transit probability timeline
        period: Period to test (days)
        nbins: Number of phase bins

    Returns:
        coherence_score: Score from 0 (incoherent) to 1 (perfectly coherent)
    """
    # Phase fold
    phase = ((time - time[0]) % period) / period

    # Bin the transit probabilities by phase
    phase_bins = np.linspace(0, 1, nbins + 1)
    binned_probs = []

    for i in range(nbins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        if np.sum(mask) > 0:
            binned_probs.append(np.mean(timeline[mask]))
        else:
            binned_probs.append(0.0)

    binned_probs = np.array(binned_probs)

    # Good period will have:
    # 1. High peak in transit probability at one phase
    # 2. Low baseline everywhere else
    # 3. Sharp transit feature (not spread out)

    peak_prob = np.max(binned_probs)
    baseline_prob = np.median(binned_probs)

    # Transit should be localized (high peak-to-baseline ratio)
    if baseline_prob > 0:
        contrast = (peak_prob - baseline_prob) / baseline_prob
    else:
        contrast = peak_prob

    # Normalize to 0-1 range
    coherence_score = min(contrast / 5.0, 1.0)

    # Bonus for having concentrated transit (not spread across many bins)
    high_bins = np.sum(binned_probs > 0.7 * peak_prob)
    if high_bins > 0 and high_bins < 10:  # Transit in 1-10 bins is good (localized)
        coherence_score *= 1.2

    return min(coherence_score, 1.0)


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    timeline: np.ndarray,
    period: float,
    epoch: Optional[float] = None,
    nbins: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase-fold data by detected period.

    Args:
        time: Time array (days)
        flux: Flux array
        timeline: Transit probability timeline
        period: Period to fold by (days)
        epoch: Reference epoch (default: first time point)
        nbins: Number of phase bins

    Returns:
        phase_bins: Phase bin centers (0-1)
        binned_flux: Median flux in each bin
        binned_prob: Median transit probability in each bin
    """
    if epoch is None:
        epoch = time[0]

    # Calculate phase
    phase = ((time - epoch) % period) / period

    # Bin by phase
    phase_edges = np.linspace(0, 1, nbins + 1)
    phase_centers = (phase_edges[:-1] + phase_edges[1:]) / 2

    binned_flux = []
    binned_prob = []

    for i in range(nbins):
        mask = (phase >= phase_edges[i]) & (phase < phase_edges[i + 1])
        if np.sum(mask) > 0:
            binned_flux.append(np.median(flux[mask]))
            binned_prob.append(np.median(timeline[mask]))
        else:
            binned_flux.append(np.nan)
            binned_prob.append(np.nan)

    return phase_centers, np.array(binned_flux), np.array(binned_prob)


def validate_detection(
    time: np.ndarray,
    flux: np.ndarray,
    timeline: np.ndarray,
    detected_period: float,
    ground_truth_period: Optional[float] = None
) -> Dict:
    """
    Validate detected period.

    Args:
        time: Time array (days)
        flux: Flux array
        timeline: Transit probability timeline
        detected_period: Detected period (days)
        ground_truth_period: Ground truth period for comparison

    Returns:
        validation_metrics: Dictionary of validation metrics
    """
    # Phase fold
    phase, binned_flux, binned_prob = phase_fold(
        time, flux, timeline, detected_period
    )

    # Remove NaN bins
    valid = np.isfinite(binned_flux) & np.isfinite(binned_prob)
    binned_flux = binned_flux[valid]
    binned_prob = binned_prob[valid]

    if len(binned_flux) < 10:
        return {
            'validation_score': 0.0,
            'transit_depth': 0.0,
            'max_prob': 0.0,
            'period_error': None
        }

    # Transit depth in phase-folded flux
    transit_depth = 1.0 - np.min(binned_flux)

    # Peak probability in phase-folded data
    max_prob = np.max(binned_prob)

    # Validation score
    validation_score = (transit_depth * 10 + max_prob) / 2
    validation_score = np.clip(validation_score, 0, 1)

    # Period error if ground truth available
    period_error = None
    if ground_truth_period is not None:
        period_error = abs(detected_period - ground_truth_period) / ground_truth_period

    return {
        'validation_score': float(validation_score),
        'transit_depth': float(transit_depth),
        'max_prob': float(max_prob),
        'period_error': float(period_error) if period_error is not None else None
    }


def aggregate_period_estimates(
    period_bls: Optional[float] = None,
    conf_bls: float = 0.0,
    period_autocorr: Optional[float] = None,
    conf_autocorr: float = 0.0,
    period_peaks: Optional[float] = None,
    conf_peaks: float = 0.0,
    harmonic_type: str = '1x'
) -> Tuple[Optional[float], str]:
    """
    Combine period estimates from multiple methods.

    BLS is preferred when available (industry standard for exoplanet detection).

    Args:
        period_bls: Period from BLS (after harmonic testing)
        conf_bls: Confidence from BLS
        period_autocorr: Period from autocorrelation
        conf_autocorr: Confidence from autocorrelation
        period_peaks: Period from peak spacing
        conf_peaks: Confidence from peak spacing
        harmonic_type: Harmonic selected by testing (e.g., '1x', '2x')

    Returns:
        best_period: Best estimate of period
        method: Which method was used (includes harmonic info for BLS)
    """
    estimates = []

    # BLS is most reliable, give it priority
    if period_bls is not None:
        # Boost BLS confidence slightly since it's designed for transits
        boosted_conf = min(conf_bls * 1.2, 1.0)
        method_name = f'bls_{harmonic_type}' if harmonic_type != '1x' else 'bls'
        estimates.append((period_bls, boosted_conf, method_name))

    if period_autocorr is not None:
        estimates.append((period_autocorr, conf_autocorr, 'autocorr'))

    if period_peaks is not None:
        estimates.append((period_peaks, conf_peaks, 'peaks'))

    if len(estimates) == 0:
        return None, 'none'

    # Use highest confidence method
    estimates.sort(key=lambda x: x[1], reverse=True)
    return estimates[0][0], estimates[0][2]