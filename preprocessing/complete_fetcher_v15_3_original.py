#!/usr/bin/env python3
"""
ExoSeeker Streaming Data Collection System v15.3 - Comprehensive Zip Fix
Downloads complete light curves, saves as CSV, uploads to Google Drive, cleans local storage
Includes improved cache handling, progress tracking, and resume functionality
Fixed lightkurve cache compatibility issues
Added comprehensive data sources: PS, TOI, KOI, K2 tables with deduplication
Added TOI 'FA' (False Alarm) disposition for additional false positive training data
Fixed K2 table name from 'k2candidates' to 'k2pandc' (adds 293 false positives + 2222 planets)

NEW IN v15.3:
CRITICAL FIX: Zip now includes ALL files from ALL runs (not just current session):
- Fixed zip logic to scan entire streaming_data_temp/lightcurves/ directory
- Includes all 2,287+ existing CSV files from previous runs
- Preserves metadata CSV in zip instead of deleting it
- Triggers on total files in directory (not just current session count)
- One comprehensive upload of everything accumulated so far

Previous v15.2 features:
Fixed upload functionality and implemented efficient zip-based batch uploads:
- Re-enabled Google Drive upload functionality
- Removed troubleshooting skip conditions for file uploads
- MAJOR OPTIMIZATION: Replaced individual file uploads with zip-based batch uploads
- Expected 50-80x faster uploads (3 minutes vs 4+ hours for 500 files)

Previous v15.1 features:
Enhanced cross-referencing and TESS support to reduce missing transit counts:
- Expanded cross-reference beyond "Kepler" names (includes TOI, KOI)
- TIC-based cross-referencing to find orbital parameters
- TESS-specific orbital parameter fetching from TOI table
- Enhanced debug logging to identify missing transit causes
- Light curve analysis fallback framework (placeholder)
- Multi-layer fallback: KOI->K2->PS->TESS->Cross-ref->LightCurve
"""

import lightkurve as lk
import numpy as np
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import os
import sys
import time
import random
import warnings
import json
import gc
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import argparse
import csv
import shutil
import zipfile
import uuid

# Google Drive imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

warnings.filterwarnings('ignore')
os.environ['ASTROQUERY_SILENCE'] = '1'

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

# ===============================
# STREAMING COLLECTION CONFIGURATION
# ===============================

# Server Resources - Optimized for zip-based batch uploads
MAX_WORKERS = 3  # Conservative for memory and upload bandwidth
BATCH_SIZE = 25  # Process in small batches but accumulate for large uploads
LARGE_BATCH_SIZE = 500  # Accumulate this many targets before zip upload
LOCAL_STORAGE_LIMIT_GB = 120  # Stop collecting if local storage exceeds this
LARGE_BATCH_THRESHOLD_GB = 50  # Create zip and upload when data reaches this size

# Network settings with more aggressive timeouts
DOWNLOAD_TIMEOUT = 180  # Reduced from 300
QUERY_TIMEOUT = 45     # Reduced from 60
UPLOAD_TIMEOUT = 1800  # 30 minutes for large uploads
MAX_RETRIES = 2        # Retry failed downloads

# Data Collection Settings
COLLECT_COMPLETE_LIGHTCURVES = True
SAVE_TRANSIT_METADATA = True
COMPRESS_DATA = True
DELETE_AFTER_UPLOAD = True

# Labels
LABEL_NON_PLANET = 1
LABEL_PLANET = 2

# Directory Structure - Temporary local storage
DATA_DIR = 'streaming_data_temp'
LIGHTCURVES_DIR = f'{DATA_DIR}/lightcurves'
METADATA_DIR = f'{DATA_DIR}/metadata'
UPLOAD_QUEUE_DIR = f'{DATA_DIR}/upload_queue'
LOG_DIR = 'streaming_logs'

# Local files
METADATA_CSV = f'{METADATA_DIR}/lightcurve_metadata.csv'
COLLECTION_LOG = f'{METADATA_DIR}/collection_progress.csv'
PROCESSED_TARGETS_LOG = f'{METADATA_DIR}/processed_targets.txt'  # New progress tracking file
UPLOAD_LOG = f'{LOG_DIR}/upload_log.csv'
LOG_FILE = f'{LOG_DIR}/streaming_fetcher.log'

# Google Drive Configuration
DRIVE_BASE_PATH = 'ExoSeeker_Data'  # Base folder in Google Drive
DRIVE_LIGHTCURVES_FOLDER = 'lightcurves'
DRIVE_METADATA_FOLDER = 'metadata'

# ===============================
# PROGRESS TRACKING
# ===============================

class ProgressTracker:
    def __init__(self):
        self.processed_targets = set()
        self.load_processed_targets()

    def load_processed_targets(self):
        """Load list of already processed targets"""
        if os.path.exists(PROCESSED_TARGETS_LOG):
            try:
                with open(PROCESSED_TARGETS_LOG, 'r') as f:
                    self.processed_targets = set(line.strip() for line in f if line.strip())
                print(f"[PROGRESS] Loaded {len(self.processed_targets)} already processed targets")
            except Exception as e:
                print(f"[PROGRESS] Error loading processed targets: {e}")
                self.processed_targets = set()

    def is_processed(self, target_id):
        """Check if target was already processed"""
        return target_id in self.processed_targets

    def mark_processed(self, target_id):
        """Mark target as processed and save to file"""
        if target_id not in self.processed_targets:
            self.processed_targets.add(target_id)
            try:
                os.makedirs(os.path.dirname(PROCESSED_TARGETS_LOG), exist_ok=True)
                with open(PROCESSED_TARGETS_LOG, 'a') as f:
                    f.write(f"{target_id}\n")
            except Exception as e:
                print(f"[PROGRESS] Error saving processed target: {e}")

    def get_stats(self):
        """Get processing statistics"""
        return len(self.processed_targets)

# Global progress tracker
progress_tracker = ProgressTracker()

# ===============================
# GOOGLE DRIVE INTEGRATION
# ===============================

class GoogleDriveManager:
    def __init__(self):
        self.service = None
        self.folder_cache = {}  # Cache folder IDs to avoid repeated queries

    def get_authenticated_service(self):
        """Initialize Google Drive service"""
        if self.service:
            return self.service

        # Check for credentials file or environment variables
        creds = None
        token_path = 'token.json'  # Common location for Google Drive tokens
        credentials_path = 'credentials.json'  # Common location for Google credentials
        
        # Load existing token if available
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/drive.file'])
            except Exception:
                print("Error loading existing token file.")
        
        # If no valid credentials, the user needs to authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    print("Credentials expired and refresh failed. Please re-authenticate.")
                    creds = None
            else:
                print("No valid credentials found. Please set up Google Drive authentication.")
                print("Follow the Google Drive API setup instructions to authenticate.")
                return None

        if creds:
            self.service = build('drive', 'v3', credentials=creds)
            return self.service
        else:
            return None

    def get_or_create_folder(self, folder_path):
        """Get or create folder path in Google Drive"""
        if folder_path in self.folder_cache:
            return self.folder_cache[folder_path]

        service = self.get_authenticated_service()
        if service is None:
            logger.error("Cannot access Google Drive: not authenticated")
            return None

        parent_id = None

        for folder_name in folder_path.split('/'):
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            else:
                query += " and 'root' in parents"

            try:
                results = service.files().list(q=query, fields='files(id, name)').execute()
                files = results.get('files', [])

                if files:
                    parent_id = files[0]['id']
                else:
                    metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
                    if parent_id:
                        metadata['parents'] = [parent_id]
                    folder = service.files().create(body=metadata, fields='id').execute()
                    parent_id = folder['id']
            except Exception as e:
                logger.error(f"Error accessing folder {folder_name}: {e}")
                return None

        self.folder_cache[folder_path] = parent_id
        return parent_id

    def upload_file(self, local_path, drive_filename, folder_path):
        """Upload file to Google Drive folder"""
        try:
            service = self.get_authenticated_service()
            if service is None:
                logger.warning(f"Cannot upload {drive_filename}: Google Drive not authenticated.")
                print(f"Upload skipped: {drive_filename} (Google Drive not authenticated)")
                return False, None
            
            folder_id = self.get_or_create_folder(folder_path)
            if folder_id is None:
                logger.error(f"Could not create or find folder: {folder_path}")
                return False, None

            file_metadata = {'name': drive_filename, 'parents': [folder_id]}

            # Determine mimetype based on extension
            if drive_filename.endswith('.csv'):
                mimetype = 'text/csv'
            elif drive_filename.endswith('.zip'):
                mimetype = 'application/zip'
            else:
                mimetype = 'application/octet-stream'

            media = MediaFileUpload(local_path, mimetype=mimetype, resumable=True)

            # Upload with progress tracking
            request = service.files().create(body=file_metadata, media_body=media, fields='id,size')
            response = None

            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    if progress % 25 == 0:  # Log every 25%
                        logger.info(f"Upload progress for {drive_filename}: {progress}%")

            file_id = response.get('id')
            file_size = response.get('size', 'Unknown')

            logger.info(f"Successfully uploaded {drive_filename} (ID: {file_id}, Size: {file_size})")

            # Log upload
            self.log_upload(drive_filename, file_id, file_size, folder_path)

            return True, file_id

        except Exception as e:
            logger.error(f"Failed to upload {drive_filename}: {e}")
            return False, None

    def log_upload(self, filename, file_id, file_size, folder_path):
        """Log successful upload to CSV"""
        upload_record = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'file_id': file_id,
            'file_size': file_size,
            'folder_path': folder_path,
            'status': 'uploaded'
        }

        upload_df = pd.DataFrame([upload_record])
        if os.path.exists(UPLOAD_LOG):
            upload_df.to_csv(UPLOAD_LOG, mode='a', header=False, index=False)
        else:
            upload_df.to_csv(UPLOAD_LOG, mode='w', header=True, index=False)

# Global Google Drive manager
drive_manager = GoogleDriveManager()

# ===============================
# LOGGING AND MONITORING
# ===============================

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger('streaming_fetcher')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = RotatingFileHandler(LOG_FILE, maxBytes=50*1024*1024, backupCount=5)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

def check_local_storage():
    """Check local storage usage and available space"""
    disk_usage = psutil.disk_usage('.')
    used_gb = disk_usage.used / (1024**3)
    free_gb = disk_usage.free / (1024**3)
    total_gb = disk_usage.total / (1024**3)

    return {
        'used_gb': used_gb,
        'free_gb': free_gb,
        'total_gb': total_gb,
        'percent_used': (used_gb / total_gb) * 100
    }

def check_data_directory_size():
    """Get size of local data directory"""
    if not os.path.exists(DATA_DIR):
        return 0

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, IOError):
                pass

    return total_size / (1024**3)  # Return in GB

def safe_print(*args, **kwargs):
    message = ' '.join(map(str, args))
    logger.info(message)
    sys.stdout.flush()

# ===============================
# ENHANCED FIRST TRANSIT DATA FUNCTIONS
# ===============================

def get_best_first_transit(target_info):
    """Get first transit time using best available source with fallback strategy"""

    # Priority 1: KOI first transit (100% success for Kepler targets)
    orbital_params = target_info.get('orbital_params', {})
    if orbital_params.get('first_transit') and target_info.get('mission') == 'Kepler':
        safe_print(f"  Using KOI first transit: {orbital_params['first_transit']}")
        return orbital_params['first_transit']

    # Priority 2: K2 first transit (100% success for K2 targets)
    if orbital_params.get('first_transit') and target_info.get('mission') == 'K2':
        safe_print(f"  Using K2 first transit: {orbital_params['first_transit']}")
        return orbital_params['first_transit']

    # Priority 3: PS table epoch (when available)
    if orbital_params.get('epoch'):
        safe_print(f"  Using PS epoch: {orbital_params['epoch']}")
        return orbital_params['epoch']

    # Priority 4: TESS-specific orbital parameter fetching
    if target_info.get('mission') == 'TESS' and target_info.get('tic'):
        tess_result = fetch_tess_orbital_parameters(target_info['tic'])
        if tess_result:
            safe_print(f"  TESS orbital parameters SUCCESS: {tess_result}")
            return tess_result

    # Priority 5: Enhanced cross-reference - try multiple approaches
    if not orbital_params.get('first_transit'):
        # Try cross-referencing by TIC ID to find Kepler matches
        if target_info.get('tic'):
            cross_ref_result = enhanced_cross_reference_by_tic(target_info['tic'])
            if cross_ref_result:
                safe_print(f"  TIC cross-reference SUCCESS: {cross_ref_result}")
                return cross_ref_result

        # Try cross-referencing by planet name (expanded beyond just "Kepler")
        target_name = target_info.get('name', '')
        if target_name and ('Kepler' in target_name or 'KOI' in target_name or 'TOI' in target_name):
            cross_ref_result = enhanced_cross_reference_by_name(target_name)
            if cross_ref_result:
                safe_print(f"  Name cross-reference SUCCESS: {cross_ref_result}")
                return cross_ref_result

    # Priority 6: Light curve analysis fallback (if we have period but no epoch)
    if orbital_params.get('period') and not orbital_params.get('epoch'):
        safe_print(f"  Attempting light curve analysis fallback...")
        lc_result = simple_light_curve_transit_detection(target_info, orbital_params.get('period'))
        if lc_result:
            safe_print(f"  Light curve analysis SUCCESS: {lc_result}")
            return lc_result

    safe_print(f"  No first transit data available after all attempts")
    return None

def simple_light_curve_transit_detection(target_info, period):
    """Simple light curve analysis to estimate first transit time when period is known"""
    try:
        safe_print(f"  Performing light curve analysis with period {period}...")

        # This is a placeholder for light curve analysis
        # In a full implementation, this would:
        # 1. Load the light curve data
        # 2. Use period to fold the light curve
        # 3. Find the phase of maximum transit depth
        # 4. Convert back to time

        # For now, return None to indicate this is not yet implemented
        safe_print(f"  Light curve analysis not yet implemented")
        return None

        # Future implementation would look like:
        # target_id = processor.get_target_id(target_info)
        # lc_file = f"streaming_data_temp/lightcurves/{target_id}.csv"
        # if os.path.exists(lc_file):
        #     df = pd.read_csv(lc_file, comment='#')
        #     time = df['time'].values
        #     flux = df['flux'].values
        #
        #     # Simple box-fitting to find transit
        #     from astropy.timeseries import BoxLeastSquares
        #     bls = BoxLeastSquares(time, flux)
        #     result = bls.power(period, 0.1)  # duration estimate
        #     return result.transit_time

    except Exception as e:
        logger.debug(f"Light curve analysis failed: {e}")

    return None

def fetch_tess_orbital_parameters(tic_id):
    """Fetch orbital parameters specifically for TESS targets from alternative sources"""
    try:
        safe_print(f"  Fetching TESS orbital parameters for TIC {tic_id}...")

        # Try TOI table with period information (some TOI entries might have periods)
        toi_data = NasaExoplanetArchive.query_criteria(
            table="toi",
            select="tid, toipfx, toi_period, toi_epoch, toi_depth",
            where=f"tid = {tic_id}"
        )

        if len(toi_data) > 0:
            row = toi_data[0]
            period = row.get('toi_period')
            epoch = row.get('toi_epoch')

            # Check if we have useful orbital data
            if period and not (hasattr(period, 'mask') and np.ma.is_masked(period)):
                safe_print(f"    Found TOI period: {period}")

                if epoch and not (hasattr(epoch, 'mask') and np.ma.is_masked(epoch)):
                    safe_print(f"    Found TOI epoch: {epoch}")
                    return float(epoch)
                else:
                    # If we have period but no epoch, try to estimate first transit
                    # This is a fallback - could implement light curve analysis here
                    safe_print(f"    TOI has period but no epoch - needs light curve analysis")

        # Try exofop-tess.ipac.caltech.edu equivalent queries if available
        # This would require additional API access

        # Try literature search for this TIC (future enhancement)
        safe_print(f"    No TESS orbital parameters found for TIC {tic_id}")

    except Exception as e:
        logger.debug(f"TESS orbital parameter fetch failed for TIC {tic_id}: {e}")

    return None

def enhanced_cross_reference_by_tic(tic_id):
    """Enhanced cross-reference using TIC ID to find orbital parameters from any source"""
    try:
        safe_print(f"  Attempting TIC cross-reference for TIC {tic_id}...")

        # Try to find this TIC in PS table with orbital parameters
        ps_data = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="pl_name, pl_orbper, pl_tranmid, pl_trandur",
            where=f"tic_id = {tic_id} and pl_tranmid is not null"
        )

        if len(ps_data) > 0:
            row = ps_data[0]
            epoch = row.get('pl_tranmid')
            if epoch and not (hasattr(epoch, 'mask') and np.ma.is_masked(epoch)):
                safe_print(f"    Found PS data for TIC {tic_id}: {row.get('pl_name')}")
                return float(epoch)

        # Try to find if this TIC corresponds to a Kepler planet via name matching
        ps_kepler = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="pl_name",
            where=f"tic_id = {tic_id} and pl_name like 'Kepler%'"
        )

        if len(ps_kepler) > 0:
            kepler_name = ps_kepler[0].get('pl_name')
            if kepler_name:
                safe_print(f"    TIC {tic_id} is {kepler_name}, checking KOI data...")
                koi_result = enhanced_cross_reference_by_name(kepler_name)
                if koi_result:
                    return koi_result

    except Exception as e:
        logger.debug(f"TIC cross-reference failed for {tic_id}: {e}")

    return None

def enhanced_cross_reference_by_name(planet_name):
    """Enhanced cross-reference by planet name - tries multiple name formats"""
    try:
        safe_print(f"  Attempting name cross-reference for {planet_name}...")

        # Clean the name
        clean_name = planet_name.strip()

        # Try exact match first
        koi_data = NasaExoplanetArchive.query_criteria(
            table="cumulative",
            select="koi_time0bk, koi_period, koi_duration",
            where=f"kepler_name = '{clean_name}'"
        )

        if len(koi_data) > 0:
            row = koi_data[0]
            first_transit = row.get('koi_time0bk')
            if first_transit and not (hasattr(first_transit, 'mask') and np.ma.is_masked(first_transit)):
                safe_print(f"    Exact match found for {clean_name}")
                return float(first_transit)

        # Try variations for TOI names
        if 'TOI' in clean_name:
            # Extract TOI number and try to find corresponding confirmed planet
            import re
            toi_match = re.search(r'TOI[- ]?(\d+)', clean_name)
            if toi_match:
                toi_num = toi_match.group(1)
                # Try to find Kepler name for this TOI
                ps_toi = NasaExoplanetArchive.query_criteria(
                    table="ps",
                    select="pl_name, pl_orbper, pl_tranmid",
                    where=f"pl_name like '%{toi_num}%' and pl_name like 'Kepler%'"
                )
                if len(ps_toi) > 0:
                    kepler_name = ps_toi[0].get('pl_name')
                    if kepler_name:
                        safe_print(f"    Found Kepler equivalent: {kepler_name}")
                        return enhanced_cross_reference_by_name(kepler_name)

    except Exception as e:
        logger.debug(f"Name cross-reference failed for {planet_name}: {e}")

    return None

def cross_reference_kepler_name(planet_name):
    """Legacy function - kept for compatibility"""
    return enhanced_cross_reference_by_name(planet_name)

# ===============================
# COMPLETE LIGHT CURVE PROCESSING (CSV FORMAT)
# ===============================

class CompleteLightCurveProcessor:
    def __init__(self):
        self.stats = {
            'planets_processed': 0,
            'non_planets_processed': 0,
            'uploads_completed': 0,
            'files_deleted': 0,
            'total_data_gb': 0,
            'cache_errors_fixed': 0
        }

    def calculate_transit_times(self, epoch, period, start_time, end_time, time_reference=None):
        """Calculate all transit times within observation window with enhanced epoch handling"""
        transit_times = []

        # Enhanced epoch format handling based on time reference system
        if time_reference:
            time_ref = str(time_reference).upper()
            if 'BJD' in time_ref and epoch < 2400000:
                # Convert BJD offset to full BJD
                epoch += 2454833
            elif 'HJD' in time_ref and epoch < 2400000:
                # Convert HJD offset to full HJD (similar conversion)
                epoch += 2454833
            elif 'JD' in time_ref and epoch < 2400000:
                # Convert JD offset
                epoch += 2454833
        else:
            # Fallback: assume standard Kepler/TESS format
            if epoch < 2400000:
                epoch += 2454833

        transit_time = epoch

        # Find transits in observation window
        while transit_time > start_time:
            transit_time -= period

        while transit_time <= end_time:
            if transit_time >= start_time:
                transit_times.append(transit_time)
            transit_time += period

        return transit_times

    def process_complete_lightcurve(self, target_info, label):
        """Download and process complete light curve - now saves as CSV"""
        target_id = self.get_target_id(target_info)

        # Check if already processed
        if progress_tracker.is_processed(target_id):
            safe_print(f"Skipping {target_id} - already processed")
            return None

        try:
            safe_print(f"Processing {target_id}...")

            # Download complete light curve with retry logic
            lc_collection, mission = self.download_complete_lightcurve_with_retry(target_info)
            if not lc_collection:
                return None

            # Stitch and clean
            if hasattr(lc_collection, 'stitch'):
                # It's a LightCurveCollection, stitch it
                lc = lc_collection.stitch().remove_nans().remove_outliers(sigma=5)
                del lc_collection  # Free memory
            else:
                # It's a single LightCurve, just clean it
                lc = lc_collection.remove_nans().remove_outliers(sigma=5)

            if len(lc.flux) < 1000:  # Minimum viable length
                del lc
                return None

            # Extract data arrays
            flux = self.safe_extract_flux(lc.flux)
            time = lc.time.value
            quality = getattr(lc, 'quality', np.zeros(len(flux), dtype=int))

            # Calculate transit metadata if available
            transit_metadata = None
            orbital_params = target_info.get('orbital_params')
            if orbital_params and orbital_params.get('period'):
                transit_metadata = self.calculate_transit_metadata(
                    time, orbital_params, target_info
                )

            # Save to CSV format instead of HDF5
            filepath = self.save_lightcurve_csv(target_id, flux, time, quality,
                                               target_info, transit_metadata, label)

            if filepath:
                # Update metadata CSV
                self.update_metadata_csv(target_id, target_info, len(flux),
                                       transit_metadata, label, mission)

                # Mark as processed
                progress_tracker.mark_processed(target_id)

                safe_print(f"Saved {target_id}: {len(flux)} points, {os.path.getsize(filepath)/(1024**2):.1f} MB")

                # Update stats
                if label == LABEL_PLANET:
                    self.stats['planets_processed'] += 1
                else:
                    self.stats['non_planets_processed'] += 1

                return filepath

        except Exception as e:
            logger.error(f"Error processing {target_id}: {e}")
            safe_print(f"DEBUG: Error details for {target_id}: {type(e).__name__}: {str(e)}")

        return None

    def download_complete_lightcurve_with_retry(self, target_info):
        """Download with retry logic and better cache handling"""
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Clear any problematic cache between attempts
                if attempt > 0:
                    safe_print(f"  Retry attempt {attempt}/{MAX_RETRIES}")
                    aggressive_cache_cleanup()
                    time.sleep(2)  # Brief pause

                result = self.download_complete_lightcurve(target_info)
                if result[0]:  # Success
                    return result

            except Exception as e:
                if "corrupt" in str(e).lower() or "size 0" in str(e).lower():
                    safe_print(f"  Cache corruption detected on attempt {attempt+1}, cleaning cache...")
                    self.stats['cache_errors_fixed'] += 1
                    aggressive_cache_cleanup()
                    if attempt == MAX_RETRIES:
                        safe_print(f"  Failed after {MAX_RETRIES+1} attempts due to cache issues")
                        return None, "Failed"
                else:
                    safe_print(f"  Non-cache error: {e}")
                    return None, "Failed"

        return None, "Failed"

    def download_complete_lightcurve(self, target_info):
        """Download complete light curve using the proven working approach"""
        search_result, mission_found = None, "None"

        # Search for light curve using the same approach as v7
        if target_info.get('kic'):
            try:
                safe_print(f"  Searching for KIC {target_info['kic']} in Kepler...")
                search_result = lk.search_lightcurve(f"KIC {target_info['kic']}", mission="Kepler", author="Kepler")
                if search_result:
                    mission_found = "Kepler"
                    safe_print(f"  Found {len(search_result)} Kepler results")
                else:
                    safe_print(f"  No Kepler results, trying K2...")
                    search_result = lk.search_lightcurve(f"KIC {target_info['kic']}", mission="K2", author="K2")
                    if search_result:
                        mission_found = "K2"
                        safe_print(f"  Found {len(search_result)} K2 results")
            except Exception as e:
                logger.debug(f"Kepler/K2 search failed for KIC {target_info['kic']}: {e}")
                safe_print(f"  KIC search failed: {e}")

        if not search_result and target_info.get('epic'):
            try:
                safe_print(f"  Searching for EPIC {target_info['epic']} in K2...")
                search_result = lk.search_lightcurve(f"EPIC {target_info['epic']}", mission="K2", author="K2")
                if search_result:
                    mission_found = "K2"
                    safe_print(f"  Found {len(search_result)} K2 EPIC results")
            except Exception as e:
                logger.debug(f"K2 EPIC search failed for EPIC {target_info['epic']}: {e}")
                safe_print(f"  EPIC search failed: {e}")

        if not search_result and target_info.get('tic'):
            try:
                safe_print(f"  Searching for TIC {target_info['tic']} in TESS...")
                search_result = lk.search_lightcurve(f"TIC {target_info['tic']}", mission="TESS", author="SPOC")
                if search_result:
                    mission_found = "TESS"
                    safe_print(f"  Found {len(search_result)} TESS results")
            except Exception as e:
                logger.debug(f"TESS search failed for TIC {target_info['tic']}: {e}")
                safe_print(f"  TIC search failed: {e}")

        if not search_result and target_info.get('name'):
            try:
                safe_print(f"  Searching for {target_info['name']} in all missions...")
                search_result = lk.search_lightcurve(target_info['name'])
                if search_result:
                    mission_found = "Multi"
                    safe_print(f"  Found {len(search_result)} results by name")
            except Exception as e:
                logger.debug(f"Name search failed for {target_info['name']}: {e}")
                safe_print(f"  Name search failed: {e}")

        if not search_result:
            safe_print(f"  No light curve data found for this target")

        # Download with hybrid flux approach and no caching
        if search_result:
            safe_print(f"  Found search results, attempting download...")
            # Decide which flux type to use (80% PDCSAP, 20% SAP)
            use_pdcsap = random.random() < 0.8
            flux_type_used = None
            lc_collection = None

            # Download and stitch light curves one by one to avoid cache corruption
            light_curves = []

            for i, search_entry in enumerate(search_result):
                try:
                    if use_pdcsap:
                        # Try PDCSAP first
                        try:
                            safe_print(f"  Downloading segment {i+1}/{len(search_result)} with PDCSAP...")
                            lc = search_entry.download(flux_column='pdcsap_flux', quality_bitmask=0, cache=False)
                            flux_type_used = 'pdcsap'
                        except (ValueError, TypeError, KeyError, AttributeError) as e:
                            # Fall back to SAP if PDCSAP not available
                            safe_print(f"  PDCSAP failed for segment {i+1}, trying SAP...")
                            lc = search_entry.download(flux_column='sap_flux', quality_bitmask=0, cache=False)
                            flux_type_used = 'sap'
                    else:
                        # Use SAP (20% of the time)
                        try:
                            safe_print(f"  Downloading segment {i+1}/{len(search_result)} with SAP...")
                            lc = search_entry.download(flux_column='sap_flux', quality_bitmask=0, cache=False)
                            flux_type_used = 'sap'
                        except (ValueError, TypeError, KeyError, AttributeError) as e:
                            # Try PDCSAP as fallback
                            safe_print(f"  SAP failed for segment {i+1}, trying PDCSAP...")
                            lc = search_entry.download(flux_column='pdcsap_flux', quality_bitmask=0, cache=False)
                            flux_type_used = 'pdcsap'

                    if lc is not None:
                        light_curves.append(lc)

                except Exception as e:
                    safe_print(f"  Failed to download segment {i+1}: {e}")
                    continue

            # Create a LightCurveCollection from individual light curves
            if light_curves:
                if len(light_curves) == 1:
                    lc_collection = light_curves[0]
                else:
                    from lightkurve import LightCurveCollection
                    lc_collection = LightCurveCollection(light_curves)
                safe_print(f"  Successfully downloaded {len(light_curves)} light curve segments")
                return lc_collection, mission_found

        return None, "None"

    def safe_extract_flux(self, flux):
        """Safely extract flux array from different formats"""
        if hasattr(flux, 'value'):
            flux = flux.value
        elif hasattr(flux, 'data'):
            flux = flux.data

        flux = np.asarray(flux, dtype=np.float32)  # Use float32 to save space

        if hasattr(flux, 'mask'):
            flux = np.ma.filled(flux, np.nan)

        return np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

    def calculate_transit_metadata(self, time, orbital_params, target_info=None):
        """Calculate transit timing metadata with enhanced first transit detection"""
        period = orbital_params.get('period')
        duration = orbital_params.get('duration', 4.0)

        # Enhanced epoch/first transit detection
        epoch = None

        # Try to get best first transit time
        if target_info:
            epoch = get_best_first_transit(target_info)

        # Fallback to original epoch if no first transit found
        if not epoch:
            epoch = orbital_params.get('epoch')

        if not period or not epoch:
            safe_print(f"  Cannot calculate transits: period={period}, epoch={epoch}")
            safe_print(f"  Target info: {target_info.get('name', 'Unknown')} (Mission: {target_info.get('mission', 'Unknown')})")
            safe_print(f"  Available orbital params: {list(orbital_params.keys()) if orbital_params else 'None'}")
            return None

        start_time = time[0]
        end_time = time[-1]

        time_reference = orbital_params.get('time_reference')
        transit_times = self.calculate_transit_times(epoch, period, start_time, end_time, time_reference)

        # Find indices closest to transit times
        transit_indices = []
        for transit_time in transit_times:
            idx = np.argmin(np.abs(time - transit_time))
            transit_indices.append(int(idx))

        return {
            'period': period,
            'epoch': epoch,
            'duration': duration,
            'transit_count': len(transit_times),
            'transit_times': transit_times,
            'transit_indices': transit_indices
        }

    def save_lightcurve_csv(self, target_id, flux, time, quality, target_info,
                           transit_metadata, label):
        """Save complete light curve to CSV format"""
        os.makedirs(LIGHTCURVES_DIR, exist_ok=True)

        filepath = os.path.join(LIGHTCURVES_DIR, f"{target_id}.csv")

        try:
            # Create DataFrame with light curve data
            df = pd.DataFrame({
                'time': time,
                'flux': flux,
                'quality': quality
            })

            # Add metadata as comment lines at the top
            metadata_lines = [
                f"# Target ID: {target_id}",
                f"# Label: {label}",
                f"# Data Length: {len(flux)}",
                f"# Time Span (days): {float(time[-1] - time[0]):.6f}",
                f"# Mission: {target_info.get('mission', 'Unknown')}",
                f"# Collection Timestamp: {datetime.now().isoformat()}",
                f"# TIC: {target_info.get('tic', 'N/A')}",
                f"# KIC: {target_info.get('kic', 'N/A')}",
                f"# EPIC: {target_info.get('epic', 'N/A')}",
                f"# Name: {target_info.get('name', 'N/A')}"
            ]

            # Add transit metadata if available
            if transit_metadata:
                metadata_lines.extend([
                    f"# Has Transit Metadata: True",
                    f"# Period: {transit_metadata['period']}",
                    f"# Epoch: {transit_metadata['epoch']}",
                    f"# Duration: {transit_metadata['duration']}",
                    f"# Transit Count: {transit_metadata['transit_count']}",
                    f"# Transit Times: {','.join(map(str, transit_metadata['transit_times']))}",
                    f"# Transit Indices: {','.join(map(str, transit_metadata['transit_indices']))}"
                ])
            else:
                metadata_lines.append("# Has Transit Metadata: False")

            # Write metadata and data
            with open(filepath, 'w', newline='') as f:
                for line in metadata_lines:
                    f.write(line + '\n')
                f.write('\n')  # Blank line before data

            # Append the DataFrame (this automatically adds headers)
            df.to_csv(filepath, mode='a', index=False)

            return filepath

        except Exception as e:
            logger.error(f"Failed to save CSV for {target_id}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return None

    def update_metadata_csv(self, target_id, target_info, data_length,
                          transit_metadata, label, mission):
        """Update the master metadata CSV"""
        os.makedirs(METADATA_DIR, exist_ok=True)

        metadata_record = {
            'target_id': target_id,
            'label': label,
            'mission': mission,
            'data_length': data_length,
            'has_orbital_params': transit_metadata is not None,
            'transit_count': transit_metadata['transit_count'] if transit_metadata else 0,
            'period': transit_metadata['period'] if transit_metadata else None,
            'collection_timestamp': datetime.now().isoformat(),
            'file_path': f"{target_id}.csv"
        }

        # Add target info
        for key in ['name', 'tic', 'kic', 'epic']:
            if key in target_info:
                metadata_record[key] = target_info[key]

        df = pd.DataFrame([metadata_record])

        if os.path.exists(METADATA_CSV):
            df.to_csv(METADATA_CSV, mode='a', header=False, index=False)
        else:
            df.to_csv(METADATA_CSV, mode='w', header=True, index=False)

    def get_target_id(self, target_info):
        """Generate consistent target ID"""
        if target_info.get('tic'):
            return f"TIC_{target_info['tic']}"
        elif target_info.get('kic'):
            return f"KIC_{target_info['kic']}"
        elif target_info.get('epic'):
            return f"EPIC_{target_info['epic']}"
        else:
            return f"TARGET_{random.randint(100000, 999999)}"

# ===============================
# STREAMING BATCH PROCESSOR
# ===============================

class StreamingBatchProcessor:
    def __init__(self):
        self.processor = CompleteLightCurveProcessor()
        self.collected_files = []
        self.large_batch_count = 0  # Track accumulated files for large batch upload

    def process_targets_batch(self, targets_with_labels, batch_name):
        """Process a batch of targets with streaming upload"""
        safe_print(f"\n{'='*60}")
        safe_print(f"Processing {batch_name}: {len(targets_with_labels)} targets")
        safe_print(f"Already processed: {progress_tracker.get_stats()} targets")

        # Check storage before starting
        storage_info = check_local_storage()
        if storage_info['percent_used'] > 80:
            safe_print(f"WARNING: Disk {storage_info['percent_used']:.1f}% full")

        successful = 0
        failed = 0
        skipped = 0

        for target_info, label in targets_with_labels:
            target_id = self.processor.get_target_id(target_info)

            # Check if already processed
            if progress_tracker.is_processed(target_id):
                skipped += 1
                continue

            # Check if we should create large batch zip upload
            local_data_size = check_data_directory_size()

            # Count total CSV files in directory (including from previous runs)
            total_csv_files = 0
            if os.path.exists(LIGHTCURVES_DIR):
                total_csv_files = len([f for f in os.listdir(LIGHTCURVES_DIR) if f.endswith('.csv')])

            if (self.large_batch_count >= LARGE_BATCH_SIZE or
                local_data_size > LARGE_BATCH_THRESHOLD_GB or
                total_csv_files >= LARGE_BATCH_SIZE):
                safe_print(f"Triggering COMPREHENSIVE zip upload:")
                safe_print(f"  Current session files: {self.large_batch_count}")
                safe_print(f"  Total CSV files found: {total_csv_files} (ALL runs)")
                safe_print(f"  Local data size: {local_data_size:.1f}GB")
                safe_print(f"  Will archive EVERYTHING and clean up")
                self.create_and_upload_zip_batch()

            # Check storage limits - force zip upload if critical
            storage_info = check_local_storage()
            if storage_info['percent_used'] > 85:
                safe_print("Storage nearly full, forcing zip upload and cleanup...")
                self.create_and_upload_zip_batch()

                # Check again after cleanup
                storage_info = check_local_storage()
                if storage_info['percent_used'] > 90:
                    safe_print("CRITICAL: Storage still full after cleanup, stopping collection")
                    break

            # Process the target
            filepath = self.processor.process_complete_lightcurve(target_info, label)

            if filepath:
                self.collected_files.append(filepath)
                self.large_batch_count += 1  # Track for large batch upload
                successful += 1

                # Memory cleanup
                gc.collect()
            else:
                failed += 1

            # Progress update
            if (successful + failed) % 10 == 0:
                cache_fixes = self.processor.stats['cache_errors_fixed']
                safe_print(f"  Progress: {successful} success, {failed} failed, {skipped} skipped, {cache_fixes} cache fixes")

        # Final upload of remaining files as zip batch
        if self.collected_files:
            safe_print("Creating final zip batch...")
            self.create_and_upload_zip_batch()

        safe_print(f"Batch complete: {successful} successful, {failed} failed, {skipped} skipped")
        return successful, failed

    def create_zip_batch(self, files_to_zip, batch_type="lightcurves"):
        """Create a compressed zip file with unique naming"""
        if not files_to_zip:
            return None

        # Generate unique batch identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = str(uuid.uuid4())[:8]  # Short unique ID
        file_count = len(files_to_zip)

        # Create unique zip filename
        zip_filename = f"exoseeker_{batch_type}_{timestamp}_{batch_id}_{file_count}files.zip"
        zip_path = os.path.join(UPLOAD_QUEUE_DIR, zip_filename)

        os.makedirs(UPLOAD_QUEUE_DIR, exist_ok=True)

        safe_print(f"Creating zip archive: {zip_filename}")
        safe_print(f"Compressing {file_count} files...")

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for file_path in files_to_zip:
                    if os.path.exists(file_path):
                        # Use just the filename in the archive (not full path)
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname)

                # Add batch metadata to zip
                metadata = {
                    'batch_type': batch_type,
                    'file_count': file_count,
                    'creation_timestamp': timestamp,
                    'batch_id': batch_id,
                    'files_included': [os.path.basename(f) for f in files_to_zip]
                }

                # Add metadata as JSON file in zip
                metadata_json = json.dumps(metadata, indent=2)
                zipf.writestr(f"batch_metadata_{batch_id}.json", metadata_json)

            # Check compression ratio
            original_size = sum(os.path.getsize(f) for f in files_to_zip if os.path.exists(f))
            compressed_size = os.path.getsize(zip_path)
            compression_ratio = (1 - compressed_size/original_size) * 100 if original_size > 0 else 0

            safe_print(f"Compression complete:")
            safe_print(f"  Original size: {original_size/(1024**2):.1f} MB")
            safe_print(f"  Compressed size: {compressed_size/(1024**2):.1f} MB")
            safe_print(f"  Compression ratio: {compression_ratio:.1f}%")

            return zip_path, {
                'original_size_mb': original_size/(1024**2),
                'compressed_size_mb': compressed_size/(1024**2),
                'compression_ratio': compression_ratio,
                'file_count': file_count,
                'batch_id': batch_id
            }

        except Exception as e:
            logger.error(f"Failed to create zip archive: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return None

    def create_and_upload_zip_batch(self):
        """Create zip archive of all data and upload as single batch"""
        if not self.collected_files:
            return

        safe_print(f"\n{'='*60}")
        safe_print("STARTING LARGE BATCH ZIP UPLOAD")
        safe_print(f"{'='*60}")
        safe_print(f"Files to archive: {len(self.collected_files)}")
        safe_print(f"Pausing processing for upload efficiency...")

        try:
            # Create comprehensive zip of entire streaming data directory
            zip_result = self.create_comprehensive_zip()
            if not zip_result:
                safe_print("Failed to create zip archive, falling back to individual uploads")
                self.upload_and_cleanup_batch()
                return

            zip_path, zip_info = zip_result

            # Upload the single zip file
            safe_print(f"\nUploading zip archive: {os.path.basename(zip_path)}")
            safe_print(f"Archive size: {zip_info['compressed_size_mb']:.1f}MB")
            safe_print(f"Compression ratio: {zip_info['compression_ratio']:.1f}%")

            success, file_id = drive_manager.upload_file(
                zip_path,
                os.path.basename(zip_path),
                f"{DRIVE_BASE_PATH}/zip_batches"
            )

            if success:
                safe_print(f"+ Successfully uploaded zip batch (ID: {file_id})")

                # Upload progress and metadata files separately
                self.upload_metadata_files()

                # Clean up everything except processed_targets.txt
                self.comprehensive_cleanup()

                # Update statistics
                self.processor.stats['uploads_completed'] += len(self.collected_files)
                self.processor.stats['files_deleted'] += len(self.collected_files)

                # Reset counters
                self.collected_files = []
                self.large_batch_count = 0

                safe_print(f"+ Cleaned up local files, ready to continue processing")

            else:
                safe_print("- Failed to upload zip, keeping local files")

            # Clean up zip file
            if os.path.exists(zip_path):
                os.remove(zip_path)

        except Exception as e:
            logger.error(f"Error in zip batch upload: {e}")
            safe_print(f"- Zip upload failed: {e}")
            safe_print("Falling back to individual file uploads...")
            self.upload_and_cleanup_batch()

    def create_comprehensive_zip(self):
        """Create zip archive of ENTIRE streaming data directory (all files from all runs)"""
        try:
            # Generate unique zip filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = str(uuid.uuid4())[:8]

            # Find ALL CSV files in the lightcurves directory (from all runs)
            all_csv_files = []
            total_size_mb = 0

            if os.path.exists(LIGHTCURVES_DIR):
                for filename in os.listdir(LIGHTCURVES_DIR):
                    if filename.endswith('.csv'):
                        filepath = os.path.join(LIGHTCURVES_DIR, filename)
                        if os.path.exists(filepath):
                            all_csv_files.append(filepath)
                            total_size_mb += os.path.getsize(filepath) / (1024**2)

            # Add metadata files size
            if os.path.exists(METADATA_CSV):
                total_size_mb += os.path.getsize(METADATA_CSV) / (1024**2)
            if os.path.exists(COLLECTION_LOG):
                total_size_mb += os.path.getsize(COLLECTION_LOG) / (1024**2)

            file_count = len(all_csv_files)
            zip_filename = f"exoseeker_comprehensive_{timestamp}_{batch_id}_{file_count}files_{total_size_mb:.0f}MB.zip"
            zip_path = os.path.join(UPLOAD_QUEUE_DIR, zip_filename)

            os.makedirs(UPLOAD_QUEUE_DIR, exist_ok=True)

            safe_print(f"Creating COMPREHENSIVE zip of ALL data: {zip_filename}")
            safe_print(f"Including ALL files from previous runs + current session")
            safe_print(f"Compressing {file_count} CSV files ({total_size_mb:.1f}MB total)...")

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                # Add ALL lightcurve CSV files found (from all runs)
                for filepath in all_csv_files:
                    if os.path.exists(filepath):
                        # Preserve directory structure in zip
                        arcname = os.path.relpath(filepath, DATA_DIR)
                        zipf.write(filepath, arcname)

                # Add ALL metadata files if they exist
                if os.path.exists(METADATA_CSV):
                    arcname = os.path.relpath(METADATA_CSV, DATA_DIR)
                    zipf.write(METADATA_CSV, arcname)
                    safe_print(f"+ Included metadata CSV in zip")

                # Add collection log if it exists
                if os.path.exists(COLLECTION_LOG):
                    arcname = os.path.relpath(COLLECTION_LOG, DATA_DIR)
                    zipf.write(COLLECTION_LOG, arcname)

                # Add processed targets log if it exists (for reference)
                if os.path.exists(PROCESSED_TARGETS_LOG):
                    arcname = os.path.relpath(PROCESSED_TARGETS_LOG, DATA_DIR)
                    zipf.write(PROCESSED_TARGETS_LOG, arcname)

                # Add comprehensive batch info
                batch_info = {
                    'batch_timestamp': timestamp,
                    'batch_id': batch_id,
                    'total_csv_files': file_count,
                    'uncompressed_size_mb': total_size_mb,
                    'includes_previous_runs': True,
                    'comprehensive_archive': True,
                    'files_included': [os.path.basename(f) for f in all_csv_files],
                    'creation_info': {
                        'version': 'v15.2',
                        'batch_type': 'comprehensive_full_archive',
                        'upload_strategy': 'all_data_zip_optimization'
                    }
                }

                batch_json = json.dumps(batch_info, indent=2)
                zipf.writestr(f"comprehensive_batch_info_{batch_id}.json", batch_json)

            # Calculate compression stats
            compressed_size = os.path.getsize(zip_path)
            original_size = total_size_mb * 1024 * 1024
            compression_ratio = (1 - compressed_size/original_size) * 100 if original_size > 0 else 0

            safe_print(f"+ COMPREHENSIVE zip creation complete:")
            safe_print(f"  Total CSV files archived: {file_count} (ALL runs)")
            safe_print(f"  Original size: {total_size_mb:.1f}MB")
            safe_print(f"  Compressed size: {compressed_size/(1024**2):.1f}MB")
            safe_print(f"  Compression ratio: {compression_ratio:.1f}%")
            safe_print(f"  Includes: CSV files + metadata + logs")

            return zip_path, {
                'compressed_size_mb': compressed_size/(1024**2),
                'original_size_mb': total_size_mb,
                'compression_ratio': compression_ratio,
                'file_count': file_count,
                'batch_id': batch_id,
                'comprehensive': True
            }

        except Exception as e:
            logger.error(f"Failed to create comprehensive zip: {e}")
            safe_print(f"- Comprehensive zip creation failed: {e}")
            return None

    def upload_metadata_files(self):
        """Upload metadata and progress files separately"""
        try:
            # Upload metadata CSV with timestamp
            if os.path.exists(METADATA_CSV):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata_filename = f"lightcurve_metadata_{timestamp}.csv"

                drive_manager.upload_file(
                    METADATA_CSV,
                    metadata_filename,
                    f"{DRIVE_BASE_PATH}/{DRIVE_METADATA_FOLDER}"
                )

            # Upload progress tracking file
            if os.path.exists(PROCESSED_TARGETS_LOG):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                progress_filename = f"processed_targets_{timestamp}.txt"

                drive_manager.upload_file(
                    PROCESSED_TARGETS_LOG,
                    progress_filename,
                    f"{DRIVE_BASE_PATH}/{DRIVE_METADATA_FOLDER}"
                )

        except Exception as e:
            logger.warning(f"Failed to upload some metadata files: {e}")

    def comprehensive_cleanup(self):
        """Clean up all local files except processed_targets.txt (metadata CSV already in zip)"""
        try:
            # Delete ALL lightcurve CSV files (they're now safely in the zip)
            deleted_count = 0
            if os.path.exists(LIGHTCURVES_DIR):
                for filename in os.listdir(LIGHTCURVES_DIR):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(LIGHTCURVES_DIR, filename)
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Could not delete {file_path}: {e}")

            # Clean up metadata CSV (it's already included in the zip)
            if os.path.exists(METADATA_CSV):
                try:
                    os.remove(METADATA_CSV)
                    safe_print("+ Cleaned up metadata CSV (already in zip)")
                except Exception as e:
                    logger.warning(f"Could not delete metadata CSV: {e}")

            # Clean up collection log (it's already included in the zip)
            if os.path.exists(COLLECTION_LOG):
                try:
                    os.remove(COLLECTION_LOG)
                    safe_print("+ Cleaned up collection log (already in zip)")
                except Exception as e:
                    logger.warning(f"Could not delete collection log: {e}")

            safe_print(f"+ Comprehensive cleanup complete:")
            safe_print(f"  CSV files deleted: {deleted_count} (all from all runs)")
            safe_print(f"  Metadata files cleaned (already in zip)")
            safe_print(f"  Preserved: {PROCESSED_TARGETS_LOG} (for resume capability)")

        except Exception as e:
            logger.error(f"Error during comprehensive cleanup: {e}")

    def upload_and_cleanup_batch(self):
        """Upload collected files to Google Drive and clean up locally"""
        if not self.collected_files:
            return

        # Removed troubleshooting mode - uploads are now enabled
        safe_print(f"Uploading {len(self.collected_files)} files to Google Drive...")

        # Upload light curve files
        uploaded_files = []
        for filepath in self.collected_files:
            filename = os.path.basename(filepath)

            success, file_id = drive_manager.upload_file(
                filepath,
                filename,
                f"{DRIVE_BASE_PATH}/{DRIVE_LIGHTCURVES_FOLDER}"
            )

            if success:
                uploaded_files.append(filepath)
                self.processor.stats['uploads_completed'] += 1
            else:
                safe_print(f"Failed to upload {filename}, keeping local copy")

        # Upload metadata files
        if os.path.exists(METADATA_CSV):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_filename = f"lightcurve_metadata_{timestamp}.csv"

            drive_manager.upload_file(
                METADATA_CSV,
                metadata_filename,
                f"{DRIVE_BASE_PATH}/{DRIVE_METADATA_FOLDER}"
            )

        # Upload progress tracking file
        if os.path.exists(PROCESSED_TARGETS_LOG):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            progress_filename = f"processed_targets_{timestamp}.txt"

            drive_manager.upload_file(
                PROCESSED_TARGETS_LOG,
                progress_filename,
                f"{DRIVE_BASE_PATH}/{DRIVE_METADATA_FOLDER}"
            )

        # Clean up successfully uploaded files
        for filepath in uploaded_files:
            try:
                os.remove(filepath)
                self.processor.stats['files_deleted'] += 1
                safe_print(f"Deleted local file: {os.path.basename(filepath)}")
            except Exception as e:
                logger.warning(f"Could not delete {filepath}: {e}")

        # Clear the collected files list
        self.collected_files = [f for f in self.collected_files if f not in uploaded_files]

        # Log storage status
        storage_info = check_local_storage()
        safe_print(f"Storage after cleanup: {storage_info['percent_used']:.1f}% used, "
                  f"{storage_info['free_gb']:.1f}GB free")

# ===============================
# DATA COLLECTION FUNCTIONS
# ===============================

def get_nasa_archive_planets_comprehensive():
    """Get comprehensive planet collection from NASA Exoplanet Archive - all tables"""
    safe_print("\n=== NASA EXOPLANET ARCHIVE - COMPREHENSIVE SEARCH ===")
    all_planets = []

    # Improved safe_float function to handle astropy quantities
    def safe_float(val):
        if val is None:
            return None
        try:
            # Handle masked values
            if hasattr(val, 'mask') and np.ma.is_masked(val):
                return None
            # Handle astropy quantities
            if hasattr(val, 'value'):
                return float(val.value)
            # Handle regular values
            return float(val)
        except (ValueError, TypeError, AttributeError):
            return None

    # 1. PS Table (Planetary Systems) - Main confirmed planets
    try:
        safe_print("Querying PS table (Planetary Systems)...")
        ps_planets = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="tic_id, pl_name, hostname, pl_orbper, pl_tranmid, pl_tranmidlim, pl_trandur, pl_trandep, pl_imppar, pl_ratdor, pl_ratror, ttv_flag, ra, dec",
            where="default_flag = 1"
        )
        safe_print(f"  Found {len(ps_planets)} in PS table")

        for row in ps_planets:
            planet = {
                'name': row.get('pl_name', 'Unknown'),
                'mission': 'Multi',
                'ra': row.get('ra'),
                'dec': row.get('dec')
            }

            if row.get('tic_id') is not None and not np.ma.is_masked(row.get('tic_id')):
                planet['tic'] = str(row['tic_id']).replace('TIC ', '').strip()

            # Add orbital parameters with enhanced fields
            orbital_params = {}
            period_val = safe_float(row.get('pl_orbper'))
            if period_val and period_val > 0:
                orbital_params['period'] = period_val

            # Enhanced epoch handling with multiple sources
            epoch_val = safe_float(row.get('pl_tranmid'))
            if epoch_val:
                orbital_params['epoch'] = epoch_val

            # Add epoch uncertainty and time reference system
            epoch_lim_val = safe_float(row.get('pl_tranmidlim'))
            if epoch_lim_val:
                orbital_params['epoch_uncertainty'] = epoch_lim_val

            # Note: pl_tranmid_systemref field not available in current PS table schema
            # Default to BJD time reference for most modern catalogs
            orbital_params['time_reference'] = 'BJD'

            duration_val = safe_float(row.get('pl_trandur'))
            if duration_val and duration_val > 0:
                orbital_params['duration'] = duration_val

            depth_val = safe_float(row.get('pl_trandep'))
            if depth_val:
                orbital_params['depth'] = depth_val

            # Add new transit modeling parameters
            impact_val = safe_float(row.get('pl_imppar'))
            if impact_val is not None:
                orbital_params['impact_parameter'] = impact_val

            ratdor_val = safe_float(row.get('pl_ratdor'))
            if ratdor_val and ratdor_val > 0:
                orbital_params['a_over_rs'] = ratdor_val

            ratror_val = safe_float(row.get('pl_ratror'))
            if ratror_val and ratror_val > 0:
                orbital_params['rp_over_rs'] = ratror_val

            # Add TTV flag
            ttv_val = row.get('ttv_flag')
            if ttv_val is not None and not np.ma.is_masked(ttv_val):
                orbital_params['has_ttv'] = bool(int(ttv_val)) if str(ttv_val).isdigit() else False

            if orbital_params:
                planet['orbital_params'] = orbital_params

            if planet.get('tic') or (planet.get('ra') is not None and not np.ma.is_masked(planet.get('ra'))):
                all_planets.append(planet)

    except Exception as e:
        safe_print(f"  Error querying PS table: {e}")
        logger.warning(f"Failed to get PS table planets: {type(e).__name__}: {str(e)}")

    # 2. TOI Table (TESS Objects of Interest) - Confirmed planets
    try:
        safe_print("Querying TOI confirmed planets...")
        toi_confirmed = NasaExoplanetArchive.query_criteria(
            table="toi",
            select="tid, toipfx, ra, dec",
            where="tfopwg_disp = 'CP'"
        )
        safe_print(f"  Found {len(toi_confirmed)} confirmed TOIs")

        for row in toi_confirmed:
            if row.get('tid'):
                all_planets.append({
                    'tic': str(row['tid']),
                    'name': f"TOI-{row.get('toipfx', 'Unknown')}",
                    'mission': 'TESS',
                    'ra': row.get('ra'),
                    'dec': row.get('dec')
                })

    except Exception as e:
        safe_print(f"  Error querying TOI: {e}")
        logger.warning(f"Failed to get TOI planets: {type(e).__name__}: {str(e)}")

    # 3. KOI Table (Kepler Objects of Interest) - Confirmed planets
    try:
        safe_print("Querying KOI confirmed planets...")
        koi_confirmed = NasaExoplanetArchive.query_criteria(
            table="cumulative",
            select="kepid, kepler_name, kepoi_name, koi_disposition, koi_period, koi_time0bk, koi_duration",
            where="koi_disposition = 'CONFIRMED'"
        )
        safe_print(f"  Found {len(koi_confirmed)} confirmed KOIs")

        for row in koi_confirmed:
            if row.get('kepid'):
                planet = {
                    'kic': str(row['kepid']),
                    'name': row.get('kepler_name', row.get('kepoi_name', 'Unknown')),
                    'mission': 'Kepler'
                }

                # Add KOI orbital parameters (first transit source)
                orbital_params = {}
                koi_period = safe_float(row.get('koi_period'))
                if koi_period and koi_period > 0:
                    orbital_params['period'] = koi_period

                koi_first_transit = safe_float(row.get('koi_time0bk'))
                if koi_first_transit:
                    orbital_params['first_transit'] = koi_first_transit  # This is the key enhancement!
                    orbital_params['epoch'] = koi_first_transit  # Also set as epoch for compatibility

                koi_duration = safe_float(row.get('koi_duration'))
                if koi_duration and koi_duration > 0:
                    orbital_params['duration'] = koi_duration

                if orbital_params:
                    planet['orbital_params'] = orbital_params

                all_planets.append(planet)

    except Exception as e:
        safe_print(f"  Error querying KOI: {e}")
        logger.warning(f"Failed to get KOI planets: {type(e).__name__}: {str(e)}")

    # 4. K2 Confirmed Planets (fixed table name)
    try:
        safe_print("Querying K2 confirmed planets...")
        k2_confirmed = NasaExoplanetArchive.query_criteria(
            table="k2pandc",
            select="pl_name, epic_hostname, ra, dec, pl_orbper, pl_tranmid, pl_trandur",
            where="disposition = 'CONFIRMED'"
        )
        safe_print(f"  Found {len(k2_confirmed)} confirmed K2 planets")

        for row in k2_confirmed:
            if row.get('epic_hostname'):
                epic_id = str(row['epic_hostname']).replace('EPIC ', '').strip()
                planet = {
                    'epic': epic_id,
                    'name': row.get('pl_name', f"EPIC-{epic_id}"),
                    'mission': 'K2',
                    'ra': row.get('ra'),
                    'dec': row.get('dec')
                }

                # Add K2 orbital parameters (first transit source)
                orbital_params = {}
                k2_period = safe_float(row.get('pl_orbper'))
                if k2_period and k2_period > 0:
                    orbital_params['period'] = k2_period

                k2_first_transit = safe_float(row.get('pl_tranmid'))
                if k2_first_transit:
                    orbital_params['first_transit'] = k2_first_transit  # This is the key enhancement!
                    orbital_params['epoch'] = k2_first_transit  # Also set as epoch for compatibility

                k2_duration = safe_float(row.get('pl_trandur'))
                if k2_duration and k2_duration > 0:
                    orbital_params['duration'] = k2_duration

                if orbital_params:
                    planet['orbital_params'] = orbital_params

                all_planets.append(planet)

    except Exception as e:
        safe_print(f"  Error querying K2 candidates: {e}")
        logger.warning(f"Failed to get K2 candidates: {type(e).__name__}: {str(e)}")

    safe_print(f"Total planets collected from NASA Archive: {len(all_planets)}")
    return all_planets

def get_comprehensive_non_planets():
    """Get comprehensive non-planet collection for negative examples"""
    safe_print("\n=== COLLECTING COMPREHENSIVE NON-PLANETS ===")
    non_planets = []

    # 1. Kepler False Positives
    try:
        safe_print("Getting Kepler FALSE POSITIVES...")
        false_positives = NasaExoplanetArchive.query_criteria(
            table="cumulative",
            select="kepid, kepoi_name",
            where="koi_disposition = 'FALSE POSITIVE'"
        )
        safe_print(f"  Found {len(false_positives)} Kepler false positives")

        for row in false_positives:
            if row.get('kepid'):
                non_planets.append({
                    'kic': str(row['kepid']),
                    'name': f"KOI {row.get('kepoi_name', 'Unknown')} (FP)",
                    'mission': 'Kepler'
                })

    except Exception as e:
        safe_print(f"  Error getting false positives: {e}")
        logger.warning(f"Failed to get false positives: {type(e).__name__}: {str(e)}")

    # 2. TOI False Positives
    try:
        safe_print("Getting TOI FALSE POSITIVES...")
        toi_false_positives = NasaExoplanetArchive.query_criteria(
            table="toi",
            select="tid, toipfx, ra, dec",
            where="tfopwg_disp = 'FP'"
        )
        safe_print(f"  Found {len(toi_false_positives)} TOI false positives")

        for row in toi_false_positives:
            if row.get('tid'):
                non_planets.append({
                    'tic': str(row['tid']),
                    'name': f"TOI-{row.get('toipfx', 'Unknown')} (FP)",
                    'mission': 'TESS',
                    'ra': row.get('ra'),
                    'dec': row.get('dec')
                })

    except Exception as e:
        safe_print(f"  Error getting TOI false positives: {e}")
        logger.warning(f"Failed to get TOI false positives: {type(e).__name__}: {str(e)}")

    # 3. TOI False Alarms (additional from testing)
    try:
        safe_print("Getting TOI FALSE ALARMS...")
        toi_false_alarms = NasaExoplanetArchive.query_criteria(
            table="toi",
            select="tid, toipfx, ra, dec",
            where="tfopwg_disp = 'FA'"
        )
        safe_print(f"  Found {len(toi_false_alarms)} TOI false alarms")

        for row in toi_false_alarms:
            if row.get('tid'):
                non_planets.append({
                    'tic': str(row['tid']),
                    'name': f"TOI-{row.get('toipfx', 'Unknown')} (FA)",
                    'mission': 'TESS',
                    'ra': row.get('ra'),
                    'dec': row.get('dec')
                })

    except Exception as e:
        safe_print(f"  Error getting TOI false alarms: {e}")
        logger.warning(f"Failed to get TOI false alarms: {type(e).__name__}: {str(e)}")

    # 4. K2 False Positives (fixed table name)
    try:
        safe_print("Getting K2 FALSE POSITIVES...")
        k2_false_positives = NasaExoplanetArchive.query_criteria(
            table="k2pandc",
            select="pl_name, epic_hostname, ra, dec",
            where="disposition = 'FALSE POSITIVE'"
        )
        safe_print(f"  Found {len(k2_false_positives)} K2 false positives")

        for row in k2_false_positives:
            if row.get('epic_hostname'):
                epic_id = str(row['epic_hostname']).replace('EPIC ', '').strip()
                non_planets.append({
                    'epic': epic_id,
                    'name': f"{row.get('pl_name', f'EPIC-{epic_id}')} (K2-FP)",
                    'mission': 'K2',
                    'ra': row.get('ra'),
                    'dec': row.get('dec')
                })

    except Exception as e:
        safe_print(f"  Error getting K2 false positives: {e}")
        logger.warning(f"Failed to get K2 false positives: {type(e).__name__}: {str(e)}")

    safe_print(f"Total non-planets collected: {len(non_planets)}")
    return non_planets

def deduplicate_targets(targets):
    """Remove duplicate targets based on identifiers"""
    seen = set()
    unique_targets = []

    for target in targets:
        key = None
        if target.get('tic'):
            key = f"TIC_{target['tic']}"
        elif target.get('kic'):
            key = f"KIC_{target['kic']}"
        elif target.get('epic'):
            key = f"EPIC_{target['epic']}"
        elif target.get('name'):
            key = f"NAME_{target['name'].replace(' ', '_')}"

        if key and key not in seen:
            seen.add(key)
            unique_targets.append(target)

    return unique_targets

# ===============================
# MAIN EXECUTION
# ===============================

def create_directory_structure():
    """Create necessary directories"""
    dirs = [DATA_DIR, LIGHTCURVES_DIR, METADATA_DIR, UPLOAD_QUEUE_DIR, LOG_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def refetch_and_update_metadata():
    """Refetch metadata for existing CSV files and update with enhanced fields"""
    safe_print("\n" + "="*60)
    safe_print("REFETCHING METADATA FOR EXISTING CSV FILES")
    safe_print("="*60)

    # Find all existing CSV files
    csv_files = []
    if os.path.exists(LIGHTCURVES_DIR):
        csv_files = [f for f in os.listdir(LIGHTCURVES_DIR) if f.endswith('.csv') and f.startswith('TIC_')]

    if not csv_files:
        safe_print("No existing CSV files found to update")
        return

    safe_print(f"Found {len(csv_files)} existing CSV files to update")

    # Get enhanced metadata for all confirmed planets
    safe_print("Fetching enhanced metadata from NASA Exoplanet Archive...")
    all_planets = get_nasa_archive_planets_comprehensive()

    # Create lookup dictionary by TIC ID
    planet_lookup = {}
    for planet in all_planets:
        if planet.get('tic'):
            planet_lookup[planet['tic']] = planet

    safe_print(f"Retrieved metadata for {len(planet_lookup)} confirmed planets")

    # Initialize processor for metadata calculations
    processor = CompleteLightCurveProcessor()
    updated_count = 0

    for csv_file in csv_files:
        try:
            # Extract TIC ID from filename
            tic_id = csv_file.replace('.csv', '').replace('TIC_', '')

            if tic_id in planet_lookup:
                safe_print(f"Updating metadata for {csv_file}...")

                planet_info = planet_lookup[tic_id]
                filepath = os.path.join(LIGHTCURVES_DIR, csv_file)

                # Read existing CSV to get flux data
                df = pd.read_csv(filepath, comment='#')
                if len(df) > 0 and 'time' in df.columns and 'flux' in df.columns:
                    time = df['time'].values
                    flux = df['flux'].values
                    quality = df['quality'].values if 'quality' in df.columns else np.zeros_like(flux)

                    # Calculate transit metadata with enhanced parameters
                    transit_metadata = None
                    orbital_params = planet_info.get('orbital_params')
                    if orbital_params and orbital_params.get('period'):
                        transit_metadata = processor.calculate_transit_metadata(time, orbital_params, planet_info)

                    # Generate target ID and update metadata
                    target_id = f"TIC_{tic_id}"
                    processor.save_lightcurve_csv(target_id, flux, time, quality,
                                                planet_info, transit_metadata, 2)

                    # Update the master metadata CSV
                    processor.update_metadata_csv(target_id, planet_info, len(flux),
                                                 transit_metadata, 2, planet_info.get('mission', 'TESS'))

                    updated_count += 1
                    safe_print(f"  + Updated {csv_file}")
                else:
                    safe_print(f"  - Skipped {csv_file} - invalid data format")
            else:
                safe_print(f"  - Skipped {csv_file} - no metadata found for TIC {tic_id}")

        except Exception as e:
            safe_print(f"  - Error updating {csv_file}: {e}")

    safe_print(f"\nMetadata update complete:")
    safe_print(f"  Files processed: {len(csv_files)}")
    safe_print(f"  Successfully updated: {updated_count}")
    safe_print(f"  Skipped/failed: {len(csv_files) - updated_count}")

def print_final_statistics(processor):
    """Print final collection statistics"""
    stats = processor.stats

    safe_print("\n" + "="*80)
    safe_print("STREAMING COLLECTION COMPLETED")
    safe_print("="*80)
    safe_print(f"Planets processed: {stats['planets_processed']}")
    safe_print(f"Non-planets processed: {stats['non_planets_processed']}")
    safe_print(f"Total files uploaded: {stats['uploads_completed']}")
    safe_print(f"Local files deleted: {stats['files_deleted']}")
    safe_print(f"Cache errors fixed: {stats['cache_errors_fixed']}")
    safe_print(f"Total targets processed from start: {progress_tracker.get_stats()}")

    # Storage status
    storage_info = check_local_storage()
    safe_print(f"\nFinal storage status:")
    safe_print(f"  Used: {storage_info['used_gb']:.1f}GB ({storage_info['percent_used']:.1f}%)")
    safe_print(f"  Free: {storage_info['free_gb']:.1f}GB")

    safe_print(f"\nGoogle Drive Structure:")
    safe_print(f"  {DRIVE_BASE_PATH}/{DRIVE_LIGHTCURVES_FOLDER}/ - Light curve CSV files")
    safe_print(f"  {DRIVE_BASE_PATH}/{DRIVE_METADATA_FOLDER}/ - Metadata and progress logs")

    safe_print(f"\nProgress tracking:")
    safe_print(f"  Processed targets log: {PROCESSED_TARGETS_LOG}")
    safe_print(f"  Resume from this point on next run")

def main():
    parser = argparse.ArgumentParser(description="ExoSeeker Streaming Data Collection v15.3")
    parser.add_argument('--planets-only', action='store_true', help="Collect only planets")
    parser.add_argument('--non-planets-only', action='store_true', help="Collect only non-planets")
    parser.add_argument('--max-targets', type=int, help="Maximum targets to process")
    parser.add_argument('--resume', action='store_true', help="Resume from previous progress")
    parser.add_argument('--refetch-metadata', action='store_true', help="Refetch and update metadata for existing CSV files with enhanced fields. Can be combined with --resume to both update existing files and continue collection.")

    args = parser.parse_args()

    safe_print("\n" + "="*80)
    safe_print("EXOSEEKER STREAMING DATA COLLECTION v15.3 - Comprehensive Zip Fix")
    safe_print("="*80)
    safe_print("MAJOR OPTIMIZATION: Zip-based batch uploads (50-80x faster than individual uploads)")
    if args.resume:
        safe_print(f"RESUMING: {progress_tracker.get_stats()} targets already processed")

    if args.refetch_metadata:
        safe_print("REFETCH METADATA MODE: Updating existing CSV files with enhanced metadata")

    create_directory_structure()

    # Test Google Drive connection
    drive_service = drive_manager.get_authenticated_service()
    if drive_service:
        safe_print("Google Drive connection successful")
    else:
        safe_print("WARNING: Google Drive connection failed - credentials not configured")
        safe_print("Continuing with local testing (files will be saved locally only)")

    # Handle refetch-metadata mode
    if args.refetch_metadata:
        refetch_and_update_metadata()
        if not args.resume and not args.planets_only and not args.non_planets_only:
            # If only refetch-metadata was requested, exit after updating
            safe_print("\nMetadata refetch completed. Exiting.")
            return

    # Initialize processor
    batch_processor = StreamingBatchProcessor()

    try:
        if not args.non_planets_only:
            # Collect planets from all sources
            safe_print("\nCOLLECTING COMPREHENSIVE CONFIRMED PLANETS...")
            all_planets = get_nasa_archive_planets_comprehensive()

            # Deduplicate planets
            safe_print(f"Deduplicating {len(all_planets)} planets...")
            unique_planets = deduplicate_targets(all_planets)
            safe_print(f"After deduplication: {len(unique_planets)} unique planets")

            if args.max_targets:
                # Use the first N real targets from the archive
                unique_planets = unique_planets[:args.max_targets]

            planet_targets = [(p, LABEL_PLANET) for p in unique_planets]
            batch_processor.process_targets_batch(planet_targets, "Comprehensive Confirmed Planets")

        if not args.planets_only:
            # Collect non-planets from all sources
            safe_print("\nCOLLECTING COMPREHENSIVE NON-PLANETS...")
            all_non_planets = get_comprehensive_non_planets()

            # Deduplicate non-planets
            safe_print(f"Deduplicating {len(all_non_planets)} non-planets...")
            unique_non_planets = deduplicate_targets(all_non_planets)
            safe_print(f"After deduplication: {len(unique_non_planets)} unique non-planets")

            if args.max_targets:
                unique_non_planets = unique_non_planets[:args.max_targets]

            non_planet_targets = [(np, LABEL_NON_PLANET) for np in unique_non_planets]
            batch_processor.process_targets_batch(non_planet_targets, "Comprehensive False Positives")

        # Final statistics
        print_final_statistics(batch_processor.processor)

    except KeyboardInterrupt:
        safe_print("\nCollection interrupted by user")
        # Upload any remaining files before exit
        if batch_processor.collected_files:
            safe_print("Uploading remaining files before exit...")
            batch_processor.upload_and_cleanup_batch()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

    finally:
        safe_print("Streaming collection ended")
        safe_print(f"Progress saved to: {PROCESSED_TARGETS_LOG}")

if __name__ == '__main__':
    random.seed(42)
    main()