import os
import sys
import numpy as np
import yaml
import mne
import datetime
import pickle
from glob import glob
from typing import List, Tuple, Dict, Optional, Union
import logging
import re

from datasets import RawDataset, BandpowerDataset, EventSegmentDataset
from data_processing import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_band_power(data: np.ndarray, 
                      sfreq: float, 
                      band_range: List[float], 
                      window_size_ms: float, 
                      overlap: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract band power using sliding windows
    
    Args:
        data: Input data in shape (channels, time)
        sfreq: Sampling frequency
        band_range: Frequency band for extraction [low, high]
        window_size_ms: Window size in milliseconds
        overlap: Overlap ratio between windows
        
    Returns:
        Tuple containing:
            - Band power data (Windows x Channels)
            - Array of window center times
    """
    from scipy.signal import butter, filtfilt
    
    n_channels = data.shape[0]
    
    # Calculate window size in samples
    window_samples = int((window_size_ms / 1000) * sfreq)
    
    # Check if window is large enough for filtering
    min_window_size = 30  # Minimum size for stable filtering
    
    if window_samples < min_window_size:
        logger.warning(f"Window size too small for filtering ({window_samples} samples).")
        logger.warning(f"Increasing window size from {window_size_ms}ms to {min_window_size/sfreq*1000:.1f}ms")
        window_samples = min_window_size
        window_size_ms = (window_samples / sfreq) * 1000
    
    # Calculate step size based on overlap
    step_size = max(1, int(window_samples * (1 - overlap)))
    
    # Calculate number of windows
    n_samples = data.shape[1]
    n_windows = (n_samples - window_samples) // step_size + 1
    
    # Design bandpass filter
    nyquist = sfreq / 2
    filter_order = 2  # Lower order for better stability with short windows
    b, a = butter(filter_order, [band_range[0] / nyquist, band_range[1] / nyquist], btype='band')
    
    # Initialize band power matrix
    band_power = np.zeros((n_windows, n_channels))
    window_times = np.zeros(n_windows)
    
    # Process each window
    for i in range(n_windows):
        # Extract window
        start_idx = i * step_size
        end_idx = start_idx + window_samples
        
        # Skip if we're at the end of the data
        if end_idx > n_samples:
            break
            
        window = data[:, start_idx:end_idx]
        
        # Apply bandpass filter - handle channels separately
        filtered_window = np.zeros_like(window)
        for ch in range(n_channels):
            filtered_window[ch] = filtfilt(b, a, window[ch])
        
        # Calculate band power (mean squared amplitude)
        power = np.mean(filtered_window ** 2, axis=1)
        band_power[i, :] = power
        
        # Store the central time point of the window
        window_times[i] = (start_idx + window_samples // 2) / sfreq
    
    return band_power, window_times

def map_events_to_windows(event_array: np.ndarray, 
                         window_times: np.ndarray, 
                         sfreq: float, 
                         window_size_ms: float,
                         chunk_start_time_abs: float) -> np.ndarray:
    """
    Map event array to windows based on window center times
    
    Args:
        event_array: Event array for raw data (samples) with values 0, 1, 2, 3
        window_times: Array of window center times (absolute time, relative to reference)
        sfreq: Sampling frequency
        window_size_ms: Window size in milliseconds
        chunk_start_time_abs: Absolute start time of the chunk corresponding to event_array
        
    Returns:
        Array of event labels for windows (preserving event types 1, 2, 3)
    """
    window_events = np.zeros(len(window_times))
    
    # Half window in samples
    half_window = int(window_size_ms * sfreq / 2000)
    
    # Debug logging
    logger.info(f"  Debug map_events_to_windows: event_array shape={event_array.shape}, sum={np.sum(event_array > 0)}")
    logger.info(f"  Debug map_events_to_windows: window_times shape={window_times.shape}")
    logger.info(f"  Debug map_events_to_windows: half_window={half_window} samples")
    
    mapped_event_count = 0
    
    for i, window_time in enumerate(window_times):
        # Calculate the time relative to the start of the chunk
        time_in_chunk = window_time - chunk_start_time_abs
        
        # Calculate the sample index corresponding to the window center *within the chunk*
        central_sample_idx_in_chunk = int(time_in_chunk * sfreq)
        
        # Use a small window around the central point *within the chunk* to check for events
        start_sample_in_chunk = max(0, central_sample_idx_in_chunk - half_window)
        end_sample_in_chunk = min(len(event_array), central_sample_idx_in_chunk + half_window)
        
        # Extract event window from the chunk's event array
        # Ensure indices are valid before slicing
        if start_sample_in_chunk < end_sample_in_chunk and start_sample_in_chunk < len(event_array) and end_sample_in_chunk > 0:
            window = event_array[start_sample_in_chunk:end_sample_in_chunk]
        else:
            window = np.array([]) # Empty window if indices are invalid
            
        # Debug log for problematic windows
        if i < 5 or (window.size > 0 and np.any(window > 0)):
            logger.debug(f"  Window {i}: time={window_time:.2f}, time_in_chunk={time_in_chunk:.2f}, " +
                        f"central_idx={central_sample_idx_in_chunk}, " +
                        f"range=[{start_sample_in_chunk}:{end_sample_in_chunk}], " +
                        f"window_sum={np.sum(window > 0) if window.size > 0 else 0}")
        
        # If any events in window, find the most important one (prioritize type 3, then 2, then 1)
        if window.size > 0 and np.any(window > 0):
            if np.any(window == 3):
                window_events[i] = 3  # Both bands (highest priority)
            elif np.any(window == 2):
                window_events[i] = 2  # High band
            else:
                window_events[i] = 1  # Low band
            mapped_event_count += 1
    
    logger.info(f"  Debug map_events_to_windows: Mapped {mapped_event_count} events out of {len(window_times)} windows")
    
    return window_events

def extract_event_segments(files: List[str], config: dict, max_event_segments: Optional[int] = None) -> EventSegmentDataset:
    """
    Extract segments around high-frequency events
    
    Args:
        files: List of data files to process
        config: Configuration dictionary
        max_event_segments: Maximum number of event segments to extract (optional)
        
    Returns:
        EventSegmentDataset with extracted segments
    """
    # Create dataset
    event_segment_dataset = EventSegmentDataset(config)
    
    # Get event detection parameters
    event_params = config.get('event_detection', {
        'low_band': [80, 250],
        'high_band': [250, 451],
        'rel_thresh': 3.0,
        'abs_thresh': 3.0,
        'min_gap': 20,
        'min_last': 20
    })
    
    # Get segment parameters
    segment_params = config.get('event_segments', {
        'fixed_segment_length_ms': 100,  # All segments will have exactly this length
    })
    
    # Get segment length parameters
    fixed_segment_length_ms = segment_params.get('fixed_segment_length_ms', 100)
    
    total_segments = 0
    all_segment_files = []
    all_segment_indices = []
    
    # Process files until we reach max_event_segments
    for file_idx, file in enumerate(files):
        if max_event_segments is not None and total_segments >= max_event_segments:
            logger.info(f"Reached maximum segments ({max_event_segments})")
            break
            
        try:
            # Parse timestamp from filename
            try:
                timestamp = parse_timestamp_from_filename(file)
            except ValueError as e:
                logger.error(f"Error parsing timestamp from {file}: {e}")
                continue
                
            # Load raw data
            raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
            
            # Log original channel names
            logger.info(f"Original channels in {os.path.basename(file)}: {raw.ch_names}")

            # Use all channels as they are already in the correct bipolar format
            eeg_picks = list(range(len(raw.ch_names)))
            logger.info(f"Using all {len(eeg_picks)} available channels")

            # Pick the selected channels (all channels in this case)
            raw.pick(eeg_picks)
            logger.info(f"Using all {len(raw.ch_names)} available channels: {raw.ch_names}")
            
            # Get original data and sampling frequency for selected channels
            original_data = raw.get_data()  # Channels x Time
            original_sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
            
            # Calculate length parameters in samples
            fixed_segment_length_samples = int(fixed_segment_length_ms * original_sfreq / 1000)
            
            # Detect events
            logger.info(f"Detecting events in {os.path.basename(file)}")
            
            # Get low band and high band event arrays
            low_band = event_params.get('low_band', [80, 250])
            high_band = event_params.get('high_band', [250, 451])
            rel_thresh = event_params.get('rel_thresh', 3.0)
            abs_thresh = event_params.get('abs_thresh', 3.0)
            min_gap = event_params.get('min_gap', 20)
            min_last = event_params.get('min_last', 20)  # in milliseconds
            
            logger.info(f"Using event detection parameters: rel_thresh={rel_thresh}, abs_thresh={abs_thresh}, min_gap={min_gap}ms, min_last={min_last}ms")
            
            # Detect events for low band
            logger.info(f"Detecting low-band events in frequency range: {low_band} Hz")
            
            # Detect events for both datasets
            file_low_band_events = detect_interictal_events(
                original_data, 
                original_sfreq,
                ch_names, 
                freq_band=low_band,
                rel_thresh=rel_thresh,
                abs_thresh=abs_thresh,
                min_gap=min_gap,
                min_last=min_last  # keep as milliseconds
            )
            
            # Detect high-band events
            logger.info(f"Detecting high-band events in frequency range: {high_band} Hz")
            file_high_band_events = detect_interictal_events(
                original_data, 
                original_sfreq,
                ch_names, 
                freq_band=high_band,
                rel_thresh=rel_thresh,
                abs_thresh=abs_thresh,
                min_gap=min_gap,
                min_last=min_last  # keep as milliseconds
            )
            
            # Combine events: where both bands have events, mark as type 3
            file_combined_events = np.zeros_like(file_low_band_events)
            file_combined_events[file_low_band_events > 0] = 1  # Low band
            file_combined_events[file_high_band_events > 0] = 2  # High band
            
            # Where both bands have events, mark as type 3
            both_bands = (file_low_band_events > 0) & (file_high_band_events > 0)
            file_combined_events[both_bands] = 3  # Both bands
            
            # Find contiguous event regions
            combined_events = find_contiguous_events(file_combined_events)
            
            # Log event counts to diagnose potential inconsistencies
            logger.info(f"Found {len(combined_events)} combined events in {os.path.basename(file)}")
            
            if not combined_events:
                logger.info(f"No events detected in {os.path.basename(file)}")
                continue
            
            # Process all event regions
            file_segments = []
            segment_event_types = []
            segment_start_idx = []
            segment_end_idx = []
            
            # Process each combined event region
            for event_idx, (start_idx, end_idx) in enumerate(combined_events):
                # Determine the event type at this position
                if file_combined_events[start_idx] == 3:
                    event_type = 3  # Both bands
                elif file_combined_events[start_idx] == 2:
                    event_type = 2  # High band
                else:
                    event_type = 1  # Low band
                
                # Get the actual event duration
                event_duration = end_idx - start_idx
                
                # Calculate segment boundaries to get exactly fixed_segment_length_samples
                if event_duration <= fixed_segment_length_samples:
                    # Event is shorter than the desired segment length
                    # Center the event in the segment
                    event_center = (start_idx + end_idx) // 2
                    half_segment = fixed_segment_length_samples // 2
                    
                    # Center the segment on the event
                    segment_start = max(0, event_center - half_segment)
                    segment_end = segment_start + fixed_segment_length_samples
                    
                    # Adjust if we go beyond the data bounds
                    if segment_end > len(file_combined_events):
                        segment_end = len(file_combined_events)
                        segment_start = max(0, segment_end - fixed_segment_length_samples)
                else:
                    # Event is longer than the desired segment length
                    # Center on the event
                    event_center = (start_idx + end_idx) // 2
                    half_segment = fixed_segment_length_samples // 2
                    
                    # Center the segment on the event
                    segment_start = max(0, event_center - half_segment)
                    segment_end = segment_start + fixed_segment_length_samples
                    
                    # Adjust if we go beyond the data bounds
                    if segment_end > len(file_combined_events):
                        segment_end = len(file_combined_events)
                        segment_start = max(0, segment_end - fixed_segment_length_samples)
                
                # Ensure exact segment length
                actual_length = segment_end - segment_start
                if actual_length < fixed_segment_length_samples:
                    # This can happen at file boundaries - skip segments that are too short
                    logger.debug(f"Skipping event segment (cannot get full length: {actual_length} < {fixed_segment_length_samples})")
                    continue
                
                # Extract segment
                segment = original_data[:, segment_start:segment_end]
                
                # Create event labels for the segment - preserves the actual event positions
                segment_events = np.zeros(segment_end - segment_start)
                
                # Map original event positions to segment positions
                rel_start = start_idx - segment_start
                rel_end = end_idx - segment_start
                
                # Make sure we're within bounds
                if rel_end > 0 and rel_start < len(segment_events):
                    rel_start = max(0, rel_start)
                    rel_end = min(len(segment_events), rel_end)
                    # Set event labels for the actual event duration
                    segment_events[rel_start:rel_end] = event_type
                
                # Add to list
                file_segments.append(segment)
                segment_event_types.append(event_type)
                segment_start_idx.append(segment_start)
                segment_end_idx.append(segment_end)
                
                # Update counter
                total_segments += 1
                
                # Check if we've reached the maximum
                if max_event_segments is not None and total_segments >= max_event_segments:
                    break
            
            # If we found segments in this file, add to dataset
            if file_segments:
                # Generate absolute time arrays for each segment
                segment_abs_times = []
                segment_rel_times = []
                segment_events_list = []
                segment_day_night = []
                
                for i, (segment, start_idx, end_idx) in enumerate(zip(file_segments, segment_start_idx, segment_end_idx)):
                    # Generate absolute time array for this segment
                    segment_length = segment.shape[1]
                    
                    # Generate time array (absolute time from file timestamp)
                    abs_time = generate_time_array(
                        timestamp,
                        segment_length,
                        original_sfreq,
                        reference_time=None  # Use file timestamp as reference
                    )
                    
                    # Adjust the time to start from the segment's start sample
                    start_time_offset = start_idx / original_sfreq
                    abs_time = abs_time + start_time_offset
                    
                    # Create relative time array starting from 0
                    rel_time = np.arange(segment_length) / original_sfreq
                    
                    # Create event labels - mark the actual event positions
                    event_labels = np.zeros(segment_length)
                    
                    # Map original event positions to segment positions
                    rel_start = start_idx - start_idx  # Will be 0
                    rel_end = end_idx - start_idx
                    
                    # Make sure we're within bounds
                    if rel_end > 0 and rel_start < segment_length:
                        rel_start = max(0, rel_start)
                        rel_end = min(segment_length, rel_end)
                        # Set event labels for the actual event duration
                        event_labels[rel_start:rel_end] = segment_event_types[i]
                    
                    # Create day/night labels
                    day_night_labels = generate_day_night_labels(
                        abs_time,
                        night_start_hour=event_segment_dataset.night_start_hour,
                        night_end_hour=event_segment_dataset.night_end_hour,
                        reference_time=None  # Use absolute timestamps directly
                    )
                    
                    segment_abs_times.append(abs_time)
                    segment_rel_times.append(rel_time)
                    segment_events_list.append(event_labels)
                    segment_day_night.append(day_night_labels)
                
                # Create source information
                segment_sources = []
                for i, (segment, start_idx, end_idx) in enumerate(zip(file_segments, segment_start_idx, segment_end_idx)):
                    source_info = {
                        'file': file,
                        'segment_idx': total_segments - len(file_segments) + i,
                        'start_sample': start_idx,
                        'end_sample': end_idx,
                        'shape': segment.shape,
                        'event_type': segment_event_types[i]
                    }
                    segment_sources.append(source_info)
                
                # Add all segments to the dataset
                for i, segment in enumerate(file_segments):
                    event_segment_dataset.add_segment(
                        segment,
                        segment_abs_times[i],
                        segment_rel_times[i],
                        segment_events_list[i],
                        segment_day_night[i],
                        segment_sources[i]
                    )
                
                all_segment_files.extend([file] * len(file_segments))
                all_segment_indices.extend(list(range(total_segments - len(file_segments), total_segments)))
                
                logger.info(f"Added {len(file_segments)} segments from {os.path.basename(file)}: {sum(t==1 for t in segment_event_types)} type 1, {sum(t==2 for t in segment_event_types)} type 2, {sum(t==3 for t in segment_event_types)} type 3")
                
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Count event types
    type1_count = 0
    type2_count = 0
    type3_count = 0
    
    if event_segment_dataset.event_labels:
        for labels in event_segment_dataset.event_labels:
            type1_count += np.sum(labels == 1)
            type2_count += np.sum(labels == 2)
            type3_count += np.sum(labels == 3)
    
    logger.info(f"Event types: {type1_count} type 1 (low band), {type2_count} type 2 (high band), {type3_count} type 3 (both bands)")
    
    # Store metadata
    event_segment_dataset.set_metadata({
        'segment_files': all_segment_files,
        'segment_indices': all_segment_indices,
        'fixed_segment_length_ms': fixed_segment_length_ms,
        'extraction_config': config
    })
    
    return event_segment_dataset

def find_contiguous_events(event_array):
    """
    Find contiguous event regions in a binary event array
    
    Args:
        event_array: Binary array where 1 indicates an event
        
    Returns:
        List of (start_idx, end_idx) tuples for each contiguous event
    """
    # Find transitions (0->1 or 1->0)
    transitions = np.diff(np.concatenate([[0], event_array.astype(int), [0]]))
    
    # Rising edges (start of events)
    rise_indices = np.where(transitions == 1)[0]
    
    # Falling edges (end of events)
    fall_indices = np.where(transitions == -1)[0]
    
    if len(rise_indices) != len(fall_indices):
        # This shouldn't happen, but just in case
        logger.warning(f"Mismatched event transitions: {len(rise_indices)} rises vs {len(fall_indices)} falls")
        # Use the minimum to avoid indexing errors
        min_len = min(len(rise_indices), len(fall_indices))
        rise_indices = rise_indices[:min_len]
        fall_indices = fall_indices[:min_len]
    
    # Create pairs of (start, end) indices
    event_regions = list(zip(rise_indices, fall_indices))
    
    return event_regions

# Add a new function to parse subject information from filename
def parse_subject_info_from_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse subject information from filename with format {subject_id}_{ID}_raw.fif
    Examples: 'fengling_IID1_raw.fif' or 'fengling_SZ1_raw.fif'
    
    Args:
        filename: The filename to parse
        
    Returns:
        Tuple of (subject_id, period, ID)
    """
    basename = os.path.basename(filename)
    
    # Pattern for format: {subject_id}_{ID}_raw.fif
    pattern = r"(.+?)_([A-Za-z]+)(\d+)_raw\.fif"
    
    match = re.match(pattern, basename)
    
    if match:
        subject_id, period, id_num = match.groups()
        return subject_id, period, id_num
    else:
        raise ValueError(f"Filename {basename} does not match expected subject format")

def construct_dataset_from_config(config_path: str) -> Tuple[Optional[RawDataset], Optional[BandpowerDataset], Optional[EventSegmentDataset]]:
    """
    Construct datasets based on configuration file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Tuple of (RawDataset, BandpowerDataset, EventSegmentDataset)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    data_dir = config.get('data_dir', './preprocessed/bipolar')
    chunk_size_sec = config.get('chunk_size_sec', 200)  # Default chunk size: 200 seconds
    max_files = config.get('max_files', None)
    save_dir = config.get('save_dir', './datasets')
    
    # Datasets to create
    create_raw_dataset = config.get('create_raw_dataset', True)
    create_bandpower_dataset = config.get('create_bandpower_dataset', True)
    create_event_segment_dataset = config.get('create_event_segment_dataset', False)

    # Check if we're using the subject-based grouping
    use_subject_format = config.get('use_subject_format', False)
    
    # Event detection parameters
    event_params = config.get('event_detection', {
        'low_band': [80, 250],
        'high_band': [250, 451],
        'rel_thresh': 3.0,
        'abs_thresh': 3.0,
        'min_gap': 20,
        'min_last': 20
    })
    
    # Bandpower parameters
    bandpower_params = config.get('bandpower', {
        'window_size_ms': 100,
        'overlap': 0.5,
        'bands': {
            'gamma': [80, 250],
            'high_gamma': [250, 400]
        }
    })
    
    # Raw dataset parameters
    raw_params = config.get('raw_dataset', {
        'resample_freq': None,
        'night_start_hour': 20,  # 8PM
        'night_end_hour': 6      # 6AM
    })
    
    # Event segment parameters
    event_segment_params = config.get('event_segments', {})
    
    # Maximum event segments to extract (check in multiple places)
    max_event_segments = config.get('max_event_segments')
    if max_event_segments is None:
        max_event_segments = event_segment_params.get('max_segments_per_file')
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize datasets based on what's enabled
    raw_dataset = None
    bandpower_dataset = None
    event_segment_dataset = None
    
    if create_raw_dataset:
        raw_dataset = RawDataset(config)
    
    if create_bandpower_dataset:
        bandpower_dataset = BandpowerDataset(config)
        
    if create_event_segment_dataset:
        event_segment_dataset = EventSegmentDataset(config)
    
    # Find files
    all_file_paths = glob(os.path.join(data_dir, "*.fif"))
    
    if not all_file_paths:
        logger.error(f"No files found in {data_dir}")
        return raw_dataset, bandpower_dataset, event_segment_dataset
    
    # Process files based on the format (standard or subject-based)
    if use_subject_format:
        # Get subject ID from config
        subject_id = config.get('subject_id', 'unknown')
        logger.info(f"Processing files for subject: {subject_id}")
        
        # Filter files by subject ID
        file_paths = []
        for file_path in all_file_paths:
            try:
                file_subject_id, period, id_num = parse_subject_info_from_filename(file_path)
                if file_subject_id == subject_id:
                    file_paths.append(file_path)
            except ValueError as e:
                logger.error(f"Error parsing subject info from {file_path}: {e}")
                continue
        
        logger.info(f"Found {len(file_paths)} files for subject {subject_id}")
        
        if not file_paths:
            logger.error(f"No files found for subject {subject_id}")
            return raw_dataset, bandpower_dataset, event_segment_dataset
        
        # Add subject_id to metadata
        if raw_dataset:
            raw_dataset.metadata['subject_id'] = subject_id
        if bandpower_dataset:
            bandpower_dataset.metadata['subject_id'] = subject_id
        if event_segment_dataset:
            event_segment_dataset.metadata['subject_id'] = subject_id
        
        # Apply max_files limit if specified
        if max_files is not None:
            file_paths = file_paths[:max_files]
            logger.info(f"Limiting to first {max_files} files")
        
        # Get reference time from config or use default
        reference_time = None
        if config.get('reference_time'):
            # Parse reference time from string
            time_str = config.get('reference_time')
            try:
                reference_time = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                logger.info(f"Using specified reference time: {reference_time}")
            except:
                logger.warning(f"Could not parse reference time: {time_str}. Using default.")
        
        # Initialize variables for tracking total counts
        total_raw_samples = 0
        total_bandpower_windows = 0
        total_event_segments = 0
        total_duration_sec = 0
        
        # Process each file for the subject
        for file_idx, file in enumerate(file_paths):
            logger.info(f"Processing file {file_idx+1}/{len(file_paths)}: {os.path.basename(file)}")
            
            try:
                # Load the raw data
                raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
                
                # Log original channel names
                logger.info(f"Original channels in {os.path.basename(file)}: {raw.ch_names}")

                # Use all channels as they are already in the correct bipolar format
                eeg_picks = list(range(len(raw.ch_names)))
                logger.info(f"Using all {len(eeg_picks)} available channels")

                # Pick the selected channels (all channels in this case)
                raw.pick(eeg_picks)
                logger.info(f"Using all {len(raw.ch_names)} available channels: {raw.ch_names}")
                
                # Get original data and sampling frequency for selected channels
                original_data = raw.get_data()  # Channels x Time
                original_sfreq = raw.info['sfreq']
                ch_names = raw.ch_names
                
                # Generate time array for original data (we will use file index for the timestamp)
                n_samples_original = original_data.shape[1]
                
                # Create a dummy timestamp for the file
                file_timestamp = datetime.datetime.now() + datetime.timedelta(seconds=file_idx)
                
                # Generate time array for the data
                time_array_original = np.arange(n_samples_original) / original_sfreq
                
                # Generate day/night labels (dummy values, all day)
                day_night_original = np.ones(n_samples_original)
                
                # Detect events if needed for event detection
                # *** Process entire file for event detection first ***
                logger.info(f"Detecting events on entire file (original data)")
                
                # Get detection parameters 
                low_band = event_params.get('low_band', [80, 250])
                high_band = event_params.get('high_band', [250, 451])
                rel_thresh = event_params.get('rel_thresh', 3.0)
                abs_thresh = event_params.get('abs_thresh', 3.0)
                min_gap = event_params.get('min_gap', 20)
                min_last = event_params.get('min_last', 20)  # in milliseconds
                
                logger.info(f"Using event detection parameters: rel_thresh={rel_thresh}, abs_thresh={abs_thresh}, min_gap={min_gap}ms, min_last={min_last}ms")
                
                # Detect low-band events
                logger.info(f"Detecting low-band events in frequency range: {low_band} Hz")
                
                # Detect events for both datasets
                file_low_band_events = detect_interictal_events(
                    original_data, 
                    original_sfreq,
                    ch_names, 
                    freq_band=low_band,
                    rel_thresh=rel_thresh,
                    abs_thresh=abs_thresh,
                    min_gap=min_gap,
                    min_last=min_last  # keep as milliseconds
                )
                
                # Detect high-band events
                logger.info(f"Detecting high-band events in frequency range: {high_band} Hz")
                file_high_band_events = detect_interictal_events(
                    original_data, 
                    original_sfreq,
                    ch_names, 
                    freq_band=high_band,
                    rel_thresh=rel_thresh,
                    abs_thresh=abs_thresh,
                    min_gap=min_gap,
                    min_last=min_last  # keep as milliseconds
                )
                
                # Combine events: where both bands have events, mark as type 3
                file_combined_events = np.zeros_like(file_low_band_events)
                file_combined_events[file_low_band_events > 0] = 1  # Low band
                file_combined_events[file_high_band_events > 0] = 2  # High band
                
                # Where both bands have events, mark as type 3
                both_bands = (file_low_band_events > 0) & (file_high_band_events > 0)
                file_combined_events[both_bands] = 3  # Both bands
                
                # Find contiguous event regions for debugging
                low_band_regions = find_contiguous_events(file_low_band_events)
                high_band_regions = find_contiguous_events(file_high_band_events)
                combined_regions = find_contiguous_events(file_combined_events)
                
                # Calculate file duration and count distinct events
                file_duration_sec = n_samples_original / original_sfreq
                event_counts = count_distinct_events(file_combined_events)
                total_events = sum(event_counts.values())
                total_duration_sec += file_duration_sec
                
                logger.info(f"File duration: {file_duration_sec/60:.1f} minutes ({file_duration_sec:.1f} seconds)")
                logger.info(f"Found {total_events} distinct events: {event_counts[1]} low-band only, {event_counts[2]} high-band only, {event_counts[3]} dual-band")
                logger.info(f"Found {len(low_band_regions)} low-band regions, {len(high_band_regions)} high-band regions, {len(combined_regions)} combined regions")
                
                # Calculate chunk size in samples and number of chunks
                chunk_samples = int(chunk_size_sec * original_sfreq)
                n_chunks = max(1, n_samples_original // chunk_samples)
                
                # Process data in chunks for raw and bandpower datasets
                if create_raw_dataset or create_bandpower_dataset:
                    for chunk_idx in range(n_chunks):
                        # Calculate chunk boundaries
                        start_sample = chunk_idx * chunk_samples
                        end_sample = min(n_samples_original, (chunk_idx + 1) * chunk_samples)
                        
                        # Skip chunk if it's too small
                        if end_sample - start_sample < chunk_samples / 2:
                            logger.info(f"  Skipping chunk {chunk_idx+1}/{n_chunks} (too small: {end_sample-start_sample} samples)")
                            continue
                        
                        # Calculate chunk duration
                        chunk_duration_sec = (end_sample - start_sample) / original_sfreq
                        
                        logger.info(f"  Processing chunk {chunk_idx+1}/{n_chunks} ({end_sample-start_sample} samples, {chunk_duration_sec:.1f} sec)")
                        
                        # Extract chunk data and time array
                        chunk_data = original_data[:, start_sample:end_sample]
                        chunk_time = time_array_original[start_sample:end_sample]
                        chunk_day_night = day_night_original[start_sample:end_sample]
                        
                        # Extract chunk events from the whole-file event arrays - use combined events
                        chunk_events = np.copy(file_combined_events[start_sample:end_sample])
                        
                        # Debug event counts in chunk
                        chunk_event_regions = find_contiguous_events(chunk_events)
                        chunk_event_counts = count_distinct_events(chunk_events)
                        total_chunk_events = sum(chunk_event_counts.values())
                        
                        logger.info(f"  Chunk contains {total_chunk_events} distinct events: " +
                                    f"{chunk_event_counts[1]} low-band, {chunk_event_counts[2]} high-band, " + 
                                    f"{chunk_event_counts[3]} dual-band, in {len(chunk_event_regions)} regions")
                        
                        # Process bandpower using ORIGINAL chunk data if needed
                        if create_bandpower_dataset:
                            logger.info(f"  Calculating bandpower on chunk (original data)")
                            # Process each band
                            band_data = {}
                            for band_name, band_range in bandpower_params['bands'].items():
                                logger.info(f"  Calculating bandpower for {band_name}: {band_range} Hz")
                                band_power, window_times = extract_band_power(
                                    chunk_data,       # Use original chunk data
                                    original_sfreq,   # Use original sampling rate
                                    band_range,
                                    bandpower_params['window_size_ms'],
                                    bandpower_params['overlap']
                                )
                                
                                # Adjust window times to be relative to reference time
                                if reference_time:
                                    time_offset = (file_timestamp - reference_time).total_seconds() + start_sample / original_sfreq
                                    window_times += time_offset
                                
                                band_data[band_name] = band_power
                            
                            # Get window day/night labels based on time array
                            if band_data:
                                window_day_night = generate_day_night_labels(
                                    window_times,
                                    night_start_hour=bandpower_params.get('night_start_hour', 20),
                                    night_end_hour=bandpower_params.get('night_end_hour', 6),
                                    reference_time=reference_time
                                )
                                
                                # Map events to windows - pass the combined events and chunk start time
                                chunk_start_time_abs = time_array_original[start_sample]
                                
                                # Debug logging to check if events are being detected
                                event_sum = np.sum(chunk_events > 0)
                                logger.info(f"  Debug: chunk_events contains {event_sum} non-zero values")
                                logger.info(f"  Debug: chunk_start_time_abs = {chunk_start_time_abs}")
                                
                                window_events = map_events_to_windows(
                                    chunk_events,
                                    window_times,
                                    original_sfreq,
                                    bandpower_params['window_size_ms'],
                                    chunk_start_time_abs=chunk_start_time_abs
                                )
                                
                                # Debug window event counts
                                window_event_counts = {}
                                for event_type in [1, 2, 3]:
                                    window_event_counts[event_type] = np.sum(window_events == event_type)
                                logger.debug(f"  Bandpower chunk {chunk_idx+1}: Window events sum = {np.sum(window_events > 0)}, Distinct mapped events = {sum(window_event_counts.values())}")
                                
                                logger.info(f"  Mapped {sum(window_event_counts.values())} events to windows: " +
                                            f"{window_event_counts[1]} low-band, {window_event_counts[2]} high-band, " + 
                                            f"{window_event_counts[3]} dual-band")
                                
                                # Add to bandpower dataset
                                bandpower_dataset.add_chunk(
                                    band_data,
                                    window_times,
                                    window_events,
                                    window_day_night,
                                    [file_idx],
                                    [file]
                                )
                                
                                # Update bandpower count
                                total_bandpower_windows += len(window_times)
                        
                        # Process raw data with optional resampling
                        if create_raw_dataset:
                            # Get resampling parameters
                            resample_freq = raw_params.get('resample_freq')
                            
                            # Resample if needed
                            if resample_freq is not None and abs(resample_freq - original_sfreq) > 1e-6:
                                logger.info(f"  Resampling chunk from {original_sfreq} Hz to {resample_freq} Hz")
                                # Create MNE Raw object for this chunk
                                chunk_info = raw.info.copy()
                                chunk_raw = mne.io.RawArray(chunk_data, chunk_info)
                                chunk_raw = chunk_raw.resample(resample_freq)
                                resampled_data = chunk_raw.get_data()
                                resampled_sfreq = resample_freq
                            else:
                                # No resampling needed
                                resampled_data = chunk_data.copy()
                                resampled_sfreq = original_sfreq
                                logger.info(f"  Using original chunk data ({original_sfreq} Hz)")
                            
                            # Generate time array for resampled data
                            n_samples_resampled = resampled_data.shape[1]
                            
                            # Create evenly spaced time points for the resampled data
                            resampled_time = np.linspace(
                                chunk_time[0],
                                chunk_time[-1],
                                n_samples_resampled
                            )
                            
                            # Generate day/night labels for resampled data
                            resampled_day_night = generate_day_night_labels(
                                resampled_time, 
                                night_start_hour=raw_params.get('night_start_hour', 20),
                                night_end_hour=raw_params.get('night_end_hour', 6),
                                reference_time=reference_time
                            )
                            
                            # Map events to resampled data timepoints
                            logger.info("  Mapping events to resampled data timepoints")
                            resampled_events = np.zeros_like(resampled_time)
                            
                            # Create a time mapping from original to resampled
                            orig_times = np.linspace(0, chunk_duration_sec, len(chunk_events))
                            resampled_times_rel = np.linspace(0, chunk_duration_sec, n_samples_resampled)
                            dt_resampled = chunk_duration_sec / n_samples_resampled
                            
                            # Map events to resampled timepoints
                            for k, resampled_time_rel in enumerate(resampled_times_rel):
                                # Define the time interval for this resampled point
                                interval_start = resampled_time_rel - dt_resampled / 2
                                interval_end = resampled_time_rel + dt_resampled / 2
                                
                                # Find original time indices that fall within this interval
                                orig_indices = np.where((orig_times >= interval_start) & (orig_times < interval_end))[0]
                                
                                if orig_indices.size > 0:
                                    # Get events within this interval from the original chunk_events
                                    events_in_interval = chunk_events[orig_indices]
                                    
                                    # Find the highest priority event type in the interval
                                    if np.any(events_in_interval == 3):
                                        resampled_events[k] = 3
                                    elif np.any(events_in_interval == 2):
                                        resampled_events[k] = 2
                                    elif np.any(events_in_interval == 1):
                                        resampled_events[k] = 1
                            
                            # Count resampled events for debugging
                            resampled_event_counts = {
                                1: np.sum(resampled_events == 1),
                                2: np.sum(resampled_events == 2),
                                3: np.sum(resampled_events == 3)
                            }
                            logger.debug(f"  Raw chunk {chunk_idx+1}: Resampled events sum = {np.sum(resampled_events > 0)}, Distinct mapped events = {sum(resampled_event_counts.values())}")
                            
                            logger.info(f"  Mapped {sum(resampled_event_counts.values())} events to resampled data: " +
                                        f"{resampled_event_counts[1]} low-band, {resampled_event_counts[2]} high-band, " + 
                                        f"{resampled_event_counts[3]} dual-band")
                            
                            # Add to raw dataset
                            raw_dataset.add_chunk(
                                resampled_data,
                                resampled_time,
                                resampled_events,
                                resampled_day_night,
                                [file_idx],
                                [file]
                            )
                            
                            # Update raw samples counter
                            total_raw_samples += n_samples_resampled
                
                # Process event segments if needed
                if create_event_segment_dataset:
                    logger.info("Extracting event segments from the file")
                    
                    # Use the extract_event_segments function to create event segments
                    try:
                        file_segment_dataset = extract_event_segments([file], config, max_event_segments)
                        
                        if file_segment_dataset and len(file_segment_dataset) > 0:
                            # Set channel names if available
                            if ch_names:
                                file_segment_dataset.channel_names = ch_names
                            
                            # Merge file dataset into the main dataset
                            if event_segment_dataset is None:
                                event_segment_dataset = file_segment_dataset
                            else:
                                # Copy segments from file dataset to main dataset
                                for i in range(len(file_segment_dataset.data)):
                                    segment_data = file_segment_dataset.data[i]
                                    abs_time = file_segment_dataset.time_arrays[i]
                                    rel_time = file_segment_dataset.rel_time_arrays[i]
                                    event_labels = file_segment_dataset.event_labels[i]
                                    day_night_labels = file_segment_dataset.day_night_labels[i]
                                    source_info = file_segment_dataset.event_sources[i]
                                    
                                    # Add segment to main dataset
                                    event_segment_dataset.add_segment(
                                        segment_data,
                                        abs_time,
                                        rel_time,
                                        event_labels,
                                        day_night_labels,
                                        source_info,
                                        ch_names
                                    )
                                    
                                    total_event_segments += 1
                            
                            # Count events in segments
                            segments_event_counts = {1: 0, 2: 0, 3: 0}
                            for labels in file_segment_dataset.event_labels:
                                segments_event_counts[1] += np.sum(labels == 1)
                                segments_event_counts[2] += np.sum(labels == 2)
                                segments_event_counts[3] += np.sum(labels == 3)
                            
                            logger.info(f"Added {len(file_segment_dataset)} segments from {os.path.basename(file)}")
                            logger.info(f"Segments contain events: {segments_event_counts[1]} low-band, {segments_event_counts[2]} high-band, {segments_event_counts[3]} dual-band")
                        else:
                            logger.info(f"No valid segments were extracted from {os.path.basename(file)}")
                    except Exception as e:
                        logger.error(f"Error extracting events from {file}: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
                import traceback
                traceback.print_exc()
        
        # Log summary statistics
        if raw_dataset:
            raw_event_counts = {1: 0, 2: 0, 3: 0}
            for labels in raw_dataset.event_labels:
                raw_event_counts[1] += np.sum(labels == 1)
                raw_event_counts[2] += np.sum(labels == 2)
                raw_event_counts[3] += np.sum(labels == 3)
            
            logger.info(f"Raw dataset contains {len(raw_dataset)} chunks with {total_raw_samples} samples")
            logger.info(f"Raw dataset events: {raw_event_counts[1]} low-band, {raw_event_counts[2]} high-band, {raw_event_counts[3]} dual-band")
        
        if bandpower_dataset:
            bp_event_counts = {1: 0, 2: 0, 3: 0}
            for labels in bandpower_dataset.event_labels:
                bp_event_counts[1] += np.sum(labels == 1)
                bp_event_counts[2] += np.sum(labels == 2)
                bp_event_counts[3] += np.sum(labels == 3)
            
            logger.info(f"Bandpower dataset contains {len(bandpower_dataset)} chunks with {total_bandpower_windows} windows")
            logger.info(f"Bandpower dataset events: {bp_event_counts[1]} low-band, {bp_event_counts[2]} high-band, {bp_event_counts[3]} dual-band")
        
        if event_segment_dataset:
            segment_event_counts = {1: 0, 2: 0, 3: 0}
            for labels in event_segment_dataset.event_labels:
                segment_event_counts[1] += np.sum(labels == 1)
                segment_event_counts[2] += np.sum(labels == 2)
                segment_event_counts[3] += np.sum(labels == 3)
            
            logger.info(f"Event segment dataset contains {len(event_segment_dataset)} segments with {total_event_segments} total event segments")
            logger.info(f"Event segment dataset events: {segment_event_counts[1]} low-band, {segment_event_counts[2]} high-band, {segment_event_counts[3]} dual-band")
        
        # Save datasets
        # Save raw dataset if it exists
        if raw_dataset is not None and len(raw_dataset) > 0:
            # Save channel_names in metadata
            if hasattr(raw_dataset, 'channel_names') and raw_dataset.channel_names:
                raw_dataset.metadata['channel_names'] = raw_dataset.channel_names
                logger.info(f"Including {len(raw_dataset.channel_names)} channel names in raw dataset")
            
            raw_filepath = os.path.join(save_dir, f"{subject_id}_raw_dataset.npz")
            try:
                raw_dataset.save(raw_filepath)
                logger.info(f"Saved raw dataset for subject {subject_id} with {len(raw_dataset)} chunks")
            except Exception as e:
                logger.error(f"Error saving raw dataset: {e}")
                import traceback
                traceback.print_exc()
        
        # Save bandpower dataset if it exists
        if bandpower_dataset is not None and len(bandpower_dataset) > 0:
            # Save channel_names in metadata
            if hasattr(bandpower_dataset, 'channel_names') and bandpower_dataset.channel_names:
                bandpower_dataset.metadata['channel_names'] = bandpower_dataset.channel_names
                logger.info(f"Including {len(bandpower_dataset.channel_names)} channel names in bandpower dataset")
            
            bp_filepath = os.path.join(save_dir, f"{subject_id}_bandpower_dataset.npz")
            try:
                bandpower_dataset.save(bp_filepath)
                logger.info(f"Saved bandpower dataset for subject {subject_id} with {len(bandpower_dataset)} chunks")
            except Exception as e:
                logger.error(f"Error saving bandpower dataset: {e}")
                import traceback
                traceback.print_exc()
        
        # Save event segment dataset if it exists
        if event_segment_dataset is not None and len(event_segment_dataset) > 0:
            # Save channel_names in metadata
            if hasattr(event_segment_dataset, 'channel_names') and event_segment_dataset.channel_names:
                event_segment_dataset.metadata['channel_names'] = event_segment_dataset.channel_names
                logger.info(f"Including {len(event_segment_dataset.channel_names)} channel names in event segment dataset")
            
            es_filepath = os.path.join(save_dir, f"{subject_id}_event_segments_dataset.npz")
            try:
                event_segment_dataset.save(es_filepath)
                logger.info(f"Saved event segment dataset for subject {subject_id} with {len(event_segment_dataset)} segments")
            except Exception as e:
                logger.error(f"Error saving event segment dataset: {e}")
                import traceback
                traceback.print_exc()

    else:
        # Original time-based file processing
        files_with_timestamps = []
        
        for file_path in all_file_paths:
            try:
                timestamp = parse_timestamp_from_filename(file_path)
                files_with_timestamps.append((file_path, timestamp))
            except ValueError as e:
                logger.error(f"Error parsing timestamp from {file_path}: {e}")
                continue
        
        # Sort by actual timestamp
        files_with_timestamps.sort(key=lambda x: x[1])  # Sort by timestamp
        
        # Extract sorted file paths
        files = [f[0] for f in files_with_timestamps]
        
        logger.info(f"Found and chronologically sorted {len(files)} files to process")
        
        # Apply max_files limit if specified
        if max_files is not None:
            files = files[:max_files]
            logger.info(f"Limiting to first {max_files} files")
        
        # Get reference time from first file if not specified
        reference_time = None
        if config.get('reference_time'):
            # Parse reference time from string
            time_str = config.get('reference_time')
            try:
                reference_time = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                logger.info(f"Using specified reference time: {reference_time}")
            except:
                logger.warning(f"Could not parse reference time: {time_str}. Using default.")
        
        # Initialize variables for tracking total counts
        total_raw_samples = 0
        total_bandpower_windows = 0
        total_event_segments = 0
        total_duration_sec = 0  # Track total processed duration in seconds
        
        # Process each file 
        for file_idx, file in enumerate(files):
            logger.info(f"Processing file {file_idx+1}/{len(files)}: {os.path.basename(file)}")
            
            try:
                # Parse timestamp from filename
                try:
                    timestamp = parse_timestamp_from_filename(file)
                except ValueError as e:
                    logger.error(f"Error parsing timestamp from {file}: {e}")
                    continue
                
                # If reference_time is not set and this is the first file, set it to 7 AM of this day
                if reference_time is None and file_idx == 0:
                    reference_time = datetime.datetime(
                        timestamp.year, timestamp.month, timestamp.day, 7, 0, 0
                    )
                    logger.info(f"Setting reference time to: {reference_time}")
                
                # Load the raw data
                raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
                
                # Log original channel names
                logger.info(f"Original channels in {os.path.basename(file)}: {raw.ch_names}")

                # Use all channels as they are already in the correct bipolar format
                eeg_picks = list(range(len(raw.ch_names)))
                logger.info(f"Using all {len(eeg_picks)} available channels")

                # Pick the selected channels (all channels in this case)
                raw.pick(eeg_picks)
                logger.info(f"Using all {len(raw.ch_names)} available channels: {raw.ch_names}")
                
                # Get original data and sampling frequency for selected channels
                original_data = raw.get_data()  # Channels x Time
                original_sfreq = raw.info['sfreq']
                ch_names = raw.ch_names
                
                # Generate time array for original data (absolute time)
                n_samples_original = original_data.shape[1]
                time_array_original = generate_time_array(
                    timestamp, 
                    n_samples_original, 
                    original_sfreq, 
                    reference_time, 
                    cyclic_24h=False
                )
                
                # Generate day/night labels for original data
                day_night_original = generate_day_night_labels(
                    time_array_original, 
                    night_start_hour=raw_params.get('night_start_hour', 20),
                    night_end_hour=raw_params.get('night_end_hour', 6),
                    reference_time=reference_time
                )
                
                # *** Process entire file for event detection first ***
                # Detect events on the entire original data (before any chunking or resampling)
                logger.info(f"Detecting events on entire file (original data)")
                
                # Get detection parameters 
                low_band = event_params.get('low_band', [80, 250])
                high_band = event_params.get('high_band', [250, 451])
                rel_thresh = event_params.get('rel_thresh', 3.0)
                abs_thresh = event_params.get('abs_thresh', 3.0)
                min_gap = event_params.get('min_gap', 20)
                min_last = event_params.get('min_last', 20)  # in milliseconds
                
                logger.info(f"Using event detection parameters: rel_thresh={rel_thresh}, abs_thresh={abs_thresh}, min_gap={min_gap}ms, min_last={min_last}ms")
                
                # Detect low-band events
                logger.info(f"Detecting low-band events in frequency range: {low_band} Hz")
                
                # Detect events for both datasets
                file_low_band_events = detect_interictal_events(
                    original_data, 
                    original_sfreq,
                    ch_names, 
                    freq_band=low_band,
                    rel_thresh=rel_thresh,
                    abs_thresh=abs_thresh,
                    min_gap=min_gap,
                    min_last=min_last  # keep as milliseconds
                )
                
                # Detect high-band events
                logger.info(f"Detecting high-band events in frequency range: {high_band} Hz")
                file_high_band_events = detect_interictal_events(
                    original_data, 
                    original_sfreq,
                    ch_names, 
                    freq_band=high_band,
                    rel_thresh=rel_thresh,
                    abs_thresh=abs_thresh,
                    min_gap=min_gap,
                    min_last=min_last  # keep as milliseconds
                )
                
                # Combine events: where both bands have events, mark as type 3
                file_combined_events = np.zeros_like(file_low_band_events)
                file_combined_events[file_low_band_events > 0] = 1  # Low band
                file_combined_events[file_high_band_events > 0] = 2  # High band
                
                # Where both bands have events, mark as type 3
                both_bands = (file_low_band_events > 0) & (file_high_band_events > 0)
                file_combined_events[both_bands] = 3  # Both bands
                
                # Find contiguous event regions for debugging
                low_band_regions = find_contiguous_events(file_low_band_events)
                high_band_regions = find_contiguous_events(file_high_band_events)
                combined_regions = find_contiguous_events(file_combined_events)
                
                # Calculate file duration and count distinct events
                file_duration_sec = n_samples_original / original_sfreq
                event_counts = count_distinct_events(file_combined_events)
                total_events = sum(event_counts.values())
                total_duration_sec += file_duration_sec
                
                logger.info(f"File duration: {file_duration_sec/60:.1f} minutes ({file_duration_sec:.1f} seconds)")
                logger.info(f"Found {total_events} distinct events: {event_counts[1]} low-band only, {event_counts[2]} high-band only, {event_counts[3]} dual-band")
                logger.info(f"Found {len(low_band_regions)} low-band regions, {len(high_band_regions)} high-band regions, {len(combined_regions)} combined regions")
                
                # Debug: Log sums of detected event arrays
                logger.debug(f"Event sums: Low={np.sum(file_low_band_events > 0)}, High={np.sum(file_high_band_events > 0)}, Combined={np.sum(file_combined_events > 0)}")
                logger.debug(f"Distinct event counts: Low={event_counts[1]}, High={event_counts[2]}, Both={event_counts[3]}")
                
                # Calculate chunk size in samples and number of chunks
                chunk_samples = int(chunk_size_sec * original_sfreq)
                n_chunks = max(1, n_samples_original // chunk_samples)
                
                # Process data in chunks for raw and bandpower datasets
                if create_raw_dataset or create_bandpower_dataset:
                    for chunk_idx in range(n_chunks):
                        # Calculate chunk boundaries
                        start_sample = chunk_idx * chunk_samples
                        end_sample = min(n_samples_original, (chunk_idx + 1) * chunk_samples)
                        
                        # Skip chunk if it's too small
                        if end_sample - start_sample < chunk_samples / 2:
                            logger.info(f"  Skipping chunk {chunk_idx+1}/{n_chunks} (too small: {end_sample-start_sample} samples)")
                            continue
                        
                        # Calculate chunk duration
                        chunk_duration_sec = (end_sample - start_sample) / original_sfreq
                        
                        logger.info(f"  Processing chunk {chunk_idx+1}/{n_chunks} ({end_sample-start_sample} samples, {chunk_duration_sec:.1f} sec)")
                        
                        # Extract chunk data and time array
                        chunk_data = original_data[:, start_sample:end_sample]
                        chunk_time = time_array_original[start_sample:end_sample]
                        chunk_day_night = day_night_original[start_sample:end_sample]
                        
                        # Extract chunk events from the whole-file event arrays
                        chunk_events = np.copy(file_combined_events[start_sample:end_sample])
                        
                        # Debug event counts in chunk
                        chunk_event_regions = find_contiguous_events(chunk_events)
                        chunk_event_counts = count_distinct_events(chunk_events)
                        total_chunk_events = sum(chunk_event_counts.values())
                        
                        logger.info(f"  Chunk contains {total_chunk_events} distinct events: " +
                                    f"{chunk_event_counts[1]} low-band, {chunk_event_counts[2]} high-band, " + 
                                    f"{chunk_event_counts[3]} dual-band, in {len(chunk_event_regions)} regions")
                        
                        # Process bandpower using ORIGINAL chunk data if needed
                        if create_bandpower_dataset:
                            logger.info(f"  Calculating bandpower on chunk (original data)")
                            # Process each band
                            band_data = {}
                            for band_name, band_range in bandpower_params['bands'].items():
                                logger.info(f"  Calculating bandpower for {band_name}: {band_range} Hz")
                                band_power, window_times = extract_band_power(
                                    chunk_data,       # Use original chunk data
                                    original_sfreq,   # Use original sampling rate
                                    band_range,
                                    bandpower_params['window_size_ms'],
                                    bandpower_params['overlap']
                                )
                                
                                # Adjust window times to be relative to reference time
                                if reference_time:
                                    time_offset = (timestamp - reference_time).total_seconds() + start_sample / original_sfreq
                                    window_times += time_offset
                                
                                band_data[band_name] = band_power
                            
                            # Get window day/night labels based on time array
                            if band_data:
                                window_day_night = generate_day_night_labels(
                                    window_times,
                                    night_start_hour=bandpower_params.get('night_start_hour', 20),
                                    night_end_hour=bandpower_params.get('night_end_hour', 6),
                                    reference_time=reference_time
                                )
                                
                                # Map events to windows - pass the combined events and chunk start time
                                chunk_start_time_abs = time_array_original[start_sample]
                                
                                # Debug logging to check if events are being detected
                                event_sum = np.sum(chunk_events > 0)
                                logger.info(f"  Debug: chunk_events contains {event_sum} non-zero values")
                                logger.info(f"  Debug: chunk_start_time_abs = {chunk_start_time_abs}")
                                
                                window_events = map_events_to_windows(
                                    chunk_events,
                                    window_times,
                                    original_sfreq,
                                    bandpower_params['window_size_ms'],
                                    chunk_start_time_abs=chunk_start_time_abs
                                )
                                
                                # Debug window event counts
                                window_event_counts = {}
                                for event_type in [1, 2, 3]:
                                    window_event_counts[event_type] = np.sum(window_events == event_type)
                                logger.debug(f"  Bandpower chunk {chunk_idx+1}: Window events sum = {np.sum(window_events > 0)}, Distinct mapped events = {sum(window_event_counts.values())}")
                                
                                logger.info(f"  Mapped {sum(window_event_counts.values())} events to windows: " +
                                            f"{window_event_counts[1]} low-band, {window_event_counts[2]} high-band, " + 
                                            f"{window_event_counts[3]} dual-band")
                                
                                # Add to bandpower dataset
                                bandpower_dataset.add_chunk(
                                    band_data,
                                    window_times,
                                    window_events,
                                    window_day_night,
                                    [file_idx],
                                    [file]
                                )
                                
                                # Update bandpower count
                                total_bandpower_windows += len(window_times)
                        
                        # Process raw data with optional resampling
                        if create_raw_dataset:
                            # Get resampling parameters
                            resample_freq = raw_params.get('resample_freq')
                            
                            # Resample if needed
                            if resample_freq is not None and abs(resample_freq - original_sfreq) > 1e-6:
                                logger.info(f"  Resampling chunk from {original_sfreq} Hz to {resample_freq} Hz")
                                # Create MNE Raw object for this chunk
                                chunk_info = raw.info.copy()
                                chunk_raw = mne.io.RawArray(chunk_data, chunk_info)
                                chunk_raw = chunk_raw.resample(resample_freq)
                                resampled_data = chunk_raw.get_data()
                                resampled_sfreq = resample_freq
                            else:
                                # No resampling needed
                                resampled_data = chunk_data.copy()
                                resampled_sfreq = original_sfreq
                                logger.info(f"  Using original chunk data ({original_sfreq} Hz)")
                            
                            # Generate time array for resampled data
                            n_samples_resampled = resampled_data.shape[1]
                            
                            # Create evenly spaced time points for the resampled data
                            resampled_time = np.linspace(
                                chunk_time[0],
                                chunk_time[-1],
                                n_samples_resampled
                            )
                            
                            # Generate day/night labels for resampled data
                            resampled_day_night = generate_day_night_labels(
                                resampled_time, 
                                night_start_hour=raw_params.get('night_start_hour', 20),
                                night_end_hour=raw_params.get('night_end_hour', 6),
                                reference_time=reference_time
                            )
                            
                            # Map events to resampled data timepoints
                            logger.info("  Mapping events to resampled data timepoints")
                            resampled_events = np.zeros_like(resampled_time)
                            
                            # Create a time mapping from original to resampled
                            orig_times = np.linspace(0, chunk_duration_sec, len(chunk_events))
                            resampled_times_rel = np.linspace(0, chunk_duration_sec, n_samples_resampled)
                            dt_resampled = chunk_duration_sec / n_samples_resampled
                            
                            # Map events to resampled timepoints
                            for k, resampled_time_rel in enumerate(resampled_times_rel):
                                # Define the time interval for this resampled point
                                interval_start = resampled_time_rel - dt_resampled / 2
                                interval_end = resampled_time_rel + dt_resampled / 2
                                
                                # Find original time indices that fall within this interval
                                orig_indices = np.where((orig_times >= interval_start) & (orig_times < interval_end))[0]
                                
                                if orig_indices.size > 0:
                                    # Get events within this interval from the original chunk_events
                                    events_in_interval = chunk_events[orig_indices]
                                    
                                    # Find the highest priority event type in the interval
                                    if np.any(events_in_interval == 3):
                                        resampled_events[k] = 3
                                    elif np.any(events_in_interval == 2):
                                        resampled_events[k] = 2
                                    elif np.any(events_in_interval == 1):
                                        resampled_events[k] = 1
                            
                            # Count resampled events for debugging
                            resampled_event_counts = {
                                1: np.sum(resampled_events == 1),
                                2: np.sum(resampled_events == 2),
                                3: np.sum(resampled_events == 3)
                            }
                            logger.debug(f"  Raw chunk {chunk_idx+1}: Resampled events sum = {np.sum(resampled_events > 0)}, Distinct mapped events = {sum(resampled_event_counts.values())}")
                            
                            logger.info(f"  Mapped {sum(resampled_event_counts.values())} events to resampled data: " +
                                        f"{resampled_event_counts[1]} low-band, {resampled_event_counts[2]} high-band, " + 
                                        f"{resampled_event_counts[3]} dual-band")
                            
                            # Add to raw dataset
                            raw_dataset.add_chunk(
                                resampled_data,
                                resampled_time,
                                resampled_events,
                                resampled_day_night,
                                [file_idx],
                                [file]
                            )
                            
                            # Update raw samples counter
                            total_raw_samples += n_samples_resampled
                
                # Process event segments directly from the whole file if enabled
                if create_event_segment_dataset:
                    logger.info("Extracting event segments from the file")
                    
                    # Use the extract_event_segments function to create event segments
                    try:
                        file_segment_dataset = extract_event_segments([file], config, max_event_segments)
                        
                        if file_segment_dataset and len(file_segment_dataset) > 0:
                            # Merge file dataset into the main dataset
                            if event_segment_dataset is None:
                                event_segment_dataset = file_segment_dataset
                            else:
                                # Copy segments from file dataset to main dataset
                                for i in range(len(file_segment_dataset.data)):
                                    segment_data = file_segment_dataset.data[i]
                                    abs_time = file_segment_dataset.time_arrays[i]
                                    rel_time = file_segment_dataset.rel_time_arrays[i]
                                    event_labels = file_segment_dataset.event_labels[i]
                                    day_night_labels = file_segment_dataset.day_night_labels[i]
                                    source_info = file_segment_dataset.event_sources[i]
                                    
                                    # Add segment to main dataset
                                    event_segment_dataset.add_segment(
                                        segment_data,
                                        abs_time,
                                        rel_time,
                                        event_labels,
                                        day_night_labels,
                                        source_info
                                    )
                                    
                                    total_event_segments += 1
                            
                            # Count events in segments
                            segments_event_counts = {1: 0, 2: 0, 3: 0}
                            for labels in file_segment_dataset.event_labels:
                                segments_event_counts[1] += np.sum(labels == 1)
                                segments_event_counts[2] += np.sum(labels == 2)
                                segments_event_counts[3] += np.sum(labels == 3)
                            
                            logger.info(f"Added {len(file_segment_dataset)} segments from {os.path.basename(file)}")
                            logger.info(f"Segments contain events: {segments_event_counts[1]} low-band, {segments_event_counts[2]} high-band, {segments_event_counts[3]} dual-band")
                        else:
                            logger.info(f"No valid segments were extracted from {os.path.basename(file)}")
                    except Exception as e:
                        logger.error(f"Error extracting event segments from {file}: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
                import traceback
                traceback.print_exc()
        
        # Log summary statistics for all datasets
        if raw_dataset:
            raw_event_counts = {1: 0, 2: 0, 3: 0}
            for labels in raw_dataset.event_labels:
                raw_event_counts[1] += np.sum(labels == 1)
                raw_event_counts[2] += np.sum(labels == 2)
                raw_event_counts[3] += np.sum(labels == 3)
            
            logger.info(f"Raw dataset contains {len(raw_dataset)} chunks with {total_raw_samples} samples")
            logger.info(f"Raw dataset events: {raw_event_counts[1]} low-band, {raw_event_counts[2]} high-band, {raw_event_counts[3]} dual-band")
        
        if bandpower_dataset:
            bp_event_counts = {1: 0, 2: 0, 3: 0}
            for labels in bandpower_dataset.event_labels:
                bp_event_counts[1] += np.sum(labels == 1)
                bp_event_counts[2] += np.sum(labels == 2)
                bp_event_counts[3] += np.sum(labels == 3)
            
            logger.info(f"Bandpower dataset contains {len(bandpower_dataset)} chunks with {total_bandpower_windows} windows")
            logger.info(f"Bandpower dataset events: {bp_event_counts[1]} low-band, {bp_event_counts[2]} high-band, {bp_event_counts[3]} dual-band")
        
        if event_segment_dataset:
            segment_event_counts = {1: 0, 2: 0, 3: 0}
            for labels in event_segment_dataset.event_labels:
                segment_event_counts[1] += np.sum(labels == 1)
                segment_event_counts[2] += np.sum(labels == 2)
                segment_event_counts[3] += np.sum(labels == 3)
            
            logger.info(f"Event segment dataset contains {len(event_segment_dataset)} segments with {total_event_segments} total event segments")
            logger.info(f"Event segment dataset events: {segment_event_counts[1]} low-band, {segment_event_counts[2]} high-band, {segment_event_counts[3]} dual-band")
        
        # Log total processing duration
        hours = total_duration_sec / 3600
        minutes = (total_duration_sec % 3600) / 60
        logger.info(f"Total data processed: {hours:.0f}h {minutes:.0f}m ({total_duration_sec:.1f} seconds)")
        
        # Save datasets if requested
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw dataset if it exists
        if raw_dataset is not None and len(raw_dataset) > 0:
            raw_filepath = os.path.join(save_dir, f"{timestamp}_raw_dataset.npz")
            try:
                raw_dataset.save(raw_filepath)
                logger.info(f"Saved raw dataset with {len(raw_dataset)} chunks and {total_raw_samples} samples")
            except Exception as e:
                logger.error(f"Error saving raw dataset: {e}")
                import traceback
                traceback.print_exc()
        
        # Save bandpower dataset if it exists
        if bandpower_dataset is not None and len(bandpower_dataset) > 0:
            bp_filepath = os.path.join(save_dir, f"{timestamp}_bandpower_dataset.npz")
            try:
                bandpower_dataset.save(bp_filepath)
                logger.info(f"Saved bandpower dataset with {len(bandpower_dataset)} chunks and {total_bandpower_windows} windows")
            except Exception as e:
                logger.error(f"Error saving bandpower dataset: {e}")
                import traceback
                traceback.print_exc()
        
        # Save event segment dataset if it exists
        if event_segment_dataset is not None and len(event_segment_dataset) > 0:
            es_filepath = os.path.join(save_dir, f"{timestamp}_event_segments_dataset.npz")
            try:
                event_segment_dataset.save(es_filepath)
                logger.info(f"Saved event segment dataset with {len(event_segment_dataset)} segments")
            except Exception as e:
                logger.error(f"Error saving event segment dataset: {e}")
                import traceback
                traceback.print_exc()
        
        return raw_dataset, bandpower_dataset, event_segment_dataset

def generate_day_night_labels(time_array: np.ndarray,
                             night_start_hour: int = 20,
                             night_end_hour: int = 6,
                             reference_time: Optional[datetime.datetime] = None) -> np.ndarray:
    """
    Generate day/night labels based on time array
    
    Args:
        time_array: Array of time points in seconds from reference time
        night_start_hour: Hour when night starts (24-hour format, default 20 = 8PM)
        night_end_hour: Hour when night ends (24-hour format, default 6 = 6AM)
        reference_time: Reference time for the time array
        
    Returns:
        Array of labels: 0 for night (8PM-6AM), 1 for day
    """
    if reference_time is None:
        # If no reference time provided, cannot determine day/night
        logger.warning("No reference time provided for day/night labeling, using all day")
        return np.ones_like(time_array)
    
    day_night_labels = np.ones_like(time_array)  # Default to day (1)
    
    for i, time_sec in enumerate(time_array):
        # Calculate absolute time
        abs_time = reference_time + datetime.timedelta(seconds=float(time_sec))
        
        # Check if it's night time (8PM-6AM)
        hour = abs_time.hour
        
        # Handle the case where night spans midnight
        if night_start_hour > night_end_hour:
            if hour >= night_start_hour or hour < night_end_hour:
                day_night_labels[i] = 0  # Night
        else:
            if night_start_hour <= hour < night_end_hour:
                day_night_labels[i] = 0  # Night
    
    return day_night_labels

def count_distinct_events(event_array):
    """
    Count distinct events (contiguous regions) in an event array
    
    Args:
        event_array: Array of event labels (0 = no event, 1-3 = event types)
        
    Returns:
        Dictionary with counts for each event type
    """
    event_counts = {1: 0, 2: 0, 3: 0}
    
    # For each event type
    for event_type in [1, 2, 3]:
        # Create a binary mask for this event type
        mask = (event_array == event_type)
        
        if not np.any(mask):
            continue
            
        # Find transitions (0->1 or 1->0)
        transitions = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        
        # Count rising edges (start of events)
        event_counts[event_type] = np.sum(transitions == 1)
    
    return event_counts

if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "dataset_config.yaml"
    
    # Check for optional subject format flag
    use_subject_format = False
    if len(sys.argv) > 2 and sys.argv[2].lower() in ['subject', 'subjects', 'subject_format', 'true']:
        use_subject_format = True
        print(f"Using subject-based file format")
    
    # Construct datasets
    raw_dataset, bandpower_dataset, event_segment_dataset = construct_dataset_from_config(config_path)
    
    if raw_dataset:
        logger.info(f"Created raw dataset with {len(raw_dataset)} chunks")
        if hasattr(raw_dataset, 'channel_names') and raw_dataset.channel_names:
            logger.info(f"Raw dataset has {len(raw_dataset.channel_names)} channels: {raw_dataset.channel_names[:5]}...")
    
    if bandpower_dataset:
        logger.info(f"Created bandpower dataset with {len(bandpower_dataset)} chunks")
        if hasattr(bandpower_dataset, 'channel_names') and bandpower_dataset.channel_names:
            logger.info(f"Bandpower dataset has {len(bandpower_dataset.channel_names)} channels: {bandpower_dataset.channel_names[:5]}...")
        
    if event_segment_dataset:
        logger.info(f"Created event segment dataset with {len(event_segment_dataset)} segments")
        if hasattr(event_segment_dataset, 'channel_names') and event_segment_dataset.channel_names:
            logger.info(f"Event segment dataset has {len(event_segment_dataset.channel_names)} channels: {event_segment_dataset.channel_names[:5]}...")

    # Print usage instructions
    print("\nUsage:")
    print("python dataset_constructor.py [config_path] [subject_format]")
    print("  config_path: Path to YAML configuration file (default: dataset_config.yaml)")
    print("  subject_format: Optional flag to use subject-based file format (default: False)")
    print("\nExample:")
    print("python dataset_constructor.py my_config.yaml subject")
    print("\nFor subject-based format, ensure your config.yaml includes:")
    print("use_subject_format: true  # Enable subject file pattern {subject_id}_{period}{ID}.fif")
        
