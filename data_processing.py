import os
import re
import numpy as np
import mne
import datetime
from glob import glob
from typing import List, Tuple, Optional, Union
import torch
import sys
from scipy.signal import butter, filtfilt

# Add the path to bqk_utils.py to make the functions available
sys.path.append('.')
try:
    from bqk_utils import return_hil_enve_norm, find_high_enveTimes
except ImportError:
    print("Warning: bqk_utils not found. Make sure it's in the correct path.")

def parse_timestamp_from_filename(filename: str) -> datetime.datetime:
    """
    Parse timestamp from filename, handling two possible formats:
    1. 2024_08_20_18_37_21_bipolar.fif or 2024_08_20_18_37_21_average.fif
    2. 24_08_13-D_14_36_48_bipolar.fif or 24_08_13-D_14_36_48_average.fif
    
    Args:
        filename: The filename to parse
        
    Returns:
        datetime object representing the timestamp
    """
    basename = os.path.basename(filename)
    
    # Pattern for format: 2024_08_20_18_37_21_bipolar.fif or 2024_08_20_18_37_21_average.fif
    pattern1 = r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(?:bipolar|average)\.fif"
    
    # Pattern for format: 24_08_13-D_14_36_48_bipolar.fif or 24_08_13-D_14_36_48_average.fif
    pattern2 = r"(\d{2})_(\d{2})_(\d{2})-D_(\d{2})_(\d{2})_(\d{2})_(?:bipolar|average)\.fif"
    
    match1 = re.match(pattern1, basename)
    match2 = re.match(pattern2, basename)
    
    if match1:
        year, month, day, hour, minute, second = map(int, match1.groups())
    elif match2:
        year_short, month, day, hour, minute, second = map(int, match2.groups())
        # Assume 20xx for the year
        year = 2000 + year_short
    else:
        raise ValueError(f"Filename {basename} does not match expected format")
    
    return datetime.datetime(year, month, day, hour, minute, second)

def generate_time_array(
    start_time: datetime.datetime, 
    n_samples: int, 
    sfreq: float, 
    reference_time: Optional[datetime.datetime] = None,
    cyclic_24h: bool = False
) -> np.ndarray:
    """
    Generate a time array in seconds, with reference to a specific time.
    
    Args:
        start_time: The start time of the recording
        n_samples: Number of samples in the data
        sfreq: Sampling frequency in Hz
        reference_time: Reference time. If None, defaults to 7:00 AM of the same day
        cyclic_24h: If True, maps all times to a 24-hour range (0-86400 seconds),
                   ignoring the date. If False, uses absolute time difference.
        
    Returns:
        Array of time points in seconds relative to reference_time
    """
    if reference_time is None:
        # Default reference time is 7:00 AM of the same day
        reference_time = datetime.datetime(
            start_time.year, start_time.month, start_time.day, 7, 0, 0
        )
    
    if cyclic_24h:
        # Only consider time of day, ignoring the date
        # Calculate seconds since midnight for both times
        ref_seconds = reference_time.hour * 3600 + reference_time.minute * 60 + reference_time.second
        start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
        
        # Calculate offset, ensuring it's within 0-86400 range (24 hours)
        start_offset = (start_seconds - ref_seconds) % 86400
        
        # For data spanning multiple days, we need to handle day transitions
        day_diff = (start_time.date() - reference_time.date()).days
        seconds_per_day = 86400  # 24 hours in seconds
        
        # Generate time array with cyclic wrapping at 24 hours
        time_points = np.arange(n_samples) / sfreq
        time_array = (start_offset + time_points) % seconds_per_day
    else:
        # Standard mode: absolute time difference
        start_offset = (start_time - reference_time).total_seconds()
        
        # Generate time array
        time_points = np.arange(n_samples) / sfreq
        time_array = start_offset + time_points
    
    return time_array

def detect_interictal_events(
    data: np.ndarray, 
    fs: float,
    ch_names: List[str] = None,
    freq_band: List[float] = [250, 451],
    rel_thresh: float = 3.0,
    abs_thresh: float = 3.0,
    min_gap: float = 20,
    min_last: float = 20,
    start_time: float = 0
) -> np.ndarray:
    """
    Detect inter-ictal events in EEG data and create binary labels.
    
    Args:
        data: Input data in shape (channels, time)
        fs: Sampling frequency
        ch_names: Channel names (for logging purposes)
        freq_band: Frequency band for filtering [low, high]
        rel_thresh: Relative threshold compared to median
        abs_thresh: Absolute threshold compared to global median
        min_gap: Minimum gap between events in ms
        min_last: Minimum duration of events in ms
        start_time: Start time of the data segment
        
    Returns:
        Binary labels indicating event presence (1) or absence (0) at each time point
    """
    if 'return_hil_enve_norm' not in globals():
        raise ImportError("bqk_utils module not found. Cannot detect events.")
    
    # Calculate the envelope of the signal in the specified frequency band
    # This helps to detect high amplitude events like IEDs
    raw_envelope = return_hil_enve_norm(data, fs, freq_band)
    
    # Detect IED events using the envelope
    ied_events = find_high_enveTimes(
        raw_envelope, 
        len(data),  # number of channels
        fs, 
        rel_thresh=rel_thresh,
        abs_thresh=abs_thresh, 
        min_gap=min_gap,
        min_last=min_last,
        start_time=start_time
    )
    
    # Count detected IED events per channel
    ied_counts = [len(events) for events in ied_events]
    if ch_names:
        print("IED events detected per channel:")
        for i, count in enumerate(ied_counts):
            if i < len(ch_names):
                print(f"{ch_names[i]}: {count} events")
            else:
                print(f"Channel {i}: {count} events")
    
    # Convert time ranges to binary labels
    # Create a time array for the whole data
    times = np.arange(data.shape[1]) / fs + start_time
    
    # Initialize binary labels
    labels = np.zeros(len(times))
    
    # For each channel's events, mark the corresponding time points in the labels
    for channel_events in ied_events:
        for start, end in channel_events:
            # Find indices where times fall within this event
            event_indices = np.where((times >= start) & (times <= end))[0]
            labels[event_indices] = 1
    
    print(f"Found {int(np.sum(labels))} time points with events out of {len(labels)} total points")
    
    return labels

def load_merged_data(
    data_dir: str, 
    max_samples: int,
    batch_size: int = -1,
    max_files: int = None,
    resample_freq: Optional[float] = None,
    reference_time: Optional[datetime.datetime] = None,
    file_pattern: str = "*.fif",
    start_file_idx: int = 0,
    custom_labels: Optional[np.ndarray] = None,
    save_data: bool = False,
    save_dir: str = "preprocessed",
    label_channels: Optional[List[str]] = None,
    label_type: str = "time",
    label_params: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load and merge EEG data into batches of fixed size.
    
    Args:
        data_dir: Directory containing the EEG files
        max_samples: Number of samples per batch
        batch_size: Number of batches to create. If -1, creates as many as possible
        max_files: Maximum number of files to read (optional). If None, reads all files.
        resample_freq: Frequency to resample the data to (optional)
        reference_time: Reference time for time array (default: 7:00 AM of the first file's day)
        file_pattern: Pattern to match files
        start_file_idx: Index of the file to start loading from
        custom_labels: Optional custom labels to use instead of time arrays
        save_data: Whether to save the preprocessed data
        save_dir: Directory to save the preprocessed data
        label_channels: Optional list of channel names to use as labels
        label_type: Type of labels to use - options: "time", "event", "custom" 
        label_params: Parameters for label generation (e.g., event detection parameters)
        
    Returns:
        Tuple containing:
            - Processed data (n_batch, Time, Channels)
            - Labels (n_batch, Time, Label_Channels) - based on specified label_type
            - File indices used in this batch
            - Next file index to start from
    """
    files = sorted(glob(os.path.join(data_dir, file_pattern)))
    
    if not files:
        raise ValueError(f"No files found in {data_dir} matching pattern {file_pattern}")
    
    if start_file_idx >= len(files):
        raise ValueError(f"Start file index {start_file_idx} exceeds available files ({len(files)})")
    
    # Apply max_files limit if specified
    if max_files is not None:
        available_files = files[start_file_idx:start_file_idx + max_files]
    else:
        available_files = files[start_file_idx:]
    
    print(f"Found {len(available_files)} available files to process")
    
    # Initialize data collection for batches
    batch_data_list = []
    batch_times_list = []
    batch_events_list = []
    used_file_indices = []
    
    # Track start and end timestamps for all processed files
    start_timestamp = None
    end_timestamp = None
    
    # Start from the provided file index
    file_idx = start_file_idx
    current_data = None
    current_times = None
    current_events = None
    batch_idx = 0
    
    # Default event detection parameters
    default_event_params = {
        'freq_band': [250, 451],
        'rel_thresh': 3.0,
        'abs_thresh': 3.0,
        'min_gap': 20,
        'min_last': 20
    }
    
    # Update with user-provided parameters
    if label_params and label_type == "event":
        default_event_params.update(label_params)
    
    # Process batches until we reach batch_size or run out of files
    while (batch_size == -1 or batch_idx < batch_size) and file_idx < start_file_idx + len(available_files):
        print(f"Processing batch {batch_idx+1}")
        current_samples = 0
        batch_file_indices = []
        
        # Reset data for new batch
        current_data = None
        current_times = None
        current_events = None
        
        # Keep adding files until we reach max_samples
        while current_samples < max_samples and file_idx < start_file_idx + len(available_files):
            file = available_files[file_idx - start_file_idx]
            print(f"  Loading file: {os.path.basename(file)}")
            
            # Parse the timestamp from the filename
            timestamp = parse_timestamp_from_filename(file)
            
            # Update start and end timestamps
            if start_timestamp is None or timestamp < start_timestamp:
                start_timestamp = timestamp
            if end_timestamp is None or timestamp > end_timestamp:
                end_timestamp = timestamp
            
            # If reference_time is not set and this is the first file, set it to 7 AM of this day
            if reference_time is None and batch_idx == 0 and current_data is None:
                reference_time = datetime.datetime(
                    timestamp.year, timestamp.month, timestamp.day, 7, 0, 0
                )
                print(f"Setting reference time to: {reference_time}")
            
            # Load the raw data
            raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
            
            # Resample if needed
            if resample_freq is not None:
                raw = raw.resample(resample_freq)
            
            # Get data and convert to numpy array
            file_data = raw.get_data()  # Channels x Time
            
            # Generate time array for this file
            n_samples = file_data.shape[1]
            sfreq = raw.info['sfreq']
            time_array = generate_time_array(timestamp, n_samples, sfreq, reference_time, cyclic_24h=False)
            
            # Detect events if requested
            if label_type == "event":
                try:
                    # For event detection, we'll use the raw data
                    params = default_event_params.copy()
                    params['start_time'] = (timestamp - reference_time).total_seconds() if reference_time else 0
                    event_array = detect_interictal_events(
                        file_data, 
                        sfreq,
                        raw.ch_names, 
                        **params
                    )
                except Exception as e:
                    print(f"Error detecting events: {e}")
                    print("Using zeros as events")
                    event_array = np.zeros(n_samples)
            
            if current_data is None:
                current_data = file_data
                current_times = time_array
                if label_type == "event":
                    current_events = event_array
            else:
                # Concatenate along time dimension (axis=1)
                current_data = np.concatenate([current_data, file_data], axis=1)
                current_times = np.concatenate([current_times, time_array])
                if label_type == "event":
                    current_events = np.concatenate([current_events, event_array])
            
            batch_file_indices.append(file_idx)
            file_idx += 1
            
            # Update current_samples
            current_samples = current_data.shape[1]
            
            # If we've reached max_samples, or even exceeded it, break
            if current_samples >= max_samples:
                break
                
            # If we've processed all available files, break
            if file_idx >= start_file_idx + len(available_files):
                print(f"  Reached end of files at batch {batch_idx+1}")
                break
        
        # If we didn't get any data for this batch, break
        if current_data is None:
            print(f"No more data available for batch {batch_idx+1}")
            break
            
        # If we didn't get enough data for a complete batch, pad with zeros
        if current_samples < max_samples:
            print(f"  Warning: Batch {batch_idx+1} is incomplete with {current_samples}/{max_samples} samples")
            pad_samples = max_samples - current_samples
            current_data = np.pad(current_data, ((0, 0), (0, pad_samples)), mode='constant')
            
            # Pad times by continuing the time sequence
            if current_times is not None and len(current_times) > 0:
                last_time = current_times[-1]
                time_step = 1.0 / sfreq if len(current_times) > 1 else 1.0 
                pad_times = last_time + np.arange(1, pad_samples + 1) * time_step
                current_times = np.concatenate([current_times, pad_times])
            
            # Pad events with zeros
            if label_type == "event" and current_events is not None:
                current_events = np.pad(current_events, (0, pad_samples), mode='constant')
        
        # If we have more than max_samples, trim
        if current_samples > max_samples:
            current_data = current_data[:, :max_samples]
            current_times = current_times[:max_samples]
            if label_type == "event":
                current_events = current_events[:max_samples]
        
        # Transpose to Time x Channels
        current_data = current_data.T
        
        # No normalization as requested
        
        # Add to batch lists
        batch_data_list.append(current_data)
        batch_times_list.append(current_times)
        if label_type == "event":
            batch_events_list.append(current_events)
        used_file_indices.extend(batch_file_indices)
        
        batch_idx += 1
        
        # If we've processed all files, break
        if file_idx >= start_file_idx + len(available_files):
            break
    
    # Convert lists to arrays
    batched_data = np.array(batch_data_list)
    
    # Process time arrays or events into 3D labels based on label_type
    if label_type == "event":
        # Use detected events as labels
        batched_labels = create_3d_labels_from_events(batch_events_list, batched_data.shape)
        print(f"Created event-based labels with shape: {batched_labels.shape}")
    elif label_type == "custom" and custom_labels is not None:
        # Use custom labels if provided
        if isinstance(custom_labels, np.ndarray):
            # Try to reshape custom labels to match batched data
            if len(custom_labels.shape) == 3:
                # Already in 3D format (n_batch, Time, Label_Channels)
                batched_labels = custom_labels
            elif len(custom_labels.shape) == 2:
                # 2D format, add channel dimension
                if custom_labels.shape[0] == batched_data.shape[0]:
                    # Matches batch dimension
                    batched_labels = custom_labels[:, :, np.newaxis]
                else:
                    # Try to reshape
                    batched_labels = np.tile(custom_labels[np.newaxis, :, :], (batched_data.shape[0], 1, 1))
            else:
                print(f"Warning: Custom labels not in expected format, using {label_type} arrays instead")
                if label_type == "event":
                    batched_labels = np.zeros((batched_data.shape[0], batched_data.shape[1], 1))
                else:  # Default to time
                    batched_labels = create_3d_labels(batch_times_list, batched_data.shape)
        else:
            print(f"Warning: Custom labels not provided or not in expected format, using {label_type} arrays instead")
            if label_type == "event":
                batched_labels = np.zeros((batched_data.shape[0], batched_data.shape[1], 1))
            else:  # Default to time
                batched_labels = create_3d_labels(batch_times_list, batched_data.shape)
    else:
        # Default: Create 3D labels from time arrays
        batched_labels = create_3d_labels(batch_times_list, batched_data.shape)
        print(f"Created time-based labels with shape: {batched_labels.shape}")
    
    print(f"Loaded {len(batch_data_list)} batches with shape: {batched_data.shape}")
    print(f"Created labels with shape: {batched_labels.shape}")
    
    # Save the preprocessed data if requested
    if save_data and start_timestamp is not None and end_timestamp is not None:
        save_preprocessed_data(batched_data, batched_labels, start_timestamp, end_timestamp, save_dir)
    
    return batched_data, batched_labels, np.array(used_file_indices), file_idx

def create_3d_labels(time_arrays, data_shape, label_channels=1):
    """
    Create 3D labels from time arrays.
    
    Args:
        time_arrays: List of time arrays
        data_shape: Shape of the batched data (n_batch, Time, Channels)
        label_channels: Number of label channels to create
        
    Returns:
        3D labels with shape (n_batch, Time, label_channels)
    """
    n_batch = len(time_arrays)
    if n_batch == 0:
        return np.array([])
    
    # Initialize 3D labels array
    labels_3d = np.zeros((n_batch, data_shape[1], label_channels))
    
    # Fill with time values
    for i, time_array in enumerate(time_arrays):
        # Ensure time_array matches the expected length
        if len(time_array) == data_shape[1]:
            labels_3d[i, :, 0] = time_array
        else:
            # Pad or truncate if needed
            if len(time_array) < data_shape[1]:
                padded = np.pad(time_array, (0, data_shape[1] - len(time_array)), mode='constant')
                labels_3d[i, :, 0] = padded
            else:
                labels_3d[i, :, 0] = time_array[:data_shape[1]]
    
    return labels_3d

def create_3d_labels_from_events(event_arrays, data_shape):
    """
    Create 3D labels from event arrays.
    
    Args:
        event_arrays: List of event arrays
        data_shape: Shape of the batched data (n_batch, Time, Channels)
        
    Returns:
        3D labels with shape (n_batch, Time, 1)
    """
    n_batch = len(event_arrays)
    if n_batch == 0:
        return np.array([])
    
    # Initialize 3D labels array
    labels_3d = np.zeros((n_batch, data_shape[1], 1))
    
    # Fill with event values
    for i, event_array in enumerate(event_arrays):
        # Ensure event_array matches the expected length
        if len(event_array) == data_shape[1]:
            labels_3d[i, :, 0] = event_array
        else:
            # Pad or truncate if needed
            if len(event_array) < data_shape[1]:
                padded = np.pad(event_array, (0, data_shape[1] - len(event_array)), mode='constant')
                labels_3d[i, :, 0] = padded
            else:
                labels_3d[i, :, 0] = event_array[:data_shape[1]]
    
    # Count events in each batch
    for i in range(n_batch):
        event_count = np.sum(labels_3d[i, :, 0] > 0)
        print(f"  Batch {i+1}: {event_count} time points with events ({event_count/data_shape[1]*100:.2f}%)")
    
    return labels_3d

def save_preprocessed_data(
    data: np.ndarray, 
    labels: np.ndarray, 
    start_timestamp: datetime.datetime, 
    end_timestamp: datetime.datetime, 
    save_dir: str = "preprocessed"
) -> str:
    """
    Save preprocessed data with timestamp-based filename.
    
    Args:
        data: Processed data to save
        labels: Labels to save
        start_timestamp: Timestamp of the first file
        end_timestamp: Timestamp of the last file
        save_dir: Directory to save the data
        
    Returns:
        Path to the saved file
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Format timestamps for filename
    start_str = start_timestamp.strftime("%Y%m%d_%H%M%S")
    end_str = end_timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{start_str}_{end_str}_PreMARBLE_dataset.npz"
    filepath = os.path.join(save_dir, filename)
    
    # Save data and labels
    np.savez(filepath, data=data, labels=labels)
    
    print(f"Saved preprocessed data to {filepath}")
    return filepath

def prepare_MARBLE_data(
    data: np.ndarray, 
    labels: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Prepare data for MARBLE training by creating position and vector lists.
    MARBLE expects data as lists of arrays, not as batched arrays.
    
    Args:
        data: Input data (n_batch, Time, Channels) or (Time, Channels)
        labels: Labels (n_batch, Time, Label_Channels) or (n_batch, Time) or (Time)
        
    Returns:
        Tuple containing:
            - List of position arrays, each with shape (Time-1, Channels)
            - List of vector arrays, each with shape (Time-1, Channels)
            - List of label arrays, each with shape (Time-1, Label_Channels) or (Time-1)
    """
    pos_list_batched = []
    x_list_batched = []
    labels_batched = []
    
    # Handle both 3D and 2D inputs
    if len(data.shape) == 3:  # (n_batch, Time, Channels)
        # Process each batch separately and create lists
        for batch_idx in range(data.shape[0]):
            # Get data for this batch
            batch_data = data[batch_idx]  # (Time, Channels)
            
            # Create position and vector for this batch
            pos = batch_data[:-1, :]  # (Time-1, Channels)
            x = np.diff(batch_data, axis=0)  # (Time-1, Channels)
            
            # Process labels based on their shape
            if len(labels.shape) == 3:  # (n_batch, Time, Label_Channels)
                # Get labels for this batch and trim last time point
                batch_labels = labels[batch_idx, :-1, :]  # (Time-1, Label_Channels)
            elif len(labels.shape) == 2:  # (n_batch, Time) or (Time, Label_Channels)
                if labels.shape[0] == data.shape[0]:
                    # (n_batch, Time) - get labels for this batch
                    batch_labels = labels[batch_idx, :-1]  # (Time-1,)
                    # Add channel dimension if needed for consistency
                    if batch_labels.ndim == 1:
                        batch_labels = batch_labels[:, np.newaxis]  # (Time-1, 1)
                else:
                    # (Time, Label_Channels) or mismatched shape
                    print(f"Warning: Labels shape {labels.shape} doesn't match data shape {data.shape}")
                    # Use the same labels for all batches
                    batch_labels = labels[:-1]
                    if batch_labels.ndim == 1:
                        batch_labels = batch_labels[:, np.newaxis]
            else:  # (Time,)
                # Use the same labels for all batches
                batch_labels = labels[:-1]
                # Add channel dimension if needed
                if batch_labels.ndim == 1:
                    batch_labels = batch_labels[:, np.newaxis]  # (Time-1, 1)
            
            # Add to lists
            pos_list_batched.append(pos)
            x_list_batched.append(x)
            labels_batched.append(batch_labels)
    else:  # (Time, Channels) - single batch
        # Create position and vector
        pos = data[:-1, :]  # (Time-1, Channels)
        x = np.diff(data, axis=0)  # (Time-1, Channels)
        
        # Process labels
        if len(labels.shape) == 3:  # (n_batch, Time, Label_Channels)
            # Use first batch
            batch_labels = labels[0, :-1, :]  # (Time-1, Label_Channels)
        elif len(labels.shape) == 2:  # (Time, Label_Channels) or (n_batch, Time)
            batch_labels = labels[:-1]  # Take first dimension
            if batch_labels.ndim == 1:
                batch_labels = batch_labels[:, np.newaxis]
        else:  # (Time,)
            batch_labels = labels[:-1]
            if batch_labels.ndim == 1:
                batch_labels = batch_labels[:, np.newaxis]
        
        # Add to lists
        pos_list_batched.append(pos)
        x_list_batched.append(x)
        labels_batched.append(batch_labels)
    
    # Print shapes for debugging
    print(f"Created {len(pos_list_batched)} batches of data")
    if len(pos_list_batched) > 0:
        print(f"Position shape: {pos_list_batched[0].shape}")
        print(f"Vector shape: {x_list_batched[0].shape}")
        print(f"Labels shape: {labels_batched[0].shape}")
    
    return pos_list_batched, x_list_batched, labels_batched

def load_preprocessed_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed data from a saved .npz file.
    
    Args:
        filepath: Path to the saved .npz file
        
    Returns:
        Tuple containing:
            - Processed data (n_batch, Time, Channels)
            - Labels (n_batch, Time)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessed data file not found: {filepath}")
    
    # Load the compressed file
    data_dict = np.load(filepath, allow_pickle=True)
    
    # Extract data and labels
    data = data_dict['data']
    labels = data_dict['labels']
    
    print(f"Loaded preprocessed data with shape: {data.shape}")
    if len(labels.shape) > 0:
        print(f"Loaded labels with shape: {labels.shape}")
    
    # Check if this is a bandpower dataset by looking for bandpower metadata
    is_bandpower = False
    metadata = {}
    for key in ['band_range', 'window_size_ms', 'overlap']:
        if key in data_dict:
            is_bandpower = True
            metadata[key] = data_dict[key]
    
    if is_bandpower:
        print("Detected bandpower dataset with parameters:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    return data, labels

def find_preprocessed_datasets(
    base_dir: str = "preprocessed", 
    pattern: str = "*PreMARBLE_dataset.npz"
) -> List[str]:
    """
    Find all preprocessed datasets in the specified directory.
    
    Args:
        base_dir: Directory to search in
        pattern: File pattern to match
        
    Returns:
        List of file paths for preprocessed datasets
    """
    if not os.path.exists(base_dir):
        print(f"Directory does not exist: {base_dir}")
        return []
    
    # Find all matching files
    file_paths = sorted(glob(os.path.join(base_dir, pattern)))
    
    if not file_paths:
        print(f"No preprocessed datasets found in {base_dir} matching pattern {pattern}")
    else:
        print(f"Found {len(file_paths)} preprocessed datasets:")
        for i, path in enumerate(file_paths):
            # Extract timestamp information from filename
            basename = os.path.basename(path)
            print(f"  {i+1}. {basename}")
    
    return file_paths

def load_bandpower_data(
    data_dir: str,
    max_samples: int,
    batch_size: int = -1,
    max_files: int = None,
    start_file_idx: int = 0,
    file_pattern: str = "*.fif",
    reference_time: Optional[datetime.datetime] = None,
    label_type: str = "event",
    save_data: bool = False,
    save_dir: str = "temp_Data",
    band_range: List[float] = [80, 250],
    window_size_ms: float = 20,
    overlap: float = 0.5,
    label_params: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load and extract band power from EEG data using sliding windows.
    
    Args:
        data_dir: Directory containing the EEG files
        max_samples: Number of samples per batch for the band power features
        batch_size: Number of batches to create. If -1, creates as many as possible
        max_files: Maximum number of files to read (optional). If None, reads all files.
        start_file_idx: Index of the file to start loading from
        file_pattern: Pattern to match files
        reference_time: Reference time for time array (default: 7:00 AM of the first file's day)
        label_type: Type of labels to use - options: "time", "event", "custom"
        save_data: Whether to save the preprocessed data
        save_dir: Directory to save the preprocessed data
        band_range: Frequency band for extraction [low, high]
        window_size_ms: Size of sliding window in milliseconds
        overlap: Overlap ratio between windows (0 to 1)
        label_params: Parameters for label generation (e.g., event detection parameters)
        
    Returns:
        Tuple containing:
            - Band power data (n_batch, n_windows, n_channels)
            - Labels (n_batch, n_windows)
            - File indices used in this batch
            - Next file index to start from
    """
    files = sorted(glob(os.path.join(data_dir, file_pattern)))
    
    if not files:
        raise ValueError(f"No files found in {data_dir} matching pattern {file_pattern}")
    
    if start_file_idx >= len(files):
        raise ValueError(f"Start file index {start_file_idx} exceeds available files ({len(files)})")
    
    # Apply max_files limit if specified
    if max_files is not None:
        available_files = files[start_file_idx:start_file_idx + max_files]
    else:
        available_files = files[start_file_idx:]
    
    print(f"Found {len(available_files)} available files to process")
    
    # Initialize data collection for batches
    batch_data_list = []
    batch_times_list = []
    batch_events_list = []
    used_file_indices = []
    
    # Track start and end timestamps for all processed files
    start_timestamp = None
    end_timestamp = None
    
    # Start from the provided file index
    file_idx = start_file_idx
    batch_idx = 0
    
    # Default event detection parameters
    default_event_params = {
        'freq_band': [250, 451],
        'rel_thresh': 3.0,
        'abs_thresh': 3.0,
        'min_gap': 20,
        'min_last': 20
    }
    
    # Update with user-provided parameters
    if label_params and label_type == "event":
        default_event_params.update(label_params)
    
    def extract_band_power(data, sfreq, band_range, window_size_ms, overlap):
        """Helper function to extract band power using sliding windows"""
        n_channels = data.shape[0]
        
        # Calculate window size in samples
        window_samples = int((window_size_ms / 1000) * sfreq)
        
        # Check if window is large enough for filtering
        # The minimum window size for filtfilt is typically 3*(filter_order*2 + 1)
        # Using a 4th order filter, this requires at least 3*(4*2+1) = 27 samples
        min_window_size = 30  # Slightly larger than the minimum to be safe
        
        if window_samples < min_window_size:
            print(f"Warning: Window size too small for filtering ({window_samples} samples).")
            print(f"Increasing window size from {window_size_ms}ms to {min_window_size/sfreq*1000:.1f}ms")
            window_samples = min_window_size
            window_size_ms = (window_samples / sfreq) * 1000
        
        # Calculate step size based on overlap
        step_size = max(1, int(window_samples * (1 - overlap)))
        
        # Calculate number of windows
        n_samples = data.shape[1]
        n_windows = (n_samples - window_samples) // step_size + 1
        
        # Design bandpass filter
        nyquist = sfreq / 2
        filter_order = 2  # Lower the filter order for better stability with short windows
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
    
    # Process batches until we reach batch_size or run out of files
    while (batch_size == -1 or batch_idx < batch_size) and file_idx < start_file_idx + len(available_files):
        print(f"Processing batch {batch_idx+1}")
        current_windows = 0
        batch_file_indices = []
        
        # Reset data for new batch
        current_data = None
        current_times = None
        current_events = None
        
        # Keep adding files until we reach max_samples (windows)
        while current_windows < max_samples and file_idx < start_file_idx + len(available_files):
            file = available_files[file_idx - start_file_idx]
            print(f"  Loading file: {os.path.basename(file)}")
            
            # Parse the timestamp from the filename
            timestamp = parse_timestamp_from_filename(file)
            
            # Update start and end timestamps
            if start_timestamp is None or timestamp < start_timestamp:
                start_timestamp = timestamp
            if end_timestamp is None or timestamp > end_timestamp:
                end_timestamp = timestamp
            
            # If reference_time is not set and this is the first file, set it to 7 AM of this day
            if reference_time is None and batch_idx == 0 and current_data is None:
                reference_time = datetime.datetime(
                    timestamp.year, timestamp.month, timestamp.day, 7, 0, 0
                )
                print(f"Setting reference time to: {reference_time}")
            
            # Load the raw data
            raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
            
            # Get data and convert to numpy array
            file_data = raw.get_data()  # Channels x Time
            
            # Get sampling frequency
            sfreq = raw.info['sfreq']
            
            # Extract band power using sliding windows
            band_power, window_times = extract_band_power(
                file_data, sfreq, band_range, window_size_ms, overlap
            )
            
            # Adjust window times to be relative to reference time
            if reference_time:
                # Calculate offset in seconds
                time_offset = (timestamp - reference_time).total_seconds()
                window_times += time_offset
            
            # Detect events if requested
            if label_type == "event":
                try:
                    # For event detection, we'll use the raw data
                    params = default_event_params.copy()
                    params['start_time'] = (timestamp - reference_time).total_seconds() if reference_time else 0
                    event_array = detect_interictal_events(
                        file_data, 
                        sfreq,
                        raw.ch_names, 
                        **params
                    )
                    
                    # Map event array to windows
                    window_events = np.zeros(len(window_times))
                    for i, window_time in enumerate(window_times):
                        # Calculate the sample index in the original data
                        central_sample_idx = int((window_time - time_offset) * sfreq)
                        
                        # Use a small window around the central point to check for events
                        half_window = int(window_size_ms * sfreq / 2000)  # Half window in samples
                        start_sample = max(0, central_sample_idx - half_window)
                        end_sample = min(len(event_array), central_sample_idx + half_window)
                        
                        # If any sample in the window has an event, mark the window
                        if np.any(event_array[start_sample:end_sample] > 0):
                            window_events[i] = 1
                    
                except Exception as e:
                    print(f"Error detecting events: {e}")
                    print("Using zeros as events")
                    window_events = np.zeros(len(window_times))
            
            if current_data is None:
                current_data = band_power
                current_times = window_times
                if label_type == "event":
                    current_events = window_events
            else:
                # Concatenate along the window dimension (axis=0)
                current_data = np.concatenate([current_data, band_power], axis=0)
                current_times = np.concatenate([current_times, window_times])
                if label_type == "event":
                    current_events = np.concatenate([current_events, window_events])
            
            batch_file_indices.append(file_idx)
            file_idx += 1
            
            # Update current_windows
            current_windows = current_data.shape[0]
            
            # If we've reached max_samples, or even exceeded it, break
            if current_windows >= max_samples:
                break
                
            # If we've processed all available files, break
            if file_idx >= start_file_idx + len(available_files):
                print(f"  Reached end of files at batch {batch_idx+1}")
                break
        
        # If we didn't get any data for this batch, break
        if current_data is None:
            print(f"No more data available for batch {batch_idx+1}")
            break
            
        # If we didn't get enough data for a complete batch, pad with zeros
        if current_windows < max_samples:
            print(f"  Warning: Batch {batch_idx+1} is incomplete with {current_windows}/{max_samples} windows")
            pad_windows = max_samples - current_windows
            current_data = np.pad(current_data, ((0, pad_windows), (0, 0)), mode='constant')
            
            # Pad times by continuing the time sequence
            if current_times is not None and len(current_times) > 0:
                last_time = current_times[-1]
                # Estimate time step from data
                time_step = (window_size_ms / 1000) * (1 - overlap) if len(current_times) > 1 else 0.01
                pad_times = last_time + np.arange(1, pad_windows + 1) * time_step
                current_times = np.concatenate([current_times, pad_times])
            
            # Pad events with zeros
            if label_type == "event" and current_events is not None:
                current_events = np.pad(current_events, (0, pad_windows), mode='constant')
        
        # If we have more than max_samples windows, trim
        if current_windows > max_samples:
            current_data = current_data[:max_samples, :]
            current_times = current_times[:max_samples]
            if label_type == "event":
                current_events = current_events[:max_samples]
        
        # Add to batch lists
        batch_data_list.append(current_data)
        batch_times_list.append(current_times)
        if label_type == "event":
            batch_events_list.append(current_events)
        used_file_indices.extend(batch_file_indices)
        
        batch_idx += 1
        
        # If we've processed all files, break
        if file_idx >= start_file_idx + len(available_files):
            break
    
    # Convert lists to arrays
    batched_data = np.array(batch_data_list)
    
    # Process time arrays or events into labels based on label_type
    if label_type == "event":
        batched_labels = np.array(batch_events_list)
        # Add channel dimension if needed
        if len(batched_labels.shape) == 2:
            batched_labels = batched_labels[:, :, np.newaxis]
        print(f"Created event-based labels with shape: {batched_labels.shape}")
    else:
        # Use time arrays as labels
        batched_labels = np.array([times[:, np.newaxis] for times in batch_times_list])
        print(f"Created time-based labels with shape: {batched_labels.shape}")
    
    print(f"Loaded {len(batch_data_list)} batches with shape: {batched_data.shape}")
    
    # Save the preprocessed data if requested
    if save_data and start_timestamp is not None and end_timestamp is not None:
        # Create filename indicating this is bandpower data
        start_str = start_timestamp.strftime("%Y%m%d_%H%M%S")
        end_str = end_timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{start_str}_{end_str}_bandpower_{band_range[0]}-{band_range[1]}Hz_PreMARBLE_dataset.npz"
        filepath = os.path.join(save_dir, filename)
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save data and labels
        np.savez(filepath, data=batched_data, labels=batched_labels, 
                band_range=band_range, window_size_ms=window_size_ms, overlap=overlap)
        
        print(f"Saved bandpower data to {filepath}")
    
    return batched_data, batched_labels, np.array(used_file_indices), file_idx