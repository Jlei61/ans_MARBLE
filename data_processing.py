import os
import re
import numpy as np
import mne
import datetime
from glob import glob
from typing import List, Tuple, Optional, Union
import torch

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

def load_merged_data(
    data_dir: str, 
    max_samples: int, 
    resample_freq: Optional[float] = None,
    reference_time: Optional[datetime.datetime] = None,
    file_pattern: str = "*.fif",
    start_file_idx: int = 0,
    prev_remainder_data: Optional[np.ndarray] = None,
    prev_remainder_times: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load and merge EEG data from multiple files to meet the maximum sample requirement.
    
    Args:
        data_dir: Directory containing the EEG files
        max_samples: Maximum number of samples to retrieve
        resample_freq: Frequency to resample the data to (optional)
        reference_time: Reference time for time array (default: 7:00 AM of the first file's day)
        file_pattern: Pattern to match files
        start_file_idx: Index of the file to start loading from
        prev_remainder_data: Remaining data from previous batch (if any)
        prev_remainder_times: Remaining time points from previous batch (if any)
        
    Returns:
        Tuple containing:
            - Processed data (Time x Channels)
            - Time array
            - File indices used in this batch
            - Next file index to start from
            - Remaining data for next batch
            - Remaining time points for next batch
    """
    files = sorted(glob(os.path.join(data_dir, file_pattern)))
    
    if not files:
        raise ValueError(f"No files found in {data_dir} matching pattern {file_pattern}")
    
    if start_file_idx >= len(files):
        raise ValueError(f"Start file index {start_file_idx} exceeds available files ({len(files)})")
    
    # Initialize data collection
    current_data = prev_remainder_data
    current_times = prev_remainder_times
    used_file_indices = []
    
    # Start from the provided file index
    file_idx = start_file_idx
    
    # If we have initial data from a previous batch, count it
    initial_samples = 0 if current_data is None else current_data.shape[1]
    
    while initial_samples < max_samples and file_idx < len(files):
        file = files[file_idx]
        print(f"Loading file: {os.path.basename(file)}")
        
        # Parse the timestamp from the filename
        timestamp = parse_timestamp_from_filename(file)
        
        # If reference_time is not set and this is the first file, set it to 7 AM of this day
        if reference_time is None and (current_data is None):
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
        time_array = generate_time_array(timestamp, n_samples, sfreq, reference_time,cyclic_24h=True)
        
        if current_data is None:
            current_data = file_data
            current_times = time_array
        else:
            # Concatenate along time dimension (axis=1)
            current_data = np.concatenate([current_data, file_data], axis=1)
            current_times = np.concatenate([current_times, time_array])
        
        used_file_indices.append(file_idx)
        file_idx += 1
        
        # Update initial_samples after adding new data
        initial_samples = current_data.shape[1]
    
    # Check if we have enough data
    if current_data.shape[1] <= max_samples:
        # Not enough data, return all of it
        batch_data = current_data
        batch_times = current_times
        remainder_data = None
        remainder_times = None
    else:
        # We have more than enough data, split it
        batch_data = current_data[:, :max_samples]
        batch_times = current_times[:max_samples]
        remainder_data = current_data[:, max_samples:]
        remainder_times = current_times[max_samples:]
    
    # Transpose to Time x Channels
    batch_data = batch_data.T
    
    # Normalize
    batch_data = (batch_data - batch_data.mean(axis=0)) / batch_data.std(axis=0)
    
    return batch_data, batch_times, np.array(used_file_indices), file_idx, remainder_data, remainder_times

def prepare_MARBLE_data(
    data: np.ndarray, 
    time_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for MARBLE training by creating position and vector lists.
    
    Args:
        data: Input data (Time x Channels)
        time_array: Array of time points
        
    Returns:
        Tuple containing:
            - Position list (Time-1 x Channels)
            - Vector list (Time-1 x Channels)
            - Labels (Time-1)
    """
    # Create position and vector lists
    pos_list = data[:-1, :]
    x_list = np.diff(data, axis=0)
    
    # Use time array as labels (exclude last point)
    labels = time_array[:-1]
    
    return pos_list, x_list, labels

def train_MARBLE_model(
    data_dir: str,
    max_samples: int = 100000,
    resample_freq: float = 200,
    k_value: int = 20,
    params: dict = None,
    reference_time: Optional[datetime.datetime] = None,
    process_all_batches: bool = False
):
    """
    Train MARBLE model on batches of data loaded from data_dir.
    
    Args:
        data_dir: Directory containing the EEG files
        max_samples: Maximum number of samples per batch
        resample_freq: Frequency to resample the data to
        k_value: k value for MARBLE dataset construction
        params: Parameters for MARBLE.net
        reference_time: Reference time for time array (default: 7:00 AM of the first file's day)
        process_all_batches: Whether to process all available data in batches
        
    Returns:
        List of trained MARBLE models and transformed data
    """
    import MARBLE
    from MARBLE import postprocessing
    
    if params is None:
        params = {
            "epochs": 50,
            "order": 1,
            "hidden_channels": [256],
            "batch_size": 256,
            "lr": 1e-3,
            "out_channels": 3,
            "inner_product_features": False,
            "emb_norm": True,
            "diffusion": True,
        }
    
    models = []
    transformed_data_list = []
    
    # Initialize variables for tracking remainder data
    file_idx = 0
    remainder_data = None
    remainder_times = None
    
    batch_count = 0
    
    while True:
        try:
            # Load and merge data from files
            batch_data, batch_times, used_files, next_file_idx, remainder_data, remainder_times = load_merged_data(
                data_dir=data_dir,
                max_samples=max_samples,
                resample_freq=resample_freq,
                reference_time=reference_time,
                start_file_idx=file_idx,
                prev_remainder_data=remainder_data,
                prev_remainder_times=remainder_times
            )
            
            # Update file index for next batch
            file_idx = next_file_idx
            
            print(f"Batch {batch_count}: Loaded data with shape {batch_data.shape}")
            
            # Prepare data for MARBLE
            pos_list, x_list, labels = prepare_MARBLE_data(batch_data, batch_times)
            
            # Construct dataset
            Dataset = MARBLE.construct_dataset(
                anchor=pos_list, 
                vector=x_list,
                label=labels,
                graph_type="cknn",
                k=k_value,  
                spacing=0.05,
            )
            
            # Train model
            model = MARBLE.net(Dataset, params=params)
            model.fit(Dataset)
            
            # Transform data
            transformed_data = model.transform(Dataset)
            transformed_data = postprocessing.embed_in_2D(transformed_data)
            
            # Store results
            models.append(model)
            transformed_data_list.append(transformed_data)
            
            batch_count += 1
            
            # If not processing all batches, exit after the first batch
            if not process_all_batches:
                break
                
            # If we've used all files and there's no remainder, exit
            if next_file_idx >= len(glob(os.path.join(data_dir, "*.fif"))) and remainder_data is None:
                print("Processed all available data")
                break
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            break
    
    return models, transformed_data_list 