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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import from data_processing.py
try:
    from data_processing import (
        parse_timestamp_from_filename,
        generate_time_array,
        detect_interictal_events
    )
except ImportError:
    logger.error("Could not import functions from data_processing.py. Make sure it's in the same directory.")
    sys.exit(1)

class Dataset:
    """Base class for datasets"""
    def __init__(self, config: Dict):
        self.config = config
        self.metadata = {
            "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": config
        }
    def save(self, filepath: str):
        """Save dataset to disk"""
        raise NotImplementedError("Subclasses must implement this method")
    
    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def __len__(self):
        """Return number of samples"""
        raise NotImplementedError("Subclasses must implement this method")

class RawDataset(Dataset):
    """Dataset containing raw iEEG data"""
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.data = []          # List of data chunks (Channel x Time)
        self.time_arrays = []   # List of time arrays (Time)
        self.event_labels = []  # List of event labels (Time)
        self.day_night_labels = []  # List of day/night labels (Time)
        self.file_indices = []  # List of file indices used
        
        # Resampling parameters
        self.resample_freq = config.get('raw_dataset', {}).get('resample_freq', None)
        
        # Day/night parameters - 8PM (20) to 6AM (6)
        self.night_start_hour = config.get('raw_dataset', {}).get('night_start_hour', 20)
        self.night_end_hour = config.get('raw_dataset', {}).get('night_end_hour', 6)
        
        self.metadata.update({
            "dataset_type": "raw",
            "resample_freq": self.resample_freq,
            "day_night_params": {
                "night_start_hour": self.night_start_hour,
                "night_end_hour": self.night_end_hour
            },
            "used_files": [],
            "batch_info": []
        })
        
    def add_chunk(self, 
                 data: np.ndarray, 
                 time_array: np.ndarray, 
                 event_labels: np.ndarray,
                 day_night_labels: np.ndarray,
                 file_indices: List[int],
                 filenames: List[str]):
        """Add a chunk of data to the dataset"""
        self.data.append(data)
        self.time_arrays.append(time_array)
        self.event_labels.append(event_labels)
        self.day_night_labels.append(day_night_labels)
        self.file_indices.append(file_indices)
        
        # Update metadata
        self.metadata["used_files"].extend(filenames)
        self.metadata["batch_info"].append({
            "shape": data.shape,
            "time_range": [float(time_array[0]), float(time_array[-1])],
            "event_count": int(np.sum(event_labels)),
            "night_time_percentage": float(np.mean(day_night_labels == 0) * 100)
        })
        
    def save(self, filepath: str):
        """Save dataset to disk"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Combine data for saving
        data_array = np.array(self.data)
        time_array = np.array(self.time_arrays)
        event_array = np.array(self.event_labels)
        day_night_array = np.array(self.day_night_labels)
        
        # Save as npz
        np.savez(
            filepath,
            data=data_array,
            time=time_array,
            events=event_array,
            day_night=day_night_array,
            metadata=pickle.dumps(self.metadata)
        )
        
        logger.info(f"Saved raw dataset to {filepath}")
        
    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        # Load the npz file
        data_dict = np.load(filepath, allow_pickle=True)
        
        # Extract data
        data_array = data_dict['data']
        time_array = data_dict['time']
        event_array = data_dict['events']
        
        # Handle backward compatibility for day_night labels
        if 'day_night' in data_dict:
            day_night_array = data_dict['day_night']
        else:
            logger.warning("No day/night labels found in dataset, using default (all day)")
            day_night_array = np.ones_like(time_array)  # Default to all day
        
        metadata = pickle.loads(data_dict['metadata'])
        
        # Create dataset instance
        dataset = cls(metadata['config'])
        dataset.data = [data_array[i] for i in range(len(data_array))]
        dataset.time_arrays = [time_array[i] for i in range(len(time_array))]
        dataset.event_labels = [event_array[i] for i in range(len(event_array))]
        dataset.day_night_labels = [day_night_array[i] for i in range(len(day_night_array))]
        dataset.metadata = metadata
        
        logger.info(f"Loaded raw dataset from {filepath} with {len(dataset)} chunks")
        return dataset
    
    def __len__(self):
        """Return number of chunks"""
        return len(self.data)

class BandpowerDataset(Dataset):
    """Dataset containing bandpower features for multiple frequency bands"""
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Store bandpower data for each band
        self.bands = {}  # Dict of band_name -> List of data chunks (Windows x Channels)
        self.time_arrays = []  # List of time arrays (Windows)
        self.event_labels = []  # List of event labels (Windows)
        self.day_night_labels = []  # List of day/night labels (Windows)
        self.file_indices = []  # List of file indices used
        
        # Initialize with band definitions from config
        self.band_definitions = config.get('bandpower', {}).get('bands', {
            'gamma': [80, 250],
            'high_gamma': [250, 400]
        })
        
        # Day/night parameters - 8PM (20) to 6AM (6)
        self.night_start_hour = config.get('bandpower', {}).get('night_start_hour', 20)
        self.night_end_hour = config.get('bandpower', {}).get('night_end_hour', 6)
        
        self.metadata.update({
            "dataset_type": "bandpower",
            "band_definitions": self.band_definitions,
            "window_size_ms": config.get('bandpower', {}).get('window_size_ms', 100),
            "overlap": config.get('bandpower', {}).get('overlap', 0.5),
            "day_night_params": {
                "night_start_hour": self.night_start_hour,
                "night_end_hour": self.night_end_hour
            },
            "used_files": [],
            "batch_info": []
        })
        
    def add_chunk(self, 
                 band_data: Dict[str, np.ndarray],
                 time_array: np.ndarray, 
                 event_labels: np.ndarray,
                 day_night_labels: np.ndarray,
                 file_indices: List[int],
                 filenames: List[str]):
        """
        Add a chunk of bandpower data to the dataset
        
        Args:
            band_data: Dictionary of band_name -> data array (Windows x Channels)
            time_array: Array of time points (Windows)
            event_labels: Array of event labels (Windows)
            day_night_labels: Array of day/night labels (Windows)
            file_indices: List of file indices
            filenames: List of filenames used
        """
        # Initialize band data lists if not already done
        for band_name in band_data.keys():
            if band_name not in self.bands:
                self.bands[band_name] = []
            self.bands[band_name].append(band_data[band_name])
        
        self.time_arrays.append(time_array)
        self.event_labels.append(event_labels)
        self.day_night_labels.append(day_night_labels)
        self.file_indices.append(file_indices)
        
        # Update metadata
        self.metadata["used_files"].extend(filenames)
        self.metadata["batch_info"].append({
            "shape": {band: data.shape for band, data in band_data.items()},
            "time_range": [float(time_array[0]), float(time_array[-1])],
            "event_count": int(np.sum(event_labels)),
            "night_time_percentage": float(np.mean(day_night_labels == 0) * 100)
        })
        
    def save(self, filepath: str):
        """Save dataset to disk"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Prepare data for saving
        save_dict = {
            'metadata': pickle.dumps(self.metadata),
            'time': np.array(self.time_arrays),
            'events': np.array(self.event_labels),
            'day_night': np.array(self.day_night_labels)
        }
        
        # Add bandpower data for each band
        for band_name, band_data in self.bands.items():
            save_dict[f'band_{band_name}'] = np.array(band_data)
        
        # Save as npz
        np.savez(filepath, **save_dict)
        
        logger.info(f"Saved bandpower dataset to {filepath} with bands: {list(self.bands.keys())}")
        
    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        # Load the npz file
        data_dict = np.load(filepath, allow_pickle=True)
        
        # Extract metadata
        metadata = pickle.loads(data_dict['metadata'])
        
        # Create dataset instance
        dataset = cls(metadata['config'])
        
        # Load time and event data
        time_array = data_dict['time']
        event_array = data_dict['events']
        
        dataset.time_arrays = [time_array[i] for i in range(len(time_array))]
        dataset.event_labels = [event_array[i] for i in range(len(event_array))]
        
        # Handle backward compatibility for day_night labels
        if 'day_night' in data_dict:
            day_night_array = data_dict['day_night']
            dataset.day_night_labels = [day_night_array[i] for i in range(len(day_night_array))]
        else:
            logger.warning("No day/night labels found in dataset, using default (all day)")
            dataset.day_night_labels = [np.ones_like(t) for t in time_array]
        
        # Load band data
        dataset.bands = {}
        for key in data_dict.keys():
            if key.startswith('band_'):
                band_name = key[5:]  # Remove 'band_' prefix
                band_data = data_dict[key]
                dataset.bands[band_name] = [band_data[i] for i in range(len(band_data))]
        
        dataset.metadata = metadata
        
        logger.info(f"Loaded bandpower dataset from {filepath} with {len(dataset)} chunks and bands: {list(dataset.bands.keys())}")
        return dataset
    
    def __len__(self):
        """Return number of chunks"""
        return len(self.time_arrays)
    
    def get_data_for_marble(self, band_name: str = None, transpose: bool = True) -> List[np.ndarray]:
        """
        Get data in format suitable for MARBLE training (Channel x Time)
        
        Args:
            band_name: Which band to use. If None, uses the first available band
            transpose: Whether to transpose from (Windows x Channels) to (Channels x Windows)
            
        Returns:
            List of arrays in (Channel x Time) format
        """
        if not self.bands:
            logger.error("No band data available")
            return []
        
        # Use specified band or first available
        if band_name is None or band_name not in self.bands:
            band_name = list(self.bands.keys())[0]
            logger.info(f"Using band '{band_name}' for MARBLE data")
        
        band_data = self.bands[band_name]
        
        if transpose:
            # Convert from (Windows x Channels) to (Channels x Windows)
            return [data.T for data in band_data]
        else:
            return band_data

class EventSegmentDataset(Dataset):
    """Dataset containing only segments with high-frequency events"""
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.data = []            # List of segments (Channel x Time)
        self.time_arrays = []     # List of absolute time arrays for each segment (Time)
        self.rel_time_arrays = [] # List of relative time arrays for each segment (Time)
        self.event_labels = []    # List of event labels (Time)
        self.day_night_labels = []# List of day/night labels (Time)
        self.event_sources = []   # Source information for segments (file, channel, start_time)
        
        # Event segment specific parameters
        self.segment_params = config.get('event_segments', {})
        self.pre_event_ms = self.segment_params.get('pre_event_ms', 1000)
        self.post_event_ms = self.segment_params.get('post_event_ms', 1000)
        
        # Event detection bands
        self.low_band = config.get('event_detection', {}).get('low_band', [80, 250])
        self.high_band = config.get('event_detection', {}).get('high_band', [250, 451])
        
        # Day/night parameters - 8PM (20) to 6AM (6)
        self.night_start_hour = config.get('event_segments', {}).get('night_start_hour', 20)
        self.night_end_hour = config.get('event_segments', {}).get('night_end_hour', 6)
        
        self.metadata.update({
            "dataset_type": "event_segments",
            "pre_event_ms": self.pre_event_ms,
            "post_event_ms": self.post_event_ms,
            "event_detection": {
                "low_band": self.low_band,
                "high_band": self.high_band
            },
            "day_night_params": {
                "night_start_hour": self.night_start_hour,
                "night_end_hour": self.night_end_hour
            },
            "used_files": [],
            "segment_info": []
        })

    def add_segments(self, segments_array, segment_indices, segment_files, event_types=None):
        """
        Add multiple segments to the dataset
        
        Args:
            segments_array: Array of segments with shape (n_segments, n_channels, n_timepoints)
            segment_indices: List of segment indices
            segment_files: List of source files for each segment
            event_types: List of event types (1=low_band, 2=high_band, 3=both). Default is all type 1.
        """
        n_segments = segments_array.shape[0]
        
        # Default to all type 1 events if not provided
        if event_types is None:
            event_types = [1] * n_segments
        
        for i in range(n_segments):
            segment = segments_array[i]
            segment_length = segment.shape[1]
            
            # Create dummy time arrays (we don't have real timestamps for these segments)
            # Absolute time is set to index values
            abs_time = np.arange(segment_length)
            # Relative time starts from 0
            rel_time = np.arange(segment_length)
            
            # Create dummy event and day/night labels
            # Mark the center of the segment with the appropriate event type
            event_labels = np.zeros(segment_length)
            center_idx = segment_length // 2
            
            # Use the event type from the list
            event_type = event_types[i]
            # Ensure it's 1, 2, or 3
            if event_type not in [1, 2, 3]:
                event_type = 1  # Default to type 1
            
            event_labels[center_idx] = event_type
            
            # Default to day
            day_night_labels = np.ones(segment_length)
            
            # Create source info
            source_info = {
                'file': segment_files[i],
                'segment_idx': segment_indices[i],
                'shape': segment.shape,
                'event_type': event_type
            }
            
            # Add to dataset
            self.data.append(segment)
            self.time_arrays.append(abs_time)
            self.rel_time_arrays.append(rel_time)
            self.event_labels.append(event_labels)
            self.day_night_labels.append(day_night_labels)
            self.event_sources.append(source_info)
            
            # Update metadata with event type counts
            event_info = {
                "shape": segment.shape,
                "segment_idx": segment_indices[i],
                "source": source_info,
                "event_type": event_type
            }
            
            self.metadata["segment_info"].append(event_info)
            
            # Add file to used files if not already present
            if segment_files[i] not in self.metadata["used_files"]:
                self.metadata["used_files"].append(segment_files[i])
            
    def add_segment(self, 
                   data: np.ndarray, 
                   abs_time_array: np.ndarray,
                   rel_time_array: np.ndarray, 
                   event_labels: np.ndarray,
                   day_night_labels: np.ndarray,
                   source_info: Dict):
        """Add a segment of data to the dataset"""
        self.data.append(data)
        self.time_arrays.append(abs_time_array)
        self.rel_time_arrays.append(rel_time_array)
        self.event_labels.append(event_labels)
        self.day_night_labels.append(day_night_labels)
        self.event_sources.append(source_info)
        
        # Update metadata
        self.metadata["segment_info"].append({
            "shape": data.shape,
            "abs_time_range": [float(abs_time_array[0]), float(abs_time_array[-1])],
            "rel_time_range": [float(rel_time_array[0]), float(rel_time_array[-1])],
            "event_type_counts": {
                "type1": int(np.sum(event_labels == 1)),
                "type2": int(np.sum(event_labels == 2)),
                "type3": int(np.sum(event_labels == 3))
            },
            "night_percentage": float(np.mean(day_night_labels == 0) * 100),
            "source": source_info
        })
        
        # Add file to used files if not already present
        if source_info['file'] not in self.metadata["used_files"]:
            self.metadata["used_files"].append(source_info['file'])
            
    def set_metadata(self, metadata_dict):
        """Update dataset metadata with custom values"""
        self.metadata.update(metadata_dict)

    def save(self, filepath: str):
        """Save dataset to disk"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Combine data for saving
        data_array = np.array(self.data)
        abs_time_array = np.array(self.time_arrays)
        rel_time_array = np.array(self.rel_time_arrays)
        event_labels_array = np.array(self.event_labels)
        day_night_array = np.array(self.day_night_labels)
        sources_array = np.array(self.event_sources, dtype=object)
        
        # Save as npz
        np.savez(
            filepath,
            data=data_array,
            abs_time=abs_time_array,
            rel_time=rel_time_array,
            events=event_labels_array,
            day_night=day_night_array,
            sources=sources_array,
            metadata=pickle.dumps(self.metadata)
        )
        
        logger.info(f"Saved event segment dataset to {filepath} with {len(self.data)} segments")
        
    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        # Load the npz file
        data_dict = np.load(filepath, allow_pickle=True)
        
        # Extract data
        data_array = data_dict['data']
        
        # Handle absolute and relative time for backward compatibility
        if 'abs_time' in data_dict:
            abs_time_array = data_dict['abs_time']
        else:
            abs_time_array = data_dict['time']  # For backward compatibility
            
        if 'rel_time' in data_dict:
            rel_time_array = data_dict['rel_time']
        else:
            # Create relative time arrays if not available
            rel_time_array = np.array([t - t[0] for t in abs_time_array])
        
        # Handle event labels for backward compatibility
        if 'events' in data_dict:
            event_labels_array = data_dict['events']
        else:
            # Default to all type 1 events
            event_labels_array = np.array([np.ones_like(t) for t in abs_time_array])
        
        # Handle day/night labels for backward compatibility
        if 'day_night' in data_dict:
            day_night_array = data_dict['day_night']
        else:
            # Default to all day
            day_night_array = np.array([np.ones_like(t) for t in abs_time_array])
        
        sources_array = data_dict['sources']
        metadata = pickle.loads(data_dict['metadata'])
        
        # Create dataset instance
        dataset = cls(metadata['config'])
        dataset.data = [data_array[i] for i in range(len(data_array))]
        dataset.time_arrays = [abs_time_array[i] for i in range(len(abs_time_array))]
        dataset.rel_time_arrays = [rel_time_array[i] for i in range(len(rel_time_array))]
        dataset.event_labels = [event_labels_array[i] for i in range(len(event_labels_array))]
        dataset.day_night_labels = [day_night_array[i] for i in range(len(day_night_array))]
        dataset.event_sources = [sources_array[i] for i in range(len(sources_array))]
        dataset.metadata = metadata
        
        logger.info(f"Loaded event segment dataset from {filepath} with {len(dataset)} segments")
        return dataset
    
    def __len__(self):
        """Return number of segments"""
        return len(self.data)
    
    def get_concatenated_data(self, use_relative_time: bool = True):
        """
        Get concatenated event segments as a single continuous array
        
        Args:
            use_relative_time: Whether to use relative time (True) or absolute time (False)
            
        Returns:
            Tuple containing:
                - Concatenated data (Channels x Time)
                - Concatenated time array (Time)
                - Concatenated event labels (Time)
                - Concatenated day/night labels (Time)
                - Segment boundaries (start, end) indices
        """
        if not self.data:
            return None, None, None, None, []
        
        # Concatenate all segments
        concatenated_data = np.concatenate(self.data, axis=1)
        
        # Create indices for segment boundaries
        boundaries = []
        current_idx = 0
        for segment in self.data:
            segment_length = segment.shape[1]
            boundaries.append((current_idx, current_idx + segment_length))
            current_idx += segment_length
        
        # Create continuous arrays for time, events, and day/night labels
        concatenated_time = np.zeros(concatenated_data.shape[1])
        concatenated_events = np.zeros(concatenated_data.shape[1])
        concatenated_day_night = np.zeros(concatenated_data.shape[1])
        
        current_idx = 0
        for i in range(len(self.data)):
            segment_length = self.data[i].shape[1]
            
            # Use relative or absolute time
            if use_relative_time:
                time_array = self.rel_time_arrays[i]
            else:
                time_array = self.time_arrays[i]
                
            concatenated_time[current_idx:current_idx+segment_length] = time_array
            concatenated_events[current_idx:current_idx+segment_length] = self.event_labels[i]
            concatenated_day_night[current_idx:current_idx+segment_length] = self.day_night_labels[i]
            
            current_idx += segment_length
            
        return concatenated_data, concatenated_time, concatenated_events, concatenated_day_night, boundaries

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
                         window_size_ms: float) -> np.ndarray:
    """
    Map event array to windows based on window center times
    
    Args:
        event_array: Event array for raw data (samples)
        window_times: Array of window center times
        sfreq: Sampling frequency
        window_size_ms: Window size in milliseconds
        
    Returns:
        Array of event labels for windows
    """
    window_events = np.zeros(len(window_times))
    
    # Half window in samples
    half_window = int(window_size_ms * sfreq / 2000)
    
    for i, window_time in enumerate(window_times):
        # Calculate the sample index in the original data
        central_sample_idx = int(window_time * sfreq)
        
        # Use a small window around the central point to check for events
        start_sample = max(0, central_sample_idx - half_window)
        end_sample = min(len(event_array), central_sample_idx + half_window)
        
        # If any sample in the window has an event, mark the window
        if start_sample < end_sample and np.any(event_array[start_sample:end_sample] > 0):
            window_events[i] = 1
    
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
    segment_params = config.get('event_segment', {
        'pre_event': 1000,  # ms before event
        'post_event': 1000  # ms after event
    })
    
    pre_event = segment_params.get('pre_event', 1000)
    post_event = segment_params.get('post_event', 1000)
    
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
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
            
            # Detect events
            logger.info(f"Detecting events in {os.path.basename(file)}")
            event_array = detect_interictal_events(
                data, 
                sfreq,
                ch_names, 
                freq_band=event_params.get('low_band', [80, 250]),
                rel_thresh=event_params.get('rel_thresh', 3.0),
                abs_thresh=event_params.get('abs_thresh', 3.0),
                min_gap=event_params.get('min_gap', 20),
                min_last=event_params.get('min_last', 20)
            )
            
            # If no events found, skip to next file
            if np.sum(event_array) == 0:
                logger.info(f"No events detected in {os.path.basename(file)}")
                continue
                
            # Find start indices of each event
            event_indices = np.where(np.diff(np.concatenate([[0], event_array])) == 1)[0]
            
            if len(event_indices) == 0:
                logger.info(f"No events detected in {os.path.basename(file)}")
                continue
                
            logger.info(f"Found {len(event_indices)} events in {os.path.basename(file)}")
            
            # Calculate segment samples
            pre_event_samples = int(pre_event * sfreq / 1000)
            post_event_samples = int(post_event * sfreq / 1000)
            segment_length = pre_event_samples + post_event_samples
            
            # Extract segments
            n_events = len(event_indices)
            file_segments = []
            
            for i, event_idx in enumerate(event_indices):
                # Check if we need more segments
                if max_event_segments is not None and total_segments >= max_event_segments:
                    break
                    
                # Calculate segment boundaries
                start_idx = max(0, event_idx - pre_event_samples)
                end_idx = min(data.shape[1], event_idx + post_event_samples)
                
                # Skip if segment would be incomplete
                if end_idx - start_idx < segment_length * 0.9:  # Allow some tolerance
                    continue
                    
                # Extract segment
                segment = data[:, start_idx:end_idx]
                
                # Pad if needed
                if segment.shape[1] < segment_length:
                    pad_before = max(0, pre_event_samples - (event_idx - start_idx))
                    pad_after = max(0, segment_length - segment.shape[1] - pad_before)
                    segment = np.pad(segment, ((0, 0), (pad_before, pad_after)), mode='constant')
                
                # Add to list
                file_segments.append(segment)
                
                # Increment total count
                total_segments += 1
                
                # Periodically report progress
                if i % 100 == 0 and i > 0:
                    logger.info(f"  Processed {i}/{n_events} events, extracted {len(file_segments)} segments")
            
            # If we found segments in this file, add to dataset
            if file_segments:
                # Stack segments for this file
                file_segments_array = np.stack(file_segments, axis=0)
                
                # Add to dataset
                segment_indices = list(range(total_segments - len(file_segments), total_segments))
                segment_files = [file] * len(file_segments)
                
                event_segment_dataset.add_segments(file_segments_array, segment_indices, segment_files)
                
                all_segment_files.extend(segment_files)
                all_segment_indices.extend(segment_indices)
                
                logger.info(f"Added {len(file_segments)} segments from {os.path.basename(file)}")
                
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Total segments extracted: {total_segments}")
    
    # Store metadata
    event_segment_dataset.set_metadata({
        'segment_files': all_segment_files,
        'segment_indices': all_segment_indices,
        'pre_event_ms': pre_event,
        'post_event_ms': post_event,
        'extraction_config': config
    })
    
    return event_segment_dataset

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
    
    # Event segment parameters (optional)
    max_event_segments = config.get('max_event_segments', config.get('event_segment', {}).get('max_event_segments'))
    
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
    files = sorted(glob(os.path.join(data_dir, "*.fif")))
    
    if not files:
        logger.error(f"No files found in {data_dir}")
        return raw_dataset, bandpower_dataset, event_segment_dataset
    
    # Apply max_files limit if specified
    if max_files is not None:
        files = files[:max_files]
    
    logger.info(f"Found {len(files)} files to process")
    
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
            
            # Get original data and sampling frequency
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
            
            # Calculate chunk size in samples and number of chunks
            chunk_samples = int(chunk_size_sec * original_sfreq)
            n_chunks = max(1, n_samples_original // chunk_samples)
            
            # Calculate file duration
            file_duration_sec = n_samples_original / original_sfreq
            logger.info(f"  File duration: {file_duration_sec/60:.1f} minutes ({file_duration_sec:.1f} seconds)")
            
            # Process data in chunks
            for chunk_idx in range(n_chunks):
                # Calculate chunk boundaries
                start_sample = chunk_idx * chunk_samples
                end_sample = min(n_samples_original, (chunk_idx + 1) * chunk_samples)
                
                # Skip chunk if it's too small
                if end_sample - start_sample < chunk_samples / 2:
                    logger.info(f"  Skipping chunk {chunk_idx+1}/{n_chunks} (too small: {end_sample-start_sample} samples)")
                    continue
                
                # Calculate chunk duration and update total
                chunk_duration_sec = (end_sample - start_sample) / original_sfreq
                total_duration_sec += chunk_duration_sec
                
                logger.info(f"  Processing chunk {chunk_idx+1}/{n_chunks} ({end_sample-start_sample} samples, {chunk_duration_sec:.1f} sec)")
                
                # Extract chunk data and time array
                chunk_data = original_data[:, start_sample:end_sample]
                chunk_time = time_array_original[start_sample:end_sample]
                chunk_day_night = day_night_original[start_sample:end_sample]
                
                # Detect events on chunk data
                logger.info(f"  Detecting events on chunk (original data)")
                chunk_events = detect_interictal_events(
                    chunk_data, 
                    original_sfreq,
                    ch_names, 
                    freq_band=event_params.get('low_band', [80, 250]),
                    rel_thresh=event_params.get('rel_thresh', 3.0),
                    abs_thresh=event_params.get('abs_thresh', 3.0),
                    min_gap=event_params.get('min_gap', 20),
                    min_last=event_params.get('min_last', 20),
                    start_time=(timestamp - reference_time).total_seconds() if reference_time else 0
                )
                
                # Process bandpower using ORIGINAL chunk data if needed
                if create_bandpower_dataset:
                    logger.info(f"  Calculating bandpower on chunk (original data)")
                    # Process each band
                    band_data = {}
                    for band_name, band_range in bandpower_params['bands'].items():
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
                        
                        # Map events to windows - using chunk events
                        window_events = map_events_to_windows(
                            chunk_events,
                            window_times,
                            original_sfreq,
                            bandpower_params['window_size_ms']
                        )
                        
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
                    chunk_duration = (end_sample - start_sample) / original_sfreq
                    
                    # Create evenly spaced time points for the resampled data
                    chunk_start_time = chunk_time[0]
                    resampled_time = np.linspace(
                        chunk_start_time,
                        chunk_start_time + chunk_duration,
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
                    
                    # Map each original event sample to resampled time
                    if np.any(chunk_events > 0):
                        event_indices = np.where(chunk_events > 0)[0]
                        
                        for event_idx in event_indices:
                            # Calculate time of this event in seconds
                            event_time = chunk_time[event_idx]
                            
                            # Find closest time in resampled data
                            closest_idx = np.argmin(np.abs(resampled_time - event_time))
                            if 0 <= closest_idx < len(resampled_events):
                                resampled_events[closest_idx] = 1
                    
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
                
                # Extract event segments if enabled (using original chunk data)
                if create_event_segment_dataset:
                    logger.info(f"  Extracting event segments from chunk")
                    
                    # Find event indices
                    event_indices = np.where(np.diff(np.concatenate([[0], chunk_events])) == 1)[0]
                    
                    if len(event_indices) == 0:
                        logger.info(f"  No events found in chunk")
                        continue
                    
                    # Get segment parameters
                    pre_event_ms = config.get('event_segments', {}).get('pre_event_ms', 1000)
                    post_event_ms = config.get('event_segments', {}).get('post_event_ms', 1000)
                    
                    # Calculate segment samples
                    pre_event_samples = int(pre_event_ms * original_sfreq / 1000)
                    post_event_samples = int(post_event_ms * original_sfreq / 1000)
                    segment_length = pre_event_samples + post_event_samples
                    
                    # Extract segments from chunk
                    chunk_segments = []
                    segment_indices = []
                    segment_files = []
                    segment_event_types = []
                    
                    # Get event detection bands
                    low_band = event_params.get('low_band', [80, 250])
                    high_band = event_params.get('high_band', [250, 451])
                    
                    # Detect events in different bands for classification
                    # Use existing events from low_band detection
                    low_band_events = chunk_events.copy() 
                    
                    # Detect events in high band
                    high_band_events = detect_interictal_events(
                        chunk_data, 
                        original_sfreq,
                        ch_names, 
                        freq_band=high_band,
                        rel_thresh=event_params.get('rel_thresh', 3.0),
                        abs_thresh=event_params.get('abs_thresh', 3.0),
                        min_gap=event_params.get('min_gap', 20),
                        min_last=event_params.get('min_last', 20)
                    )
                    
                    for i, event_idx in enumerate(event_indices):
                        # Calculate segment boundaries relative to chunk
                        start_idx = max(0, event_idx - pre_event_samples)
                        end_idx = min(chunk_data.shape[1], event_idx + post_event_samples)
                        
                        # Skip if segment would be incomplete
                        if end_idx - start_idx < segment_length * 0.9:  # Allow some tolerance
                            continue
                        
                        # Extract segment
                        segment = chunk_data[:, start_idx:end_idx]
                        
                        # Pad if needed
                        if segment.shape[1] < segment_length:
                            pad_before = max(0, pre_event_samples - (event_idx - start_idx))
                            pad_after = max(0, segment_length - segment.shape[1] - pad_before)
                            segment = np.pad(segment, ((0, 0), (pad_before, pad_after)), mode='constant')
                        
                        # Determine event type by checking high_band_events at this index
                        # Check for high band event within a small window around this event
                        window_start = max(0, event_idx - 10)
                        window_end = min(len(high_band_events), event_idx + 10)
                        
                        # Determine event type
                        if np.any(high_band_events[window_start:window_end] > 0):
                            # Both bands have activity - type 3
                            event_type = 3
                        else:
                            # Only low band has activity - type 1
                            event_type = 1
                        
                        # Add to lists
                        chunk_segments.append(segment)
                        segment_indices.append(total_event_segments + len(chunk_segments) - 1)
                        segment_files.append(file)
                        segment_event_types.append(event_type)
                    
                    # Now check for high-band only events (type 2)
                    high_band_event_indices = np.where(np.diff(np.concatenate([[0], high_band_events])) == 1)[0]
                    
                    for event_idx in high_band_event_indices:
                        # Skip if this is also a low-band event (already processed)
                        window_start = max(0, event_idx - 10) 
                        window_end = min(len(low_band_events), event_idx + 10)
                        
                        if np.any(low_band_events[window_start:window_end] > 0):
                            continue  # Skip as this was already counted as type 3
                            
                        # Calculate segment boundaries relative to chunk
                        start_idx = max(0, event_idx - pre_event_samples)
                        end_idx = min(chunk_data.shape[1], event_idx + post_event_samples)
                        
                        # Skip if segment would be incomplete
                        if end_idx - start_idx < segment_length * 0.9:  # Allow some tolerance
                            continue
                        
                        # Extract segment
                        segment = chunk_data[:, start_idx:end_idx]
                        
                        # Pad if needed
                        if segment.shape[1] < segment_length:
                            pad_before = max(0, pre_event_samples - (event_idx - start_idx))
                            pad_after = max(0, segment_length - segment.shape[1] - pad_before)
                            segment = np.pad(segment, ((0, 0), (pad_before, pad_after)), mode='constant')
                        
                        # Add to lists
                        chunk_segments.append(segment)
                        segment_indices.append(total_event_segments + len(chunk_segments) - 1)
                        segment_files.append(file)
                        segment_event_types.append(2)  # Type 2 (high band only)
                    
                    # If we found segments in this chunk, add to dataset
                    if chunk_segments:
                        # Stack segments
                        segments_array = np.stack(chunk_segments, axis=0)
                        
                        # Add to dataset with event types
                        event_segment_dataset.add_segments(
                            segments_array, 
                            segment_indices, 
                            segment_files,
                            segment_event_types
                        )
                        
                        # Update counter
                        n_added = len(chunk_segments)
                        total_event_segments += n_added
                        logger.info(f"  Added {n_added} event segments from chunk")
                        
                        # Log counts of each event type
                        type1_count = segment_event_types.count(1)
                        type2_count = segment_event_types.count(2)
                        type3_count = segment_event_types.count(3)
                        logger.info(f"  Event types: {type1_count} type 1 (low band), {type2_count} type 2 (high band), {type3_count} type 3 (both bands)")
                
                # Log progress after each chunk
                hours = total_duration_sec / 3600
                minutes = (total_duration_sec % 3600) / 60
                logger.info(f"  Total processed: {hours:.0f}h {minutes:.0f}m ({total_duration_sec:.1f} seconds)")
            
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save datasets if requested
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if raw_dataset is not None and len(raw_dataset) > 0:
        raw_filepath = os.path.join(save_dir, f"{timestamp}_raw_dataset.npz")
        raw_dataset.save(raw_filepath)
        logger.info(f"Saved raw dataset with {len(raw_dataset)} chunks and {total_raw_samples} samples")
    
    if bandpower_dataset is not None and len(bandpower_dataset) > 0:
        bp_filepath = os.path.join(save_dir, f"{timestamp}_bandpower_dataset.npz")
        bandpower_dataset.save(bp_filepath)
        logger.info(f"Saved bandpower dataset with {len(bandpower_dataset)} chunks and {total_bandpower_windows} windows")
    
    if event_segment_dataset is not None and len(event_segment_dataset) > 0:
        es_filepath = os.path.join(save_dir, f"{timestamp}_event_segments_dataset.npz")
        event_segment_dataset.save(es_filepath)
        logger.info(f"Saved event segment dataset with {len(event_segment_dataset)} segments")
        
        # Log total counts of each event type
        if len(event_segment_dataset.event_labels) > 0:
            type1_count = sum(np.sum(labels == 1) for labels in event_segment_dataset.event_labels)
            type2_count = sum(np.sum(labels == 2) for labels in event_segment_dataset.event_labels)
            type3_count = sum(np.sum(labels == 3) for labels in event_segment_dataset.event_labels)
            logger.info(f"Event type distribution: {type1_count} type 1 (low band), {type2_count} type 2 (high band), {type3_count} type 3 (both bands)")
    
    # Log total processing duration
    hours = total_duration_sec / 3600
    minutes = (total_duration_sec % 3600) / 60
    logger.info(f"Total data processed: {hours:.0f}h {minutes:.0f}m ({total_duration_sec:.1f} seconds)")
    
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

if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "dataset_config.yaml"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Creating example configuration file...")
        
        # Create example config
        example_config = {
            "data_dir": "./preprocessed/bipolar",
            "max_samples": 10000,
            "chunk_size_sec": 200,
            "max_files": None,
            "save_dir": "./datasets",
            "create_raw_dataset": True,
            "create_bandpower_dataset": True,
            "create_event_segment_dataset": True,
            "max_event_segments": 1000,  # Maximum event segments to extract
            "reference_time": None,
            "event_detection": {
                "freq_band": [80, 250],
                "rel_thresh": 3.0,
                "abs_thresh": 3.0,
                "min_gap": 20,
                "min_last": 20
            },
            "event_segments": {
                "pre_event_ms": 1000,      # ms before event
                "post_event_ms": 1000,     # ms after event
                "min_duration_ms": 10,    # Minimum event duration to include
                "max_segments_per_file": 100  # Maximum segments per file
            },
            "bandpower": {
                "window_size_ms": 100,
                "overlap": 0.5,
                "bands": {
                    "gamma": [80, 250],
                    "high_gamma": [250, 400]
                }
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False)
        
        logger.info(f"Example configuration file created at {config_path}")
        logger.info("Edit the configuration file and run again.")
        sys.exit(0)
    
    # Construct datasets
    try:
        raw_dataset, bandpower_dataset, event_segment_dataset = construct_dataset_from_config(config_path)
        
        if raw_dataset:
            logger.info(f"Created raw dataset with {len(raw_dataset)} chunks")
        
        if bandpower_dataset:
            logger.info(f"Created bandpower dataset with {len(bandpower_dataset)} chunks")
            
        if event_segment_dataset:
            logger.info(f"Created event segment dataset with {len(event_segment_dataset)} segments")
            
    except Exception as e:
        logger.error(f"Error constructing datasets: {e}")
        import traceback
        traceback.print_exc() 