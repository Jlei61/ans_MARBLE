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

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        try:
            # Load the npz file
            data_dict = np.load(filepath, allow_pickle=True)
            
            # Extract metadata
            metadata = pickle.loads(data_dict['metadata'])
            
            # Create dataset instance
            dataset = cls(metadata['config'])
            
            # Check if this is the alternative format (with individual chunks)
            if 'n_chunks' in data_dict:
                # Load using alternative format
                n_chunks = int(data_dict['n_chunks'])
                logger.info(f"Loading dataset using alternative format with {n_chunks} chunks")
                
                dataset.data = []
                dataset.time_arrays = []
                dataset.event_labels = []
                dataset.day_night_labels = []
                
                for i in range(n_chunks):
                    dataset.data.append(data_dict[f'data_{i}'])
                    dataset.time_arrays.append(data_dict[f'time_{i}'])
                    dataset.event_labels.append(data_dict[f'events_{i}'])
                    dataset.day_night_labels.append(data_dict[f'day_night_{i}'])
            else:
                # Load using original format
                data_array = data_dict['data']
                time_array = data_dict['time']
                event_array = data_dict['events']
                
                # Handle backward compatibility for day_night labels
                if 'day_night' in data_dict:
                    day_night_array = data_dict['day_night']
                else:
                    logger.warning("No day/night labels found in dataset, using default (all day)")
                    day_night_array = np.ones_like(time_array)  # Default to all day
                
                # Handle potentially empty arrays
                n_chunks = len(data_array) if hasattr(data_array, '__len__') else 0
                
                if n_chunks > 0:
                    # Convert numpy arrays back to lists
                    try:
                        dataset.data = [data_array[i] for i in range(n_chunks)]
                        dataset.time_arrays = [time_array[i] for i in range(n_chunks)]
                        dataset.event_labels = [event_array[i] for i in range(n_chunks)]
                        dataset.day_night_labels = [day_night_array[i] for i in range(n_chunks)]
                    except Exception as e:
                        logger.error(f"Error converting arrays to lists: {e}")
                        # Try alternative access in case these are already lists
                        dataset.data = list(data_array)
                        dataset.time_arrays = list(time_array)
                        dataset.event_labels = list(event_array)
                        dataset.day_night_labels = list(day_night_array)
                else:
                    logger.warning("Dataset contains no chunks")
                    dataset.data = []
                    dataset.time_arrays = []
                    dataset.event_labels = []
                    dataset.day_night_labels = []
            
            dataset.metadata = metadata
            
            # Log shapes for debugging
            if dataset.data:
                logger.info(f"Loaded raw dataset shapes:")
                for i, d in enumerate(dataset.data):
                    logger.info(f"  Chunk {i}: shape {d.shape}")
            
            logger.info(f"Loaded raw dataset from {filepath} with {len(dataset)} chunks")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def __len__(self):
        """Return number of chunks"""
        return len(self.time_arrays)
        
    def get_batched_data(self, start_idx: int = 0, num_chunks: int = None, 
                        batch_size: int = None, transpose: bool = False,
                        chunks_per_batch_item: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get concatenated data from the dataset with optional batching.
        
        Args:
            start_idx: Index of first chunk to include
            num_chunks: Number of chunks to include (None for all available from start_idx)
            batch_size: If specified, shape output as (batch_size, time_steps, channels)
            transpose: Whether to transpose the default output format
            chunks_per_batch_item: Number of chunks to combine into each batch item
            
        Returns:
            Tuple of (data, time_arrays, event_labels) with consistent shape:
            - If batch_size is None: (time_steps, channels) or (channels, time_steps) if transpose=True
            - If batch_size is specified: (batch_size, time_steps, channels) or (batch_size, channels, time_steps)
        """
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
        
        # Count distinct events
        event_counts = count_distinct_events(event_labels)
        total_events = sum(event_counts.values())
        
        # Update metadata
        self.metadata["used_files"].extend(filenames)
        self.metadata["batch_info"].append({
            "shape": data.shape,
            "time_range": [float(time_array[0]), float(time_array[-1])],
            "event_count": total_events,
            "event_type_counts": {
                "type1": event_counts[1],
                "type2": event_counts[2],
                "type3": event_counts[3]
            },
            "night_time_percentage": float(np.mean(day_night_labels == 0) * 100)
        })
        
    def save(self, filepath: str):
        """Save dataset to disk"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Check if we have any data to save
        if not self.data:
            logger.warning("No data to save in RawDataset")
            # Save empty arrays
            np.savez(
                filepath,
                data=np.array([], dtype=object),
                time=np.array([], dtype=object),
                events=np.array([], dtype=object),
                day_night=np.array([], dtype=object),
                metadata=pickle.dumps(self.metadata)
            )
            logger.info(f"Saved empty raw dataset to {filepath}")
            return
        
        try:
            # First debug the shapes of our data
            logger.info(f"Saving {len(self.data)} data chunks with shapes:")
            for i, d in enumerate(self.data):
                logger.info(f"  Chunk {i}: shape {d.shape}")
            
            # Create a pickled list of data arrays instead of trying to stack them
            # This avoids broadcasting errors with inconsistent shapes
            data_list = self.data  # Keep as Python list
            time_arrays_list = self.time_arrays
            event_labels_list = self.event_labels
            day_night_labels_list = self.day_night_labels
            
            # Save as npz with pickle for lists
            np.savez(
                filepath,
                data=np.array(data_list, dtype=object),  # This should work with dtype=object
                time=np.array(time_arrays_list, dtype=object),
                events=np.array(event_labels_list, dtype=object),
                day_night=np.array(day_night_labels_list, dtype=object),
                metadata=pickle.dumps(self.metadata)
            )
            
            logger.info(f"Saved raw dataset to {filepath}")
        except Exception as e:
            logger.error(f"Error saving raw dataset: {e}")
            import traceback
            traceback.print_exc()
            
            # Try alternative saving method
            try:
                logger.info("Trying alternative saving method...")
                # Create a dictionary with individual chunks to avoid array creation
                save_dict = {
                    'metadata': pickle.dumps(self.metadata)
                }
                
                # Save each chunk separately
                for i, (data, time, events, day_night) in enumerate(zip(
                    self.data, self.time_arrays, self.event_labels, self.day_night_labels)):
                    save_dict[f'data_{i}'] = data
                    save_dict[f'time_{i}'] = time
                    save_dict[f'events_{i}'] = events
                    save_dict[f'day_night_{i}'] = day_night
                
                # Add chunk count
                save_dict['n_chunks'] = len(self.data)
                
                # Save as npz
                np.savez(filepath, **save_dict)
                logger.info(f"Saved raw dataset using alternative method to {filepath}")
            except Exception as e2:
                logger.error(f"Error using alternative save method: {e2}")
                traceback.print_exc()
                raise
    
    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        try:
            # Load the npz file
            data_dict = np.load(filepath, allow_pickle=True)
            
            # Extract metadata
            metadata = pickle.loads(data_dict['metadata'])
            
            # Create dataset instance
            dataset = cls(metadata['config'])
            
            # Check if this is the alternative format (with individual chunks)
            if 'n_chunks' in data_dict:
                # Load using alternative format
                n_chunks = int(data_dict['n_chunks'])
                logger.info(f"Loading dataset using alternative format with {n_chunks} chunks")
                
                dataset.data = []
                dataset.time_arrays = []
                dataset.event_labels = []
                dataset.day_night_labels = []
                
                for i in range(n_chunks):
                    dataset.data.append(data_dict[f'data_{i}'])
                    dataset.time_arrays.append(data_dict[f'time_{i}'])
                    dataset.event_labels.append(data_dict[f'events_{i}'])
                    dataset.day_night_labels.append(data_dict[f'day_night_{i}'])
            else:
                # Load using original format
                data_array = data_dict['data']
                time_array = data_dict['time']
                event_array = data_dict['events']
                
                # Handle backward compatibility for day_night labels
                if 'day_night' in data_dict:
                    day_night_array = data_dict['day_night']
                else:
                    logger.warning("No day/night labels found in dataset, using default (all day)")
                    day_night_array = np.ones_like(time_array)  # Default to all day
                
                # Handle potentially empty arrays
                n_chunks = len(data_array) if hasattr(data_array, '__len__') else 0
                
                if n_chunks > 0:
                    # Convert numpy arrays back to lists
                    try:
                        dataset.data = [data_array[i] for i in range(n_chunks)]
                        dataset.time_arrays = [time_array[i] for i in range(n_chunks)]
                        dataset.event_labels = [event_array[i] for i in range(n_chunks)]
                        dataset.day_night_labels = [day_night_array[i] for i in range(n_chunks)]
                    except Exception as e:
                        logger.error(f"Error converting arrays to lists: {e}")
                        # Try alternative access in case these are already lists
                        dataset.data = list(data_array)
                        dataset.time_arrays = list(time_array)
                        dataset.event_labels = list(event_array)
                        dataset.day_night_labels = list(day_night_array)
                else:
                    logger.warning("Dataset contains no chunks")
                    dataset.data = []
                    dataset.time_arrays = []
                    dataset.event_labels = []
                    dataset.day_night_labels = []
            
            dataset.metadata = metadata
            
            # Log shapes for debugging
            if dataset.data:
                logger.info(f"Loaded raw dataset shapes:")
                for i, d in enumerate(dataset.data):
                    logger.info(f"  Chunk {i}: shape {d.shape}")
            
            logger.info(f"Loaded raw dataset from {filepath} with {len(dataset)} chunks")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def __len__(self):
        """Return number of chunks"""
        return len(self.data)
    
    def get_batched_data(self, start_idx: int = 0, num_chunks: int = None, 
                        batch_size: int = 1, transpose: bool = False,
                        chunks_per_batch_item: int = 1) -> Tuple[List, List, List]:
        """
        Get concatenated data from raw dataset with optional batching.
        
        Args:
            start_idx: Index of first chunk to include
            num_chunks: Number of chunks to include (None for all available from start_idx)
            batch_size: If specified, shape output as batch of items
            transpose: Whether to transpose from (Channel x Time) to (Time x Channel)
            chunks_per_batch_item: Number of chunks to combine into each batch item
            
        Returns:
            Tuple of (data, time_arrays, event_labels) as lists of arrays
        """
        if not self.data:
            logger.warning("No data available in RawDataset")
            return [], [], []
        
        # Determine how many chunks to process
        end_idx = len(self.data) if num_chunks is None else min(start_idx + num_chunks, len(self.data))
        chunks_to_process = list(range(start_idx, end_idx))
        
        if not chunks_to_process:
            logger.warning(f"No valid chunks in range start_idx={start_idx}, num_chunks={num_chunks}")
            return [], [], []
        
        # Calculate number of batch items needed
        total_batch_items = len(chunks_to_process) // chunks_per_batch_item
        
        # Initialize batch arrays - these will be our final output
        batch_data = []
        batch_times = []
        batch_events = []
        
        # Process batch items
        for item_idx in range(total_batch_items):
            # Get chunks for this batch item
            start_chunk = start_idx + (item_idx * chunks_per_batch_item)
            end_chunk = start_chunk + chunks_per_batch_item
            item_chunks = list(range(start_chunk, min(end_chunk, len(self.data))))
            
            # Skip if no valid chunks
            if not item_chunks:
                continue
            
            # Concatenate chunks for this batch item
            item_data_chunks = [self.data[j] for j in item_chunks]
            item_time_chunks = [self.time_arrays[j] for j in item_chunks]
            item_event_chunks = [self.event_labels[j] for j in item_chunks]
            
            item_data = np.concatenate(item_data_chunks, axis=1)
            item_data = item_data.astype(np.float32) # Ensure float32 dtype
            item_time = np.concatenate(item_time_chunks)
            item_events = np.concatenate(item_event_chunks)
            
            # Transpose if needed
            if transpose:
                item_data = item_data.T
                
            batch_data.append(item_data)
            batch_times.append(item_time)
            batch_events.append(item_events)
        
        if not batch_data:
            logger.warning("No valid batches created")
            return [], [], []
            
        # Return the batch lists
        return batch_data, batch_times, batch_events

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
        
        # Count distinct events
        event_counts = count_distinct_events(event_labels)
        total_events = sum(event_counts.values())
        
        # Update metadata
        self.metadata["used_files"].extend(filenames)
        self.metadata["batch_info"].append({
            "shape": {band: data.shape for band, data in band_data.items()},
            "time_range": [float(time_array[0]), float(time_array[-1])],
            "event_count": total_events,
            "event_type_counts": {
                "type1": event_counts[1],
                "type2": event_counts[2],
                "type3": event_counts[3]
            },
            "night_time_percentage": float(np.mean(day_night_labels == 0) * 100)
        })
        
    def save(self, filepath: str):
        """Save dataset to disk"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Prepare data for saving
        save_dict = {
            'metadata': pickle.dumps(self.metadata),
            'time': np.array(self.time_arrays, dtype=object),
            'events': np.array(self.event_labels, dtype=object),
            'day_night': np.array(self.day_night_labels, dtype=object)
        }
        
        # Add bandpower data for each band
        for band_name, band_data in self.bands.items():
            save_dict[f'band_{band_name}'] = np.array(band_data, dtype=object)
        
        # Save as npz
        np.savez(filepath, **save_dict)
        
        logger.info(f"Saved bandpower dataset to {filepath} with bands: {list(self.bands.keys())}")
        
    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        try:
            # Load the npz file
            data_dict = np.load(filepath, allow_pickle=True)
            
            # Extract metadata
            metadata = pickle.loads(data_dict['metadata'])
            
            # Create dataset instance
            dataset = cls(metadata['config'])
            
            # Load time and event data
            time_array = data_dict['time']
            event_array = data_dict['events']
            
            # Handle potentially empty arrays
            n_chunks = len(time_array) if hasattr(time_array, '__len__') else 0
            
            if n_chunks > 0:
                dataset.time_arrays = [time_array[i] for i in range(n_chunks)]
                dataset.event_labels = [event_array[i] for i in range(n_chunks)]
                
                # Handle backward compatibility for day_night labels
                if 'day_night' in data_dict:
                    day_night_array = data_dict['day_night']
                    dataset.day_night_labels = [day_night_array[i] for i in range(n_chunks)]
                else:
                    logger.warning("No day/night labels found in dataset, using default (all day)")
                    dataset.day_night_labels = [np.ones_like(t) for t in dataset.time_arrays]
                
                # Load band data
                dataset.bands = {}
                for key in data_dict.keys():
                    if key.startswith('band_'):
                        band_name = key[5:]  # Remove 'band_' prefix
                        band_data = data_dict[key]
                        dataset.bands[band_name] = [band_data[i] for i in range(n_chunks)]
            else:
                logger.warning("Dataset contains no chunks")
                dataset.time_arrays = []
                dataset.event_labels = []
                dataset.day_night_labels = []
                dataset.bands = {}
            
            dataset.metadata = metadata
            
            logger.info(f"Loaded bandpower dataset from {filepath} with {len(dataset)} chunks and bands: {list(dataset.bands.keys())}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def __len__(self):
        """Return number of chunks"""
        return len(self.time_arrays)
    
    def get_batched_data(self, start_idx: int = 0, num_chunks: int = None, 
                        batch_size: int = None, transpose: bool = False, 
                        band_name: str = None, chunks_per_batch_item: int = 1) -> Tuple[List, List, List]:
        """
        Get concatenated bandpower data with optional batching.
        
        Args:
            start_idx: Index of first chunk to include
            num_chunks: Number of chunks to include (None for all available from start_idx)
            batch_size: If specified, shape output as batch of items
            transpose: Whether to transpose from (Windows x Channel) to (Channel x Windows)
                      Default is False, which returns (Windows x Channel)
            band_name: Which frequency band to use (if None, uses first available)
            chunks_per_batch_item: Number of chunks to combine into each batch item
            
        Returns:
            Tuple of (data, time_arrays, event_labels) as lists of arrays
        """
        if not self.bands:
            logger.warning("No band data available in BandpowerDataset")
            return [], [], []
        
        # Use specified band or first available
        if band_name is None or band_name not in self.bands:
            band_name = list(self.bands.keys())[0]
            logger.info(f"Using band '{band_name}' for data")
        
        # Determine how many chunks to process
        end_idx = len(self.time_arrays) if num_chunks is None else min(start_idx + num_chunks, len(self.time_arrays))
        chunks_to_process = list(range(start_idx, end_idx))
        
        if not chunks_to_process:
            logger.warning(f"No valid chunks in range start_idx={start_idx}, num_chunks={num_chunks}")
            return [], [], []
        
        # Calculate number of batch items needed
        total_batch_items = len(chunks_to_process) // chunks_per_batch_item
        
        # Initialize batch arrays - these will be our final output
        batch_data = []
        batch_times = []
        batch_events = []
        
        # Process batch items
        for item_idx in range(total_batch_items):
            # Get chunks for this batch item
            start_chunk = start_idx + (item_idx * chunks_per_batch_item)
            end_chunk = start_chunk + chunks_per_batch_item
            item_chunks = list(range(start_chunk, min(end_chunk, len(self.time_arrays))))
            
            # Skip if no valid chunks
            if not item_chunks:
                continue
            
            # Concatenate chunks for this batch item
            item_data_chunks = [self.bands[band_name][j] for j in item_chunks]
            item_time_chunks = [self.time_arrays[j] for j in item_chunks]
            item_event_chunks = [self.event_labels[j] for j in item_chunks]
            
            item_data = np.concatenate(item_data_chunks, axis=0)
            item_data = item_data.astype(np.float32) # Ensure float32 dtype
            item_time = np.concatenate(item_time_chunks)
            item_events = np.concatenate(item_event_chunks)
            
            # Transpose if needed
            if transpose:
                item_data = item_data.T
                
            # Add directly to the batch lists
            batch_data.append(item_data)
            batch_times.append(item_time)
            batch_events.append(item_events)
        
        if not batch_data:
            logger.warning("No valid batch items created")
            return [], [], []
            
        # Return the batch lists
        return batch_data, batch_times, batch_events


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
        
        try:
            # Check if all segments have the same shape
            if self.data:
                shapes = [segment.shape for segment in self.data]
                first_shape = shapes[0]
                
                # If segments have different shapes, log a warning
                if not all(shape == first_shape for shape in shapes):
                    logger.warning(f"Segments have inconsistent shapes: {set(shapes)}")
                    logger.info("Converting data to object arrays to handle inconsistent shapes")
            
            # Combine data for saving - use dtype=object to handle different shapes
            data_array = np.array(self.data, dtype=object)
            abs_time_array = np.array(self.time_arrays, dtype=object)
            rel_time_array = np.array(self.rel_time_arrays, dtype=object)
            event_labels_array = np.array(self.event_labels, dtype=object)
            day_night_array = np.array(self.day_night_labels, dtype=object)
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
        
        except Exception as e:
            logger.error(f"Error saving event segment dataset: {e}")
            logger.error("Trying alternative save method...")
            
            try:
                # Alternative save method: save each segment separately
                save_dict = {
                    'metadata': pickle.dumps(self.metadata),
                    'n_segments': len(self.data)
                }
                
                # Save each segment separately
                for i in range(len(self.data)):
                    save_dict[f'data_{i}'] = self.data[i]
                    save_dict[f'abs_time_{i}'] = self.time_arrays[i]
                    save_dict[f'rel_time_{i}'] = self.rel_time_arrays[i]
                    save_dict[f'events_{i}'] = self.event_labels[i]
                    save_dict[f'day_night_{i}'] = self.day_night_labels[i]
                    save_dict[f'source_{i}'] = self.event_sources[i]
                
                # Save as npz
                np.savez(filepath, **save_dict)
                logger.info(f"Saved event segment dataset using alternative method to {filepath} with {len(self.data)} segments")
            
            except Exception as e2:
                logger.error(f"Alternative save method also failed: {e2}")
                import traceback
                traceback.print_exc()
                raise
    
    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        try:
            # Load the npz file
            data_dict = np.load(filepath, allow_pickle=True)
            
            # Extract metadata
            metadata = pickle.loads(data_dict['metadata'])
            
            # Create dataset instance
            dataset = cls(metadata['config'])
            
            # Check if this is the alternative format (with individual segments)
            if 'n_segments' in data_dict:
                # Load using alternative format
                n_segments = int(data_dict['n_segments'])
                logger.info(f"Loading dataset using alternative format with {n_segments} segments")
                
                dataset.data = []
                dataset.time_arrays = []
                dataset.rel_time_arrays = []
                dataset.event_labels = []
                dataset.day_night_labels = []
                dataset.event_sources = []
                
                for i in range(n_segments):
                    dataset.data.append(data_dict[f'data_{i}'])
                    dataset.time_arrays.append(data_dict[f'abs_time_{i}'])
                    dataset.rel_time_arrays.append(data_dict[f'rel_time_{i}'])
                    dataset.event_labels.append(data_dict[f'events_{i}'])
                    dataset.day_night_labels.append(data_dict[f'day_night_{i}'])
                    if f'source_{i}' in data_dict:
                        dataset.event_sources.append(data_dict[f'source_{i}'])
                    else:
                        # Handle missing source info
                        dataset.event_sources.append({
                            'file': 'unknown',
                            'segment_idx': i,
                            'shape': dataset.data[i].shape
                        })
            else:
                # Extract segment data using original format
                data_array = data_dict['data']
                
                # Use correct field names to match what's saved in save()
                abs_time_array = data_dict['abs_time']
                rel_time_array = data_dict['rel_time']
                event_array = data_dict['events']
                sources_array = data_dict['sources']
                
                # Handle potentially empty arrays
                n_segments = len(data_array) if hasattr(data_array, '__len__') else 0
                
                if n_segments > 0:
                    try:
                        dataset.data = [data_array[i] for i in range(n_segments)]
                        dataset.time_arrays = [abs_time_array[i] for i in range(n_segments)]
                        dataset.rel_time_arrays = [rel_time_array[i] for i in range(n_segments)]
                        dataset.event_labels = [event_array[i] for i in range(n_segments)]
                        dataset.event_sources = [sources_array[i] for i in range(n_segments)]
                    except Exception as e:
                        logger.error(f"Error converting arrays to lists: {e}")
                        # Try alternative access in case arrays have inconsistent shapes
                        dataset.data = list(data_array)
                        dataset.time_arrays = list(abs_time_array)
                        dataset.rel_time_arrays = list(rel_time_array)
                        dataset.event_labels = list(event_array)
                        dataset.event_sources = list(sources_array)
                    
                    # Handle backward compatibility for day_night labels
                    if 'day_night' in data_dict:
                        day_night_array = data_dict['day_night']
                        try:
                            dataset.day_night_labels = [day_night_array[i] for i in range(n_segments)]
                        except Exception as e:
                            logger.error(f"Error converting day/night arrays: {e}")
                            dataset.day_night_labels = list(day_night_array)
                    else:
                        logger.warning("No day/night labels found in dataset, using default (all day)")
                        dataset.day_night_labels = [np.ones_like(t) for t in dataset.time_arrays]
                    
                    # Get segment info from metadata if available
                    if 'segment_files' in metadata:
                        dataset.segment_files = metadata['segment_files']
                    if 'segment_indices' in metadata:
                        dataset.segment_indices = metadata['segment_indices']
                else:
                    logger.warning("Dataset contains no segments")
                    dataset.data = []
                    dataset.time_arrays = []
                    dataset.rel_time_arrays = []
                    dataset.event_labels = []
                    dataset.day_night_labels = []
                    dataset.event_sources = []
            
            dataset.metadata = metadata
            
            # Verify data integrity
            if dataset.data:
                logger.info(f"Loaded event segment dataset shapes:")
                shape_counts = {}
                for i, d in enumerate(dataset.data):
                    shape = d.shape
                    if shape in shape_counts:
                        shape_counts[shape] += 1
                    else:
                        shape_counts[shape] = 1
                
                logger.info(f"  Shape distribution: {shape_counts}")
            
            logger.info(f"Loaded event segment dataset from {filepath} with {len(dataset)} segments")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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
    
    def get_batched_data(self, start_idx: int = 0, num_chunks: int = None, 
                        batch_size: int = None, transpose: bool = False,
                        chunks_per_batch_item: int = 1, use_relative_time: bool = True) -> Tuple[List, List, List]:
        """
        Get concatenated or batched event segment data.
        
        Args:
            start_idx: Index of first segment to include
            num_chunks: Number of segments to include (None for all available from start_idx)
            batch_size: If specified, shape output as batch of items
            transpose: Whether to transpose from (Channel x Time) to (Time x Channel)
                      Default is False, which returns (Time x Channel) when concatenated
            chunks_per_batch_item: Number of chunks/segments to combine into each batch item
            use_relative_time: Whether to use relative time (True) or absolute time (False)
            
        Returns:
            Tuple of (data, time_arrays, event_labels) as lists of arrays
        """
        if not self.data:
            logger.warning("No data available in EventSegmentDataset")
            return [], [], []
        
        # Determine how many segments to process
        end_idx = len(self.data) if num_chunks is None else min(start_idx + num_chunks, len(self.data))
        segments_to_process = list(range(start_idx, end_idx))
        
        if not segments_to_process:
            logger.warning(f"No valid segments in range start_idx={start_idx}, num_chunks={num_chunks}")
            return [], [], []
        
        # Calculate number of batch items needed
        total_batch_items = len(segments_to_process) // chunks_per_batch_item
        
        # Initialize batch arrays - these will be our final output
        batch_data = []
        batch_times = []
        batch_events = []
        
        # Process batch items
        for item_idx in range(total_batch_items):
            # Get segments for this batch item
            start_segment = start_idx + (item_idx * chunks_per_batch_item)
            end_segment = start_segment + chunks_per_batch_item
            item_segments = list(range(start_segment, min(end_segment, len(self.data))))
            
            # Skip if no valid segments
            if not item_segments:
                continue
            
            # Concatenate segments for this batch item
            item_data_segments = [self.data[j] for j in item_segments]
            # Use relative or absolute time based on parameter
            item_time_segments = [self.rel_time_arrays[j] if use_relative_time else self.time_arrays[j] for j in item_segments]
            item_event_segments = [self.event_labels[j] for j in item_segments]
            
            item_data = np.concatenate(item_data_segments, axis=1)
            item_data = item_data.astype(np.float32) # Ensure float32 dtype
            item_time = np.concatenate(item_time_segments)
            item_events = np.concatenate(item_event_segments)
            
            # Transpose if needed
            if transpose:
                item_data = item_data.T
                
            # Add directly to the batch lists
            batch_data.append(item_data)
            batch_times.append(item_time)
            batch_events.append(item_events)
        
        if not batch_data:
            logger.warning("No valid batch items created")
            return [], [], []
            
        # Return the batch lists
        return batch_data, batch_times, batch_events

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
        
        # If any events in window, find the most important one (prioritize type 3, then 2, then 1)
        if window.size > 0 and np.any(window > 0):
            if np.any(window == 3):
                window_events[i] = 3  # Both bands (highest priority)
            elif np.any(window == 2):
                window_events[i] = 2  # High band
            else:
                window_events[i] = 1  # Low band
    
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
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
            
            # Calculate length parameters in samples
            fixed_segment_length_samples = int(fixed_segment_length_ms * sfreq / 1000)
            
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
                data, 
                sfreq,
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
                data, 
                sfreq,
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
                segment = data[:, segment_start:segment_end]
                
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
                        sfreq,
                        reference_time=None  # Use file timestamp as reference
                    )
                    
                    # Adjust the time to start from the segment's start sample
                    start_time_offset = start_idx / sfreq
                    abs_time = abs_time + start_time_offset
                    
                    # Create relative time array starting from 0
                    rel_time = np.arange(segment_length) / sfreq
                    
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
    file_paths = glob(os.path.join(data_dir, "*.fif"))
    
    if not file_paths:
        logger.error(f"No files found in {data_dir}")
        return raw_dataset, bandpower_dataset, event_segment_dataset
    
    # Sort files chronologically by actual timestamp in the filename
    # instead of lexicographically
    files_with_timestamps = []
    
    for file_path in file_paths:
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
    
    # Log the first few files to verify sorting
    for i in range(min(5, len(files))):
        try:
            timestamp = parse_timestamp_from_filename(files[i])
            logger.info(f"File {i+1}: {os.path.basename(files[i])} - Timestamp: {timestamp}")
        except Exception:
            logger.info(f"File {i+1}: {os.path.basename(files[i])}")
    
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
                    
                    # Extract chunk events from the whole-file event arrays - use combined events
                    # IMPORTANT: Make sure to correctly slice the events array
                    chunk_events = np.copy(file_combined_events[start_sample:end_sample])
                    
                    # Debug event counts in chunk - this helps diagnose if events are being lost
                    chunk_event_regions = find_contiguous_events(chunk_events)
                    chunk_event_counts = count_distinct_events(chunk_events)
                    total_chunk_events = sum(chunk_event_counts.values())
                    logger.debug(f"  Chunk {chunk_idx+1}: Event sum = {np.sum(chunk_events > 0)}, Distinct events = {total_chunk_events}")
                    
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
                        
                        # Create a time mapping from original to resampled
                        orig_times = np.linspace(0, chunk_duration, len(chunk_events))
                        resampled_times_rel = np.linspace(0, chunk_duration, n_samples_resampled) # Relative to chunk start
                        dt_resampled = chunk_duration / n_samples_resampled
                        
                        # --- Improved Event Mapping Logic --- 
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
                                # else: remains 0 (no event)
                        # --- End Improved Event Mapping --- 

                        # --- Original Nearest Neighbor Mapping (commented out) ---
                        # # For each type of event (1, 2, 3), find where it occurs and map to resampled timeline
                        # for event_type in [1, 2, 3]:
                        #     # Find where this event type occurs in original data
                        #     event_indices = np.where(chunk_events == event_type)[0]
                        #     
                        #     if len(event_indices) > 0:
                        #         # For each event, map to nearest timepoint in resampled data
                        #         for idx in event_indices:
                        #             # Get time of this event in seconds from chunk start
                        #             event_time = orig_times[idx]
                        #             
                        #             # Find nearest time in resampled timeline
                        #             nearest_idx = np.argmin(np.abs(resampled_times_rel - event_time))
                        #             
                        #             # Set the event in resampled data, prioritizing higher types
                        #             if 0 <= nearest_idx < len(resampled_events):
                        #                 resampled_events[nearest_idx] = max(resampled_events[nearest_idx], event_type)
                        # --- End Original Mapping --- 

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
            
            # Log total counts of each event type
            if len(event_segment_dataset.event_labels) > 0:
                type1_count = sum(np.sum(labels == 1) for labels in event_segment_dataset.event_labels)
                type2_count = sum(np.sum(labels == 2) for labels in event_segment_dataset.event_labels)
                type3_count = sum(np.sum(labels == 3) for labels in event_segment_dataset.event_labels)
                logger.info(f"Event type distribution: {type1_count} type 1 (low band), {type2_count} type 2 (high band), {type3_count} type 3 (both bands)")
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