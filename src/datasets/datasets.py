import os
import pickle
import numpy as np
import datetime
from typing import Dict, List, Tuple
from data_processing import *
import logging

logger = logging.getLogger(__name__)

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

class Dataset:
    """Base class for datasets"""
    def __init__(self, config: Dict):
        self.config = config
        self.metadata = {
            "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": config
        }
        
        # Add channel names storage
        self.channel_names = []

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
        self.channel_names = []  # List of channel names
        
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
                 filenames: List[str],
                 channel_names: List[str] = None):
        """Add a chunk of data to the dataset"""
        self.data.append(data)
        self.time_arrays.append(time_array)
        self.event_labels.append(event_labels)
        self.day_night_labels.append(day_night_labels)
        self.file_indices.append(file_indices)
        if channel_names is not None:
            self.channel_names = channel_names  # Store channel names
        
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
                channel_names=np.array(self.channel_names),
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
                channel_names=np.array(self.channel_names),
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
                    'metadata': pickle.dumps(self.metadata),
                    'channel_names': np.array(self.channel_names)
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
                
                # Load channel names if available
                if 'channel_names' in data_dict:
                    dataset.channel_names = data_dict['channel_names'].tolist() if isinstance(data_dict['channel_names'], np.ndarray) else data_dict['channel_names']
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
                
                # Load channel names if available
                if 'channel_names' in data_dict:
                    dataset.channel_names = data_dict['channel_names'].tolist() if isinstance(data_dict['channel_names'], np.ndarray) else data_dict['channel_names']
            
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
            # Use relative or absolute time based on parameter
            item_time_chunks = [self.time_arrays[j] for j in item_chunks]
            item_event_chunks = [self.event_labels[j] for j in item_chunks]
            
            item_data = np.concatenate(item_data_chunks, axis=1)
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
        self.channel_names = []  # List of channel names
        
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
                 filenames: List[str],
                 channel_names: List[str] = None):
        """
        Add a chunk of bandpower data to the dataset
        
        Args:
            band_data: Dictionary of band_name -> data array (Windows x Channels)
            time_array: Array of time points (Windows)
            event_labels: Array of event labels (Windows)
            day_night_labels: Array of day/night labels (Windows)
            file_indices: List of file indices
            filenames: List of filenames used
            channel_names: List of channel names
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
        if channel_names is not None:
            self.channel_names = channel_names  # Store channel names
        
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
            'day_night': np.array(self.day_night_labels, dtype=object),
            'channel_names': np.array(self.channel_names)
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
                
                # Load channel names if available
                if 'channel_names' in data_dict:
                    dataset.channel_names = data_dict['channel_names'].tolist() if isinstance(data_dict['channel_names'], np.ndarray) else data_dict['channel_names']
                
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
                dataset.channel_names = []
            
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
        self.channel_names = []   # List of channel names
        
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

    def add_segments(self, segments_array, segment_indices, segment_files, event_types=None, channel_names=None):
        """
        Add multiple segments to the dataset
        
        Args:
            segments_array: Array of segments with shape (n_segments, n_channels, n_timepoints)
            segment_indices: List of segment indices
            segment_files: List of source files for each segment
            event_types: List of event types (1=low_band, 2=high_band, 3=both). Default is all type 1.
            channel_names: List of channel names
        """
        n_segments = segments_array.shape[0]
        
        # Store channel names if provided
        if channel_names is not None:
            self.channel_names = channel_names
        
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
                   source_info: Dict,
                   channel_names: List[str] = None):
        """Add a segment of data to the dataset"""
        self.data.append(data)
        self.time_arrays.append(abs_time_array)
        self.rel_time_arrays.append(rel_time_array)
        self.event_labels.append(event_labels)
        self.day_night_labels.append(day_night_labels)
        self.event_sources.append(source_info)
        
        # Store channel names if provided
        if channel_names is not None:
            self.channel_names = channel_names
        
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
                channel_names=np.array(self.channel_names),
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
                    'n_segments': len(self.data),
                    'channel_names': np.array(self.channel_names)
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
                
                # Load channel names if available
                if 'channel_names' in data_dict:
                    dataset.channel_names = data_dict['channel_names'].tolist() if isinstance(data_dict['channel_names'], np.ndarray) else data_dict['channel_names']
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
                    
                    # Load channel names if available
                    if 'channel_names' in data_dict:
                        dataset.channel_names = data_dict['channel_names'].tolist() if isinstance(data_dict['channel_names'], np.ndarray) else data_dict['channel_names']
                else:
                    logger.warning("Dataset contains no segments")
                    dataset.data = []
                    dataset.time_arrays = []
                    dataset.rel_time_arrays = []
                    dataset.event_labels = []
                    dataset.day_night_labels = []
                    dataset.event_sources = []
                
                # Load channel names if available
                if 'channel_names' in data_dict:
                    dataset.channel_names = data_dict['channel_names'].tolist() if isinstance(data_dict['channel_names'], np.ndarray) else data_dict['channel_names']
            
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
