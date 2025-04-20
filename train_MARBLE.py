#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yaml
import torch
import pickle
from glob import glob
import logging

import MARBLE
from MARBLE import postprocessing, plotting

from data_processing import *
from dataset_constructor import RawDataset, BandpowerDataset, EventSegmentDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("marble_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("./datasets/MARBLE", exist_ok=True)
os.makedirs("./models", exist_ok=True)

def prepare_marble_data(dataset, band_name='gamma', batch_number=100, chunk_per_batch=10, time_label=False):
    """
    Prepare data for MARBLE training from a dataset
    
    Args:
        dataset: The dataset object (RawDataset, BandpowerDataset, or EventSegmentDataset)
        band_name: Band name for bandpower dataset (only used for BandpowerDataset)
        batch_number: Number of batches to create
        chunk_per_batch: Number of chunks per batch item
        time_label: If True, use batch time as label instead of batch events
        
    Returns:
        anchor_list, vector_list, labels_list for MARBLE training
    """
    if isinstance(dataset, BandpowerDataset) and band_name is not None:
        # For bandpower dataset, we need to specify the band
        batch_data, batch_time, batch_events = dataset.get_batched_data(
            band_name=band_name,
            start_idx=0,
            num_chunks=batch_number*chunk_per_batch,
            batch_size=batch_number,
            chunks_per_batch_item=chunk_per_batch
        )
    else:
        # For raw or event segment dataset
        batch_data, batch_time, batch_events = dataset.get_batched_data(
            start_idx=0, 
            num_chunks=batch_number*chunk_per_batch,
            batch_size=batch_number,
            chunks_per_batch_item=chunk_per_batch
        )
    
    # Process each batch separately
    pos_list_batched = []
    x_list_batched = []
    labels_batched = []

    # Process each batch separately
    for batch_idx in range(len(batch_data)):
        # Get data and events for this batch
        data = batch_data[batch_idx]  # (Time, Channels)
        
        if time_label:
            # Use time as labels (make it relative to first time point)
            time = batch_time[batch_idx]  # (Time)
            relative_time = time - time[0]  # Make relative to first time point
            labels = relative_time
        else:
            # Use events as labels
            labels = batch_events[batch_idx]  # (Time)
        
        # Create position and vector for this batch
        pos = data[:-1, :]  # (Time-1, Channels)
        x = np.diff(data, axis=0)  # (Time-1, Channels)
        
        # Process labels - trim last time point to match position/vector shapes
        batch_labels = labels[:-1]
        
        # Add to lists
        pos_list_batched.append(pos)
        x_list_batched.append(x)
        labels_batched.append(batch_labels)
    
    return pos_list_batched, x_list_batched, labels_batched

def train_marble_model(anchor_list, vector_list, labels_list, config, dataset_type):
    """
    Train a MARBLE model on the given data
    
    Args:
        anchor_list: List of position data
        vector_list: List of vector data  
        labels_list: List of labels
        config: Configuration dictionary
        dataset_type: Type of dataset ('raw', 'bandpower', or 'event_segments')
        
    Returns:
        Trained MARBLE model
    """
    # Extract parameters from config
    marble_params = config['marble_params']
    graph_params = config['graph_params']
    output_dirs = config['output_dirs']
    batch_number = config['batch_number']
    chunk_per_batch = config['chunk_per_batch']
    device = config.get('device', 'cuda:0')
    
    # Extract early stopping patience parameter if present
    if 'patience' in config:
        marble_params['patience'] = config['patience']
    
    # Check if we should load a pre-constructed dataset
    prebuilt_dataset_path = config.get('prebuilt_dataset_path', None)
    
    # Construct dataset save path with more information
    dataset_save_path = os.path.join(
        output_dirs['dataset'], 
        f"marble_{dataset_type}_b{batch_number}_c{chunk_per_batch}_dataset.pkl"
    )
    model_save_path = os.path.join(
        output_dirs['model'], 
        f"marble_{dataset_type}_b{batch_number}_c{chunk_per_batch}_model.pkl"
    )
    
    # Either load or construct MARBLE dataset
    if prebuilt_dataset_path and os.path.exists(prebuilt_dataset_path):
        logger.info(f"Loading pre-constructed MARBLE dataset from {prebuilt_dataset_path}")
        try:
            with open(prebuilt_dataset_path, 'rb') as f:
                marble_dataset = pickle.load(f)
                logger.info(f"Successfully loaded pre-constructed MARBLE dataset")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    else:
        # Construct MARBLE dataset
        logger.info(f"Constructing MARBLE dataset for {dataset_type} data...")
        try:
            marble_dataset = MARBLE.construct_dataset(
                anchor=anchor_list,
                vector=vector_list,
                label=labels_list,
                graph_type=graph_params['graph_type'],
                k=graph_params['k'],
                Sampling=graph_params['sampling'],
                number_of_eigenvectors=graph_params['number_of_eigenvectors']
            )
            
            # Save dataset
            with open(dataset_save_path, 'wb') as f:
                pickle.dump(marble_dataset, f)
            logger.info(f"MARBLE dataset saved to {dataset_save_path}")
        except Exception as e:
            logger.error(f"Error constructing dataset: {e}")
            return None
    
    # Train MARBLE model
    logger.info(f"Training MARBLE model on {dataset_type} data...")
    try:
        # Set the specified device
        marble_params['device'] = device
        logger.info(f"Using device: {device}")
            
        model = MARBLE.net(marble_dataset, params=marble_params)
        model.fit(marble_dataset)
        
        # Move model back to CPU before saving
        model = model.cpu()
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_save_path}")
        
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

def main():
    """Main function to load datasets and train MARBLE models"""
    # Load configuration
    with open('train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dataset type from config
    dataset_type = config.get('dataset_type', 'raw')
    if dataset_type not in ['raw', 'bandpower', 'event_segments']:
        logger.error(f"Invalid dataset type: {dataset_type}. Must be one of: raw, bandpower, event_segments")
        return
    
    # Check if we're using a prebuilt dataset
    prebuilt_dataset_path = config.get('prebuilt_dataset_path', None)
    if prebuilt_dataset_path and os.path.exists(prebuilt_dataset_path):
        logger.info(f"Using pre-constructed MARBLE dataset: {prebuilt_dataset_path}")
        # Pass empty lists to train_marble_model since they won't be used
        model = train_marble_model(
            [], [], [], 
            config, dataset_type
        )
        if model:
            logger.info(f"MARBLE model training completed successfully for pre-constructed dataset")
        return
    
    # Process raw data if no prebuilt dataset is specified
    # Get dataset path from config
    dataset_path = config['dataset_paths'].get(dataset_type)
    if not dataset_path:
        logger.error(f"No dataset path specified for type: {dataset_type}")
        return
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return
    
    # Extract parameters
    batch_number = config['batch_number']
    chunk_per_batch = config['chunk_per_batch']
    time_label = config.get('time_label', False)
    
    logger.info(f"Processing {dataset_type} dataset: {os.path.basename(dataset_path)}")
    
    # Process based on dataset type
    if dataset_type == 'raw':
        dataset = RawDataset.load(dataset_path)
        anchor_list, vector_list, labels_list = prepare_marble_data(
            dataset, 
            batch_number=batch_number, 
            chunk_per_batch=chunk_per_batch,
            time_label=time_label
        )
        
    elif dataset_type == 'bandpower':
        dataset = BandpowerDataset.load(dataset_path)
        
        # Get band from config or use default
        band_name = config.get('band_name', 'gamma')

        logger.info(f"Using band: {band_name}")
        
        anchor_list, vector_list, labels_list = prepare_marble_data(
            dataset, 
            band_name=band_name,
            batch_number=batch_number, 
            chunk_per_batch=chunk_per_batch,
            time_label=time_label
        )
        
    elif dataset_type == 'event_segments':
        dataset = EventSegmentDataset.load(dataset_path)
        anchor_list, vector_list, labels_list = prepare_marble_data(
            dataset, 
            batch_number=batch_number, 
            chunk_per_batch=chunk_per_batch,
            time_label=time_label
        )
    
    # Check if data was prepared successfully
    logger.info(f"Created {len(anchor_list)} batches of {dataset_type} data")
    if len(anchor_list) == 0:
        logger.error(f"No data batches created for {dataset_type} dataset")
        return
    
    logger.info(f"Data shapes - Anchor: {anchor_list[0].shape}, Vector: {vector_list[0].shape}, Labels: {labels_list[0].shape}")
    
    # Train model
    model = train_marble_model(
        anchor_list, vector_list, labels_list, 
        config, dataset_type
    )
    if model:
        logger.info(f"MARBLE model training completed successfully for {dataset_type} dataset")

if __name__ == "__main__":
    # Create results directory for plots
    os.makedirs("results", exist_ok=True)
    main() 