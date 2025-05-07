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
import sys

# Add parent directory to path to import modules
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path) # Prioritize local src directory
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) # This line might not be necessary if imports are absolute from src
from MARBLE import postprocessing, plotting
from MARBLE import net, construct_dataset # Import specific functions

from utils.data_processing import *
from datasets.datasets_class import RawDataset, BandpowerDataset, EventSegmentDataset

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "marble_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories

def prepare_marble_data(dataset, start_chunk_idx=0, band_name='gamma', batch_number=100, chunk_per_batch=10, time_label=False, use_pca=False, n_components=None, mask_label_ratio=0.0):
    """
    Prepare data for MARBLE training from a dataset
    
    Args:
        dataset: The dataset object (RawDataset, BandpowerDataset, or EventSegmentDataset)
        start_chunk_idx: The starting index for chunks to fetch
        band_name: Band name for bandpower dataset (only used for BandpowerDataset)
        batch_number: Number of batches to create
        chunk_per_batch: Number of chunks per batch item
        time_label: If True, use batch time as label instead of batch events
        use_pca: Whether to apply PCA dimensionality reduction to channel dimension
        n_components: Number of principal components to keep (if use_pca is True)
        mask_label_ratio: Ratio of labeled points (label > 0) to mask. Default 0.0 (no masking).

    Returns:
        anchor_list, vector_list, labels_list, masks_list for MARBLE training
    """
    if isinstance(dataset, BandpowerDataset) and band_name is not None:
        # For bandpower dataset, we need to specify the band
        batch_data, batch_time, batch_events = dataset.get_batched_data(
            band_name=band_name,
            start_idx=start_chunk_idx,
            num_chunks=batch_number*chunk_per_batch,
            batch_size=batch_number,
            chunks_per_batch_item=chunk_per_batch
        )
    else:
        # For raw or event segment dataset
        batch_data, batch_time, batch_events = dataset.get_batched_data(
            start_idx=start_chunk_idx,
            num_chunks=batch_number*chunk_per_batch,
            batch_size=batch_number,
            chunks_per_batch_item=chunk_per_batch,
            transpose=True,
        )
    
    # Process each batch separately
    pos_list_batched = []
    x_list_batched = []
    labels_batched = []

    # Apply PCA if requested
    if use_pca and n_components is not None:
        from sklearn.decomposition import PCA
        logger.info(f"Applying PCA to reduce dimensions from {batch_data[0].shape[1]} to {n_components} components")
        
        # Concatenate all batch data for fitting PCA
        all_data = np.vstack([batch.reshape(-1, batch.shape[-1]) for batch in batch_data])
        
        # Fit PCA on all data
        pca = PCA(n_components=n_components)
        pca.fit(all_data)
        logger.info(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # Transform each batch separately
        batch_data_pca = []
        for batch in batch_data:
            original_shape = batch.shape
            # Reshape to 2D for PCA, then back to original shape but with reduced dimensions
            transformed = pca.transform(batch.reshape(-1, original_shape[-1]))
            batch_data_pca.append(transformed.reshape(original_shape[0], n_components))
        
        # Replace original batch data with PCA-transformed data
        batch_data = batch_data_pca

    # Process each batch separately
    for batch_idx in range(len(batch_data)):
        # Get data and events for this batch
        data = batch_data[batch_idx].astype(np.float32)  # (Time, Channels)
        print(f"Batch {batch_idx} data shape: {data.shape}")
        
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
        pos_list_batched.append(pos.astype(np.float32))
        x_list_batched.append(x.astype(np.float32))
        labels_batched.append(batch_labels.astype(np.float32)) # Store as float initially

    # Generate masks based on labels and ratio
    masks_list = []
    if mask_label_ratio > 0 and not time_label: # Only apply if ratio > 0 and not using time as labels
        logger.info(f"Generating masks with label ratio: {mask_label_ratio}")
        for labels_np in labels_batched: # Iterate through each sample's labels (still numpy)
            labels_tensor = torch.from_numpy(labels_np) # Convert to tensor for processing
            mask = torch.zeros_like(labels_tensor, dtype=torch.bool)
            # Identify indices of points with labels > 0
            # Adjust this condition if 'main features' correspond to different label values
            target_indices = torch.where(labels_tensor > 2)[0]

            if len(target_indices) > 0:
                num_to_mask = int(len(target_indices) * mask_label_ratio)
                # Ensure num_to_mask does not exceed available target indices
                num_to_mask = min(num_to_mask, len(target_indices))

                if num_to_mask > 0:
                     # Randomly select indices from the target group to mask
                    masked_indices_indices = torch.randperm(len(target_indices))[:num_to_mask]
                    masked_indices = target_indices[masked_indices_indices]
                    mask[masked_indices] = True
            masks_list.append(mask) # Add the torch bool tensor mask
        logger.info(f"Generated {len(masks_list)} label-based masks.")
    else:
        if mask_label_ratio > 0 and time_label:
             logger.warning("mask_label_ratio > 0 but time_label is True. Skipping label-based mask generation.")
        else:
             logger.info("Mask label ratio is 0 or time_label is True, skipping label-based mask generation.")
        # Create default all-False masks if ratio is 0 or time_label is True
        masks_list = [torch.zeros(len(l), dtype=torch.bool) for l in labels_batched] # Create torch bool tensor masks

    # Debug - print the shape of the anchor list (first batch)
    if len(pos_list_batched) > 0:
        logger.info(f"Anchor list shape: {pos_list_batched[0].shape}, Vector list shape: {x_list_batched[0].shape}, Labels list shape: {labels_batched[0].shape}, Mask list shape: {masks_list[0].shape if masks_list else 'N/A'}")

    return pos_list_batched, x_list_batched, labels_batched, masks_list

def train_marble_model(anchor_list, vector_list, labels_list, masks_list, config, dataset_type, iteration_num):
    """
    Train a MARBLE model on the given data
    
    Args:
        anchor_list: List of position data
        vector_list: List of vector data  
        labels_list: List of labels
        masks_list: List of masks (boolean tensors) corresponding to labels
        config: Configuration dictionary
        dataset_type: Type of dataset ('raw', 'bandpower', or 'event_segments')
        iteration_num: The sequential iteration number (for naming)
        
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
    mask_label_ratio = config.get('mask_label_ratio', 0.0) # Get mask ratio from main config
    
    # Extract early stopping patience parameter if present
    if 'patience' in config:
        marble_params['patience'] = config['patience']
    
    # Check if we should load a pre-constructed dataset
    prebuilt_dataset_path = config.get('prebuilt_dataset_path', None)
    
    # Get embedding type abbreviation
    inner_product = marble_params.get('inner_product_features', False)
    embedding_type = "EA" if inner_product else "EAG"  # EA: Embedding-Aware, EAG: Embedding-AGnostic
    
    # --- Create filename suffix for masking ---
    mask_suffix = ""
    if mask_label_ratio > 0:
        # Format ratio for filename (e.g., 0.8 -> 0p8)
        ratio_str = str(mask_label_ratio).replace('.', 'p')
        mask_suffix = f"_mask{ratio_str}"
    # --- End suffix creation ---

    # --- Create iteration suffix ---
    iter_suffix = f"_iter{iteration_num}"
    # --- End iteration suffix ---

    # Construct dataset save path with more information
    if dataset_type == 'bandpower':
        band_name = config.get('band_name', 'gamma')
        dataset_filename = f"marble_{dataset_type}_{band_name}_b{batch_number}_c{chunk_per_batch}_{embedding_type}{mask_suffix}{iter_suffix}_dataset.pkl"
        model_filename = f"marble_{dataset_type}_{band_name}_b{batch_number}_c{chunk_per_batch}_{embedding_type}{mask_suffix}{iter_suffix}_model.pkl"
    else:
        dataset_filename = f"marble_{dataset_type}_b{batch_number}_c{chunk_per_batch}_{embedding_type}{mask_suffix}{iter_suffix}_dataset.pkl"
        model_filename = f"marble_{dataset_type}_b{batch_number}_c{chunk_per_batch}_{embedding_type}{mask_suffix}{iter_suffix}_model.pkl"

    dataset_save_path = os.path.join(output_dirs['dataset'], dataset_filename)
    model_save_path = os.path.join(output_dirs['model'], model_filename)
    
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
        logger.info(f"Constructing MARBLE dataset for {dataset_type} data (mask ratio: {mask_label_ratio})...") # Log ratio used
        try:
            marble_dataset = construct_dataset( # Use imported function directly
                anchor=anchor_list,
                vector=vector_list,
                label=labels_list,
                mask=masks_list, # Pass the generated masks here
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
            
        model = net(marble_dataset, params=marble_params) # Use imported function directly
        model.fit(marble_dataset)
        
        # Ensure model is on CPU before saving
        if isinstance(model, torch.nn.Module):
            model.cpu()
        elif hasattr(model, 'to_device'): # Handle potential custom device setting methods
            model.to_device('cpu')
        
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
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'train_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dataset type from config
    dataset_type = config.get('dataset_type', 'raw')
    if dataset_type not in ['raw', 'bandpower', 'event_segments']:
        logger.error(f"Invalid dataset type: {dataset_type}. Must be one of: raw, bandpower, event_segments")
        return
    
    # Check if we're using a prebuilt dataset
    prebuilt_dataset_path = config.get('prebuilt_dataset_path', None)
    if prebuilt_dataset_path:
        # Make path absolute if it's relative
        if not os.path.isabs(prebuilt_dataset_path):
            prebuilt_dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), prebuilt_dataset_path)
            config['prebuilt_dataset_path'] = prebuilt_dataset_path
            
        if os.path.exists(prebuilt_dataset_path):
            logger.info(f"Using pre-constructed MARBLE dataset: {prebuilt_dataset_path}")
            # Pass empty lists to train_marble_model since they won't be used for construction
            # Note: If using a prebuilt dataset, the masking logic needs to have been applied
            # when that dataset was originally built. This flow assumes construction.
            model = train_marble_model(
                [], [], [], [], # Pass empty mask list as well
                config, dataset_type, 1
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
    
    # Make dataset path absolute if it's relative
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), dataset_path)
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return
    
    # Make output directories absolute if they're relative
    for key, path in config['output_dirs'].items():
        if not os.path.isabs(path):
            config['output_dirs'][key] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), path)
            # Create the directory if it doesn't exist
            os.makedirs(config['output_dirs'][key], exist_ok=True)
    
    # Extract parameters
    batch_number = config['batch_number']
    chunk_per_batch = config['chunk_per_batch']
    time_label = config.get('time_label', False)
    mask_label_ratio = config.get('mask_label_ratio', 0.0) # Extract mask ratio
    num_sequential_iterations = config.get('num_sequential_iterations', 1) # Get sequential control
    
    # Extract PCA parameters from config
    use_pca = config.get('use_pca', False)
    n_components = config.get('pca_components', None)
    
    logger.info(f"Processing {dataset_type} dataset: {os.path.basename(dataset_path)}")
    if use_pca:
        logger.info(f"Using PCA with {n_components} components")
    if mask_label_ratio > 0 and not time_label:
        logger.info(f"Masking {mask_label_ratio*100:.1f}% of labeled points (label > 0)")
    elif mask_label_ratio > 0 and time_label:
        logger.warning("mask_label_ratio > 0 but time_label is True. No label-based masking will be applied.")
    
    # Load the dataset based on type
    logger.info(f"Loading full {dataset_type} dataset from {dataset_path}...")
    try:
        if dataset_type == 'raw':
            full_dataset = RawDataset.load(dataset_path)
        elif dataset_type == 'bandpower':
            full_dataset = BandpowerDataset.load(dataset_path)
        elif dataset_type == 'event_segments':
            full_dataset = EventSegmentDataset.load(dataset_path)
        else:
             # This case should have been caught earlier, but good to double-check
            logger.error(f"Invalid dataset type encountered during loading: {dataset_type}")
            return
        logger.info(f"Dataset loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}")
        return
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_path}: {e}")
        return

    # Determine total number of chunks available
    total_chunks = 0
    if hasattr(full_dataset, 'time_arrays') and full_dataset.time_arrays is not None:
        total_chunks = len(full_dataset.time_arrays)
    elif dataset_type == 'event_segments' and hasattr(full_dataset, 'segments'):
        # EventSegmentDataset might store segments differently, adjust if needed
        # Assuming len(segments) corresponds to chunks for now
        total_chunks = len(full_dataset.segments)
        # TODO: Verify how EventSegmentDataset chunking works if different from time_arrays

    if total_chunks == 0:
        logger.error("Could not determine the number of chunks in the loaded dataset.")
        return
    logger.info(f"Total chunks available in dataset: {total_chunks}")

    # --- Sequential Training Loop ---
    current_start_chunk_idx = 0
    iteration_count = 0
    chunks_per_iteration = batch_number * chunk_per_batch

    while True:
        iteration_count += 1
        logger.info(f"--- Starting Sequential Iteration {iteration_count} ---")

        # Check if maximum iterations reached (if specified)
        if num_sequential_iterations > 0 and iteration_count > num_sequential_iterations:
            logger.info(f"Reached specified maximum of {num_sequential_iterations} iterations. Stopping.")
            break

        # Check if enough chunks remain for this iteration
        if current_start_chunk_idx + chunks_per_iteration > total_chunks:
            logger.info(f"Not enough chunks remaining ({total_chunks - current_start_chunk_idx}) for a full iteration requiring {chunks_per_iteration}. Stopping.")
            break
        
        logger.info(f"Processing chunks from index {current_start_chunk_idx} to {current_start_chunk_idx + chunks_per_iteration - 1}")

        # Prepare data for the current iteration
        anchor_list, vector_list, labels_list, masks_list = [], [], [], [] # Reset lists for each iteration
        try:
            prepare_args = {
                'dataset': full_dataset,
                'start_chunk_idx': current_start_chunk_idx,
                'batch_number': batch_number,
                'chunk_per_batch': chunk_per_batch,
                'time_label': time_label,
                'use_pca': use_pca,
                'n_components': n_components,
                'mask_label_ratio': mask_label_ratio
            }
            if dataset_type == 'bandpower':
                band_name = config.get('band_name', 'gamma')
                logger.info(f"Using band: {band_name}")
                prepare_args['band_name'] = band_name

            anchor_list, vector_list, labels_list, masks_list = prepare_marble_data(**prepare_args)
        
        except Exception as e:
            logger.error(f"Error preparing data for iteration {iteration_count}: {e}")
            break # Stop processing if data preparation fails

        # Check if data was prepared successfully for this iteration
        if len(anchor_list) == 0:
            logger.warning(f"No data batches created for iteration {iteration_count}. Skipping training for this iteration.")
            # Update start index even if skipped, to avoid infinite loop on problematic data segment
            current_start_chunk_idx += chunks_per_iteration
            continue # Move to next potential iteration
            
        logger.info(f"Iteration {iteration_count}: Created {len(anchor_list)} batches.")
        logger.info(f"Iteration {iteration_count} Data shapes - Anchor: {anchor_list[0].shape}, Vector: {vector_list[0].shape}, Labels: {labels_list[0].shape}, Masks: {masks_list[0].shape}")
        
        # Train model for the current iteration
        try:
            model = train_marble_model(
                anchor_list, vector_list, labels_list, masks_list,
                config, dataset_type, iteration_count # Pass iteration number
            )
            if model:
                logger.info(f"MARBLE model training completed successfully for iteration {iteration_count}")
            else:
                logger.error(f"MARBLE model training failed for iteration {iteration_count}. Stopping.")
                break # Stop if training fails
        except Exception as e:
            logger.error(f"Error during training for iteration {iteration_count}: {e}")
            break # Stop processing if training throws an error

        # Update start index for the next iteration
        current_start_chunk_idx += chunks_per_iteration

    logger.info(f"--- Completed Sequential Training ({iteration_count - 1} successful iterations) ---")

    # Process based on dataset type
    # if dataset_type == 'raw':
    #     dataset = RawDataset.load(dataset_path)
    #     anchor_list, vector_list, labels_list, masks_list = prepare_marble_data( # Unpack masks_list
    #         dataset,
    #         batch_number=batch_number,
    #         chunk_per_batch=chunk_per_batch,
    #         time_label=time_label,
    #         use_pca=use_pca,
    #         n_components=n_components,
    #         mask_label_ratio=mask_label_ratio # Pass mask ratio
    #     )
    
    # elif dataset_type == 'event_segments':
    #     dataset = EventSegmentDataset.load(dataset_path)
    #     anchor_list, vector_list, labels_list, masks_list = prepare_marble_data( # Unpack masks_list
    #         dataset,
    #         batch_number=batch_number,
    #         chunk_per_batch=chunk_per_batch,
    #         time_label=time_label,
    #         use_pca=use_pca,
    #         n_components=n_components,
    #         mask_label_ratio=mask_label_ratio # Pass mask ratio
    #     )
    
    # Check if data was prepared successfully
    # logger.info(f"Created {len(anchor_list)} batches of {dataset_type} data")
    # if len(anchor_list) == 0:
    #     logger.error(f"No data batches created for {dataset_type} dataset")
    #     return
    
    # logger.info(f"Data shapes - Anchor: {anchor_list[0].shape}, Vector: {vector_list[0].shape}, Labels: {labels_list[0].shape}, Masks: {masks_list[0].shape}")
    
    # Train model
    # model = train_marble_model(
    #     anchor_list, vector_list, labels_list, masks_list, # Pass masks_list
    #     config, dataset_type
    # )
    # if model:
    #     logger.info(f"MARBLE model training completed successfully for {dataset_type} dataset")

if __name__ == "__main__":
    # Create results directory for plots
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
    os.makedirs(results_dir, exist_ok=True)
    main() 