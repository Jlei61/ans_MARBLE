import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from glob import glob
import mne
import torch
import pickle

import MARBLE
from MARBLE import postprocessing, plotting, geometry

from data_processing import *

# Create directories for output if they don't exist
os.makedirs("temp_Data", exist_ok=True)
os.makedirs("temp_Figures", exist_ok=True)

# Choose data loading method
use_bandpower = True  # Set to True to use band power extraction method, False for original method
load_dataset = True  # Whether to load preprocessed dataset

# Define parameters
data_dir = "./preprocessed/bipolar"
max_samples = 10000  # Use a smaller number for demonstration
batch_size = 50  # Number of batches to create
max_files = None  # Maximum number of files to read
resample_freq = 50  # Resampling frequency in Hz (only used for original method)
save_data = True  # Whether to save preprocessed data
save_dir = "temp_Data"  # Directory to save preprocessed data
label_type = "event"

# Band power parameters (only used if use_bandpower is True)
band_range = [80, 250]  # Frequency band for extraction [low, high] in Hz
window_size_ms = 100  # Window size in milliseconds (increased from 20ms to avoid filtering issues)
overlap = 0.5  # Overlap ratio between windows


# Define flags for loading saved data/model
load_dataset = False 
load_model = False

if load_dataset:
    # Find available preprocessed datasets
    if use_bandpower:
        # Look specifically for bandpower datasets
        dataset_pattern = "*bandpower*" + str(band_range[0]) + "-" + str(band_range[1]) + "Hz_PreMARBLE_dataset.npz"
        dataset_files = find_preprocessed_datasets(save_dir, pattern=dataset_pattern)
    else:
        # Look for regular datasets (excluding bandpower ones)
        dataset_files = find_preprocessed_datasets(save_dir, pattern="*PreMARBLE_dataset.npz")
        # Filter out bandpower datasets
        dataset_files = [f for f in dataset_files if "bandpower" not in f]
    
    if dataset_files:
        # Load the most recent dataset
        batch_data, batch_times = load_preprocessed_data(dataset_files[-1])
        print(f"Loaded preprocessed data from {dataset_files[-1]}")
        used_files = []  # No file tracking needed when loading preprocessed
        next_file_idx = 0
    else:
        if use_bandpower:
            print(f"No bandpower preprocessed datasets found for band {band_range[0]}-{band_range[1]}Hz, loading from raw files...")
        else:
            print("No preprocessed datasets found, loading from raw files...")
        load_dataset = False

if not load_dataset:
    if use_bandpower:
        # Load data using band power extraction method
        print(f"Using band power extraction method with band {band_range} Hz")
        batch_data, batch_times, used_files, next_file_idx = load_bandpower_data(
            data_dir=data_dir,
            max_samples=max_samples,
            batch_size=batch_size,
            max_files=max_files,
            save_data=save_data,
            save_dir=save_dir,
            label_type=label_type,
            band_range=band_range,
            window_size_ms=window_size_ms,
            overlap=overlap,
        )
    else:
        # Load data using original method
        print("Using original data loading method")
        batch_data, batch_times, used_files, next_file_idx = load_merged_data(
            data_dir=data_dir,
            max_samples=max_samples,
            batch_size=batch_size,
            max_files=max_files,
            resample_freq=resample_freq,
            save_data=save_data,
            save_dir=save_dir,
            label_type=label_type,
        )

print(f"Loaded data shape: {batch_data.shape}")
print(f"Time array shape: {batch_times.shape}")
if not load_dataset:
    print(f"Used {len(used_files)} file(s)")
    print(f"Next file index: {next_file_idx}")

# Plot samples from first batch to check data quality
plt.figure(figsize=(12, 6))

# Update method_suffix based on actual data type
method_suffix = "bandpower" if use_bandpower else "raw"

if use_bandpower:
    plt.subplot(2, 1, 1)
    # For band power data, plot first 100 windows (or all if less than 100) for the first 3 channels
    plt_points = min(100, batch_data.shape[1])
    plt.plot(batch_data[0, :plt_points, :3])  
    plt.title(f'Band Power Data ({band_range[0]}-{band_range[1]} Hz, first {plt_points} windows, first 3 channels)')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    # For band power data, plot corresponding time points
    plt.plot(batch_times[0, :plt_points, 0] if batch_times.ndim > 2 else batch_times[0, :plt_points])
    plt.title(f'Time Array (first {plt_points} windows)')
    plt.xlabel('Window')
    plt.ylabel('Time (seconds from reference)')
else:
    plt.subplot(2, 1, 1)
    plt_points = min(1000, batch_data.shape[1])
    plt.plot(batch_data[0, :plt_points, :3])  # First batch, first 1000 samples, first 3 channels
    plt.title(f'EEG Data (first {plt_points} samples, first 3 channels)')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(batch_times[0, :plt_points, 0] if batch_times.ndim > 2 else batch_times[0, :plt_points])
    plt.title(f'Time Array (first {plt_points} samples)')
    plt.xlabel('Sample')
    plt.ylabel('Time (seconds from reference)')

plt.grid(True)
plt.tight_layout()
plt.savefig('temp_Figures/raw_data_preview.png', dpi=300, bbox_inches='tight')

# Prepare data for MARBLE
pos_list, x_list, labels = prepare_MARBLE_data(batch_data, batch_times)


# Define save paths based on method
method_suffix = "bandpower" if use_bandpower else "raw"
save_path = f'temp_Data/eeg_{method_suffix}_dataset.pkl'
model_path = f'temp_Data/eeg_{method_suffix}_model.pkl'

# Define MARBLE parameters
params = {
    "epochs": 50,  # Use fewer epochs for demonstration
    "order": 2,
    "hidden_channels": [128,64],
    "batch_size": 256,
    "lr": 1e-3,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "out_channels": 5,
    "inner_product_features": False,
    "diffusion": False,
    "batch_norm": True,
    "seed": 42,
}

# Construct or load dataset
k_value = 50
if not load_dataset:
    try:
        Dataset = MARBLE.construct_dataset(
            anchor=pos_list, 
            vector=x_list,
            label=labels,  # Use time-based labels
            graph_type="cknn",
            k=k_value,  
            Sampling=False,
            number_of_eigenvectors=500,
        )
        with open(save_path, 'wb') as f:
            pickle.dump(Dataset, f)
        print(f"Dataset saved to {save_path}")
    except Exception as e:
        print(f"Error constructing dataset: {e}")
        import sys
        sys.exit(1)
else:
    try:
        with open(save_path, 'rb') as f:
            Dataset = pickle.load(f)
        print(f"Dataset loaded from {save_path}")
    except FileNotFoundError:
        print(f"Dataset file not found: {save_path}")
        print("Please run again with load_dataset=False to create the dataset")
        import sys
        sys.exit(1)

# Train or load model
try:
    model = MARBLE.net(Dataset, params=params)

    if not load_model:
        model.fit(Dataset)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Error in model training/loading: {e}")
    import sys
    sys.exit(1)

# Transform data and visualize embeddings
try:
    transformed_data = model.transform(Dataset)
    transformed_data = postprocessing.embed_in_2D(transformed_data)
    transformed_data = MARBLE.distribution_distances(Dataset)

    # Visualize embeddings
    plt.figure(figsize=(4, 4))
    ax = plotting.embedding(transformed_data, transformed_data.y.numpy().astype(int))
    plt.title(f'MARBLE Embedding ({method_suffix} data)')
    plt.savefig(f'temp_Figures/eeg_{method_suffix}_marble_embedding.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(6.4, 4.8))
    im = plt.imshow(transformed_data.dist)
    plt.colorbar(im)
    plt.title(f'Distance Matrix ({method_suffix} data)')
    plt.savefig(f'temp_Figures/eeg_{method_suffix}_distance_matrix.png', dpi=300, bbox_inches='tight')

    # Plot different embeddings
    embed_types = ['PCA', 'tsne', 'umap', 'Isomap', 'MDS']

    for embed_typ in embed_types:
        emb, _ = geometry.embed(transformed_data.dist, embed_typ=embed_typ)
        plt.figure(figsize=(4, 4))
        ax = plotting.embedding(emb, np.array([0,1]), s=30, alpha=1)
        plt.title(f'{embed_typ} Embedding ({method_suffix} data)')
        plt.savefig(f'temp_Figures/eeg_{method_suffix}_{embed_typ.lower()}_embedding.png', dpi=300, bbox_inches='tight')
except Exception as e:
    print(f"Error in data transformation or visualization: {e}")
    import sys
    sys.exit(1)

print("=" * 40)
print(f"Results saved to temp_Figures/ directory (using {method_suffix} method)")

# If run as main script
if __name__ == "__main__":
    print("Script completed successfully")