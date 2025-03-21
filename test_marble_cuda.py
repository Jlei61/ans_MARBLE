#!/usr/bin/env python
"""
Test script to evaluate CUDA acceleration in MARBLE.
This script processes EEG data files from a single day (24_08_13)
and uses CUDA for dataset construction and processing.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
from glob import glob
from tqdm import tqdm

import MARBLE
from MARBLE import postprocessing, plotting, preprocessing

def check_cuda_availability():
    """Check if CUDA is available and print device information."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available! Using device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("CUDA is not available. Using CPU instead.")
        return False

def process_files_with_cuda(files, output_dir="marble_cuda_results", max_samples=None):
    """Process multiple EEG files using CUDA-accelerated MARBLE.
    
    Args:
        files: List of files to process
        output_dir: Directory to save results
        max_samples: Maximum number of samples to use from each file (None=all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Track timing for performance comparison
    timings = {
        'cpu_memory_efficient': [],
        'cuda_memory_efficient': [],
        'processed_files': []  # Track which files were successfully processed
    }
    
    # Process each file with progress bar
    for file_idx, file in enumerate(tqdm(files, desc="Processing files", unit="file")):
        print(f"\n[{file_idx+1}/{len(files)}] Processing file: {os.path.basename(file)}")
        
        try:
            # Load data
            raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
            data = raw.get_data().T
            
            # Apply max_samples limit if specified
            if max_samples is not None and len(data) > max_samples:
                print(f"Limiting data to {max_samples} samples (original: {len(data)})")
                data = data[:max_samples]
                
            data = (data - data.mean(axis=0)) / data.std(axis=0)
            
            # Create position and vector lists
            pos_list = [data[:-1, :]]
            x_list = [np.diff(data, axis=0)]
            
            # Prepare parameters
            params = {
                "epochs": 50,
                "order": 1,
                "hidden_channels": [32],
                "batch_size": 256,
                "lr": 1e-4,
                "out_channels": 3,
                "inner_product_features": False,
                "emb_norm": True,
                "diffusion": True,
            }
            
            # Process with memory-efficient CPU
            print("\n--- Processing with memory-efficient CPU ---")
            start_time = time.time()
            cpu_dataset = MARBLE.construct_dataset(
                anchor=pos_list, 
                vector=x_list,
                graph_type="cknn",
                spacing=0.1,
                memory_efficient=True,
                use_cuda=False,
            )
            cpu_time = time.time() - start_time
            timings['cpu_memory_efficient'].append(cpu_time)
            print(f"CPU processing time: {cpu_time:.2f} seconds")
            
            # Process with CUDA-accelerated version
            print("\n--- Processing with CUDA-accelerated version ---")
            start_time = time.time()
            with tqdm(total=100, desc="CUDA processing", unit="%") as pbar:
                cuda_dataset = MARBLE.construct_dataset(
                    anchor=pos_list, 
                    vector=x_list,
                    graph_type="cknn",
                    spacing=0.1,
                    memory_efficient=True,
                    use_cuda=True,
                )
                pbar.update(100)  # Update to 100% when done
            cuda_time = time.time() - start_time
            timings['cuda_memory_efficient'].append(cuda_time)
            timings['processed_files'].append(file)
            print(f"CUDA processing time: {cuda_time:.2f} seconds")
            print(f"Speed improvement: {cpu_time/cuda_time:.2f}x")
            
            # Save the dataset to a pickle file
            filename = os.path.basename(file).replace('.fif', '')
            dataset_path = f"{output_dir}/dataset_{filename}.pkl"
            with open(dataset_path, 'wb') as f:
                pickle.dump(cuda_dataset, f)
            print(f"Dataset saved to {dataset_path}")
            
            # Train model on CUDA dataset
            print("\n--- Training model with CUDA dataset ---")
            model = MARBLE.net(cuda_dataset, params=params)
            with tqdm(total=params["epochs"], desc="Training model", unit="epoch") as pbar:
                def update_progress(epoch, loss):
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss:.4f}")
                model.fit(cuda_dataset, progress_callback=update_progress)
            
            # Transform and process
            print("\n--- Transforming data ---")
            with tqdm(total=3, desc="Post-processing", unit="step") as pbar:
                transformed_data = model.transform(cuda_dataset)
                pbar.update(1)
                transformed_data = postprocessing.embed_in_2D(transformed_data)
                pbar.update(1)
                transformed_data = postprocessing.cluster(transformed_data, n_clusters=5)
                pbar.update(1)
            
            # Visualize and save results
            plt.figure(figsize=(10, 8))
            plotting.state_space(transformed_data)
            plt.title(f"MARBLE Analysis with CUDA: {filename}")
            plt.savefig(f"{output_dir}/marble_cuda_{filename}.png")
            plt.close()
            
            # Save model
            torch.save(model.state_dict(), f"{output_dir}/model_{filename}.pt")
            
            # Save transformation data
            np.save(f"{output_dir}/transformed_data_{filename}.npy", {
                'embeddings': transformed_data.emb,
                'clusters': transformed_data.clusters,
            })
            
            # Also save transformed data as pickle for easier loading
            with open(f"{output_dir}/transformed_data_{filename}.pkl", 'wb') as f:
                pickle.dump(transformed_data, f)
            
        except Exception as e:
            print(f"\nError processing file {file}: {str(e)}")
            continue
    
    # Plot timing comparison
    if len(timings['cpu_memory_efficient']) > 0:
        plt.figure(figsize=(10, 6))
        x = range(len(timings['cpu_memory_efficient']))
        plt.plot(x, timings['cpu_memory_efficient'], 'b-o', label='CPU Memory-Efficient')
        plt.plot(x, timings['cuda_memory_efficient'], 'r-o', label='CUDA Memory-Efficient')
        plt.xlabel("File Index")
        plt.ylabel("Processing Time (seconds)")
        plt.title("MARBLE Processing Time Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/timing_comparison.png")
        plt.close()
        
        # Calculate average speedup
        avg_speedup = np.mean(np.array(timings['cpu_memory_efficient']) / 
                             np.array(timings['cuda_memory_efficient']))
        print(f"\nAverage CUDA speedup: {avg_speedup:.2f}x")
    else:
        print("\nNo files were successfully processed for timing comparison.")
    
    return timings

def main():
    """Main function to run the test."""
    # Check CUDA availability
    cuda_available = check_cuda_availability()
    
    # Get EEG files from 24_08_13
    data_dir = "./preprocessed/bipolar"
    files = sorted(glob(os.path.join(data_dir, "24_08_13*.fif")))
    
    if not files:
        print(f"No files found matching the pattern in {data_dir}")
        return
    
    print(f"Found {len(files)} files to process:")
    for i, file in enumerate(files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Process files (only if CUDA is available, we'll still run the CPU version for comparison)
    if cuda_available:
        
        max_samples = None
        
        print(f"\nProcessing with max_samples={max_samples}" if max_samples else "\nProcessing all samples")
        timings = process_files_with_cuda(files, max_samples=max_samples)
        
        # Print summary of successful runs
        if len(timings['cpu_memory_efficient']) > 0:
            print("\n--- Processing Summary ---")
            print(f"Successfully processed files: {len(timings['processed_files'])}/{len(files)}")
            
            for i, file in enumerate(timings['processed_files']):
                cpu_time = timings['cpu_memory_efficient'][i]
                cuda_time = timings['cuda_memory_efficient'][i]
                speedup = cpu_time / cuda_time
                print(f"{os.path.basename(file)}: CPU={cpu_time:.2f}s, CUDA={cuda_time:.2f}s, Speedup={speedup:.2f}x")
    else:
        print("\nCUDA is not available. Cannot perform acceleration test.")
        print("Please make sure CUDA is installed and available to PyTorch.")

if __name__ == "__main__":
    main() 