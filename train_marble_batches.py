#!/usr/bin/env python
"""
Script to train MARBLE models on batches of EEG data from .fif files.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime
from data_processing import train_MARBLE_model, parse_timestamp_from_filename

def main():
    parser = argparse.ArgumentParser(description="Train MARBLE models on batches of EEG data")
    parser.add_argument("--data_dir", type=str, default="./preprocessed/bipolar",
                        help="Directory containing EEG .fif files")
    parser.add_argument("--max_samples", type=int, default=100000,
                        help="Maximum samples per batch")
    parser.add_argument("--resample_freq", type=float, default=200,
                        help="Frequency to resample data to")
    parser.add_argument("--k_value", type=int, default=20,
                        help="k value for MARBLE dataset construction")
    parser.add_argument("--reference_hour", type=int, default=7,
                        help="Reference hour (0-23) for time array")
    parser.add_argument("--all_batches", action="store_true",
                        help="Process all available data in batches")
    parser.add_argument("--output_dir", type=str, default="./marble_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set reference time if provided
    reference_time = None
    if args.reference_hour is not None:
        # We'll initialize with a placeholder date, which will be updated
        # based on the first file's date when processing starts
        reference_time = datetime.datetime(2000, 1, 1, args.reference_hour, 0, 0)
    
    print(f"Training MARBLE models with:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Max samples per batch: {args.max_samples}")
    print(f"  Resample frequency: {args.resample_freq} Hz")
    print(f"  Reference hour: {args.reference_hour}:00")
    print(f"  Processing all batches: {args.all_batches}")
    
    # Train MARBLE models
    models, transformed_data_list = train_MARBLE_model(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        resample_freq=args.resample_freq,
        k_value=args.k_value,
        reference_time=reference_time,
        process_all_batches=args.all_batches
    )
    
    print(f"Training complete. Trained {len(models)} models.")
    
    # Plot and save results for each batch
    for i, (model, transformed_data) in enumerate(zip(models, transformed_data_list)):
        # Plot transformed data colored by time
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            transformed_data[:, 0],
            transformed_data[:, 1],
            c=model.data.label.numpy(),
            cmap='viridis',
            alpha=0.5,
            s=5
        )
        
        # Add colorbar to show time progression
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (seconds from reference)')
        
        # Set title and labels
        ax.set_title(f'MARBLE Embedding - Batch {i}')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'marble_embedding_batch{i}.png'))
        plt.close(fig)
        
        # Save model and data
        import pickle
        with open(os.path.join(args.output_dir, f'model_batch{i}.pkl'), 'wb') as f:
            pickle.dump(model, f)
        
        # Save transformed data
        np.save(os.path.join(args.output_dir, f'transformed_data_batch{i}.npy'), transformed_data)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 