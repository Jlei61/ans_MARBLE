# MARBLE Training

This repository contains code for training MARBLE (MAnifold-based Representation of Brain LandscapE) models on neural datasets.

## Dataset Types

The code supports three types of datasets:
- **Raw**: Raw neural signal data
- **Bandpower**: Spectral power in different frequency bands
- **Event Segments**: Data segmented around specific events

## Usage

1. Prepare your dataset using the provided dataset constructors
2. Configure training in `train_config.yaml`
3. Run the training script:

```bash
python train_marble_datasets.py
```

## Configuration

The configuration file (`train_config.yaml`) contains the following parameters:

- `dataset_type`: Which dataset to use ('raw', 'bandpower', or 'event_segments')
- `time_label`: Whether to use time as label instead of events (default: false)
- `band_name`: For bandpower datasets, specify which frequency band to use
- `device`: GPU device to use for training (e.g., 'cuda:0', 'cuda:1')
- `batch_number`: Number of batches to create
- `chunk_per_batch`: Number of chunks per batch item
- `dataset_paths`: Paths to your prepared datasets
- `output_dirs`: Directories to save outputs
- `marble_params`: Parameters for MARBLE model training
- `graph_params`: Parameters for graph construction

## Outputs

The training script generates:
- MARBLE dataset pickle file (in `output_dirs.dataset`)
- Trained model pickle file (in `output_dirs.model`)

All output filenames include information about the dataset type, batch size, and chunks per batch.

## Examples

### Training on raw data

```yaml
dataset_type: 'raw'
device: 'cuda:0'
```

### Training on bandpower data (gamma band)

```yaml
dataset_type: 'bandpower'
band_name: 'gamma'
device: 'cuda:1'
```

### Using time as label

```yaml
dataset_type: 'event_segments'
time_label: true
device: 'cuda:0'
``` 