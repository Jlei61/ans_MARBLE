# iEEG Dataset Construction for MARBLE

This dataset construction system creates structured datasets from iEEG data for use with MARBLE. The system supports creating three types of datasets:

1. **Raw Dataset**: Contains raw iEEG signals with optional resampling and day/night labels
2. **Bandpower Dataset**: Contains frequency band power features extracted using sliding windows
3. **Event Segment Dataset**: Contains only segments around detected high-frequency events with multi-class event labels

## Features

- Configurable dataset parameters via YAML configuration
- Multiple frequency band definitions
- Day/night cycle labeling based on time of day
- Multi-class event detection (80-250Hz and 250-451Hz bands)
- Relative time tracking for event segments
- Optional resampling for raw data
- Processing data in manageable chunks to handle memory constraints
- Comprehensive metadata tracking
- Targeted extraction of segments containing high-frequency events

## Usage

### 1. Set up the configuration file

Edit the `dataset_config.yaml` file to set your desired parameters:

```bash
# Edit the configuration
nano dataset_config.yaml
```

### 2. Run the dataset constructor

```bash
python dataset_constructor.py [config_file_path]
```

If no config file path is provided, it will use `dataset_config.yaml` in the current directory.

### 3. Use the generated datasets

The constructor will create:
- `TIMESTAMP_raw_dataset.npz`: Raw iEEG data with time arrays, event labels, and day/night labels
- `TIMESTAMP_bandpower_dataset.npz`: Bandpower features with time arrays and labels
- `TIMESTAMP_event_segments_dataset.npz`: Segments containing high-frequency events with multi-class labels

## Loading a Dataset

```python
from dataset_constructor import RawDataset, BandpowerDataset, EventSegmentDataset

# Load a raw dataset
raw_dataset = RawDataset.load("./datasets/20240820_123456_raw_dataset.npz")

# Load a bandpower dataset
bandpower_dataset = BandpowerDataset.load("./datasets/20240820_123456_bandpower_dataset.npz")

# Load an event segment dataset
event_dataset = EventSegmentDataset.load("./datasets/20240820_123456_event_segments_dataset.npz")

# Access raw data
for i in range(len(raw_dataset)):
    chunk_data = raw_dataset.data[i]             # Channel x Time
    chunk_time = raw_dataset.time_arrays[i]      # Time
    chunk_events = raw_dataset.event_labels[i]   # Time
    chunk_day_night = raw_dataset.day_night_labels[i]  # Time (0=night, 1=day)

# For bandpower datasets
for i in range(len(bandpower_dataset)):
    gamma_power = bandpower_dataset.bands['gamma'][i]      # Windows x Channels
    chunk_time = bandpower_dataset.time_arrays[i]          # Windows
    chunk_events = bandpower_dataset.event_labels[i]       # Windows
    chunk_day_night = bandpower_dataset.day_night_labels[i]  # Windows (0=night, 1=day)

# Format bandpower data for MARBLE (Channel x Time format)
marble_ready_data = bandpower_dataset.get_data_for_marble('gamma')  # List of (Channel x Windows)

# For event segment datasets
for i in range(len(event_dataset)):
    segment_data = event_dataset.data[i]              # Channel x Time
    segment_abs_time = event_dataset.time_arrays[i]   # Absolute Time
    segment_rel_time = event_dataset.rel_time_arrays[i]  # Relative Time (from segment start)
    segment_events = event_dataset.event_labels[i]    # Event class labels (1=low band, 2=high band, 3=both)
    segment_day_night = event_dataset.day_night_labels[i]  # Day/night labels (0=night, 1=day)
    source_info = event_dataset.event_sources[i]      # Source metadata

# Get concatenated event segments
data, time, events, day_night, boundaries = event_dataset.get_concatenated_data(use_relative_time=True)
```

## Data Structure

### Raw Dataset

- `data`: List of data chunks, each with shape (Channel, Time)
- `time_arrays`: List of time arrays, each with shape (Time)
- `event_labels`: List of event label arrays, each with shape (Time)
- `day_night_labels`: List of day/night label arrays, each with shape (Time) - 0=night, 1=day
- `metadata`: Dictionary containing dataset information including resampling parameters

### Bandpower Dataset

- `bands`: Dictionary of band_name → List of data chunks, each with shape (Windows, Channels)
- `time_arrays`: List of time arrays, each with shape (Windows)
- `event_labels`: List of event label arrays, each with shape (Windows)
- `day_night_labels`: List of day/night label arrays, each with shape (Windows) - 0=night, 1=day
- `metadata`: Dictionary containing dataset information, including band definitions
- `get_data_for_marble()`: Method to get data in Channel x Time format for MARBLE

### Event Segment Dataset

- `data`: List of event segments, each with shape (Channel, Time)
- `time_arrays`: List of absolute time arrays, each with shape (Time)
- `rel_time_arrays`: List of relative time arrays, each with shape (Time)
- `event_labels`: List of multi-class event labels, each with shape (Time) - 0=none, 1=low band, 2=high band, 3=both
- `day_night_labels`: List of day/night label arrays, each with shape (Time) - 0=night, 1=day
- `event_sources`: List of dictionaries with source information (file, event indices, timestamps)
- `metadata`: Dictionary containing dataset information, including event parameters
- `get_concatenated_data()`: Method to get all segments as a single concatenated array

## Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Directory containing iEEG .fif files |
| `max_samples` | Maximum samples per chunk |
| `batch_size` | Number of chunks to create (-1 for unlimited) |
| `max_files` | Maximum number of files to read (null for all) |
| `save_dir` | Directory to save datasets |
| `create_raw_dataset` | Whether to create raw dataset |
| `create_bandpower_dataset` | Whether to create bandpower dataset |
| `create_event_segment_dataset` | Whether to create event segment dataset |
| `max_event_segments` | Maximum number of event segments to extract |
| `reference_time` | Reference time for absolute timing |
| `raw_dataset` | Raw dataset specific parameters (resampling, day/night) |
| `event_detection` | Event detection parameters |
| `event_segments` | Event segment extraction parameters |
| `bandpower` | Bandpower extraction parameters |

## Raw Dataset Parameters

| Parameter | Description |
|-----------|-------------|
| `resample_freq` | Frequency to resample to (Hz), null for no resampling |
| `night_start_hour` | Hour when night starts (24-hour format, 20 = 8PM) |
| `night_end_hour` | Hour when night ends (24-hour format, 6 = 6AM) |

## Event Detection Parameters

| Parameter | Description |
|-----------|-------------|
| `low_band` | Low frequency band for detection [min, max] Hz |
| `high_band` | High frequency band for detection [min, max] Hz |
| `rel_thresh` | Relative threshold compared to median |
| `abs_thresh` | Absolute threshold compared to global median |
| `min_gap` | Minimum gap between events in ms |
| `min_last` | Minimum duration of events in ms |

## Event Segment Parameters

| Parameter | Description |
|-----------|-------------|
| `pre_event_ms` | Time before event to include (milliseconds) |
| `post_event_ms` | Time after event to include (milliseconds) |
| `min_duration_ms` | Minimum event duration to include (milliseconds) |
| `max_segments_per_file` | Maximum segments to extract per file |
| `night_start_hour` | Hour when night starts (24-hour format, 20 = 8PM) |
| `night_end_hour` | Hour when night ends (24-hour format, 6 = 6AM) |

## Bandpower Parameters

| Parameter | Description |
|-----------|-------------|
| `window_size_ms` | Window size for bandpower calculation (milliseconds) |
| `overlap` | Overlap ratio between windows (0-1) |
| `bands` | Dictionary of band names to frequency ranges [min, max] Hz |
| `night_start_hour` | Hour when night starts (24-hour format, 20 = 8PM) |
| `night_end_hour` | Hour when night ends (24-hour format, 6 = 6AM) |

## Multi-class Event Labels

For event segment datasets, events are labeled based on which frequency band they were detected in:

| Label | Description |
|-------|-------------|
| 0 | No event detected |
| 1 | Event detected in low band only (default: 80-250 Hz) |
| 2 | Event detected in high band only (default: 250-451 Hz) |
| 3 | Event detected in both bands simultaneously |

## Notes

- Event detection uses high-frequency oscillation (HFO) detection method
- Day/night labels are based on time of day (default: 8PM-6AM = night)
- Time arrays contain absolute time values relative to the reference time
- All chunks within a dataset have the same length (max_samples)
- Shorter chunks are padded with zeros
- Event segments contain data from all channels, even if the event was only detected in one channel
- Bandpower data in (Windows x Channels) format can be transposed for MARBLE using get_data_for_marble() method 