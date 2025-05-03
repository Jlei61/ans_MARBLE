import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
from scipy import signal
from scipy.signal import spectrogram


def plot_seeg_channels(fif_file, duration=10, save_path=None):
    """
    Plot SEEG channels from a .fif file with vertical separation and custom styling.
    
    Parameters:
    - fif_file: Path to the .fif file
    - duration: Duration of data to plot (in seconds)
    - save_path: Path to save the plot (optional, if None, displays plot)
    """
    # Load the .fif file
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    
    # Get sampling frequency and calculate number of samples
    sfreq = raw.info['sfreq']
    n_samples = int(duration * sfreq)
    
    # Extract data (limit to specified duration)
    data, times = raw[:, :n_samples]
    
    # Standardize data to [-1, 1]
    data_standardized = standardize_data(data)
    
    # Number of channels
    n_channels = len(raw.ch_names)
    
    # Vertical offset for each channel
    offset = 2.0
    offsets = np.arange(0, n_channels * offset, offset)
    
    # Define colors: blue tones for electrode 1, red tones for electrode 2
    # Assuming bipolar channels: 0-2 from electrode 1, 3-5 from electrode 2
    colors = []
    for i in range(n_channels):
        if i < n_channels // 2:  # First half (electrode 1)
            cmap = cm.Blues
            shade = 0.4 + (i / (n_channels / 2)) * 0.5  # Range from 0.4 to 0.9
        else:  # Second half (electrode 2)
            cmap = cm.Reds
            shade = 0.4 + ((i - n_channels / 2) / (n_channels / 2)) * 0.5
        colors.append(cmap(shade))
    
    # Create figure
    plt.figure(figsize=(24, n_channels * 1.2))
    
    # Plot each channel
    for i in range(n_channels):
        channel_data = data_standardized[i] + offsets[i]
        plt.plot(times, channel_data, color=colors[i], linewidth=0.8, alpha=0.7)
    
    # Customize plot
    plt.xlim(0, duration)  # Set x-axis from start to end
    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    plt.title(f'SEEG Data: {fif_file} (First {duration} seconds)')
    
    # Set y-ticks to channel names at their offsets
    plt.yticks(offsets, raw.ch_names)
    
    # Add a subtle grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
        
def plot_hfo_overlay(data, mask, fs, Chn_names, cmp, zoom_factor=1):
    """
    Plot SEEG data with HFO activity overlay.

    Parameters:
    - data: ndarray (channels x samples)
    - mask: List of lists where each inner list indicates start and end times of an HFO event
    - fs: Sampling frequency
    """
    n_channels, n_samples = data.shape
    time = np.arange(n_samples) / fs
    y_offsets = zoom_factor*np.arange(n_channels)

    # Plot data with offsets
    plt.figure(figsize=(25,15))
    for i in range(n_channels):
        plt.plot(time, data[i, :] + y_offsets[i], cmp[2], lw=0.5)

    # Overlay HFO activity
    for i, events in enumerate(mask):
        for event in events:
            start_time, end_time = event
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            
            # Check if indices are within data length
            start_idx = min(start_idx, n_samples - 1)
            end_idx = min(end_idx, n_samples - 1)
            
            plt.axvspan(time[start_idx], time[end_idx], color='red', 
                        ymin=(zoom_factor * (i-0.5)) / (zoom_factor * n_channels),
                        ymax=(zoom_factor * (i+0.5)) / (zoom_factor * n_channels))

    # Adjusting yticks
    tick_positions = [zoom_factor*i for i in range(n_channels)]
    tick_labels = Chn_names.tolist()

    plt.yticks(tick_positions, tick_labels, fontsize=10)
    plt.xlabel("Time (s)")
    plt.ylabel("Channels")
    plt.xlim([time[0], time[-1]])  # Assuming time is an array of time points
    plt.ylim([0-1, zoom_factor * n_channels])
    plt.title("SEEG Data with HFO Overlay")
    plt.tight_layout()
    plt.show()

def plot_hfo_event_with_signal(ax_signal, ax_spectrogram, Seeg_data, Sxx, t, f, channel, start_time, end_time, Chn_names, fs, padding=0.1, cmp='viridis'):
    """
    Plots HFO event both in raw signal and its corresponding spectrogram.

    Parameters:
    - ax_signal: Axes object for raw signal plot
    - ax_spectrogram: Axes object for spectrogram plot
    - Seeg_data: Raw signal data, shape (n_channels, n_samples)
    - Sxx: Spectrogram data, shape (n_channels, n_frequencies, n_samples)
    - t: Time array for the spectrogram
    - f: Frequency array for the spectrogram
    - channel: Channel number to be plotted
    - start_time: HFO event start time (in seconds)
    - end_time: HFO event end time (in seconds)
    - Chn_names: List of channel names
    - fs: Sampling frequency (in Hz)
    - padding: Additional time (in seconds) to be plotted around the HFO event
    - cmp: Color map for the spectrogram
    """

    # Compute time for the raw data
    n_channels, n_samples = Seeg_data.shape
    time_seeg = np.arange(n_samples) / fs
    
    # Convert time to indices for raw data
    start_idx_seeg = max(0, int((start_time - padding) * fs))
    end_idx_seeg = min(n_samples, int((end_time + padding) * fs))
    
    # Convert time to indices for spectrogram
    start_idx_sxx = np.searchsorted(t, start_time - padding)
    end_idx_sxx = np.searchsorted(t, end_time + padding)

    # Plot raw signal data
    ax_signal.plot(time_seeg[start_idx_seeg:end_idx_seeg], Seeg_data[channel, start_idx_seeg:end_idx_seeg])
    ax_signal.axvline(start_time, color='red', linestyle='--')
    ax_signal.axvline(end_time, color='red', linestyle='--')
    ax_signal.set_title(f'{Chn_names[channel]} Signal [{start_time:.2f}-{end_time:.2f}s]')

    # Plot spectrogram
    pcm = ax_spectrogram.pcolormesh(t[start_idx_sxx:end_idx_sxx], f, 10 * np.log10(Sxx[channel, :, start_idx_sxx:end_idx_sxx]), shading='gouraud', cmap=cmp)
    ax_spectrogram.axvline(start_time, color='red', linestyle='--')
    ax_spectrogram.axvline(end_time, color='red', linestyle='--')
    fig = plt.gcf()
    fig.colorbar(pcm, ax=ax_spectrogram, label='Intensity (dB)')
    ax_spectrogram.set_ylim([0, 500])  # Limit to 500Hz for visualization of HFO
    ax_spectrogram.set_title(f'{Chn_names[channel]} Spectrogram [{start_time:.2f}-{end_time:.2f}s]')

def standardize_data(data):
    """
    Standardize data to the range [-1, 1].
    """
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data)
    standardized = 2 * (data - data_min) / (data_max - data_min) - 1
    return standardized

def plot_ied_overlay(data, ied_events, fs, Chn_names, cmp=['b','r','g'], zoom_factor=1):
    """
    Plot SEEG data with IED events overlay.

    Parameters:
    - data: ndarray (channels x samples) - Raw SEEG data
    - ied_events: List of lists where each inner list indicates start and end times of an IED event
    - fs: Sampling frequency
    - Chn_names: List of channel names
    - cmp: List of colors [baseline_color, event_color, highlight_color]
    - zoom_factor: Vertical zoom factor for visualization
    """
    n_channels, n_samples = data.shape
    time = np.arange(n_samples) / fs
    y_offsets = zoom_factor * np.arange(n_channels)

    # Standardize data for better visualization
    data_std = standardize_data(data)

    # Plot data with offsets
    plt.figure(figsize=(25, 15))
    for i in range(n_channels):
        plt.plot(time, data_std[i, :] + y_offsets[i], color=cmp[0], lw=0.5)

    # Overlay IED activity with red color
    for i, events in enumerate(ied_events):
        for event in events:
            start_time, end_time = event
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            
            # Check if indices are within data length
            start_idx = min(start_idx, n_samples - 1)
            end_idx = min(end_idx, n_samples - 1)
            
            # Highlight the IED event area with a red background
            plt.axvspan(time[start_idx], time[end_idx], color=cmp[1], alpha=0.3,
                        ymin=(y_offsets[i] - 0.4) / (zoom_factor * n_channels),
                        ymax=(y_offsets[i] + 0.4) / (zoom_factor * n_channels))
            
            # Plot the IED signal portion with a different color to make it stand out
            plt.plot(time[start_idx:end_idx+1], 
                     data_std[i, start_idx:end_idx+1] + y_offsets[i], 
                     color=cmp[2], lw=1.0)

    # Adjusting yticks
    tick_positions = y_offsets
    plt.yticks(tick_positions, Chn_names, fontsize=10)
    plt.xlabel("Time (s)")
    plt.ylabel("Channels")
    plt.xlim([time[0], time[-1]])
    plt.ylim([-1, zoom_factor * n_channels])
    plt.title("SEEG Data with IED Events Overlay")
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

def plot_hfo_event(data, fs, channel, start_time, end_time, Chn_names, baseline_duration=0.5, cmap='hot'):
    """
    Plots HFO event in both time and frequency domains, with pre-event baseline.
    Uses external axes for more flexibility.
    Uses wavelet transform for time-frequency analysis.
    Subtracts the baseline to highlight event-specific changes.
    Assumes data may already be filtered for HFO detection.

    Parameters:
    - data: Raw signal data, shape (n_channels, n_samples)
    - fs: Sampling frequency (in Hz)
    - channel: Channel number to be plotted
    - start_time: HFO event start time (in seconds)
    - end_time: HFO event end time (in seconds)
    - Chn_names: List of channel names
    - baseline_duration: Duration of baseline to include before the event (in seconds)
    - cmap: Color map for the wavelet scalogram
    """
    # Create figure and axes
    fig, (ax_signal, ax_wavelet) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Compute time for the raw data
    n_channels, n_samples = data.shape
    time_seeg = np.arange(n_samples) / fs
    
    # Calculate baseline start time
    baseline_start = max(0, start_time - baseline_duration)
    
    # Convert time to indices for raw data
    baseline_idx = max(0, int(baseline_start * fs))
    start_idx = int(start_time * fs)
    end_idx = min(n_samples, int(end_time * fs))
    
    # Extract baseline and event signals
    baseline_signal = data[channel, baseline_idx:start_idx]
    
    # Define a window that includes baseline and event with some padding
    display_start_idx = baseline_idx
    display_end_idx = min(n_samples, end_idx + int(0.2 * fs))  # 0.2s after event
    
    # Extract the full segment for display
    full_segment = data[channel, display_start_idx:display_end_idx]
    full_time = time_seeg[display_start_idx:display_end_idx]
    
    # Calculate baseline mean
    baseline_mean = np.mean(baseline_signal)
    
    # Subtract baseline mean from the full segment
    baseline_corrected_signal = full_segment - baseline_mean
    
    # Plot baseline-corrected signal
    ax_signal.plot(full_time, baseline_corrected_signal)
    
    # Highlight the baseline period
    ax_signal.axvspan(baseline_start, start_time, color='lightblue', alpha=0.3, label='Baseline')
    
    # Highlight the HFO event
    ax_signal.axvspan(start_time, end_time, color='red', alpha=0.3, label='HFO Event')
    
    # Add vertical lines at event boundaries
    ax_signal.axvline(start_time, color='red', linestyle='--')
    ax_signal.axvline(end_time, color='red', linestyle='--')
    
    ax_signal.set_title(f'{Chn_names[channel]} Signal [{start_time:.2f}-{end_time:.2f}s] (Baseline Corrected)')
    ax_signal.set_xlabel('Time (s)')
    ax_signal.set_ylabel('Amplitude (baseline corrected)')
    ax_signal.legend()
    ax_signal.grid(True, linestyle='--', alpha=0.3)
    
    # Compute wavelet transform directly on baseline-corrected signal
    # Use scales optimized for higher frequencies
    widths = np.arange(1, 101)  # Wider range of scales
    cwtmatr = signal.cwt(baseline_corrected_signal, signal.morlet2, widths)
    
    # Convert scales to frequencies (approximate)
    frequencies = fs / (2 * widths)
    
    # Extract baseline and event segments for normalization
    baseline_wavelet_idx = np.where((full_time >= baseline_start) & (full_time < start_time))[0]
    event_wavelet_idx = np.where((full_time >= start_time) & (full_time <= end_time))[0]
    
    # Calculate normalized power (relative to baseline)
    normalized_cwt = np.zeros_like(cwtmatr)
    for i in range(cwtmatr.shape[0]):
        # Get baseline power for this frequency
        if len(baseline_wavelet_idx) > 0:
            baseline_power = np.mean(np.abs(cwtmatr[i, baseline_wavelet_idx]))
            if baseline_power > 0:
                # Calculate power relative to baseline (fold change)
                normalized_cwt[i, :] = np.abs(cwtmatr[i, :]) / baseline_power
            else:
                normalized_cwt[i, :] = np.abs(cwtmatr[i, :])
        else:
            normalized_cwt[i, :] = np.abs(cwtmatr[i, :])
    
    # Plot wavelet scalogram (normalized)
    pcm = ax_wavelet.pcolormesh(full_time, frequencies, normalized_cwt, 
                               cmap=cmap, shading='gouraud', 
                               vmin=0.5, vmax=5)  # Adjust scale to highlight power increases
    
    # Add vertical lines at event boundaries
    ax_wavelet.axvline(start_time, color='white', linestyle='--')
    ax_wavelet.axvline(end_time, color='white', linestyle='--')
    
    # Add colorbar
    cbar = fig.colorbar(pcm, ax=ax_wavelet)
    cbar.set_label('Power Relative to Baseline')
    
    # Focus on HFO frequency range
    min_freq = 80
    max_freq = 500
    
    # Check if we have enough high frequencies in our transform
    if np.any(frequencies >= 250):
        ax_wavelet.set_ylim([min_freq, max_freq])
    else:
        # If not, use available range
        ax_wavelet.set_ylim([min_freq, max(frequencies)])
    
    ax_wavelet.set_title(f'{Chn_names[channel]} Wavelet Scalogram [{start_time:.2f}-{end_time:.2f}s]')
    ax_wavelet.set_xlabel('Time (s)')
    ax_wavelet.set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax_signal, ax_wavelet)
