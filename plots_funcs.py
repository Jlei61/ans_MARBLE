import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


def create_violin_df(data, label, node_names,ez_idx):
    df = pd.DataFrame({'Node': [node for node_name in node_names for node in [node_name] * len(data[node_names.index(node_name)])],
                       'Value': [val for sublist in data for val in sublist],
                       'Type': label})
    df['is_ez'] = df['Node'].isin([node_names[i] for i in ez_idx])
    return df

def feature_violinplot(df1,df2,title1,title2,cmp):
    plt.figure(figsize=(25, 12))
    # Plot for rp_durations
    plt.subplot(2,1,1)
    sns.violinplot(x='Node', y='Value',hue='is_ez', data=df1,palette=[cmp[2], cmp[0]])
    plt.title(title1)
    plt.xlabel('Contacts')
    plt.ylabel('Value')

    plt.subplot(2,1,2)
    sns.violinplot(x='Node', y='Value',hue='is_ez', data=df2,palette=[cmp[2], cmp[0]])
    plt.title(title2)
    plt.xlabel('Contacts')
    plt.ylabel('Value')


def feature_barplot(barplot_data,COI_bi_id,title,cmp):
    # Convert to DataFrame
    df_barplot = pd.DataFrame(barplot_data)

    df_barplot['is_ez'] = df_barplot['contacts_id'].isin(COI_bi_id)

    # Create the bar plot
    plt.figure(figsize=(24, 6))
    sns.barplot(x='contacts_name', y='events', hue='is_ez', data=df_barplot, palette=[cmp[2], cmp[0]])

    plt.title(title)
    plt.xlabel('Contact(#)')
    plt.ylabel('Frequency')
    plt.legend(title='EZ Index', labels=['Not in EZ', 'In EZ'])


def SEM_violinplot(nn, eta_true, ez_idx, pz_idx, eta_posterior, eta_c, delta_eta ):
    parts= plt.violinplot(eta_posterior, widths=0.7, showmeans=True, showextrema=True);
    plt.plot(np.r_[0:nn]+1,eta_true ,'o', color='k', alpha=0.9, markersize=4)
    plt.axhline(y=eta_c, linewidth=.8, color = 'r', linestyle='--')
    plt.axhline(y=eta_c-delta_eta, linewidth=.8, color = 'y', linestyle='--')
    plt.yticks(fontsize=14) 
    plt.xticks(fontsize=14) 
    #plt.xticks(np.r_[1:nn+1], np.r_[1:nn+1], rotation=90, fontsize=14)  
    #plt.xticks(np.arange(1,nn+1, step=2),np.arange(1, nn+1, step=2), fontsize=12, rotation=0)
    plt.ylabel(' Posterior ' +r'${(\eta_i)}$', fontsize=22);  
    plt.xlabel('Brain nodes', fontsize=22); 

    for pc in parts['bodies'][0:nn]:
        pc.set_facecolor('g')
        pc.set_edgecolor('g')
        pc.set_alpha(0.5)
    i = 0
    while i < len(ez_idx):
        for pc in parts['bodies'][ez_idx[i]:ez_idx[i]+1]:
            pc.set_facecolor('r')
            pc.set_edgecolor('r')
            pc.set_alpha(0.8)
        i += 1

    j = 0
    while j < len(pz_idx):
        for pc in parts['bodies'][pz_idx[j]:pz_idx[j]+1]:
            pc.set_facecolor('y')
            pc.set_edgecolor('y')
            pc.set_alpha(0.8)
        j += 1
    plt.tight_layout()
    

def plot_distribution_samples(distribution, n_samples=1000, tail_start=-8):
    """
    Plot samples from a custom distribution, with different colors for the normal and tail parts.

    :param distribution: The custom distribution object.
    :param n_samples: Number of samples to generate for plotting.
    :param tail_start: The value where the tail starts.
    """
    samples = distribution.sample((n_samples,))

    # Split samples into normal and tail parts based on tail_start
    normal_samples = samples[samples <= tail_start]
    tail_samples = samples[samples > tail_start]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(normal_samples, bins=50, color='blue', alpha=0.7, label='Normal Part')
    plt.hist(tail_samples, bins=50, color='red', alpha=0.7, label='Long Tail Part')
    plt.title('Visualization of Distribution Samples')
    plt.xlabel('Sample Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def Res_violin(data, ROI_names, ez_bar, pz_bar, cmp, title):
    # Set the figure size and title
    plt.figure(figsize=(12, 24))
    plt.title(title)

    # Determine the mean of each parameter across samples
    means = np.mean(data, axis=0)

    # Create a DataFrame for easy plotting with Seaborn
    df = pd.DataFrame(data, columns=ROI_names)

    # Melt the DataFrame for horizontal violin plots
    df_melted = df.melt(var_name='Parameter', value_name='Value')

    # Default color for all violins
    palette = [cmp[2]] * data.shape[1]

    # Adjust colors based on the mean and thresholds
    for i, mean in enumerate(means):
        if ez_bar is not None and mean > ez_bar:
            palette[i] = cmp[0]
        elif pz_bar is not None and mean > pz_bar:
            palette[i] = cmp[1]

    # Create a horizontal violin plot
    sns.violinplot(x='Value', y='Parameter', data=df_melted, palette=palette, inner=None)

    # Adding vertical lines for ez_bar and pz_bar, if they are not None
    if ez_bar is not None:
        plt.axvline(x=ez_bar, color=cmp[0], linestyle='--', label='EZ Threshold')
    if pz_bar is not None:
        plt.axvline(x=pz_bar, color=cmp[1], linestyle='--', label='PZ Threshold')

    # Show the legend if there are threshold lines
    if ez_bar is not None or pz_bar is not None:
        plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

def plot_sim_seeg(data,Chn_names,COI_id,cmp,Figure_folder,fs = 2000,T = 10):
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    nc = data.shape[0]
    times = np.arange(0, T + 1/fs, 1/fs)

    for i in range(0, nc):
        if i in COI_id:
            plt.plot(times,data[i,:]+1.5*i, cmp[0], lw=0.5)
        else:
            plt.plot(times,data[i,:]+1.5*i, cmp[2], lw=0.5)
    # Adjusting yticks
    tick_positions = [1.5*i for i in range(nc)]
    tick_labels = Chn_names

    plt.yticks(tick_positions, tick_labels, fontsize=10)

    plt.xticks(np.arange(0,11,10),fontsize=16)
    plt.title("Simulation SEEG",fontsize=18)
    plt.xlabel('Time(s)',fontsize=25)
    plt.ylabel('Electrodes(#)',fontsize=25)
    plt.ylim([0-3, 1.5*nc+5])
    plt.xlim([0, T])

    plt.savefig(Figure_folder+'SEEG_SIM.png',dpi = 300)