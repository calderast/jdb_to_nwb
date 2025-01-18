def plot_raw_ratiometric_and_565_photometry_signals(raw_green, raw_405, raw_red, raw_ratio_signal, sampling_rate, pulse_times_in_mins): 
    """
    Plots the raw 470, 405, 565 and ratiometric 470/405 fluorescence signals.
    
    Parameters
    ----------
    raw_green : array-like
        The raw 470 fluorescence signal.
    raw_405 : array-like
        The raw 405 fluorescence signal.
    raw_red : array-like
        The raw 565 fluorescence signal.
    raw_ratio_signal : array-like
        The ratiometric 470/405 fluorescence signal.
    sampling_rate : int
        The sampling rate of the photometry data.
    pulse_times_in_mins : array-like
        The times of the reward cues in minutes.
        
    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt    

    pulse_times_in_mins = [time / 60000 for time in ppd_data['pulse_times_1']]
    xvals = np.arange(0,len(raw_green))/sampling_rate/60 

    raw = plt.figure(figsize=(16, 10))
    plt.suptitle('Raw & ratiometric 470/405 fluorescence signals', fontsize=16)

    ax1 = raw.add_subplot(411) # Three integers (nrows, ncols, index).
    ax1.plot(xvals,raw_green,color='blue',lw=1.5,label='raw 470 (V)')
    ax1.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(raw_green)), label='Reward Cue', color='w', marker="|", mec='k', ms=10)

    ax1.legend()

    ax2 = raw.add_subplot(412, sharex=ax1)
    ax2.plot(xvals,raw_405,color='purple',lw=1.5,label='raw 405 (V)')

    ax2.legend()

    ax3 = raw.add_subplot(413, sharex=ax1)
    ax3.plot(xvals,raw_red,color='green',lw=1.5,label='raw 565 (V)')
    ax3.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(raw_red)), label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax3.legend()

    ax4 = raw.add_subplot(414, sharex=ax1)
    ax4.plot(xvals,raw_ratio_signal,color='black',lw=1.5,label='470/405 (V)')
    ax4.set_xlabel('time (min)')
    ax4.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(raw_ratio_signal)), label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax4.legend()

    plt.show()

    return None

def plot_trial_time_distribution(tritimes):
    """
    Plots the distribution of trial times.
    """

    import matplotlib.pyplot as plt
    tts = plt.figure()
    plt.title('Distribution of trial times')
    plt.hist(tritimes,bins=100)
    plt.ylabel('# of trials')
    plt.xlabel('trial time (s)')
    tts.show()

    return None

def plot_405_470_correlation(raw_405, raw_green):
    """
    Plots the correlation between the 405 and 470 signals.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    slope, intercept, r_value, p_value, std_err = linregress(x=raw_405, y=raw_green)
    plt.figure()
    plt.scatter(raw_405[::5], raw_green[::5],alpha=0.1, marker='.')
    x = np.array(plt.xlim())
    plt.plot(x, intercept+slope*x)
    plt.xlabel('ACh3.8 405 signal')
    plt.ylabel('ACh3.8 470 signal')
    plt.title('470 - 405 correlation.')

    # Add text box with slope and R-squared values
    textstr = f'Slope: {slope:.3f}\nR-squared: {r_value**2:.3f}'
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()
    return None

def plot_405_565_correlation(raw_405, raw_red):
    """
    Plots the correlation between the 405 and 565 signals.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    slope, intercept, r_value, p_value, std_err = linregress(x=raw_405, y=raw_red)
    plt.figure()
    plt.scatter(raw_405[::5], raw_red[::5],alpha=0.1, marker='.')
    x = np.array(plt.xlim())
    plt.plot(x, intercept+slope*x)
    plt.xlabel('ACh3.8 405 signal')
    plt.ylabel('ACh3.8 565 signal')
    plt.title('565 - 405 correlation.')

    # Add text box with slope and R-squared values
    textstr = f'Slope: {slope:.3f}\nR-squared: {r_value**2:.3f}'
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()
    return None

def plot_470_565_correlation(raw_green, raw_red):
    """
    Plots the correlation between the 470 and 565 signals.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    slope, intercept, r_value, p_value, std_err = linregress(x=raw_green, y=raw_red)
    plt.figure()
    plt.scatter(raw_green[::5], raw_red[::5],alpha=0.1, marker='.')
    x = np.array(plt.xlim())
    plt.plot(x, intercept+slope*x)
    plt.xlabel('ACh3.8 470 signal')
    plt.ylabel('ACh3.8 565 signal')
    plt.title('565 - 470 correlation.')

    # Add text box with slope and R-squared values
    textstr = f'Slope: {slope:.3f}\nR-squared: {r_value**2:.3f}'
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()
    return None

def plot_ratio_565_correlation(filtered_ratio, filtered_red):
    """
    Plots the correlation between the 470/405 ratio and the 565 signal.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    slope, intercept, r_value, p_value, std_err = linregress(x=filtered_red, y=filtered_ratio)
    plt.figure(figsize=(13, 10))
    plt.scatter(filtered_red[::5], filtered_ratio[::5],alpha=0.1, marker='.')
    x = np.array(plt.xlim())
    plt.plot(x, intercept+slope*x)
    plt.xlabel('rDA3m')
    plt.ylabel('GACh3.8 470/405 ratio')
    plt.title('Ratiometric GACh3.8 - rDA3m correlation.')

    # Add text box with slope and R-squared values
    textstr = f'Slope: {slope:.3f}\nR-squared: {r_value**2:.3f}'
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()
    return None

def plot_interactive_filtered_signals(time_seconds, filtered_ratio, filtered_green, filtered_405, filtered_red, pulse_times_in_mins):
    """
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig,ax1=plt.subplots(figsize=(16, 10))  
    plot1=ax1.plot(time_seconds, filtered_ratio, 'b', label='470/405')
    plot2=ax1.plot(time_seconds, filtered_green, 'g', label='470 GACh3.8')
    plot3=ax1.plot(time_seconds, filtered_405, 'y', label='405')

    ax2=plt.twinx()
    plot4=ax2.plot(time_seconds, filtered_red, color='r', label='565 rDA3m') 

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('ACh & 405 Signals (V)', color='k')
    ax2.set_ylabel('565 DA Signal (V)', color='r')
    ax1.set_title('Denoised and Bleach Corrected by Highpass Filtering at 0.001Hz and Lowpass Filtering at 10Hz')

    lines = plot1+plot2+plot3+plot4
    labels = [l.get_label() for l in lines]  
    ax2.set_ylim(ax1.get_ylim())

    # Setup the plot

    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to give space for the slider

    ax2.set_ylabel('')  # This removes the y-axis label for ax2
    ax2.yaxis.set_ticks([]) 


    # Set the limits for the initial plot
    ax1.set_xlim(0, 120)

    # Create the slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])  # position for the slider
    slider = Slider(ax_slider, 'Time', 0, max(time_seconds) - 120, valinit=0, valstep=0.1)

    # Update function for the slider
    def update(val):
        start = slider.val
        ax1.set_xlim(start, start + 120)
        ax2.set_xlim(start, start + 120)
        fig.canvas.draw_idle()

    # Register the update function with the slider
    slider.on_changed(update)

    # Show legends
    lines = plot1 + plot2 + plot3 + plot4 # Combine all plot handles
    labels = [line.get_label() for line in lines]  # Extract labels
    ax1.legend(lines, labels, loc='upper right')

    # Show the plot
    plt.show()
    return None

def plot_signals_aligned_port_entry(time_seconds, GACh_highpass, highpass_405, rDA3m_highpass, ratio_highpass, ppd_data, sampledata):
    """
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters
    sampling_rate = 86  # Hz
    time_window = 5  # seconds around port entry
    samples_window = time_window * sampling_rate
    total_rewarded_trials = (sampledata['rwd'] == 1).sum()
    print("Total rewarded trials: "+str(total_rewarded_trials))

    total_omitted_trials = (sampledata['rwd'] == 0).sum()
    print("Total omitted trials: "+str(total_omitted_trials))
    # Get port entry indices (non-zero values in data['digital_1'])
    port_entry_indices = ppd_data['pulse_inds_1'][1:]

    # Initialize lists for aligned traces
    aligned_470_405 = []
    aligned_470_GACh = []
    aligned_405 = []
    aligned_565_rDA3m = []

    # Align windows
    for idx in port_entry_indices:
        if idx - samples_window >= 0 and idx + samples_window < len(time_seconds):  # Ensure indices are in bounds
            aligned_470_405.append(ratio_highpass[idx - samples_window:idx + samples_window])
            aligned_470_GACh.append(GACh_highpass[idx - samples_window:idx + samples_window])
            aligned_405.append(highpass_405[idx - samples_window:idx + samples_window])
            aligned_565_rDA3m.append(rDA3m_highpass[idx - samples_window:idx + samples_window])

    # Convert to numpy arrays for easier averaging
    aligned_470_405 = np.array(aligned_470_405)
    aligned_470_GACh = np.array(aligned_470_GACh)
    aligned_405 = np.array(aligned_405)
    aligned_565_rDA3m = np.array(aligned_565_rDA3m)

    # Calculate means and SEMs
    mean_470_405 = np.mean(aligned_470_405, axis=0)
    sem_470_405 = np.std(aligned_470_405, axis=0) / np.sqrt(len(aligned_470_405))

    mean_470_GACh = np.mean(aligned_470_GACh, axis=0)
    sem_470_GACh = np.std(aligned_470_GACh, axis=0) / np.sqrt(len(aligned_470_GACh))

    mean_405 = np.mean(aligned_405, axis=0)
    sem_405 = np.std(aligned_405, axis=0) / np.sqrt(len(aligned_405))

    mean_565_rDA3m = np.mean(aligned_565_rDA3m, axis=0)
    sem_565_rDA3m = np.std(aligned_565_rDA3m, axis=0) / np.sqrt(len(aligned_565_rDA3m))

    # Generate time axis for aligned data
    aligned_time = np.linspace(-time_window, time_window, 2 * samples_window)

    # Plot averaged traces
    plt.figure(figsize=(16, 10))
    plt.title("Average Traces Aligned to Port Entry ("+str(len(port_entry_indices))+" total trials)")

    # Plot each trace with SEM shading
    plt.plot(aligned_time, mean_470_405, label='470/405', color='b')
    plt.fill_between(aligned_time, mean_470_405 - sem_470_405, mean_470_405 + sem_470_405, color='b', alpha=0.2)

    plt.plot(aligned_time, mean_470_GACh, label='470 GACh3.8', color='g')
    plt.fill_between(aligned_time, mean_470_GACh - sem_470_GACh, mean_470_GACh + sem_470_GACh, color='g', alpha=0.2)

    plt.plot(aligned_time, mean_405, label='405', color='y')
    plt.fill_between(aligned_time, mean_405 - sem_405, mean_405 + sem_405, color='y', alpha=0.2)

    plt.plot(aligned_time, mean_565_rDA3m, label='565 rDA3m', color='r')
    plt.fill_between(aligned_time, mean_565_rDA3m - sem_565_rDA3m, mean_565_rDA3m + sem_565_rDA3m, color='r', alpha=0.2)

    # Add vertical line at port entry
    plt.axvline(x=0, ls='--', color='k', label='Port Entry ('+str(total_rewarded_trials)+' RWD; '+str(total_omitted_trials)+' OMT)')

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Volts (V)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None

def plot_normalized_signals(pulse_times_in_mins, green_zscored, zscored_405, red_zscored, ratio_zscored, xvals):
    """
    """
    import numpy as np
    import matplotlib.pyplot as plt

    zscrd = plt.figure(figsize=(16, 10))
    plt.suptitle('Z-scored signals calculated after preprocessing raw singals \n by applying a high-pass filter at 0.001 Hz and a low-pass filter at 10 Hz')

    ax1 = zscrd.add_subplot(411) # Three integers (nrows, ncols, index).
    ax1.plot(xvals,green_zscored,color='blue',lw=1.5,label='ACh3.8 470')
    ax1.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(green_zscored)), label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax1.legend()

    ax2 = zscrd.add_subplot(412, sharex=ax1)
    ax2.plot(xvals,zscored_405,color='purple',lw=1.5,label='ACh3.8 405')
    ax2.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(zscored_405)), label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax2.legend()

    ax3 = zscrd.add_subplot(413, sharex=ax1)
    ax3.plot(xvals,red_zscored,color='green',lw=1.5,label='rDA3m 565')
    ax3.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(red_zscored)), label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax3.legend()

    ax3 = zscrd.add_subplot(414, sharex=ax1)
    ax3.plot(xvals,ratio_zscored,color='black',lw=1.5,label='470/405')
    ax3.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(ratio_zscored)), label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax3.legend()
    ax3.set_xlabel('time (min)')

    zscrd.text(0.04, 0.5, 'Z-Score', va='center', rotation='vertical')

    plt.show()

    return None

def plot_ratio_and_565_signals_aligned_port_entry():
    """
    """
    ratio_traces = []
    red_traces = [] 
    time_window = int(time_window)

    xvals = np.arange(-time_window*86,time_window*86+1)/86

    # Loop through each index in visinds
    for i in visinds:
        # Extract a time window of data around the current index
        start_idx = i - 86 * time_window
        end_idx = i + 86 * time_window
        ratio_trace = sampledata.loc[start_idx:end_idx, "ratio_z_scored"].values
        red_trace = sampledata.loc[start_idx:end_idx, "red_z_scored"].values

        # Only use traces of the correct length (matching xvals)
        if len(ratio_trace) == len(xvals):
            ratio_traces.append(ratio_trace)

        if len(red_trace) == len(xvals):
            red_traces.append(red_trace)

    # Calculate mean and SEM 
    if ratio_traces: 
        ratio_traces_mean = np.mean(ratio_traces, axis=0)
        ratio_traces_sem = np.std(ratio_traces, axis=0) / np.sqrt(len(ratio_traces))
    else:  # If no traces, create empty arrays
        ratio_traces_rwd_mean = np.zeros_like(xvals)
        ratio_traces_rwd_sem = np.zeros_like(xvals)

    # Calculate mean and SEM 
    if red_traces: 
        red_traces_mean = np.mean(red_traces, axis=0)
        red_traces_sem = np.std(red_traces, axis=0) / np.sqrt(len(red_traces))
    else:  # If no traces, create empty arrays
        red_traces_mean = np.zeros_like(xvals)
        red_traces_sem = np.zeros_like(xvals)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(19, 10))

    plt.suptitle(f'Z-scored ratiometric 470/405 & 565 fluorescence signals {rat} {date} {time_window}sec', fontsize=16)

    ax1.plot(xvals, ratio_traces_mean, color='g')
    ax1.fill_between(xvals, ratio_traces_mean - ratio_traces_sem, ratio_traces_mean + ratio_traces_sem, color='g', alpha=0.2)
    ax1.axvline(x=0, ls='--', color='k', label='Port Entry')
    ax1.set_title('470/405 Ratio')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Z-score')

    ax2.plot(xvals, red_traces_mean, color='r')
    ax2.fill_between(xvals, red_traces_mean - red_traces_sem, red_traces_mean + red_traces_sem, color='r', alpha=0.2)
    ax2.axvline(x=0, ls='--', color='k', label=str(len(visinds))+' Port Entries')
    ax2.set_title('565 rDA3m')
    ax2.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.legend()
    plt.plot()

    return None

def plot_ratio_and_565_signals_aligned_port_entry_separated_by_rewarded_or_omitted():
    """
    """
    ratio_rwd_traces = []
    ratio_om_traces = []
    red_rwd_traces = []
    red_om_traces = []

    # Loop through each index in visinds
    for i in visinds:
        # Extract a time window of data around the current index
        start_idx = i - 86 * time_window
        end_idx = i + 86 * time_window
        ratio_trace = sampledata.loc[start_idx:end_idx, "ratio_z_scored"].values
        red_trace = sampledata.loc[start_idx:end_idx, "red_z_scored"].values

        # Only use traces of the correct length (matching xvals)
        if len(ratio_trace) == len(xvals):
            # Check if the trial is rewarded or omitted
            if sampledata.loc[i, 'rwd'] == 1:
                ratio_rwd_traces.append(ratio_trace)  # Add to rewarded traces
            else:
                ratio_om_traces.append(ratio_trace)  # Add to omitted traces

            # Only use traces of the correct length (matching xvals)
        if len(red_trace) == len(xvals):
            # Check if the trial is rewarded or omitted
            if sampledata.loc[i, 'rwd'] == 1:
                red_rwd_traces.append(red_trace)  # Add to rewarded traces
            else:
                red_om_traces.append(red_trace)  # Add to omitted traces

    # Calculate mean and SEM for rewarded traces
    if ratio_rwd_traces:  # If there are any rewarded traces
        ratio_rwd_mean = np.mean(ratio_rwd_traces, axis=0)
        ratio_rwd_sem = np.std(ratio_rwd_traces, axis=0) / np.sqrt(len(ratio_rwd_traces))
    else:  # If no rewarded traces, create empty arrays
        ratio_rwd_mean = np.zeros_like(xvals)
        ratio_rwd_sem = np.zeros_like(xvals)

    # Calculate mean and SEM for rewarded traces
    if red_rwd_traces:  # If there are any rewarded traces
        red_rwd_mean = np.mean(red_rwd_traces, axis=0)
        red_rwd_sem = np.std(red_rwd_traces, axis=0) / np.sqrt(len(red_rwd_traces))
    else:  # If no rewarded traces, create empty arrays
        red_rwd_mean = np.zeros_like(xvals)
        red_rwd_sem = np.zeros_like(xvals)

    # Calculate mean and SEM for omitted traces
    if ratio_om_traces:  # If there are any omitted traces
        ratio_om_mean = np.mean(ratio_om_traces, axis=0)
        ratio_om_sem = np.std(ratio_om_traces, axis=0) / np.sqrt(len(ratio_om_traces))
    else:  # If no omitted traces, create empty arrays
        ratio_om_mean = np.zeros_like(xvals)
        ratio_om_sem = np.zeros_like(xvals)

    # Calculate mean and SEM for omitted traces
    if red_om_traces:  # If there are any omitted traces
        red_om_mean = np.mean(red_om_traces, axis=0)
        red_om_sem = np.std(red_om_traces, axis=0) / np.sqrt(len(red_om_traces))
    else:  # If no omitted traces, create empty arrays
        red_om_mean = np.zeros_like(xvals)
        red_om_sem = np.zeros_like(xvals)


    xvals = np.arange(-time_window*86,time_window*86+1)/86

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(19, 10))

    plt.suptitle(f'Z-scored ratiometric 470/405 & 565 fluorescence signals {rat} {date} {time_window}sec', fontsize=16)

    ax1.plot(xvals, ratio_rwd_mean, label=str(total_rewarded_trials)+' Rewarded', color='r')
    ax1.plot(xvals, ratio_om_mean, label=str(total_omitted_trials)+' Omitted', color='b')
    ax1.fill_between(xvals, ratio_rwd_mean - ratio_rwd_sem, ratio_rwd_mean + ratio_rwd_sem, color='r', alpha=0.2)
    ax1.fill_between(xvals, ratio_om_mean - ratio_om_sem, ratio_om_mean + ratio_om_sem, color='b', alpha=0.2)
    ax1.axvline(x=0, ls='--', color='k', label='Port Entry')
    ax1.set_title('470/405 Ratio')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Z-score')

    ax2.plot(xvals, red_rwd_mean, label=str(total_rewarded_trials)+' Rewarded', color='r')
    ax2.plot(xvals, red_om_mean, label=str(total_omitted_trials)+' Omitted', color='b')
    ax2.fill_between(xvals, red_rwd_mean - red_rwd_sem, red_rwd_mean + red_rwd_sem, color='r', alpha=0.2)
    ax2.fill_between(xvals, red_om_mean - red_om_sem, red_om_mean + red_om_sem, color='b', alpha=0.2)
    ax2.axvline(x=0, ls='--', color='k', label='Port Entry')
    ax2.set_title('565 rDA3m')
    ax2.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.legend()
    plt.plot()