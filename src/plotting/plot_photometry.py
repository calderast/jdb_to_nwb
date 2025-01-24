import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import linregress

def plot_raw_photometry_signals(visits, raw_green, raw_red, raw_405, relative_raw_signal, sampling_rate): 
    """
    Plots the raw 470, 405, 565 and ratiometric 470/405 fluorescence signals.
    """
    xvals = np.arange(0,len(raw_green))/sampling_rate/60 
    pulse_times_in_mins = [time / 60000 for time in visits]

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
    ax4.plot(xvals,relative_raw_signal,color='black',lw=1.5,label='470/405 (V)')
    ax4.set_xlabel('time (min)')
    ax4.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(relative_raw_signal)), label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax4.legend()

    plt.show()

    return None


def plot_405_470_correlation(raw_405, raw_green, slope_405x470, intercep_405x470, r_value_405x470):
    """
    Plots the correlation between the 405 and 470 signals.
    """
    slope_405x470, intercep_405x470, r_value_405x470 = linregress(x=raw_405, y=raw_green)
    plt.figure()
    plt.scatter(raw_405[::5], raw_green[::5],alpha=0.1, marker='.')
    x = np.array(plt.xlim())
    plt.plot(x, intercep_405x470+ slope_405x470 * x)
    plt.xlabel('ACh3.8 405 signal')
    plt.ylabel('ACh3.8 470 signal')
    plt.title('470 - 405 correlation.')

    # Add text box with slope and R-squared values
    textstr = f'Slope: {slope_405x470:.3f}\nR-squared: {r_value_405x470**2:.3f}'
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()
    return None

def plot_405_565_correlation(raw_405, raw_red, slope_405x565, intercep_405x565, r_value_405x565):
    """
    Plots the correlation between the 405 and 565 signals.
    """
    slope_405x565, intercep_405x565, r_value_405x565 = linregress(x=raw_405, y=raw_red)
    plt.figure()
    plt.scatter(raw_405[::5], raw_red[::5],alpha=0.1, marker='.')
    x = np.array(plt.xlim())
    plt.plot(x, intercep_405x565 + slope_405x565 * x)
    plt.xlabel('ACh3.8 405 signal')
    plt.ylabel('ACh3.8 565 signal')
    plt.title('565 - 405 correlation.')

    # Add text box with slope and R-squared values
    textstr = f'Slope: {slope_405x565:.3f}\nR-squared: {r_value_405x565**2:.3f}'
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()
    return None

def plot_470_565_correlation(raw_green, raw_red, slope_470x565, intercep_470x565, r_value_470x565):
    """
    Plots the correlation between the 470 and 565 signals.
    """
    slope_470x565, intercep_470x565, r_value_470x565 = linregress(x=raw_green, y=raw_red)
    plt.figure()
    plt.scatter(raw_green[::5], raw_red[::5],alpha=0.1, marker='.')
    x = np.array(plt.xlim())
    plt.plot(x, intercep_470x565 + slope_470x565 * x)
    plt.xlabel('ACh3.8 470 signal')
    plt.ylabel('ACh3.8 565 signal')
    plt.title('565 - 470 correlation.')

    # Add text box with slope and R-squared values
    textstr = f'Slope: {slope_470x565:.3f}\nR-squared: {r_value_470x565**2:.3f}'
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()
    return None

def plot_ratio_565_correlation(ratio_highpass, red_highpass, slope_filtered, intercept_filtered, r_value_filtered):
    """
    Plots the correlation between the 470/405 ratio and the 565 signal.
    """
    slope_filtered, intercept_filtered, r_value_filtered = linregress(x=red_highpass, y=ratio_highpass)
    plt.figure(figsize=(13, 10))
    plt.scatter(red_highpass[::5], ratio_highpass[::5],alpha=0.1, marker='.')
    x = np.array(plt.xlim())
    plt.plot(x, intercept_filtered + slope_filtered * x)
    plt.xlabel('rDA3m')
    plt.ylabel('GACh3.8 470/405 ratio')
    plt.title('Ratiometric GACh3.8 - rDA3m correlation.')

    # Add text box with slope and R-squared values
    textstr = f'Slope: {slope_filtered:.3f}\nR-squared: {r_value_filtered**2:.3f}'
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()
    return None

def plot_interactive_filtered_signals(time_seconds, filtered_ratio, filtered_green, filtered_405, filtered_red):
    """
    """

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

def plot_signals_aligned_port_entry(sampling_rate, time_seconds, visits, port_entry_indices, aligned_time,ratio_highpass, highpass_405, green_highpass, red_highpass, mean_ratio, sem_ratio, mean_green, sem_green, mean_405, sem_405, mean_red, sem_red, total_rewarded_trials, total_omitted_trials):
    """
    """
    time_window_in_sec = 10
    
    samples_window = time_window_in_sec * sampling_rate
    total_rewarded_trials = (rwd == 1).sum()
    print("Total rewarded trials: "+str(total_rewarded_trials))

    total_omitted_trials = (rwd == 0).sum()
    print("Total omitted trials: "+str(total_omitted_trials))

    # Initialize lists for aligned traces
    aligned_ratio = []
    aligned_green = []
    aligned_405 = []
    aligned_red = []

#################################################################################
    # QUESTION FOR STEPH:
    # I need this code to run some of the plotting from this point forward.
    # Is there a way to get this information from convert_behvaior?
    # For now, I'll keep everything the way it is using "visinds" and "port_entry_indeces"
    visinds = sampledata.loc[sampledata.port.notnull()].index.values
    port_entry_indices = visinds 
#################################################################################

    # Align windows
    for idx in visits:
        if idx - samples_window >= 0 and idx + samples_window < len(time_seconds):  # Ensure indices are in bounds
            aligned_ratio.append(ratio_highpass[idx - samples_window:idx + samples_window])
            aligned_green.append(green_highpass[idx - samples_window:idx + samples_window])
            aligned_405.append(highpass_405[idx - samples_window:idx + samples_window])
            aligned_red.append(red_highpass[idx - samples_window:idx + samples_window])

    # Convert to numpy arrays for easier averaging
    aligned_ratio = np.array(aligned_ratio)
    aligned_green = np.array(aligned_green)
    aligned_405 = np.array(aligned_405)
    aligned_red = np.array(aligned_red)

    # Calculate means and SEMs
    mean_ratio = np.mean(aligned_ratio, axis=0)
    sem_ratio = np.std(aligned_ratio, axis=0) / np.sqrt(len(aligned_ratio))

    mean_green = np.mean(aligned_green, axis=0)
    sem_green = np.std(aligned_green, axis=0) / np.sqrt(len(aligned_green))

    mean_405 = np.mean(aligned_405, axis=0)
    sem_405 = np.std(aligned_405, axis=0) / np.sqrt(len(aligned_405))

    mean_red = np.mean(aligned_red, axis=0)
    sem_red = np.std(aligned_red, axis=0) / np.sqrt(len(aligned_red))

    # Generate time axis for aligned data
    aligned_time = np.linspace(-time_window_in_sec, time_window_in_sec, 2 * samples_window)
    # Plot averaged traces
    plt.figure(figsize=(16, 10))
    plt.title("Average Traces Aligned to Port Entry ("+str(len(port_entry_indices))+" total trials)")

    # Plot each trace with SEM shading
    plt.plot(aligned_time, mean_ratio, label='470/405', color='b')
    plt.fill_between(aligned_time, mean_ratio - sem_ratio, mean_ratio + sem_ratio, color='b', alpha=0.2)

    plt.plot(aligned_time, mean_green, label='470 GACh3.8', color='g')
    plt.fill_between(aligned_time, mean_green - sem_green, mean_green + sem_green, color='g', alpha=0.2)

    plt.plot(aligned_time, mean_405, label='405', color='y')
    plt.fill_between(aligned_time, mean_405 - sem_405, mean_405 + sem_405, color='y', alpha=0.2)

    plt.plot(aligned_time, mean_red, label='565 rDA3m', color='r')
    plt.fill_between(aligned_time, mean_red - sem_red, mean_red + sem_red, color='r', alpha=0.2)

    # Add vertical line at port entry
    plt.axvline(x=0, ls='--', color='k', label='Port Entry ('+str(total_rewarded_trials)+' RWD; '+str(total_omitted_trials)+' OMT)')

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Volts (V)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None

def plot_normalized_signals(sampling_rate, pulse_times_in_mins, green_zscored, zscored_405, red_zscored, ratio_zscored):
    """
    """
    xvals = np.arange(0,len(green_zscored))/sampling_rate/60 
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

def plot_ratio_and_565_signals_aligned_port_entry(rat, date, time_window, time_window_xvals, ratio_traces_mean, ratio_traces_sem, red_traces_mean, red_traces_sem, visinds): 
    """
    """
    ratio_traces = []
    red_traces = [] 
    time_window_in_sec = 10

    time_window_xvals = np.arange(-time_window_in_sec*86,time_window_in_sec*86+1)/86

    # Loop through each index in visinds
    for i in visinds:
        # Extract a time window of data around the current index
        start_idx = i - 86 * time_window_in_sec
        end_idx = i + 86 * time_window_in_sec
        ratio_trace = sampledata.loc[start_idx:end_idx, "ratio_z_scored"].values
        red_trace = sampledata.loc[start_idx:end_idx, "red_z_scored"].values

        # Only use traces of the correct length (matching xvals)
        if len(ratio_trace) == len(time_window_xvals):
            ratio_traces.append(ratio_trace)

        if len(red_trace) == len(time_window_xvals):
            red_traces.append(red_trace)

    # Calculate mean and SEM 
    if ratio_traces: 
        ratio_traces_mean = np.mean(ratio_traces, axis=0)
        ratio_traces_sem = np.std(ratio_traces, axis=0) / np.sqrt(len(ratio_traces))
    else:  # If no traces, create empty arrays
        ratio_traces_rwd_mean = np.zeros_like(time_window_xvals)
        ratio_traces_rwd_sem = np.zeros_like(time_window_xvals)

    # Calculate mean and SEM 
    if red_traces: 
        red_traces_mean = np.mean(red_traces, axis=0)
        red_traces_sem = np.std(red_traces, axis=0) / np.sqrt(len(red_traces))
    else:  # If no traces, create empty arrays
        red_traces_mean = np.zeros_like(time_window_xvals)
        red_traces_sem = np.zeros_like(time_window_xvals)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(19, 10))

    plt.suptitle(f'Z-scored ratiometric 470/405 & 565 fluorescence signals {rat} {date} {time_window}sec', fontsize=16)

    ax1.plot(time_window_xvals, ratio_traces_mean, color='g')
    ax1.fill_between(time_window_xvals, ratio_traces_mean - ratio_traces_sem, ratio_traces_mean + ratio_traces_sem, color='g', alpha=0.2)
    ax1.axvline(x=0, ls='--', color='k', label='Port Entry')
    ax1.set_title('470/405 Ratio')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Z-score')

    ax2.plot(time_window_xvals, red_traces_mean, color='r')
    ax2.fill_between(time_window_xvals, red_traces_mean - red_traces_sem, red_traces_mean + red_traces_sem, color='r', alpha=0.2)
    ax2.axvline(x=0, ls='--', color='k', label=str(len(visinds))+' Port Entries')
    ax2.set_title('565 rDA3m')
    ax2.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.legend()
    plt.plot()

    return None

def plot_ratio_and_565_signals_aligned_port_entry_separated_by_rewarded_or_omitted(rat, date, visinds, time_window_in_sec, xvals, ratio_rwd_mean, ratio_rwd_sem, ratio_om_mean, ratio_om_sem, red_rwd_mean, red_rwd_sem, red_om_mean, red_om_sem, total_rewarded_trials, total_omitted_trials):
    """
    """
    ratio_rwd_traces = []
    ratio_om_traces = []
    red_rwd_traces = []
    red_om_traces = []

    time_window_in_sec = 10

    time_window_xvals = np.arange(-time_window_in_sec*86,time_window_in_sec*86+1)/86

    # Loop through each index in visinds
    for i in visinds:
        # Extract a time window of data around the current index
        start_idx = i - 86 * time_window_in_sec
        end_idx = i + 86 * time_window_in_sec
        ratio_trace = sampledata.loc[start_idx:end_idx, "ratio_z_scored"].values
        red_trace = sampledata.loc[start_idx:end_idx, "red_z_scored"].values

        # Only use traces of the correct length (matching xvals)
        if len(ratio_trace) == len(time_window_xvals):
            # Check if the trial is rewarded or omitted
            if sampledata.loc[i, 'rwd'] == 1:
                ratio_rwd_traces.append(ratio_trace)  # Add to rewarded traces
            else:
                ratio_om_traces.append(ratio_trace)  # Add to omitted traces

            # Only use traces of the correct length (matching xvals)
        if len(red_trace) == len(time_window_xvals):
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
        ratio_rwd_mean = np.zeros_like(time_window_xvals)
        ratio_rwd_sem = np.zeros_like(time_window_xvals)

    # Calculate mean and SEM for rewarded traces
    if red_rwd_traces:  # If there are any rewarded traces
        red_rwd_mean = np.mean(red_rwd_traces, axis=0)
        red_rwd_sem = np.std(red_rwd_traces, axis=0) / np.sqrt(len(red_rwd_traces))
    else:  # If no rewarded traces, create empty arrays
        red_rwd_mean = np.zeros_like(time_window_xvals)
        red_rwd_sem = np.zeros_like(time_window_xvals)

    # Calculate mean and SEM for omitted traces
    if ratio_om_traces:  # If there are any omitted traces
        ratio_om_mean = np.mean(ratio_om_traces, axis=0)
        ratio_om_sem = np.std(ratio_om_traces, axis=0) / np.sqrt(len(ratio_om_traces))
    else:  # If no omitted traces, create empty arrays
        ratio_om_mean = np.zeros_like(time_window_xvals)
        ratio_om_sem = np.zeros_like(time_window_xvals)

    # Calculate mean and SEM for omitted traces
    if red_om_traces:  # If there are any omitted traces
        red_om_mean = np.mean(red_om_traces, axis=0)
        red_om_sem = np.std(red_om_traces, axis=0) / np.sqrt(len(red_om_traces))
    else:  # If no omitted traces, create empty arrays
        red_om_mean = np.zeros_like(time_window_xvals)
        red_om_sem = np.zeros_like(time_window_xvals)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(19, 10))

    plt.suptitle(f'Z-scored ratiometric 470/405 & 565 fluorescence signals {rat} {date} {time_window_in_sec}sec', fontsize=16)

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