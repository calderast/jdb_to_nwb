import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


# The following plots are called by process_and_add_pyphotometry_to_nwb

def plot_raw_photometry_signals(visits, raw_green, raw_red, raw_405, 
                                relative_raw_signal, sampling_rate, fig_dir=None): 
    """
    Plots the raw 470, 405, 565 and ratiometric 470/405 fluorescence signals.
    """
    xvals = np.arange(0,len(raw_green)) / sampling_rate / 60 
    pulse_times_in_mins = [time / 60 for time in visits]

    raw = plt.figure(figsize=(16, 10))
    plt.suptitle('Raw & ratiometric 470/405 fluorescence signals', fontsize=16)

    ax1 = raw.add_subplot(411) # Three integers (nrows, ncols, index).
    ax1.plot(xvals,raw_green,color='blue',lw=1.5,label='raw 470 (V)')
    ax1.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(raw_green)), 
             label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax1.legend()

    ax2 = raw.add_subplot(412, sharex=ax1)
    ax2.plot(xvals,raw_405,color='purple',lw=1.5,label='raw 405 (V)')
    ax1.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(raw_green)), 
             label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax2.legend()

    ax3 = raw.add_subplot(413, sharex=ax1)
    ax3.plot(xvals,raw_red,color='green',lw=1.5,label='raw 565 (V)')
    ax3.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(raw_red)), 
             label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax3.legend()

    ax4 = raw.add_subplot(414, sharex=ax1)
    ax4.plot(xvals,relative_raw_signal,color='black',lw=1.5,label='470/405 (V)')
    ax4.set_xlabel('time (min)')
    ax4.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(relative_raw_signal)), 
             label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax4.legend()

    if fig_dir:
        save_path = os.path.join(fig_dir, "raw_pyphotometry_signals.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_405_470_correlation(raw_405, raw_green, fig_dir=None):
    """
    Plots the correlation between the raw 405 and 470 signals.
    """
    slope_405x470, intercep_405x470, r_value_405x470, _ , _ = linregress(x=raw_405, y=raw_green)
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

    if fig_dir:
        save_path = os.path.join(fig_dir, "raw_405_470_correlation.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_405_565_correlation(raw_405, raw_red, fig_dir=None):
    """
    Plots the correlation between the raw 405 and 565 signals.
    """
    slope_405x565, intercep_405x565, r_value_405x565, _ , _ = linregress(x=raw_405, y=raw_red)
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

    if fig_dir:
        save_path = os.path.join(fig_dir, "raw_405_565_correlation.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_470_565_correlation(raw_green, raw_red, fig_dir=None):
    """
    Plots the correlation between the raw 470 and 565 signals.
    """
    slope_470x565, intercep_470x565, r_value_470x565, _ , _ = linregress(x=raw_green, y=raw_red)
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

    if fig_dir:
        save_path = os.path.join(fig_dir, "raw_470_565_correlation.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_ratio_565_correlation(ratio_highpass, red_highpass, fig_dir=None):
    """
    Plots the correlation between the filtered 470/405 ratio and the 565 signals.
    """
    slope_filtered, intercept_filtered, r_value_filtered, _ , _ = linregress(x=red_highpass, y=ratio_highpass)
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

    if fig_dir:
        save_path = os.path.join(fig_dir, "filtered_ratio_565_correlation.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_normalized_signals(visits, green_zscored, zscored_405, red_zscored, 
                            ratio_zscored, sampling_rate, fig_dir=None):
    """
    """
    xvals = np.arange(0,len(green_zscored))/ sampling_rate / 60
    pulse_times_in_mins = [time / 60 for time in visits]
    
    zscrd = plt.figure(figsize=(16, 10))
    plt.suptitle("Z-scored signals calculated after preprocessing raw signals \n "
                 "by applying a high-pass filter at 0.001 Hz and a low-pass filter at 10 Hz")

    ax1 = zscrd.add_subplot(411) # Three integers (nrows, ncols, index).
    ax1.plot(xvals,green_zscored,color='blue',lw=1.5,label='ACh3.8 470')
    ax1.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(green_zscored)), 
             label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax1.legend()

    ax2 = zscrd.add_subplot(412, sharex=ax1)
    ax2.plot(xvals,zscored_405,color='purple',lw=1.5,label='ACh3.8 405')
    ax2.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(zscored_405)), 
             label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax2.legend()

    ax3 = zscrd.add_subplot(413, sharex=ax1)
    ax3.plot(xvals,red_zscored,color='green',lw=1.5,label='rDA3m 565')
    ax3.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(red_zscored)), 
             label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax3.legend()

    ax3 = zscrd.add_subplot(414, sharex=ax1)
    ax3.plot(xvals,ratio_zscored,color='black',lw=1.5,label='470/405')
    ax3.plot(pulse_times_in_mins, np.full(np.size(pulse_times_in_mins), np.max(ratio_zscored)), 
             label='Reward Cue', color='w', marker="|", mec='k', ms=10)
    ax3.legend()
    ax3.set_xlabel('time (min)')

    zscrd.text(0.04, 0.5, 'Z-Score', va='center', rotation='vertical')

    if fig_dir:
        save_path = os.path.join(fig_dir, "processed_pyphotometry_signals.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# Commenting out the following plot for now.
# We will probably not do interactive with the conversion pipeline.
# But keeping this here for now
# # from matplotlib.widgets import Slider


# def plot_interactive_filtered_signals(time_seconds, filtered_ratio, 
#                                       filtered_green, filtered_405, filtered_red):
#     """
#     """

#     fig,ax1=plt.subplots(figsize=(16, 10))  
#     plot1=ax1.plot(time_seconds, filtered_ratio, 'b', label='470/405')
#     plot2=ax1.plot(time_seconds, filtered_green, 'g', label='470 GACh3.8')
#     plot3=ax1.plot(time_seconds, filtered_405, 'y', label='405')

#     ax2=plt.twinx()
#     plot4=ax2.plot(time_seconds, filtered_red, color='r', label='565 rDA3m') 

#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('ACh & 405 Signals (V)', color='k')
#     ax2.set_ylabel('565 DA Signal (V)', color='r')
#     ax1.set_title('Denoised and Bleach Corrected by Highpass Filtering at 0.001Hz and Lowpass Filtering at 10Hz')

#     lines = plot1+plot2+plot3+plot4
#     labels = [l.get_label() for l in lines]  
#     ax2.set_ylim(ax1.get_ylim())

#     # Setup the plot

#     plt.subplots_adjust(bottom=0.25)  # Adjust bottom to give space for the slider

#     ax2.set_ylabel('')  # This removes the y-axis label for ax2
#     ax2.yaxis.set_ticks([]) 


#     # Set the limits for the initial plot
#     ax1.set_xlim(0, 120)

#     # Create the slider
#     ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05])  # position for the slider
#     slider = Slider(ax_slider, 'Time', 0, max(time_seconds) - 120, valinit=0, valstep=0.1)

#     # Update function for the slider
#     def update(val):
#         start = slider.val
#         ax1.set_xlim(start, start + 120)
#         ax2.set_xlim(start, start + 120)
#         fig.canvas.draw_idle()

#     # Register the update function with the slider
#     slider.on_changed(update)

#     # Show legends
#     lines = plot1 + plot2 + plot3 + plot4 # Combine all plot handles
#     labels = [line.get_label() for line in lines]  # Extract labels
#     ax1.legend(lines, labels, loc='upper right')

#     # Show the plot
#     plt.show()
#     return None


