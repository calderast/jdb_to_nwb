# Some plots for combined photometry and behavior (things aligned to port entry, rewarded/omitted etc!)
# Implement eventually. For now, comment it all out! 

# import numpy as np
# import matplotlib.pyplot as plt

# def plot_signals_aligned_port_entry(sampling_rate, time_seconds, visits, 
#                                     port_entry_indices, aligned_time,ratio_highpass, 
#                                     highpass_405, green_highpass, red_highpass, 
#                                     mean_ratio, sem_ratio, mean_green, sem_green, 
#                                     mean_405, sem_405, mean_red, sem_red, 
#                                     total_rewarded_trials, total_omitted_trials):
#     """
#     """
#     time_window_in_sec = 10
    
#     samples_window = time_window_in_sec * sampling_rate
#     total_rewarded_trials = (rwd == 1).sum()
#     print("Total rewarded trials: "+str(total_rewarded_trials))

#     total_omitted_trials = (rwd == 0).sum()
#     print("Total omitted trials: "+str(total_omitted_trials))

#     # Initialize lists for aligned traces
#     aligned_ratio = []
#     aligned_green = []
#     aligned_405 = []
#     aligned_red = []

# #################################################################################
#     # QUESTION FOR STEPH:
#     # I need this code to run some of the plotting from this point forward.
#     # Is there a way to get this information from convert_behvaior?
#     # For now, I'll keep everything the way it is using "visinds" and "port_entry_indices"
#     visinds = sampledata.loc[sampledata.port.notnull()].index.values
#     port_entry_indices = visinds 
# #################################################################################

#     # Align windows
#     for idx in visits:
#         if idx - samples_window >= 0 and idx + samples_window < len(time_seconds):  # Ensure indices are in bounds
#             aligned_ratio.append(ratio_highpass[idx - samples_window:idx + samples_window])
#             aligned_green.append(green_highpass[idx - samples_window:idx + samples_window])
#             aligned_405.append(highpass_405[idx - samples_window:idx + samples_window])
#             aligned_red.append(red_highpass[idx - samples_window:idx + samples_window])

#     # Convert to numpy arrays for easier averaging
#     aligned_ratio = np.array(aligned_ratio)
#     aligned_green = np.array(aligned_green)
#     aligned_405 = np.array(aligned_405)
#     aligned_red = np.array(aligned_red)

#     # Calculate means and SEMs
#     mean_ratio = np.mean(aligned_ratio, axis=0)
#     sem_ratio = np.std(aligned_ratio, axis=0) / np.sqrt(len(aligned_ratio))

#     mean_green = np.mean(aligned_green, axis=0)
#     sem_green = np.std(aligned_green, axis=0) / np.sqrt(len(aligned_green))

#     mean_405 = np.mean(aligned_405, axis=0)
#     sem_405 = np.std(aligned_405, axis=0) / np.sqrt(len(aligned_405))

#     mean_red = np.mean(aligned_red, axis=0)
#     sem_red = np.std(aligned_red, axis=0) / np.sqrt(len(aligned_red))

#     # Generate time axis for aligned data
#     aligned_time = np.linspace(-time_window_in_sec, time_window_in_sec, 2 * samples_window)
#     # Plot averaged traces
#     plt.figure(figsize=(16, 10))
#     plt.title("Average Traces Aligned to Port Entry ("+str(len(port_entry_indices))+" total trials)")

#     # Plot each trace with SEM shading
#     plt.plot(aligned_time, mean_ratio, label='470/405', color='b')
#     plt.fill_between(aligned_time, mean_ratio - sem_ratio, mean_ratio + sem_ratio, color='b', alpha=0.2)

#     plt.plot(aligned_time, mean_green, label='470 GACh3.8', color='g')
#     plt.fill_between(aligned_time, mean_green - sem_green, mean_green + sem_green, color='g', alpha=0.2)

#     plt.plot(aligned_time, mean_405, label='405', color='y')
#     plt.fill_between(aligned_time, mean_405 - sem_405, mean_405 + sem_405, color='y', alpha=0.2)

#     plt.plot(aligned_time, mean_red, label='565 rDA3m', color='r')
#     plt.fill_between(aligned_time, mean_red - sem_red, mean_red + sem_red, color='r', alpha=0.2)

#     # Add vertical line at port entry
#     plt.axvline(x=0, ls='--', color='k', 
#                 label='Port Entry ('+str(total_rewarded_trials)+' RWD; '+str(total_omitted_trials)+' OMT)')

#     # Add labels and legend
#     plt.xlabel("Time (s)")
#     plt.ylabel("Volts (V)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     return None


# def plot_ratio_and_565_signals_aligned_port_entry(rat, date, time_window, time_window_xvals, 
#                                                   ratio_traces_mean, ratio_traces_sem, red_traces_mean, 
#                                                   red_traces_sem, visinds): 
#     """
#     """
#     ratio_traces = []
#     red_traces = [] 
#     time_window_in_sec = 10

#     time_window_xvals = np.arange(-time_window_in_sec*86,time_window_in_sec*86+1)/86

#     # Loop through each index in visinds
#     for i in visinds:
#         # Extract a time window of data around the current index
#         start_idx = i - 86 * time_window_in_sec
#         end_idx = i + 86 * time_window_in_sec
#         ratio_trace = sampledata.loc[start_idx:end_idx, "ratio_z_scored"].values
#         red_trace = sampledata.loc[start_idx:end_idx, "red_z_scored"].values

#         # Only use traces of the correct length (matching xvals)
#         if len(ratio_trace) == len(time_window_xvals):
#             ratio_traces.append(ratio_trace)

#         if len(red_trace) == len(time_window_xvals):
#             red_traces.append(red_trace)

#     # Calculate mean and SEM 
#     if ratio_traces: 
#         ratio_traces_mean = np.mean(ratio_traces, axis=0)
#         ratio_traces_sem = np.std(ratio_traces, axis=0) / np.sqrt(len(ratio_traces))
#     else:  # If no traces, create empty arrays
#         ratio_traces_rwd_mean = np.zeros_like(time_window_xvals)
#         ratio_traces_rwd_sem = np.zeros_like(time_window_xvals)

#     # Calculate mean and SEM 
#     if red_traces: 
#         red_traces_mean = np.mean(red_traces, axis=0)
#         red_traces_sem = np.std(red_traces, axis=0) / np.sqrt(len(red_traces))
#     else:  # If no traces, create empty arrays
#         red_traces_mean = np.zeros_like(time_window_xvals)
#         red_traces_sem = np.zeros_like(time_window_xvals)
#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(19, 10))

#     plt.suptitle(f'Z-scored ratiometric 470/405 & 565 fluorescence signals {rat} {date} {time_window}sec', 
#                  fontsize=16)

#     ax1.plot(time_window_xvals, ratio_traces_mean, color='g')
#     ax1.fill_between(time_window_xvals, 
#                      ratio_traces_mean - ratio_traces_sem, 
#                      ratio_traces_mean + ratio_traces_sem, 
#                      color='g', alpha=0.2)
#     ax1.axvline(x=0, ls='--', color='k', label='Port Entry')
#     ax1.set_title('470/405 Ratio')
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel('Z-score')

#     ax2.plot(time_window_xvals, red_traces_mean, color='r')
#     ax2.fill_between(time_window_xvals, 
#                      red_traces_mean - red_traces_sem, 
#                      red_traces_mean + red_traces_sem, 
#                      color='r', alpha=0.2)
#     ax2.axvline(x=0, ls='--', color='k', label=str(len(visinds))+' Port Entries')
#     ax2.set_title('565 rDA3m')
#     ax2.set_xlabel('Time (s)')

#     plt.tight_layout()
#     plt.legend()
#     plt.plot()

#     return None

# def plot_ratio_and_565_signals_aligned_port_entry_separated_by_rewarded_or_omitted(rat, date, visinds, 
#         time_window_in_sec, xvals, ratio_rwd_mean, ratio_rwd_sem, ratio_om_mean, ratio_om_sem, 
#         red_rwd_mean, red_rwd_sem, red_om_mean, red_om_sem, total_rewarded_trials, total_omitted_trials):
#     """
#     """
#     ratio_rwd_traces = []
#     ratio_om_traces = []
#     red_rwd_traces = []
#     red_om_traces = []

#     time_window_in_sec = 10

#     time_window_xvals = np.arange(-time_window_in_sec*86,time_window_in_sec*86+1)/86

#     # Loop through each index in visinds
#     for i in visinds:
#         # Extract a time window of data around the current index
#         start_idx = i - 86 * time_window_in_sec
#         end_idx = i + 86 * time_window_in_sec
#         ratio_trace = sampledata.loc[start_idx:end_idx, "ratio_z_scored"].values
#         red_trace = sampledata.loc[start_idx:end_idx, "red_z_scored"].values

#         # Only use traces of the correct length (matching xvals)
#         if len(ratio_trace) == len(time_window_xvals):
#             # Check if the trial is rewarded or omitted
#             if sampledata.loc[i, 'rwd'] == 1:
#                 ratio_rwd_traces.append(ratio_trace)  # Add to rewarded traces
#             else:
#                 ratio_om_traces.append(ratio_trace)  # Add to omitted traces

#             # Only use traces of the correct length (matching xvals)
#         if len(red_trace) == len(time_window_xvals):
#             # Check if the trial is rewarded or omitted
#             if sampledata.loc[i, 'rwd'] == 1:
#                 red_rwd_traces.append(red_trace)  # Add to rewarded traces
#             else:
#                 red_om_traces.append(red_trace)  # Add to omitted traces

#     # Calculate mean and SEM for rewarded traces
#     if ratio_rwd_traces:  # If there are any rewarded traces
#         ratio_rwd_mean = np.mean(ratio_rwd_traces, axis=0)
#         ratio_rwd_sem = np.std(ratio_rwd_traces, axis=0) / np.sqrt(len(ratio_rwd_traces))
#     else:  # If no rewarded traces, create empty arrays
#         ratio_rwd_mean = np.zeros_like(time_window_xvals)
#         ratio_rwd_sem = np.zeros_like(time_window_xvals)

#     # Calculate mean and SEM for rewarded traces
#     if red_rwd_traces:  # If there are any rewarded traces
#         red_rwd_mean = np.mean(red_rwd_traces, axis=0)
#         red_rwd_sem = np.std(red_rwd_traces, axis=0) / np.sqrt(len(red_rwd_traces))
#     else:  # If no rewarded traces, create empty arrays
#         red_rwd_mean = np.zeros_like(time_window_xvals)
#         red_rwd_sem = np.zeros_like(time_window_xvals)

#     # Calculate mean and SEM for omitted traces
#     if ratio_om_traces:  # If there are any omitted traces
#         ratio_om_mean = np.mean(ratio_om_traces, axis=0)
#         ratio_om_sem = np.std(ratio_om_traces, axis=0) / np.sqrt(len(ratio_om_traces))
#     else:  # If no omitted traces, create empty arrays
#         ratio_om_mean = np.zeros_like(time_window_xvals)
#         ratio_om_sem = np.zeros_like(time_window_xvals)

#     # Calculate mean and SEM for omitted traces
#     if red_om_traces:  # If there are any omitted traces
#         red_om_mean = np.mean(red_om_traces, axis=0)
#         red_om_sem = np.std(red_om_traces, axis=0) / np.sqrt(len(red_om_traces))
#     else:  # If no omitted traces, create empty arrays
#         red_om_mean = np.zeros_like(time_window_xvals)
#         red_om_sem = np.zeros_like(time_window_xvals)

#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(19, 10))

#     plt.suptitle(f'Z-scored ratiometric 470/405 & 565 fluorescence signals {rat} {date} {time_window_in_sec}sec', 
#                  fontsize=16)

#     ax1.plot(xvals, ratio_rwd_mean, label=str(total_rewarded_trials)+' Rewarded', color='r')
#     ax1.plot(xvals, ratio_om_mean, label=str(total_omitted_trials)+' Omitted', color='b')
#     ax1.fill_between(xvals, ratio_rwd_mean - ratio_rwd_sem, ratio_rwd_mean + ratio_rwd_sem, color='r', alpha=0.2)
#     ax1.fill_between(xvals, ratio_om_mean - ratio_om_sem, ratio_om_mean + ratio_om_sem, color='b', alpha=0.2)
#     ax1.axvline(x=0, ls='--', color='k', label='Port Entry')
#     ax1.set_title('470/405 Ratio')
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel('Z-score')

#     ax2.plot(xvals, red_rwd_mean, label=str(total_rewarded_trials)+' Rewarded', color='r')
#     ax2.plot(xvals, red_om_mean, label=str(total_omitted_trials)+' Omitted', color='b')
#     ax2.fill_between(xvals, red_rwd_mean - red_rwd_sem, red_rwd_mean + red_rwd_sem, color='r', alpha=0.2)
#     ax2.fill_between(xvals, red_om_mean - red_om_sem, red_om_mean + red_om_sem, color='b', alpha=0.2)
#     ax2.axvline(x=0, ls='--', color='k', label='Port Entry')
#     ax2.set_title('565 rDA3m')
#     ax2.set_xlabel('Time (s)')

#     plt.tight_layout()
#     plt.legend()
#     plt.plot()