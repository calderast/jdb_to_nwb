import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


def plot_photometry_signal_aligned_to_port_entry(nwbfile, signal_name, fig_dir=None):
    """
    Plots an average photometry signal (DA or ACh) aligned to port entry, 
    split by rewarded and unrewarded trials. Also plots the full-session signal trace
    with rewarded and unrewarded poke times indicated.
    """
    session_id = nwbfile.session_id

    # Get trial and reward data
    trials = nwbfile.intervals["trials"]
    poke_in_times = trials["poke_in"].data[:]
    rewards = trials["reward"].data[:]

    # Get photometry signal trace
    photometry_signal_object = nwbfile.acquisition[signal_name]
    signal_trace = photometry_signal_object.data[:]
    timestamps = photometry_signal_object.get_timestamps()
    sampling_rate = photometry_signal_object.rate  

    # Define time window
    time_window = 3  # seconds
    num_samples = int(2 * time_window * sampling_rate)
    time_vector = np.arange(-time_window, time_window, 1/sampling_rate)[:num_samples]

    # Split by rewarded vs unrewarded trials
    rewarded, un_rewarded = [], []

    for poke_in, reward in zip(poke_in_times, rewards):
        
        # Extract signal trace centered on poke_in
        idx = np.where(timestamps == poke_in)[0][0]
        start_idx = max(0, idx - num_samples // 2)
        end_idx = start_idx + num_samples

        if end_idx <= len(signal_trace):  # Ensure within bounds
            trace = signal_trace[start_idx:end_idx]
            (rewarded if reward == 1 else un_rewarded).append(trace)

    # Convert to arrays
    rewarded, un_rewarded = map(np.array, (rewarded, un_rewarded))
    n_trials = len(poke_in_times)
    n_rewarded = rewarded.shape[0]
    n_unrewarded = un_rewarded.shape[0]

    # Get mean and SEM
    rewarded_mean = rewarded.mean(axis=0)
    rewarded_sem = sem(rewarded, axis=0)
    unrewarded_mean = un_rewarded.mean(axis=0)
    unrewarded_sem = sem(un_rewarded, axis=0)

    # First plot: Average signal response across all trials
    plt.figure(figsize=(8, 5))
    plt.plot(time_vector, rewarded_mean, label=f"rewarded (n={n_rewarded})", color="red")
    plt.fill_between(time_vector, rewarded_mean - rewarded_sem, rewarded_mean + rewarded_sem, color="red", alpha=0.3)
    plt.plot(time_vector, unrewarded_mean, label=f"unrewarded (n={n_unrewarded})", color="blue")
    plt.fill_between(time_vector, unrewarded_mean - unrewarded_sem, unrewarded_mean + unrewarded_sem, color="blue", alpha=0.3)
    plt.axvline(0, linestyle="--", color="black", label="Poke In")
    plt.xlabel("Time (s)")
    plt.ylabel(f"{signal_name}")
    plt.title(f"{signal_name} aligned to port entry ({session_id}, {n_trials} trials)")
    plt.legend()
    plt.tight_layout()

    if fig_dir:
        save_path = os.path.join(fig_dir, f"{signal_name}_aligned_to_port_entry.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # Second plot: Full session signal trace
    plt.figure(figsize=(20, 5))
    plt.plot(timestamps, signal_trace)
    for poke_in, reward in zip(poke_in_times, rewards):
        color = "red" if reward == 1 else "blue"
        plt.axvline(poke_in, linestyle="--", color=color)
    plt.xlabel("Time (s)")
    plt.ylabel(f"{signal_name}")
    plt.title(f"Full session {signal_name} and port entries")
    plt.ylim([-2, 10])
    plt.xlim([min(timestamps)-10, max(timestamps)+10])
    plt.tight_layout()

    if fig_dir:
        save_path = os.path.join(fig_dir, f"{signal_name}_full_session_trace.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()