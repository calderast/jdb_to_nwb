import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from hexmaze import plot_hex_maze


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
    plt.fill_between(time_vector, rewarded_mean - rewarded_sem, 
                     rewarded_mean + rewarded_sem, color="red", alpha=0.3)
    plt.plot(time_vector, unrewarded_mean, label=f"unrewarded (n={n_unrewarded})", color="blue")
    plt.fill_between(time_vector, unrewarded_mean - unrewarded_sem, 
                     unrewarded_mean + unrewarded_sem, color="blue", alpha=0.3)
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
    plt.title(f"Full session {signal_name} and port entries ({session_id}, {n_trials} trials)")
    plt.ylim([-2, 10])
    plt.xlim([min(timestamps)-10, max(timestamps)+10])
    plt.tight_layout()

    if fig_dir:
        save_path = os.path.join(fig_dir, f"{signal_name}_full_session_trace.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_rat_position_heatmap(nwbfile, spatial_series_name, fig_dir=None):
    """
    For each block, create a heatmap of rat position overlaid on the hex maze.
    Also create one big plot for the session where position is split by first vs second half
    of each block, to (hopefully) see refinement of trajectories as the rat adapts to the new config.
    """
    session_id = nwbfile.session_id

    # Get hex centroids and convert to a dict so we can plot the hex maze with custom centroids
    behavior_module = nwbfile.processing["behavior"]
    if "hex_centroids" in behavior_module.data_interfaces:
        centroids_df = behavior_module.data_interfaces["hex_centroids"].to_dataframe()
        centroids_dict = centroids_df.set_index('hex')[['x', 'y']].apply(tuple, axis=1).to_dict()
    else:
        # If we have no centroids, still make the plot, just without the maze background
        # (It is way worse this way... but I'll still allow it I guess)
        centroids_dict = None

    # Get position data for the given spatial series
    position = behavior_module.data_interfaces["position"].spatial_series[spatial_series_name]
    position_df = pd.DataFrame(position.data, columns=["x", "y"]) 
    position_df["timestamp"] = position.timestamps

    # Get block data
    block_data = nwbfile.intervals["block"].to_dataframe()
    n_blocks = len(block_data)

    # Set up n_blocks x 2 plot to plot rat position by (first half, second half) of each block
    fig, axs = plt.subplots(n_blocks, 2, figsize=(8, 4 * n_blocks), sharex=True, sharey=True)

    for row, block in enumerate(block_data.itertuples(index=False)):
        # Get maze configuration and reward probabilities for this block
        maze = block.maze_configuration
        reward_probs = [block.pA, block.pB, block.pC]

        # Filter position data for this block (excluding nans)
        block_times = (position_df["timestamp"] >= block.start_time) & (position_df["timestamp"] <= block.stop_time)
        block_positions = position_df[block_times].dropna(subset=['x', 'y'])

        # Set up plot of rat position heatmap for this block
        fig_full, ax_full = plt.subplots(figsize=(6, 6))

        # Create 2D histogram (aka heatmap) of the rat's x, y positions in this block
        heatmap_full, xedges, yedges = np.histogram2d(
            block_positions['x'].values, block_positions['y'].values, bins=100
        )
        heatmap_full_masked = np.ma.masked_where(heatmap_full == 0, heatmap_full)
        log_heatmap_full = np.log1p(heatmap_full_masked)

        # Plot maze layout (open hexes only) using custom centroids if they exist
        if centroids_dict is not None:
            plot_hex_maze(
                barriers=maze, centroids=centroids_dict, ax=ax_full, show_hex_labels=False,
                show_barriers=False, show_choice_points=False, reward_probabilities=reward_probs,
                invert_yaxis=True
            )
        # Plot rat position heatmap on top of the hexes
        ax_full.imshow(
            log_heatmap_full.T, origin='lower', cmap='viridis',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect='equal', zorder=1
        )
        ax_full.set_xticks([])
        ax_full.set_yticks([])
        ax_full.set_title(f"Rat position heatmap ({session_id}, block {block.block})")
        fig_full.tight_layout()

        if fig_dir:
            save_path = os.path.join(fig_dir, f"{spatial_series_name}_block_{block.block}_heatmap.png")
            fig_full.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig_full)

        # Now do the same thing, but split into first/second half so we can see behavioral adaptation

        # Split block into first and second half
        block_midpoint = len(block_positions) // 2
        halves = [block_positions.iloc[:block_midpoint], block_positions.iloc[block_midpoint:]]

        # Add this block half to our big plot
        for col, half in enumerate(halves):
            # Create 2D histogram (aka heatmap) of the rat's x, y positions in this block half
            heatmap, xedges, yedges = np.histogram2d(half['x'].values, half['y'].values, bins=100)
            heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)
            log_heatmap = np.log1p(heatmap_masked)

            ax = axs[row, col]
            # Plot maze layout (open hexes only) using custom centroids if they exist
            if centroids_dict is not None:
                plot_hex_maze(
                    barriers=maze, centroids=centroids_dict, ax=ax, show_hex_labels=False,
                    show_barriers=False, show_choice_points=False, reward_probabilities=reward_probs,
                    invert_yaxis=True
                )
            # Plot rat position heatmap on top of the hexes
            ax.imshow(
                log_heatmap.T, origin='lower', cmap='viridis', 
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect='equal', zorder=1
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Block {block.block} ({'first' if col == 0 else 'second'} half)")

    fig.suptitle(f"{session_id} rat position heatmaps", fontsize=16)
    fig.tight_layout()

    if fig_dir:
            save_path = os.path.join(fig_dir, f"{spatial_series_name}_all_blocks_heatmap.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)