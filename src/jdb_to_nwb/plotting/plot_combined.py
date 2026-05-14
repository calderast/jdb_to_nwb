import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from hexmaze import plot_hex_maze


def plot_photometry_signal_aligned_to_port_entry(nwbfile, signal_name, fig_dir=None):
    """
    Plot a photometry signal (DA or ACh) aligned to port entry, split by rewarded vs. unrewarded trials.

    Produces two figures:
    1. Mean ± SEM signal in a ±3 s window around poke-in, separately for rewarded and unrewarded trials.
    2. Full-session processed signal trace with vertical lines marking each poke-in (red = rewarded, blue = unrewarded).

    Parameters:
        nwbfile: NWBFile object containing trials and photometry data.
        signal_name (str): Key in nwbfile.acquisition for the processed photometry TimeSeries
            (e.g. "dLight1.3b_dFF").
        fig_dir (str): Optional directory to save figures. Saved as
            {signal_name}_aligned_to_port_entry.png and {signal_name}_full_session_trace.png.

    Returns:
        tuple[plt.Figure, plt.Figure]: (aligned average figure, full session trace figure).
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

    # Get trace centered on poke_in
    for poke_in, reward in zip(poke_in_times, rewards):

        # Find the photometry sample closest to poke_in (nearest-sample rather than exact equality,
        # because extrapolated visit times may not exactly match a photometry sample timestamp)
        idx = np.argmin(np.abs(timestamps - poke_in))
        start_idx = max(0, idx - num_samples // 2)
        end_idx = start_idx + num_samples

        # Exclude this trial from the average if we don't have photometry signal within bounds
        if end_idx <= len(signal_trace):
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
    fig1 = plt.figure(figsize=(8, 5))
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
        fig1.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig1)

    # Second plot: Full session signal trace
    fig2 = plt.figure(figsize=(20, 5))
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
        fig2.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)

    return fig1, fig2


def plot_rat_position_heatmap(nwbfile, spatial_series_name, fig_dir=None):
    """
    Plot 2D heatmaps of rat position overlaid on the hex maze, one per block.

    Produces two sets of figures per block:
    1. One full-block heatmap per block (log-scaled 2D histogram overlaid on maze layout).
    2. A single session-level figure with each block split into first and second half,
       to (theoretically) visualize trajectory refinement as the rat adapts to each maze configuration.

    Parameters:
        nwbfile: NWBFile object containing behavior, position, and block data.
        spatial_series_name (str): Name of the SpatialSeries in the behavior position module
            (e.g. "cap_front").
        fig_dir (str): Optional directory to save figures. Per-block full heatmaps are saved as
            {spatial_series_name}_block_{N}_heatmap.png; the session summary is saved as
            {spatial_series_name}_all_blocks_heatmap.png.

    Returns:
        tuple[plt.Figure, list[plt.Figure]]: (session summary figure, list of per-block figures).
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
    fig_fulls = []

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
                ax=ax_full,
                barriers=maze,
                centroids=centroids_dict,
                snap_centroids=True,
                show_hex_labels=False,
                show_barriers=False,
                show_choice_points=False,
                reward_probabilities=reward_probs,
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
        fig_fulls.append(fig_full)

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
                    ax=ax,
                    barriers=maze,
                    centroids=centroids_dict,
                    snap_centroids=True,
                    show_hex_labels=False,
                    show_barriers=False,
                    show_choice_points=False,
                    reward_probabilities=reward_probs,
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

    return fig, fig_fulls


def plot_rat_position_by_trial(nwbfile, spatial_series_name, fig_dir=None):
    """
    For each block, plot the rat's position trajectory for every individual trial
    on a grid of subplots overlaid on the hex maze.

    Parameters:
        nwbfile: NWBFile object containing behavior, position, block, and trial data.
        spatial_series_name (str): Name of the SpatialSeries in the behavior position module
            (e.g. "cap_front_position", "cap_back_position").
        fig_dir (str): Optional directory to save figures. Each block is saved as a separate
            file named {spatial_series_name}_block_{N}_position_by_trial.png.

    Returns:
        list[plt.Figure]: One figure per block.
    """

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

    # Get block and trial data
    block_data = nwbfile.intervals["block"].to_dataframe()
    trial_data = nwbfile.intervals["trials"].to_dataframe()

    figs = []
    for i, block in enumerate(block_data.itertuples(index=False)):
        # Get maze configuration and reward probabilities for this block
        maze = block.maze_configuration
        reward_probs = [block.pA, block.pB, block.pC]

        # Filter trials to only those in this block
        block_trials = trial_data[trial_data["block"] == block.block]
        n_trials = len(block_trials)

        # Set up square-ish grid for trials
        ncols = int(np.ceil(np.sqrt(n_trials)))
        nrows = int(np.ceil(n_trials / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

        # Make sure axes is 1D so flatten doesn't break
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        else:
            axes = np.array(axes).flatten()

        # Loop over trials in this block
        for i, trial in enumerate(block_trials.itertuples(index=False)):
            # Get trial start/end
            start, end = trial.start_time, trial.stop_time
            df_trial = position_df[
                (position_df["timestamp"] >= start) & (position_df["timestamp"] <= end)
            ]

            axes[i].set_title(f"Trial {i+1}")
            plot_hex_maze(
                ax=axes[i],
                barriers=maze,
                centroids=centroids_dict,
                snap_centroids=True,
                invert_yaxis=True,
                show_barriers=False,
                show_hex_labels=False,
                show_stats=True,
                reward_probabilities=reward_probs,
            )

            # Plot rat position for this trial in black
            axes[i].scatter(df_trial["x"], df_trial["y"], s=1, color='k')

        # Hide unused axes
        for j in range(n_trials, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{nwbfile.session_id} rat position by trial (block {block.block})", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if fig_dir:
            save_path = os.path.join(fig_dir, f"{spatial_series_name}_block_{block.block}_by_trial.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        figs.append(fig)

    return figs


def plot_maze_on_video_frame(nwbfile, video_file_path, fig_dir=None):
    """
    For each block, plot the hex maze layout (using the stored hex centroids) overlaid on a video frame.

    The frame chosen for each block is the one closest to the block's midpoint in time,
    using the aligned video timestamps stored in the NWB. For probability change sessions, 
    we use the midpoint of the entire session (because the maze config does not change).

    Useful for verifying that hex centroids are correctly aligned with the camera view
    and that the maze configuration matches the actual maze used.

    Parameters:
        nwbfile: NWBFile object containing hex_centroids, block data, and video timestamps.
        video_file_path (str): Full path to the behavior video file (.avi or .mp4).
            The NWB only stores the video filename without path, so this must be provided explicitly.
        fig_dir (str): Optional directory to save figures as maze_overlay_block_{N}.png.

    Returns:
        list[plt.Figure]: One figure per block.
    """
    session_id = nwbfile.session_id
    behavior_module = nwbfile.processing["behavior"]

    # Get hex centroids in pixel coordinates
    if "hex_centroids" not in behavior_module.data_interfaces:
        raise ValueError("No hex_centroids found in nwbfile!")
    centroids_df = behavior_module.data_interfaces["hex_centroids"].to_dataframe()
    centroids_dict = centroids_df.set_index("hex")[["x", "y"]].apply(tuple, axis=1).to_dict()

    # Get aligned video timestamps from NWB to map block times to frame indices
    video_timestamps = nwbfile.processing["video_files"]["video"]["behavior_video"].timestamps[:]
    block_data = nwbfile.intervals["block"].to_dataframe()

    # For probability change sessions the maze config is the same every block,
    # so one plot (using the session midpoint) is sufficient.
    is_prob_change = block_data["task_type"].iloc[0] == "probability change"
    blocks_to_plot = block_data.iloc[:1] if is_prob_change else block_data

    figs = []
    for _, block in blocks_to_plot.iterrows():
        maze = block["maze_configuration"]
        block_num = int(block["block"])
        # Don't add reward probs for prob change sessions because we are only plotting one block and they change
        reward_probs = [block["pA"], block["pB"], block["pC"]] if not is_prob_change else None

        # Pick the frame closest to the midpoint of this block (or whole session for prob-change)
        if is_prob_change:
            session_mid = (block_data["start_time"].iloc[0] + block_data["stop_time"].iloc[-1]) / 2
            frame_idx = int(np.argmin(np.abs(video_timestamps - session_mid)))
        else:
            block_mid = (block["start_time"] + block["stop_time"]) / 2
            frame_idx = int(np.argmin(np.abs(video_timestamps - block_mid)))

        cap = cv2.VideoCapture(video_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(frame_rgb)

        # Overlay maze in pixel coordinates
        plot_hex_maze(
            barriers=maze,
            centroids=centroids_dict,
            ax=ax,
            show_hex_labels=True,
            show_barriers=True,
            reward_probabilities=reward_probs,
            invert_yaxis=False,
        )
        # Make hex fills translucent so the video frame is visible underneath.
        # Edges stay fully opaque so the maze outline is clear.
        for patch in ax.patches:
            fc = patch.get_facecolor()
            patch.set_facecolor((*fc[:3], 0.3))
        ax.set_xticks([])
        ax.set_yticks([])
        title = (
            f"{session_id} maze configuration overlay" if is_prob_change 
            else f"{session_id} maze configuration overlay (block {block_num})"
        )
        ax.set_title(title)
        fig.tight_layout()

        if fig_dir:
            fname = (
                "maze_config_video_overlay.png" if is_prob_change 
                else f"maze_config_video_overlay_block_{block_num}.png"
            )
            fig.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)

        figs.append(fig)

    return figs
