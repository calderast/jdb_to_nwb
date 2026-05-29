import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os


_OPENEPHYS_FS = 30_000        # Open Ephys sampling rate (Hz)
_INTAN_UV_PER_BIT = 0.195     # Intan RHD conversion factor
_TOTAL_CHANNELS = 264         # 256 CH + 8 ADC for Berke Lab probes

# Full-saturation and desaturated (blended 50% toward white) versions of the two shank colors
_SHANK_COLORS = ["black", "#cc0000"]
_SHANK_COLORS_FADED = ["#aaaaaa", "#dd9999"]



def _draw_traces(ax, t, data_slice, elec_sorted, scale):
    """Plot sorted traces onto ax and configure y-axis labels.

    elec_sorted must have columns:
        'shank', 'data_col', 'bad_channel', 'electrode_name', 'intan_channel'.
    """
    n = len(elec_sorted)
    shank_ids = elec_sorted["shank"].values
    bad_flags = elec_sorted["bad_channel"].values

    shank_changes = np.concatenate(([True], np.diff(shank_ids) != 0))
    color_idx = np.cumsum(shank_changes) - 1

    # Traces: faded for bad channels; labels: always full-saturation shank color
    trace_colors = [(_SHANK_COLORS_FADED if bad_flags[i] else _SHANK_COLORS)[color_idx[i] % 2]
                    for i in range(n)]
    label_colors = [_SHANK_COLORS[color_idx[i] % 2] for i in range(n)]

    for row_pos in range(n):
        trace = data_slice[:, int(elec_sorted.iloc[row_pos]["data_col"])]
        y_offset = float(n - 1 - row_pos)
        ax.plot(t, trace / scale + y_offset, color=trace_colors[row_pos], linewidth=0.5, rasterized=True)

    # Labels: electrode name left-padded to 6 chars, channel number right-padded to 3 digits
    # ytick i sits at y=i; row_pos r is plotted at y=n-1-r, so label for ytick i = row n-1-i
    tick_labels = [
        f"{elec_sorted.iloc[n - 1 - i]['electrode_name']:<6}  "
        f"{int(elec_sorted.iloc[n - 1 - i]['intan_channel']) + 1:>3}"
        for i in range(n)
    ]
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=6, fontfamily="monospace")
    ax.set_ylim(-0.7, n - 0.3)

    # Color each tick label to match its shank (full saturation, regardless of bad_channel)
    for mpl_label, color in zip(ax.get_yticklabels(), [label_colors[n - 1 - i] for i in range(n)]):
        mpl_label.set_color(color)


def _split_by_shank(elec_sorted):
    """Split a sorted electrode DataFrame into two halves by shank for side-by-side plotting."""
    shanks = sorted(elec_sorted["shank"].unique())
    mid = len(shanks) // 2
    left = elec_sorted[elec_sorted["shank"].isin(shanks[:mid])].reset_index(drop=True)
    right = elec_sorted[elec_sorted["shank"].isin(shanks[mid:])].reset_index(drop=True)
    return left, right


def _make_trace_figure(t, data_slice, elec_sorted, scale, start_time, duration):
    """Create a side-by-side trace figure split across shank halves."""
    left, right = _split_by_shank(elec_sorted)
    n_rows = max(len(left), len(right))

    trace_h_in = 0.15
    fig_h = max(6.0, n_rows * trace_h_in + 1.0)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(24, fig_h), sharey=False)

    _draw_traces(ax_l, t, data_slice, left, scale)
    _draw_traces(ax_r, t, data_slice, right, scale)

    for ax in (ax_l, ax_r):
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Raw ephys traces — {start_time}s to {start_time + duration}s "
        f"({len(elec_sorted)} channels, scale ±{scale:.0f} µV)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def load_electrode_info(metadata):
    """
    Load electrode geometry and impedance from the CSV files referenced in metadata.

    Replicates the core data loading from the conversion pipeline without requiring
    a logger or triggering any plots — intended for notebook / pre-conversion use.

    Parameters:
        metadata (dict):
            Full metadata dict. Requires 'ephys' subdict with keys
            'probe', 'impedance_file_path', and optionally 'plug_order',
            'min_impedance_ohms', 'max_impedance_ohms'.

    Returns:
        pd.DataFrame:
            One row per electrode with columns: electrode_name, shank, electrode,
            intan_channel, x_um, y_um, Channel Name, Port, Enabled,
            Impedance Magnitude at 1000 Hz (ohms), Impedance Phase at 1000 Hz (degrees),
            Series RC equivalent R (Ohms), Series RC equivalent C (Farads), bad_channel.
    """
    # Lazy import to avoid circular dependency (convert_raw_ephys imports from this module)
    from ..convert_raw_ephys import BERKE_LAB_PROBES, MIN_IMPEDANCE_OHMS, MAX_IMPEDANCE_OHMS

    probe_name = metadata["ephys"]["probe"][0]
    if probe_name not in BERKE_LAB_PROBES:
        raise ValueError(f"Unknown probe '{probe_name}'. Valid probes: {list(BERKE_LAB_PROBES)}")

    probe_info = BERKE_LAB_PROBES[probe_name]
    channel_map_df = pd.read_csv(probe_info["channel_map"])
    electrode_coords_df = pd.read_csv(probe_info["electrode_coords"])
    channel_coords_df = pd.merge(channel_map_df, electrode_coords_df, on=["shank", "electrode"], how="outer")

    plug_order = metadata["ephys"].get("plug_order", "chip_first")
    if plug_order == "chip_first":
        channel_coords_df = channel_coords_df.drop(columns=["cable_first"])
        channel_coords_df = channel_coords_df.rename(columns={"chip_first": "intan_channel"})
    else:
        channel_coords_df = channel_coords_df.drop(columns=["chip_first"])
        channel_coords_df = channel_coords_df.rename(columns={"cable_first": "intan_channel"})

    impedance_data = pd.read_csv(metadata["ephys"]["impedance_file_path"])
    impedance_data = impedance_data.drop(columns=["Channel Number"])
    impedance_data["intan_channel"] = range(len(impedance_data))

    electrode_info = channel_coords_df.merge(impedance_data, on="intan_channel", how="left")

    min_imp = float(metadata["ephys"].get("min_impedance_ohms", MIN_IMPEDANCE_OHMS))
    max_imp = float(metadata["ephys"].get("max_impedance_ohms", MAX_IMPEDANCE_OHMS))
    electrode_info["bad_channel"] = (
        electrode_info["Impedance Magnitude at 1000 Hz (ohms)"].lt(min_imp) |
        electrode_info["Impedance Magnitude at 1000 Hz (ohms)"].gt(max_imp)
    ).astype(int)

    return electrode_info


def plot_raw_ephys_traces_from_metadata(metadata, start_time=100.0, duration=1.5, fig_dir=None):
    """
    Plot a short window of raw ephys traces directly from Open Ephys files.

    Reads electrode geometry from the channel map and coordinate CSVs, loads
    a time slice of the continuous.dat via memmap, and plots traces ordered by
    electrode geometry. 
    
    No NWB conversion required — intended for pre-conversion quality check.

    Channels are sorted left-to-right by shank, then top-to-bottom within each
    shank by vertical position. Colors alternate black / red between shanks.

    Parameters:
        metadata (dict):
            Full metadata dict with an 'ephys' subdict containing at least
            'probe', 'impedance_file_path', and 'openephys_folder_path'.
        start_time (float):
            Start of the window to plot in seconds. Default 100.0.
        duration (float):
            Duration of the window in seconds. Default 1.5.
        fig_dir (str):
            Optional directory to save the figure as 'raw_ephys_traces.png'.

    Returns:
        plt.Figure
    """
    from ..convert_raw_ephys import find_open_ephys_paths

    electrode_info = load_electrode_info(metadata)

    # Locate the continuous.dat file
    openephys_folder_path = metadata["ephys"]["openephys_folder_path"]
    open_ephys_paths = find_open_ephys_paths(openephys_folder_path)
    recording_files = open_ephys_paths["recording_files"]
    if len(recording_files) != 1:
        raise ValueError(f"Expected 1 recording file, got {len(recording_files)}: {list(recording_files)}")
    continuous_dat_path = next(iter(recording_files.values()))

    # Load a time window from the continuous.dat via memory-map
    start_frame = int(start_time * _OPENEPHYS_FS)
    end_frame = int((start_time + duration) * _OPENEPHYS_FS)
    raw = np.memmap(continuous_dat_path, dtype="int16", mode="r").reshape(-1, _TOTAL_CHANNELS)
    # Only CH channels (columns 0-255), convert to µV
    data_slice = raw[start_frame:end_frame, :256].astype(float) * _INTAN_UV_PER_BIT
    t = np.arange(end_frame - start_frame) / _OPENEPHYS_FS

    # Sort left-to-right by shank, then top-to-bottom within shank (highest y_um first)
    elec_sorted = electrode_info.sort_values(
        ["shank", "y_um"], ascending=[True, False]
    ).reset_index(drop=True)

    # data_slice columns are indexed by intan_channel directly
    elec_sorted["data_col"] = elec_sorted["intan_channel"]
    
    # Set scale based on good channels only
    good_cols = elec_sorted.loc[elec_sorted["bad_channel"] == 0, "data_col"].astype(int).values
    good_data = data_slice[:, good_cols] if len(good_cols) > 0 else data_slice
    scale = max(np.nanstd(good_data) * 4, 1.0)

    fig = _make_trace_figure(t, data_slice, elec_sorted, scale, start_time, duration)

    if fig_dir:
        fig.savefig(os.path.join(fig_dir, "raw_ephys_traces.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_raw_ephys_traces(nwbfile, start_time=100.0, duration=1.5, fig_dir=None):
    """
    Plot a short window of raw ephys traces ordered by electrode geometry.

    Channels are sorted left-to-right by shank index, then top-to-bottom within
    each shank by vertical position (rel_y descending). Trace color alternates
    black and red between shanks so shank boundaries are immediately visible.

    Parameters:
        nwbfile (NWBFile):
            NWB file with an ElectricalSeries in acquisition and an electrodes table
            containing columns 'probe_shank', 'rel_y', and 'electrode_name'.
        start_time (float):
            Start of the window to plot, in seconds. Default 100.0.
        duration (float):
            Duration of the window in seconds. Default 1.5.
        fig_dir (str):
            Optional directory to save the figure as 'raw_ephys_traces.png'.

    Returns:
        plt.Figure
    """
    eseries = nwbfile.acquisition["ElectricalSeries"]
    timestamps = np.array(eseries.timestamps)

    start_frame = int(np.searchsorted(timestamps, start_time))
    end_frame = int(np.searchsorted(timestamps, start_time + duration))
    t = timestamps[start_frame:end_frame] - timestamps[start_frame]

    # Load the data window: shape (n_time, n_channels), in uV.
    # When loaded from HDF5, eseries.data is an h5py.Dataset (subscriptable).
    # During conversion it is H5DataIO(MicrovoltsSpikeInterfaceRecordingDataChunkIterator),
    # and H5DataIO.__getitem__ delegates to the iterator which is not subscriptable.
    try:
        data_slice = np.array(eseries.data[start_frame:end_frame, :], dtype=float)
    except TypeError:
        inner = getattr(eseries.data, 'data', eseries.data)
        if not hasattr(inner, 'conversion_factor_uv'):
            return None
        raw = inner.recording.get_traces(start_frame=start_frame, end_frame=end_frame, return_scaled=False)
        data_slice = raw.astype(float) * inner.conversion_factor_uv

    # Build per-column electrode info using the electrode table region.
    # eseries.electrodes.data[c] = electrode table row for data column c.
    electrode_table_df = nwbfile.electrodes.to_dataframe()
    region_indices = list(eseries.electrodes.data)
    elec_df = electrode_table_df.iloc[region_indices].reset_index(drop=True)
    elec_df["data_col"] = range(len(elec_df))

    # Sort left-to-right by shank, then top-to-bottom within shank (highest rel_y first)
    # Rename to match the shared helper's expected column names
    elec_sorted = elec_df.sort_values(
        ["probe_shank", "rel_y"], ascending=[True, False]
    ).reset_index(drop=True)
    elec_sorted = elec_sorted.rename(columns={
        "probe_shank": "shank", "rel_y": "y_um", "intan_channel_number": "intan_channel"
    })

    # Set scale based on good channels only
    good_cols = elec_sorted.loc[elec_sorted["bad_channel"] == 0, "data_col"].astype(int).values
    good_data = data_slice[:, good_cols] if len(good_cols) > 0 else data_slice
    scale = max(np.nanstd(good_data) * 4, 1.0)

    fig = _make_trace_figure(t, data_slice, elec_sorted, scale, start_time, duration)

    if fig_dir:
        fig.savefig(os.path.join(fig_dir, "raw_ephys_traces.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_raw_ephys_snippet(nwbfile, fig_dir=None):
    """
    Plot a raw ephys trace snippet around the 5th rewarded poke. Called during NWB conversion.

    Skips silently if the NWB file has no ElectricalSeries, no trials table, or fewer than
    5 rewarded trials. The window is centered 0.5s before the poke and runs for 1.5s total.
    I picked the 5th rewarded poke because ideally we would see theta on approach and maybe ripples after.

    Parameters:
        nwbfile (NWBFile): Fully assembled NWB file (ephys + behavior both added).
        fig_dir (str): Optional directory to save the figure.
        logger: Optional logger for warnings.
    """
    if "ElectricalSeries" not in nwbfile.acquisition or nwbfile.trials is None:
        return

    trials_df = nwbfile.trials.to_dataframe()
    rewarded = trials_df[trials_df["reward"] == 1]
    # If we have < 5 rewarded trials, skip
    if len(rewarded) < 5:
        return

    poke_time = float(rewarded.iloc[4]["poke_in"])
    start_time = max(0.0, poke_time - 1)
    plot_raw_ephys_traces(nwbfile=nwbfile, start_time=start_time, duration=2, fig_dir=fig_dir)


def plot_channel_map(probe_name, channel_coords, fig_dir=None):
    """
    Plot the channel map with channel numbers at their corresponding coordinates.

    Parameters:
        probe_name (str): 
            The name of the probe.
        channel_coords (pd.DataFrame): 
            Dataframe with columns 'x_um', 'y_um', and 'intan_channel'.
        fig_dir (str): 
            The directory to save the figure. If None, the figure will not be saved.
    """
    fig = plt.figure(figsize=(16, 6))
    plt.scatter(channel_coords["x_um"], channel_coords["y_um"], s=10, alpha=0)
    for _, row in channel_coords.iterrows():
        plt.text(row["x_um"], row["y_um"], row["intan_channel"], fontsize=8, ha='center', va='center')

    plt.title(f'{probe_name} Channel Map')
    plt.xlim(min(channel_coords["x_um"]) - 100, max(channel_coords["x_um"]) + 100)
    plt.ylim(min(channel_coords["y_um"]) - 100, max(channel_coords["y_um"]) + 100)
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')

    if fig_dir:
        save_name = f"{probe_name.lower().replace(' ', '_').replace(',', '').replace('-','')}"
        save_path = os.path.join(fig_dir, f"{save_name}_channel_coords.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_channel_impedances(probe_name, electrode_info, min_impedance, max_impedance, fig_dir=None):
    """
    Plot the channel map with colored squares based on impedance values.

    Parameters:
        probe_name (str): 
            The name of the probe.
        electrode_info (pd.DataFrame): 
            DataFrame with columns 'x_um', 'y_um', 'intan_channel', and 'Impedance Magnitude at 1000 Hz (ohms)'
        min_impedance (float): 
            Minimum acceptable impedance (Ohms)
        max_impedance (float): 
            Maximum acceptable impedance (Ohms)
        fig_dir (str): 
            Directory to save the figure. If None, the figure will not be saved.
    """

    def get_color(imp):
        if imp < min_impedance:
            return 'blue'
        if imp > max_impedance:
            return 'black'
        return 'green'

    # Get impedances and corresponding color
    impedance = electrode_info["Impedance Magnitude at 1000 Hz (ohms)"]
    colors = impedance.apply(get_color)

    # Channels without electrode coordinates appear in the impedance scatter but not the channel map
    # Mark them with an orange outline so we can distinguish them from recording electrodes
    has_coords = electrode_info[["x_um", "y_um"]].notna().all(axis=1)
    edge_colors = has_coords.map({True: 'k', False: 'orange'})

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Plot channel map colored by impedance
    axes[0].scatter(electrode_info["x_um"], electrode_info["y_um"], c=colors, s=200, marker='s', edgecolors='k')
    for _, row in electrode_info.iterrows():
        axes[0].text(row["x_um"], row["y_um"], str(row["intan_channel"]),
                     fontsize=8, ha='center', va='center', color='w')
    axes[0].set_title(f'{probe_name} Channel Impedances')
    axes[0].set_xlabel('X Position (μm)')
    axes[0].set_ylabel('Y Position (μm)')
    axes[0].set_xlim(electrode_info["x_um"].min() - 100, electrode_info["x_um"].max() + 100)
    axes[0].set_ylim(electrode_info["y_um"].min() - 100, electrode_info["y_um"].max() + 150)

    # Plot impedance for each channel
    axes[1].scatter(electrode_info["intan_channel"], impedance, c=colors, s=40, edgecolors=edge_colors, linewidths=1)
    axes[1].axhline(min_impedance, color='blue', linestyle='--', lw=0.8)
    axes[1].axhline(max_impedance, color='black', linestyle='--', lw=0.8)
    axes[1].set_xlabel("Intan Channel Number")
    axes[1].set_ylabel("Impedance (Ω)")
    axes[1].grid(True)

    # Legend
    legend_handles = [
        mpatches.Patch(color='blue', label=f'< {min_impedance/1e6:.1f} MΩ'),
        mpatches.Patch(color='green', label='Within range'),
        mpatches.Patch(color='black', label=f'> {max_impedance/1e6:.1f} MΩ'),
    ]
    axes[0].legend(handles=legend_handles, title="Electrode Impedance", loc='upper right')

    if fig_dir:
        save_name = f"{probe_name.lower().replace(' ', '_').replace(',', '').replace('-', '')}"
        save_path = os.path.join(fig_dir, f"{save_name}_channel_impedances.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_neuropixels(all_neuropixels_electrodes: pd.DataFrame, channel_info: pd.DataFrame, 
                     probe_name=None, fig_dir=None):
    """
    1. Plot all neuropixels electrodes in black with the recording electrodes highlighted in red
    2. Plot channel names on the recording electrode layout, colored by channel number

    Parameters:
        all_neuropixels_electrodes (pd.DataFrame): 
            Dataframe of all 5120 Neuropixels recording sites with columns 'x_um' and 'y_um'
        channel_info (pd.DataFrame): 
            Dataframe of 384 recording electrodes with columns 'x_um', 'y_um', 'channel_num', and 'channel_name'
        probe_name (str):
            Optional name of the probe, used in plot title and saved filename
        fig_dir (str): 
            The directory to save the figure. If None, the figure will not be saved.
    """
    # Plot all electrodes with recording electrodes highlighted
    fig1 = plt.figure(figsize=(10, 25))
    plt.scatter(all_neuropixels_electrodes["x_um"], all_neuropixels_electrodes["y_um"],
                color="black", s=1, label="All electrodes")
    plt.scatter(channel_info["x_um"], channel_info["y_um"], color="red", s=1, label="Recording electrodes")
    plt.title(f"Neuropixels 2.0 (multishank) Recording Electrode Layout{f' — {probe_name}' if probe_name else ''}")
    plt.xlabel("X (µm)")
    plt.ylabel("Y (µm)")
    plt.xlim(all_neuropixels_electrodes['x_um'].min() - 50, all_neuropixels_electrodes['x_um'].max() + 50)
    plt.legend()

    if fig_dir:
        save_path = os.path.join(fig_dir, f"neuropixels_recording_sites{f'_{probe_name}' if probe_name else ''}.png")
        fig1.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig1)

    # Plot channel names on the electrode layout (colored by channel number)
    fig2 = plt.figure(figsize=(10, 25))
    sc = plt.scatter(channel_info["x_um"], channel_info["y_um"], c=channel_info["channel_num"], cmap="viridis", s=10)
    cbar = plt.colorbar(sc)
    cbar.set_label("Channel Number")
    for _, row in channel_info.iterrows():
        plt.text(row["x_um"] + 1, row["y_um"], row["channel_name"], va="center", fontsize=5)
    plt.title(f"Neuropixels 2.0 (multishank) Channel Layout{f' — {probe_name}' if probe_name else ''}")
    plt.xlabel("X (µm)")
    plt.ylabel("Y (µm)")
    plt.xlim(all_neuropixels_electrodes['x_um'].min() - 50, all_neuropixels_electrodes['x_um'].max() + 50)

    if fig_dir:
        save_path = os.path.join(fig_dir, f"neuropixels_channel_layout{f'_{probe_name}' if probe_name else ''}.png")
        fig2.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)

    return fig1, fig2
