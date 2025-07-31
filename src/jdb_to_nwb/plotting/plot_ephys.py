import matplotlib.pyplot as plt
import pandas as pd
import os


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
    plt.figure(figsize=(16, 6))
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


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
    plt.figure(figsize=(10, 25))
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # Plot channel names on the electrode layout (colored by channel number)
    plt.figure(figsize=(10, 25))
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
