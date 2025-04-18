import matplotlib.pyplot as plt
import os


def plot_channel_map(probe_name, channel_geometry, fig_dir=None):
    """
    Plot the channel map with channel numbers at their corresponding coordinates.

    Args:
        probe_name (str): The name of the probe.
        channel_geometry (pd.DataFrame): The channel geometry with columns 'x' and 'y'.
        fig_dir (str): The directory to save the figure. If None, the figure will not be saved.
    """
    plt.figure(figsize=(10, 6))
    
    # First establish the plot limits using scatter (otherwise the text labels will take forever to draw)
    plt.scatter(channel_geometry['x'], channel_geometry['y'], alpha=0)  # invisible points to set boundaries
    
    # Then plot the text labels
    for channel_num in channel_geometry.index:
        x = channel_geometry.iloc[channel_num]['x']
        y = channel_geometry.iloc[channel_num]['y']
        plt.text(x, y, str(channel_num), fontsize=10, ha='center', va='center')

    plt.title(f'{probe_name} Channel Map')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')

    if fig_dir:
        save_path = os.path.join(fig_dir, "channel_coords.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    import os
    import pandas as pd
    import yaml

    with open("resources/ephys_devices.yaml", "r") as f:
        ephys_devices = yaml.safe_load(f)

    for probe in ephys_devices["probes"]:
        probe_name = probe["name"]
        channel_geometry = pd.read_csv(f"resources/{probe['electrode_coords']}")
        plot_channel_map(probe_name, channel_geometry, fig_dir="resources")
        os.rename("resources/channel_coords.png", f"resources/{probe_name}_channel_coords.png")
