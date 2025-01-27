import matplotlib.pyplot as plt
import os


def plot_channel_map(probe_name, channel_map, channel_geometry, fig_dir=None):
    """
    Plot the channel map with channel numbers at their corresponding coordinates.

    Args:
        probe_name (str): The name of the probe.
        channel_map (list): The channel map.
        channel_geometry (pd.DataFrame): The channel geometry with columns 'x' and 'y'.
        fig_dir (str): The directory to save the figure. If None, the figure will not be saved.
    """
    plt.figure(figsize=(10, 6))
    
    # First establish the plot limits using scatter (otherwise the text labels will take forever to draw)
    x_coords = channel_geometry.iloc[channel_map]['x']
    y_coords = channel_geometry.iloc[channel_map]['y']
    plt.scatter(x_coords, y_coords, alpha=0)  # invisible points to set boundaries
    
    # Then plot the text labels
    for channel_num in channel_map:
        x = channel_geometry.iloc[channel_num]['x']
        y = channel_geometry.iloc[channel_num]['y']
        plt.text(x, y, str(channel_num), fontsize=10, ha='center', va='center')

    plt.title(f'{probe_name} Channel Map')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')

    if fig_dir:
        save_path = os.path.join(fig_dir, "channel_map.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()