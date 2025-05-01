import os
import numpy as np
import matplotlib.pyplot as plt
from hexmaze import plot_hex_maze


def plot_maze_configs(nwbfile, fig_dir=None):
    """
    Plots an average photometry signal (DA or ACh) aligned to port entry, 
    split by rewarded and unrewarded trials. Also plots the full-session signal trace
    with rewarded and unrewarded poke times indicated.
    """
    session_id = nwbfile.session_id

    # Get trial and reward data
    block_data = nwbfile.intervals["blocks"].to_dataframe()
    maze_configurations = block_data["maze_configuration"]
    
    print(maze_configurations)

    # Now plot!!

    # # First plot: Average signal response across all trials
    # plt.figure(figsize=(8, 5))
    # plt.plot(time_vector, rewarded.mean(axis=0), label=f"rewarded (n={n_rewarded})", color="red")
    # plt.plot(time_vector, un_rewarded.mean(axis=0), label=f"unrewarded (n={n_unrewarded})", color="blue")
    # plt.axvline(0, linestyle="--", color="black", label="Poke In")
    # plt.xlabel("Time (s)")
    # plt.ylabel(f"{signal_name}")
    # plt.title(f"{signal_name} aligned to port entry ({session_id}, {n_trials} trials)")
    # plt.legend()
    # plt.tight_layout()

    # if fig_dir:
    #     save_path = os.path.join(fig_dir, f"{signal_name}_aligned_to_port_entry.png")
    #     plt.savefig(save_path, dpi=300, bbox_inches="tight")
    #     plt.close()
