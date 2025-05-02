import os
import matplotlib.pyplot as plt
from hexmaze import plot_hex_maze


def plot_trial_time_histogram(trial_data, fig_dir=None):
    """
    Plot a histogram of trial times.
    """
    # Get trial durations
    durations = [event['end_time'] - event['start_time'] for event in trial_data]

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(durations, bins=40, color='skyblue', edgecolor='black')
    plt.xlabel('Trial duration (s)')
    plt.ylabel('Number of trials')
    plt.title('Histogram of trial durations')
    plt.tight_layout()

    if fig_dir:
        plt.savefig(os.path.join(fig_dir, "histogram_of_trial_durations.png"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_maze_configurations(block_data, fig_dir=None):
    """
    Creates 2 plots of the maze configurations for each block.
    One plot shows all barriers (including permanent barriers), and
    the other shows only open hexes with choice points and optimal paths highlighted.
    """

    # Set up 2 1x(num blocks) plot so we can put each maze in a subplot
    fig1, axs1 = plt.subplots(1, len(block_data), figsize=((len(block_data) * 4, 4)))
    fig2, axs2 = plt.subplots(1, len(block_data), figsize=((len(block_data) * 4, 4)))

    for i, block in enumerate(block_data):
        # Get maze configuration and reward probabilities
        maze = block["maze_configuration"]
        reward_probs = [block["pA"], block["pB"], block["pC"]]

        # Plot the maze for this block (permanent barriers shown)
        plot_hex_maze(barriers=maze, ax=axs1[i], reward_probabilities=reward_probs, 
                        show_choice_points=False, show_hex_labels=False, show_permanent_barriers=True)
        axs1[i].set_title(f"Block {i+1}")
        axs1[i].set_xlabel(f"Barriers: {maze}")

        # Plot the maze for this block (only open hexes shown, optimal paths and choice points highlighted)
        plot_hex_maze(barriers=maze, ax=axs2[i], reward_probabilities=reward_probs, show_barriers=False,
                      show_hex_labels=False, show_choice_points=True, show_optimal_paths=True)
        axs2[i].set_title(f"Block {i+1}")
        axs2[i].set_xlabel(f"Barriers: {maze}")
    fig1.tight_layout()
    fig2.suptitle("Maze configurations with optimal paths and choice points highlighted", fontsize=14)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    if fig_dir:
        fig1.savefig(os.path.join(fig_dir, "maze_configurations.png"), dpi=300, bbox_inches="tight")
        fig2.savefig(os.path.join(fig_dir, "maze_configurations_with_optimal_paths.png"), dpi=300, bbox_inches="tight")
        plt.close(fig1)
        plt.close(fig2)
