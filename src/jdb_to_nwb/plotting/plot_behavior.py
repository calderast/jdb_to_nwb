import os
import numpy as np
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


def plot_probability_matching(trial_data, block_data, fig_dir=None):
    """
    Plot within-session probability matching behavior.
    
    Shows:
    - Rolling average of port choice frequencies over trials
    - Reward delivery events at each port (as bars)
    - Block transitions and reward probabilities
    
    Parameters
    ----------
    trial_data : list of dict
        List of trial dictionaries from parse_arduino_text.
        Each dict should have keys: 'end_port', 'reward', 'block', 'trial_within_session'
    block_data : list of dict
        List of block dictionaries from parse_arduino_text.
        Each dict should have keys: 'block', 'pA', 'pB', 'pC'
    fig_dir : str or None
        Directory to save the figure. If None, figure is not saved.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Create mapping from port letters to indices
    port_map = {'A': 0, 'B': 1, 'C': 2}
    
    # Extract data from trial_data list
    num_trials = len(trial_data)
    x1 = np.linspace(0, num_trials, num_trials)
    
    # Create arrays for reward events at each port
    yA1 = np.zeros(num_trials)
    yB1 = np.zeros(num_trials)
    yC1 = np.zeros(num_trials)
    
    # Create arrays for port choice indicators
    choose_A = np.zeros(num_trials)
    choose_B = np.zeros(num_trials)
    choose_C = np.zeros(num_trials)
    
    # Process each trial
    for i, trial in enumerate(trial_data):
        end_port = trial['end_port']
        reward = trial['reward']
        
        # Mark reward events (offset by +2 for visibility)
        if reward == 1:
            if end_port == 'A':
                yA1[i] = reward + 2
            elif end_port == 'B':
                yB1[i] = reward + 2
            elif end_port == 'C':
                yC1[i] = reward + 2
        
        # Mark port choices
        if end_port == 'A':
            choose_A[i] = 1
        elif end_port == 'B':
            choose_B[i] = 1
        elif end_port == 'C':
            choose_C[i] = 1
    
    # Calculate rolling window averages for port choice frequency
    window = 10
    
    # Use pandas-like rolling calculation with numpy
    def rolling_mean(arr, window):
        """Calculate rolling mean with min_periods=1"""
        result = np.zeros(len(arr))
        for i in range(len(arr)):
            start_idx = max(0, i - window + 1)
            result[i] = np.mean(arr[start_idx:i+1])
        return result
    
    freq_A = rolling_mean(choose_A, window)
    freq_B = rolling_mean(choose_B, window)
    freq_C = rolling_mean(choose_C, window)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle('Within-Session Probability Matching', fontweight='bold', fontsize=26)
    
    # Main plot for port visit frequencies
    ax4 = plt.subplot2grid((18, 1), (3, 0), colspan=1, rowspan=15)
    ax4.plot(x1, freq_A, label='A', alpha=0.8, color='blue')
    ax4.plot(x1, freq_B, label='B', alpha=0.8, color='orange')
    ax4.plot(x1, freq_C, label='C', alpha=0.8, color='green')
    ax4.set_ylabel('Port Visits/trial', fontsize=20, fontweight='bold')
    ax4.set_ylim(0, 0.7)
    ax4.legend(bbox_to_anchor=(0.9, 1.4), loc=2, borderaxespad=0.)
    
    # Add block transition lines and probability labels
    for i, block in enumerate(block_data):
        block_num = block['block']
        # Get trials for this block
        block_trials = [t for t in trial_data if t['block'] == block_num]
        
        if block_trials:
            # Get the last trial index of this block
            last_trial_idx = block_trials[-1]['trial_within_session'] - 1
            first_trial_idx = block_trials[0]['trial_within_session'] - 1
            
            # Calculate midpoint for text placement
            xmid = int(np.mean([first_trial_idx, last_trial_idx]))
            
            # Draw vertical line at block boundary (after last trial)
            if i < len(block_data) - 1:  # Don't draw line after last block
                xstart = last_trial_idx + 1
                if i == 0:
                    ax4.axvline(x=xstart, color='r', linestyle='--', label='Block Change')
                else:
                    ax4.axvline(x=xstart, color='r', linestyle='--')
            
            # Add probability text labels using axes coordinates for y position
            # Place text at 90% of the y-axis height
            plt.text(xmid - 12, 0.9, str(int(block['pA'])) + ': ', 
                    fontsize='xx-large', fontweight='bold', color='b', 
                    transform=ax4.get_xaxis_transform())
            plt.text(xmid, 0.9, str(int(block['pB'])) + ': ',
                    fontsize='xx-large', fontweight='bold', color='orange',
                    transform=ax4.get_xaxis_transform())
            plt.text(xmid + 12, 0.9, str(int(block['pC'])),
                    fontsize='xx-large', fontweight='bold', color='g',
                    transform=ax4.get_xaxis_transform())
    
    ax4.legend()
    
    # Top subplot: Rewards at port A (blue bars)
    ax1 = plt.subplot2grid((18, 1), (0, 0), colspan=1, rowspan=1, sharex=ax4)
    ax1.bar(x1, yA1, color='blue')
    ax1.axis('off')
    
    # Middle subplot: Rewards at port B (orange bars)
    ax2 = plt.subplot2grid((18, 1), (1, 0), colspan=1, rowspan=1, sharex=ax4)
    ax2.bar(x1, yB1, color='orange')
    ax2.axis('off')
    
    # Bottom subplot: Rewards at port C (green bars)
    ax3 = plt.subplot2grid((18, 1), (2, 0), colspan=1, rowspan=1, sharex=ax4)
    ax3.bar(x1, yC1, color='g')
    ax3.axis('off')
    
    if fig_dir:
        plt.savefig(os.path.join(fig_dir, "probability_matching.png"), dpi=300, bbox_inches="tight")
    
    return fig
