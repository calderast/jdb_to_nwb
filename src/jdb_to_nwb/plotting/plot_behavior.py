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
    fig = plt.figure(figsize=(6, 4))
    plt.hist(durations, bins=40, color='skyblue', edgecolor='black')
    plt.xlabel('Trial duration (s)')
    plt.ylabel('Number of trials')
    plt.title('Histogram of trial durations')
    plt.tight_layout()

    if fig_dir:
        plt.savefig(os.path.join(fig_dir, "histogram_of_trial_durations.png"), dpi=300, bbox_inches="tight")
        plt.close()

    return fig


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

    return fig1, fig2


PORT_COLORS = {"A": "#1a3a8c", "B": "#3aa65a", "C": "#f5a623"}
PORT_Y = {"A": 3, "B": 2, "C": 1}

# Fixed margins in inches around each axes panel (for labels, title, etc.)
_LMARGIN, _RMARGIN, _BMARGIN, _TMARGIN = 0.55, 0.15, 0.55, 0.40
_ROW_GAP = 0.50  # vertical gap between rows in the by-block plot


def _draw_port_choices_on_ax(ax, block_trials, cell, n_cols=None, show_xlabel=True, row_label=None):
    """Draw port choice dots and grid on a pre-existing axes.

    n_cols sets the total x-axis width in data units. When n_cols > len(block_trials)
    the axes extends to the right with empty cells, keeping all rows left-justified
    to a common width in the by-block plot.
    """
    n = len(block_trials)
    if n_cols is None:
        n_cols = n
    dot_size = 80 * (cell / 0.15) ** 2
    open_lw = 1.4 * (cell / 0.15)

    for i, trial in enumerate(block_trials, start=1):
        port, reward = trial["end_port"], trial["reward"]
        color = PORT_COLORS[port]
        y = PORT_Y[port]
        if reward:
            ax.scatter(i, y, color=color, s=dot_size, zorder=2, linewidths=0)
        else:
            ax.scatter(i, y, facecolors="white", edgecolors=color, s=dot_size, linewidths=open_lw, zorder=2)

    # Ticks and grid span the full n_cols width; labels only shown for actual trials
    ax.set_xticks(range(1, n_cols + 1))
    ax.set_yticks([1, 2, 3])
    tick_labels = [str(j) if j <= n else "" for j in range(1, n_cols + 1)]
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticklabels(["C", "B", "A"], fontsize=9)
    ax.set_xticks(np.arange(0.5, n_cols + 1.5), minor=True)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5], minor=True)
    ax.set_xlim(0.5, n_cols + 0.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_aspect("equal")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, color="#cccccc", zorder=0)
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(True)
    for j, label in enumerate(ax.xaxis.get_ticklabels()):
        label.set_visible(j < n and (j + 1) % 10 == 0)
    if show_xlabel:
        ax.set_xlabel("Trial number", fontsize=10)
    ax.set_ylabel("Port", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if row_label:
        ax.set_title(row_label, fontsize=10, loc="left", pad=4)


def plot_port_choices(trial_data, fig_dir=None):
    """
    Plot port choices and reward outcomes across all trials as one long strip.

    Each trial is a circle on the row for its port (A/B/C): filled = rewarded,
    open = unrewarded. Red vertical lines mark block transitions.
    Cell width is capped so the figure stays under ~40 inches wide.

    Parameters:
        trial_data (list[dict]): Trial dicts with keys 'end_port', 'reward',
            'block', and 'trial_within_session'.
        fig_dir (str): Optional directory to save as port_choices_and_outcomes.png.

    Returns:
        tuple[plt.Figure, plt.Axes]
    """
    block_breaks = [trial["trial_within_session"] for i, trial in enumerate(trial_data[:-1])
                    if trial["block"] != trial_data[i + 1]["block"]]
    n_trials = len(trial_data)
    cell = min(0.15, 40 / n_trials)

    axes_w = n_trials * cell
    axes_h = 3 * cell
    fig_w = axes_w + _LMARGIN + _RMARGIN
    fig_h = axes_h + _BMARGIN + _TMARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([_LMARGIN / fig_w, _BMARGIN / fig_h, axes_w / fig_w, axes_h / fig_h])

    _draw_port_choices_on_ax(ax, trial_data, cell)

    for b in block_breaks:
        ax.axvline(b + 0.5, color="red", linewidth=1.0, zorder=3)
        ax.scatter(b + 0.5, 3.8, marker="v", color="red", s=60, zorder=4, clip_on=False)

    fig.suptitle("Port choices and reward outcomes", fontsize=11,
                 x=(_LMARGIN + axes_w / 2) / fig_w, y=1 - (_TMARGIN / 2) / fig_h)

    if fig_dir:
        fig.savefig(os.path.join(fig_dir, "port_choices_and_outcomes.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


def plot_port_choices_by_block(trial_data, fig_dir=None):
    """
    Plot port choices and reward outcomes with one row per block.

    Same dot encoding as plot_port_choices, but each block occupies its own
    row so the figure is shorter and wider blocks are easy to compare.

    Parameters:
        trial_data (list[dict]): Trial dicts with keys 'end_port', 'reward',
            'block', and 'trial_within_session'.
        fig_dir (str): Optional directory to save as port_choices_by_block.png.

    Returns:
        plt.Figure
    """
    blocks = {}
    for trial in trial_data:
        blocks.setdefault(trial["block"], []).append(trial)
    blocks = dict(sorted(blocks.items()))
    n_blocks = len(blocks)
    max_trials = max(len(v) for v in blocks.values())

    cell = min(0.15, 40 / max_trials)
    axes_w = max_trials * cell
    row_h = 3 * cell
    fig_w = axes_w + _LMARGIN + _RMARGIN
    fig_h = n_blocks * row_h + (n_blocks - 1) * _ROW_GAP + _BMARGIN + _TMARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, (block_num, block_trials) in enumerate(blocks.items()):
        # Stack rows from top to bottom
        bottom = (_BMARGIN + (n_blocks - 1 - i) * (row_h + _ROW_GAP)) / fig_h
        ax = fig.add_axes([_LMARGIN / fig_w, bottom, axes_w / fig_w, row_h / fig_h])
        _draw_port_choices_on_ax(ax, block_trials, cell, n_cols=max_trials,
                                 show_xlabel=(i == n_blocks - 1),
                                 row_label=f"Block {block_num}")

    fig.suptitle("Port choices and reward outcomes by block", fontsize=11,
                 x=(_LMARGIN + axes_w / 2) / fig_w, y=1 - (_TMARGIN / 2) / fig_h)

    if fig_dir:
        fig.savefig(os.path.join(fig_dir, "port_choices_by_block.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig