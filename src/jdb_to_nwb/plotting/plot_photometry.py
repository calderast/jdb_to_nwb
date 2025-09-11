import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def plot_photometry_signals(visits, sampling_rate, signals, signal_labels, title, 
                            signal_colors=None, signal_units=None, overlay_signals=None, fig_dir=None):
    """
    Plots photometry signals and port visit times.

    This function can be used to plot any signals, but our current use case is:
    - LabVIEW has 470nm and 405nm (dLight) 
    - pyPhotometry has 470nm, 405nm, and 470/405 ratio (gACh4h), and 565nm (rDA3m)

    Parameters:
        visits (list or np.array): Port visits in photometry sample time
        sampling_rate (float): Sampling rate of signals and port visits in Hz
        signals (list[np.array]): List of 1D arrays of signals to plot
        signal_labels (list[str]): List of names for each signal (e.g., ['470', '405', '565', '470/405'])
        title (str): Plot title
        signal_colors (list[str]): Optional list of colors for each signal. Defaults to matplotlib blue
        signal_units (str or list[str]): Optional unit to use for all signals, or list
                of units if different signals have different units. Defaults to a.u.
        overlay_signals (list[tuple]): Optional list of signals to overlay (used to show airPLS baselines).
                Each tuple contains:
                - The signal to overlay
                - The index of the subplot where the signal should be overlaid
                - The color for the overlay signal
                - The label for the overlay signal
    """
    # Sanity check
    assert len(signals) == len(signal_labels), "Each signal must have a corresponding label"

    # If no signal units are specified, default to arbitrary units (a.u.)
    if signal_units is None:
        signal_units = ["a.u."] * len(signals)
    # If a single unit, use it for all signals
    elif isinstance(signal_units, str):
        signal_units = [signal_units] * len(signals)
    # Or if it's a list, it must match the number of signals
    else:
        assert len(signal_units) == len(signals), "Length of signal_units must match the number of signals"

    # Set signal colors
    if signal_colors is None:
        signal_colors = ["#1f77b4"] * len(signals) # Default blue
    else:
        assert len(signal_colors) == len(signals), "Length of signal_colors must match the number of signals"

    # Convert timestamps to minutes
    xvals = np.arange(len(signals[0])) / sampling_rate / 60
    port_visit_times = np.array(visits) / sampling_rate / 60

    # Set up figure with a subplot for each signal
    n_signals = len(signals)
    fig, axs = plt.subplots(n_signals, 1, figsize=(16, 3 * n_signals + 2), sharex=True)
    fig.suptitle(title, fontsize=18)
    axs = np.atleast_1d(axs)

    # Plot each signal with its respective label, unit, and color
    for i, (signal, label, unit, color) in enumerate(zip(signals, signal_labels, signal_units, signal_colors)):
        # Plot the signal
        axs[i].plot(xvals, signal, color=color, lw=1.5, label=f'{label}')

        # Mark port visit times with short vertical lines (top 5% of the y axis limits)
        ymin_val = axs[i].get_ylim()[1] - (axs[i].get_ylim()[1] - axs[i].get_ylim()[0]) * 0.05
        axs[i].vlines(port_visit_times, ymin=ymin_val, ymax=axs[i].get_ylim()[1], color='k', lw=1, label='Port visits')

        # Overlay signals if specified
        if overlay_signals:
            for overlay_signal, overlay_subplot_idx, overlay_color, overlay_label in overlay_signals:
                if overlay_subplot_idx == i:
                    axs[i].plot(xvals, overlay_signal, color=overlay_color, 
                                lw=1.5, label=f'{overlay_label}')

        # Set title and ylabel
        axs[i].set_ylabel(f'{unit}', fontsize=10)
        axs[i].set_title(f'{label}', fontsize=15)
        axs[i].legend(loc='upper right')

    axs[-1].set_xlabel('Time (minutes)', fontsize=12)

    # Crop the x axis to only signal times
    margin = 0.01 * (xvals[-1] - xvals[0])
    axs[0].set_xlim(xvals[0] - margin, xvals[-1] + margin)
    plt.tight_layout()

    if fig_dir:
        save_file_name = f"{title.lower().replace(' ', '_').replace('/', '_')}.png"
        save_path = os.path.join(fig_dir, save_file_name)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_signal_correlation(sig1, sig2, label1, label2, fig_dir=None):
    """
    Plots the correlation between two photometry signals (e.g raw 470 and raw 405)

    Parameters:
        sig1 (list or np.array): The first signal to correlate
        sig2 (list or np.array): The second signal to correlate
        label1 (str): The name of the first signal
        label2 (str): The name of the second signal
    """
    slope, intercept, r_value, _, _ = linregress(x=sig1, y=sig2)

    # Plot correlation between sig1 and sig2
    fig, ax = plt.subplots()
    plt.scatter(sig1[::5], sig2[::5], alpha=0.1, marker='.')
    x = np.array(plt.xlim())
    ax.plot(x, intercept + slope * x, color='red')
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_title(f'{label2} vs {label1} correlation')

    # Add slope and R squared to the plot
    textstr = f'Slope: {slope:.3f}\nR-squared: {r_value**2:.3f}'
    ax.text(0.70, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    if fig_dir:
        save_file_name = f"{label1.lower().replace(' ', '_').replace('/', '_')}_vs_" \
                         f"{label2.lower().replace(' ', '_').replace('/', '_')}_correlation.png"
        save_path = os.path.join(fig_dir, save_file_name)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig