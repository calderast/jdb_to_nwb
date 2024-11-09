from pynwb import NWBFile
import pandas as pd
import numpy as np
import warnings
import scipy.io  # for loading in signals.mat files
from scipy.sparse import diags, eye, csc_matrix  # for creating sparse matrices
from scipy.sparse.linalg import spsolve  # for solving sparse linear systems
from sklearn.linear_model import Lasso  # for Lasso regression

# Some of these imports are unused for now but will be used for photometry metadata
from ndx_fiber_photometry import (
    Indicator,
    OpticalFiber,
    ExcitationSource,
    Photodetector,
    DichroicMirror,
    BandOpticalFilter,
    EdgeOpticalFilter,
    FiberPhotometry,
    FiberPhotometryTable,
    FiberPhotometryResponseSeries,
    CommandedVoltageSeries,
)


def whittaker_smooth(data, binary_mask, lambda_):
    """
    Penalized least squares algorithm for fitting a smooth background to noisy data.
    Used by airPLS to adaptively fit a baseline to photometry data.

    Fits a smooth background to the input data by minimizing the sum of
    squared differences between the data and the background.
    Uses a binary mask to identify the signal regions in the data
    that are not part of the background calculation.
    Uses a penalty term lambda that discourages rapid changes and enforces smoothness.

    Args:
    data: 1D array representing the signal data
    binary_mask: binary mask indicating which points are signals (peaks) and which are background (0 = signal/peak, 1=background)
    lambda_: Smoothing parameter. A larger value results in a smoother baseline.

    Returns:
    the fitted background vector
    """

    data_matrix = np.matrix(data)
    data_size = data_matrix.size  # Size of the data matrix
    # Create an identity matrix the size of the data matrix in compressed sparse column (csc) format
    identity_matrix = eye(data_size, format="csc")
    # numpy.diff() does not work with sparse matrix. This is a workaround.
    diff_matrix = identity_matrix[1:] - identity_matrix[:-1]
    # Creates a diagonal matrix with the binary mask values on the diagonal
    diagonal_matrix = diags(binary_mask, 0, shape=(data_size, data_size))
    # Represents the combination of the diagonal matrix and the smoothness penalty term lambda_
    A = csc_matrix(diagonal_matrix + (lambda_ * diff_matrix.T * diff_matrix))
    # Represents the weighted data
    B = csc_matrix(diagonal_matrix * data_matrix.T)
    # Solves the linear system of equations to find the smoothed baseline
    smoothed_baseline = spsolve(A, B)
    return np.array(smoothed_baseline)


def airPLS(data, lambda_=1e8, max_iterations=50):
    """
    Adaptive iteratively reweighted Penalized Least Squares for baseline fitting (airPLS).
    DOI: 10.1039/b922045c

    This function is used to fit a baseline to the input data using
    adaptive iteratively reweighted Penalized Least Squares (airPLS).
    The baseline is fitted by minimizing the weighted sum of the squared differences
    between the input data and the baseline using a penalized least squares approach
    (Whittaker smoothing) with a penalty term lambda that encourages smoothness.
    Iteratively adjusts the weights and the baseline until convergence is achieved.

    Args:
    data: input data (i.e. photometry signal)
    lambda_: Smoothing parameter. A larger value results in a smoother baseline.
    max_iterations: Maximum number of iterations to adjust the weights and fit the baseline.

    Returns:
    the fitted baseline vector
    """

    num_data_points = data.shape[0]
    weights = np.ones(num_data_points)  # Set the intial weights to 1 to treat all points equally

    # Loop runs up to 'max_iterations' times to adjust the weights and fit the baseline
    for i in range(1, max_iterations + 1):
        # Use Whittaker smoothing to fit a baseline to the data using the updated weights
        baseline = whittaker_smooth(data, weights, lambda_)
        # Difference between data and baseline to calculate residuals. delta > 0 == peak. delta < 0 == background.
        delta = data - baseline
        # Calculate how much data is below the baseline
        sum_of_neg_deltas = np.abs(delta[delta < 0].sum())

        # Convergence check: if sum_of_neg_deltas < 0.1% of the total data, or if the maximum number of iterations is reached
        if sum_of_neg_deltas < 0.001 * (abs(data)).sum() or i == max_iterations:
            if i == max_iterations:
                warnings.warn(
                    f"Reached maximum iterations before convergence was achieved! "
                    f"Wanted sum_of_neg_deltas < {0.001 * (abs(data)).sum()}, got sum_of_neg_deltas = {sum_of_neg_deltas}"
                )
            break
        # Delta >= 0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        weights[delta >= 0] = 0
        # Updates the weights for the negative deltas. Gives more weight to larger residuals using an exponential function.
        weights[delta < 0] = np.exp(i * np.abs(delta[delta < 0]) / sum_of_neg_deltas)
        # Updates the weights for the first and last data points to ensure edges of data are not ignored or underweighed.
        weights[0] = np.exp(i * (delta[delta < 0]).max() / sum_of_neg_deltas)
        weights[-1] = weights[0]
    return baseline


def add_photometry_metadata(nwbfile: NWBFile, metadata: dict):
    # TODO for Ryan - add photometry metadata to NWB :)
    # https://github.com/catalystneuro/ndx-fiber-photometry/tree/main
    pass


def add_photometry(nwbfile: NWBFile, metadata: dict):
    print("Adding photometry...")

    # Get path to signals.mat (photometry data) from metadata file
    signals_mat_file_path = metadata["photometry"]["signals_mat_file_path"]
    signals = scipy.io.loadmat(signals_mat_file_path, matlab_compatible=True)

    # Downsample the raw data from 10 kHz to 250 Hz by taking every 40th sample
    print("Downsampling raw data to 250 Hz...")
    SR = 10000  # Original sampling rate of the photometry system (Hz)
    Fs = 250  # Target downsample frequency (Hz)
    raw_reference = pd.Series(signals["ref"][0][:: int(SR / Fs)])
    raw_green = pd.Series(signals["sig1"][0][:: int(SR / Fs)])
    port_visits = np.divide(signals["visits"][0], SR / Fs).astype(int)

    # Smooth the signals using a rolling mean
    print("Smoothing the photometry signals using a rolling mean...")
    smooth_window = int(Fs / 30)
    min_periods = 1  # Minimum number of observations required for a valid computation
    reference = np.array(raw_reference.rolling(window=smooth_window, min_periods=min_periods).mean()).reshape(
        len(raw_reference), 1
    )
    signal_green = np.array(raw_green.rolling(window=smooth_window, min_periods=min_periods).mean()).reshape(
        len(raw_green), 1
    )

    # Calculate a smoothed baseline for each signal using airPLS
    print("Calculating a smoothed baseline using airPLS...")
    lam = 1e8  # Parameter to control how smooth the resulting baseline should be
    max_iter = 50
    ref_baseline = airPLS(data=raw_reference.T, lambda_=lam, max_iterations=max_iter).reshape(len(raw_reference), 1)
    green_baseline = airPLS(data=raw_green.T, lambda_=lam, max_iterations=max_iter).reshape(len(raw_green), 1)

    # Subtract the respective airPLS baseline from the smoothed signal and reference
    print("Subtracting the smoothed baseline...")
    remove = 0  # Number of samples to remove from the beginning of the signals
    baseline_subtracted_ref = reference[remove:] - ref_baseline[remove:]
    baseline_subtracted_green = signal_green[remove:] - green_baseline[remove:]

    # Standardize by Z-scoring the signals (assumes signals are Gaussian distributed)
    print("Standardizing the signals by Z-scoring...")
    z_scored_reference = (baseline_subtracted_ref - np.median(baseline_subtracted_ref)) / np.std(
        baseline_subtracted_ref
    )
    z_scored_green = (baseline_subtracted_green - np.median(baseline_subtracted_green)) / np.std(
        baseline_subtracted_green
    )

    # Remove the contribution of signal artifacts from the green signal using a Lasso regression
    alpha = 0.0001  # Parameter to control the regularization strength. A larger value means more regularization
    # Create the Lasso model (Lasso = type of linear regression that uses L1 regularization to prevent overfitting)
    lin = Lasso(alpha=alpha, precompute=True, max_iter=1000, positive=True, random_state=9999, selection="random")
    # Fit the model (learn the relationship between the reference signal and green signal)
    lin.fit(z_scored_reference, z_scored_green)
    # Predict what the values of z_scored_green should be given z_scored_reference
    z_scored_reference_fitted = lin.predict(z_scored_reference).reshape(len(z_scored_reference), 1)
    # We use these predicted values from the reference signal as our baseline for the dF/F calculation because
    # it accounts for the changes in the green signal that are accompanied by changes in the reference signal
    # due to photobleaching, rat head movements, etc.

    # Calculate deltaF/F for the green signal
    print("Calculating deltaF/F...")
    z_scored_green_dFF = z_scored_green - z_scored_reference_fitted

    # Add photometry metadata to the NWB
    print("Adding photometry metadata to NWB ...")
    add_photometry_metadata(NWBFile, metadata)

    # Add actual photometry data to the NWB
    print("Adding photometry signals to NWB ...")

    # Create NWB FiberPhotometryResponseSeries objects for the relevant photometry signals
    z_scored_green_dFF_response_series = FiberPhotometryResponseSeries(
        name="z_scored_green_dFF",
        description="Z-scored green signal (470 nm) dF/F",
        data=z_scored_green_dFF.T[0],
        unit="dF/F",
        rate=float(Fs),
    )
    z_scored_reference_fitted_response_series = FiberPhotometryResponseSeries(
        name="z_scored_reference_fitted",
        description="Fitted Z-scored reference signal. This is the baseline for the dF/F calculation.",
        data=z_scored_reference_fitted.T[0],
        unit="F",
        rate=float(Fs),
    )
    raw_green_response_series = FiberPhotometryResponseSeries(
        name="raw_green",
        description="Raw green signal, 470nm",
        data=raw_green.to_numpy(),
        unit="F",
        rate=float(Fs),
    )
    raw_reference_response_series = FiberPhotometryResponseSeries(
        name="raw_reference",
        description="Raw reference signal (isosbestic control), 405nm",
        data=raw_reference.to_numpy(),
        unit="F",
        rate=float(Fs),
    )

    # Add the FiberPhotometryResponseSeries objects to the NWB
    nwbfile.add_acquisition(z_scored_green_dFF_response_series)
    nwbfile.add_acquisition(z_scored_reference_fitted_response_series)
    nwbfile.add_acquisition(raw_green_response_series)
    nwbfile.add_acquisition(raw_reference_response_series)

    # Return port visits in downsampled photometry time (250 Hz) to use for alignment
    return port_visits
