from pynwb import NWBFile
import struct
import pandas as pd
import numpy as np
import warnings
import scipy.io
from scipy.signal import butter, lfilter, hilbert
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import Lasso

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


def read_phot_data(phot_file_path):
    """Parse .phot file from Labview into a dict"""

    phot = {}
    with open(phot_file_path, "rb") as fid:
        # Read binary data from the file, specifying the big-endian data format '>'
        phot["magic_key"] = struct.unpack(">I", fid.read(4))[0]
        phot["header_size"] = struct.unpack(">h", fid.read(2))[0]
        phot["main_version"] = struct.unpack(">h", fid.read(2))[0]
        phot["secondary_version"] = struct.unpack(">h", fid.read(2))[0]
        phot["sampling_rate"] = struct.unpack(">h", fid.read(2))[0]
        phot["bytes_per_sample"] = struct.unpack(">h", fid.read(2))[0]
        phot["num_channels"] = struct.unpack(">h", fid.read(2))[0]

        # After reading the character arrays, decode them from utf-8 and stripping the null characters (\x00)
        phot["file_name"] = fid.read(256).decode("utf-8").strip("\x00")
        phot["date"] = fid.read(256).decode("utf-8").strip("\x00")
        phot["time"] = fid.read(256).decode("utf-8").strip("\x00")

        # Loop through the four channels and extract the location, signal, frequency, max and min values in the same way
        phot["channels"] = []

        # Initialize a list of empty dictionaries for the channels
        for i in range(4):
            phot["channels"].append({})

        # Read and decode the location for all channels first
        for i in range(4):
            phot["channels"][i]["location"] = fid.read(256).decode("utf-8", errors="ignore").strip("\x00")

        # Read and decode the signal for all channels
        for i in range(4):
            phot["channels"][i]["signal"] = fid.read(256).decode("utf-8", errors="ignore").strip("\x00")

        # Read frequency for all channels
        for i in range(4):
            phot["channels"][i]["freq"] = struct.unpack(">h", fid.read(2))[0]

        # Read max voltage for all channels
        for i in range(4):
            phot["channels"][i]["max_v"] = struct.unpack(">h", fid.read(2))[0] / 32767.0

        # Read min voltage for all channels
        for i in range(4):
            phot["channels"][i]["min_v"] = struct.unpack(">h", fid.read(2))[0] / 32767.0

        phot["signal_label"] = []
        for signal in range(8):
            # phot['signal_label'].append(fid.read(256).decode('utf-8').strip('\x00'))
            signal_label = fid.read(256).decode("utf-8").strip("\x00")
            phot["signal_label"].append(signal_label)

        # Handle the padding by reading until the header size is reached
        position = fid.tell()
        pad_size = phot["header_size"] - position
        phot["pad"] = fid.read(pad_size)

        # Reshape the read data into a 2D array where the number of channels is the first dimension
        data = np.fromfile(fid, dtype=np.dtype(">i2"))
        phot["data"] = np.reshape(data, (phot["num_channels"], -1), order="F")

    return phot


def read_box_data(box_file_path):
    """Parse .box file from Labview into a dict"""

    box = {}
    with open(box_file_path, "rb") as fid:
        # Read binary data from the file, specifying the big-endian data format '>'
        box["magic_key"] = struct.unpack(">I", fid.read(4))[0]
        box["header_size"] = struct.unpack(">h", fid.read(2))[0]
        box["main_version"] = struct.unpack(">h", fid.read(2))[0]
        box["secondary_version"] = struct.unpack(">h", fid.read(2))[0]
        box["sampling_rate"] = struct.unpack(">h", fid.read(2))[0]
        box["bytes_per_sample"] = struct.unpack(">h", fid.read(2))[0]
        box["num_channels"] = struct.unpack(">h", fid.read(2))[0]

        # Read and decode file name, date, and time
        box["file_name"] = fid.read(256).decode("utf-8").strip("\x00")
        box["date"] = fid.read(256).decode("utf-8").strip("\x00")
        box["time"] = fid.read(256).decode("utf-8").strip("\x00")

        # Read channel locations
        box["ch1_location"] = fid.read(256).decode("utf-8").strip("\x00")
        box["ch2_location"] = fid.read(256).decode("utf-8").strip("\x00")
        box["ch3_location"] = fid.read(256).decode("utf-8").strip("\x00")

        # Get current file position
        position = fid.tell()

        # Calculate pad size and read padding
        pad_size = box["header_size"] - position
        box["pad"] = fid.read(pad_size)

        # Read the remaining data and reshape it
        data = np.fromfile(fid, dtype=np.uint8)
        box["data"] = np.reshape(data, (box["num_channels"], -1), order="F")

    return box


def process_pulses(box):
    """Extract port visit times from the box dict and return visit timestamps in 10kHz sample time"""
    diff_data = np.diff(box["data"][2, :].astype(np.int16))
    start = np.where(diff_data < -1)[0][0]
    pulses = np.where(diff_data > 1)[0]
    visits = pulses - start
    return visits


def lockin_detection(input_signal, exc1, exc2, Fs, tau=10, filter_order=5, detrend=False, full=True):
    tau /= 1000  # Convert to seconds
    Fc = 1 / (2 * np.pi * tau)
    fL = 0.01

    # High-pass filter design (Same as MATLAB filter design)
    b, a = butter(filter_order, Fc / (Fs / 2), "high")

    # Single-direction filtering to match MATLAB's 'filter'
    input_signal = lfilter(b, a, input_signal)

    # Demodulation
    demod1 = input_signal * exc1
    demod2 = input_signal * exc2

    # Trend filter design
    if detrend:
        b, a = butter(filter_order, [fL, Fc] / (Fs / 2))
    else:
        b, a = butter(filter_order, Fc / (Fs / 2))

    if not full:
        # Use lfilter for single-direction filtering
        sig1 = lfilter(b, a, demod1)
        sig2 = lfilter(b, a, demod2)
    else:
        # Full mode
        sig1x = lfilter(b, a, demod1)
        sig2x = lfilter(b, a, demod2)

        # Get imaginary part of the Hilbert transform for phase-shifted signal
        exc1_hilbert = np.imag(hilbert(exc1))
        exc2_hilbert = np.imag(hilbert(exc2))

        demod1 = input_signal * exc1_hilbert
        demod2 = input_signal * exc2_hilbert

        if detrend:
            b, a = butter(filter_order, [fL, Fc] / (Fs / 2))
        else:
            b, a = butter(filter_order, Fc / (Fs / 2))

        sig1y = lfilter(b, a, demod1)
        sig2y = lfilter(b, a, demod2)

        # Combine signals using Pythagorean theorem
        sig1 = np.sqrt(sig1x**2 + sig1y**2)
        sig2 = np.sqrt(sig2x**2 + sig2y**2)

    return sig1, sig2


def run_lockin_detection(phot):
    """Run lockin detection to extract the modulated photometry signals"""
    # Default args for lockin detection
    tau = 10
    filter_order = 5

    # Get the necessary data from the phot structure
    detector = phot["data"][5, :]
    exc1 = phot["data"][6, :]
    exc2 = phot["data"][7, :]

    # Call lockin_detection function
    sig1, ref = lockin_detection(
        detector, exc1, exc2, phot["sampling_rate"], tau=tau, filter_order=filter_order, detrend=False, full=True
    )

    detector = phot["data"][2, :]
    exc1 = phot["data"][0, :]
    exc2 = phot["data"][1, :]

    # Call lockin_detection function for the second set of signals
    sig2, ref2 = lockin_detection(
        detector, exc1, exc2, phot["sampling_rate"], tau=tau, filter_order=filter_order, detrend=False, full=True
    )

    # Cut off the beginning of the signals to match behavioral data
    # NOTE: We no longer do this - it will likely be removed in a future PR but keeping it for posterity for now
    remove = 0  # Number of samples to remove from the beginning of the signals
    sig1 = sig1[remove:]
    sig2 = sig2[remove:]
    ref = ref[remove:]
    ref2 = ref2[remove:]

    loc = phot["channels"][2]["location"][:15]  # First 15 characters of the location

    # Create a dictionary with the relevant signals to match signals.mat returned by the original MATLAB processing code
    # NOTE: We don't actually use sig2 and ref2 - these may be removed in a future PR but are kept for posterity for now
    signals = {"sig1": sig1, "ref": ref, "sig2": sig2, "ref2": ref2, "loc": loc}
    return signals


def process_raw_photometry_signals(phot_file_path, box_file_path):
    """Process the .phot and .box files from Labview into a "signals" dict, replacing former MATLAB preprocessing code that created signals.mat"""

    # Read .phot file from Labview into a dict
    phot_dict = read_phot_data(phot_file_path)

    # Read .box file from Labview into a dict
    box_dict = read_box_data(box_file_path)

    # Run lockin detection to extract the modulated photometry signals
    signals = run_lockin_detection(phot_dict)

    # Get timestamps of port visits in 10 kHz photometry sample time
    visits = process_pulses(box_dict)
    signals["visits"] = visits

    # Return signals dict equivalent to signals.mat
    return signals


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


def add_photometry(nwbfile: NWBFile, metadata: dict, preprocessed: bool = True):
    """
    Add photometry data to the NWB.

    Option to do the initial photometry processing separately in MATLAB (specify preprocessed=True),
    in which case "signals_mat_file_path" must exist in the metadata dict.
    Alternatively, create "signals" from the raw .phot and .box files returned by Labview (preprocessed=False),
    in which case "phot_file_path" and "box_file_path" must exist in the metadata dict.
    Argument "preprocessed" is True by default (starts from existing signals.mat file)

    We may ultimately remove the "preprocessed" argument and make this decision based on if signals.mat
    exists or not. Note the preprocessed=False mode has not yet been extensively tested.
    """

    print("Adding photometry...")

    # If we don't have signals.mat, process the raw .phot and .box files from Labview
    if not preprocessed:
        # Process photometry data from Labview to create dict of relevant photometry signals
        phot_file_path = metadata["photometry"]["phot_file_path"]
        box_file_path = metadata["photometry"]["box_file_path"]
        signals = process_raw_photometry_signals(phot_file_path, box_file_path)
    # If we do have signals.mat, load it and go from there!
    else:
        # Load signals.mat created by external MATLAB photometry processing code
        signals_mat_file_path = metadata["photometry"]["signals_mat_file_path"]
        signals = scipy.io.loadmat(signals_mat_file_path, matlab_compatible=True)

    # Downsample the raw data from 10 kHz to 250 Hz by taking every 40th sample
    print("Downsampling raw data to 250 Hz...")
    SR = 10000  # Original sampling rate of the photometry system (Hz)
    Fs = 250  # Target downsample frequency (Hz)
    # Use np.squeeze to deal with the fact that signals from our dict are 1D but signals.mat are 2D
    raw_reference = pd.Series(np.squeeze(signals["ref"])[:: int(SR / Fs)])
    raw_green = pd.Series(np.squeeze(signals["sig1"])[:: int(SR / Fs)])
    port_visits = np.divide(np.squeeze(signals["visits"]), SR / Fs).astype(int)

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
