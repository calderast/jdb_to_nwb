from pynwb import NWBFile
import struct
import pandas as pd
import numpy as np
import os
import re
import json
import scipy.io
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from importlib.resources import files
from scipy.signal import butter, lfilter, hilbert, filtfilt
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import Lasso

from ndx_fiber_photometry import (
    FiberPhotometryResponseSeries,
    Indicator,
    ExcitationSource,
    OpticalFiber,
    Photodetector,
    DichroicMirror,
    FiberPhotometryTable,
    FiberPhotometry,
)
from .plotting.plot_photometry import (
    plot_signal_correlation,
    plot_photometry_signals,
)

# Get the location of the resources directory when the package is installed from pypi
__location_of_this_file = Path(files(__name__))
RESOURCES_DIR = __location_of_this_file / "resources" / "photometry"

# If the resources directory does not exist, we are probably running the code from the source directory
if not RESOURCES_DIR.exists():
    RESOURCES_DIR = __location_of_this_file.parent.parent / "resources" / "photometry"

DEVICES_PATH = RESOURCES_DIR / "photometry_devices.yaml"
MAPPINGS_PATH = RESOURCES_DIR / "photometry_mappings.yaml"
VIRUSES_PATH = RESOURCES_DIR / "virus_info.yaml"


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

        # Loop through the four channels and extract the
        # location, signal, frequency, max and min values in the same way
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
    return visits, start


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


def run_lockin_detection(phot, start, logger):
    """Run lockin detection to extract the modulated photometry signals

    Args:
    phot (dict): Dictionary of photometry data
    start (int): Number of samples to remove from the start of photometry signals
    for alignment
    """
    # Default args for lockin detection
    tau = 10
    filter_order = 5
    detrend=False
    full=True

    logger.info(f"Running lockin detection to extract modulated photometry signals with args: \n"
                f"tau={tau}, filter_order={filter_order}, detrend={detrend}, full={full}")

    # Get the necessary data from the phot structure
    detector = phot["data"][5, :]
    exc1 = phot["data"][6, :]
    exc2 = phot["data"][7, :]

    # Call lockin_detection function
    sig1, ref = lockin_detection(
        detector, exc1, exc2, phot["sampling_rate"], tau=tau, filter_order=filter_order, detrend=detrend, full=full
    )

    detector = phot["data"][2, :]
    exc1 = phot["data"][0, :]
    exc2 = phot["data"][1, :]

    # Call lockin_detection function for the second set of signals
    sig2, ref2 = lockin_detection(
        detector, exc1, exc2, phot["sampling_rate"], tau=tau, filter_order=filter_order, detrend=detrend, full=full
    )

    # Cut off the beginning of the signals to match behavioral data
    sig1 = sig1[start:]
    sig2 = sig2[start:]
    ref = ref[start:]
    ref2 = ref2[start:]

    loc = phot["channels"][2]["location"][:15]  # First 15 characters of the location
    logger.debug(f"The location of the fiber is: {loc}")

    # Create a dict with the relevant signals to match signals.mat returned by the original MATLAB processing code
    # NOTE: We don't use sig2 and ref2 - these may be removed in a future PR but are kept for posterity for now
    signals = {"sig1": sig1, "ref": ref, "sig2": sig2, "ref2": ref2, "loc": loc}
    return signals


def process_raw_labview_photometry_signals(phot_file_path, box_file_path, logger):
    """
    Process the .phot and .box files from Labview into a "signals" dict,
    replacing former MATLAB preprocessing code that created signals.mat

    Also adds the start time of the Labview recording as a datetime object
    to the returned signals dict
    """

    # Read .phot file from Labview into a dict
    logger.info("Reading LabVIEW .phot file into a dictionary...")
    phot_dict = read_phot_data(phot_file_path)

    # Print .phot file values to debug file
    logger.debug(f"Data read from LabVIEW .phot file at {phot_file_path}:")
    for phot_key in phot_dict:
        if phot_key == "pad":
            continue
        logger.debug(f"{phot_key}: {phot_dict[phot_key]}")

    # Read .box file from Labview into a dict
    logger.info("Reading LabVIEW .box file into a dictionary...")
    box_dict = read_box_data(box_file_path)

    # Print .box file values to debug file
    logger.debug(f"Data read from LabVIEW .box file at {box_file_path}:")
    for box_key in box_dict:
        if box_key == "pad":
            continue
        logger.debug(f"{box_key}: {box_dict[box_key]}")

    # Get timestamps of port visits in 10 kHz photometry sample time
    # And the start sample of the photometry signal
    visits, start = process_pulses(box_dict)

    # Run lockin detection to extract the modulated photometry signals
    signals = run_lockin_detection(phot_dict, start, logger)

    # Add visit times to the signals dict
    signals["visits"] = visits

    # Convert Labview photometry start time to datetime object and set timezone to Pacific Time
    photometry_start = datetime.strptime(f"{phot_dict['date']} {phot_dict['time']}".strip(), "%Y-%m-%d %H-%M-%S")
    photometry_start = photometry_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    logger.info(f"LabVIEW photometry start time: {photometry_start}")
    signals["photometry_start"] = photometry_start

    # Return signals dict equivalent to signals.mat (with added photometry_start datetime object)
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
    binary_mask: binary mask indicating which points are signals (peaks) and which are background
    (0 = signal/peak, 1=background)
    lambda_: Smoothing parameter. A larger value results in a smoother baseline.

    Returns:
    the fitted background vector
    """

    data_matrix = np.array(data)
    data_matrix = np.expand_dims(data_matrix, axis=0)  # Convert to 2D array (1, num_samples)
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


def airPLS(data, logger, lambda_=1e8, max_iterations=50):
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
    weights = np.ones(num_data_points)  # Set the initial weights to 1 to treat all points equally

    # Loop runs up to 'max_iterations' times to adjust the weights and fit the baseline
    for i in range(1, max_iterations + 1):
        # Use Whittaker smoothing to fit a baseline to the data using the updated weights
        baseline = whittaker_smooth(data, weights, lambda_)
        # Difference between data and baseline to calculate residuals. delta > 0 == peak, delta < 0 == background
        delta = data - baseline
        # Calculate how much data is below the baseline
        sum_of_neg_deltas = np.abs(delta[delta < 0].sum())

        # Convergence check: if sum_of_neg_deltas < 0.1% of the total data, or if max iterations is reached
        if sum_of_neg_deltas < 0.001 * (abs(data)).sum() or i == max_iterations:
            if i == max_iterations:
                logger.warning(
                    f"When calculating adaptive photometry baseline with airPLS, we "
                    f"reached maximum iterations ({max_iterations}) before convergence was achieved! "
                    f"Wanted sum_of_neg_deltas < {0.001 * (abs(data)).sum()}, "
                    f"got sum_of_neg_deltas = {sum_of_neg_deltas}"
                )
            break
        # Delta >= 0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        weights[delta >= 0] = 0
        # Update weights for the negative deltas. Gives more weight to larger residuals using an exponential
        weights[delta < 0] = np.exp(i * np.abs(delta[delta < 0]) / sum_of_neg_deltas)
        # Update weights for the first and last data points to ensure edges of data are not ignored or underweighed
        weights[0] = np.exp(i * (delta[delta < 0]).max() / sum_of_neg_deltas)
        weights[-1] = weights[0]
    return baseline


def import_ppd(ppd_file_path):
    '''
    Credit to the homie: https://github.com/ThomasAkam/photometry_preprocessing.git
    Edited so that this function only returns the data dictionary without the filtered data.
    Raw data is then filtered by the process_ppd_photometry function.

        Function to import pyPhotometry binary data files into Python.
        Returns a dictionary with the following items:
            'filename'      - Data filename
            'subject_ID'    - Subject ID
            'date_time'     - Recording start date and time (ISO 8601 format string)
            'end_time'      - Recording end date and time (ISO 8601 format string)
            'mode'          - Acquisition mode
            'sampling_rate' - Sampling rate (Hz)
            'LED_current'   - Current for LEDs 1 and 2 (mA)
            'version'       - Version number of pyPhotometry
            'analog_1'      - Raw analog signal 1 (volts)
            'analog_2'      - Raw analog signal 2 (volts)
            'analog_3'      - Raw analog signal 3 (if present, volts)
            'digital_1'     - Digital signal 1
            'digital_2'     - Digital signal 2 (if present)
            'pulse_inds_1'  - Locations of rising edges on digital input 1 (samples).
            'pulse_inds_2'  - Locations of rising edges on digital input 2 (samples).
            'pulse_times_1' - Times of rising edges on digital input 1 (ms).
            'pulse_times_2' - Times of rising edges on digital input 2 (ms).
            'time'          - Time of each sample relative to start of recording (ms)
    '''
    with open(ppd_file_path, "rb") as f:
        header_size = int.from_bytes(f.read(2), "little")
        data_header = f.read(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype("<u2"))
    # Extract header information
    header_dict = json.loads(data_header)
    volts_per_division = header_dict["volts_per_division"]
    sampling_rate = header_dict["sampling_rate"]
    # Extract signals.
    analog = data >> 1  # Analog signal is most significant 15 bits.
    digital = ((data & 1) == 1).astype(int)  # Digital signal is least significant bit.
    # Alternating samples are different signals.
    if "n_analog_signals" in header_dict.keys():
        n_analog_signals = header_dict["n_analog_signals"]
        n_digital_signals = header_dict["n_digital_signals"]
    else:  # Pre version 1.0 data file.
        n_analog_signals = 2
        n_digital_signals = 2
    analog_1 = analog[::n_analog_signals] * volts_per_division[0]
    analog_2 = analog[1::n_analog_signals] * volts_per_division[1]
    analog_3 = analog[2::n_analog_signals] * volts_per_division[0] if n_analog_signals == 3 else None
    digital_1 = digital[::n_analog_signals]
    digital_2 = digital[1::n_analog_signals] if n_digital_signals == 2 else None
    time = np.arange(analog_1.shape[0]) * 1000 / sampling_rate  # Time relative to start of recording (ms).

    # Extract rising edges for digital inputs.
    pulse_inds_1 = 1 + np.where(np.diff(digital_1) == 1)[0]
    pulse_inds_2 = 1 + np.where(np.diff(digital_2) == 1)[0] if n_digital_signals == 2 else None
    pulse_times_1 = pulse_inds_1 * 1000 / sampling_rate
    pulse_times_2 = pulse_inds_2 * 1000 / sampling_rate if n_digital_signals == 2 else None
    # Return signals + header information as a dictionary.
    data_dict = {
        "filename": os.path.basename(ppd_file_path),
        "analog_1": analog_1,
        "analog_2": analog_2,
        "digital_1": digital_1,
        "digital_2": digital_2,
        "pulse_inds_1": pulse_inds_1,
        "pulse_inds_2": pulse_inds_2,
        "pulse_times_1": pulse_times_1,
        "pulse_times_2": pulse_times_2,
        "time": time,
    }
    if n_analog_signals == 3:
        data_dict.update(
            {
                "analog_3": analog_3,
            }
        )
    data_dict.update(header_dict)
    return data_dict


def process_and_add_pyphotometry_to_nwb(nwbfile: NWBFile, ppd_file_path, logger, fig_dir=None):
    """
    Process pyPhotometry data from a .ppd file and add the processed signals to the NWB file.

    TODO: Update this function to have options for isosbestic vs ratiometric
    correction depending on the identities of analog_1, analog_2, etc.
    Wait until we know more about those use cases.

    For now, we assume the following (Jose's setup):
    analog_1: 470 nm (gACh4h)
    analog_2: 565 nm (rDA3m)
    analog_3: 405 nm (for ratiometric correction of gACh4h)

    Returns:
    dict with keys
    - sampling_rate: int (Hz)
    - port_visits: list of port visit times in seconds
    - photometry_start: datetime object marking the start time of photometry recording
    """

    logger.info("Assuming pyPhotometry signals: analog_1: 470 nm (gACh4h), analog_2: 565 nm (rDA3m), analog_3: 405 nm")
    ppd_data = import_ppd(ppd_file_path)
    raw_green = pd.Series(ppd_data['analog_1'])
    raw_red = pd.Series(ppd_data['analog_2'])
    raw_405 = pd.Series(ppd_data['analog_3'])
    relative_raw_signal = raw_green / raw_405

    # Get port visits (in photometry sample time) and sampling rate from ppd file
    visits = ppd_data['pulse_inds_1'][1:]
    logger.debug(f"There were {len(visits)} port visits recorded by pyPhotometry")
    sampling_rate = ppd_data['sampling_rate']
    logger.info(f"pyPhotometry sampling rate: {sampling_rate} Hz")

    # Convert pyphotometry photometry start time to datetime object and set timezone to Pacific Time
    photometry_start = datetime.strptime(ppd_data['date_time'], "%Y-%m-%dT%H:%M:%S.%f")
    photometry_start = photometry_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    logger.info(f"pyPhotometry start time: {photometry_start}")

    logger.debug("Read data from ppd file:")
    for phot_key in ppd_data:
        logger.debug(f"{phot_key}: {ppd_data[phot_key]}")

    # Plot the raw pyPhotometry signals
    plot_photometry_signals(visits=visits,
                            sampling_rate=sampling_rate,
                            signals=[raw_green, raw_405, relative_raw_signal, raw_red],
                            signal_labels=["Raw 470", "Raw 405", "Raw 470/405 ratio", "Raw 565"],
                            signal_colors=["blue", "purple", "grey", "red"],
                            title="Raw pyPhotometry signals",
                            signal_units=["V", "V", "ratio", "V"],
                            fig_dir=fig_dir)

    # Plot the correlation between the raw signals
    plot_signal_correlation(sig1=raw_405, sig2=raw_green, label1='Raw 405', label2='Raw 470', fig_dir=fig_dir)
    plot_signal_correlation(sig1=raw_405, sig2=raw_red, label1='Raw 405', label2='Raw 565', fig_dir=fig_dir)
    plot_signal_correlation(sig1=raw_green, sig2=raw_red, label1='Raw 470', label2='Raw 565', fig_dir=fig_dir)

    # Low pass filter at 10Hz to remove high frequency noise
    print('Filtering data...')
    lowpass_cutoff = 10
    logger.info(f'Filtering photometry signals with a low pass filter at {lowpass_cutoff} Hz'
                ' to remove high frequency noise...')
    b,a = butter(2, lowpass_cutoff, btype='low', fs=sampling_rate)
    green_denoised = filtfilt(b,a, raw_green)
    red_denoised = filtfilt(b,a, raw_red)
    ratio_denoised = filtfilt(b,a, relative_raw_signal)
    denoised_405 = filtfilt(b,a, raw_405)

    # High pass filter at 0.001Hz to removes drift due to photobleaching
    # Note that this will also remove any physiological variation in the signal on very slow timescales
    highpass_cutoff = 0.001
    logger.info(f"Filtering photometry signals with a high pass filter at {highpass_cutoff} Hz "
                "to remove drift due to photobleaching...")
    b,a = butter(2, highpass_cutoff, btype='high', fs=sampling_rate)
    green_highpass = filtfilt(b,a, green_denoised, padtype='even')
    red_highpass = filtfilt(b,a, red_denoised, padtype='even')
    ratio_highpass = filtfilt(b,a, ratio_denoised, padtype='even')
    highpass_405 = filtfilt(b,a, denoised_405, padtype='even')

    # Plot the correlation between the filtered signals of interest (ACh and DA)
    plot_signal_correlation(sig1=ratio_highpass, sig2=red_highpass,
                            label1='gACh4h 470/405 ratio', label2='rDA3m', fig_dir=fig_dir)

    # Z-score each signal to normalize the data
    print('Z-scoring photometry signals...')
    logger.info('Z-scoring filtered photometry signals...')
    green_zscored = np.divide(np.subtract(green_highpass,green_highpass.mean()),green_highpass.std())
    red_zscored = np.divide(np.subtract(red_highpass,red_highpass.mean()),red_highpass.std())
    zscored_405 = np.divide(np.subtract(highpass_405,highpass_405.mean()),highpass_405.std())
    ratio_zscored = np.divide(np.subtract(ratio_highpass,ratio_highpass.mean()),ratio_highpass.std())

    # Plot the processed pyPhotometry signals
    plot_photometry_signals(visits=visits,
                            sampling_rate=sampling_rate,
                            signals=[green_zscored, zscored_405, ratio_zscored, red_zscored],
                            signal_labels=["gACh4h 470nm", "gACh4h 405nm", "gACh4h 470/405 ratio", "rDA3m 565nm"],
                            signal_colors=["blue", "purple", "grey", "red"],
                            title="Processed pyPhotometry signals",
                            signal_units="Z-score",
                            fig_dir=fig_dir)

    # Add photometry signals to the NWB
    print("Adding photometry signals to NWB...")
    logger.info("Adding photometry signals to NWB...")

    # Find the rows of the FiberPhotometryTable that correspond to the blue (470nm), purple (405nm), 
    # and green (565 nm) LEDs. Our current setup uses PyPhotometry in the new maze room, so we assume 
    # we are using the Doric Blue LED, the Doric Purple LED, and the Doric Green LED. 

    # This is necessary to create the FiberPhotometryTableRegion objects for the raw signals
    # and to ensure that each signal is correctly associated with an excitation source
    # in the fiber photometry table. Note that for processed signals (e.g. 470/405 ratio),
    # we choose the max "signal" wavelength (e.g. 470) as the LED to associate with the series,
    # even though this series should technically be associated with both 470nm and 405nm LEDs.
    fiber_photometry_table = nwbfile.get_lab_meta_data("fiber_photometry").fiber_photometry_table

    blue_led_table_region = None
    purple_led_table_region = None
    green_led_table_region = None

    for row_index, excitation_source_obj in enumerate(fiber_photometry_table.excitation_source.data):
        name_lower = excitation_source_obj.name.lower()
        if "blue led" in name_lower:
            blue_led_table_region = fiber_photometry_table.create_fiber_photometry_table_region(
                region=[row_index], description="Blue LED"
            )
        elif "purple led" in name_lower:
            purple_led_table_region = fiber_photometry_table.create_fiber_photometry_table_region(
                region=[row_index], description="Purple LED"
            )
        elif "green led" in name_lower:
            green_led_table_region = fiber_photometry_table.create_fiber_photometry_table_region(
                region=[row_index], description="Green LED"
            )

    if blue_led_table_region is None:
        logger.error("Could not find a blue LED in fiber photometry table. Please check the devices.yaml file.")
        raise ValueError("Blue LED not found in fiber photometry table.")

    if purple_led_table_region is None:
        logger.error("Could not find a purple LED in fiber photometry table. Please check the devices.yaml file.")
        raise ValueError("Purple LED not found in fiber photometry table.")

    if green_led_table_region is None:
        logger.error("Could not find a green LED in fiber photometry table. Please check the devices.yaml file.")
        raise ValueError("Green LED not found in fiber photometry table.")

    raw_470_response_series = FiberPhotometryResponseSeries(
        name="raw_470",
        description="Raw 470 nm",
        data=raw_green.to_numpy(),
        unit="V",
        rate=float(sampling_rate),
        fiber_photometry_table_region=blue_led_table_region,
    )

    z_scored_470_response_series = FiberPhotometryResponseSeries(
        name="z_scored_470",
        description="Z-scored 470 nm",
        data=green_zscored,
        unit="z-score",
        rate=float(sampling_rate),
        fiber_photometry_table_region=blue_led_table_region,
    )

    raw_405_response_series = FiberPhotometryResponseSeries(
        name="raw_405",
        description="Raw 405 nm",
        data=raw_405.to_numpy(),
        unit="V",
        rate=float(sampling_rate),
        fiber_photometry_table_region=purple_led_table_region,
    )

    z_scored_405_response_series = FiberPhotometryResponseSeries(
        name="zscored_405",
        description="Z-scored 405nm. This is used to calculate the ratiometric index when using gACh4h",
        data=zscored_405,
        unit="z-score",
        rate=float(sampling_rate),
        fiber_photometry_table_region=purple_led_table_region,
    )

    raw_565_response_series = FiberPhotometryResponseSeries(
        name="raw_565",
        description="Raw 565 nm",
        data=raw_red.to_numpy(),
        unit="V",
        rate=float(sampling_rate),
        fiber_photometry_table_region=green_led_table_region,
    )

    z_scored_565_response_series = FiberPhotometryResponseSeries(
        name="zscored_565",
        description="Z-scored 565nm",
        data=red_zscored,
        unit="z-score",
        rate=float(sampling_rate),
        fiber_photometry_table_region=green_led_table_region,
    )

    raw_ratio_response_series = FiberPhotometryResponseSeries(
        name="raw_470_405_ratio",
        description="Raw ratiometric index of 470nm and 405nm",
        data=relative_raw_signal.to_numpy(),
        unit="V",
        rate=float(sampling_rate),
        fiber_photometry_table_region=blue_led_table_region,
    )

    z_scored_ratio_response_series = FiberPhotometryResponseSeries(
        name="zscored_470_405_ratio",
        description="Z-scored ratiometric index of 470nm and 405nm",
        data=ratio_zscored,
        unit="z-score",
        rate=float(sampling_rate),
        fiber_photometry_table_region=blue_led_table_region,
    )

    # Add the FiberPhotometryResponseSeries objects to the NWB
    nwbfile.add_acquisition(raw_405_response_series)
    nwbfile.add_acquisition(raw_470_response_series)
    nwbfile.add_acquisition(raw_565_response_series)
    nwbfile.add_acquisition(raw_ratio_response_series)
    nwbfile.add_acquisition(z_scored_405_response_series)
    nwbfile.add_acquisition(z_scored_470_response_series)
    nwbfile.add_acquisition(z_scored_565_response_series)
    nwbfile.add_acquisition(z_scored_ratio_response_series)

    # Convert port visits to seconds to use for alignment
    visits_in_seconds = [visit_time / sampling_rate for visit_time in visits]

    # Return photometry start time, sampling rate, and port visit times in seconds to use for alignment
    # Add 'signals_to_plot' indicating processed signals to plot aligned to port entry (after behavior is parsed)
    signals_to_plot = ["zscored_565", "zscored_470_405_ratio"]
    return {'sampling_rate': sampling_rate, 'port_visits': visits_in_seconds,
            'photometry_start': photometry_start, 'signals_to_plot': signals_to_plot}


def process_and_add_labview_to_nwb(nwbfile: NWBFile, signals, logger, fig_dir=None):
    """
    Process LabVIEW signals and add the processed signals to the NWB file.

    Assumes
    sig1: 470 nm (dLight signal wavelength)
    ref: 405 nm (dLight isosbestic wavelength)

    Returns:
    dict with keys
    - sampling_rate: int (Hz)
    - port_visits: list of port visit times in seconds
    - photometry_start: datetime object marking the start time of photometry recording,
    or None if we are starting from processed signals.mat so no start time was found
    """
    logger.debug("Using signals mat: ")
    logger.debug(signals)

    # Downsample the raw data from 10 kHz to 250 Hz by taking every 40th sample
    SR = 10000  # Original sampling rate of the photometry system (Hz)
    Fs = 250  # Target downsample frequency (Hz)
    print(f"Downsampling raw LabVIEW data to {Fs} Hz...")
    logger.info(f"Downsampling raw LabVIEW data from 10 kHz to {Fs} Hz by taking every {int(SR / Fs)}th sample...")
    # Use np.squeeze to deal with the fact that signals from our dict are 1D but signals.mat are 2D
    raw_reference = pd.Series(np.squeeze(signals["ref"])[:: int(SR / Fs)])
    raw_green = pd.Series(np.squeeze(signals["sig1"])[:: int(SR / Fs)])
    port_visits = np.divide(np.squeeze(signals["visits"]), SR / Fs).astype(int)

    # Get raw signal length and desired crop length (if one was specified)
    raw_signal_length_mins = len(raw_reference) / Fs / 60
    phot_end_time_mins = signals.get("phot_end_time_mins", 0) # default is 0 if the user didn't set one

    # We can't crop the photometry signal if the desired end time is after the signal ends! Warn if so
    if phot_end_time_mins > raw_signal_length_mins:
        logger.warning(f"Specified `phot_end_time_mins` ({phot_end_time_mins}) is longer than the raw signal length "
                       f"({raw_signal_length_mins} mins). The photometry signal will not be cropped.")
        phot_end_time_mins = raw_signal_length_mins
    # Log at debug level if we aren't cropping
    elif phot_end_time_mins == 0:
        logger.debug("No `phot_end_time_mins` specified, so the photometry signal will not be cropped "
                     "(this is the normal case).")
        phot_end_time_mins = raw_signal_length_mins
    # Log if we are cropping!
    else:
        logger.info(f"Cropping photometry signal from raw length ({raw_signal_length_mins} mins) "
                    f"to {phot_end_time_mins} mins.")

    # Convert end time into samples to crop
    samples_to_keep = int(phot_end_time_mins * 60 * Fs)

    # Crop raw photometry signals
    raw_reference = raw_reference[:samples_to_keep]
    raw_green = raw_green[:samples_to_keep]

    # Plot the raw LabVIEW signals
    plot_photometry_signals(visits=port_visits,
                            sampling_rate=Fs,
                            signals=[raw_green, raw_reference],
                            signal_labels=["Raw 470nm signal", "Raw 405nm signal"],
                            signal_colors=["blue", "purple"],
                            title="Raw LabVIEW photometry signals",
                            signal_units="a.u.",
                            fig_dir=fig_dir)

    # Plot the correlation between the raw 405nm and 470nm signals
    plot_signal_correlation(sig1=raw_green, sig2=raw_reference,
                            label1="Raw 470", label2="Raw 405", fig_dir=fig_dir)

    # Smooth the signals using a rolling mean
    smooth_window = int(Fs / 30)
    min_periods = 1  # Minimum number of observations required for a valid computation
    print("Smoothing the photometry signals using a rolling mean...")
    logger.info("Smoothing the photometry signals using a rolling mean "
                f"with smooth_window={int(Fs / 30)} and min_periods={min_periods}...")
    reference = np.array(raw_reference.rolling(window=smooth_window, min_periods=min_periods).mean()).reshape(
        len(raw_reference), 1
    )
    signal_green = np.array(raw_green.rolling(window=smooth_window, min_periods=min_periods).mean()).reshape(
        len(raw_green), 1
    )

    # Calculate a smoothed baseline for each signal using airPLS
    lam = 1e8  # Parameter to control how smooth the resulting baseline should be
    max_iter = 50
    print("Calculating a smoothed baseline using airPLS...")
    logger.info("Calculating a smoothed baseline using airPLS with "
                f"lambda={lam} and max_iterations={max_iter}...")
    ref_baseline = airPLS(data=raw_reference.T, logger=logger, lambda_=lam, max_iterations=max_iter).reshape(
        len(raw_reference), 1
    )
    green_baseline = airPLS(data=raw_green.T, logger=logger, lambda_=lam, max_iterations=max_iter).reshape(
        len(raw_green), 1
    )

    # Subtract the respective airPLS baseline from the smoothed signal and reference
    print("Subtracting the smoothed baseline...")
    logger.info("Subtracting the smoothed baseline...")
    baseline_subtracted_ref = reference - ref_baseline
    baseline_subtracted_green = signal_green - green_baseline

    # Standardize by Z-scoring the signals (assumes signals are Gaussian distributed)
    print("Standardizing the signals by Z-scoring...")
    logger.info("Standardizing the signals by Z-scoring (using median instead of mean)...")
    z_scored_reference = (baseline_subtracted_ref - np.median(baseline_subtracted_ref)) / np.std(
        baseline_subtracted_ref
    )
    z_scored_green = (baseline_subtracted_green - np.median(baseline_subtracted_green)) / np.std(
        baseline_subtracted_green
    )

    logger.info("Removing the contribution of movement artifacts from the green signal using a Lasso regression")
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
    # due to rat head movements, etc.

    # Calculate deltaF/F for the green signal
    print("Calculating deltaF/F...")
    logger.info("Calculating deltaF/F via subtraction (z_scored_green - z_scored_reference_fitted)...")
    z_scored_green_dFF = z_scored_green - z_scored_reference_fitted

    # Plot the processing steps for 470nm wavelength
    signals_to_plot = [raw_green, signal_green, baseline_subtracted_green, z_scored_green]
    signal_labels = ["Raw 470nm signal", "Smoothed 470nm signal",
                     "Baseline-subtracted 470nm signal", "Z-scored 470nm signal"]
    plot_photometry_signals(visits=port_visits,
                            sampling_rate=Fs,
                            signals=signals_to_plot,
                            signal_labels=signal_labels,
                            title="470nm signal processing",
                            signal_units=["a.u.", "a.u.", "a.u.", "Z-score"],
                            overlay_signals=[(green_baseline, 1, "red", "airPLS baseline")],
                            fig_dir=fig_dir)

    # Plot the processing steps for 405nm wavelength
    signals_to_plot = [raw_reference, reference, baseline_subtracted_ref, z_scored_reference]
    signal_labels = ["Raw 405nm signal", "Smoothed 405nm signal",
                     "Baseline-subtracted 405nm signal", "Z-scored 405nm signal"]
    plot_photometry_signals(visits=port_visits,
                            sampling_rate=Fs,
                            signals=signals_to_plot,
                            signal_labels=signal_labels,
                            title="405nm signal processing",
                            signal_units=["a.u.", "a.u.", "a.u.", "Z-score"],
                            overlay_signals=[(ref_baseline, 1, "red", "airPLS baseline")],
                            fig_dir=fig_dir)

    # Plot steps of isosbestic correction
    signals_to_plot = [z_scored_green, z_scored_reference, z_scored_reference_fitted, z_scored_green_dFF]
    signal_labels = ["Z-scored 470nm signal", "Z-scored 405nm signal",
                     "Predicted 470nm signal from 405nm signal", "Z-scored dF/F (post isosbestic correction)"]
    plot_photometry_signals(visits=port_visits,
                            sampling_rate=Fs,
                            signals=signals_to_plot,
                            signal_labels=signal_labels,
                            signal_colors=["blue", "purple", "gray", "green"],
                            title="dLight isosbestic correction",
                            signal_units=["Z-score", "Z-score", "Z-score", "Z-score"],
                            fig_dir=fig_dir)

    # Add photometry signals to the NWB
    print("Adding photometry signals to NWB...")
    logger.info("Adding photometry signals to NWB...")

    # Find the rows of the FiberPhotometryTable that correspond to the blue (470nm) 
    # and purple (405nm) LEDs. Our current setup uses LabVIEW in the old maze room, 
    # so we assume we are using the Thorlabs Blue LED and Thorlabs Purple LED (no green LED).

    # This is necessary to create the FiberPhotometryTableRegion objects for the raw signals
    # and to ensure that each signal is correctly associated with an excitation source
    # in the fiber photometry table. Note that for processed signals (e.g. dLight dF/F),
    # we choose the max "signal" wavelength (e.g. 470) as the LED to associate with the series,
    # even though this series should technically be associated with both 470nm and 405nm LEDs.
    fiber_photometry_table = nwbfile.get_lab_meta_data("fiber_photometry").fiber_photometry_table

    blue_led_table_region = None
    purple_led_table_region = None

    for row_index, excitation_source_obj in enumerate(fiber_photometry_table.excitation_source.data):
        name_lower = excitation_source_obj.name.lower()
        if "blue led" in name_lower:
            blue_led_table_region = fiber_photometry_table.create_fiber_photometry_table_region(
                region=[row_index], description="Blue LED"
            )
        elif "purple led" in name_lower:
            purple_led_table_region = fiber_photometry_table.create_fiber_photometry_table_region(
                region=[row_index], description="Purple LED"
            )
        elif "green led" in name_lower:
            logger.warning(f"Found green LED '{excitation_source_obj.name}' in fiber photometry table! "
                           "This is not expected for recording with LabVIEW in our setup and will be ignored.")

    if blue_led_table_region is None:
        logger.error("Could not find a blue LED in fiber photometry table. Please check the devices.yaml file.")
        raise ValueError("Blue LED not found in fiber photometry table.")

    if purple_led_table_region is None:
        logger.error("Could not find a purple LED in fiber photometry table. Please check the devices.yaml file.")
        raise ValueError("Purple LED not found in fiber photometry table.")

    # Create NWB FiberPhotometryResponseSeries objects for the relevant photometry signals
    z_scored_green_dFF_response_series = FiberPhotometryResponseSeries(
        name="z_scored_green_dFF",
        description="Z-scored green signal (470 nm) dF/F",
        data=z_scored_green_dFF.T[0],
        unit="dF/F",
        rate=float(Fs),
        fiber_photometry_table_region=blue_led_table_region,
        # Not a perfect mapping - this series combines data from both 470nm and 405nm,
        # but valuable for accessing these linked metadata
    )
    z_scored_reference_fitted_response_series = FiberPhotometryResponseSeries(
        name="z_scored_reference_fitted",
        description="Fitted Z-scored reference signal. This is the baseline for the dF/F calculation.",
        data=z_scored_reference_fitted.T[0],
        unit="F",
        rate=float(Fs),
        fiber_photometry_table_region=purple_led_table_region,
    )
    raw_green_response_series = FiberPhotometryResponseSeries(
        name="raw_green",
        description="Raw green signal, 470nm",
        data=raw_green.to_numpy(),
        unit="F",
        rate=float(Fs),
        fiber_photometry_table_region=blue_led_table_region,
    )
    raw_reference_response_series = FiberPhotometryResponseSeries(
        name="raw_reference",
        description="Raw reference signal (isosbestic control), 405nm",
        data=raw_reference.to_numpy(),
        unit="F",
        rate=float(Fs),
        fiber_photometry_table_region=purple_led_table_region,
    )

    # Add the FiberPhotometryResponseSeries objects to the NWB
    nwbfile.add_acquisition(z_scored_green_dFF_response_series)
    nwbfile.add_acquisition(z_scored_reference_fitted_response_series)
    nwbfile.add_acquisition(raw_green_response_series)
    nwbfile.add_acquisition(raw_reference_response_series)

    # Convert port visits to seconds to use for alignment
    visits_in_seconds = [visit_time / Fs for visit_time in port_visits]

    # Return photometry start time, sampling rate, and port visit times in seconds to use for alignment
    # Add 'signals_to_plot' indicating processed signals to plot aligned to port entry (after behavior is parsed)
    signals_to_plot = ['z_scored_green_dFF']
    return {'sampling_rate': Fs, 'port_visits': visits_in_seconds,
            'photometry_start': signals.get('photometry_start'), 'signals_to_plot': signals_to_plot}


def add_photometry_metadata(nwbfile: NWBFile, metadata: dict, logger):
    """Add photometry metadata to the NWB file.

    The keys in the metadata YAML are assumed to be the same as the keyword arguments used for the corresponding
    classes in the ndx-fiber-photometry extension. See the extension for more details:
    https://github.com/catalystneuro/ndx-fiber-photometry/tree/main
    """

    with open(DEVICES_PATH, "r") as f:
        devices = yaml.safe_load(f)

    with open(MAPPINGS_PATH, "r") as f:
        mappings = yaml.safe_load(f)

    with open(VIRUSES_PATH, "r") as f:
        viruses = yaml.safe_load(f)

    # Add excitation sources to the nwb
    added_excitation_sources = dict()
    if "excitation_sources" in metadata["photometry"]:
        excitation_source_names = metadata["photometry"]["excitation_sources"]
        for excitation_source_name in excitation_source_names:
            # Find the matching device by name in the photometry devices list
            for device in devices["excitation_sources"]:
                if device["name"] == excitation_source_name:
                    logger.info(
                        f"Excitation source '{excitation_source_name}' found in resources/photometry_devices.yaml"
                    )
                    # Create the ExcitationSource object and add it to the nwb
                    excitation_source_obj = ExcitationSource(**device)
                    nwbfile.add_device(excitation_source_obj)
                    added_excitation_sources[excitation_source_name] = excitation_source_obj
                    break
            else:
                logger.error(
                    f"Excitation source '{excitation_source_name}' not found " "in resources/photometry_devices.yaml"
                )
                raise ValueError(
                    f"Excitation source '{excitation_source_name}' not found in resources/photometry_devices.yaml"
                )
    else:
        logger.warning("No 'excitation_sources' found in photometry metadata.")

    # Add optic fibers to the nwb
    if "optic_fiber_implant_sites" in metadata["photometry"]:
        fiber_implant_sites = metadata["photometry"]["optic_fiber_implant_sites"]

        # Our current pipeline assumes that we only record from one fiber at a time
        # Quality check that exactly one fiber implant site has recording: true
        count_recording_sites = 0
        for fiber_implant_site in fiber_implant_sites:
            if fiber_implant_site.get("recording", False):
                count_recording_sites += 1
        if count_recording_sites != 1:
            logger.error(
                "There should be exactly one optic fiber implant site with 'recording: true'. "
                f"Found {count_recording_sites} sites with 'recording: true'."
            )
            raise ValueError(
                "There should be exactly one optic fiber implant site with 'recording: true'. "
                f"Found {count_recording_sites} sites with 'recording: true'."
            )

        for fiber_implant_site in fiber_implant_sites:
            fiber_name = fiber_implant_site["optic_fiber"]

            # Find the matching device by name in the photometry devices list
            for device in devices["optic_fibers"]:
                if device["name"] == fiber_name:
                    logger.info(f"Optic fiber '{fiber_name}' found in resources/photometry_devices.yaml")
                    # NWB does not allow duplicate device names, but we implant most fibers bilaterally.
                    # We add the hemisphere based on the ML coordinate + targeted location in parenthesis 
                    # after the name of the fiber to get around this, 
                    # e.g. 'Doric 0.66mm Flat 40mm Optic Fiber (left NAcc)'
                    hemisphere = "left" if fiber_implant_site["ml_in_mm"] < 0 else "right"
                    fiber_name = f"{fiber_name} ({hemisphere} {fiber_implant_site["targeted_location"]})"

                    # Create the OpticalFiber object and add it to the nwb
                    fiber_metadata = device.copy()
                    fiber_metadata["name"] = fiber_name
                    fiber_obj = OpticalFiber(**fiber_metadata)
                    nwbfile.add_device(fiber_obj)
                    break
            else:
                logger.error(f"Optic fiber '{fiber_name}' not found in resources/photometry_devices.yaml")
                raise ValueError(
                    f"Optic fiber '{fiber_name}' not found in resources/photometry_devices.yaml"
                )

            # Save the targeted location and coordinates of the recording fiber
            # This will be used in the fiber photometry table
            if fiber_implant_site.get("recording", False):
                recorded_fiber_obj = fiber_obj
                recorded_fiber_target_location = fiber_implant_site["targeted_location"]
                recorded_fiber_coordinates = (
                    fiber_implant_site["ap_in_mm"],
                    fiber_implant_site["ml_in_mm"],
                    fiber_implant_site["dv_in_mm"],
                )
    else:
        logger.warning("No 'optic_fibers' found in photometry metadata.")

    # Add photodetector to the nwb
    if "photodetector" in metadata["photometry"]:
        photodetector_name = metadata["photometry"]["photodetector"]

        # Find the matching device by name in the photometry devices list
        for device in devices["photodetectors"]:
            if device["name"] == photodetector_name:
                logger.info(f"Photodetector '{photodetector_name}' found in resources/photometry_devices.yaml")
                # Create the Photodetector object and add it to the nwb
                # Also add a DichroicMirror object based on photodetector info
                photodetector_obj = Photodetector(**device)
                dichroic_mirror_obj = DichroicMirror(
                    name=photodetector_obj.name + " Built-in Dichroic Mirror",
                    description="Built-in dichroic mirror for photodetector",
                    manufacturer=photodetector_obj.manufacturer,
                )
                nwbfile.add_device(photodetector_obj)
                nwbfile.add_device(dichroic_mirror_obj)
                break
        else:
            logger.error(f"Photodetector '{photodetector_name}' not found in resources/photometry_devices.yaml")
            raise ValueError(f"Photodetector '{photodetector_name}' not found in resources/photometry_devices.yaml")
    else:
        logger.warning("No 'photodetectors' found in photometry metadata.")

    # Add indicators to the nwb
    added_indicators = dict()
    if "virus_injections" in metadata["photometry"]:
        virus_injections = metadata["photometry"]["virus_injections"]
        for virus_injection in virus_injections:
            virus_name = virus_injection["virus_name"]

            # Find the matching virus by name in the indicator list
            for virus in viruses["indicators"]:
                if virus["name"] == virus_name:
                    logger.info(f"Virus '{virus_name}' found in indicators in resources/virus_info.yaml")
                    indicator_metadata = virus
                    break
            else:
                # NOTE: Un-comment this once opsins list is non-empty. For now, iterating on None breaks!
                # Find the matching virus by name in the opsin list
                # for virus in viruses["opsins"]:
                #     if virus["name"] == virus_name:
                #         logger.warning(
                #             f"Virus '{virus_name}' found in opsins list in resources/virus_info.yaml. "
                #             "Optogenetics is not yet implemented so this virus will be ignored."
                #         )
                #         break
                # else:
                # If the virus was not found in either list, raise an error
                logger.error(f"Virus '{virus_name}' not found in resources/virus_info.yaml")
                raise ValueError(f"Virus '{virus_name}' not found in resources/virus_info.yaml")

            if "titer_in_vg_per_mL" not in virus_injection:
                logger.warning(
                    f"Virus injection for '{virus_name}' does not have a 'titer_in_vg_per_mL' field in metadata."
                )
            if "volume_in_uL" not in virus_injection:
                logger.warning(f"Virus injection for '{virus_name}' does not have a 'volume_in_uL' field in metadata.")
            
            # NWB does not allow duplicate device names, but we inject most indicators bilaterally.
            # We add the hemisphere based on the ML coordinate + targeted location in parenthesis 
            # after the name of the indicator to get around this (e.g. 'dLight1.3b (left NAcc)')
            hemisphere = "left" if virus_injection["ml_in_mm"] < 0 else "right"
            indicator_name = f"{indicator_metadata['name']} ({hemisphere} {virus_injection['targeted_location']})"

            # Add the indicator to the nwb
            # There is no standard way to store titer/volume, so we add it to the description
            indicator = Indicator(
                name=indicator_name,
                description=(
                    f"{indicator_metadata['description']}. "
                    f"Titer in vg/mL: {virus_injection.get('titer_in_vg_per_mL', 'unknown')}. "
                    f"Volume in uL: {virus_injection.get('volume_in_uL', 'unknown')}."
                ),
                label=indicator_metadata["construct_name"],
                manufacturer=indicator_metadata["manufacturer"],
                injection_location=virus_injection["targeted_location"],
                injection_coordinates_in_mm=(
                    virus_injection["ap_in_mm"],
                    virus_injection["ml_in_mm"],
                    virus_injection["dv_in_mm"],
                ),
            )
            nwbfile.add_device(indicator)
            added_indicators[indicator.name] = indicator
    
    # Set up fiber photometry table with one row for each recorded indicator + excitation source combination
    # These mappings come from resources/photometry_mappings.yaml
    fiber_photometry_table = FiberPhotometryTable(
        name="fiber_photometry_table",
        description="fiber photometry table",
    )

    for indicator_obj in added_indicators.values():
        # We have added all optical fibers and indicators to the nwbfile as devices
        # However, we only record from a single optical fiber per session
        # So only add the indicators that are actually recorded by that fiber to the fiber photometry table
        # If the indicator injection coords are >0.5mm from the recorded fiber coordinates, don't add it
        distance_threshold_mm = 0.5
        injection_coords = np.array(indicator_obj.injection_coordinates_in_mm)
        fiber_coords = np.array(recorded_fiber_coordinates)
        distance = np.linalg.norm(injection_coords - fiber_coords)

        if distance > distance_threshold_mm:
            logger.info(
                f"Indicator '{indicator_obj.name}' at '{injection_coords}' is too far from "
                f"recording fiber '{recorded_fiber_obj.name}' at {fiber_coords} to have been recorded in this session. "
                f"(Distance {distance:.2f}mm exceeds threshold of {distance_threshold_mm}mm)"
            )
            logger.info(f"Skipping adding '{indicator_obj.name}' to the fiber photometry table, as it "
                        "was not recorded from in this session. It still exists in the nwbfile under 'Devices'.")
            continue  # don't add to fiber_photometry_table
        else:
            logger.info(f"Adding indicator '{indicator_obj.name}' to the fiber photometry table")

        # Remove the last parenthesis we added after the indicator name so we can find the correct mapping
        logger.debug(f"Finding mapping for indicator '{indicator_obj.name}'")
        indicator_name_for_mapping = re.sub(r'\s*\([^()]*\)\s*$', '', indicator_obj.name)

        if indicator_name_for_mapping in mappings:
            logger.debug(f"Found mapping for indicator '{indicator_name_for_mapping}'")
            mapped_excitation_sources = mappings[indicator_name_for_mapping]

            added_rows = 0
            for excitation_source_name in mapped_excitation_sources:
                if excitation_source_name in added_excitation_sources:
                    logger.info(
                        f"Mapping excitation source '{excitation_source_name}' to indicator '{indicator_obj.name}'"
                    )
                    excitation_source_obj = added_excitation_sources[excitation_source_name]

                    fiber_photometry_table.add_row(
                        location=recorded_fiber_target_location,
                        coordinates=recorded_fiber_coordinates,
                        optical_fiber=recorded_fiber_obj,
                        photodetector=photodetector_obj,
                        dichroic_mirror=dichroic_mirror_obj,
                        indicator=indicator_obj,  # <-- this changes in the loop
                        excitation_source=excitation_source_obj,  # <-- this changes in the loop
                    )
                    added_rows += 1
            if added_rows == 0:
                logger.error(
                    f"None of the mapped excitation sources for indicator '{indicator_name_for_mapping}' "
                    f"were found in the excitation sources added to the NWB file."
                )
                raise ValueError(
                    f"None of the mapped excitation sources for indicator '{indicator_name_for_mapping}' "
                    f"were found in the excitation sources added to the NWB file."
                )
        else:
            logger.error(
                f"No mapping found for the indicator {indicator_name_for_mapping} in resources/photometry_mapping.yaml"
            )
            raise ValueError(
                f"No mapping found for the indicator {indicator_name_for_mapping} in resources/photometry_mapping.yaml"
            )

    fiber_photometry_lab_meta_data = FiberPhotometry(
        name="fiber_photometry",
        fiber_photometry_table=fiber_photometry_table,
    )

    nwbfile.add_lab_meta_data(fiber_photometry_lab_meta_data)


def add_photometry(nwbfile: NWBFile, metadata: dict, logger, fig_dir=None):
    """
    Add photometry data to the NWB and return port visits
    in downsampled photometry time to use for alignment.

    The processing differs based on what photometry data
    is available as specified by the metadata dictionary:

    If "phot_file_path" and "box_file_path" exist in the metadata dict:
    - We are using LabVIEW and have not done any preprocessing to extract the
    modulated photometry signals
    - Read raw data from the LabVIEW files and run lockin detection to extract visits,
    raw green signal, and raw reference signal, then do signal processing and dF/F

    If "signals_mat_file_path" exists in the metadata dict:
    - We are using LabVIEW and the raw .phot and .box files have already
    been processed in MATLAB to create signals.mat (this is true for older recordings)
    - Load the "signals.mat" dictionary that contains raw green signal, raw reference
    signal, and port visit times and do the signal processing and dF/F
    - Note that if "signals_mat_file_path" and both "phot_file_path" and "box_file_path"
    have been specified, we default to processing the raw LabVIEW data and ignore signals.mat

    If "ppd_file_path" exists in the metadata dict:
    - We are using pyPhotometry and do processing accordingly. This is currently
    only implemented for Jose's case (gACh4h, ratiometric instead of isosbestic correction)
    """

    if "photometry" not in metadata:
        print("No photometry metadata found for this session. Skipping photometry conversion.")
        logger.info("No photometry metadata found for this session. Skipping photometry conversion.")
        return {}

    # Add photometry metadata to the NWB
    print("Adding photometry metadata to NWB...")
    logger.info("Adding photometry metadata to NWB...")
    add_photometry_metadata(nwbfile, metadata, logger)

    # If we have raw LabVIEW data (.phot and .box files)
    if "phot_file_path" in metadata["photometry"] and "box_file_path" in metadata["photometry"]:
        # Process photometry data from LabVIEW to create a signals dict of relevant photometry signals
        logger.info("Using LabVIEW for photometry!")
        logger.info("Processing raw .phot and .box files from LabVIEW...")
        print("Processing raw .phot and .box files from LabVIEW...")
        phot_file_path = metadata["photometry"]["phot_file_path"]
        box_file_path = metadata["photometry"]["box_file_path"]
        signals = process_raw_labview_photometry_signals(phot_file_path, box_file_path, logger)
        # Get the desired end time if we need to crop phot signals (default 0 is processed as no cropping)
        signals["phot_end_time_mins"] = metadata["photometry"].get("phot_end_time_mins", 0)
        photometry_data_dict = process_and_add_labview_to_nwb(nwbfile, signals, logger, fig_dir)

    # If we have already processed the LabVIEW .phot and .box files into signals.mat (true for older recordings)
    elif "signals_mat_file_path" in metadata["photometry"]:
        # Load signals.mat created by external MATLAB photometry processing code
        logger.info("Using LabVIEW for photometry!")
        logger.info("Processing signals.mat file of photometry signals from LabVIEW...")
        logger.warning("Using signals.mat instead of raw .phot and .box file means the exact photometry start time\n"
                       "(time of day) is not recorded. Using the raw LabVIEW data is preferred for this reason.")
        print("Processing signals.mat file of photometry signals from LabVIEW...")
        signals_mat_file_path = metadata["photometry"]["signals_mat_file_path"]
        signals = scipy.io.loadmat(signals_mat_file_path, matlab_compatible=True)
        # Get the desired end time if we need to crop phot signals (default 0 is processed as no cropping)
        signals["phot_end_time_mins"] = metadata["photometry"].get("phot_end_time_mins", 0)
        photometry_data_dict = process_and_add_labview_to_nwb(nwbfile, signals, logger, fig_dir)

    # If we have a ppd file from pyPhotometry
    elif "ppd_file_path" in metadata["photometry"]:
        # Process ppd file from pyPhotometry and add signals to the NWB
        logger.info("Using pyPhotometry for photometry!")
        logger.info("Processing ppd file from pyPhotometry...")
        print("Processing ppd file from pyPhotometry...")
        ppd_file_path = metadata["photometry"]["ppd_file_path"]
        photometry_data_dict = process_and_add_pyphotometry_to_nwb(nwbfile, ppd_file_path, logger, fig_dir)

    else:
        logger.error("The required photometry subfields do not exist in the metadata dictionary.")
        raise ValueError(
            "The required photometry subfields do not exist in the metadata dictionary.\n"
            "Remove the 'photometry' field from metadata if you do not have photometry data "
            "for this session, or specify the following: \n"
            "If you are using LabVIEW, you must include both 'phot_file_path' and 'box_file_path' "
            "to process raw LabVIEW data,\n"
            "OR 'signals_mat_file_path' if the initial preprocessing has already been done in MATLAB.\n"
            "If you are using pyPhotometry, you must include 'ppd_file_path'."
        )

    # Photometry visit times are now our ground truth visit times
    metadata["ground_truth_time_source"] = "photometry"
    metadata["ground_truth_visit_times"] = photometry_data_dict.get("port_visits")

    return photometry_data_dict
