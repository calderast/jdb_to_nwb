from pynwb import NWBFile
import struct
import pandas as pd
import numpy as np
import os
import re
import json
import scipy.io
import yaml
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from importlib.resources import files
from scipy.signal import butter, lfilter, hilbert, filtfilt
from scipy.optimize import curve_fit
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
PRESETS_PATH = RESOURCES_DIR / "processing_presets.yaml"

# Map wavelength strings to LED color keywords for fiber photometry table lookup
WAVELENGTH_TO_LED_COLOR = {
    "470nm": "blue",
    "405nm": "purple",
    "565nm": "green",
}

# Plot colors per indicator. The reference channel (405nm isosbestic) always uses REFERENCE_PLOT_COLOR.
# Add new indicators here as needed; unknown indicators fall back to DEFAULT_SIGNAL_PLOT_COLOR.
INDICATOR_PLOT_COLORS = {
    "dLight1.3b":    "#2CA02C", # green
    "dLight3.8":     "#2CA02C", # green
    "gACh4h":        "#2CA02C", # green
    "rDA3m (AAV9)":  "#D62728", # red
    "rDA3m (rAAV)":  "#D62728", # red
}
REFERENCE_PLOT_COLOR = "#8FBF8F"   # 405nm reference for dLight and gACh4h. green, but desaturated.
DEFAULT_SIGNAL_PLOT_COLOR = "#1f77b4"   # matplotlib default blue


############################ Functions to read raw photometry data ############################

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
    start (int): Number of samples to remove from the start of photometry signals for alignment
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

    # Cut off the beginning of the signals to match behavioral data
    sig1 = sig1[start:]
    ref = ref[start:]

    loc = phot["channels"][2]["location"][:15]  # First 15 characters of the location
    logger.debug(f"The location of the fiber is: {loc}")

    # Create a dict with the relevant signals to match signals.mat returned by the original MATLAB processing code
    signals = {"sig1": sig1, "ref": ref, "loc": loc}
    return signals


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


@dataclass
class PhotometrySignalBundle:
    """Standardized container for loaded photometry data, before processing."""
    signals: dict  # {"470nm": np.ndarray, "405nm": np.ndarray, ...}
    sampling_rate: float
    port_visits: np.ndarray
    photometry_start: datetime | None
    source: str  # "labview_raw", "labview_mat", "pyphotometry"


def load_labview_raw(phot_file_path, box_file_path, logger, phot_end_time_mins=0):
    """Load raw LabVIEW .phot + .box files, run lock-in detection, downsample to 250 Hz.

    Returns a PhotometrySignalBundle with signals keyed by wavelength.
    """
    # Read .phot file
    logger.info("Reading LabVIEW .phot file into a dictionary...")
    phot_dict = read_phot_data(phot_file_path)
    logger.debug(f"Data read from LabVIEW .phot file at {phot_file_path}:")
    for phot_key in phot_dict:
        if phot_key == "pad":
            continue
        logger.debug(f"{phot_key}: {phot_dict[phot_key]}")

    # Read .box file
    logger.info("Reading LabVIEW .box file into a dictionary...")
    box_dict = read_box_data(box_file_path)
    logger.debug(f"Data read from LabVIEW .box file at {box_file_path}:")
    for box_key in box_dict:
        if box_key == "pad":
            continue
        logger.debug(f"{box_key}: {box_dict[box_key]}")

    # Get timestamps of port visits in 10 kHz photometry sample time
    # And the start sample of the photometry signal
    visits, start = process_pulses(box_dict)

    # Run lockin detection to extract the modulated photometry signals
    lockin_signals = run_lockin_detection(phot_dict, start, logger)

    # Downsample from 10 kHz to 250 Hz
    SR = 10000
    Fs = 250
    print(f"Downsampling raw LabVIEW data to {Fs} Hz...")
    logger.info(f"Downsampling raw LabVIEW data from 10 kHz to {Fs} Hz by taking every {int(SR / Fs)}th sample...")
    raw_470 = np.squeeze(lockin_signals["sig1"])[:: int(SR / Fs)]
    raw_405 = np.squeeze(lockin_signals["ref"])[:: int(SR / Fs)]
    port_visits = np.divide(np.squeeze(visits), SR / Fs).astype(int)

    # Handle cropping (if needed)
    raw_470, raw_405 = crop_signals([raw_470, raw_405], Fs, phot_end_time_mins, logger)

    # Convert Labview photometry start time to datetime object and set timezone to Pacific Time
    photometry_start = datetime.strptime(f"{phot_dict['date']} {phot_dict['time']}".strip(), "%Y-%m-%d %H-%M-%S")
    photometry_start = photometry_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    logger.info(f"LabVIEW photometry start time: {photometry_start}")

    return PhotometrySignalBundle(
        signals={"470nm": raw_470, "405nm": raw_405},
        sampling_rate=Fs,
        port_visits=port_visits,
        photometry_start=photometry_start,
        source="labview_raw",
    )


def load_labview_mat(signals_mat_path, logger, phot_end_time_mins=0):
    """Load preprocessed LabVIEW signals.mat file.

    Returns a PhotometrySignalBundle with signals keyed by wavelength.
    """
    logger.warning(
        "Using signals.mat instead of raw .phot and .box file means the exact photometry start time\n"
        "(time of day) is not recorded. Using the raw LabVIEW data is preferred for this reason."
    )
    signals = scipy.io.loadmat(signals_mat_path, matlab_compatible=True)
    logger.debug("Using signals mat: ")
    logger.debug(signals)

    # Downsample from 10 kHz to 250 Hz
    SR = 10000
    Fs = 250
    print(f"Downsampling raw LabVIEW data to {Fs} Hz...")
    logger.info(f"Downsampling raw LabVIEW data from 10 kHz to {Fs} Hz by taking every {int(SR / Fs)}th sample...")
    raw_470 = np.squeeze(signals["sig1"])[:: int(SR / Fs)]
    raw_405 = np.squeeze(signals["ref"])[:: int(SR / Fs)]
    port_visits = np.divide(np.squeeze(signals["visits"]), SR / Fs).astype(int)

    # Handle cropping (if needed)
    raw_470, raw_405 = crop_signals([raw_470, raw_405], Fs, phot_end_time_mins, logger)

    return PhotometrySignalBundle(
        signals={"470nm": raw_470, "405nm": raw_405},
        sampling_rate=Fs,
        port_visits=port_visits,
        photometry_start=None,  # No start time from signals.mat
        source="labview_mat",
    )


def load_pyphotometry(ppd_file_path, logger, phot_end_time_mins=0):
    """Load pyPhotometry .ppd file.

    Returns a PhotometrySignalBundle with signals keyed by wavelength.
    2-signal case: {"470nm": analog_1, "405nm": analog_2}
    3-signal case: {"470nm": analog_1, "565nm": analog_2, "405nm": analog_3}
    """
    ppd_data = import_ppd(ppd_file_path)

    logger.debug("Read data from ppd file:")
    for phot_key in ppd_data:
        logger.debug(f"{phot_key}: {ppd_data[phot_key]}")

    # Get port visits and sampling rate
    visits = ppd_data['pulse_inds_1'][1:]
    logger.debug(f"There were {len(visits)} port visits recorded by pyPhotometry")
    sampling_rate = ppd_data['sampling_rate']
    logger.info(f"pyPhotometry sampling rate: {sampling_rate} Hz")

    # Convert start time
    photometry_start = datetime.strptime(ppd_data['date_time'], "%Y-%m-%dT%H:%M:%S.%f")
    photometry_start = photometry_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    logger.info(f"pyPhotometry start time: {photometry_start}")

    # Map channels to wavelengths based on signal count
    if ppd_data.get('analog_3') is not None:
        logger.info("Detected 3-signal pyPhotometry: analog_1=470nm, analog_2=565nm, analog_3=405nm")
        wavelength_keys = ["470nm", "565nm", "405nm"]
        raw_signals = [ppd_data['analog_1'], ppd_data['analog_2'], ppd_data['analog_3']]
    else:
        logger.info("Detected 2-signal pyPhotometry: analog_1=470nm, analog_2=405nm")
        wavelength_keys = ["470nm", "405nm"]
        raw_signals = [ppd_data['analog_1'], ppd_data['analog_2']]

    # Handle cropping (if needed)
    cropped_signals = crop_signals(raw_signals, sampling_rate, phot_end_time_mins, logger)
    signals = dict(zip(wavelength_keys, cropped_signals))

    return PhotometrySignalBundle(
        signals=signals,
        sampling_rate=sampling_rate,
        port_visits=visits,
        photometry_start=photometry_start,
        source="pyphotometry",
    )


def crop_signals(signal_list, sampling_rate, phot_end_time_mins, logger):
    """Crop a list of signals to the desired end time. Returns cropped signals."""
    raw_signal_length_mins = len(signal_list[0]) / sampling_rate / 60

    if phot_end_time_mins > raw_signal_length_mins:
        logger.warning(
            f"Specified `phot_end_time_mins` ({phot_end_time_mins}) is longer than the raw signal length "
            f"({raw_signal_length_mins} mins). The photometry signal will not be cropped."
        )
        phot_end_time_mins = raw_signal_length_mins
    elif phot_end_time_mins == 0:
        logger.debug(
            "No `phot_end_time_mins` specified, so the photometry signal will not be cropped "
            "(this is the normal case)."
        )
        phot_end_time_mins = raw_signal_length_mins
    else:
        logger.info(
            f"Cropping photometry signal from raw length ({raw_signal_length_mins} mins) "
            f"to {phot_end_time_mins} mins."
        )

    samples_to_keep = int(phot_end_time_mins * 60 * sampling_rate)
    return tuple(sig[:samples_to_keep] for sig in signal_list)


############################ Signal processing functions ############################

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


def apply_rolling_mean(signal, sampling_rate, window_fraction=0.0333):
    """Smooth signal using a rolling mean. Returns 1D numpy array."""
    window = max(1, int(sampling_rate * window_fraction))
    smoothed = pd.Series(signal).rolling(window=window, min_periods=1).mean().to_numpy()
    return smoothed


def apply_lowpass_filter(signal, sampling_rate, cutoff_hz=10, order=2):
    """Low-pass Butterworth filter."""
    b, a = butter(order, cutoff_hz, btype='low', fs=sampling_rate)
    return filtfilt(b, a, signal)


def apply_highpass_filter(signal, sampling_rate, cutoff_hz=0.001, order=2):
    """High-pass Butterworth filter for baseline removal."""
    b, a = butter(order, cutoff_hz, btype='high', fs=sampling_rate)
    return filtfilt(b, a, signal, padtype='even')


def apply_airpls_baseline(signal, logger, baseline_signal=None, lambda_=1e8, max_iterations=50, mode="subtract"):
    """Compute airPLS baseline and remove it from the signal.

    If baseline_signal is provided, the baseline is estimated from baseline_signal (not signal).
    We do this because historically, we have estimated the airPLS baseline from the raw data signal,
    then subtract it from the already-smoothed signal.

    mode:
        "subtract": signal - baseline  (removes drift, keeps signal in raw units)
        "dff": (signal - baseline) / baseline  (true dF/F, normalizes for signal magnitude)

    Returns (corrected, baseline).
    """
    data_for_baseline = baseline_signal if baseline_signal is not None else signal
    baseline = airPLS(data=data_for_baseline, logger=logger, lambda_=lambda_, max_iterations=max_iterations)
    if mode == "dff":
        corrected = (signal - baseline) / np.maximum(baseline, 1e-6)
    else:
        corrected = signal - baseline
    return corrected, baseline


def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    """Double exponential with constant offset, used for photobleaching baseline fitting."""
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def apply_double_exponential_baseline(signal, sampling_rate, mode="subtract", tau_slow=3600, tau_multiplier=0.1):
    """Fit a double exponential to the signal and use it as a photobleaching baseline.
    From Thomas Akam (https://github.com/ThomasAkam/photometry_preprocessing/):
    In practice we find that a double exponential fit is preferable to a single exponential fit because there are 
    typically multiple sources of fluorescence that contribute to the bleaching (e.g. autofluorescence from fiber, 
    autofluorescence from brain tissue, and flurophore fluorescence), which may bleach at different rates, 
    so a single exponential fit can be overly restrictive.
    
    Parameters:
        signal:          1D signal array (smoothed)
        sampling_rate:   Hz, used to construct the time vector
        mode:            "subtract" or "dff" — same semantics as apply_airpls_baseline
        tau_slow:        Initial guess for slow time constant in seconds (default: 3600)
        tau_multiplier:  Initial guess for fast/slow tau ratio (default: 0.1)

    Returns (corrected, baseline).
    """
    t = np.arange(len(signal)) / sampling_rate
    max_sig = np.max(signal)
    p0 = [max_sig / 2, max_sig / 4, max_sig / 4, tau_slow, tau_multiplier]
    bounds = (
        [0,       0,       0,       600,        0],
        [max_sig, max_sig, max_sig, 36000,      1],
    )
    params, _ = curve_fit(double_exponential, t, signal, p0=p0, bounds=bounds, maxfev=1000)
    baseline = double_exponential(t, *params)

    if mode == "dff":
        corrected = (signal - baseline) / np.maximum(baseline, 1e-6)
    else:
        corrected = signal - baseline
    return corrected, baseline


def apply_zscore_median_std(signal):
    """Z-score using median and std: (signal - median) / std"""
    return (signal - np.median(signal)) / np.std(signal)


def apply_zscore_mean_std(signal):
    """Z-score using mean and std: (signal - mean) / std"""
    return (signal - np.mean(signal)) / np.std(signal)


def apply_isosbestic_correction(signal, reference, alpha=0.0001):
    """Isosbestic correction via Lasso regression.

    Fits the reference signal to the signal using Lasso regression,
    then subtracts the fitted reference to produce dF/F.

    Returns (corrected_signal, fitted_reference).
    """
    signal_2d = signal.reshape(-1, 1)
    reference_2d = reference.reshape(-1, 1)
    lin = Lasso(alpha=alpha, precompute=True, max_iter=1000, positive=True, random_state=9999, selection="random")
    lin.fit(reference_2d, signal_2d)
    fitted_reference = lin.predict(reference_2d).ravel()
    corrected = signal - fitted_reference
    return corrected, fitted_reference


def apply_ratiometric_correction(signal, reference):
    """Ratiometric correction: signal / reference."""
    return signal / reference


############################ Processing pipeline for a given signal ############################

# Dispatch tables mapping method names to functions
SMOOTHING_METHODS = {
    "rolling_mean": apply_rolling_mean,
    "lowpass": apply_lowpass_filter,
    "none": None,
}

BASELINE_METHODS = {
    "airpls": apply_airpls_baseline,
    "double_exp": apply_double_exponential_baseline,
    "highpass": apply_highpass_filter,
    "none": None,
}

NORMALIZATION_METHODS = {
    "median_zscore": apply_zscore_median_std,
    "mean_zscore": apply_zscore_mean_std,
    "none": None,
}

CORRECTION_METHODS = {
    "isosbestic_lasso": apply_isosbestic_correction,
    "ratiometric": apply_ratiometric_correction,
    "none": None,
}


def load_processing_config(preset_name, overrides=None):
    """Load a processing preset from processing_presets.yaml and apply any overrides.

    Returns a fully resolved config dict with method names and parameters for each step.
    """
    with open(PRESETS_PATH, "r") as f:
        presets_yaml = yaml.safe_load(f)

    method_defaults = presets_yaml.get("method_defaults", {})
    presets = presets_yaml.get("presets", {})

    if preset_name not in presets:
        raise ValueError(
            f"Processing preset '{preset_name}' not found. "
            f"Available presets: {list(presets.keys())}"
        )

    preset = presets[preset_name]
    preset_param_overrides = preset.get("param_overrides", {})

    # Build resolved config: for each step, get the method name and its default params
    config = {}
    for step in ["smoothing", "baseline", "normalization", "correction"]:
        method_name = preset.get(step, "none")
        # Get default params for this method
        step_defaults = method_defaults.get(step, {})
        params = dict(step_defaults.get(method_name, {}))
        # Apply preset-level param overrides (e.g. a different lambda for a specific preset)
        if step in preset_param_overrides:
            params.update(preset_param_overrides[step])
        config[step] = {"method": method_name, "params": params}

    config["description"] = preset.get("description", "")

    # Apply session-level overrides (from processing_overrides in metadata)
    if overrides:
        for step, override_params in overrides.items():
            if step in config:
                config[step]["params"].update(override_params)

    return config


def resolve_indicator_configs(metadata, bundle, logger):
    """Determine which indicators to process and their preset configs.

    Returns a list of dicts, each with:
        - indicator_name: str
        - signal_wavelength: str (e.g. "470nm")
        - reference_wavelength: str or None
        - config: resolved processing config dict
    """
    with open(MAPPINGS_PATH, "r") as f:
        mappings = yaml.safe_load(f)

    phot_meta = metadata["photometry"]
    user_presets = phot_meta.get("processing_presets", {})
    overrides = phot_meta.get("processing_overrides")

    # Determine which indicators are present from virus_injections
    indicator_configs = []
    seen_indicators = set()

    if "virus_injections" in phot_meta:
        for virus_injection in phot_meta["virus_injections"]:
            virus_name = virus_injection["virus_name"]
            # Skip duplicates (bilateral injections of the same indicator)
            if virus_name in seen_indicators:
                continue
            seen_indicators.add(virus_name)

            if virus_name not in mappings:
                logger.warning(f"Indicator '{virus_name}' not found in photometry_mappings.yaml, skipping processing")
                continue

            mapping = mappings[virus_name]
            signal_wl = f"{mapping['signal_wavelength_nm']}nm"
            ref_wl_nm = mapping.get("reference_wavelength_nm")
            reference_wl = f"{ref_wl_nm}nm" if ref_wl_nm is not None else None

            # Check that the required wavelengths for this indicator are in the signals bundle
            if signal_wl not in bundle.signals:
                logger.warning(
                    f"Indicator '{virus_name}' requires {signal_wl} but it's not in the loaded signals "
                    f"({list(bundle.signals.keys())}). Skipping."
                )
                continue
            if reference_wl and reference_wl not in bundle.signals:
                logger.warning(
                    f"Indicator '{virus_name}' requires reference {reference_wl} but it's not in the loaded signals. "
                    "Skipping."
                )
                continue

            # Determine preset: user-specified > default from mappings
            preset_name = user_presets.get(virus_name, mapping.get("default_preset"))
            if preset_name is None:
                logger.error(f"No processing preset found for indicator '{virus_name}'")
                raise ValueError(f"No processing preset found for indicator '{virus_name}'")

            logger.info(f"Using preset '{preset_name}' for indicator '{virus_name}'")
            config = load_processing_config(preset_name, overrides)

            indicator_configs.append({
                "indicator_name": virus_name,
                "signal_wavelength": signal_wl,
                "reference_wavelength": reference_wl,
                "config": config,
            })

    if not indicator_configs:
        raise ValueError(
            "No indicator configs could be resolved. "
            "Please specify 'virus_injections' in the photometry metadata."
        )

    return indicator_configs


def process_single_signal(signal, sampling_rate, config, logger):
    """Apply smoothing -> baseline correction -> normalization to a single signal.

    Returns dict with 'smoothed', 'baseline_subtracted', 'baseline' (if applicable), and 'normalized'.
    """
    result = {}
    raw = signal.ravel()

    # Smoothing
    smooth_method = config["smoothing"]["method"]
    smooth_fn = SMOOTHING_METHODS.get(smooth_method)
    if smooth_fn:
        logger.info(f"Applying smoothing: {smooth_method} with params {config['smoothing']['params']}")
        smoothed = smooth_fn(raw, sampling_rate=sampling_rate, **config["smoothing"]["params"])
    else:
        smoothed = raw
    result["smoothed"] = smoothed

    # Baseline correction
    baseline_method = config["baseline"]["method"]
    baseline_fn = BASELINE_METHODS.get(baseline_method)
    if baseline_fn:
        logger.info(f"Applying baseline correction: {baseline_method} with params {config['baseline']['params']}")
        if baseline_method == "airpls":
            # airPLS computes the baseline from the RAW signal (not the smoothed signal),
            # then subtracts it from the smoothed signal
            params = config["baseline"]["params"]
            baseline_subtracted, baseline = apply_airpls_baseline(
                smoothed,
                logger=logger,
                baseline_signal=raw,
                lambda_=params.get("lambda", 1e8),
                max_iterations=params.get("max_iterations", 50),
                mode=params.get("mode", "subtract"),
            )
            result["baseline"] = baseline
        elif baseline_method == "double_exp":
            params = config["baseline"]["params"]
            baseline_subtracted, baseline = apply_double_exponential_baseline(
                smoothed,
                sampling_rate=sampling_rate,
                mode=params.get("mode", "subtract"),
                tau_slow=params.get("tau_slow", 3600),
                tau_multiplier=params.get("tau_multiplier", 0.1),
            )
            result["baseline"] = baseline
        else:
            # highpass filter returns the filtered signal directly (baseline is implicit)
            baseline_subtracted = baseline_fn(
                smoothed, sampling_rate=sampling_rate, **config["baseline"]["params"]
            )
    else:
        baseline_subtracted = smoothed
    result["baseline_subtracted"] = baseline_subtracted

    # Normalization
    norm_method = config["normalization"]["method"]
    norm_fn = NORMALIZATION_METHODS.get(norm_method)
    if norm_fn:
        logger.info(f"Applying normalization: {norm_method}")
        normalized = norm_fn(baseline_subtracted)
    else:
        normalized = baseline_subtracted
    result["normalized"] = normalized

    return result


def run_processing_pipeline(bundle, indicator_configs, logger, fig_dir=None):
    """Execute the full processing pipeline on loaded signals.

    For each indicator, the processing depends on the correction method:

    Isosbestic (e.g. dLight):
        1. Process signal and reference individually: smoothing -> baseline -> zscore
        2. Fit reference to signal via Lasso, subtract to get dF/F

    Ratiometric (e.g. gACh4h):
        1. Compute ratio from RAW signals (always positive, so ratio is well-behaved)
        2. Process ratio, signal, and reference individually: smoothing -> baseline -> zscore

    None (e.g. rDA3m):
        1. Process signal individually: smoothing -> baseline -> zscore

    Returns a dict of results keyed by indicator name.
    """
    all_results = {}

    for ind_config in indicator_configs:
        indicator_name = ind_config["indicator_name"]
        signal_wl = ind_config["signal_wavelength"]
        ref_wl = ind_config["reference_wavelength"]
        config = ind_config["config"]

        logger.info(f"Processing indicator '{indicator_name}' (signal={signal_wl}, reference={ref_wl})")
        logger.info(f"Processing config: {config['description']}")

        raw_signal = bundle.signals[signal_wl]
        raw_reference = bundle.signals[ref_wl] if ref_wl else None

        # Build plot label from indicator + wavelength (e.g. "dLight 470nm")
        def label(wl):
            return f"{indicator_name} {wl}"
        # Get color to plot this signal
        sig_color = INDICATOR_PLOT_COLORS.get(indicator_name, DEFAULT_SIGNAL_PLOT_COLOR)

        correction_method = config["correction"]["method"]
        correction_fn = CORRECTION_METHODS.get(correction_method)
        corrected = None
        fitted_reference = None
        raw_ratio = None
        ratio_result = None

        # Process signal
        logger.info(f"Processing {signal_wl} signal for {indicator_name}...")
        sig_result = process_single_signal(raw_signal, bundle.sampling_rate, config, logger)

        # Process reference (if applicable)
        ref_result = None
        if ref_wl:
            logger.info(f"Processing {ref_wl} reference for {indicator_name}...")
            ref_result = process_single_signal(raw_reference, bundle.sampling_rate, config, logger)

        # Apply isosbestic or ratiometric correction if applicable
        if correction_method == "isosbestic_lasso" and correction_fn and ref_result:
            logger.info("Applying isosbestic correction via Lasso regression")
            corrected, fitted_reference = correction_fn(
                sig_result["normalized"], ref_result["normalized"],
                **config["correction"]["params"],
            )

        elif correction_method == "ratiometric" and raw_reference is not None:
            # Ratiometric: compute ratio from RAW signals, then process the ratio
            # Raw signals are always positive so the ratio is numerically stable
            logger.info(f"Computing ratiometric correction: raw {signal_wl} / raw {ref_wl}")
            raw_ratio = apply_ratiometric_correction(raw_signal, raw_reference)

            logger.info(f"Processing {signal_wl}/{ref_wl} ratio for {indicator_name}...")
            ratio_result = process_single_signal(raw_ratio, bundle.sampling_rate, config, logger)
            corrected = ratio_result["normalized"]

        all_results[indicator_name] = {
            "signal_wavelength": signal_wl,
            "reference_wavelength": ref_wl,
            "raw_signal": raw_signal,
            "raw_reference": raw_reference,
            "raw_ratio": raw_ratio if correction_method == "ratiometric" else None,
            "processed_signal": sig_result["normalized"],
            "processed_reference": ref_result["normalized"] if ref_result else None,
            "corrected": corrected,
            "fitted_reference": fitted_reference,
            "correction_method": correction_method,
            "config": config,
        }

        # Plotting!
        signal_units = "a.u." if bundle.source != "pyphotometry" else "V"
        # If we have a reference wavelength (e.g. dLight, gACh4h)
        if ref_wl:
            # Plot raw signal and reference on same plot (2 subplots)
            plot_photometry_signals(
                visits=bundle.port_visits, sampling_rate=bundle.sampling_rate,
                signals=[raw_signal, raw_reference],
                signal_labels=[f"Raw {label(signal_wl)}", f"Raw {label(ref_wl)}"],
                signal_colors=[sig_color, REFERENCE_PLOT_COLOR],
                title=f"Raw signals for {indicator_name}",
                signal_units=signal_units,
                fig_dir=fig_dir,
            )
            # Plot correlation between signal and reference
            plot_signal_correlation(
                sig1=raw_signal, sig2=raw_reference,
                label1=label(signal_wl), label2=label(ref_wl), fig_dir=fig_dir,
            )
        # If no reference wavelength (e.g. rDA3m)
        else:
            # Just plot raw signal
            plot_photometry_signals(
                visits=bundle.port_visits, sampling_rate=bundle.sampling_rate,
                signals=[raw_signal],
                signal_labels=[f"Raw {label(signal_wl)}"],
                signal_colors=[sig_color],
                title=f"Raw signals for {indicator_name}",
                signal_units=signal_units,
                fig_dir=fig_dir,
            )
        # Plot processing steps for this signal (raw, smoothing, baseline, normalization)
        plot_processing_steps(
            raw_signal, sig_result, bundle.port_visits, bundle.sampling_rate,
            label(signal_wl), fig_dir, signal_color=sig_color, config=config,
        )
        # Plot processing steps for the reference wavelength (raw, smoothing, baseline, normalization) if it exists
        if ref_result:
            plot_processing_steps(
                raw_reference, ref_result, bundle.port_visits, bundle.sampling_rate,
                label(ref_wl), fig_dir, signal_color=REFERENCE_PLOT_COLOR, config=config,
            )
        # If we did isosbestic correction, plot the steps (signal, reference, predicted signal from ref, dF/F)
        if correction_method == "isosbestic_lasso" and corrected is not None:
            plot_photometry_signals(
                visits=bundle.port_visits, sampling_rate=bundle.sampling_rate,
                signals=[sig_result["normalized"], ref_result["normalized"], fitted_reference, corrected],
                signal_labels=[
                    f"Normalized {label(signal_wl)}", f"Normalized {label(ref_wl)}",
                    f"Predicted {label(signal_wl)} from {label(ref_wl)}", "dF/F (corrected)",
                ],
                signal_colors=[sig_color, REFERENCE_PLOT_COLOR, "gray", sig_color],
                title=f"{indicator_name} isosbestic correction",
                signal_units="normalized",
                fig_dir=fig_dir,
            )
        # If we did ratiometric correction
        elif correction_method == "ratiometric" and ratio_result is not None:
            # Plot processing steps for the ratio (raw, smoothing, baseline, normalization)
            plot_processing_steps(
                raw_ratio, ratio_result, bundle.port_visits, bundle.sampling_rate,
                label(f"{signal_wl}/{ref_wl} ratio"), fig_dir, signal_color=sig_color, config=config,
            )
            # Plot the signal (numerator), reference (denominator), and ratio
            plot_photometry_signals(
                visits=bundle.port_visits, sampling_rate=bundle.sampling_rate,
                signals=[sig_result["normalized"], ref_result["normalized"], corrected],
                signal_labels=[
                    f"Normalized {label(signal_wl)}",
                    f"Normalized {label(ref_wl)}",
                    f"Ratio {label(signal_wl)}/{label(ref_wl)} (corrected)",
                ],
                signal_colors=[sig_color, REFERENCE_PLOT_COLOR, sig_color],
                title=f"{indicator_name} ratiometric correction",
                signal_units="normalized",
                fig_dir=fig_dir,
            )

    return all_results


def plot_processing_steps(raw, result, visits, sampling_rate, signal_label, fig_dir, signal_color=None, config=None):
    """Plot the processing steps for a single signal."""
    signals_to_plot = [raw, result["smoothed"], result["baseline_subtracted"], result["normalized"]]
    labels = [
        f"Raw {signal_label}",
        f"Smoothed {signal_label}",
        f"Baseline-subtracted {signal_label}",
        f"Normalized {signal_label}",
    ]
    units = ["a.u.", "a.u.", "a.u.", "normalized"]
    overlay = None
    if "baseline" in result:
        overlay = [(result["baseline"], 1, "black", "baseline")]

    colors = [signal_color] * 4 if signal_color else None

    if config:
        smooth = config["smoothing"]["method"]
        baseline = config["baseline"]["method"]
        norm = config["normalization"]["method"]
        baseline_mode = config["baseline"]["params"].get("mode", "subtract")
        step2_label = "dF/F" if baseline_mode == "dff" else "Baseline-subtracted"
        labels[1] = f"Smoothed {signal_label} ({smooth})"
        labels[2] = f"{step2_label} {signal_label} ({baseline})"
        labels[3] = f"Normalized {signal_label} ({norm})"

    return plot_photometry_signals(
        visits=visits, sampling_rate=sampling_rate,
        signals=signals_to_plot, signal_labels=labels,
        title=f"{signal_label} processing steps",
        signal_units=units,
        signal_colors=colors,
        overlay_signals=overlay,
        fig_dir=fig_dir,
    )


############################ Add signals to nwb ############################

def find_led_table_region(nwbfile, led_color, logger):
    """Find the FiberPhotometryTableRegion for a given LED color.

    Args:
        led_color: One of "blue", "purple", "green"
    """
    fiber_photometry_table = nwbfile.get_lab_meta_data("fiber_photometry").fiber_photometry_table

    for row_index, excitation_source_obj in enumerate(fiber_photometry_table.excitation_source.data):
        if f"{led_color} led" in excitation_source_obj.name.lower():
            return fiber_photometry_table.create_fiber_photometry_table_region(
                region=[row_index], description=f"{led_color.capitalize()} LED"
            )

    logger.error(f"Could not find a {led_color} LED in fiber photometry table.")
    raise ValueError(f"{led_color.capitalize()} LED not found in fiber photometry table.")


def write_photometry_to_nwb(nwbfile, bundle, processing_results, logger):
    """Write raw and processed photometry signals to the NWB file.

    Returns a dict with keys: sampling_rate, port_visits, photometry_start, signals_to_plot
    """
    print("Adding photometry signals to NWB...")
    logger.info("Adding photometry signals to NWB...")

    # Write raw signals (one per wavelength in the bundle, written once)
    written_raw_wavelengths = set()
    for ind_result in processing_results.values():
        for wl_key, raw_data in [
            (ind_result["signal_wavelength"], ind_result["raw_signal"]),
            (ind_result["reference_wavelength"], ind_result["raw_reference"]),
        ]:
            if wl_key is None or wl_key in written_raw_wavelengths or raw_data is None:
                continue

            led_color = WAVELENGTH_TO_LED_COLOR[wl_key]
            led_region = find_led_table_region(nwbfile, led_color, logger)
            wl_num = wl_key.replace("nm", "")
            series_name = f"raw_{wl_num}"

            raw_np = raw_data.to_numpy() if hasattr(raw_data, 'to_numpy') else np.asarray(raw_data).ravel()
            series = FiberPhotometryResponseSeries(
                name=series_name,
                description=f"Raw {wl_key}",
                data=raw_np,
                unit="V" if bundle.source == "pyphotometry" else "a.u.",
                rate=float(bundle.sampling_rate),
                fiber_photometry_table_region=led_region,
            )
            nwbfile.add_acquisition(series)
            written_raw_wavelengths.add(wl_key)
            logger.info(f"Added raw signal '{series_name}' to NWB")

    # Write processed and corrected signals for each indicator
    signals_to_plot = []

    for indicator_name, ind_result in processing_results.items():
        signal_wl = ind_result["signal_wavelength"]
        ref_wl = ind_result["reference_wavelength"]
        signal_wl_num = signal_wl.replace("nm", "")
        signal_led_color = WAVELENGTH_TO_LED_COLOR[signal_wl]
        signal_led_region = find_led_table_region(nwbfile, signal_led_color, logger)

        # Write processed signal
        processed_name = f"processed_{signal_wl_num}"
        series = FiberPhotometryResponseSeries(
            name=processed_name,
            description=f"Processed {signal_wl} for {indicator_name}",
            data=ind_result["processed_signal"],
            unit="z-score",
            rate=float(bundle.sampling_rate),
            fiber_photometry_table_region=signal_led_region,
        )
        nwbfile.add_acquisition(series)
        logger.info(f"Added processed signal '{processed_name}' to NWB")

        # Write processed reference (if applicable)
        if ref_wl and ind_result["processed_reference"] is not None:
            ref_wl_num = ref_wl.replace("nm", "")
            ref_led_color = WAVELENGTH_TO_LED_COLOR[ref_wl]
            ref_led_region = find_led_table_region(nwbfile, ref_led_color, logger)

            processed_ref_name = f"processed_{ref_wl_num}"
            # Only write if not already written by another indicator
            existing_names = [acq.name for acq in nwbfile.acquisition.values()]
            if processed_ref_name not in existing_names:
                series = FiberPhotometryResponseSeries(
                    name=processed_ref_name,
                    description=f"Processed {ref_wl} reference for {indicator_name}",
                    data=ind_result["processed_reference"],
                    unit="z-score",
                    rate=float(bundle.sampling_rate),
                    fiber_photometry_table_region=ref_led_region,
                )
                nwbfile.add_acquisition(series)
                logger.info(f"Added processed reference '{processed_ref_name}' to NWB")

        # Write corrected signal (if applicable)
        correction_method = ind_result["correction_method"]
        if ind_result["corrected"] is not None:
            if correction_method == "isosbestic_lasso":
                corrected_name = f"corrected_{signal_wl_num}_dFF"
                series = FiberPhotometryResponseSeries(
                    name=corrected_name,
                    description=f"Isosbestic-corrected dF/F for {indicator_name} ({signal_wl})",
                    data=ind_result["corrected"],
                    unit="dF/F",
                    rate=float(bundle.sampling_rate),
                    fiber_photometry_table_region=signal_led_region,
                )
                nwbfile.add_acquisition(series)
                logger.info(f"Added corrected signal '{corrected_name}' to NWB")
                signals_to_plot.append(corrected_name)

                # Write fitted reference
                if ind_result["fitted_reference"] is not None:
                    ref_wl_num = ref_wl.replace("nm", "")
                    ref_led_color = WAVELENGTH_TO_LED_COLOR[ref_wl]
                    ref_led_region = find_led_table_region(nwbfile, ref_led_color, logger)
                    fitted_name = f"fitted_{ref_wl_num}"
                    series = FiberPhotometryResponseSeries(
                        name=fitted_name,
                        description=f"Lasso-fitted {ref_wl} reference used for isosbestic correction",
                        data=ind_result["fitted_reference"],
                        unit="z-score",
                        rate=float(bundle.sampling_rate),
                        fiber_photometry_table_region=ref_led_region,
                    )
                    nwbfile.add_acquisition(series)
                    logger.info(f"Added fitted reference '{fitted_name}' to NWB")

            elif correction_method == "ratiometric":
                ref_wl_num = ref_wl.replace("nm", "")

                # Write raw ratio
                raw_ratio_name = f"raw_{signal_wl_num}_{ref_wl_num}_ratio"
                series = FiberPhotometryResponseSeries(
                    name=raw_ratio_name,
                    description=f"Raw ratiometric signal ({signal_wl}/{ref_wl}) for {indicator_name}",
                    data=ind_result["raw_ratio"],
                    unit="ratio",
                    rate=float(bundle.sampling_rate),
                    fiber_photometry_table_region=signal_led_region,
                )
                nwbfile.add_acquisition(series)
                logger.info(f"Added raw ratio '{raw_ratio_name}' to NWB")

                # Write normalized ratio
                corrected_name = f"corrected_{signal_wl_num}_{ref_wl_num}_ratio"
                series = FiberPhotometryResponseSeries(
                    name=corrected_name,
                    description=f"Normalized ratiometric correction ({signal_wl}/{ref_wl}) for {indicator_name}",
                    data=ind_result["corrected"],
                    unit="z-score",
                    rate=float(bundle.sampling_rate),
                    fiber_photometry_table_region=signal_led_region,
                )
                nwbfile.add_acquisition(series)
                logger.info(f"Added corrected signal '{corrected_name}' to NWB")
                signals_to_plot.append(corrected_name)
        else:
            # No correction - the processed signal is the final output
            signals_to_plot.append(processed_name)

    # Convert port visits to seconds
    visits_in_seconds = [visit_time / bundle.sampling_rate for visit_time in bundle.port_visits]

    return {
        'sampling_rate': bundle.sampling_rate,
        'port_visits': visits_in_seconds,
        'photometry_start': bundle.photometry_start,
        'signals_to_plot': signals_to_plot,
    }


############################ Add metadata to nwb ############################


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
            mapped_excitation_sources = mappings[indicator_name_for_mapping]["excitation_sources"]

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


############################ Entry point ############################


def add_photometry(nwbfile: NWBFile, metadata: dict, logger, fig_dir=None):
    """
    Add photometry data to the NWB and return port visits
    in downsampled photometry time to use for alignment.

    Supports three data sources:
    - LabVIEW raw (.phot + .box files)
    - LabVIEW preprocessed (signals.mat)
    - pyPhotometry (.ppd file)

    Processing is driven by configurable presets defined in
    resources/photometry/processing_presets.yaml. The preset for each
    indicator is determined from photometry_mappings.yaml (default_preset)
    or can be overridden via processing_presets in the session metadata.
    """

    if "photometry" not in metadata:
        print("No photometry metadata found for this session. Skipping photometry conversion.")
        logger.info("No photometry metadata found for this session. Skipping photometry conversion.")
        return {}

    # 1. Add photometry metadata to NWB
    print("Adding photometry metadata to NWB...")
    logger.info("Adding photometry metadata to NWB...")
    add_photometry_metadata(nwbfile, metadata, logger)

    # 2. Load photometry signals into PhotometrySignalBundle
    phot_meta = metadata["photometry"]
    # If we have raw LabVIEW data (.phot and .box files)
    if "phot_file_path" in phot_meta and "box_file_path" in phot_meta:
        logger.info("Using LabVIEW for photometry!")
        print("Processing raw .phot and .box files from LabVIEW...")
        bundle = load_labview_raw(
            phot_meta["phot_file_path"], phot_meta["box_file_path"],
            logger, phot_meta.get("phot_end_time_mins", 0),
        )
    # If we have already processed the LabVIEW .phot and .box files into signals.mat (legacy, will be deprecated)
    elif "signals_mat_file_path" in phot_meta:
        logger.info("Using LabVIEW for photometry!")
        print("Processing signals.mat file of photometry signals from LabVIEW...")
        bundle = load_labview_mat(
            phot_meta["signals_mat_file_path"],
            logger, phot_meta.get("phot_end_time_mins", 0),
        )
    # If we have a ppd file from pyPhotometry
    elif "ppd_file_path" in phot_meta:
        logger.info("Using pyPhotometry for photometry!")
        print("Processing ppd file from pyPhotometry...")
        bundle = load_pyphotometry(
            phot_meta["ppd_file_path"], logger, phot_meta.get("phot_end_time_mins", 0),
        )
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

    # 3. Resolve processing config for each indicator
    indicator_configs = resolve_indicator_configs(metadata, bundle, logger)

    # 4. Run processing pipeline
    processing_results = run_processing_pipeline(bundle, indicator_configs, logger, fig_dir)

    # 5. Write to NWB
    photometry_data_dict = write_photometry_to_nwb(nwbfile, bundle, processing_results, logger)

    # 6. Set ground truth time source
    metadata["ground_truth_time_source"] = "photometry"
    metadata["ground_truth_visit_times"] = photometry_data_dict.get("port_visits")

    return photometry_data_dict
