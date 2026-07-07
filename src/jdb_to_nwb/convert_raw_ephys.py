# NOTE: We could have used NeuroConv to convert the ephys data but we want to use the Frank Lab's 
# ndx-franklab-novela extension to store the Probe information for maximal integration with Spyglass, 
# so we are doing the conversion manually using PyNWB.

import os
import re
import glob
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import xml.etree.ElementTree as ET
from importlib.resources import files
from hdmf.backends.hdf5 import H5DataIO

from neuroconv.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
    SpikeInterfaceRecordingDataChunkIterator,
)
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor

from .utils import get_logger_directory, log_and_print
from .timestamps_alignment import align_via_interpolation
from .plotting.plot_ephys import plot_channel_map, plot_channel_impedances, plot_neuropixels
from ndx_franklab_novela import AssociatedFiles, Probe, NwbElectrodeGroup, Shank, ShanksElectrode

VOLTS_PER_MICROVOLT = 1e-6

MIN_IMPEDANCE_OHMS = 1e5
MAX_IMPEDANCE_OHMS = 3e6

# Neuropixels port visits are recorded on the separate OneBox-ADC stream on ADC2 (in volts)
# (ADC0 carries a 1 Hz sync clock, ADC1 is used for future optotagging)
# Both the channel and the threshold are overridable via metadata["ephys"]
NPX_DEFAULT_PORT_VISITS_ADC_CHANNEL = 2
NPX_PORT_VISITS_THRESHOLD_VOLTS = 4.0

# Get the location of the resources directory when the package is installed from pypi
__location_of_this_file = Path(files(__name__))
RESOURCES_DIR = __location_of_this_file / "resources" / "electrophysiology"

# If the resources directory does not exist, we are probably running the code from the source directory
if not RESOURCES_DIR.exists():
    RESOURCES_DIR = __location_of_this_file.parent.parent / "resources" / "electrophysiology"

BERKE_256CH_PROBE_CHANNEL_MAP_PATH = RESOURCES_DIR / "256ch_silicon_probe_channel_map.csv"
BERKE_252CH_PROBE_CHANNEL_MAP_PATH = RESOURCES_DIR  / "252ch_silicon_probe_channel_map.csv"
ELECTRODE_COORDS_PATH_256CH_3MM_PROBE = RESOURCES_DIR  / "256ch_probe_3mm_length_66um_pitch_coords.csv"
ELECTRODE_COORDS_PATH_256CH_6MM_PROBE = RESOURCES_DIR / "256ch_probe_6mm_length_80um_pitch_coords.csv"
ELECTRODE_COORDS_PATH_252CH_4MM_PROBE = RESOURCES_DIR / "252ch_probe_4mm_length_80um_pitch_coords.csv"
ELECTRODE_COORDS_PATH_252CH_10MM_PROBE = RESOURCES_DIR / "252ch_probe_10mm_length_100um_pitch_coords.csv"
ELECTRODE_COORDS_PATH_NPX_MULTISHANK = RESOURCES_DIR / "neuropixels_2.0_multishank_electrode_coords.csv"
DEVICES_PATH = RESOURCES_DIR / "ephys_devices.yaml"

# Names of custom Berke Lab probes
BERKE_LAB_PROBES = {
    "256-ch Silicon Probe, 3mm length, 66um pitch": {
        "channel_map": BERKE_256CH_PROBE_CHANNEL_MAP_PATH,
        "electrode_coords": ELECTRODE_COORDS_PATH_256CH_3MM_PROBE
    },
    "256-ch Silicon Probe, 6mm length, 80um pitch": {
        "channel_map": BERKE_256CH_PROBE_CHANNEL_MAP_PATH,
        "electrode_coords": ELECTRODE_COORDS_PATH_256CH_6MM_PROBE
    },
    "252-ch Silicon Probe, 4mm length, 80um pitch": {
        "channel_map": BERKE_252CH_PROBE_CHANNEL_MAP_PATH,
        "electrode_coords": ELECTRODE_COORDS_PATH_252CH_4MM_PROBE
    },
    "252-ch Silicon Probe, 10mm length, 100um pitch": {
        "channel_map": BERKE_252CH_PROBE_CHANNEL_MAP_PATH,
        "electrode_coords": ELECTRODE_COORDS_PATH_252CH_10MM_PROBE
    },
}

class MicrovoltsSpikeInterfaceRecordingDataChunkIterator(SpikeInterfaceRecordingDataChunkIterator):
    def __init__(self, iterator: SpikeInterfaceRecordingDataChunkIterator, conversion_factor_uv):
        self.iterator = iterator
        self.conversion_factor_uv = conversion_factor_uv
        super().__init__(recording=iterator.recording)

    def _get_default_chunk_shape(self, chunk_mb: float = 10.0) -> tuple[int, int]:
        return self.iterator._get_default_chunk_shape(chunk_mb)

    def _get_data(self, selection: tuple[slice]):
        data = self.iterator._get_data(selection)
        return (data * self.conversion_factor_uv).astype("int16")

    def _get_dtype(self):
        return np.dtype("int16")

    def _get_maxshape(self):
        return self.iterator._get_maxshape()


def find_open_ephys_paths(open_ephys_folder_path, experiment_number=1) -> dict:
    """
    Given the Open Ephys folder path, find the relevant settings.xml file and all associated continuous.dat files.

    For custom Berke Lab probes, we expect a single probe folder with a single continuous.dat. For example:
    open_ephys_folder_path/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat

    For Neuropixels, we may have multiple "probe name" folders, each with a different continuous.dat file
    This will be the case for the ADC, and also if we implant 2 probes bilatrerally. For example:
    open_ephys_folder_path/Record Node 101/experiment2/recording1/continuous/OneBox-100.OneBox-ADC/continuous.dat
    open_ephys_folder_path/Record Node 101/experiment2/recording1/continuous/OneBox-100.ProbeA/continuous.dat

    For both cases, the settings.xml file lives at the same level as the experiment{number} folder.
    The suffix for settings.xml matches the number of the experiment if >=2, or no suffix for experiment1
    Example paths:
    open_ephys_folder_path/settings.xml (Berke Lab probe)
    open_ephys_folder_path/Record Node 101/settings_2.xml (Neuropixels)

    Parameters:
        open_ephys_folder_path: 
            Path to the Open Ephys output folder (the auto-generated one with the
            recording start time in the folder name)
        experiment_number (int):
            Optional. If multiple recordings in the same output folder (which get
            auto-named experiment1, experiment2, etc), which one to use. Defaults to 1

    Returns:
        dict:
            "settings_file": path to settings.xml file
            "recording_files": dict with probe_name: path to the continuous.dat for that probe
    """
    # Determine the suffix for settings.xml (empty for experiment1, "_2" for experiment2, etc.)
    suffix = "" if experiment_number == 1 else f"_{experiment_number}"

    # Look for the settings.xml file
    settings_file_pattern = os.path.join(open_ephys_folder_path, f"**/settings{suffix}.xml")
    settings_files = glob.glob(settings_file_pattern, recursive=True)

    # Make sure there is a single settings.xml file
    if not settings_files:
        raise FileNotFoundError(f"No settings.xml found for experiment {experiment_number} in {open_ephys_folder_path}")
    if len(settings_files) > 1:
        raise ValueError(
            f"Found {len(settings_files)} for experiment {experiment_number} in {open_ephys_folder_path},"
            f"expected 1: {settings_files}"
            )
    settings_xml_file = settings_files[0]

    # Look for the continuous.dat files
    experiment_folder = f"experiment{experiment_number}"
    continuous_base = os.path.join(open_ephys_folder_path, "**", experiment_folder, "recording*", "continuous")
    probe_folders = glob.glob(os.path.join(continuous_base, "*"), recursive=True)

    # Inside each "probe name" folder, there should be a single continuous.dat
    continuous_dat_files = {}
    for probe_folder in probe_folders:
        if not os.path.isdir(probe_folder):
            continue
        probe_name = os.path.basename(probe_folder)
        dat_files = glob.glob(os.path.join(probe_folder, "continuous.dat"))
        if len(dat_files) == 0:
            print(f"No continuous.dat found in {probe_folder}!")
            continue
        if len(dat_files) > 1:
            raise ValueError(f"Multiple continuous.dat files found in {probe_folder}: {dat_files}")
        continuous_dat_files[probe_name] = dat_files[0]

    return {"settings_file": settings_xml_file, "recording_files": continuous_dat_files}


def read_oebin_params(continuous_dat_file_path: Path, logger) -> dict:
    """
    Read per-stream recording parameters from the structure.oebin file.

    The structure.oebin is a JSON file at the recording level (the same folder that contains the
    'continuous' folder). It describes every continuous stream (probe streams and the ADC stream),
    including the number of channels, sampling rate, and the per-channel bit_volts conversion factor.
    This works the same for Berke Lab probes and Neuropixels.

    Note: structure.oebin does NOT contain the bandpass filtering info (LowCut/HighCut) -
    afaik that lives only in settings.xml, so filtering info still comes from read_open_ephys_settings_xml.

    Parameters:
        continuous_dat_file_path (Path):
            Path to a 'continuous.dat' file. The structure.oebin is found relative to it
            (.../recording{n}/continuous/<stream>/continuous.dat -> .../recording{n}/structure.oebin)
        logger (Logger):
            Logger to track conversion progress

    Returns:
        dict:
            Params for the stream matching this continuous.dat:
            "channel_count" (int), "sample_rate" (float), "bit_volts" (float, the uV conversion
            factor, shared across channels), "channel_names" (list[str]), and "oebin_path" (Path to
            the structure.oebin file itself, so we can save it as an AssociatedFile)
    """
    continuous_dat_file_path = Path(continuous_dat_file_path)
    # .../recording{n}/continuous/<stream folder>/continuous.dat -> stream folder name and oebin path
    stream_folder_name = continuous_dat_file_path.parent.name
    oebin_path = continuous_dat_file_path.parent.parent.parent / "structure.oebin"
    if not oebin_path.exists():
        raise FileNotFoundError(f"Could not find structure.oebin at expected location {oebin_path}")

    with open(oebin_path, "r") as f:
        oebin = json.load(f)

    # Log a summary of every continuous stream in the oebin 
    logger.debug(f"structure.oebin at {oebin_path} (Open Ephys GUI version {oebin.get('GUI version', '?')})")
    for s in oebin["continuous"]:
        logger.debug(f"  continuous stream: folder_name={s['folder_name']!r} "
                     f"num_channels={s['num_channels']} sample_rate={s['sample_rate']}")

    # Find the continuous stream whose folder_name matches this continuous.dat's stream folder
    # (oebin folder_name has a trailing slash, e.g. "Rhythm_FPGA-100.0/", so strip it to compare)
    for stream in oebin["continuous"]:
        if stream["folder_name"].rstrip("/") == stream_folder_name:
            break
    else:
        raise ValueError(
            f"Could not find a continuous stream matching '{stream_folder_name}' in {oebin_path}. "
            f"Available streams: {[s['folder_name'] for s in oebin['continuous']]}"
        )

    # bit_volts (the uV conversion factor) should be the same for every channel in the stream
    bit_volts_per_channel = {ch["bit_volts"] for ch in stream["channels"]}
    if len(bit_volts_per_channel) != 1:
        logger.warning(f"Stream '{stream_folder_name}' has multiple bit_volts values: {bit_volts_per_channel}")
    bit_volts = float(stream["channels"][0]["bit_volts"])

    params = {
        "channel_count": int(stream["num_channels"]),
        "sample_rate": float(stream["sample_rate"]),
        "bit_volts": bit_volts,
        "channel_names": [ch["channel_name"] for ch in stream["channels"]],
        "oebin_path": oebin_path,  # so we can save structure.oebin as an AssociatedFile
    }
    logger.info(f"Read oebin params for stream '{stream_folder_name}': "
                f"{params['channel_count']} channels at {params['sample_rate']} Hz, bit_volts={bit_volts}")
    # List every channel name found in this stream
    logger.debug(f"structure.oebin channel names for '{stream_folder_name}': {params['channel_names']}")
    return params


def validate_oebin_channel_names(channel_names: list[str], expected_channel_names: set[str], logger) -> None:
    """
    Sanity-check the channel names read from structure.oebin against what we expect for this probe.

    We source this from structure.oebin (which lists each channel's name) rather than settings.xml,
    since oebin is format-independent and we already read it. Logs warnings but does NOT raise on a
    mismatch: a few missing/unexpected channels usually just get excluded downstream by impedance
    anyway (e.g. IM-1875 is missing CH1-CH8, replaced by duplicated ADC1-ADC8).

    Parameters:
        channel_names (list[str]):
            Ordered channel names from structure.oebin (from read_oebin_params)
        expected_channel_names (set[str]):
            The channel names we expect for this probe (e.g. CH1-CH256 + ADC1-ADC8 for a Berke probe)
        logger (Logger):
            Logger to track conversion progress
    """
    found_channel_names = set(channel_names)

    # Check for missing or unexpected channel names
    missing_channel_names = expected_channel_names - found_channel_names
    unexpected_channel_names = found_channel_names - expected_channel_names
    if missing_channel_names or unexpected_channel_names:
        logger.warning("Channel names in structure.oebin do not match expectations!!!!")
        if missing_channel_names:
            logger.warning(f"Missing channel names: {sorted(missing_channel_names)}")
        if unexpected_channel_names:
            logger.warning(f"Unexpected channel names: {sorted(unexpected_channel_names)}")
    else:
        logger.info(f"All {len(expected_channel_names)} expected channel names found in structure.oebin")

    # Check for duplicate channel names
    duplicate_names = [name for name, count in Counter(channel_names).items() if count > 1]
    if duplicate_names:
        logger.warning("Duplicate channel names found in structure.oebin!!!!")
        logger.warning(f"Duplicate channel names: {sorted(duplicate_names)}")


def get_port_visits(continuous_dat_file_path: Path,
                    total_channels: int,
                    port_visits_channel_num: int,
                    sample_rate: float,
                    logger,
                    pulse_high_threshold: float = 10_000) -> tuple[list[float], int]:
    """
    Extract port visit times from an OpenEphys continuous.dat file.

    Works for both Berke Lab custom probes and Neuropixels. The difference is just which file/channel
    the port visits live on, plus the sample rate and threshold:
      - Berke Lab probes: port visits are on ADC1 (channel 256) of the probe's own continuous.dat
        (264 channels, 30000 Hz, raw int16 threshold ~10000).
      - Neuropixels (OneBox): port visits are on the SEPARATE OneBox-ADC continuous.dat (ADC2,
        12 channels, 30300.5 Hz). ADC data is stored as raw int16 like everything else, so pass a
        threshold in raw int16 units (volts / bit_volts).
        
    NOTE: We return a time (not a sample count) of bonsai start because the neuropixels port visit file 
    (e.g. the OneBox-ADC at 30300.5 Hz) has a different sample rate than the probe file (30000 Hz). 
    Sample rates are same for Berke Lab probes (ADC1 lives in the same contimnuous.dat file as neural data)

    Parameters:
        continuous_dat_file_path (Path):
            Path to the 'continuous.dat' OpenEphys binary recording that holds the port visit channel
        total_channels (int):
            The total number of channels in that continuous.dat file
        port_visits_channel_num (int):
            The number of the channel (zero-indexed) that port visits are recorded on
        sample_rate (float):
            The sample rate (Hz) of that continuous.dat file (from structure.oebin)
        logger (Logger):
            Logger to track conversion progress
        pulse_high_threshold (float):
            Raw int16 value above which the port visit channel is considered "high". Defaults to 10000

    Returns:
        port_visits (list[float]):
            List of port visit times in seconds (relative to the bonsai start pulse)
        bonsai_start_time (float):
            Time (seconds) of the start pulse marking bonsai start, relative to the start of this
            continuous.dat. The caller converts this to a sample count in the neural recording's
            sample rate to trim data before bonsai start
    """
    # Downsample to ~1000 Hz before searching for pulses. 1000ish is low enough to speed things up
    # (this takes ~2mins on my machine) but high enough that we definitely don't miss any port visit pulses.
    # For reference, a typical port visit keeps the channel high for ~10ms, so 10 samples at 1000 Hz.
    # Sample rates aren't always a clean multiple of 1000 (the OneBox ADC is 30300.5 Hz), so compute the
    # integer decimation factor and the actual downsampled rate it gives us (used for samples->seconds).
    downsampled_fs_target = 1000
    decimation = max(1, round(sample_rate / downsampled_fs_target))
    downsampled_fs = sample_rate / decimation

    # Memory-map the large .dat file to avoid loading everything into memory. Reshape to (samples, channels)
    data_for_all_channels = np.memmap(continuous_dat_file_path, dtype='int16', mode='c').reshape(-1, total_channels)

    logger.debug(f"Reading Open Ephys port visits from channel {port_visits_channel_num}")
    logger.debug(f"Downsampling data from {sample_rate} Hz to {downsampled_fs} Hz so it isn't huge")

    # Extract the channel that records port visits and downsample so the data isn't massive
    # Force a contiguous copy of just the one channel up front (much faster than slicing the memmap column)
    visits_channel_data = np.array(data_for_all_channels[:, port_visits_channel_num])  # copies into RAM contiguously
    visits_channel_data = visits_channel_data[::decimation]

    # Find indices where the visits channel is high (aka port visit times)
    logger.debug(f"Recording times the channel is high (>{pulse_high_threshold})")
    pulse_above_threshold = np.where(visits_channel_data > pulse_high_threshold)[0]

    # If no port visits were found, return an empty list of visits and bonsai start time 0
    if pulse_above_threshold.size == 0:
        return [], 0.0

    # Find pulse boundaries (breaks in the sequence of threshold crossings)
    breaks = np.where(np.diff(pulse_above_threshold) != 1)[0] + 1

    # Get the durations of each pulse (each contiguous block where the visits channel is high)
    pulse_starts = np.insert(pulse_above_threshold[breaks], 0, pulse_above_threshold[0])
    pulse_ends = np.append(pulse_above_threshold[breaks - 1], pulse_above_threshold[-1])
    pulse_durations = pulse_ends - pulse_starts + 1

    logger.debug(f"Found {len(pulse_starts)} visit pulses.")

    # Only keep pulses at least 5ms long just in case (typical port visit keeps the channel high for ~10ms)
    min_pulse_duration = int(downsampled_fs * 0.005)
    pulse_starts = pulse_starts[pulse_durations > min_pulse_duration]

    logger.debug(f"Only keeping pulses at least 5ms long ({min_pulse_duration} samples at {downsampled_fs} Hz).")
    logger.debug(f"Kept {len(pulse_starts)} pulses.")
    logger.debug(f"Pulse durations: {pulse_durations}")

    # Set first pulse (which marks bonsai start time) as time=0 and convert to seconds
    # The duration of the first pulse is longer than a normal pulse (~500ms instead of ~10ms)
    if pulse_durations[0] < 300:
        logger.warning("Expected the first pulse marking the start time to have duration >= 300! "
                       f"Got pulse duration {pulse_durations[0]} - this may indicate a problem with the start pulse!")

    # Bonsai start time (seconds) relative to the start of this continuous.dat
    bonsai_start_time = float(pulse_starts[0] / downsampled_fs)
    logger.info(f"Bonsai start time occurred {bonsai_start_time}s after ephys was started.")
    logger.info("This will be set to time=0 and samples before this will be removed.")

    # Convert port visit times to seconds relative to bonsai start pulse
    port_visits = pulse_starts[1:] - pulse_starts[0]
    port_visits = [float(visit / downsampled_fs) for visit in port_visits]

    return port_visits, bonsai_start_time


def get_raw_ephys_data(
    folder_path: Path,
    logger,
    exclude_channels: list[str] = [],
    samples_to_remove: int = 0
) -> tuple[SpikeInterfaceRecordingDataChunkIterator, float, np.ndarray, list[str]]:
    """
    Get the raw ephys data from the OpenEphys binary recording.

    Parameters:
        folder_path (Path):
            Path to the folder containing the OpenEphys binary recording. The folder
            should have the date in the name and contain a file called "settings.xml".
        logger (Logger):
            Logger to track conversion progress
        exclude_channels (list[str]):
            List of channel names to exclude, if any. 
            Channel names are 1-based and should start with 'CH', e.g. 'CH64'
        samples_to_remove (int):
            Number of samples to trim off of the start of the raw ephys data.
            Used to remove data before bonsai start time

    Returns:
        traces_as_iterator (SpikeInterfaceRecordingDataChunkIterator):
            To be used as the data argument in pynwb.ecephys.ElectricalSeries.
        channel_conversion_factor (float):
            The conversion factor from the raw data to uV.
        original_timestamps (np.ndarray):
            Array that could be used as the timestamps argument in pynwb.ecephys.ElectricalSeries
            or may need to be time aligned with the other data streams in the NWB file.
            Should start at time=0, which is the bonsai start time.
    """
    # Create a SpikeInterface recording extractor for the OpenEphys binary data
    # NOTE: We could write our own extractor to handle the relatively simple OpenEphys binary format
    # and surrounding files but it is nice to build on the well-tested code of others when possible.
    # However, we should remember that external code may not be well-maintained or may have bugs of their own.
    streams, _ = OpenEphysBinaryRecordingExtractor.get_streams(folder_path=folder_path)
    logger.debug(f"Found streams in the OpenEphys binary data: {streams}")

    streams_without_adc = [s for s in streams if not s.endswith("ADC")]
    assert len(streams_without_adc) == 1, \
        (f"More than one non-ADC stream found in the OpenEphys binary data: {streams_without_adc}")

    # NOTE: For bilateral Neuropixels we will have 2 streams without ADC (one for each probe)
    # We will deal with this later. Probably just do all of the following in a loop (once per probe)?

    # Ignore the "ADC" channels
    recording = OpenEphysBinaryRecordingExtractor(folder_path=folder_path, stream_name=streams_without_adc[0])
    # Exclude extra channels
    if exclude_channels:
        logger.debug(f"Excluding {len(exclude_channels)} channels: {exclude_channels}")
    channel_ids_to_convert = [
        ch for ch in recording.channel_ids if ch.startswith("CH") and ch not in exclude_channels
    ]
    recording_sliced = recording.select_channels(channel_ids=channel_ids_to_convert)

    # Confirm all channel names start with "CH"
    assert all([ch.startswith("CH") for ch in recording_sliced.channel_ids]), \
        (f"Some channels do not start with 'CH': {recording_sliced.channel_ids}")

    logger.debug(
        f"Found {len(recording_sliced.channel_ids)} Open Ephys channels to convert: "
        f"{recording_sliced.channel_ids}"
    )

    # Get the channel conversion factor
    channel_conversion_factors_uv = recording_sliced.get_channel_gains()
    # Warn if the channel conversion factors are not the same for all channels
    if not all(channel_conversion_factors_uv == channel_conversion_factors_uv[0]):
        log_and_print(
            logger,
            "The channel conversion factors are not the same for all channels. "
            "This is unexpected and may indicate a problem with the conversion factors.",
            level="warning",
        )
    # Just grab the first one, because it should be the same for all channels
    channel_conversion_factor_uv = channel_conversion_factors_uv[0]
    logger.debug(f"Channel conversion factor in uV: {channel_conversion_factor_uv}")

    # NOTE channel offsets should be 0 for all channels in openephys data
    channel_conversion_offsets = recording_sliced.get_channel_offsets()
    assert all(channel_conversion_offsets == 0), "Channel conversion offsets are not all 0."

    fs = recording_sliced.get_sampling_frequency()
    logger.debug(f"Open Ephys sampling frequency: {fs}")

    # Remove ephys samples before bonsai start time
    recording_sliced = recording_sliced.frame_slice(start_frame=samples_to_remove, end_frame=None)
    logger.debug(f"Trimmed {samples_to_remove} samples ({float(samples_to_remove/fs)}s) from the start of "
                 "ephys data to exclude data before bonsai start.")

    # Get the original timestamps (in seconds)
    # NOTE THAT THESE TIMESTAMPS DO NOT START FROM 0
    original_timestamps = recording_sliced.get_times()

    # We set the first timestamp (the bonsai start time) as time=0 
    # so the timestamps match the sync pulses for alignment with photometry
    original_timestamps = original_timestamps - original_timestamps[0]

    # Create a SpikeInterfaceRecordingDataChunkIterator using all default buffering and
    # chunking options. This will be passed to the pynwb.ecephys.ElectricalSeries
    # constructor.
    traces_as_iterator = SpikeInterfaceRecordingDataChunkIterator(recording=recording_sliced)

    return (
        traces_as_iterator,
        channel_conversion_factor_uv,
        original_timestamps,
    )


def add_associated_file(nwbfile: NWBFile, name: str, description: str, content: str, logger,
                        task_epochs: str = "0", log_filename: str = None) -> None:
    """
    Add a file to the nwbfile's 'associated_files' processing module (creating the module if needed),
    and optionally save a copy of it alongside the log files

    We save several raw provenance files this way (the Open Ephys settings.xml and structure.oebin, the
    electrode info CSV, etc). This helper centralizes the boilerplate and logs each save.

    Parameters:
        nwbfile (NWBFile): The NWB file being assembled.
        name (str): Name for the AssociatedFiles object
        description (str): Human-readable description of the file
        content (str): The file contents as a string
        logger (Logger): Logger to track conversion progress
        task_epochs (str): Task epoch the file belongs to. Defaults to "0" (Berke Lab has one epoch per day)
        log_filename (str): Optional. If given, also write a copy of the content to the log directory
            under this filename (e.g. "settings.xml"). Skipped if the logger has no logfile.
    """
    if "associated_files" not in nwbfile.processing:
        logger.debug("Creating nwb processing module for associated files")
        nwbfile.create_processing_module(name="associated_files", description="Contains all associated files")
    logger.info(f"Saving '{name}' as an AssociatedFiles object in the nwb ({len(content)} chars)")
    nwbfile.processing["associated_files"].add(AssociatedFiles(
        name=name,
        description=description,
        content=content,
        task_epochs=task_epochs,
    ))

    # Also save a copy to the logging directory
    # Skip if the logger writes only to stdout (no log directory)
    if log_filename is not None:
        try:
            log_dir = get_logger_directory(logger)
        except ValueError:
            logger.debug(f"No log directory available; not saving a copy of '{log_filename}' alongside logs")
            return
        save_path = os.path.join(log_dir, log_filename)
        with open(save_path, "w") as f:
            f.write(content)
        logger.info(f"Saved a copy of '{log_filename}' to the log directory: {save_path}")


def add_probe_info(nwbfile: NWBFile, metadata: dict, logger):
    """
    Add the Probe as a device to the nwbfile

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled.
        metadata (dict):
            Metadata dictionary
        logger (Logger):
            Logger to track conversion progress

    Returns:
        probe_metadata (dict):
            Metadata for this Probe read from resources/ephys_devices.yaml
        probe_obj (Probe):
            The Probe object added to the nwbfile
    """
    # Load probe metadata
    with open(DEVICES_PATH, "r") as f:
        devices = yaml.safe_load(f)

    probe_args = metadata["ephys"]["probe"]
    assert len(probe_args) == 1, "Only one probe is supported at this time."
    probe_name = probe_args[0]
    logger.info(f"Probe name is: {probe_name}")
    # Find the matching device by name in the devices list
    for device in devices["probe"]:
        if device["name"] == probe_name:
            probe_metadata = device
            break
    else:
        logger.error(f"Probe '{probe_name}' not found in resources/ephys_devices.yaml")
        raise ValueError(f"Probe '{probe_name}' not found in resources/ephys_devices.yaml")

    logger.debug("Probe metadata:")
    for key in probe_metadata:
        logger.debug(f"{key}: {probe_metadata[key]}")

    required_probe_keys = {
        "name",
        "description",
        "manufacturer",
        "contact_side_numbering",
        "contact_size",
        "units",
        "shanks",
    }
    assert required_probe_keys.issubset(probe_metadata.keys()), (
        f"Probe is missing required keys {required_probe_keys - probe_metadata.keys()}"
    )
    # Create the Probe device and add it to the nwb
    probe_obj = Probe(
        id=0,  # int
        name=probe_metadata["name"],  # str
        probe_type=probe_metadata["name"],  # str
        probe_description=probe_metadata["description"],  # str
        manufacturer=probe_metadata["manufacturer"],  # str
        contact_side_numbering=probe_metadata["contact_side_numbering"],  # bool
        contact_size=float(probe_metadata["contact_size"]),  # float
        units=probe_metadata["units"],  # str (um or mm)
    )
    nwbfile.add_device(probe_obj)
    return probe_metadata, probe_obj


### Functions for Berke Lab custom probes

def read_open_ephys_settings_xml(settings_file_path: Path, logger) -> str:
    """
    Read the OpenEphys settings.xml file to get the filtering applied across all channels.

    Supports two settings.xml formats:

    Old format (Open Ephys < 1.0):
        <PROCESSOR name="Sources/Rhythm FPGA" ...>
          <CHANNEL_INFO>
            <CHANNEL name="CH1" number="0" gain="0.19499999284744263"/>
            ...
          </CHANNEL_INFO>
          <EDITOR ... LowCut="..." HighCut="..." />
        </PROCESSOR>

    Note that only the information in the "Sources/Rhythm FPGA" is relevant -
    the other processors ("Filters/Channel Map", "Filters/Bandpass Filter", etc.)
    reflect OpenEphys display settings only

    New format (Open Ephys >= 1.0):
        <PROCESSOR name="Acquisition Board" ...>
          <STREAM name="acquisition_board" ... channel_count="264"/>
          <EDITOR ... LowCut="..." HighCut="..." />
        </PROCESSOR>

    NOTE: All we actually use from settings.xml is the filtering info (a single string stored on
    every electrode). We used to also validate the channel numbers/names against the expected
    256 CH + 8 ADC layout, but we stopped because for some of our rats (IM-1875), we did actually 
    have some weird stuff going on there (missing CH1-CH8, which was replaced by a duplicated ADC1-ADC8),
    and we ignored the warnings. A better and more consistent way to do this would be to read that info
    from the structure.oebin instead. I think that would work for old and newer open ephys versions, and
    both Berke lab and neuropixels. Not sure, but putting my thoughts here for posterity (7/6/26) -S

    Parameters:
        settings_file_path (Path):
            Path to the OpenEphys settings.xml file
        logger (Logger):
            Logger to track conversion progress

    Returns:
        filtering_info (str):
            Filtering applied to all channels
    """
    settings_tree = ET.parse(settings_file_path)
    settings_root = settings_tree.getroot()

    # The EDITOR (with the filtering info) lives inside the acquisition processor, whose name
    # differs between the old format ("Sources/Rhythm FPGA") and the new format ("Acquisition Board")
    
    # Try old format first: "Sources/Rhythm FPGA" processor
    acq_processor = settings_root.find(".//PROCESSOR[@name='Sources/Rhythm FPGA']")
    if acq_processor is None:
        # Try new format: "Acquisition Board" processor
        acq_processor = settings_root.find(".//PROCESSOR[@name='Acquisition Board']")
    if acq_processor is None:
        logger.error(
            "Could not find 'Sources/Rhythm FPGA' or 'Acquisition Board' processor in the settings.xml file."
        )
        raise ValueError(
            "Could not find 'Sources/Rhythm FPGA' or 'Acquisition Board' processor in the settings.xml file."
        )

    # Get the filtering info from the EDITOR element (same location in both formats)
    editor = acq_processor.find("EDITOR")
    if editor is not None and "LowCut" in editor.attrib and "HighCut" in editor.attrib:
        lowcut = float(editor.attrib.get("LowCut"))
        highcut = float(editor.attrib.get("HighCut"))
        filtering_info = f"Filter with highcut={highcut} Hz, lowcut={lowcut} Hz"
        logger.info(f"Filtering info from settings.xml: {filtering_info}")
    else:
        logger.warning("EDITOR tag with LowCut/HighCut not found in settings.xml, no filtering info for channels!")
        filtering_info = "Unknown"

    return filtering_info


def get_electrode_info(metadata: dict, logger, fig_dir: Path = None) -> pd.DataFrame:
    """
    Create a DataFrame of complete electrode information for a Berke Lab silicon probe.
    Combines channel map, electrode coordinates, and impedance data.

    Parameters:
        metadata (dict):
            Full metadata dictionary (from user-specified yaml)
        logger (Logger):
            Logger to track conversion progress
        fig_dir (Path):
            Optional. The directory to save the figure. If None, the figure will not be saved.

    Returns:
        pd.DataFrame:
            DataFrame of electrode information including electrode name, shank, electrode number,
            intan channel, relative x and y coordinates, impedance info, and 'bad channel' tag
    """

    # Get info for this probe
    probe_args = metadata["ephys"]["probe"]
    assert len(probe_args) == 1, "Only one probe is supported at this time."
    probe_name = probe_args[0]

    try:
        probe_info = BERKE_LAB_PROBES[probe_name]
    except KeyError:
        logger.error(f"Unknown probe '{probe_name}'. Valid probes are: {list(BERKE_LAB_PROBES)}")
        raise ValueError(f"Unknown probe '{probe_name}'. Valid probes are: {list(BERKE_LAB_PROBES)}")

    # Combine channel map and electrode coords into a single dataframe
    channel_map_df = pd.read_csv(probe_info["channel_map"])
    electrode_coords_df = pd.read_csv(probe_info["electrode_coords"])
    channel_coords_df = pd.merge(channel_map_df, electrode_coords_df, on=["shank", "electrode"], how="outer")

    # Make sure we have exactly 256 rows (one per Intan channel)
    if len(channel_coords_df) != 256:
        logger.error(f"Expected 256 rows in merged probe info, but got {len(channel_coords_df)}")
        raise ValueError(f"Expected 256 rows in merged probe info, but got {len(channel_coords_df)}")

    # Choose the correct channel map based on how the rat was plugged in (assume "chip_first" if none specified)
    plug_order = metadata["ephys"].get("plug_order", "chip_first")
    logger.info(f"Plug order is: {plug_order}")

    # Rename the correct channel map column to 'intan_channel' and remove the unused channel map column
    if plug_order == "chip_first":
        channel_coords_df = channel_coords_df.drop(columns=["cable_first"])
        channel_coords_df = channel_coords_df.rename(columns={"chip_first": "intan_channel"})
    elif plug_order == "cable_first":
        channel_coords_df = channel_coords_df.drop(columns=["chip_first"])
        channel_coords_df = channel_coords_df.rename(columns={"cable_first": "intan_channel"})
    else:
        logger.error(f"Unknown plug order: {plug_order}, expected 'chip_first' or 'cable_first'")
        raise ValueError(f"Unknown plug order: {plug_order}, expected 'chip_first' or 'cable_first'")

    # Add another channel name column based on intan_channel to match Open Ephys 1-based 'CH##' channel names
    channel_coords_df["open_ephys_channel_string"] = channel_coords_df["intan_channel"].apply(lambda i: f"CH{i + 1}")

    # Plot the channel coordinates
    plot_channel_map(probe_name=probe_name, channel_coords=channel_coords_df, fig_dir=fig_dir)

    # Now load impedance data
    impedance_file_path = metadata["ephys"]["impedance_file_path"]
    impedance_data = pd.read_csv(impedance_file_path)

    # Check that the expected columns are present in order and no extra columns are present
    expected_columns = [
        "Channel Number",
        "Channel Name",
        "Port",
        "Enabled",
        "Impedance Magnitude at 1000 Hz (ohms)",
        "Impedance Phase at 1000 Hz (degrees)",
        "Series RC equivalent R (Ohms)",
        "Series RC equivalent C (Farads)",
    ]

    assert impedance_data.columns.tolist() == expected_columns, (
        f"Impedance file has columns {impedance_data.columns.tolist()}, "
        f"does not match expected columns {expected_columns}"
    )
    logger.debug(f"Impedance file has expected columns {expected_columns}")

    # Drop the first column which should be the same as the second column
    assert (impedance_data["Channel Number"] == impedance_data["Channel Name"]).all(), (
        "First column is not the same as the second column."
    )
    impedance_data.drop(columns=["Channel Number"], inplace=True)

    # Make sure the channel coords dataframe and impedance dataframe have the same length (256)
    assert len(channel_coords_df) == len(impedance_data), (
        "Mismatch in lengths: "
        f"channel coordinates ({len(channel_coords_df)}), "
        f"impedance data ({len(impedance_data)})"
    )
    logger.debug(f"Channel coordinates and impedance data have the same length ({len(channel_coords_df)})")

    # Create a column 'intan_channel' (zero-indexed) in the impedance df so we can merge with channel coords df
    impedance_data["intan_channel"] = range(len(impedance_data))
    full_electrode_info_df = channel_coords_df.merge(impedance_data, on="intan_channel", how="left")
    assert len(full_electrode_info_df) == len(channel_coords_df) == 256, (
        "Full electrode info dataframe does not have 256 rows"
    )

    # Sanity check that the channel numbers from the "Channel Name" column match the "intan_channel" column
    # We expect that the impedance file contains either Port B and Port C, or Port A and Port D
    port_prefixes = sorted(full_electrode_info_df["Channel Name"].str.split("-").str[0].unique())
    assert port_prefixes in [["A", "D"], ["B", "C"]], f"Unexpected port prefixes: {port_prefixes}"

    # Get the channel number based on the "Channel Name" column
    # channel nums = the num for Port A/B channels (e.g. B-001 is 1) and num+128 for Port C/D channels (C-001 is 129)
    expected_channel_nums = full_electrode_info_df["Channel Name"].apply(
        lambda s: int(s.split("-")[1]) + (128 if s.startswith(("C-", "D-")) else 0)
    )

    mismatched_channel_nums = full_electrode_info_df["intan_channel"] != expected_channel_nums
    if mismatched_channel_nums.any():
        logger.warning(f"{mismatched_channel_nums.sum()} rows have mismatched intan_channel values")
        logger.warning(full_electrode_info_df[mismatched_channel_nums])
    else:
        logger.debug("All intan channel numbers from coordinates file match channel names from the impedance file!")

    # Mark electrodes with impedances outside the allowed range as "bad channel"
    # The default impedance range is between >=0.1 MOhms or <=3.0 MOhms
    min_impedance = float(metadata["ephys"].get("min_impedance_ohms", MIN_IMPEDANCE_OHMS))
    max_impedance = float(metadata["ephys"].get("max_impedance_ohms", MAX_IMPEDANCE_OHMS))

    logger.info(f"Marking channels with impedance > {max_impedance} Ohms or < {min_impedance} Ohms as 'bad_channel'")
    full_electrode_info_df["bad_channel"] = (
        full_electrode_info_df["Impedance Magnitude at 1000 Hz (ohms)"].lt(min_impedance) |
        full_electrode_info_df["Impedance Magnitude at 1000 Hz (ohms)"].gt(max_impedance)
    ).astype(int)

    # Log the number of good and bad channels
    bad_channel_count = full_electrode_info_df["bad_channel"].sum()
    good_channel_count = len(full_electrode_info_df) - bad_channel_count
    logger.info(f"There are {good_channel_count} good channels and {bad_channel_count} bad channels based on impedance")

    # Plot channel impedances
    plot_channel_impedances(probe_name=probe_name, electrode_info=full_electrode_info_df, 
                            min_impedance=min_impedance, max_impedance=max_impedance, fig_dir=fig_dir)

    return full_electrode_info_df


def add_electrode_data_berke_probe(
    *,
    nwbfile: NWBFile,
    filtering_info: str,
    metadata: dict,
    probe_metadata: dict,
    probe_obj,
    logger,
    fig_dir: Path = None,
) -> list[pd.Series]:
    """
    Add the electrode data from the impedance and channel geometry files.
    Specific to Berke Lab custom probes

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled.
        filtering_info (str):
            The filtering applied to all channels.
        metadata (dict):
            Full metadata dictionary (from user-specified yaml)
        probe_metadata (dict):
            Metadata for this Probe read from resources/ephys_devices.yaml
        probe_obj (Probe):
            The Probe object added to the nwbfile
        logger (Logger):
            Logger to track conversion progress
        fig_dir (Path):
            Optional. The directory to save the figure. If None, the figure will not be saved.
    Returns:
        list[pd.Series]:
            List of information about channels excluded from the electrode table, if any.
            For now, we exclude the extra 4 channels that can be connected to ECoG screws in the 252-channel probes
    """

    # Get dataframe of electrode information (channel map, x/y coordinates, and impedance info)
    electrode_info = get_electrode_info(metadata=metadata, logger=logger, fig_dir=fig_dir)

    # Save electrode info as an AssociatedFiles object in the NWB (and save a copy to the log directory)
    add_associated_file(
        nwbfile,
        name="electrode_info",
        description="Electrode channel map and impedance information (CSV)",
        content=electrode_info.to_csv(index=False),
        logger=logger,
        log_filename="electrode_info.csv",
    )

    # Get general metadata for the Probe
    electrodes_location = metadata["ephys"].get("electrodes_location", "unspecified")
    if electrodes_location == "unspecified":
        logger.warning("No 'electrodes_location' in ephys metadata, setting to 'unspecified'!")
    else:
        logger.info(f"Electrodes location is '{electrodes_location}'")
    targeted_x = metadata["ephys"].get("targeted_x")
    targeted_y = metadata["ephys"].get("targeted_y")
    targeted_z = metadata["ephys"].get("targeted_z")
    logger.info(f"Targeted location is {targeted_x}, {targeted_y}, {targeted_z}")

    electrode_to_shank_map = {}
    electrode_groups_by_shank = {}

    # Process each Shank on the Probe
    for shank_dict in probe_metadata["shanks"]:
        for shank_index, shank_electrode_indices in shank_dict.items():
            shank = Shank(name=str(shank_index))
            logger.debug(f"Adding shank {shank_index} with electrodes {shank_electrode_indices}")

            # Make an ElectrodeGroup for this Shank and add it to the nwb
            electrode_group = NwbElectrodeGroup(
                name=str(shank_index),
                description=f"Electrodes on shank {shank_index}",
                location=electrodes_location,
                targeted_location=electrodes_location,
                targeted_x=float(targeted_x),
                targeted_y=float(targeted_y),
                targeted_z=float(targeted_z),
                units="mm",
                device=probe_obj,
            )
            nwbfile.add_electrode_group(electrode_group)
            # Store the group so we can reference it when adding electrodes to the electrodes table
            electrode_groups_by_shank[shank_index] = electrode_group

            # Add each Shank and its ShanksElectrodes to the Probe 
            for electrode_index in shank_electrode_indices:
                # NOTE: We follow the ndx-franklab-novela/trodes-to-nwb usage of ShanksElectrode
                # even though it is redundant with NWB's electrode table fields, for consistency
                # with Frank Lab when using Spyglass
                shank.add_shanks_electrode(
                    ShanksElectrode(  
                        name=str(electrode_index),
                        rel_x=float(electrode_info.iloc[electrode_index]['x_um']),
                        rel_y=float(electrode_info.iloc[electrode_index]['y_um']),
                        rel_z=0.0,
                    )
                )
                electrode_to_shank_map[electrode_index] = shank_index
            probe_obj.add_shank(shank)

    # Add the electrode data to the NWB file, one column at a time
    nwbfile.add_electrode_column(
        name="electrode_name",
        description="The name of the electrode, in 'S(shank number)E(electrode number)' format",
    )
    nwbfile.add_electrode_column(
        name="intan_channel_number",
        description="The intan channel number (0-indexed) for the electrode",
    )
    nwbfile.add_electrode_column(
        name="open_ephys_channel_str",
        description="The Open Ephys channel name (1-based) for the electrode",
    )
    nwbfile.add_electrode_column(
        name="imp_file_channel_name",
        description="The name of the channel from the impedance file",
    )
    nwbfile.add_electrode_column(
        name="port",
        description="The port of the electrode from the impedance file",
    )
    nwbfile.add_electrode_column(
        name="imp",
        description="The impedance of the electrode (Impedance Magnitude at 1000 Hz (ohms))",
    )
    nwbfile.add_electrode_column(
        name="imp_phase",
        description="The phase of the impedance of the electrode (Impedance Phase at 1000 Hz (degrees))",
    )
    nwbfile.add_electrode_column(
        name="series_resistance_in_ohms",
        description="The series resistance of the electrode (Series RC equivalent R (Ohms))",
    )
    nwbfile.add_electrode_column(
        name="series_capacitance_in_farads",
        description="The series capacitance of the electrode (Series RC equivalent C (Farads))",
    )
    nwbfile.add_electrode_column(
        name="bad_channel",
        description="Whether the channel is a bad channel based on too low or too high impedance",
    )
    nwbfile.add_electrode_column(
        name="rel_x",
        description="The relative x coordinate of the electrode (um)",
    )
    nwbfile.add_electrode_column(
        name="rel_y",
        description="The relative y coordinate of the electrode (um)",
    )
    nwbfile.add_electrode_column(
        name="filtering",
        description="The filtering applied to the electrode",
    )
    nwbfile.add_electrode_column(
        name="probe_electrode",
        description= (
            "The index of the electrode on the probe (0-indexed). Equivalent to electrode number-1"
        ),
    )
    nwbfile.add_electrode_column(
        name="probe_shank",
        description="The index of the shank this electrode is on",
    )
    nwbfile.add_electrode_column(
        name="ref_elect_id",
        description=(
            "The index of the reference electrode in this table. "
            "-1 if not set. Used by Spyglass."
        ),
    )

    # TODO: Figure out an elegant(ish) way to handle ECoG screws.
    # For now I am excluding them entirely.

    # # Make an ElectrodeGroup for unconnected Intan channels (potentially used for ECoG screws)
    # # This is only needed for the 252-channel probes. 
    # # We may re-evaluate this later (they should maybe have a separate Probe object? This is a hack for now.)
    # extra_electrode_group = NwbElectrodeGroup(
    #     name="Other",
    #     description="Intan channels not connected to probe electrodes (may be fully unconnected or ECoG screws)",
    #     location="unknown",
    #     targeted_location="unknown",
    #     targeted_x=0.0,
    #     targeted_y=0.0,
    #     targeted_z=0.0,
    #     units="mm",
    #     device=probe_obj,
    # )
    # added_extra_electrode_group = False

    # Keep track of which channels we are excluding, if any.
    # For now, we exclude channels that have no ElectrodeGroup (aka the extra 4 channels on our 252ch probes)
    # Every row in the ElectricalSeries needs an associated electrode, 
    # so we need to keep track of the channels we exclude to exclude those rows later as well.
    excluded_channels = []

    for i, row in electrode_info.iterrows():
        # Get the Shank and ElectrodeGroup for this electrode
        shank_index = electrode_to_shank_map.get(i, -1)
        electrode_group = electrode_groups_by_shank.get(shank_index, None)

        # TODO: See above note on ECoG screws. Keeping this here for now:

        # # If we have no electrode_group, this is one of the 4 extra channels on the 252-ch probes.
        # # So add the extra_electrode_group to the nwb if we haven't already and assign this electrode to it
        # if electrode_group is None:
        #     if not added_extra_electrode_group:
        #         nwbfile.add_electrode_group(extra_electrode_group)
        #         added_extra_electrode_group = True
        #         logger.debug("Adding extra electrode group for unconnected Intan channels")
        #     electrode_group = extra_electrode_group

        if electrode_group is not None:
            nwbfile.add_electrode(
                electrode_name=row["electrode_name"],
                intan_channel_number=row["intan_channel"],
                open_ephys_channel_str=row["open_ephys_channel_string"],
                imp_file_channel_name=row["Channel Name"],
                port=row["Port"],
                imp=row["Impedance Magnitude at 1000 Hz (ohms)"],
                imp_phase=row["Impedance Phase at 1000 Hz (degrees)"],
                series_resistance_in_ohms=row["Series RC equivalent R (Ohms)"],
                series_capacitance_in_farads=row["Series RC equivalent C (Farads)"],
                bad_channel=row["bad_channel"],  # used by Spyglass
                rel_x=float(row["x_um"]),
                rel_y=float(row["y_um"]),
                group=electrode_group,
                location=electrodes_location, # same for all electrodes
                filtering=filtering_info, # same for all electrodes
                probe_electrode=i,  # used by Spyglass
                probe_shank=shank_index,  # used by Spyglass
                ref_elect_id=-1,  # used by Spyglass
            )
        else:
            logger.debug(f"Excluding '{row["electrode_name"]}' from electrode table:")
            logger.debug(row)
            excluded_channels.append(row)

    return excluded_channels


### Functions for Neuropixels

# Neuropixels 2.0 has 4 shanks with 1280 electrodes each, for 5120 total recording sites.
# The global electrode index (0..5119) encodes the shank: shank = electrode_index // 1280.
NPX_ELECTRODES_PER_SHANK = 1280


def _parse_np_probe_element(probe, tag: str, logger) -> dict:
    """
    Parse one per-channel element of an NP_PROBE (e.g. CHANNELS, ELECTRODE_INDEX, ELECTRODE_XPOS).

    These elements store one attribute per recording channel, keyed by channel name ("CH0", "CH1", ...).
    Returns a dict of {channel_num (int): value (str)} for the "CH#" attributes.
    """
    element = probe.find(tag)
    if element is None:
        logger.error(f"Could not find {tag} in the NP_PROBE in the settings.xml file.")
        raise ValueError(f"Could not find {tag} in the NP_PROBE in the settings.xml file.")
    return {int(name[2:]): value for name, value in element.attrib.items() if name.startswith("CH")}


def get_channel_map_neuropixels(settings_file_path, logger, fig_dir=None) -> dict:
    """
    Read the OpenEphys settings.xml for a Neuropixels 2.0 multishank probe and build a channel map:
    which physical electrode each recording channel (CH0..CH383) is wired to, plus its geometry.

    Each NP_PROBE element in settings.xml stores, per channel:
        CHANNELS        : "bank:shank" (we take the shank)
        ELECTRODE_INDEX : global electrode index (0..5119) = shank * 1280 + local_electrode
        ELECTRODE_XPOS / ELECTRODE_YPOS : physical position on the probe (um)

    We treat ELECTRODE_INDEX as the authoritative identity and merge it against our canonical coordinate
    table (neuropixels_2.0_multishank_electrode_coords.csv, one row per global electrode) to get the
    shank, shank_column, shank_row, and canonical x_um/y_um. We merge on the electrode index rather than
    on x/y: the settings.xml x positions use a slightly different origin than our coords table (a constant
    ~19um offset), so I need to check this.

    Parameters:
        settings_file_path (Path):
            Path to the OpenEphys settings.xml file
        logger (Logger):
            Logger to track conversion progress
        fig_dir (Path):
            Optional. The directory to save associated figures. If None, the figures will not be saved

    Returns:
        dict:
            probe_id -> DataFrame with one row per recording channel and columns:
            channel_num, channel_name, bank, shank, electrode, shank_column, shank_row, x_um, y_um
    """
    # Load canonical coordinates for all 5120 potential Neuropixels recording sites
    all_neuropixels_electrodes_info = pd.read_csv(ELECTRODE_COORDS_PATH_NPX_MULTISHANK)

    settings_root = ET.parse(settings_file_path).getroot()
    probes = settings_root.findall(".//NP_PROBE")
    if not probes:
        logger.error("Could not find any NP_PROBE elements in the settings.xml file.")
        raise ValueError("Could not find any NP_PROBE elements in the settings.xml file.")

    # Parse each probe (there will be more than one if we ever do bilateral recordings)
    neuropixels_channel_data = {}
    for i, probe in enumerate(probes):
        probe_id = f"probe_{i}"
        logger.debug(f"NP_PROBE {i}: electrodeConfigurationPreset = '{probe.get('electrodeConfigurationPreset')}'")

        # Parse the per-channel elements, each keyed by channel number
        bank_shank = _parse_np_probe_element(probe, "CHANNELS", logger)  # "bank:shank"
        electrode_index = _parse_np_probe_element(probe, "ELECTRODE_INDEX", logger)
        xpos = _parse_np_probe_element(probe, "ELECTRODE_XPOS", logger)
        ypos = _parse_np_probe_element(probe, "ELECTRODE_YPOS", logger)

        channel_map = {}
        for channel_num, bank_shank_str in bank_shank.items():
            bank_str, shank_str = bank_shank_str.split(":")
            channel_map[channel_num] = {
                "channel_name": f"CH{channel_num}",
                "bank": int(bank_str),
                "shank": int(shank_str),
                "electrode": int(electrode_index[channel_num]),
                "x_um_settings": int(xpos[channel_num]),
                "y_um_settings": int(ypos[channel_num]),
            }

        channel_df = pd.DataFrame.from_dict(channel_map, orient="index")
        channel_df["channel_num"] = channel_df.index.astype(int)
        channel_df = channel_df.sort_values("channel_num").reset_index(drop=True)

        # Cross-check: the shank from CHANNELS should match the shank implied by the electrode index
        shank_from_electrode = channel_df["electrode"] // NPX_ELECTRODES_PER_SHANK
        if not (channel_df["shank"] == shank_from_electrode).all():
            logger.warning(f"{probe_id}: shank from CHANNELS disagrees with electrode_index // "
                           f"{NPX_ELECTRODES_PER_SHANK} for some channels!")

        # Merge with the canonical coordinate table on the global electrode index to validate and enrich.
        # This pulls in shank_column, shank_row, and canonical x_um/y_um (the 'shank' also comes from here)
        logger.debug(f"Validating electrode indices for Neuropixels {probe_id} against the coords table...")
        merged = channel_df.drop(columns=["shank"]).merge(
            all_neuropixels_electrodes_info[["electrode", "shank", "shank_column", "shank_row", "x_um", "y_um"]],
            on="electrode", how="left", validate="one_to_one",
        )

        # Check for electrode indices we couldn't match to a known recording site
        unmatched = merged[merged["shank_row"].isnull()]
        if len(unmatched) > 0:
            logger.error(f"{probe_id}: {len(unmatched)} channels have electrode indices not found in the "
                         f"coords table:\n{unmatched[['channel_num', 'electrode']]}")
            raise ValueError(f"{probe_id}: electrode indices not found in coords table: "
                             f"{sorted(unmatched['electrode'])}")

        # Sanity-check our canonical coords against what OpenEphys reports in settings.xml. 
        # They should match exactly, warn if a future probe config ever disagrees
        # Then drop the now-redundant settings columns to keep the channel info clean
        x_mismatch = (merged["x_um"] != merged["x_um_settings"]).sum()
        y_mismatch = (merged["y_um"] != merged["y_um_settings"]).sum()
        if x_mismatch or y_mismatch:
            logger.warning(f"{probe_id}: coords table disagrees with settings.xml for "
                           f"{x_mismatch} x and {y_mismatch} y positions - check the coords file!")
        merged = merged.drop(columns=["x_um_settings", "y_um_settings"])

        assert len(merged) == 384, f"Expected 384 channels, got {len(merged)}"
        logger.info(f"Neuropixels {probe_id}: all {len(merged)} channels matched to electrodes successfully!")
        logger.debug(f"{probe_id}: recording channels per shank: {merged.groupby('shank').size().to_dict()}")
        logger.debug(f"{probe_id}: electrode index range {int(merged['electrode'].min())}-"
                     f"{int(merged['electrode'].max())}, y range {int(merged['y_um'].min())}-"
                     f"{int(merged['y_um'].max())} um")
        neuropixels_channel_data[probe_id] = merged

        # Plot the electrode layout and channel info for this probe
        plot_neuropixels(all_neuropixels_electrodes=all_neuropixels_electrodes_info,
                         channel_info=merged,
                         probe_name=probe_id,
                         fig_dir=fig_dir)

    return neuropixels_channel_data


def add_electrode_data_neuropixels(
    *,
    nwbfile: NWBFile,
    channel_info: pd.DataFrame,
    filtering_info: str,
    metadata: dict,
    probe_obj,
    logger,
) -> None:
    """
    Add electrode groups and the electrode table for a Neuropixels probe.

    Analogous to add_electrode_data_berke_probe, but simpler: Neuropixels has no per-session impedance
    file, so we don't mark bad channels here (all channels are treated as good). We create one
    ElectrodeGroup per shank that has recording sites, and add one electrode per recording channel.

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled.
        channel_info (pd.DataFrame):
            One row per recording channel (from get_channel_map_neuropixels), with columns
            channel_num, shank, electrode, shank_column, shank_row, x_um, y_um
        filtering_info (str):
            The filtering applied to all channels (Neuropixels applies hardware filtering)
        metadata (dict):
            Full metadata dictionary (from user-specified yaml)
        probe_obj (Probe):
            The Probe object added to the nwbfile
        logger (Logger):
            Logger to track conversion progress
    """
    # Save electrode info as an AssociatedFiles object in the NWB (and save a copy to the log directory)
    add_associated_file(
        nwbfile,
        name="electrode_info",
        description="Neuropixels channel map (recording channel -> electrode geometry) (CSV)",
        content=channel_info.to_csv(index=False),
        logger=logger,
        log_filename="electrode_info.csv",
    )

    # General metadata for the electrode groups
    electrodes_location = metadata["ephys"].get("electrodes_location", "unspecified")
    if electrodes_location == "unspecified":
        logger.warning("No 'electrodes_location' in ephys metadata, setting to 'unspecified'!")
    else:
        logger.info(f"Electrodes location is '{electrodes_location}'")
    targeted_x = metadata["ephys"].get("targeted_x")
    targeted_y = metadata["ephys"].get("targeted_y")
    targeted_z = metadata["ephys"].get("targeted_z")
    logger.info(f"Targeted location is {targeted_x}, {targeted_y}, {targeted_z}")

    # Make an ElectrodeGroup and a Shank for each shank that actually has recording sites.
    # NOTE: we group one ElectrodeGroup per shank (mirrors the Berke probe per-shank structure). 
    # We might later change to an electrode group per contiguous y block.
    electrode_groups_by_shank = {}
    shanks_by_shank = {}
    for shank_index in sorted(channel_info["shank"].unique()):
        electrode_group = NwbElectrodeGroup(
            name=str(shank_index),
            description=f"Electrodes on shank {shank_index}",
            location=electrodes_location,
            targeted_location=electrodes_location,
            targeted_x=float(targeted_x),
            targeted_y=float(targeted_y),
            targeted_z=float(targeted_z),
            units="mm",
            device=probe_obj,
        )
        nwbfile.add_electrode_group(electrode_group)
        electrode_groups_by_shank[shank_index] = electrode_group
        shanks_by_shank[shank_index] = Shank(name=str(shank_index))
        logger.debug(f"Adding shank {shank_index}")

    # Add the electrode data columns (mirrors the Berke table, minus the impedance columns)
    nwbfile.add_electrode_column(
        name="electrode_name",
        description="The name of the electrode, in 'S(shank number)E(electrode number)' format",
    )
    nwbfile.add_electrode_column(
        name="intan_channel_number",
        description="The recording channel number (0-indexed) for the electrode (CH0..CH383 -> 0..383)",
    )
    nwbfile.add_electrode_column(
        name="electrode_index",
        description="The global Neuropixels electrode index (0-indexed, 0..5119)",
    )
    nwbfile.add_electrode_column(
        name="bad_channel",
        description="Whether the channel is a bad channel. Always False for Neuropixels (no impedance file)",
    )
    nwbfile.add_electrode_column(
        name="rel_x",
        description="The relative x coordinate of the electrode (um)",
    )
    nwbfile.add_electrode_column(
        name="rel_y",
        description="The relative y coordinate of the electrode (um)",
    )
    nwbfile.add_electrode_column(
        name="filtering",
        description="The filtering applied to the electrode",
    )
    nwbfile.add_electrode_column(
        name="probe_electrode",
        description="The global index of the electrode on the probe (0-indexed)",
    )
    nwbfile.add_electrode_column(
        name="probe_shank",
        description="The index of the shank this electrode is on",
    )
    nwbfile.add_electrode_column(
        name="ref_elect_id",
        description="The index of the reference electrode in this table. -1 if not set. Used by Spyglass.",
    )

    # Add electrodes in channel_num order (0..383) so the electrode table row -> raw data row mapping
    # (built downstream from intan_channel_number) is correct
    for _, row in channel_info.sort_values("channel_num").iterrows():
        shank_index = int(row["shank"])
        electrode_index = int(row["electrode"])
        local_electrode = electrode_index % NPX_ELECTRODES_PER_SHANK

        # Follow the ndx-franklab-novela/trodes-to-nwb usage of ShanksElectrode (for Spyglass consistency)
        shanks_by_shank[shank_index].add_shanks_electrode(
            ShanksElectrode(
                name=str(local_electrode),
                rel_x=float(row["x_um"]),
                rel_y=float(row["y_um"]),
                rel_z=0.0,
            )
        )
        nwbfile.add_electrode(
            electrode_name=f"S{shank_index:02d}E{local_electrode}",
            intan_channel_number=int(row["channel_num"]),
            electrode_index=electrode_index,
            bad_channel=False,  # no impedance file for Neuropixels, so no channels are marked bad
            rel_x=float(row["x_um"]),
            rel_y=float(row["y_um"]),
            group=electrode_groups_by_shank[shank_index],
            location=electrodes_location,  # same for all electrodes
            filtering=filtering_info,  # same for all electrodes
            probe_electrode=electrode_index,  # used by Spyglass
            probe_shank=shank_index,  # used by Spyglass
            ref_elect_id=-1,  # used by Spyglass
        )

    # Add each populated Shank to the Probe
    for shank in shanks_by_shank.values():
        probe_obj.add_shank(shank)

    logger.info(f"Added {len(channel_info)} Neuropixels electrodes across {len(electrode_groups_by_shank)} "
                f"shank ElectrodeGroup(s): {sorted(electrode_groups_by_shank)}")


def add_raw_ephys(
    *,
    nwbfile: NWBFile,
    metadata: dict,
    logger,
    fig_dir: Path = None,
) -> None:
    """Add the raw ephys data to a NWB file.

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled.
        metadata (dict):
            Metadata dictionary
        logger (Logger):
            Logger to track conversion progress
        fig_dir (Path):
            Optional. The directory to save associated figures. If None, the figures will not be saved.
    """

    if "ephys" not in metadata:
        log_and_print(logger, "No ephys metadata found for this session. Skipping ephys conversion.")
        return {}

    # If we do have "ephys" in metadata, check for the required keys.
    # (impedance_file_path is required for Berke Lab probes but not Neuropixels, so it is checked
    # in the Berke branch below rather than here.)
    required_ephys_keys = {"openephys_folder_path", "probe"}
    missing_keys = required_ephys_keys - metadata["ephys"].keys()
    if missing_keys:
        log_and_print(
            logger,
            "The required ephys subfields do not exist in the metadata dictionary.\n"
            "Remove the 'ephys' field from metadata if you do not have ephys data "
            f"for this session, \nor specify the following missing subfields: {missing_keys}",
            level="warning",
        )
        return {}

    log_and_print(logger, "Adding raw ephys...")
    openephys_folder_path = metadata["ephys"]["openephys_folder_path"]

    # Get Open Ephys start time as datetime object based on the time specified in the folder name
    openephys_folder_name = openephys_folder_path.split('/')[-1]
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", openephys_folder_name)
    if match:
        datetime_str = match.group(0)
        open_ephys_start = datetime.strptime(datetime_str, "%Y-%m-%d_%H-%M-%S")
    else:
        raise ValueError(f"No valid datetime string found in path: {openephys_folder_path}")

    open_ephys_start = open_ephys_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    logger.info(f"Open Ephys start time: {open_ephys_start}")

    # Get paths to the settings.xml file and the continuous.dat files
    open_ephys_paths_dict = find_open_ephys_paths(open_ephys_folder_path=openephys_folder_path, experiment_number=1)
    settings_file_path = open_ephys_paths_dict["settings_file"]
    recording_file_paths_dict = open_ephys_paths_dict["recording_files"]

    # Separate the neural data stream(s) from the ADC stream. For Berke Lab probes, the ADC channels live
    # inside the single probe continuous.dat (so there is no separate ADC stream). For Neuropixels (OneBox),
    # the ADC is its own stream/folder (e.g. "OneBox-100.OneBox-ADC") separate from the probe ("...ProbeA").
    adc_file_paths = {name: path for name, path in recording_file_paths_dict.items() if "ADC" in name.upper()}
    probe_file_paths = {name: path for name, path in recording_file_paths_dict.items() if "ADC" not in name.upper()}

    # For now, only allow a single probe
    if len(probe_file_paths) != 1:
        logger.error(f"Expected exactly one continuous.dat, found: {probe_file_paths}")
        raise NotImplementedError("Currently only one continuous.dat file is supported.")
    probe_stream_name, continuous_dat_file_path = next(iter(probe_file_paths.items()))
    logger.info(f"Found continuous.dat for '{probe_stream_name}' at {continuous_dat_file_path}")

    logger.info("Adding probe...")
    probe_metadata, probe_obj = add_probe_info(nwbfile=nwbfile, metadata=metadata, logger=logger)
    
    # Read recording params (channel count, sample rate, conversion factor) from structure.oebin
    # This works the same for Berke lab probes and Neuropixels
    oebin_params = read_oebin_params(continuous_dat_file_path=continuous_dat_file_path, logger=logger)
    total_channels = oebin_params["channel_count"]

    # Berke Lab probes and Neuropixels store their electrode info and port visits differently, so each
    # branch sets up its electrode table and tells us where to find the port visit pulses:
    #   port_visits_* describe the continuous.dat file/channel/rate that the port visit pulses live on
    #   pulse_high_threshold is the raw int16 value above which the port visit channel is considered "high"
    #   exclude_channel_names are neural channels to drop from the ElectricalSeries (Berke ECoG screws)
    if probe_metadata["name"] in BERKE_LAB_PROBES:
        log_and_print(logger, f"Processing '{probe_metadata['name']}' as a Berke Lab custom probe.")

        # Berke Lab probes require a per-session impedance file (used to mark bad channels)
        if "impedance_file_path" not in metadata["ephys"]:
            raise ValueError("Berke Lab probes require 'impedance_file_path' in the ephys metadata.")

        # Sanity-check the oebin channels against the expected 256 "CH" + 8 "ADC" layout
        expected_channel_names = {f"CH{i}" for i in range(1, 257)} | {f"ADC{i}" for i in range(1, 9)}
        validate_oebin_channel_names(oebin_params["channel_names"], expected_channel_names, logger)

        # For Berke Lab probes, port visits are recorded on ADC1, which lives in the same continuous.dat
        # as the neural data channels (256 "CH" + 8 "ADC" = 264 total channels)
        port_visits_dat_file_path = continuous_dat_file_path
        port_visits_total_channels = total_channels
        port_visits_sample_rate = oebin_params["sample_rate"]
        port_visits_channel_num = 256  # Port visits are recorded on ADC1, aka channel 256 (zero-indexed)
        pulse_high_threshold = 10_000  # raw int16 threshold

        # Read the settings.xml file to get filtering info
        filtering_info = read_open_ephys_settings_xml(settings_file_path=settings_file_path, logger=logger)

        # Create electrode groups and add electrode data to the NWB file
        excluded_channels = add_electrode_data_berke_probe(
                                    nwbfile=nwbfile,
                                    filtering_info=filtering_info,
                                    metadata=metadata,
                                    probe_metadata=probe_metadata,
                                    probe_obj=probe_obj,
                                    logger=logger,
                                    fig_dir=fig_dir,
                                )

        # Get Open Ephys names of channels to exclude for use by OpenEphysBinaryRecordingExtractor
        exclude_channel_names = [row["open_ephys_channel_string"] for row in excluded_channels]

    else:
        log_and_print(logger, f"Processing '{probe_metadata['name']}' as a Neuropixels probe.")

        # For Neuropixels (OneBox), port visits are recorded on a SEPARATE OneBox-ADC stream (its own
        # continuous.dat), not in the probe's continuous.dat. Find that ADC stream and read its params.
        if not adc_file_paths:
            raise ValueError("No OneBox-ADC stream found for Neuropixels; cannot read port visits.")
        if len(adc_file_paths) > 1:
            raise NotImplementedError(f"Expected exactly one ADC stream, found: {adc_file_paths}")
        adc_stream_name, adc_dat_file_path = next(iter(adc_file_paths.items()))
        logger.info(f"Found OneBox-ADC continuous.dat for '{adc_stream_name}' at {adc_dat_file_path}")
        adc_oebin_params = read_oebin_params(continuous_dat_file_path=adc_dat_file_path, logger=logger)

        # ADC data is stored as raw int16 like everything else; convert our desired volts threshold to raw
        # int16 using the ADC bit_volts. Port visits default to ADC0 (the "selected" ADC channel in
        # settings.xml); both the channel and threshold are overridable via metadata.
        port_visits_dat_file_path = adc_dat_file_path
        port_visits_total_channels = adc_oebin_params["channel_count"]
        port_visits_sample_rate = adc_oebin_params["sample_rate"]
        port_visits_channel_num = metadata["ephys"].get("port_visits_adc_channel", NPX_DEFAULT_PORT_VISITS_ADC_CHANNEL)
        threshold_volts = metadata["ephys"].get("port_visits_threshold_volts", NPX_PORT_VISITS_THRESHOLD_VOLTS)
        pulse_high_threshold = threshold_volts / adc_oebin_params["bit_volts"]
        logger.info(f"Reading Neuropixels port visits from ADC channel {port_visits_channel_num} "
                    f"with threshold {threshold_volts} V ({pulse_high_threshold:.0f} raw int16)")

        # Neuropixels applies hardware filtering (not recorded in settings.xml like the Berke bandpass)
        filtering_info = "Neuropixels 2.0 hardware filtering (see IMEC Neuropixels documentation)"

        # Build the channel map, then create electrode groups and add electrode data to the NWB file.
        # We only support a single probe for now, so take the one probe's channel info.
        channel_map = get_channel_map_neuropixels(settings_file_path=settings_file_path,
                                                  logger=logger, fig_dir=fig_dir)
        if len(channel_map) != 1:
            raise NotImplementedError(f"Expected exactly one Neuropixels probe, found {len(channel_map)}.")
        channel_info = next(iter(channel_map.values()))

        add_electrode_data_neuropixels(
            nwbfile=nwbfile,
            channel_info=channel_info,
            filtering_info=filtering_info,
            metadata=metadata,
            probe_obj=probe_obj,
            logger=logger,
        )

        # All Neuropixels recording channels are kept (no ECoG-screw channels to exclude)
        exclude_channel_names = []

    # Get port visits recorded by Open Ephys for timestamp alignment
    logger.info("Getting port visits recorded by Open Ephys...")
    logger.debug(f"Port visit source: channel {port_visits_channel_num} of {port_visits_dat_file_path} "
                 f"({port_visits_total_channels} channels @ {port_visits_sample_rate} Hz, "
                 f"threshold {pulse_high_threshold:.0f} raw int16)")
    ephys_visit_times, bonsai_start_time = get_port_visits(continuous_dat_file_path=port_visits_dat_file_path,
                                                           total_channels=port_visits_total_channels,
                                                           port_visits_channel_num=port_visits_channel_num,
                                                           sample_rate=port_visits_sample_rate,
                                                           logger=logger,
                                                           pulse_high_threshold=pulse_high_threshold)
    log_and_print(logger, f"Open Ephys recorded {len(ephys_visit_times)} port visits.")
    logger.debug(f"Open Ephys port visits: {ephys_visit_times}")

    # Convert the bonsai start time (seconds) to a number of samples in the probe's sample rate.
    # This matters for Neuropixels: port visits come from the OneBox-ADC (30300.5 Hz) but we trim the
    # probe data (30000 Hz), so we need to convert
    probe_sample_rate = oebin_params["sample_rate"]
    samples_to_remove = round(bonsai_start_time * probe_sample_rate)
    logger.debug(f"Trimming {samples_to_remove} samples ({bonsai_start_time}s at {probe_sample_rate} Hz) "
                 "from the start of the probe data")

    # Get raw ephys data (with times before bonsai start removed)
    (
        traces_as_iterator,
        channel_conversion_factor_uv,
        original_timestamps,
    ) = get_raw_ephys_data(folder_path=openephys_folder_path, 
                           logger=logger, 
                           exclude_channels=exclude_channel_names,
                           samples_to_remove=samples_to_remove
                           )
    num_samples, num_channels = traces_as_iterator.maxshape

    # Check that the number of electrodes in the NWB file is the same as the number of channels in traces_as_iterator
    assert (len(nwbfile.electrodes) == num_channels), (
        f"Number of electrodes in NWB file ({len(nwbfile.electrodes)}) does not match number of channels "
        f"in traces_as_iterator ({num_channels})."
    )
    logger.debug(f"There are {len(nwbfile.electrodes)} electrodes in the nwbfile "
                 "(same as number of channels in traces_as_iterator)")

    # Get electrodes table (added to nwb in `add_electrode_data_berke_probe`)
    electrodes = nwbfile.electrodes
    # Get the mapping of electrode to channel number (0-indexed)
    electrode_table_row_to_raw_data_row = electrodes['intan_channel_number'].data[:]
    # Get the reverse mapping of row in the raw ElectricalSeries to row in the electrode table
    raw_data_row_to_electrode_table_row = np.argsort(electrode_table_row_to_raw_data_row)

    # Create the electrode table region encompassing all electrodes
    electrode_table_region = nwbfile.create_electrode_table_region(
        region=list(raw_data_row_to_electrode_table_row),
        description="Electrodes used in raw ElectricalSeries recording",
    )
    logger.info(f"Raw ephys data is {num_samples} samples x {num_channels} channels "
                f"(conversion factor {channel_conversion_factor_uv} uV/bit)")

    # Convert to uV without loading the whole thing at once
    uv_traces_as_iterator = MicrovoltsSpikeInterfaceRecordingDataChunkIterator(traces_as_iterator,
                                                                               channel_conversion_factor_uv)

    # A chunk of shape (81920, 64) and dtype int16 (2 bytes) is ~10 MB, which is the recommended chunk size
    # by the NWB team.
    # We could also add compression here. zstd/blosc-zstd are recommended by the NWB team, but
    # they require the hdf5plugin library to be installed. gzip is available by default.
    # Use gzip for now, but consider zstd/blosc-zstd in the future.
    data_data_io = H5DataIO(
        data=uv_traces_as_iterator,
        chunks=(min(num_samples, 81920), min(num_channels, 64)),
        compression="gzip",
    )
    logger.info(f"ElectricalSeries data will be written with chunks "
                f"{(min(num_samples, 81920), min(num_channels, 64))} and gzip compression")

    # If we have ground truth port visit times (photometry), align timestamps to that
    ground_truth_time_source = metadata.get("ground_truth_time_source")
    if ground_truth_time_source is not None:
        logger.info(f"Aligning ephys visit times to ground truth ({ground_truth_time_source})")
        ground_truth_visit_times = metadata.get("ground_truth_visit_times")
        ephys_timestamps = align_via_interpolation(unaligned_timestamps=original_timestamps,
                                                   unaligned_visit_times=ephys_visit_times,
                                                   ground_truth_visit_times=ground_truth_visit_times,
                                                   logger=logger)
    else:
        # If we don't have photometry, keep the original timestamps
        ephys_timestamps = original_timestamps
        logger.info("No ground truth visit times (photometry) found. "
                    "Ephys visit times are now our ground truth visit times.")

        # Ephys visit times are now our ground truth visit times
        metadata["ground_truth_time_source"] = "ephys"
        metadata["ground_truth_visit_times"] = ephys_visit_times

    # Create the ElectricalSeries
    # For now, we do not chunk or compress the timestamps, which are relatively small
    eseries = ElectricalSeries(
        name="ElectricalSeries",
        description="Raw ephys data from OpenEphys recording, in uV (multiply by conversion factor to get data in V).",
        data=data_data_io,
        timestamps=ephys_timestamps,
        electrodes=electrode_table_region,
        conversion=VOLTS_PER_MICROVOLT, # conversion from uV to V
    )

    # Add the ElectricalSeries to the NWBFile
    logger.info("Adding raw ephys to the nwbfile as an ElectricalSeries")
    nwbfile.add_acquisition(eseries)

    # Save the raw Open Ephys settings.xml and structure.oebin as AssociatedFiles
    with open(settings_file_path, "r") as settings_file:
        raw_settings_xml = settings_file.read()
    add_associated_file(
        nwbfile,
        name="open_ephys_settings_xml",
        description="Raw settings.xml file from OpenEphys",
        content=raw_settings_xml,
        logger=logger,
        log_filename="settings.xml",
    )

    with open(oebin_params["oebin_path"], "r") as oebin_file:
        raw_structure_oebin = oebin_file.read()
    add_associated_file(
        nwbfile,
        name="open_ephys_structure_oebin",
        description="Raw structure.oebin file from OpenEphys (recording metadata: streams, channels, "
                    "bit_volts, sample rates)",
        content=raw_structure_oebin,
        logger=logger,
        log_filename="structure.oebin",
    )

    log_and_print(logger, "Finished adding raw ephys to the nwb.")
    return {"ephys_start": open_ephys_start, "port_visits": ephys_visit_times}
