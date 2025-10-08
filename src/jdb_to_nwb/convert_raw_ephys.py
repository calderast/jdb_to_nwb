# NOTE: We could have used NeuroConv to convert the ephys data but we want to use the Frank Lab's 
# ndx-franklab-novela extension to store the Probe information for maximal integration with Spyglass, 
# so we are doing the conversion manually using PyNWB.

import os
import re
import glob
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

from .utils import get_logger_directory
from .timestamps_alignment import align_via_interpolation
from .plotting.plot_ephys import plot_channel_map, plot_channel_impedances, plot_neuropixels
from ndx_franklab_novela import AssociatedFiles, Probe, NwbElectrodeGroup, Shank, ShanksElectrode

MICROVOLTS_PER_VOLT = 1e6
VOLTS_PER_MICROVOLT = 1 / MICROVOLTS_PER_VOLT

MIN_IMPEDANCE_OHMS = 1e5
MAX_IMPEDANCE_OHMS = 3e6

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


def get_port_visits(continuous_dat_file_path: Path, 
                    total_channels: int, 
                    port_visits_channel_num: int, 
                    logger) -> list[float]:
    """
    Extract port visit times from OpenEphys continuous.dat file.

    TODO: Make sure this works for both Berke Lab custom probes and Neuropixels.
    This function has not yet been tested for Neuropixels.
    Ideally, making this work for Neuropixels (or any other probe) is as simple as changing 
    total_channels and port_visits_channel_num to match that configuration.
    But other things like the pulse_high_threshold etc may need to be adjusted as well.
    Hold off on testing this until we actually decide to use Neuropixels or a different probe.

    For Berke Lab probes:
    total_channels = 264  # 256 "CH" + 8 "ADC"
    port_visits_channel_num = 256 # Port visits are recorded on ADC1, aka channel 256 (zero-indexed)

    Parameters:
        continuous_dat_file_path (Path):
            Path to the 'continuous.dat' OpenEphys binary recording
        total_channels (int):
            The total number of channels in the continuous.dat file
        port_visits_channel_num (int):
            The number of the channel (zero-indexed) that port visits are recorded on.
        logger (Logger):
            Logger to track conversion progress

    Returns:
        port_visits (list[float]):
            List of port visit times in seconds
    """

    pulse_high_threshold = 10_000
    openephys_fs = 30_000
    downsampled_fs = 1000
    # 1000ish is a reasonable downsample frequency, because it's low enough to speeds things up
    # (this takes ~2mins on my machine) but high enough that we definitely don't miss any port visit pulses 
    # For reference, a typical port visit keeps the channel high for ~10ms, so 10 samples at 1000 Hz

    # Memory-map the large .dat file to avoid loading everything into memory. Reshape to (samples, channels)
    data_for_all_channels = np.memmap(continuous_dat_file_path, dtype='int16',mode='c').reshape(-1, total_channels) 

    logger.debug(f"Reading Open Ephys port visits from channel {port_visits_channel_num}")
    logger.debug(f"Downsampling data to from {openephys_fs} Hz to {downsampled_fs} Hz so it isn't huge")

    # Extract the channel that records port visits and downsample so the data isn't massive
    visits_channel_data = data_for_all_channels[:, port_visits_channel_num]
    visits_channel_data = visits_channel_data[::int(openephys_fs / downsampled_fs)]

    # Find indices where the visits channel is high (aka port visit times)
    logger.debug(f"Recording times the channel is high (>{pulse_high_threshold})")
    pulse_above_threshold = np.where(visits_channel_data > pulse_high_threshold)[0]

    # If no port visits were found, return an empty list
    if pulse_above_threshold.size == 0:
        return []

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

    # Align to first pulse (which marks bonsai start time) and convert to seconds
    # NOTE: The duration of the first pulse is longer than a normal pulse (~500ms instead of ~10ms)
    # Should we add a check for this to ensure we have identified the correct "start" pulse?
    logger.info("Removing the first ephys pulse, as it marks start time and not a true port visit")
    port_visits = pulse_starts[1:] - pulse_starts[0]
    port_visits = [float(visit / downsampled_fs) for visit in port_visits]
    return port_visits


def get_raw_ephys_data(
    folder_path: Path,
    logger,
    exclude_channels: list[str] = []
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

    Returns:
        traces_as_iterator (SpikeInterfaceRecordingDataChunkIterator):
            To be used as the data argument in pynwb.ecephys.ElectricalSeries.
        channel_conversion_factor (float):
            The conversion factor from the raw data to volts.
        original_timestamps (np.ndarray):
            Array that could be used as the timestamps argument in pynwb.ecephys.ElectricalSeries
            or may need to be time aligned with the other data streams in the NWB file.
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

    # TODO: For bilateral Neuropixels we will have 2 streams without ADC (one for each probe)
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
        print(
            "The channel conversion factors are not the same for all channels. "
            "This is unexpected and may indicate a problem with the conversion factors."
        )
        logger.warning(
            "The channel conversion factors are not the same for all channels. "
            "This is unexpected and may indicate a problem with the conversion factors."
        )
    channel_conversion_factor_v = channel_conversion_factors_uv[0] * VOLTS_PER_MICROVOLT
    logger.debug(f"Channel conversion factor in V: {channel_conversion_factor_v}")

    # NOTE channel offsets should be 0 for all channels in openephys data
    channel_conversion_offsets = recording_sliced.get_channel_offsets()
    assert all(channel_conversion_offsets == 0), "Channel conversion offsets are not all 0."

    # Get the original timestamps (in seconds)
    original_timestamps = recording_sliced.get_times()

    logger.debug(f"Open Ephys sampling frequency: {recording_sliced.get_sampling_frequency()}")

    # Create a SpikeInterfaceRecordingDataChunkIterator using all default buffering and
    # chunking options. This will be passed to the pynwb.ecephys.ElectricalSeries
    # constructor.
    traces_as_iterator = SpikeInterfaceRecordingDataChunkIterator(recording=recording_sliced)

    return (
        traces_as_iterator,
        channel_conversion_factor_v,
        original_timestamps,
    )


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

def read_open_ephys_settings_xml(settings_file_path: Path, logger) -> tuple[dict, str]:
    """
    Get the raw ephys metadata from the OpenEphys binary recording.

    Read the OpenEphys settings.xml file to get the mapping of channel number to channel name
    and the filtering applied across all channels. 
    
    Note that only the information in the "Sources/Rhythm FPGA" is relevant - 
    the other processors ("Filters/Channel Map", "Filters/Bandpass Filter", etc.) 
    reflect OpenEphys display settings only 

    <PROCESSOR name="Sources/Rhythm FPGA" ...>
      <CHANNEL_INFO>
        <CHANNEL name="CH1" number="0" gain="0.19499999284744263"/>
        ...
      </CHANNEL_INFO>
      ...
    </PROCESSOR>

    Parameters:
        settings_file_path (Path):
            Path to the OpenEphys settings.xml file
        logger (Logger):
            Logger to track conversion progress

    Returns:
        channel_number_to_channel_name (dict):
            Dict mapping channel number (0-263) to channel name (e.g. 'CH1', 'CH2', etc.)
        filtering_info (str):
            Filtering applied to all channels
    """
    settings_tree = ET.parse(settings_file_path)
    settings_root = settings_tree.getroot()
    rhfp_processor = settings_root.find(".//PROCESSOR[@name='Sources/Rhythm FPGA']")
    if rhfp_processor is None:
        logger.error("Could not find the Rhythm FPGA processor in the settings.xml file.")
        raise ValueError("Could not find the Rhythm FPGA processor in the settings.xml file.")
    channel_info = rhfp_processor.find("CHANNEL_INFO")
    if channel_info is None:
        logger.error("Could not find the CHANNEL_INFO node in the settings.xml file.")
        raise ValueError("Could not find the CHANNEL_INFO node in the settings.xml file.")
    channel_number_to_channel_name = {
        int(channel.attrib["number"]): channel.attrib["name"] for channel in channel_info.findall("CHANNEL")
    }

    # Check that the channel numbers and channel names are what we expect
    expected_channel_numbers = set(range(264))  # 0 to 263, for 256 channels + 8 ADC
    expected_channel_names = {f"CH{i}" for i in range(1, 257)} | {f"ADC{i}" for i in range(1, 9)}

    channel_numbers_from_settings_xml = set(channel_number_to_channel_name.keys())
    channel_names_from_settings_xml = set(channel_number_to_channel_name.values())

    # Check for missing or unexpected channel numbers
    missing_channel_numbers = expected_channel_numbers - channel_numbers_from_settings_xml
    unexpected_channel_numbers = channel_numbers_from_settings_xml - expected_channel_numbers
    if missing_channel_numbers or unexpected_channel_numbers:
        logger.warning("Channel numbers in settings.xml do not match expectations!!!!")
        if missing_channel_numbers:
            logger.warning(f"Missing channel numbers: {sorted(missing_channel_numbers)}")
        if unexpected_channel_numbers:
            logger.warning(f"Unexpected channel numbers: {sorted(unexpected_channel_numbers)}")
    else:
        logger.info("All expected channel numbers found in OpenEphys settings.xml")
        logger.debug(f"Found channels: {sorted(channel_numbers_from_settings_xml)}")

    # Check for missing or unexpected channel names
    missing_channel_names = expected_channel_names - channel_names_from_settings_xml
    unexpected_channel_names = channel_names_from_settings_xml - expected_channel_names
    if missing_channel_names or unexpected_channel_names:
        logger.warning("Channel names in settings.xml do not match expectations!!!!")
        if missing_channel_names:
            logger.warning(f"Missing channel names: {sorted(missing_channel_names)}")
        if unexpected_channel_names:
            logger.warning(f"Unexpected channel names: {sorted(unexpected_channel_names)}")
    else:
        logger.info("All expected channel names found in OpenEphys settings.xml")
        logger.debug(f"Found channels: {sorted(channel_names_from_settings_xml)}")

    # Check for duplicate channel names
    channel_name_counts = Counter(list(channel_number_to_channel_name.values()))
    duplicate_names = [name for name, count in channel_name_counts.items() if count > 1]
    if duplicate_names:
        logger.warning("Duplicate channel names found in settings.xml!!!!")
        logger.warning(f"Duplicate channel names: {sorted(duplicate_names)}")

    # Get the filtering info
    editor = rhfp_processor.find("EDITOR")
    if editor is not None:
        lowcut = float(editor.attrib.get("LowCut"))
        highcut = float(editor.attrib.get("HighCut"))
        filtering_info = f"Filter with highcut={highcut} Hz, lowcut={lowcut} Hz"
        logger.info(f"Filtering info from settings.xml: {filtering_info}")
    else:
        logger.warning("EDITOR tag not found in settings.xml, no filtering info for channels!")
        filtering_info = "Unknown"

    return (
        channel_number_to_channel_name,
        filtering_info,
    )


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
    # channel nums = the number for Port B channels (e.g. B-001 is 1) and number+128 for Port C channels (C-001 is 129)
    expected_channel_nums = full_electrode_info_df["Channel Name"].apply(
        lambda s: int(s.split("-")[1]) + (128 if s.startswith("C-") else 0)
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

    # Save electrode info to the log directory
    log_dir = get_logger_directory(logger)
    save_path = os.path.join(log_dir, "electrode_info.csv")
    full_electrode_info_df.to_csv(save_path, index=False)
    logger.info(f"Saved electrode info to {save_path}")

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

def get_channel_map_neuropixels(settings_file_path, logger, fig_dir=None) -> dict:
    """
    Read the openephys settings.xml file for a Neuropixels 2.0 mulitshank probe
    to get the mapping of channel number to channel name, electrode_xpos, and electrode_ypos.

    Merge with data for all Neuropixels electrodes to get shank row, shank column, 
    and electrode ID for each channel.

    Parameters:
        settings_file_path (Path):
            Path to the OpenEphys settings.xml file
        logger (Logger):
            Logger to track conversion progress
        fig_dir (Path):
            Optional. The directory to save associated figures. If None, the figures will not be saved

    Returns:
        dict:
            Dictionary of probe_id: dataframe of channel info
    """
    # Load electrode coordinates for all 5120 potential Neuropixels recording sites
    all_neuropixels_electrodes_info = pd.read_csv(ELECTRODE_COORDS_PATH_NPX_MULTISHANK)

    # Initialize dict to store channel info for each probe from the settings.xml
    neuropixels_channel_data = {}

    # Load the settings.xml
    settings_tree = ET.parse(settings_file_path)
    settings_root = settings_tree.getroot()

    # Parse each PROBE
    probes = settings_root.findall(".//NP_PROBE")
    for i, probe in enumerate(probes):
        # Make an id for each probe in case there are multiple (there will be 2 if we do bilateral recordings)
        probe_id = f"probe_{i}"

        # Get electrode config preset name
        logger.debug(f"NP_PROBE {i}: electrodeConfigurationPreset = '{probe.get('electrodeConfigurationPreset')}'")

        # Initialize channel map for this probe
        channel_map = {}

        # Parse CHANNELS
        channels_element = probe.find("CHANNELS")
        if channels_element is not None:
            for channel_name, shank_info in channels_element.attrib.items():
                if channel_name.startswith("CH"):
                    channel_num = int(channel_name[2:])
                    bank_str, shank_str = shank_info.split(":")
                    channel_map[channel_num] = {
                        "channel_name": channel_name, 
                        "shank": int(shank_str), 
                        "bank(?)": int(bank_str) # TODO: I don't actually know what this is
                    }
        else:
            logger.error("Could not find CHANNELS in the settings.xml file.")
            raise ValueError("Could not find CHANNELS in the settings.xml file.")

        # Parse ELECTRODE_XPOS
        xpos_element = probe.find("ELECTRODE_XPOS")
        if xpos_element is not None:
            for channel_name, xpos in xpos_element.attrib.items():
                if channel_name.startswith("CH"):
                    channel_num = int(channel_name[2:])
                    channel_map[channel_num]["x_um"] = int(xpos)
        else:
            logger.error("Could not find ELECTRODE_XPOS in the settings.xml file.")
            raise ValueError("Could not find ELECTRODE_XPOS in the settings.xml file.")

        # Parse ELECTRODE_YPOS
        ypos_element = probe.find("ELECTRODE_YPOS")
        if ypos_element is not None:
            for channel_name, ypos in ypos_element.attrib.items():
                if channel_name.startswith("CH"):
                    channel_num = int(channel_name[2:])
                    channel_map[channel_num]["y_um"] = int(ypos)
        else:
            logger.error("Could not find ELECTRODE_YPOS in the settings.xml file.")
            raise ValueError("Could not find ELECTRODE_YPOS in the settings.xml file.")

        # Create dataframe of channel info from the settings.xml
        channel_info_from_settings = pd.DataFrame.from_dict(channel_map, orient='index')
        channel_info_from_settings['channel_num'] = channel_info_from_settings.index.astype(int)
        channel_info_from_settings = channel_info_from_settings.sort_values(by='channel_num').reset_index(drop=True)
        logger.debug(f"Channel info from settings.xml for Neuropixels probe {probe_id}:")
        logger.debug(channel_info_from_settings)

        # Merge this probe's channel info with all electrode coordinates
        logger.debug(f"Validating electrode coordinates for Neuropixels {probe_id} "
                    "by merging with data for all possible Neuropixels recording sites...")
        full_channel_info_df = pd.merge(
            channel_info_from_settings,
            all_neuropixels_electrodes_info,
            on=["x_um", "y_um", "shank"],
            how="left",  # ensures we detect any unmatched rows
            validate="one_to_one"
        )

        # Check for unmatched electrodes
        if full_channel_info_df["electrode"].isnull().any():
            missing = full_channel_info_df[full_channel_info_df["electrode"].isnull()]
            logger.error(
                f"Missing electrode match for {len(missing)} channels in {probe_id}:\n"
                f"{missing[['channel_num', 'x_um', 'y_um', 'shank']]}"
            )
            raise ValueError(
                f"Missing electrode match for {len(missing)} channels in {probe_id}:\n"
                f"{missing[['channel_num', 'x_um', 'y_um', 'shank']]}"
            )

        # Check that we have info for 384 recording sites as expected
        assert len(full_channel_info_df) == 384, f"Expected 384 channels, got {len(full_channel_info_df)}"
        logger.info(f"Neuropixels {probe_id}: all 384 channels matched and merged successfully!")
        neuropixels_channel_data[probe_id] = full_channel_info_df

        # Plot electrode layour and channel info for this probe
        plot_neuropixels(all_neuropixels_electrodes=all_neuropixels_electrodes_info,
                         channel_info=channel_info_from_settings,
                         probe_name=probe_id,
                         fig_dir=fig_dir)
    return neuropixels_channel_data


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
        print("No ephys metadata found for this session. Skipping ephys conversion.")
        logger.info("No ephys metadata found for this session. Skipping ephys conversion.")
        return {}

    # If we do have "ephys" in metadata, check for the required keys
    required_ephys_keys = {"openephys_folder_path", "probe", "impedance_file_path"}
    missing_keys = required_ephys_keys - metadata["ephys"].keys()
    if missing_keys:
        print(
            "The required ephys subfields do not exist in the metadata dictionary.\n"
            "Remove the 'ephys' field from metadata if you do not have ephys data "
            f"for this session, \nor specify the following missing subfields: {missing_keys}"
        )
        logger.warning(
            "The required ephys subfields do not exist in the metadata dictionary.\n"
            "Remove the 'ephys' field from metadata if you do not have ephys data "
            f"for this session, \nor specify the following missing subfields: {missing_keys}"
        )
        return {}

    print("Adding raw ephys...")
    logger.info("Adding raw ephys...")
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
    # This is set up for future flexibility (finds the correct paths for multiple probes and multiple recordings),
    # but we don't use that flexibility yet. I will implement processing for multiple probes if we decide to do
    # bilateral neuropixels recordings.
    open_ephys_paths_dict = find_open_ephys_paths(open_ephys_folder_path=openephys_folder_path, experiment_number=1)
    settings_file_path = open_ephys_paths_dict["settings_file"]
    recording_file_paths_dict = open_ephys_paths_dict["recording_files"]

    # For now, only allow a single continuous.dat file
    if len(recording_file_paths_dict) == 1:
        probe_name, continuous_dat_file_path = next(iter(recording_file_paths_dict.items()))
        logger.info(f"Found continuous.dat for probe '{probe_name}' at {continuous_dat_file_path}")
    else:
        logger.error("Currently only one continuous.dat file is supported!")
        logger.error(f"Found multiple probes/files: {recording_file_paths_dict}")
        raise NotImplementedError("Currently only one continuous.dat file is supported.")
    
    logger.info("Adding probe...")
    probe_metadata, probe_obj = add_probe_info(nwbfile=nwbfile, metadata=metadata, logger=logger)
    
    # We have different information in different places for Berke Lab probes vs other probe types (Neuropixels).
    # This is a bit messy because our old code assumed Berke Lab probes only. But is ok for now.
    if probe_metadata["name"] in BERKE_LAB_PROBES:
        # For Berke Lab probes:
        total_channels = 264  # 256 "CH" + 8 "ADC"
        port_visits_channel_num = 256 # Port visits are recorded on ADC1, aka channel 256 (zero-indexed)

        # Read the settings.xml file to get electrode info
        (
            channel_number_to_channel_name,
            filtering_info
        ) = read_open_ephys_settings_xml(settings_file_path=settings_file_path, logger=logger)

        # NOTE: We currently don't use channel_number_to_channel_name (except to complain when things don't match).
        # Ultimately this should be passed to add_electrode_data_berke_probe and used for validation.
        # I have excluded this for now because for our current rats (IM-1875), we do actually have some weird 
        # stuff going on there (missing CH1-CH8, which is replaced by a duplicated ADC1-ADC8).
        # This is reflected in both the settings.xml and the structure.oebin files, and (rightfully) breaks
        # a bunch of stuff. A current workaround is just to rename the offending channels in these files
        # (which is bad practice!! but ok for now because they would be excluded by impedance criteria anyway)
        # If you do this please note it and also save a copy of the original files. 
        # But for now we must choose our battles and I'm skipping this one.

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
        # For Neuropixels (NOTE these are fake placeholder values):
        total_channels = 392 # I actually have no idea. Guessing 384 "CH" + 8(??) "ADC" ?
        port_visits_channel_num = 384 # Also no idea. 

        # Read the settings.xml file to get channel map
        channel_info = get_channel_map_neuropixels(settings_file_path=settings_file_path, 
                                                   logger=logger, 
                                                   fig_dir=fig_dir)

        logger.debug("Neuropixels channel info:")
        logger.debug(channel_info)
        # TODO: Use channel map to add electrode data to the nwbfile
        # Figure out impedance and filtering info - not sure where that lives
        # Work with Sam to figure out how to name the probe and create custom electrode groups
        # based on the geometry of each channel map
        raise NotImplementedError("Full processing for Neuropixels not yet implemented.")

    # Get raw ephys data
    (
        traces_as_iterator,
        channel_conversion_factor_v,
        original_timestamps,
    ) = get_raw_ephys_data(folder_path=openephys_folder_path, logger=logger, exclude_channels=exclude_channel_names)
    num_samples, num_channels = traces_as_iterator.maxshape

    # Get port visits recorded by Open Ephys for timestamp alignment
    logger.info("Getting port visits recorded by Open Ephys...")
    ephys_visit_times = get_port_visits(continuous_dat_file_path=continuous_dat_file_path, 
                                        total_channels=total_channels,
                                        port_visits_channel_num=port_visits_channel_num,
                                        logger=logger)
    print(f"Open Ephys recorded {len(ephys_visit_times)} port visits.")
    logger.info(f"Open Ephys recorded {len(ephys_visit_times)} port visits.")
    logger.debug(f"Open Ephys port visits: {ephys_visit_times}")

    # Check that the number of electrodes in the NWB file is the same as the number of channels in traces_as_iterator
    assert (len(nwbfile.electrodes) == num_channels), (
        f"Number of electrodes in NWB file ({len(nwbfile.electrodes)}) does not match number of channels "
        f"in traces_as_iterator ({num_channels})."
    )
    logger.debug(f"There are {len(nwbfile.electrodes)} electrodes in the nwbfile "
                 "(same as number of channels in traces_as_iterator)")

    # Create the electrode table region encompassing all electrodes
    electrode_table_region = nwbfile.create_electrode_table_region(
        region=list(range(len(nwbfile.electrodes))),
        description="Electrodes used in raw ElectricalSeries recording",
    )

    # A chunk of shape (81920, 64) and dtype int16 (2 bytes) is ~10 MB, which is the recommended chunk size
    # by the NWB team.
    # We could also add compression here. zstd/blosc-zstd are recommended by the NWB team, but
    # they require the hdf5plugin library to be installed. gzip is available by default.
    # Use gzip for now, but consider zstd/blosc-zstd in the future.
    data_data_io = H5DataIO(
        traces_as_iterator,
        chunks=(min(num_samples, 81920), min(num_channels, 64)),
        compression="gzip",
    )

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
        description="Raw ephys data from OpenEphys recording (multiply by conversion factor to get data in volts).",
        data=data_data_io,
        timestamps=ephys_timestamps,
        electrodes=electrode_table_region,
        conversion=channel_conversion_factor_v,
    )

    # Add the ElectricalSeries to the NWBFile
    logger.info("Adding raw ephys to the nwbfile as an ElectricalSeries")
    nwbfile.add_acquisition(eseries)

    # Get the raw settings.xml file as a string to be used to create an AssociatedFiles object
    with open(settings_file_path, "r") as settings_file:
        raw_settings_xml = settings_file.read()

    # Create an AssociatedFiles object to save settings.xml
    raw_settings_xml_file = AssociatedFiles(
        name="open_ephys_settings_xml",
        description="Raw settings.xml file from OpenEphys",
        content=raw_settings_xml,
        task_epochs="0",  # Berke Lab only has one epoch (session) per day
    )

    # If it doesn't exist already, make a processing module for associated files
    if "associated_files" not in nwbfile.processing:
        logger.debug("Creating nwb processing module for associated files")
        nwbfile.create_processing_module(name="associated_files", description="Contains all associated files")

    # Add settings.xml to the nwb as an associated file
    logger.debug("Saving the Open Ephys settings.xml file as an AssociatedFiles object")
    nwbfile.processing["associated_files"].add(raw_settings_xml_file)

    return {"ephys_start": open_ephys_start, "port_visits": ephys_visit_times}
