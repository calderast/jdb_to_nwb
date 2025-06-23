# NOTE: We could have used NeuroConv to convert the ephys data but we want to use the Frank Lab's 
# ndx-franklab-novela extension to store the Probe information for maximal integration with Spyglass, 
# so we are doing the conversion manually using PyNWB.

import os
import re
import glob
from datetime import datetime
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET
from pathlib import Path
from importlib.resources import files
import pandas as pd
from hdmf.backends.hdf5 import H5DataIO
import numpy as np
import yaml
from neuroconv.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
    SpikeInterfaceRecordingDataChunkIterator,
)
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor

from .timestamps_alignment import align_via_interpolation
from .plotting.plot_ephys import plot_channel_map
from ndx_franklab_novela import AssociatedFiles, Probe, NwbElectrodeGroup, Shank, ShanksElectrode

MICROVOLTS_PER_VOLT = 1e6
VOLTS_PER_MICROVOLT = 1 / MICROVOLTS_PER_VOLT

MIN_IMPEDANCE_OHMS = 1e5
MAX_IMPEDANCE_OHMS = 3e6

# Names of custom Berke Lab probes
BERKE_LAB_PROBES = {"256-ch Silicon Probe, 3mm length, 66um pitch", "256-ch Silicon Probe, 6mm length, 80um pitch"}

# Get the location of the resources directory when the package is installed from pypi
__location_of_this_file = Path(files(__name__))
RESOURCES_DIR = __location_of_this_file / "resources"

# If the resources directory does not exist, we are probably running the code from the source directory
if not RESOURCES_DIR.exists():
    RESOURCES_DIR = __location_of_this_file.parent.parent / "resources"

BERKE_PROBE_CHANNEL_MAP_PATH = RESOURCES_DIR / "berke_probe_channel_map.csv"
ELECTRODE_COORDS_PATH_3MM_PROBE = RESOURCES_DIR / "3mm_probe_66um_pitch_electrode_coords.csv"
ELECTRODE_COORDS_PATH_6MM_PROBE = RESOURCES_DIR / "6mm_probe_80um_pitch_electrode_coords.csv"
ELECTRODE_COORDS_PATH_NPX_MULTISHANK = RESOURCES_DIR / "neuropixels_2.0_multishank_electrode_coords.csv"
DEVICES_PATH = RESOURCES_DIR / "ephys_devices.yaml"


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

    # Align to first pulse (which marks photometry(?) start time) and convert to seconds
    # NOTE: The duration of the first pulse is longer than a normal pulse (~500ms instead of ~10ms)
    # should we add a check for this to ensure we have identified the correct "start" pulse?
    logger.info("Removing the first ephys pulse, as it marks start time and not a true port visit")
    port_visits = pulse_starts[1:] - pulse_starts[0]
    port_visits = [float(visit / downsampled_fs) for visit in port_visits]
    return port_visits


def get_raw_ephys_data(
    folder_path: Path,
    logger
) -> tuple[SpikeInterfaceRecordingDataChunkIterator, float, np.ndarray, list[str]]:
    """
    Get the raw ephys data from the OpenEphys binary recording.

    Parameters:
        folder_path (Path):
            Path to the folder containing the OpenEphys binary recording. The folder
            should have the date in the name and contain a file called "settings.xml".
        logger (Logger):
            Logger to track conversion progress

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
    channel_ids_to_convert = [ch for ch in recording.channel_ids if ch.startswith("CH")]
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

def get_raw_ephys_metadata_berke_probe(settings_file_path: Path, logger) -> tuple[list[str], list[int], list[int]]:
    """
    Get the raw ephys metadata from the OpenEphys binary recording.

    Read the openephys settings.xml file to get the mapping of channel number to channel name.

    <PROCESSOR name="Sources/Rhythm FPGA" ...>
      <CHANNEL_INFO>
        <CHANNEL name="CH1" number="0" gain="0.19499999284744263"/>
        ...
      </CHANNEL_INFO>
      ...
    </PROCESSOR>

    Read the settings.xml file to get the filtering applied to each channel.

    Read the settings.xml file to get the channel map.

    Parameters:
        settings_file_path (Path):
            Path to the OpenEphys settings.xml file
        logger (Logger):
            Logger to track conversion progress

    Returns:
        filtering_list (list[str]):
            The filtering applied to each channel.
        headstage_channel_numbers (list[int]):
            The headstage channel numbers for each channel.
        reference_daq_channel_indices (list[int]):
            The reference DAQ channel indices for each channel (-1 if not set).
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
        channel.attrib["number"]: channel.attrib["name"] for channel in channel_info.findall("CHANNEL")
    }

    # Get the filtering info
    filtering_list = get_filtering_info_berke_probe(settings_root, channel_number_to_channel_name, logger)

    # Get the channel map info
    headstage_channel_numbers, reference_daq_channel_indices = get_channel_map_info_berke_probe(
        settings_root, channel_number_to_channel_name
    )

    return (
        filtering_list,
        headstage_channel_numbers,
        reference_daq_channel_indices,
    )


def get_filtering_info_berke_probe(settings_root: ET.Element, 
                                   channel_number_to_channel_name: dict[str, str], 
                                   logger) -> list[str]:
    """
    Get the filtering applied to each channel from the settings.xml file.

    Read the settings.xml file to get the filtering applied to each channel - map channel number to filter description

    <PROCESSOR name="Filters/Bandpass Filter" ...>
      <CHANNEL name="0" number="0">
        <SELECTIONSTATE param="1" record="0" audio="0"/>
        <PARAMETERS highcut="6000" lowcut="1" shouldFilter="1"/>
      </CHANNEL>
      ...
    </PROCESSOR>

    Parameters:
        settings_root (ET.Element):
            The root of the settings.xml file.
        channel_number_to_channel_name (dict[str, str]):
            Mapping of channel number to channel name.
        logger (Logger):
            Logger to track conversion progress

    Returns:
        filtering_list (list[str]):
            The filtering applied to each channel.
    """
    bandpass_filter = settings_root.find(".//PROCESSOR[@name='Filters/Bandpass Filter']")
    filtering = {}
    if bandpass_filter is not None:
        for channel in bandpass_filter.findall("CHANNEL"):
            # Ignore the ADC channels
            if not channel_number_to_channel_name.get(channel.attrib["number"]).startswith("CH"):
                continue
            highcut = channel.find("PARAMETERS").attrib["highcut"]
            lowcut = channel.find("PARAMETERS").attrib["lowcut"]
            should_filter = channel.find("PARAMETERS").attrib["shouldFilter"]
            if should_filter == "1":
                filtering[channel.attrib["number"]] = (
                    f"2nd-order Butterworth filter with highcut={highcut} Hz and lowcut={lowcut} Hz"
                )
            else:
                filtering[channel.attrib["number"]] = "No filtering"
    else:
        logger.error("No bandpass filter found in the settings.xml file.")
        raise ValueError("No bandpass filter found in the settings.xml file.")

    # Check that the channel numbers in filtering go from "0" to "N-1"
    # where N is the number of channels
    assert all(
        int(ch_num) == i for i, ch_num in enumerate(filtering.keys())
    ), "Channel numbers in filtering do not go from 0 to N-1."
    filtering_list = list(filtering.values())

    # Warn if the channel filtering is not the same for all channels
    if not all(f == filtering_list[0] for f in filtering_list):
        print(
            "The channel filtering is not the same for all channels. "
            "This is unexpected and may indicate a problem with the filtering settings."
        )
        logger.warning(
            "The channel filtering is not the same for all channels. "
            "This is unexpected and may indicate a problem with the filtering settings."
        )
    return filtering_list


def get_channel_map_info_berke_probe(
    settings_root: ET.Element, 
    channel_number_to_channel_name: dict[str, str]
) -> tuple[list[int], list[int]]:
    """
    Get the channel map info from the settings.xml file.

    Read the settings.xml file to get the mapping of daq / data channel index (0-indexed) to headstage channel
    index (1-indexed) and get the reference channels for each channel
    The GUI for this older version of the OpenEphys Channel Map filter is documented here:
    https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/950421/Channel+Map

    <PROCESSOR name="Filters/Channel Map" ...>
      <CHANNEL name="0" number="0">
        <SELECTIONSTATE param="0" record="0" audio="0"/>
      </CHANNEL>
      ...
      <EDITOR isCollapsed="0" displayName="Channel Map" Type="ChannelMappingEditor">
        <SETTING Type="visibleChannels" Value="6"/>
        <CHANNEL Number="0" Mapping="1" Reference="-1" Enabled="1"/>
        <CHANNEL Number="1" Mapping="2" Reference="-1" Enabled="1"/>
        <CHANNEL Number="2" Mapping="3" Reference="-1" Enabled="1"/>
        ...
        <REFERENCE Number="0" Channel="2"/>
        <REFERENCE Number="1" Channel="-1"/>
        ...
      </EDITOR>
    </PROCESSOR>

    Parameters:
        settings_root (ET.Element):
            The root of the settings.xml file.
        channel_number_to_channel_name (dict[str, str]):
            Mapping of channel number to channel name.

    Returns:
        headstage_channel_numbers (list[int]):
            The headstage channel numbers for each channel.
        reference_daq_channel_indices (list[int]):
            The reference DAQ channel indices for each channel (-1 if not set).
    """
    channel_map = settings_root.find(".//PROCESSOR[@name='Filters/Channel Map']")
    if channel_map is None:
        raise ValueError("Could not find the Channel Map processor in the settings.xml file.")
    channel_map_editor = channel_map.find("EDITOR")
    if channel_map_editor is None:
        raise ValueError("Could not find the EDITOR node in the settings.xml file.")

    # Get the reference channel if set
    reference_channels = list()
    for i, reference in enumerate(channel_map_editor.findall("REFERENCE")):
        assert int(reference.attrib["Number"]) == i, "Reference number does not match index."
        reference_channels.append(int(reference.attrib["Channel"]))
        # Ryan believes the reference channel is the daq / data channel index and not the headstage channel index

    # Get the channel map
    headstage_channel_numbers = list()
    reference_daq_channel_indices = list()
    for i, channel in enumerate(channel_map_editor.findall("CHANNEL")):
        # There should not be any disabled channels and this code was not tested with disabled channels.
        # TODO Before removing this assertion, check that the code below works when a channel is disabled.
        assert int(channel.attrib["Enabled"]) == 1, "Channel is not enabled."
        assert int(channel.attrib["Number"]) == i, "Channel number does not match index."

        # Ignore the ADC channels
        if not channel_number_to_channel_name.get(channel.attrib["Number"]).startswith("CH"):
            continue

        # This code was not tested with reference channels.
        # TODO Before removing this assertion, check that the code below works when a channel is a reference.
        assert int(channel.attrib["Reference"]) == -1, "Channel is a reference."
        if int(channel.attrib["Reference"]) != -1:
            reference_daq_channel_indices.append(reference_channels[int(channel.attrib["Reference"])])
        else:
            reference_daq_channel_indices.append(-1)

        headstage_channel_numbers.append(int(channel.attrib["Mapping"]))

    return headstage_channel_numbers, reference_daq_channel_indices


def add_electrode_data_berke_probe(
    *,
    nwbfile: NWBFile,
    filtering_list: list[str],
    headstage_channel_numbers: list[int],
    reference_daq_channel_indices: list[int],
    metadata: dict,
    probe_metadata: dict,
    probe_obj,
    logger,
    fig_dir: Path = None,
):
    """
    Add the electrode data from the impedance and channel geometry files.
    Specific to Berke Lab custom probes

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled.
        filtering_list (list[str]):
            The filtering applied to each channel.
        headstage_channel_numbers (list[int]):
            The headstage channel numbers for each channel.
        reference_daq_channel_indices (list[int]):
            The reference DAQ channel indices for each channel.
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
    """

    # Get the channel map based on how the rat was plugged in (assume "chip_first" if none specified)
    plug_order = metadata["ephys"].get("plug_order", "chip_first")
    logger.info(f"Plug order is: {plug_order}")

    channel_map_df = pd.read_csv(BERKE_PROBE_CHANNEL_MAP_PATH)
    channel_map = np.array(channel_map_df[plug_order])

    # Under the "chip_first" channel map, the first channel has index 191 (0-indexed). 
    # The coordinates for this channel are at row index 191 in the channel_geometry dataframe 
    # (electrode coords CSV).

    # Get electrode coordinates as a (2, 256) array based on the probe name
    # The first column is the relative x coordinate, and the second column is the relative y coordinate
    probe_name = probe_metadata["name"]
    if probe_name == "256-ch Silicon Probe, 3mm length, 66um pitch":
        electrode_coords_path = ELECTRODE_COORDS_PATH_3MM_PROBE
    elif probe_name == "256-ch Silicon Probe, 6mm length, 80um pitch":
        electrode_coords_path = ELECTRODE_COORDS_PATH_6MM_PROBE
    else:
        raise ValueError(f"Unknown probe '{probe_name}' has no associated electrode coordinates file.")

    logger.info(f"Using electrode coords specified in {electrode_coords_path}")
    channel_geometry = pd.read_csv(electrode_coords_path)

    assert len(channel_geometry) == len(channel_map), (
        "Mismatch in lengths: "
        f"channel_geometry ({len(channel_geometry)}), "
        f"channel_map ({len(channel_map)}), "
    )
    logger.debug(f"Lengths of channel geometry and channel map match (length={len(channel_map)})!")

    plot_channel_map(probe_name, channel_geometry, fig_dir=fig_dir)

    # Get general metadata for the Probe
    electrodes_location = metadata["ephys"].get("electrodes_location")
    logger.info(f"Electrodes location is {electrodes_location}")
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
                        rel_x=float(channel_geometry["x"][channel_map[electrode_index]]),
                        rel_y=float(channel_geometry["y"][channel_map[electrode_index]]),
                        rel_z=0.0,
                    )
                )
                electrode_to_shank_map[electrode_index] = shank_index
            probe_obj.add_shank(shank)

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

    assert len(channel_map) == len(impedance_data), (
        "Mismatch in lengths: "
        f"channel_map ({len(channel_map)}), "
        f"impedance_data ({len(impedance_data)})"
    )
    logger.debug(f"Channel map and impedance data have the same length ({len(channel_map)})")
    
    # Drop the first column which should be the same as the second column
    assert (impedance_data["Channel Number"] == impedance_data["Channel Name"]).all(), (
        "First column is not the same as the second column."
    )
    impedance_data.drop(columns=["Channel Number"], inplace=True)

    # Check that the filtering list has the same length as the number of channels
    assert len(filtering_list) == len(impedance_data), (
        f"Filtering list does not have the same length ({len(filtering_list)}) "
        f"as the number of channels ({len(impedance_data)})."
    )
    logger.debug(f"Filtering list has the same length ({len(filtering_list)}) as number of channels")

    assert len(headstage_channel_numbers) == len(impedance_data), (
        f"Headstage channel numbers do not have the same length ({len(headstage_channel_numbers)}) "
        f"as the number of channels ({len(impedance_data)})."
    )
    logger.debug(
        f"Headstage channel numbers have the same length ({len(headstage_channel_numbers)}) as number of channels"
    )

    assert len(reference_daq_channel_indices) == len(impedance_data), (
        f"Reference DAQ channel indices do not have the same length ({len(reference_daq_channel_indices)}) "
        f"as the number of channels ({len(impedance_data)})."
    )
    logger.debug(
        f"Reference DAQ channel indices have the same length ({len(reference_daq_channel_indices)}) "
        "as number of channels"
    )

    # Convert the headstage channel numbers to 0-indexed
    headstage_channel_indices = np.array(headstage_channel_numbers) - 1

    # Check that the headstage channel indices are equal to the channel map
    # The channel map should have been encoded in the OpenEphys settings file during the recording
    # and that should match the channel map from the resources directory.
    if not np.all(headstage_channel_indices == channel_map):
        logger.warning(
            "Headstage channel indices are not equal to the channel map. "
            "This is unexpected and may indicate a problem with the channel map."
        )

    # Apply the channel map to the reference DAQ channel indices to get the reference electrode ID (0-indexed)
    ref_elect_id = []
    for i in reference_daq_channel_indices:
        if i != -1:
            ref_elect_id.append(headstage_channel_indices[i])  # TODO test this
        else:
            ref_elect_id.append(-1)

    # Mark electrodes with impedance that is less than 0.1 MOhms or more than 3.0 MOhms
    # as bad electrodes
    logger.info(f"Marking channels with impedance>{MAX_IMPEDANCE_OHMS} or <{MIN_IMPEDANCE_OHMS} as 'bad_channel'")
    bad_channel_mask = (
        impedance_data["Impedance Magnitude at 1000 Hz (ohms)"] < MIN_IMPEDANCE_OHMS) | (
        impedance_data["Impedance Magnitude at 1000 Hz (ohms)"] > MAX_IMPEDANCE_OHMS
    )

    # Add the electrode data to the NWB file, one column at a time
    nwbfile.add_electrode_column(
        name="channel_name",
        description="The name of the channel from the impedance file",
    )
    nwbfile.add_electrode_column(
        name="port",
        description="The port of the electrode from the impedance file",
    )
    nwbfile.add_electrode_column(
        name="enabled",
        description="Whether the channel is enabled from the impedance file",
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
        description="The relative x coordinate of the electrode",
    )
    nwbfile.add_electrode_column(
        name="rel_y",
        description="The relative y coordinate of the electrode",
    )
    nwbfile.add_electrode_column(
        name="filtering",
        description="The filtering applied to the electrode",
    )
    nwbfile.add_electrode_column(
        name="headstage_channel_number",
        description="The headstage channel number (1-indexed) for the electrode",
    )
    nwbfile.add_electrode_column(
        name="probe_electrode",
        description= (
            "The index of the electrode on the probe. "
            "There is only one probe, so this is equivalent to electrode index"
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
            "-1 if not set. Also known as 'reference_daq_channel_index'"
        ),
    )

    for i, row in impedance_data.iterrows():
        # Get the Shank and ElectrodeGroup for this electrode
        shank_index = electrode_to_shank_map[i]
        electrode_group = electrode_groups_by_shank[shank_index]
        channel_num = channel_map[i]

        nwbfile.add_electrode(
            channel_name=row["Channel Name"],
            port=row["Port"],
            enabled=bool(row["Enabled"]),
            imp=row["Impedance Magnitude at 1000 Hz (ohms)"],
            imp_phase=row["Impedance Phase at 1000 Hz (degrees)"],
            series_resistance_in_ohms=row["Series RC equivalent R (Ohms)"],
            series_capacitance_in_farads=row["Series RC equivalent C (Farads)"],
            bad_channel=bad_channel_mask[i],  # used by Spyglass
            rel_x=float(channel_geometry["x"][channel_num]),
            rel_y=float(channel_geometry["y"][channel_num]),
            group=electrode_group,
            location=electrodes_location,
            filtering=filtering_list[i],
            headstage_channel_number=headstage_channel_numbers[i],
            ref_elect_id=ref_elect_id[i],  # used by Spyglass
            probe_electrode=i,  # used by Spyglass
            probe_shank=shank_index,  # used by Spyglass
        )


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
    required_ephys_keys = {"openephys_folder_path", "probe"}
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
        filtering_list,
        headstage_channel_numbers,
        reference_daq_channel_indices,
        ) = get_raw_ephys_metadata_berke_probe(settings_file_path, logger)
        
        # Create electrode groups and add electrode data to the NWB file
        add_electrode_data_berke_probe(
            nwbfile=nwbfile,
            filtering_list=filtering_list,
            headstage_channel_numbers=headstage_channel_numbers,
            reference_daq_channel_indices=reference_daq_channel_indices,
            metadata=metadata,
            probe_metadata=probe_metadata,
            probe_obj=probe_obj,
            logger=logger,
            fig_dir=fig_dir,
        )

    else:
        # For Neuropixels (NOTE these are fake placeholder values):
        total_channels = 392 # No ideal acually. 384 "CH" + 8(??) "ADC" ?
        port_visits_channel_num = 384 # Also no idea. 
        raise NotImplementedError("Full processing for Neuropixels not yet implemented.")

    # Get raw ephys data
    (
        traces_as_iterator,
        channel_conversion_factor_v,
        original_timestamps,
    ) = get_raw_ephys_data(openephys_folder_path, logger)
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
