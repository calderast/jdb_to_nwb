# NOTE: We could have used NeuroConv to convert the ephys data but we want to use the Frank Lab's 
# ndx-franklab-novela extension to store the Probe information for maximal integration with Spyglass, 
# so we are doing the conversion manually using PyNWB.

import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET
from pathlib import Path
from importlib.resources import files
import pandas as pd
from hdmf.backends.hdf5 import H5DataIO
import numpy as np
from neuroconv.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
    SpikeInterfaceRecordingDataChunkIterator,
)
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor
from .plotting.plot_ephys import plot_channel_map

MICROVOLTS_PER_VOLT = 1e6
VOLTS_PER_MICROVOLT = 1 / MICROVOLTS_PER_VOLT

MIN_IMPEDANCE_OHMS = 1e5
MAX_IMPEDANCE_OHMS = 3e6

# Get the location of the resources directory when the package is installed from pypi
__location_of_this_file = Path(files(__name__))
RESOURCES_DIR = __location_of_this_file / "resources"

# If the resources directory does not exist, we are probably running the code from the source directory
if not RESOURCES_DIR.exists():
    RESOURCES_DIR = __location_of_this_file.parent.parent / "resources"

CHANNEL_MAP_PATH = RESOURCES_DIR / "channel_map.csv"
ELECTRODE_COORDS_PATH_3MM_PROBE = RESOURCES_DIR / "3mm_probe_66um_pitch_electrode_coords.csv"
ELECTRODE_COORDS_PATH_6MM_PROBE = RESOURCES_DIR / "6mm_probe_80um_pitch_electrode_coords.csv"

def add_electrode_data(
    *,
    nwbfile: NWBFile,
    filtering_list: list[str],
    metadata: dict,
    fig_dir: Path = None,
):
    """
    Add the electrode data from the impedance and channel geometry files.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file being assembled.
    filtering_list : list of str
        The filtering applied to each channel.
    metadata : dict
        Metadata dictionary.
    fig_dir : Path, optional
        The directory to save the figure. If None, the figure will not be saved.
    """

    device_args = metadata["ephys"]["device"]
    assert set(device_args.keys()) == {
        "name",
        "description",
        "manufacturer",
    }, "Device arguments do not match expected keys."
    device = nwbfile.create_device(**device_args)

    electrodes_location = metadata["ephys"].get("electrodes_location")

    # Create an NWB ElectrodeGroup for all electrodes
    # TODO: confirm that all electrodes are in the same group
    electrode_group = nwbfile.create_electrode_group(
        name="ElectrodeGroup",
        description="All electrodes",
        location=electrodes_location,
        device=device,
    )

    impedance_file_path = metadata["ephys"]["impedance_file_path"]
    electrode_data = pd.read_csv(impedance_file_path)

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
    assert electrode_data.columns.tolist() == expected_columns, (
        f"Impedance file has columns {electrode_data.columns.tolist()}, "
        f"does not match expected columns {expected_columns}"
    )
    
    # Check that the filtering list has the same length as the number of channels
    assert len(filtering_list) == len(electrode_data), (
        f"Filtering list does not have the same length ({len(filtering_list)}) "
        "as the number of channels ({len(electrode_data)})."
    )

    # Drop the first column which should be the same as the second column
    assert (
        electrode_data["Channel Number"] == electrode_data["Channel Name"]
    ).all(), "First column is not the same as the second column."
    electrode_data.drop(columns=["Channel Number"], inplace=True)

    # Get the channel map based on how the rat was plugged in (assume "chip_first" if none specified)
    plug_order = metadata["ephys"].get("plug_order", "chip_first")
    channel_map_df = pd.read_csv(CHANNEL_MAP_PATH)
    channel_map = np.array(channel_map_df[plug_order])

    # Get electrode coordinates as a (2, 256) array based on the probe name
    # The first column is the relative x coordinate, and the second column is the relative y coordinate
    probe_name = metadata["ephys"]["device"].get("name")
    if "3mm" in probe_name:
        channel_geometry = pd.read_csv(ELECTRODE_COORDS_PATH_3MM_PROBE)
    elif "6mm" in probe_name:
        channel_geometry = pd.read_csv(ELECTRODE_COORDS_PATH_6MM_PROBE)
    else:
        raise ValueError(f"Expected either '3mm' or '6mm' in device name '{probe_name}'")

    assert len(channel_geometry) == len(channel_map) == len(electrode_data), (
        "Mismatch in lengths: "
        f"channel_geometry ({len(channel_geometry)}), "
        f"channel_map ({len(channel_map)}), "
        f"electrode_data ({len(electrode_data)})"
    )

    plot_channel_map(probe_name, channel_map, channel_geometry, fig_dir=fig_dir)
    
    # Append the x and y coordinates to the impedance data using the channel map
    # For example: Under the "chip_first" channel map, the first channel has index 191 (0-indexed). 
    # The coordinates for this channel are at row index 191 in the channel_geometry dataframe 
    # (electrode coords CSV).
    # TODO: Make sure Stephanie's understanding of the channel map indexing is correct!!
    electrode_data["rel_x"] = [channel_geometry.iloc[idx]["x"] for idx in channel_map]
    electrode_data["rel_y"] = [channel_geometry.iloc[idx]["y"] for idx in channel_map]

    # Mark electrodes with impedance that is less than 0.1 MOhms or more than 3.0 MOhms
    # as bad electrodes
    electrode_data["bad_channel"] = (
        electrode_data["Impedance Magnitude at 1000 Hz (ohms)"] < MIN_IMPEDANCE_OHMS) | (
        electrode_data["Impedance Magnitude at 1000 Hz (ohms)"] > MAX_IMPEDANCE_OHMS
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
    for i, row in electrode_data.iterrows():
        nwbfile.add_electrode(
            channel_name=row["Channel Name"],
            port=row["Port"],
            enabled=bool(row["Enabled"]),
            imp=row["Impedance Magnitude at 1000 Hz (ohms)"],
            imp_phase=row["Impedance Phase at 1000 Hz (degrees)"],
            series_resistance_in_ohms=row["Series RC equivalent R (Ohms)"],
            series_capacitance_in_farads=row["Series RC equivalent C (Farads)"],
            bad_channel=row["bad_channel"],
            rel_x=float(row["rel_x"]),
            rel_y=float(row["rel_y"]),
            group=electrode_group,
            filtering=filtering_list[i],
            location=electrodes_location,
        )


def get_port_visits(folder_path: Path):
    """
    Extract port visit times from OpenEphys channel ADC1

    Args:
        folder_path: Path to the folder containing the OpenEphys binary recording. The folder
            should contain subfolders leading to a file named 'continuous.dat'

    Returns:
        list: list of port visit times in seconds
    """

    total_channels = 264  # 256 "CH" + 8 "ADC"
    port_visits_channel_num = 256 # Port visits are recorded on ADC1, aka channel 256 (zero-indexed)
    pulse_high_threshold = 10_000
    openephys_fs = 30_000
    downsampled_fs = 1000
    # 1000ish is a reasonable downsample frequency, because it's low enough to speeds things up
    # (this takes ~2mins on my machine) but high enough that we definitely don't miss any port visit pulses 
    # For reference, a typical port visit keeps the channel high for ~10ms, so 10 samples at 1000 Hz

    # Memory-map the large .dat file to avoid loading everything into memory. Reshape to (samples, channels)
    file_path = folder_path + "/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat"
    data_for_all_channels = np.memmap(file_path, dtype='int16',mode='c').reshape(-1, total_channels) 

    # Extract the channel that records port visits and downsample so the data isn't massive
    visits_channel_data = data_for_all_channels[:, port_visits_channel_num]
    visits_channel_data = visits_channel_data[::int(openephys_fs / downsampled_fs)]

    # Find indices where the visits channel is high (aka port visit times)
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

    # Only keep pulses at least 5ms long just in case (typical port visit keeps the channel high for ~10ms)
    min_pulse_duration = int(downsampled_fs * 0.005)
    pulse_starts = pulse_starts[pulse_durations > min_pulse_duration]

    # Align to first pulse (which marks photometry(?) start time) and convert to seconds
    # NOTE: The duration of the first pulse is longer than a normal pulse (~500ms instead of ~10ms)
    # should we add a check for this to ensure we have identified the correct "start" pulse?
    # Maybe just log its duration when we add logging?
    port_visits = pulse_starts[1:] - pulse_starts[0]
    port_visits = [float(visit / downsampled_fs) for visit in port_visits]
    return port_visits


def get_raw_ephys_data(
    folder_path: Path,
) -> tuple[SpikeInterfaceRecordingDataChunkIterator, float, np.ndarray, list[str]]:
    """
    Get the raw ephys data from the OpenEphys binary recording.

    Args:
        folder_path: Path to the folder containing the OpenEphys binary recording. The folder
            should have the date in the name and contain a file called "settings.xml".

    Returns:
        traces_as_iterator: SpikeInterfaceRecordingDataChunkIterator, to be used as the
            data argument in pynwb.ecephys.ElectricalSeries.
        channel_conversion_factor: float, the conversion factor from the raw data to volts.
        original_timestamps: np.ndarray, that could be used as the timestamps argument in
            pynwb.ecephys.ElectricalSeries or may need to be time aligned with the other
            data streams in the NWB file.
        filtering_list: list of str, the filtering applied to each channel
    """
    # Create a SpikeInterface recording extractor for the OpenEphys binary data
    recording = OpenEphysBinaryRecordingExtractor(folder_path=folder_path)

    # Select only the channels that start with "CH"
    # Ignore the "ADC" channels
    channel_ids_to_convert = [ch for ch in recording.channel_ids if ch.startswith("CH")]
    recording_sliced = recording.select_channels(channel_ids=channel_ids_to_convert)

    # Get the channel conversion factor
    channel_conversion_factors_uv = recording_sliced.get_channel_gains()
    # Warn if the channel conversion factors are not the same for all channels
    if not all(channel_conversion_factors_uv == channel_conversion_factors_uv[0]):
        warnings.warn(
            "The channel conversion factors are not the same for all channels. "
            "This is unexpected and may indicate a problem with the conversion factors."
        )
    channel_conversion_factor_v = channel_conversion_factors_uv[0] * VOLTS_PER_MICROVOLT

    # NOTE channel offsets should be 0 for all channels in openephys data
    channel_conversion_offsets = recording_sliced.get_channel_offsets()
    assert all(channel_conversion_offsets == 0), "Channel conversion offsets are not all 0."

    # Get the original timestamps (in seconds)
    original_timestamps = recording_sliced.get_times()

    # TODO when we add logging, log sample frequency
    # recording.get_sampling_frequency()

    # Create a SpikeInterfaceRecordingDataChunkIterator using all default buffering and
    # chunking options. This will be passed to the pynwb.ecephys.ElectricalSeries
    # constructor.
    traces_as_iterator = SpikeInterfaceRecordingDataChunkIterator(recording=recording_sliced)

    # Read the openephys settings.xml file to get the mapping of channel number to channel name
    # <PROCESSOR name="Sources/Rhythm FPGA" ...>
    #   <CHANNEL_INFO>
    #     <CHANNEL name="CH1" number="0" gain="0.19499999284744263"/>
    #     ...
    #   </CHANNEL_INFO>
    #   ...
    # </PROCESSOR>
    settings_file_path = Path(folder_path) / "settings.xml"
    settings_tree = ET.parse(settings_file_path)
    settings_root = settings_tree.getroot()
    rhfp_processor = settings_root.find(".//PROCESSOR[@name='Sources/Rhythm FPGA']")
    if rhfp_processor is None:
        raise ValueError("Could not find the Rhythm FPGA processor in the settings.xml file.")
    channel_info = rhfp_processor.find("CHANNEL_INFO")
    if channel_info is None:
        raise ValueError("Could not find the CHANNEL_INFO node in the settings.xml file.")
    channel_number_to_channel_name = {
        channel.attrib["number"]: channel.attrib["name"] for channel in channel_info.findall("CHANNEL")
    }

    # Read the settings.xml file to get the filtering applied to each channel - 
    # map channel number to filter description
    # <PROCESSOR name="Filters/Bandpass Filter" ...>
    #   <CHANNEL name="0" number="0">
    #     <SELECTIONSTATE param="1" record="0" audio="0"/>
    #     <PARAMETERS highcut="6000" lowcut="1" shouldFilter="1"/>
    #   </CHANNEL>
    #   ...
    # </PROCESSOR>
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
        raise ValueError("No bandpass filter found in the settings.xml file.")

    # Check that the channel numbers in filtering go from "0" to "N-1"
    # where N is the number of channels
    assert all(
        int(ch_num) == i for i, ch_num in enumerate(filtering.keys())
    ), "Channel numbers in filtering do not go from 0 to N-1."
    filtering_list = list(filtering.values())

    # Warn if the channel filtering is not the same for all channels
    if not all(f == filtering_list[0] for f in filtering_list):
        warnings.warn(
            "WARNING: The channel filtering is not the same for all channels. "
            "This is unexpected and may indicate a problem with the filtering settings."
        )

    # TODO: save reference information from settings.xml

    # TODO: save settings.xml as an associated file using the ndx-franklab-novela extension

    return (
        traces_as_iterator,
        channel_conversion_factor_v,
        original_timestamps,
        filtering_list,
    )


def add_raw_ephys(
    *,
    nwbfile: NWBFile,
    metadata: dict = None,
    fig_dir: Path = None,
) -> None:
    """Add the raw ephys data to a NWB file. Must be called after add_electrode_groups

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file being assembled.
    metadata : dict, optional
        Metadata dictionary,by default None
    fig_dir: Path, optional
        The directory to save the figure. If None, the figure will not be saved.
    """

    if "ephys" not in metadata:
        print("No ephys metadata found for this session. Skipping ephys conversion.")
        return {}

    # If we do have "ephys" in metadata, check for the required keys
    required_ephys_keys = {"openephys_folder_path", "device", "impedance_file_path"}
    missing_keys = required_ephys_keys - metadata["ephys"].keys()
    if missing_keys:
        print(
            "The required ephys subfields do not exist in the metadata dictionary.\n"
            "Remove the 'ephys' field from metadata if you do not have ephys data "
            f"for this session, \nor specify the following missing subfields:{missing_keys}"
        )
        return {}

    print("Adding raw ephys...")
    openephys_folder_path = metadata["ephys"]["openephys_folder_path"]

    # Get Open Ephys start time as datetime object based on the time specified in the path
    datetime_str = openephys_folder_path.split('/')[-1] # The path ends with the date and time
    open_ephys_start = datetime.strptime(datetime_str, "%Y-%m-%d_%H-%M-%S")
    open_ephys_start = open_ephys_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))

    (
        traces_as_iterator,
        channel_conversion_factor_v,
        original_timestamps,
        filtering_list,
    ) = get_raw_ephys_data(openephys_folder_path)
    num_samples, num_channels = traces_as_iterator.maxshape
    
    # Get port visits recorded by Open Ephys for timestamp alignment
    ephys_visit_times = get_port_visits(openephys_folder_path)
    print(f"Open Ephys recorded {len(ephys_visit_times)} port visits.")

    # Create electrode groups and add electrode data to the NWB file
    add_electrode_data(nwbfile=nwbfile, filtering_list=filtering_list, metadata=metadata, fig_dir=fig_dir)

    # Check that the number of electrodes in the NWB file is the same as the number of channels in traces_as_iterator
    assert (len(nwbfile.electrodes) == num_channels), (
        f"Number of electrodes in NWB file ({len(nwbfile.electrodes)}) does not match number of channels "
        f"in traces_as_iterator ({num_channels})."
    )

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

    # If we have visit times from photometry, align timestamps to that
    # Based on: https://github.com/catalystneuro/neuroconv/blob/8d274f434c152d34ea0bcc5d120765639c6771a7/src/
    # neuroconv/datainterfaces/text/timeintervalsinterface.py#L192
    photometry_visit_times = metadata.get("photometry_visit_times")
    if photometry_visit_times is not None:
        # Make sure we have the same number of ephys and photometry visit times
        assert len(ephys_visit_times) == len(photometry_visit_times), (
            f"Expected the same number of port visits recorded by ephys and photometry! \n"
            f"Got {len(ephys_visit_times)} ephys visits, but {len(photometry_visit_times)} photometry visits!!!"
        )
        # Align ephys timestamps to photometry via interpolation
        ephys_timestamps = np.interp(
            x=original_timestamps,
            xp=ephys_visit_times,
            fp=photometry_visit_times,
            left=2 * photometry_visit_times[0] - photometry_visit_times[1],  # For values outside bounds,
            right=2 * photometry_visit_times[-1] - photometry_visit_times[-2],  # use the most recent diff to regress
        )
    else:
        # If we don't have photometry, keep the original timestamps
        ephys_timestamps = original_timestamps

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
    nwbfile.add_acquisition(eseries)

    return {"ephys_start": open_ephys_start, "port_visits": ephys_visit_times}
