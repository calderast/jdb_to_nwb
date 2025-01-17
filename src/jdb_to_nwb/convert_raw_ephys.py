# NOTE: We could have used NeuroConv to convert the ephys data but we want to use the Frank Lab's 
# ndx-franklab-novela extension to store the Probe information for maximal integration with Spyglass, 
# so we are doing the conversion manually using PyNWB.

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from hdmf.backends.hdf5 import H5DataIO
import numpy as np
from neuroconv.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
    SpikeInterfaceRecordingDataChunkIterator,
)
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor

MICROVOLTS_PER_VOLT = 1e6
VOLTS_PER_MICROVOLT = 1 / MICROVOLTS_PER_VOLT

MIN_IMPEDANCE_OHMS = 1e5
MAX_IMPEDANCE_OHMS = 3e6


def get_channel_map(plug_order : str = "chip_first"):
    """
    Return the channel map for the probe electrodes based on how the rat was plugged in.
    
    Optional argument plug_order can be "chip_first" (default) or "cable_first".
    """
    
    if plug_order == "cable_first":
        # Cable first, only if plugged in wrong by accident (eg T1 / IM1520)
        channel_map = np.array([
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 64, 
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 
        122, 123, 124, 125, 126, 127, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 
        211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 
        236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 151, 152, 153, 154, 155, 
        156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 
        181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 
        136, 135, 134, 133, 132, 131, 130, 129, 128])
    else:
        # Assume chip first if not otherwise specified, this is the regular case for all other rats (eg wt43 / IM1586)
        channel_map = np.array([
        191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 128,
        129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
        153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
        248, 249, 250, 251, 252, 253, 254, 255, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
        104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
        47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    return channel_map


def get_electrode_coords():
    """Get the relative x and y coordinates of the electrodes."""
    # TODO: These are different for each probe! 
    # Add argument "probe" and then get coords for that probe specifically
    # Likely want to set up coordinate files for each probe and then just read from those
    # Maybe can store those in a berke lab metadata file in this repo somewhere
    
    # Define the base arrays
    array1 = np.arange(240, 29, -30) # [240, 210, 180, 150, 120, 90, 60, 30]
    array2 = array1 - 15 # [225, 195, 165, 135, 105, 75, 45, 15]

    # Repeat each array 16 times to get 2 arrays of shape (16, 8)
    reshaped_array1 = np.tile(array1, (16, 1))
    reshaped_array2 = np.tile(array2, (16, 1))

    # Create y coordinates by interdigitating both sequences
    # (fill even rows with reshaped_array1 and odd rows with reshaped_array2)
    y_coords = np.zeros((32, 8), dtype=int)
    y_coords[::2], y_coords[1::2] = reshaped_array1, reshaped_array2
    y_coords = y_coords.flatten()

    # Create x coordinates from 100 to 3200 with steps of 100, with each number repeated 8 times
    x_coords = np.repeat(np.arange(100, 3201, 100), 8)

    # Stack x and y coordinates as columns to form a 2D array
    coords = np.column_stack((x_coords, y_coords))
    return coords


def add_electrode_data(
    *,
    nwbfile: NWBFile,
    filtering_list: list[str],
    metadata: dict,
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
    """

    device_args = metadata["ephys"]["device"]
    assert set(device_args.keys()) == {
        "name",
        "description",
        "manufacturer",
    }, "Device arguments do not match expected keys."
    device = nwbfile.create_device(**device_args)

    electrodes_location = metadata["ephys"]["electrodes_location"]

    # Create an NWB ElectrodeGroup for all electrodes
    # TODO: confirm that all electrodes are in the same group
    electrode_group = nwbfile.create_electrode_group(
        name="ElectrodeGroup",
        description="All electrodes",
        location=electrodes_location,
        device=device,
    )

    impedance_file_path = metadata["ephys"]["impedance_file_path"]
    # channel_geometry_file_path = metadata["ephys"]["channel_geometry_file_path"]

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
    assert electrode_data.columns.tolist() == expected_columns, "Impedance file does not have the expected columns."

    # Check that the filtering list has the same length as the number of channels
    assert len(filtering_list) == len(
        electrode_data
    ), "Filtering list does not have the same length as the number of channels."

    # Drop the first column which should be the same as the second column
    assert (
        electrode_data["Channel Number"] == electrode_data["Channel Name"]
    ).all(), "First column is not the same as the second column."
    electrode_data.drop(columns=["Channel Number"], inplace=True)

    # Get electrode coordinates as a (2, 256) array
    # The first column is the relative x coordinate, and the second column is the relative y coordinate
    channel_geometry = get_electrode_coords() # formerly pd.read_csv(channel_geometry_file_path, header=None)
    
    # Get the channel map based on how the rat was plugged in
    plug_order = metadata["ephys"].get("plug_order", "chip_first")
    channel_map = get_channel_map(plug_order=plug_order)
    
    assert len(channel_geometry) == len(channel_map) == len(electrode_data), (
    "Mismatch in lengths: "
    f"channel_geometry ({len(channel_geometry)}), "
    f"channel_map ({len(channel_map)}), "
    f"electrode_data ({len(electrode_data)})"
    )
    
    # Append the x and y coordinates to the impedance data using the channel map
    
    # TODO: Make sure Stephanie's understanding of the channel map indexing is correct!!
    electrode_data["rel_x"] = [channel_geometry[idx][0] for idx in channel_map] # formerly channel_geometry.iloc[:, 0]
    electrode_data["rel_y"] = [channel_geometry[idx][1] for idx in channel_map] # formerly channel_geometry.iloc[:, 1]

    # Mark electrodes with impedance that is less than 0.1 MOhms or more than 3.0 MOhms
    # as bad electrodes
    electrode_data["bad_channel"] = (electrode_data["Impedance Magnitude at 1000 Hz (ohms)"] < MIN_IMPEDANCE_OHMS) | (
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

    # Get the original timestamps
    original_timestamps = recording_sliced.get_times()

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

    # Read the settings.xml file to get the filtering applied to each channel - map channel number to filter description
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
                raise ValueError(f"Channel {channel.attrib['number']}: No filtering")
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

    # TODO: handle channel map

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
) -> None:
    """Add the raw ephys data to a NWB file. Must be called after add_electrode_groups

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file being assembled.
    metadata : dict, optional
        Metadata dictionary,by default None
    """
    print("Adding raw ephys...")
    openephys_folder_path = metadata["ephys"]["openephys_folder_path"]
    (
        traces_as_iterator,
        channel_conversion_factor_v,
        original_timestamps,
        filtering_list,
    ) = get_raw_ephys_data(openephys_folder_path)
    num_samples, num_channels = traces_as_iterator.maxshape

    # Create electrode groups and add electrode data to the NWB file
    add_electrode_data(nwbfile=nwbfile, filtering_list=filtering_list, metadata=metadata)

    # Check that the number of electrodes in the NWB file is the same as the number of channels in traces_as_iterator
    assert (
        len(nwbfile.electrodes) == num_channels
    ), "Number of electrodes in NWB file does not match number of channels in traces_as_iterator"

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

    # Create the ElectricalSeries
    # For now, we do not chunk or compress the timestamps, which are relatively small
    # TODO: replace the timestamps with timestamps aligned with photometry data if photometry
    # data was also collected during the same recording session
    eseries = ElectricalSeries(
        name="ElectricalSeries",
        description=(
            "Raw ephys data from OpenEphys recording (multiply by conversion factor to get data in volts). "
            "Timestamps are the original timestamps from the OpenEphys recording."
        ),
        data=data_data_io,
        timestamps=original_timestamps,
        electrodes=electrode_table_region,
        conversion=channel_conversion_factor_v,
    )

    # Add the ElectricalSeries to the NWBFile
    nwbfile.add_acquisition(eseries)
