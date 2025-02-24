from datetime import datetime

import numpy as np
from dateutil import tz
from zoneinfo import ZoneInfo
from pynwb import NWBFile

from jdb_to_nwb.convert_raw_ephys import add_electrode_data, add_raw_ephys, get_raw_ephys_data


def test_add_electrode_data():
    """
    Test the add_electrode_data function.

    File `tests/test_data/processed_ephys/impedance.csv` must be created first 
    by running `python tests/test_data/create_raw_ephys_test_data.py`.
    """
    # Create a test metadata dictionary
    metadata = {}
    metadata["ephys"] = {}
    metadata["ephys"]["impedance_file_path"] = "tests/test_data/processed_ephys/impedance.csv"
    metadata["ephys"]["electrodes_location"] = "Hippocampus CA1"
    metadata["ephys"]["device"] = {
        "name": "3mm Probe",
        "description": "Test Probe",
        "manufacturer": "Test Manufacturer",
    }

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Create a test filtering list
    filtering_list = ["Bandpass Filter"] * 256

    # Add electrode data to the NWBFile
    add_electrode_data(nwbfile=nwbfile, filtering_list=filtering_list, metadata=metadata)

    # Test that the nwbfile has the expected device
    assert "3mm Probe" in nwbfile.devices
    device = nwbfile.devices["3mm Probe"]
    assert device is not None
    assert device.description == "Test Probe"
    assert device.manufacturer == "Test Manufacturer"

    # Test that the nwbfile has the expected electrode group
    assert len(nwbfile.electrode_groups) == 1
    assert "ElectrodeGroup" in nwbfile.electrode_groups
    eg = nwbfile.electrode_groups["ElectrodeGroup"]
    assert eg is not None
    assert eg.description == "All electrodes"
    assert eg.location == "Hippocampus CA1"
    assert eg.device is device

    # Test that the nwbfile has the expected electrodes after filtering
    assert len(nwbfile.electrodes) == 256
    expected_B_channels = [f"B-{i:03d}" for i in range(128)]
    expected_C_channels = [f"C-{i:03d}" for i in range(128)]
    expected_channels = expected_B_channels + expected_C_channels
    assert nwbfile.electrodes.channel_name.data[:] == expected_channels
    assert nwbfile.electrodes.port.data[:] == ["Port B"] * 128 + ["Port C"] * 128
    assert nwbfile.electrodes.enabled.data[:] == [True] * 256

    # Check first electrode data
    assert nwbfile.electrodes.imp.data[0] == 2.24e+06
    assert nwbfile.electrodes.imp_phase.data[0] == -43
    assert nwbfile.electrodes.series_resistance_in_ohms.data[0] == 1.63e+06
    assert nwbfile.electrodes.series_capacitance_in_farads.data[0] == 1.04e-10
    assert not nwbfile.electrodes.bad_channel.data[0]
    assert nwbfile.electrodes.rel_x.data[0] == 66.0
    assert nwbfile.electrodes.rel_y.data[0] == 211.0

    # Check last electrode data
    assert nwbfile.electrodes.imp.data[-1] == 6.45e+06
    assert nwbfile.electrodes.imp_phase.data[-1] == -69
    assert nwbfile.electrodes.series_resistance_in_ohms.data[-1] == 2.31e+06
    assert nwbfile.electrodes.series_capacitance_in_farads.data[-1] == 2.64e-11
    assert nwbfile.electrodes.bad_channel.data[-1]
    assert nwbfile.electrodes.rel_x.data[-1] == 2112.0
    assert nwbfile.electrodes.rel_y.data[-1] == -14.0

    assert nwbfile.electrodes.group.data[:] == [eg] * 256
    assert nwbfile.electrodes.group_name.data[:] == ["ElectrodeGroup"] * 256
    assert nwbfile.electrodes.filtering.data[:] == filtering_list
    assert nwbfile.electrodes.location.data[:] == ["Hippocampus CA1"] * 256


def test_get_raw_ephys_data():
    """
    Test the get_raw_ephys_data function.

    File `tests/test_data/raw_ephys/2022-07-25_15-30-00` must be created first by running
    `python tests/test_data/create_raw_ephys_test_data.py`.
    """
    folder_path = "tests/test_data/raw_ephys/2022-07-25_15-30-00"
    traces_as_iterator, channel_conversion_factor, original_timestamps, filtering_list = get_raw_ephys_data(folder_path)
    assert traces_as_iterator.maxshape == (3_000, 256)
    np.testing.assert_allclose(channel_conversion_factor, [0.19499999284744263 * 1e-6] * 256)
    assert filtering_list == ["2nd-order Butterworth filter with highcut=6000 Hz and lowcut=1 Hz"] * 256
    assert len(original_timestamps) == 3_000


def test_add_raw_ephys():
    """
    Test the add_raw_ephys function.

    Test raw ephys data must be created first by running `python tests/test_data/create_raw_ephys_test_data.py`.
    """
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    metadata = {}
    metadata["ephys"] = {}
    metadata["ephys"]["openephys_folder_path"] = "tests/test_data/raw_ephys/2022-07-25_15-30-00"
    metadata["ephys"]["impedance_file_path"] = "tests/test_data/processed_ephys/impedance.csv"
    metadata["ephys"]["electrodes_location"] = "Hippocampus CA1"
    metadata["ephys"]["device"] = {
        "name": "3mm Probe",
        "description": "Test Probe",
        "manufacturer": "Test Manufacturer",
    }

    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata)

    assert len(nwbfile.electrodes) == 256
    assert len(nwbfile.electrode_groups) == 1
    assert len(nwbfile.acquisition) == 1
    assert "ElectricalSeries" in nwbfile.acquisition
    es = nwbfile.acquisition["ElectricalSeries"]
    assert es.description == (
        "Raw ephys data from OpenEphys recording (multiply by conversion factor to get data in volts)."
    )
    assert es.data.maxshape == (3_000, 256)
    assert es.data.dtype == np.int16
    assert es.electrodes.data == list(range(256))
    assert es.timestamps.shape == (3_000,)
    assert es.conversion == 0.19499999284744263 * 1e-6
    
    expected_ephys_start = datetime.strptime("2022-07-25_15-30-00", "%Y-%m-%d_%H-%M-%S")
    expected_ephys_start = expected_ephys_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    assert ephys_data_dict.get("ephys_start") == expected_ephys_start


def test_add_ephys_with_incomplete_metadata(capsys):
    """
    Test that the add_raw_ephys function responds appropriately to missing or incomplete metadata.
    
    If no 'ephys' key is in the metadata dictionary, it should print that we are skipping 
    ephys conversion and move on without raising any errors.
    
    If there is a 'ephys' key in the metadata dict but the required paths to ephys data
    are not present, raise a ValueError telling the user which keys must be present in the dict.
    """

    # Create a test metadata dictionary with no ephys key
    metadata = {}

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # If we call the add_raw_ephys function with no 'ephys' key in metadata,
    # It should print that we are skipping ephys conversion
    # This should not raise any errors, as omitting the 'ephys' key is a 
    # valid way to specify that we have no ephys data for this session.
    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata)
    captured = capsys.readouterr() # capture stdout

    # Check that the correct message was printed to stdout
    assert "No ephys metadata found for this session. Skipping ephys conversion." in captured.out
    assert ephys_data_dict == {}

    # Create a test metadata dictionary with an ephys field but no ephys data
    metadata["ephys"] = {}

    # Check that add_raw_ephys raises a ValueError about missing fields in the metadata dictionary
    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata)
    captured = capsys.readouterr()
    assert ephys_data_dict == {}
    assert "The required ephys subfields do not exist in the metadata dictionary" in captured.out