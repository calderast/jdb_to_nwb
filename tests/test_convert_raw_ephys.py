from datetime import datetime

import numpy as np
from dateutil import tz
from pynwb import NWBFile

from jdb_to_nwb.convert_raw_ephys import add_electrode_data, add_raw_ephys, get_raw_ephys_data


def test_add_electrode_data():
    """
    Test the add_electrode_data function.

    Files `tests/test_data/processed_ephys/impedance.csv` and `tests/test_data/processed_ephys/geom.csv` must be
    created first by running `python tests/test_data/create_raw_ephys_test_data.py`.
    """
    # Create a test metadata dictionary
    metadata = {}
    metadata["impedance_file_path"] = "tests/test_data/processed_ephys/impedance.csv"
    metadata["channel_geometry_file_path"] = "tests/test_data/processed_ephys/geom.csv"
    metadata["electrodes_location"] = "Nucleus Accumbens core"
    metadata["device"] = {
        "name": "Probe",
        "description": "Berke Lab Probe",
        "manufacturer": "My Manufacturer",
    }

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Create a test filtering list
    filtering_list = ["Bandpass Filter"] * 4

    # Add electrode data to the NWBFile
    add_electrode_data(nwbfile=nwbfile, filtering_list=filtering_list, metadata=metadata)

    # Test that the nwbfile has the expected device
    assert "Probe" in nwbfile.devices
    device = nwbfile.devices["Probe"]
    assert device is not None
    assert device.description == "Berke Lab Probe"
    assert device.manufacturer == "My Manufacturer"

    # Test that the nwbfile has the expected electrode group
    assert len(nwbfile.electrode_groups) == 1
    assert "ElectrodeGroup" in nwbfile.electrode_groups
    eg = nwbfile.electrode_groups["ElectrodeGroup"]
    assert eg is not None
    assert eg.description == "All electrodes"
    assert eg.location == metadata["electrodes_location"]
    assert eg.device is device

    # Test that the nwbfile has the expected electrodes after filtering
    assert len(nwbfile.electrodes) == 4
    assert nwbfile.electrodes.channel_name.data[:] == ["B-000", "B-001", "B-002", "B-003"]
    assert nwbfile.electrodes.port.data[:] == ["Port B", "Port B", "Port B", "Port B"]
    assert nwbfile.electrodes.enabled.data[:] == [True, True, True, True]
    assert nwbfile.electrodes.imp.data[:] == [9999, 1e5, 3e6, 4e6]
    assert nwbfile.electrodes.imp_phase.data[:] == [-1, -2, -3, -4]
    assert nwbfile.electrodes.series_resistance_in_ohms.data[:] == [
        0.1,
        0.15,
        0.25,
        0.3,
    ]
    assert nwbfile.electrodes.series_capacitance_in_farads.data[:] == [
        0.0001,
        0.00015,
        0.00025,
        0.0003,
    ]
    assert nwbfile.electrodes.bad_channel.data[:] == [True, False, False, True]
    assert nwbfile.electrodes.rel_x.data[:] == [1056, 1056, 1056, 1056]
    assert nwbfile.electrodes.rel_y.data[:] == [-14, 16, 46, 76]
    assert nwbfile.electrodes.group.data[:] == [eg] * 4
    assert nwbfile.electrodes.group_name.data[:] == ["ElectrodeGroup"] * 4
    assert nwbfile.electrodes.filtering.data[:] == filtering_list
    assert nwbfile.electrodes.location.data[:] == ["Nucleus Accumbens core"] * 4


def test_get_raw_ephys_data():
    """
    Test the get_raw_ephys_data function.

    File `tests/test_data/raw_ephys/2022-07-25_15-30-00` must be created first by running
    `python tests/test_data/create_raw_ephys_test_data.py`.
    """
    folder_path = "tests/test_data/raw_ephys/2022-07-25_15-30-00"
    traces_as_iterator, channel_conversion_factor, original_timestamps, filtering_list = get_raw_ephys_data(folder_path)
    assert traces_as_iterator.maxshape == (30_000, 4)
    np.testing.assert_allclose(channel_conversion_factor, [0.19499999284744263 * 1e-6] * 4)
    assert filtering_list == ["2nd-order Butterworth filter with highcut=6000 Hz and lowcut=1 Hz"] * 4
    assert len(original_timestamps) == 30_000


def test_add_raw_ephys():
    """
    Test the add_raw_ephys function.

    Files `tests/test_data/processed_ephys/impedance.csv` and `tests/test_data/processed_ephys/geom.csv` must be
    created first by running `python tests/test_data/create_raw_ephys_test_data.py`.
    """
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    metadata = {}
    metadata["openephys_folder_path"] = "tests/test_data/raw_ephys/2022-07-25_15-30-00"
    metadata["impedance_file_path"] = "tests/test_data/processed_ephys/impedance.csv"
    metadata["channel_geometry_file_path"] = "tests/test_data/processed_ephys/geom.csv"
    metadata["electrodes_location"] = "Nucleus Accumbens core"
    metadata["device"] = {
        "name": "Probe",
        "description": "Berke Lab Probe",
        "manufacturer": "My Manufacturer",
    }

    add_raw_ephys(nwbfile=nwbfile, metadata=metadata)

    assert len(nwbfile.electrodes) == 4
    assert len(nwbfile.electrode_groups) == 1
    assert len(nwbfile.acquisition) == 1
    assert "ElectricalSeries" in nwbfile.acquisition
    es = nwbfile.acquisition["ElectricalSeries"]
    assert es.description == (
        "Raw ephys data from OpenEphys recording (multiply by conversion factor to get data in volts). "
        "Timestamps are the original timestamps from the OpenEphys recording."
    )
    assert es.data.maxshape == (30_000, 4)
    assert es.data.dtype == np.int16
    assert es.electrodes.data == [0, 1, 2, 3]
    assert es.timestamps.shape == (30_000,)
    assert es.conversion == 0.19499999284744263 * 1e-6
