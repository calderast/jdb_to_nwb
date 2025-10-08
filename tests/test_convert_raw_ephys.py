from datetime import datetime
from pathlib import Path
import numpy as np
from dateutil import tz
from zoneinfo import ZoneInfo
from pynwb import NWBFile
import pytest
import re

from jdb_to_nwb.convert_raw_ephys import (
    add_raw_ephys, 
    get_raw_ephys_data,
    add_probe_info,
#    get_electrode_info, # TODO add test for this
    read_open_ephys_settings_xml,
    add_electrode_data_berke_probe,
)


def test_add_probe_info(dummy_logger):
    """
    Test the add_probe_info function.
    """
    # Create a test metadata dictionary
    metadata = {}
    metadata["ephys"] = {}
    metadata["ephys"]["probe"] = ["256-ch Silicon Probe, 3mm length, 66um pitch"]

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Add probe to the nwbfile
    probe_metadata, probe_obj = add_probe_info(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Test that the nwbfile has the expected probe info under 'devices'
    assert "256-ch Silicon Probe, 3mm length, 66um pitch" in nwbfile.devices
    probe = nwbfile.devices["256-ch Silicon Probe, 3mm length, 66um pitch"]
    assert probe is not None
    assert probe.probe_description.startswith("32 shanks, 8 electrodes per shank. Each shank is 3mm long.")
    assert probe.manufacturer == "Daniel Egert, Berke Lab"
    assert probe.contact_side_numbering
    assert probe.contact_size == 15
    assert probe.units == "um"

    # Test that probe_metadata includes the probe info
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
    assert probe_metadata["name"] == "256-ch Silicon Probe, 3mm length, 66um pitch"
    assert probe_metadata["description"].startswith("32 shanks, 8 electrodes per shank. Each shank is 3mm long.")
    assert probe_metadata["manufacturer"] == "Daniel Egert, Berke Lab"
    assert probe_metadata["contact_side_numbering"]
    assert probe_metadata["contact_size"] == 15
    assert probe_metadata["units"] == "um"

    # Test that the returned probe_obj is the probe from the nwb
    probe = nwbfile.devices["256-ch Silicon Probe, 3mm length, 66um pitch"]
    assert probe_obj == probe


def test_add_electrode_data_berke_probe(dummy_logger):
    """
    Test the add_electrode_data_berke_probe function.
    Note that this depends on add_probe_info
    """
    # Create a test metadata dictionary
    metadata = {}
    metadata["ephys"] = {}
    metadata["ephys"]["impedance_file_path"] = "tests/test_data/processed_ephys/impedance.csv"
    metadata["ephys"]["electrodes_location"] = "Hippocampus CA1"
    metadata["ephys"]["targeted_x"] = 4.5  # AP in mm
    metadata["ephys"]["targeted_y"] = 2.2  # ML in mm
    metadata["ephys"]["targeted_z"] = 2.0  # DV in mm
    metadata["ephys"]["probe"] = ["256-ch Silicon Probe, 3mm length, 66um pitch"]
    metadata["ephys"]["plug_order"] = "chip_first"

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    filtering_info = "Filter with highcut=7603.76512183337 Hz, lowcut=2.495988241877236 Hz"
    expected_intan_channel_numbers = np.array([
        191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,175,174,173,172,171,170,169,128,129,130,
        131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,
        157,158,159,160,161,162,163,164,165,166,167,168,192,193,194,195,196,197,198,199,200,201,202,203,204,205,
        206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,
        232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,64,65,66,
        67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,
        101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,
        127,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,
        56,57,58,59,60,61,62,63,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
    ])
    
    # Set up expected channel names from the impedance file based on the ordering of Intan channel numbers
    # Channels < 128 are B-###, channels >= 128 are C-###
    expected_channel_names = [
        f"{'B' if n < 128 else 'C'}-{n if n < 128 else n - 128:03d}" for n in expected_intan_channel_numbers
    ]
    expected_port_labels = ["Port B" if n < 128 else "Port C" for n in expected_intan_channel_numbers]

    # First add the probe to the nwbfile
    probe_metadata, probe_obj = add_probe_info(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Add electrode data to the NWBFile
    add_electrode_data_berke_probe(
        nwbfile=nwbfile,
        filtering_info=filtering_info,
        metadata=metadata,
        probe_metadata=probe_metadata,
        probe_obj=probe_obj,
        logger=dummy_logger,
    )

    # Test that the nwbfile has the expected probe info under 'devices'
    assert "256-ch Silicon Probe, 3mm length, 66um pitch" in nwbfile.devices
    probe = nwbfile.devices["256-ch Silicon Probe, 3mm length, 66um pitch"]
    assert probe is not None
    assert probe.probe_description.startswith("32 shanks, 8 electrodes per shank. Each shank is 3mm long.")
    assert probe.manufacturer == "Daniel Egert, Berke Lab"
    assert probe.contact_side_numbering
    assert probe.contact_size == 15
    assert probe.units == "um"

    # Test that the nwbfile has the expected 32 electrode groups (one per shank)
    assert len(nwbfile.electrode_groups) == 32
    found_shanks = set()

    # Check data for each electrode group
    for name, egroup in nwbfile.electrode_groups.items():
        assert egroup.location == "Hippocampus CA1"
        assert egroup.targeted_x == 4.5
        assert egroup.targeted_y == 2.2
        assert egroup.targeted_z == 2.0
        assert egroup.device is probe

        # Electrode group description should be "Electrodes on shank {shank_index}"
        match = re.match(r"Electrodes on shank (\d+)", egroup.description)
        assert match, f"Unexpected description in group {name}: {egroup.description}"
        # Add the shank to our set of found shanks
        shank = int(match.group(1))
        found_shanks.add(shank)

    # Check that we found an electrode group for each of the 32 shanks
    assert found_shanks == set(range(32))

    # Test that the nwbfile has the expected electrodes after filtering
    assert len(nwbfile.electrodes) == 256
    expected_electrode_names = [
        f"S{shank:02d}E{electrode:01d}"
        for shank in range(1, 33)
        for electrode in range(1, 9)
    ]
    # Strip any suffix like "/SCREW#" from actual electrode names
    electrode_names = [name.split('/')[0] for name in nwbfile.electrodes.electrode_name.data[:]]
    assert electrode_names == expected_electrode_names
    assert nwbfile.electrodes.imp_file_channel_name.data[:] == expected_channel_names
    assert nwbfile.electrodes.port.data[:] == expected_port_labels
    assert (nwbfile.electrodes.intan_channel_number.data[:] == expected_intan_channel_numbers).all()

    # Check first electrode data (S01E1)
    assert nwbfile.electrodes.imp.data[0] == 6.50e06
    assert nwbfile.electrodes.imp_phase.data[0] == -68
    assert nwbfile.electrodes.series_resistance_in_ohms.data[0] == 2.44e06
    assert nwbfile.electrodes.series_capacitance_in_farads.data[0] == 2.64e-11
    assert nwbfile.electrodes.bad_channel.data[0]
    assert nwbfile.electrodes.rel_x.data[0] == 66.0
    assert nwbfile.electrodes.rel_y.data[0] == 255.0

    # Check last electrode data (S32E8)
    assert nwbfile.electrodes.imp.data[-1] == 2.24e06
    assert nwbfile.electrodes.imp_phase.data[-1] == -43
    assert nwbfile.electrodes.series_resistance_in_ohms.data[-1] == 1.63e06
    assert nwbfile.electrodes.series_capacitance_in_farads.data[-1] == 1.04e-10
    assert not nwbfile.electrodes.bad_channel.data[-1]
    assert nwbfile.electrodes.rel_x.data[-1] == 2112.0
    assert nwbfile.electrodes.rel_y.data[-1] == 30.0

    # Expected electrode group is 0-31 (32 shanks), each repeated 8 times (8 electrodes per shank)
    expected_names = [str(i) for i in range(32) for _ in range(8)]
    assert nwbfile.electrodes.group_name.data[:] == expected_names
    assert nwbfile.electrodes.filtering.data[:] == [filtering_info] * 256
    assert nwbfile.electrodes.location.data[:] == ["Hippocampus CA1"] * 256
    assert nwbfile.electrodes.ref_elect_id.data[:] == [-1] * 256


def test_get_raw_ephys_data(dummy_logger):
    """
    Test the get_raw_ephys_data function.

    File `tests/test_data/raw_ephys/2022-07-25_15-30-00` must be created first by running
    `python tests/test_data/create_raw_ephys_test_data.py`.
    """
    folder_path = "tests/test_data/raw_ephys/2022-07-25_15-30-00"
    traces_as_iterator, channel_conversion_factor, original_timestamps = get_raw_ephys_data(folder_path, dummy_logger)
    assert traces_as_iterator.maxshape == (3_000, 256)
    np.testing.assert_allclose(channel_conversion_factor, [0.19499999284744263 * 1e-6] * 256)
    assert len(original_timestamps) == 3_000


def test_read_open_ephys_settings_xml(dummy_logger):
    """
    Test that the read_open_ephys_settings_xml function extracts the correct metadata from the settings.xml file.
    """
    # Read settings file to get filtering info and map of channel number to channel name
    settings_file_path = "tests/test_data/raw_ephys/2022-07-25_15-30-00/settings.xml"
    (
        channel_number_to_channel_name,
        filtering_info
    ) = read_open_ephys_settings_xml(settings_file_path, dummy_logger)

    # Check correct filtering info (single string for all electrodes)
    assert filtering_info == "Filter with highcut=7603.76512183337 Hz, lowcut=2.495988241877236 Hz"

    # We expect channel numbers 0-255 mapping to CH1-CH256, then channels 256-263 mapping to ADC1-ADC8
    expected_channel_dict = {i-1:f"CH{i}" for i in range(1, 257)}
    expected_channel_dict.update({i+255 : f"ADC{i}" for i in range(1, 9)})
    assert channel_number_to_channel_name == expected_channel_dict


def test_add_raw_ephys(dummy_logger):
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
    metadata["ephys"]["targeted_x"] = 4.5  # AP in mm
    metadata["ephys"]["targeted_y"] = 2.2  # ML in mm
    metadata["ephys"]["targeted_z"] = 2.0  # DV in mm
    metadata["ephys"]["probe"] = ["256-ch Silicon Probe, 3mm length, 66um pitch"]

    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    assert len(nwbfile.electrodes) == 256
    assert len(nwbfile.electrode_groups) == 32
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

    # Test that the nwbfile has the expected associated files
    assert "associated_files" in nwbfile.processing
    assert "open_ephys_settings_xml" in nwbfile.processing["associated_files"].data_interfaces
    assert nwbfile.processing["associated_files"]["open_ephys_settings_xml"].description == (
        "Raw settings.xml file from OpenEphys"
    )

    settings_file_path = Path(metadata["ephys"]["openephys_folder_path"]) / "settings.xml"
    with open(settings_file_path, "r") as settings_file:
        expected_raw_settings_xml = settings_file.read()
    assert nwbfile.processing["associated_files"]["open_ephys_settings_xml"].content == expected_raw_settings_xml

    expected_ephys_start = datetime.strptime("2022-07-25_15-30-00", "%Y-%m-%d_%H-%M-%S")
    expected_ephys_start = expected_ephys_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    assert ephys_data_dict.get("ephys_start") == expected_ephys_start


def test_add_ephys_with_incomplete_metadata(dummy_logger, capsys):
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
    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)
    captured = capsys.readouterr() # capture stdout

    # Check that the correct message was printed to stdout
    assert "No ephys metadata found for this session. Skipping ephys conversion." in captured.out
    assert ephys_data_dict == {}

    # Create a test metadata dictionary with an ephys field but no ephys data
    metadata["ephys"] = {}

    # Check that add_raw_ephys raises a ValueError about missing fields in the metadata dictionary
    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)
    captured = capsys.readouterr()
    assert ephys_data_dict == {}
    assert "The required ephys subfields do not exist in the metadata dictionary" in captured.out


@pytest.mark.slow
def test_add_raw_ephys_complete_data():
    """
    Test the add_raw_ephys function with an actual (big) OpenEphys recording and its settings.xml file.

    The only difference between this test and test_add_raw_ephys is that this test uses an actual
    OpenEphys recording. The shape of the ElectricalSeries data and timestamps are different. Everything else
    should be the same.

    Only run this test locally where the large file is present (change the path below as needed).

    NOTE: This test does not actually write the file to disk. That should be tested as well.
    """
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    metadata = {}
    metadata["ephys"] = {}
    metadata["ephys"]["openephys_folder_path"] = "/Users/rly/Documents/NWB/berke-lab-to-nwb/data/2022-07-25_15-30-00"
    metadata["ephys"]["impedance_file_path"] = "tests/test_data/processed_ephys/impedance.csv"
    metadata["ephys"]["electrodes_location"] = "Hippocampus CA1"
    metadata["ephys"]["targeted_x"] = 4.5  # AP in mm
    metadata["ephys"]["targeted_y"] = 2.2  # ML in mm
    metadata["ephys"]["targeted_z"] = 2.0  # DV in mm
    metadata["ephys"]["probe"] = ["256-ch Silicon Probe, 3mm length, 66um pitch"]
    
    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata)

    assert len(nwbfile.electrodes) == 256
    assert len(nwbfile.electrode_groups) == 32
    assert len(nwbfile.acquisition) == 1
    assert "ElectricalSeries" in nwbfile.acquisition
    es = nwbfile.acquisition["ElectricalSeries"]
    assert es.description == (
        "Raw ephys data from OpenEphys recording (multiply by conversion factor to get data in volts)."
    )
    assert es.data.maxshape == (157_733_308, 256)
    assert es.data.dtype == np.int16
    assert es.electrodes.data == list(range(256))
    assert es.timestamps.shape == (157_733_308,)
    assert es.conversion == 0.19499999284744263 * 1e-6

    # Test that the nwbfile has the expected associated files
    assert "associated_files" in nwbfile.processing
    assert "open_ephys_settings_xml" in nwbfile.processing["associated_files"].data_interfaces
    assert nwbfile.processing["associated_files"]["open_ephys_settings_xml"].description == (
        "Raw settings.xml file from OpenEphys"
    )

    settings_file_path = Path(metadata["ephys"]["openephys_folder_path"]) / "settings.xml"
    with open(settings_file_path, "r") as settings_file:
        expected_raw_settings_xml = settings_file.read()
    assert nwbfile.processing["associated_files"]["open_ephys_settings_xml"].content == expected_raw_settings_xml

    expected_ephys_start = datetime.strptime("2022-07-25_15-30-00", "%Y-%m-%d_%H-%M-%S")
    expected_ephys_start = expected_ephys_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    assert ephys_data_dict.get("ephys_start") == expected_ephys_start
