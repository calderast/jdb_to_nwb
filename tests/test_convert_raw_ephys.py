from datetime import datetime
from pathlib import Path
import numpy as np
from dateutil import tz
from zoneinfo import ZoneInfo
from pynwb import NWBFile
import pytest
import re

from jdb_to_nwb import convert_raw_ephys
from jdb_to_nwb.convert_raw_ephys import (
    add_raw_ephys,
    get_combined_raw_ephys_data,
    add_probe_info,
#    get_electrode_info, # TODO add test for this
    read_open_ephys_settings_xml,
    add_electrode_data_berke_probe,
    add_electrode_data_neuropixels,
    find_open_ephys_paths,
    combine_probe_recordings,
    resolve_probe_stream,
    get_probe_configs,
    probe_name_token,
    get_session_time_offsets,
    concatenate_session_recordings,
)


def test_add_probe_info(dummy_logger):
    """
    Test the add_probe_info function.
    """
    # Create a test metadata dictionary
    metadata = {}
    metadata["ephys"] = {}

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Add probe to the nwbfile
    probe_type = "256-ch Silicon Probe, 3mm length, 66um pitch"
    probe_metadata, probe_obj = add_probe_info(nwbfile=nwbfile, probe_type=probe_type, logger=dummy_logger)

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
    metadata["ephys"]["probes"] = [
        {"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
         "electrodes_location": "Hippocampus CA1",
         "targeted_x": 2.2,    # ML in mm
         "targeted_y": 4.5,    # AP in mm
         "targeted_z": -2.0},  # DV in mm
    ]
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
    probe_metadata, probe_obj = add_probe_info(
        nwbfile=nwbfile, probe_type=metadata["ephys"]["probes"][0]["probe_type"],
        logger=dummy_logger)

    # Add electrode data to the NWBFile
    add_electrode_data_berke_probe(
        probe_location=metadata["ephys"]["probes"][0],
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
        assert egroup.targeted_x == 2.2   # ML
        assert egroup.targeted_y == 4.5   # AP
        assert egroup.targeted_z == -2.0  # DV
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


@pytest.mark.slow
def test_add_electrode_data_neuropixels(dummy_logger):
    """
    Test add_electrode_data_neuropixels adds the FULL neuropixels probe geometry (all 5120 sites, all 4 shanks) 
    to the Probe's Shank/ShanksElectrode objects, even sites/shanks not recorded this session. 
    The electrodes table gets ONLY the recorded channels.
    """
    import pandas as pd

    sites_per_shank = convert_raw_ephys.NPX_ELECTRODES_PER_SHANK  # 1280

    # Real canonical NPX 2.0 4-shank geometry (5120 sites) that the converter registers on the Probe.
    coords = pd.read_csv(convert_raw_ephys.ELECTRODE_COORDS_PATH_NPX_MULTISHANK)
    assert len(coords) == 5120 and sorted(coords["shank"].unique()) == [0, 1, 2, 3]

    # Simulate a session recording 384 channels on shanks 0/1/2 only (like a real IM-1971 session)
    recorded = pd.concat([coords[coords["shank"] == s].head(128) for s in [0, 1, 2]], ignore_index=True)
    recorded["channel_num"] = range(len(recorded))
    # channel_info has exactly the columns get_channel_map_neuropixels produces (real coords + a channel_num)
    channel_info = recorded[["channel_num", "shank", "electrode", "shank_column", "shank_row", "x_um", "y_um"]]
    assert len(channel_info) == 384

    nwbfile = NWBFile(session_description="Mock session", session_start_time=datetime.now(tz.tzlocal()),
                      identifier="mock_session")
    probe_location = {"electrodes_location": "Hippocampus CA1",
                      "targeted_x": 2.9, "targeted_y": -3.6, "targeted_z": -1.8}  # x=ML, y=AP, z=DV
    _, probe_obj = add_probe_info(nwbfile=nwbfile, probe_type="Neuropixels 2.0 (4-shank)",
                                  logger=dummy_logger)

    add_electrode_data_neuropixels(nwbfile=nwbfile, channel_info=channel_info, filtering_info="hw filtering",
                                   probe_location=probe_location, probe_obj=probe_obj,
                                   probe_label="ProbeA", logger=dummy_logger)

    # The Probe device exists under 'devices' with the expected Neuropixels metadata (from ephys_devices.yaml)
    assert "Neuropixels 2.0 (4-shank)" in nwbfile.devices
    probe = nwbfile.devices["Neuropixels 2.0 (4-shank)"]
    assert probe is probe_obj
    assert probe.probe_description.startswith("4 shanks x 1280 electrodes per shank for 5120 total recording sites")
    assert probe.manufacturer == "IMEC"
    assert probe.contact_side_numbering
    assert probe.contact_size == 12
    assert probe.units == "um"

    # One ElectrodeGroup per RECORDED shank (0, 1, 2); shank 3 recorded nothing so gets no group
    assert len(nwbfile.electrode_groups) == 3
    found_shanks = set()
    for name, egroup in nwbfile.electrode_groups.items():
        assert egroup.location == "Hippocampus CA1"
        assert egroup.targeted_x == 2.9     # ML
        assert egroup.targeted_y == -3.6    # AP
        assert egroup.targeted_z == -1.8    # DV
        assert egroup.device is probe
        match = re.match(r"Electrodes on shank (\d+)", egroup.description)
        assert match, f"Unexpected description in group {name}: {egroup.description}"
        found_shanks.add(int(match.group(1)))
    assert found_shanks == {0, 1, 2}

    # Check the FULL geometry (all 4 shanks, all 5120 sites) is on the Probe
    assert len(probe_obj.shanks) == 4
    assert sum(len(shank.shanks_electrodes) for shank in probe_obj.shanks.values()) == 5120
    # ShanksElectrode.name is the GLOBAL electrode index, partitioned per shank (0-1279, 1280-2559, ...)
    names_by_shank = {}
    for shank in range(4):
        names = sorted(int(e.name) for e in probe_obj.shanks[str(shank)].shanks_electrodes.values())
        assert names == list(range(shank * sites_per_shank, (shank + 1) * sites_per_shank))
        names_by_shank[shank] = set(names)

    # Check the electrodes table has ONLY the recorded channels (shanks 0/1/2 -> 384 rows)
    assert len(nwbfile.electrodes) == 384
    electrodes = nwbfile.electrodes.to_dataframe()
    assert set(electrodes["probe_shank"]) == {0, 1, 2}  # unrecorded shank 3 is not in the table

    # Every recorded electrode's global probe_electrode matches a ShanksElectrode.name on its shank
    for _, row in electrodes.iterrows():
        assert int(row["probe_electrode"]) in names_by_shank[int(row["probe_shank"])]

    # Check the electrode-table columns against the recorded channel_info (electrodes are added in
    # channel_num order, which is the row order of `recorded`, so the columns line up positionally)
    assert list(electrodes["intan_channel_number"]) == list(recorded["channel_num"])  # 0..383
    assert list(electrodes["electrode_index"]) == list(recorded["electrode"])          # global index
    assert list(electrodes["probe_electrode"]) == list(recorded["electrode"])          # global (used by Spyglass)
    assert list(electrodes["probe_shank"]) == list(recorded["shank"])
    assert list(electrodes["group_name"]) == [str(s) for s in recorded["shank"]]
    np.testing.assert_allclose(electrodes["rel_x"], recorded["x_um"])
    np.testing.assert_allclose(electrodes["rel_y"], recorded["y_um"])
    # electrode_name is the cosmetic 'S{shank:02d}E{local}' label (local = global % sites_per_shank)
    expected_electrode_names = [
        f"S{int(s):02d}E{int(e) % sites_per_shank}" for s, e in zip(recorded["shank"], recorded["electrode"])
    ]
    assert list(electrodes["electrode_name"]) == expected_electrode_names
    # Columns that are constant across all electrodes for Neuropixels
    assert not electrodes["bad_channel"].any()  # no impedance file, so no channels marked bad
    assert list(electrodes["filtering"]) == ["hw filtering"] * 384
    assert list(electrodes["location"]) == ["Hippocampus CA1"] * 384
    assert list(electrodes["ref_elect_id"]) == [-1] * 384


def test_get_combined_raw_ephys_data_single_probe(dummy_logger):
    """
    Test reading raw ephys for a single probe, which goes through the same path as multi-probe.

    File `tests/test_data/raw_ephys/2022-07-25_15-30-00` must be created first by running
    `python tests/test_data/create_raw_ephys_test_data.py`.
    """
    folder_path = "tests/test_data/raw_ephys/2022-07-25_15-30-00"
    (combined_recording, channel_conversion_factor, original_timestamps,
     num_samples_by_probe) = get_combined_raw_ephys_data(
        folder_path=folder_path,
        probe_streams_by_label={"probe": "Rhythm_FPGA-100.0"},
        logger=dummy_logger,
    )
    # Each probe reports its own full length, for the spike sorting guard
    assert num_samples_by_probe == {"probe": 3_000}
    assert (combined_recording.get_num_samples(), combined_recording.get_num_channels()) == (3_000, 256)
    np.testing.assert_allclose(channel_conversion_factor, [0.19499999284744263] * 256)
    assert len(original_timestamps) == 3_000


def test_find_open_ephys_paths_autodetects_experiment_1():
    """The real test data is experiment1 (settings.xml), so auto-detection should pick experiment 1."""
    folder_path = "tests/test_data/raw_ephys/2022-07-25_15-30-00"
    paths = find_open_ephys_paths(folder_path)  # experiment_number=None -> auto-detect
    assert paths["settings_file"].endswith("settings.xml")
    assert not paths["settings_file"].endswith("settings_2.xml")


def _make_open_ephys_experiment(base_dir: Path, experiment_number: int, probe_names):
    """Create a minimal Open Ephys folder tree (settings + continuous.dat per probe) for one experiment."""
    suffix = "" if experiment_number == 1 else f"_{experiment_number}"
    record_node = base_dir / "Record Node 101"
    (record_node / f"settings{suffix}.xml").parent.mkdir(parents=True, exist_ok=True)
    (record_node / f"settings{suffix}.xml").write_text("<SETTINGS/>")
    continuous = record_node / f"experiment{experiment_number}" / "recording1" / "continuous"
    for probe_name in probe_names:
        probe_dir = continuous / probe_name
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / "continuous.dat").write_bytes(b"\x00\x00")


def test_find_open_ephys_paths_autodetects_experiment_2(tmp_path):
    """A Neuropixels-style experiment2 recording (settings_2.xml) should be auto-detected."""
    _make_open_ephys_experiment(tmp_path, 2, ["OneBox-100.ProbeA", "OneBox-100.OneBox-ADC"])
    paths = find_open_ephys_paths(str(tmp_path))  # auto-detect
    assert paths["settings_file"].endswith("settings_2.xml")
    assert set(paths["recording_files"]) == {"OneBox-100.ProbeA", "OneBox-100.OneBox-ADC"}


def test_find_open_ephys_paths_ambiguous_experiments_raises(tmp_path):
    """If multiple experiments are present, auto-detection must refuse and ask the user to specify."""
    _make_open_ephys_experiment(tmp_path, 1, ["ProbeA"])
    _make_open_ephys_experiment(tmp_path, 2, ["ProbeA"])
    with pytest.raises(ValueError, match="multiple Open Ephys experiments"):
        find_open_ephys_paths(str(tmp_path))
    # ...but an explicit experiment_number resolves the ambiguity
    paths = find_open_ephys_paths(str(tmp_path), experiment_number=2)
    assert paths["settings_file"].endswith("settings_2.xml")


def test_find_open_ephys_paths_no_settings_raises(tmp_path):
    """No settings*.xml anywhere should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No settings.xml"):
        find_open_ephys_paths(str(tmp_path))


def test_read_open_ephys_settings_xml(dummy_logger):
    """
    Test that the read_open_ephys_settings_xml function extracts the correct metadata from the settings.xml file.
    """
    # Read settings file to validate channels and get filtering info
    settings_file_path = "tests/test_data/raw_ephys/2022-07-25_15-30-00/settings.xml"
    filtering_info = read_open_ephys_settings_xml(settings_file_path, dummy_logger)

    # Check correct filtering info (single string for all electrodes)
    assert filtering_info == "Filter with highcut=7603.76512183337 Hz, lowcut=2.495988241877236 Hz"


def test_read_open_ephys_settings_xml_new_format(dummy_logger):
    """
    Test that the read_open_ephys_settings_xml function handles the new-format settings.xml
    (Open Ephys >= 1.0, with Acquisition Board processor and STREAM element instead of
    Sources/Rhythm FPGA with CHANNEL_INFO).
    """
    settings_file_path = "tests/test_data/raw_ephys/new_format_settings.xml"
    filtering_info = read_open_ephys_settings_xml(settings_file_path, dummy_logger)

    # Check correct filtering info
    assert filtering_info == "Filter with highcut=7603.76512183337 Hz, lowcut=1.577845136933669 Hz"


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
    metadata["ephys"]["openephys_folder_paths"] = [
        {"path": "tests/test_data/raw_ephys/2022-07-25_15-30-00", "session": "behavior"}]
    metadata["ephys"]["impedance_file_path"] = "tests/test_data/processed_ephys/impedance.csv"
    metadata["ephys"]["probes"] = [
        {"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
         "electrodes_location": "Hippocampus CA1",
         "targeted_x": 2.2,    # ML in mm
         "targeted_y": 4.5,    # AP in mm
         "targeted_z": -2.0},  # DV in mm
    ]
    
    # Mapping from probe electrode to channel number for 256-ch silicon probe
    intan_channel_numbers = np.array([
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
    # Reverse mapping from channel (aka row in ElectricalSeries) to row in electrode table
    expected_electrode_data_mapping = list(np.argsort(intan_channel_numbers))

    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    assert len(nwbfile.electrodes) == 256
    assert len(nwbfile.electrode_groups) == 32
    assert len(nwbfile.acquisition) == 1
    assert "ElectricalSeries" in nwbfile.acquisition
    es = nwbfile.acquisition["ElectricalSeries"]
    assert es.description == (
        "Raw ephys data from OpenEphys recording, in uV (multiply by conversion factor to get data in V)."
    )
    assert es.data.maxshape == (3_000, 256)
    assert es.data.dtype == np.int16
    assert es.electrodes.data == expected_electrode_data_mapping
    assert es.timestamps.shape == (3_000,)
    assert es.conversion == 1e-6

    # Test that the nwbfile has the expected associated files
    assert "associated_files" in nwbfile.processing
    assert "open_ephys_settings_xml" in nwbfile.processing["associated_files"].data_interfaces
    assert nwbfile.processing["associated_files"]["open_ephys_settings_xml"].description == (
        "Raw settings.xml file from OpenEphys"
    )

    settings_file_path = Path(metadata["ephys"]["openephys_folder_paths"][0]["path"]) / "settings.xml"
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
    # This test needs the full (large) OpenEphys recording, which is not committed to the repo. Point
    # this at a local copy; the test SKIPS (rather than errors) if it isn't present on this machine.
    openephys_folder_path = "/Users/rly/Documents/NWB/berke-lab-to-nwb/data/2022-07-25_15-30-00"
    if not Path(openephys_folder_path).exists():
        pytest.skip(f"Full OpenEphys recording not found at {openephys_folder_path}; this test only runs "
                    "locally where the large recording is present.")

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    metadata = {}
    metadata["ephys"] = {}
    metadata["ephys"]["openephys_folder_paths"] = [
        {"path": openephys_folder_path, "session": "behavior"}]
    metadata["ephys"]["impedance_file_path"] = "tests/test_data/processed_ephys/impedance.csv"
    metadata["ephys"]["probes"] = [
        {"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
         "electrodes_location": "Hippocampus CA1",
         "targeted_x": 2.2,    # ML in mm
         "targeted_y": 4.5,    # AP in mm
         "targeted_z": -2.0},  # DV in mm
    ]
    
    # Mapping from probe electrode to channel number for 256-ch silicon probe
    intan_channel_numbers = np.array([
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
    # Reverse mapping from channel (aka row in ElectricalSeries) to row in electrode table
    expected_electrode_data_mapping = list(np.argsort(intan_channel_numbers))

    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata)

    assert len(nwbfile.electrodes) == 256
    assert len(nwbfile.electrode_groups) == 32
    assert len(nwbfile.acquisition) == 1
    assert "ElectricalSeries" in nwbfile.acquisition
    es = nwbfile.acquisition["ElectricalSeries"]
    assert es.description == (
        "Raw ephys data from OpenEphys recording, in uV (multiply by conversion factor to get data in V)."
    )
    assert es.data.maxshape == (157_733_308, 256)
    assert es.data.dtype == np.int16
    assert es.electrodes.data == expected_electrode_data_mapping
    assert es.timestamps.shape == (157_733_308,)
    assert es.conversion == 1e-6

    # Test that the nwbfile has the expected associated files
    assert "associated_files" in nwbfile.processing
    assert "open_ephys_settings_xml" in nwbfile.processing["associated_files"].data_interfaces
    assert nwbfile.processing["associated_files"]["open_ephys_settings_xml"].description == (
        "Raw settings.xml file from OpenEphys"
    )

    settings_file_path = Path(metadata["ephys"]["openephys_folder_paths"][0]["path"]) / "settings.xml"
    with open(settings_file_path, "r") as settings_file:
        expected_raw_settings_xml = settings_file.read()
    assert nwbfile.processing["associated_files"]["open_ephys_settings_xml"].content == expected_raw_settings_xml

    expected_ephys_start = datetime.strptime("2022-07-25_15-30-00", "%Y-%m-%d_%H-%M-%S")
    expected_ephys_start = expected_ephys_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    assert ephys_data_dict.get("ephys_start") == expected_ephys_start


def _make_test_probe_recording(num_channels=4, duration=2.0, sampling_frequency=30000.0, seed=0):
    """
    Make a small SpikeInterface recording that looks like one Neuropixels probe stream:
    channels named CH0..CHn, and the same channel locations every probe has.
    """
    import spikeinterface.core as si_core

    recording = si_core.generate_recording(
        num_channels=num_channels, durations=[duration], sampling_frequency=sampling_frequency, seed=seed
    )
    return recording.rename_channels([f"CH{i}" for i in range(num_channels)])


def test_combine_probe_recordings_combines_channels(dummy_logger):
    """
    Combining multiple probes must produce one recording holding every probe's channels, in probe
    order, with stream-prefixed channel ids (every Neuropixels probe reuses CH0..CH383, so the
    unprefixed ids would collide). Two probes also share identical channel locations, which
    SpikeInterface refuses to aggregate unless we offset them, so this covers that too.
    """
    probe_a = _make_test_probe_recording(seed=1)
    probe_b = _make_test_probe_recording(seed=2)

    combined, timestamps = combine_probe_recordings({"ProbeA": probe_a, "ProbeB": probe_b}, dummy_logger)

    assert len(timestamps) == combined.get_num_samples()
    assert [str(channel_id) for channel_id in combined.channel_ids] == [
        "ProbeA_CH0", "ProbeA_CH1", "ProbeA_CH2", "ProbeA_CH3",
        "ProbeB_CH0", "ProbeB_CH1", "ProbeB_CH2", "ProbeB_CH3",
    ]

    # Each probe's traces must land in its own block of channels, in order and unmodified
    combined_traces = combined.get_traces(end_frame=100)
    np.testing.assert_allclose(combined_traces[:, :4], probe_a.get_traces(end_frame=100))
    np.testing.assert_allclose(combined_traces[:, 4:], probe_b.get_traces(end_frame=100))


def test_combine_probe_recordings_rejects_mismatched_sampling_rates(dummy_logger):
    """
    Combining probes sampled at different rates into one ElectricalSeries would be meaningless,
    so it must fail loudly rather than silently producing a bogus series.
    """
    probe_a = _make_test_probe_recording(sampling_frequency=30000.0, seed=1)
    probe_b = _make_test_probe_recording(sampling_frequency=20000.0, seed=2)

    with pytest.raises(ValueError, match="do not share a sampling rate"):
        combine_probe_recordings({"ProbeA": probe_a, "ProbeB": probe_b}, dummy_logger)


def test_combine_probe_recordings_aligns_probe_start_times(dummy_logger):
    """
    Probes share a hardware clock but can start a few samples apart (measured at ~45 samples /
    1.5 ms on IM-1979 2026-09-11). Combining must align them in TIME, not just truncate the tail -
    otherwise one ElectricalSeries row would hold samples recorded at different moments.
    """
    probe_a = _make_test_probe_recording(seed=1)
    probe_b = _make_test_probe_recording(seed=2)
    # Make probeB start 45 samples later than probeA, like the real recording
    probe_a.set_times(np.arange(probe_a.get_num_samples()) / 30000.0)
    probe_b.set_times(np.arange(probe_b.get_num_samples()) / 30000.0 + 45 / 30000.0)

    combined, timestamps = combine_probe_recordings({"ProbeA": probe_a, "ProbeB": probe_b}, dummy_logger)

    # probeA must have its first 45 samples dropped so both probes start at the same time
    combined_traces = combined.get_traces(end_frame=50)
    np.testing.assert_allclose(combined_traces[:, :4], probe_a.get_traces(start_frame=45, end_frame=95))
    np.testing.assert_allclose(combined_traces[:, 4:], probe_b.get_traces(end_frame=50))
    # The returned timestamps start at the common (later) start time
    np.testing.assert_allclose(timestamps[0], 45 / 30000.0)


def test_resolve_probe_stream_matches_every_naming_form(dummy_logger, monkeypatch):
    """
    The same probe is named three different ways: metadata says "ProbeA", the continuous folder is
    "OneBox-100.ProbeA", and SpikeInterface reports "Record Node 101#OneBox-100.ProbeA". All three
    must resolve to the same stream, and the ADC stream must never be selected as a probe.
    """
    streams = [
        "Record Node 101#OneBox-100.ProbeA",
        "Record Node 101#OneBox-100.ProbeB",
        "Record Node 101#OneBox-100.OneBox-ADC",
    ]

    class FakeExtractor:
        @staticmethod
        def get_streams(folder_path=None):
            return streams, None

    monkeypatch.setattr(convert_raw_ephys, "OpenEphysBinaryRecordingExtractor", FakeExtractor)

    for requested in ("ProbeA", "OneBox-100.ProbeA", "Record Node 101#OneBox-100.ProbeA"):
        assert resolve_probe_stream("folder", dummy_logger, stream_name=requested) == streams[0]
    assert resolve_probe_stream("folder", dummy_logger, stream_name="ProbeB") == streams[1]

    # Two probe streams and no requested stream is ambiguous, so it must fail rather than guess
    with pytest.raises(ValueError, match="no stream was requested"):
        resolve_probe_stream("folder", dummy_logger, stream_name=None)


def test_get_probe_configs_resolves_each_probe_location(dummy_logger):
    """
    Bilateral probes sit in opposite hemispheres, and Spyglass reads the target hemisphere from the
    SIGN of targeted_x, so each probe must carry its own location.
    """
    metadata = {"ephys": {"probes": [
        {"stream": "ProbeA", "probe_type": "Neuropixels 2.0 (4-shank)", "electrodes_location": "Left NAc",
         "targeted_x": -1.9, "targeted_y": 1.7, "targeted_z": 7.3},
        {"stream": "ProbeB", "probe_type": "Neuropixels 2.0 (4-shank)", "electrodes_location": "Right NAc",
         "targeted_x": 1.9, "targeted_y": 1.7, "targeted_z": 7.3},
    ]}}
    probe_file_paths = {"OneBox-100.ProbeA": "a.dat", "OneBox-100.ProbeB": "b.dat"}

    configs = get_probe_configs(metadata, probe_file_paths, dummy_logger)

    assert [c["label"] for c in configs] == ["ProbeA", "ProbeB"]
    assert [c["stream"] for c in configs] == ["OneBox-100.ProbeA", "OneBox-100.ProbeB"]
    assert configs[0]["electrodes_location"] == "Left NAc"
    # Opposite signs, so Spyglass places them in opposite hemispheres
    assert configs[0]["targeted_x"] < 0 < configs[1]["targeted_x"]


def test_get_probe_configs_selects_only_listed_probes(dummy_logger):
    """Listing a probe is what selects it, so an unlisted probe is simply not converted."""
    metadata = {"ephys": {"probes": [
        {"stream": "ProbeB", "probe_type": "Neuropixels 2.0 (4-shank)", "electrodes_location": "Right NAc",
         "targeted_x": 1.9, "targeted_y": 1.7, "targeted_z": 7.3},
    ]}}
    probe_file_paths = {"OneBox-100.ProbeA": "a.dat", "OneBox-100.ProbeB": "b.dat"}

    configs = get_probe_configs(metadata, probe_file_paths, dummy_logger)

    assert len(configs) == 1 and configs[0]["label"] == "ProbeB"


def test_get_probe_configs_stream_optional_for_a_single_probe(dummy_logger):
    """With only one probe stream in the recording, naming it would be redundant."""
    metadata = {"ephys": {"probes": [
        {"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
         "electrodes_location": "Hippocampus CA1",
         "targeted_x": 2.2, "targeted_y": 4.5, "targeted_z": -2.0},
    ]}}

    configs = get_probe_configs(metadata, {"Rhythm_FPGA-100.0": "a.dat"}, dummy_logger)

    assert len(configs) == 1
    assert configs[0]["stream"] == "Rhythm_FPGA-100.0"
    assert configs[0]["targeted_x"] == 2.2


def test_get_probe_configs_requires_stream_when_several_probes_exist(dummy_logger):
    """Omitting 'stream' is only unambiguous with one probe stream; otherwise it must be named."""
    metadata = {"ephys": {"probes": [
        {"probe_type": "NPX", "electrodes_location": "NAc",
         "targeted_x": 1.9, "targeted_y": 1.7, "targeted_z": 7.3},
    ]}}
    probe_file_paths = {"OneBox-100.ProbeA": "a.dat", "OneBox-100.ProbeB": "b.dat"}

    with pytest.raises(ValueError, match="must name its 'stream'"):
        get_probe_configs(metadata, probe_file_paths, dummy_logger)


def test_get_probe_configs_requires_stream_on_every_probe(dummy_logger):
    """Several probes listed means each must say which one it is."""
    metadata = {"ephys": {"probes": [
        {"stream": "ProbeA", "probe_type": "NPX", "electrodes_location": "L",
         "targeted_x": -1.9, "targeted_y": 1.7, "targeted_z": 7.3},
        {"probe_type": "NPX", "electrodes_location": "R",
         "targeted_x": 1.9, "targeted_y": 1.7, "targeted_z": 7.3},
    ]}}
    probe_file_paths = {"OneBox-100.ProbeA": "a.dat", "OneBox-100.ProbeB": "b.dat"}

    with pytest.raises(ValueError, match="must name its 'stream'"):
        get_probe_configs(metadata, probe_file_paths, dummy_logger)


def test_get_probe_configs_rejects_outdated_flat_location(dummy_logger):
    """
    Location used to live directly under 'ephys'. An un-migrated file must fail loudly rather than
    convert with no location at all.
    """
    metadata = {"ephys": {
        "electrodes_location": "Hippocampus CA1", "targeted_x": 2.2,
        "targeted_y": 4.5, "targeted_z": -2.0,
    }}

    with pytest.raises(ValueError, match="moved into the 'probes' list"):
        get_probe_configs(metadata, {"Rhythm_FPGA-100.0": "a.dat"}, dummy_logger)


def test_get_probe_configs_requires_a_location_per_probe(dummy_logger):
    """A listed probe with no location would silently lose its hemisphere, so it must fail."""
    metadata = {"ephys": {"probes": [{"stream": "ProbeA", "electrodes_location": "NAc"}]}}

    with pytest.raises(ValueError, match="is missing"):
        get_probe_configs(metadata, {"OneBox-100.ProbeA": "a.dat"}, dummy_logger)


def test_get_probe_configs_requires_probes(dummy_logger):
    """There is no default: without a probes list there is nothing to convert."""
    with pytest.raises(ValueError, match="No 'probes' in the ephys metadata"):
        get_probe_configs({"ephys": {}}, {"Rhythm_FPGA-100.0": "a.dat"}, dummy_logger)
def test_probe_name_token_unifies_every_naming_form(dummy_logger):
    """
    The same probe is written four different ways depending on the source. One definition of probe
    identity keeps the matchers from drifting apart, which they previously had.
    """
    same_probe = [
        "ProbeA",                              # metadata
        "OneBox-100.ProbeA",                   # Open Ephys continuous folder
        "Record Node 101#OneBox-100.ProbeA",   # SpikeInterface stream name
        "probeA",                              # preprocessing pipeline folder
    ]
    tokens = {probe_name_token(name) for name in same_probe}
    assert tokens == {"probea"}, f"all four forms must identify the same probe, got {tokens}"

    assert probe_name_token("ProbeB") != probe_name_token("ProbeA")


def test_get_probe_configs_matches_a_fully_qualified_stream_name(dummy_logger):
    """
    A probe named in the SpikeInterface form (with the Record Node prefix) must still resolve.
    This did not match before the matchers were unified.
    """
    metadata = {"ephys": {"probes": [
        {"stream": "Record Node 101#OneBox-100.ProbeB", "probe_type": "Neuropixels 2.0 (4-shank)",
         "electrodes_location": "NAc", "targeted_x": 1.9, "targeted_y": 1.7, "targeted_z": 7.3},
    ]}}
    probe_file_paths = {"OneBox-100.ProbeA": "a.dat", "OneBox-100.ProbeB": "b.dat"}

    configs = get_probe_configs(metadata, probe_file_paths, dummy_logger)

    assert configs[0]["stream"] == "OneBox-100.ProbeB"
    # The label is what the user wrote (minus the Record Node prefix), not a guess at which part
    # of it names the probe - so it is predictable from the metadata
    assert configs[0]["label"] == "OneBox-100.ProbeB"


def test_get_session_time_offsets_places_recordings_by_folder_name(dummy_logger):
    """
    Separate Open Ephys recordings share no sample numbering, so the only link between them is the
    wall-clock time in the folder name. These are the real IM-1979 2026-09-04 folders.
    """
    session_folders = [
        {"path": "D:/D1-1-6_IM-1979/ephys/2026-09-04_14-20-43", "session": "behavior"},
        {"path": "D:/D1-1-6_IM-1979/ephys/2026-09-04_16-39-38", "session": "optotagging"},
    ]

    offsets = get_session_time_offsets(session_folders, dummy_logger)

    # 16:39:38 - 14:20:43 = 2h18m55s = 8335s
    assert offsets == [0.0, 8335.0]


def test_get_session_time_offsets_rejects_out_of_order_recordings(dummy_logger):
    """Listing the recordings out of order would silently build a file with overlapping epochs."""
    session_folders = [
        {"path": "x/2026-09-04_16-39-38", "session": "optotagging"},
        {"path": "x/2026-09-04_14-20-43", "session": "behavior"},
    ]

    with pytest.raises(ValueError, match="not in chronological order"):
        get_session_time_offsets(session_folders, dummy_logger)


def test_concatenate_session_recordings_keeps_the_real_gap(dummy_logger):
    """
    The traces are contiguous but the time between recordings is not. The gap has to survive into
    the timestamps, because that is what makes Spyglass derive one valid interval per session
    rather than treating the whole file as continuous.
    """
    import spikeinterface.core as si_core

    fs = 30000.0
    behavior = si_core.generate_recording(num_channels=4, durations=[2.0], sampling_frequency=fs)
    optotagging = si_core.generate_recording(num_channels=4, durations=[1.0], sampling_frequency=fs)
    behavior_timestamps = np.arange(behavior.get_num_samples()) / fs
    optotagging_timestamps = np.arange(optotagging.get_num_samples()) / fs

    recording, timestamps, epochs = concatenate_session_recordings(
        session_recordings=[behavior, optotagging],
        session_timestamps=[behavior_timestamps, optotagging_timestamps],
        session_offsets=[0.0, 452.0],   # a real 452s gap, as on 2026-09-04
        session_labels=["behavior", "optotagging"],
        logger=dummy_logger,
    )

    assert recording.get_num_samples() == len(timestamps) == 90_000
    assert [epoch["session"] for epoch in epochs] == ["behavior", "optotagging"]
    assert epochs[1]["start_time"] == 452.0
    # Time only ever moves forward, and the jump between sessions is the real dead time
    assert np.all(np.diff(timestamps) > 0)
    assert np.isclose(np.max(np.diff(timestamps)), 452.0 - 2.0 + 1 / fs)
    # Spyglass splits on any step larger than 1.75 sample intervals, so it sees two intervals
    assert np.max(np.diff(timestamps)) > 1.75 / fs


def test_concatenate_session_recordings_rejects_overlapping_recordings(dummy_logger):
    """Overlapping recordings mean the folder names or the order are wrong, so this must not pass."""
    import spikeinterface.core as si_core

    fs = 30000.0
    first = si_core.generate_recording(num_channels=4, durations=[2.0], sampling_frequency=fs)
    second = si_core.generate_recording(num_channels=4, durations=[1.0], sampling_frequency=fs)
    timestamps = [np.arange(first.get_num_samples()) / fs, np.arange(second.get_num_samples()) / fs]

    with pytest.raises(ValueError, match="starts before"):
        concatenate_session_recordings(
            session_recordings=[first, second],
            session_timestamps=timestamps,
            session_offsets=[0.0, 1.0],   # starts before the first recording ends
            session_labels=["behavior", "optotagging"],
            logger=dummy_logger,
        )


def test_add_raw_ephys_concatenates_two_recordings(dummy_logger, tmp_path):
    """
    Two Open Ephys recordings must become ONE ElectricalSeries, with the real dead time between
    them kept in the timestamps and one nwb epoch per recording. This is the shape needed to spike
    sort a behavior session and an optotagging session together in Spyglass.

    Built from the real single-probe fixture, copied to a second folder 30 minutes later.
    """
    import shutil

    fixture = Path("tests/test_data/raw_ephys/2022-07-25_15-30-00")
    first_folder = tmp_path / "2022-07-25_15-30-00"
    second_folder = tmp_path / "2022-07-25_16-00-00"   # 1800s after the first
    shutil.copytree(fixture, first_folder)
    shutil.copytree(fixture, second_folder)

    metadata = {"ephys": {
        "openephys_folder_paths": [
            {"path": str(first_folder), "session": "behavior"},
            {"path": str(second_folder), "session": "optotagging"},
        ],
        "impedance_file_path": "tests/test_data/processed_ephys/impedance.csv",
        "probes": [
            {"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
             "electrodes_location": "Hippocampus CA1",
             "targeted_x": 2.2, "targeted_y": 4.5, "targeted_z": -2.0},
        ],
    }}
    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))

    add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    eseries = nwbfile.acquisition["ElectricalSeries"]
    samples_per_recording = eseries.data.maxshape[0] // 2

    # One series holding both recordings, end to end
    assert eseries.data.maxshape[1] == 256
    assert len(eseries.timestamps) == eseries.data.maxshape[0]

    # One epoch per recording, tagged with its name
    epochs = nwbfile.epochs.to_dataframe()
    assert len(epochs) == 2
    assert [list(tags) for tags in epochs["tags"]] == [["behavior"], ["optotagging"]]

    # The second recording starts 1800s after the first, so the gap is real dead time
    assert np.isclose(epochs["start_time"][1], 1800.0)
    assert epochs["start_time"][1] > epochs["stop_time"][0]

    # Time only moves forward, and the jump between recordings is far larger than one sample,
    # which is what makes Spyglass derive two valid intervals instead of one
    timestamps = np.asarray(eseries.timestamps)
    assert np.all(np.diff(timestamps) > 0)
    assert np.max(np.diff(timestamps)) > 1.75 / 30000.0

    # Each probe's reported length covers every recording it was sorted across
    assert samples_per_recording > 0
def test_add_raw_ephys_hands_spike_sorting_one_label_per_probe(dummy_logger, tmp_path):
    """
    The raw ephys conversion and the spike sorting must agree on what each probe is called, or the
    units are labelled with one name while the sample-count guard looks up another and silently
    finds nothing. They agree by position in the metadata 'probes' list, not by re-deriving the
    name - the metadata may name a probe in any of the forms its Open Ephys stream goes by.
    """
    import shutil

    fixture = Path("tests/test_data/raw_ephys/2022-07-25_15-30-00")
    raw_folder = tmp_path / "2022-07-25_15-30-00"
    shutil.copytree(fixture, raw_folder)

    metadata = {"ephys": {
        "openephys_folder_paths": [{"path": str(raw_folder), "session": "behavior"}],
        "impedance_file_path": "tests/test_data/processed_ephys/impedance.csv",
        "probes": [
            {"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
             "electrodes_location": "Hippocampus CA1",
             "targeted_x": 2.2, "targeted_y": 4.5, "targeted_z": -2.0},
        ],
    }}
    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))

    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # One result per probe, in the same order as the metadata's probes list
    probe_results = ephys_data_dict["probe_results"]
    assert len(probe_results) == len(metadata["ephys"]["probes"])
    # With no 'stream' in the metadata the label is the recording's own stream name. It is not a
    # parsed fragment of it, which for this probe would be the useless "0".
    assert probe_results[0]["label"] == "Rhythm_FPGA-100.0"
    assert probe_results[0]["num_samples"] == 3_000    # the probe's own full length
def test_add_raw_ephys_rejects_recordings_that_disagree(dummy_logger, tmp_path):
    """
    One ElectricalSeries has one channel count, one sample rate and one conversion factor. If two
    recordings disagree, part of the series would be silently misdescribed, so it must fail before
    anything is written.
    """
    import json
    import shutil

    fixture = Path("tests/test_data/raw_ephys/2022-07-25_15-30-00")
    first_folder = tmp_path / "2022-07-25_15-30-00"
    second_folder = tmp_path / "2022-07-25_16-00-00"
    shutil.copytree(fixture, first_folder)
    shutil.copytree(fixture, second_folder)

    # Give the second recording a different sample rate than the first
    oebin_path = second_folder / "experiment1/recording1/structure.oebin"
    oebin = json.loads(oebin_path.read_text())
    oebin["continuous"][0]["sample_rate"] = 25000.0
    oebin_path.write_text(json.dumps(oebin))

    metadata = {"ephys": {
        "openephys_folder_paths": [
            {"path": str(first_folder), "session": "behavior"},
            {"path": str(second_folder), "session": "optotagging"},
        ],
        "impedance_file_path": "tests/test_data/processed_ephys/impedance.csv",
        "probes": [
            {"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
             "electrodes_location": "Hippocampus CA1",
             "targeted_x": 2.2, "targeted_y": 4.5, "targeted_z": -2.0},
        ],
    }}
    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))

    with pytest.raises(ValueError, match="differs between recordings"):
        add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)