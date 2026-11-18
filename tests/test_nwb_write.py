"""
Write a converted nwb to disk and read it back.

Every other test in this suite checks the NWBFile in memory, which is not where the structures
this conversion builds can fail. hdmf only resolves a unit's 'electrodes' region and its
'electrode_group' link, and settles every column's dtype, when the file is WRITTEN; pynwb only
validates a file on disk, and so does nwbinspector. Two tests here, both through a real write:

  - the ephys structures on their own, assembled from the fixtures the converter tests use, so
    the units, the laser events and the raw electrical series are checked without a slow full
    conversion behind them;
  - create_nwbs end to end on a real session, which is the only test that runs the whole
    conversion and the only one that checks the file it produces.
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, validate

from jdb_to_nwb.convert import create_nwbs
from jdb_to_nwb.convert_optogenetics import add_laser_events
from jdb_to_nwb.convert_raw_ephys import add_raw_ephys
from jdb_to_nwb.convert_spikes import add_spikes

from .test_convert_optogenetics import PROBE_OFFSET_SAMPLES, _laser_fixture
from .test_convert_spikes import _kilosort_bombcell_fixture

FS = 30000.0


def _converted_nwbfile(tmp_path, logger):
    """An nwb holding what the ephys converters add: a Units table and the laser events."""
    metadata, nwbfile = _kilosort_bombcell_fixture(tmp_path, fs=FS)
    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=logger)

    processed = _laser_fixture(tmp_path, fs=FS)
    offset = PROBE_OFFSET_SAMPLES / FS
    add_laser_events(
        nwbfile=nwbfile,
        processed_ephys_path=processed,
        session_boundaries=json.loads((processed / "session_boundaries.json").read_text()),
        probe_start_times={"ProbeA": {"behavior": 0.0, "optotagging": 500.0},
                           "ProbeB": {"behavior": -offset, "optotagging": 500.0 - offset}},
        session_hardware_starts={"behavior": 0.0, "optotagging": 500.0},
        session_nwb_starts={"behavior": 0.0, "optotagging": 452.0},
        logger=logger,
    )
    return nwbfile


def _write(nwbfile, tmp_path):
    """Write the nwb the way the conversion does, and return the path."""
    nwb_path = tmp_path / "write_and_read.nwb"
    with NWBHDF5IO(nwb_path, mode="w") as io:
        io.write(nwbfile)
    return nwb_path


def test_a_converted_nwb_survives_a_write_and_read(dummy_logger, tmp_path):
    """
    Read the written file back and check the parts that only exist once it is on disk: the Units
    table's own attributes, a unit's reference into the electrodes table, and the laser events.
    """
    nwb_path = _write(_converted_nwbfile(tmp_path, dummy_logger), tmp_path)

    with NWBHDF5IO(nwb_path, mode="r") as io:
        nwbfile = io.read()

        units = nwbfile.units
        assert len(units) == 3
        # Set when the table was created, so a write and read is the only proof they were kept
        assert units.resolution == 1 / FS
        assert units.waveform_rate == FS

        units_df = units.to_dataframe()
        peak_row = units_df["peak_electrode"][0]
        # 'electrodes' is an object reference into the electrodes table, resolved on read: it must
        # come back pointing at the same electrode as the row number, on that electrode's shank
        assert list(units_df["electrodes"][0].index) == [peak_row]
        assert (units_df["electrode_group"][0].name
                == nwbfile.electrodes["group"][peak_row].name)
        # The mean waveform on that electrode, still 1D over samples
        assert units_df["waveform_mean"][0].ndim == 1

        laser_events = nwbfile.intervals["laser_events"].to_dataframe()
        assert len(laser_events) == 2
        assert list(laser_events["color"]) == ["blue_laser", "blue_laser"]
        assert laser_events["start_time"][0] == pytest.approx(452.0 + 30 / FS)


def test_a_written_nwb_has_no_critical_nwbinspector_messages(dummy_logger, tmp_path):
    """
    Run the written file through nwbinspector, which is what DANDI and Spyglass users will run.

    nwbinspector is not a dependency of this package: it requires pynwb >= 3.1 while we pin
    pynwb == 3.0, so this test skips where it is not installed rather than forcing that upgrade.

    Only CRITICAL and above fail here, which is what says the file itself is wrong - a unit's
    electrodes region pointing outside the electrodes table, for one. The suggestions below that
    are about the metadata a real session is given (subject, experimenter, institution), which a
    mock file does not have and this test is not about.
    """
    nwbinspector = pytest.importorskip("nwbinspector")
    nwb_path = _write(_converted_nwbfile(tmp_path, dummy_logger), tmp_path)

    messages = list(nwbinspector.inspect_nwbfile(
        nwbfile_path=str(nwb_path),
        importance_threshold=nwbinspector.Importance.CRITICAL,
    ))

    assert messages == [], "\n".join(f"{m.importance.name} {m.check_function_name} "
                                     f"({m.location}): {m.message}" for m in messages)


def _raw_ephys_nwbfile(logger):
    """An nwb holding what add_raw_ephys adds: the electrodes table and the ElectricalSeries."""
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime(2022, 7, 25, 15, 30, tzinfo=ZoneInfo("America/Los_Angeles")),
        identifier="mock_session",
    )
    metadata = {"ephys": {
        "openephys_folder_paths": [{"path": "tests/test_data/raw_ephys/2022-07-25_15-30-00",
                                    "session": "behavior"}],
        "impedance_file_path": "tests/test_data/processed_ephys/impedance.csv",
        "probes": [{"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
                    "electrodes_location": "Hippocampus CA1",
                    "targeted_x": 2.2,    # ML in mm
                    "targeted_y": 4.5,    # AP in mm
                    "targeted_z": -2.0}],  # DV in mm
    }}
    add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=logger)
    return nwbfile


def test_a_written_nwb_keeps_the_electrical_series_and_its_electrodes(dummy_logger, tmp_path):
    """
    Write the raw ephys structures and read them back.

    This is the largest thing the conversion writes and the part a write can go wrong in without
    anything in memory looking different: the samples are written lazily, straight out of the Open
    Ephys recording an iterator at a time, and each column of them is tied to its electrode by a
    reference into the electrodes table that is only a reference once it is on disk. The in-memory
    test of add_raw_ephys checks the row numbers it put in that region; this checks they still
    point at the same electrodes after a round trip.
    """
    nwbfile = _raw_ephys_nwbfile(dummy_logger)
    # The channel-to-electrode-row mapping add_raw_ephys worked out, kept before the write so the
    # read can be held to it
    expected_electrode_rows = list(nwbfile.acquisition["ElectricalSeries"].electrodes.data)
    expected_group_names = [nwbfile.electrodes["group"][row].name for row in expected_electrode_rows]
    nwb_path = _write(nwbfile, tmp_path)

    with NWBHDF5IO(nwb_path, mode="r") as io:
        nwbfile = io.read()

        assert len(nwbfile.electrodes) == 256
        assert len(nwbfile.electrode_groups) == 32   # one per shank of the 256-ch probe

        electrical_series = nwbfile.acquisition["ElectricalSeries"]
        # The samples, written an iterator chunk at a time: still every channel of every sample,
        # still the int16 the Intan box digitized, with the uV -> V factor kept beside them
        assert electrical_series.data.shape == (3_000, 256)
        assert electrical_series.data.dtype == np.int16
        assert electrical_series.conversion == 1e-6
        assert electrical_series.timestamps.shape == (3_000,)

        # Column i of that data is the electrode in row expected_electrode_rows[i]: the region must
        # come back resolving to the same electrodes, on the same shanks, in the same order
        assert list(electrical_series.electrodes.data) == expected_electrode_rows
        assert electrical_series.electrodes.table.name == "electrodes"
        read_group_names = [electrical_series.electrodes.table["group"][row].name
                            for row in electrical_series.electrodes.data]
        assert read_group_names == expected_group_names

        # The settings the recording was made with, embedded as a file rather than parsed away
        assert ("open_ephys_settings_xml"
                in nwbfile.processing["associated_files"].data_interfaces)


# Tim's IM-1478 07/25/2022 session, which is where the rest of the test data in this repo comes
# from: LabVIEW photometry, arduino behavior, DeepLabCut position and MountainSort spikes
TIM_SESSION = Path("tests/test_data/downloaded/IM-1478/07252022")

FULL_CONVERSION_METADATA = f"""
experimenter:
  - Tim Krausz
experiment_description: Hex maze task
date: 07/25/2022
maze: old_maze
subject:
  subject_id: IM-1478
  species: Rattus norvegicus
  genotype: Wildtype
  sex: M
  age: 6mo
  description: Long Evans Rat

# Raw ephys is left out: the committed Open Ephys folder is a fraction of a second long, so its
# ADC holds no port visits, and there is nothing to align the other datastreams to it by.
# Spike sorting output does not need the raw recording on the MountainSort path.
ephys:
  mountain_sort_output_file_path: "tests/test_data/processed_ephys/firings.mda"
  sampling_frequency: 30000

photometry:
  phot_file_path: "{TIM_SESSION}/IM-1478_2022-07-25_15-24-22____Tim_Conditioning.phot"
  box_file_path: "{TIM_SESSION}/IM-1478_2022-07-25_15-24-22____Tim_Conditioning.box"
  optic_fiber_implant_sites:
    - optic_fiber: Doric 0.66mm Flat 40mm Optic Fiber
      targeted_location: NAcc
      ap_in_mm: 1.7
      ml_in_mm: -1.7
      dv_in_mm: -6.0
      recording: true
  virus_injections:
    - virus_name: dLight1.3b
      targeted_location: NAcc
      ap_in_mm: 1.7
      ml_in_mm: -1.7
      dv_in_mm: -6.2
      volume_in_uL: 1.0
      titer_in_vg_per_mL: 2e12

behavior:
  arduino_text_file_path: "tests/test_data/behavior/arduinoraw0.txt"
  arduino_timestamps_file_path: "tests/test_data/behavior/ArduinoStamps0.csv"
  maze_configuration_file_path: "tests/test_data/behavior/barriers.txt"

video:
  video_timestamps_file_path: "{TIM_SESSION}/testvidtimes0.csv"
  dlc_path: "{TIM_SESSION}/Behav_Vid0DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000.h5"
  hex_centroids_file_path: "{TIM_SESSION}/hex_coordinates_IM-1478_07252022.csv"
"""


def test_a_full_conversion_writes_a_readable_nwb(tmp_path):
    """
    Run create_nwbs on a real session and read the nwb it writes.

    The converter tests each check their own datastream on an NWBFile they built themselves, so
    nothing else checks what create_nwbs assembles: which clock the file ends up on, the log and
    metadata files it embeds, and whether all of it survives being written. This is also the only
    test that runs the conversion the way a person does, from a metadata yaml.

    It takes a few minutes, almost all of it photometry, and is deliberately NOT marked slow: a
    conversion test that CI skips would not have caught the conversion breaking.
    """
    metadata_path = tmp_path / "IM-1478_20220725_metadata.yaml"
    metadata_path.write_text(FULL_CONVERSION_METADATA)

    create_nwbs(metadata_file_path=metadata_path, output_nwb_dir=tmp_path)

    # Named from the subject and date in the metadata, beside the figures and logs
    nwb_path = tmp_path / "IM-1478_20220725.nwb"
    assert nwb_path.is_file()
    assert (tmp_path / "IM-1478_20220725_figures").is_dir()

    with NWBHDF5IO(nwb_path, mode="r") as io:
        nwbfile = io.read()

        assert nwbfile.session_id == "IM-1478_20220725"
        # The clock the whole file is on: this session has photometry, so its exact start wins
        # over the ephys start and over the default midnight of the session date
        assert nwbfile.session_start_time == datetime(2022, 7, 25, 15, 24, 22,
                                                      tzinfo=ZoneInfo("America/Los_Angeles"))
        assert nwbfile.timestamps_reference_time == nwbfile.session_start_time
        # Written by the behavior conversion, over the placeholder create_nwbs starts with
        assert nwbfile.session_description == ("barrier change session for the hex maze task "
                                               "with 3 blocks and 188 trials.")
        assert nwbfile.source_script.startswith("jdb_to_nwb ")
        assert nwbfile.subject.subject_id == "IM-1478"
        assert nwbfile.experimenter == ("Tim Krausz",)

        # Photometry: the raw, processed and corrected signals
        for series in ("raw_470", "raw_405", "processed_470", "fitted_405", "corrected_470_dFF"):
            assert series in nwbfile.acquisition

        # Behavior: the trials and blocks, the task, and the run epoch
        assert len(nwbfile.trials) == 188          # the same 188 the behavior test counts
        assert "block" in nwbfile.intervals
        assert "task_0" in nwbfile.processing["tasks"].data_interfaces
        assert list(nwbfile.epochs.to_dataframe()["tags"][0]) == ["00_r1"]

        # Position and the maze, from DeepLabCut and the hex centroids
        behavior_module = nwbfile.processing["behavior"].data_interfaces
        assert "position" in behavior_module
        assert "hex_centroids" in behavior_module
        assert set(behavior_module["position"].spatial_series) == {"cap_front_position",
                                                                  "cap_back_position"}

        # Spikes, on the photometry clock, with the resolution the Units table was created with
        assert len(nwbfile.units) == 17            # the same 17 the MountainSort test counts
        assert nwbfile.units.resolution == 1 / 30_000

        # The file says how it was made: the metadata yaml and all three conversion logs are in it
        associated_files = nwbfile.processing["associated_files"].data_interfaces
        for embedded in ("metadata_yaml", "info_log", "warning_log", "debug_log",
                         "arduino_text", "arduino_timestamps"):
            assert embedded in associated_files
        assert associated_files["metadata_yaml"].content == FULL_CONVERSION_METADATA

    # Validate by path, not through an open NWBHDF5IO: this process has already imported
    # ndx-fiber-photometry, and validating an open file then picks that namespace over core and
    # fails looking for NWBFile in it
    validation_errors = validate(path=str(nwb_path))

    # Every photometry nwb this converter writes fails validation the same way: with
    # ndx-fiber-photometry 0.1.0 an OpticalFiber wants an OpticalFiberModel and an
    # ExcitationSource wants a wavelength and an ExcitationSourceModel, and the photometry
    # conversion writes none of the three. That is a gap in the photometry metadata, so it is
    # listed here rather than ignored: anything else that fails validation fails this test.
    known_photometry_gaps = ("missing data type OpticalFiberModel",
                             "missing data type ExcitationSourceModel",
                             "ExcitationSource/wavelength_in_nm")
    unexpected = [str(error) for error in validation_errors
                  if not any(gap in str(error) for gap in known_photometry_gaps)]
    assert unexpected == [], f"the written nwb does not validate: {unexpected}"
