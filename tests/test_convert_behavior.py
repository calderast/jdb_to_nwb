from datetime import datetime
from dateutil import tz
from pynwb import NWBFile

from jdb_to_nwb.convert_behavior import add_behavior


def test_convert_behavior():
    """Test the add_behavior function."""

    # Test data is copied from /Volumes/Tim/Photometry/IM-1478/07252022/
    metadata = {}
    metadata["behavior"] = {}
    metadata["behavior"]["arduino_text_file_path"] = "tests/test_data/behavior/arduinoraw0.txt"
    metadata["behavior"]["arduino_timestamps_file_path"] = "tests/test_data/behavior/ArduinoStamps0.csv"
    metadata["behavior"]["maze_configuration_file_path"] = "tests/test_data/behavior/barriers.txt"

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_behavior(nwbfile, metadata)

    # Note that we do not run validation tests on the trial- and block-level data (e.g. trials/blocks are
    # enumerated corrrectly, block metadata is valid, number of trials per block adds to the
    # total number of trials, block start/end times are aligned to trial start/end times)
    # because these checks are already run by validate_trial_and_block_data every time add_behavior is run

    # Test that the nwbfile has the expected associated files
    assert "associated_files" in nwbfile.processing
    assert "arduino_text" in nwbfile.processing["associated_files"].data_interfaces
    assert "arduino_timestamps" in nwbfile.processing["associated_files"].data_interfaces
    assert nwbfile.processing["associated_files"]["arduino_text"].description == "Raw arduino text"

    # Test that the session description has been updated
    assert (
        nwbfile.experiment_description == "Barrier change session for the hex maze task with 3 blocks and 188 trials."
    )

    # Test that expected block data has been added to the nwbfile
    assert "block" in nwbfile.intervals
    expected_block_columns = {
        "start_time",
        "stop_time",
        "block",
        "maze_configuration",
        "prob_A",
        "prob_B",
        "prob_C",
        "num_trials_in_block",
    }
    assert set(nwbfile.intervals["block"].colnames) == expected_block_columns
    assert len(nwbfile.intervals["block"].start_time) == 3  # there are 3 blocks in this session
    # Check block data
    assert nwbfile.intervals["block"].num_trials_in_block[0] == 68
    assert nwbfile.intervals["block"].maze_configuration[0] == "{35, 11, 12, 45, 14, 15, 18, 22, 29, 31}"
    assert nwbfile.intervals["block"].prob_A[0] == 10
    assert nwbfile.intervals["block"].prob_B[0] == 50
    assert nwbfile.intervals["block"].prob_C[0] == 90

    # Test that expected trial data has been added to the nwbfile
    assert nwbfile.trials is not None
    expected_trial_columns = {
        "start_time",
        "stop_time",
        "duration",
        "trial",
        "trial_within_session",
        "block",
        "start_port",
        "end_port",
        "reward",
        "beam_break_start",
        "beam_break_end",
    }
    assert set(nwbfile.trials.colnames) == expected_trial_columns
    assert len(nwbfile.trials.start_time) == 188  # there are 188 trials in this session
