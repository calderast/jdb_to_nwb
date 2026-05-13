import pytest
from datetime import datetime
from dateutil import tz
from pynwb import NWBFile
from pynwb.file import ProcessingModule
from hdmf.common.table import DynamicTable, VectorData

from jdb_to_nwb.convert_behavior import add_behavior


def test_convert_behavior(dummy_logger):
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

    behavior_data_dict = add_behavior(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Test that we got the expected photometry start time for this session
    assert behavior_data_dict.get("photometry_start_in_arduino_time") == 55520059.6736
    
    # Test that we got the expected number of port entries
    assert len(behavior_data_dict.get("port_visits")) == 188

    # Note that we do not run validation tests on the trial- and block-level data (e.g. trials/blocks are
    # enumerated correctly, block metadata is valid, number of trials per block adds to the
    # total number of trials, block start/end times are aligned to trial start/end times)
    # because these checks are already run by validate_trial_and_block_data every time add_behavior is run
    
    # Test that the tasks processing module has been added
    assert "tasks" in nwbfile.processing
    tasks_module = nwbfile.processing["tasks"]
    assert isinstance(tasks_module, ProcessingModule)
    assert tasks_module.name == "tasks"
    assert tasks_module.description == "Contains all tasks information"

    # We should have a single task named task_0
    assert len(tasks_module.data_interfaces) == 1
    task = tasks_module.data_interfaces["task_0"]
    assert isinstance(task, DynamicTable)
    assert task.name == "task_0"
    assert task.description == ""

    # Check if the task metadata columns were added correctly
    for val in task.columns:
        assert isinstance(val, VectorData)
    expected_task_columns = {"task_name", "task_description", "task_epochs", "task_environment", "camera_id"}
    assert set(task.colnames) == expected_task_columns, (
        f"Task columns {set(task.colnames)} did not match expected {expected_task_columns}"
    )

    # Test that the nwbfile has the expected associated files
    assert "associated_files" in nwbfile.processing
    assert "arduino_text" in nwbfile.processing["associated_files"].data_interfaces
    assert "arduino_timestamps" in nwbfile.processing["associated_files"].data_interfaces
    assert nwbfile.processing["associated_files"]["arduino_text"].description == "Raw arduino text"

    # Test that the session description has been updated
    assert (
        nwbfile.session_description == "barrier change session for the hex maze task with 3 blocks and 188 trials."
    )

    # Test that expected block data has been added to the nwbfile
    assert "block" in nwbfile.intervals
    expected_block_columns = {
        'start_time', 'stop_time', 'epoch', 'block', 
        'maze_configuration', 'pA', 'pB', 'pC', 'num_trials', 'task_type'}
    assert set(nwbfile.intervals["block"].colnames) == expected_block_columns, (
        f"Block columns {nwbfile.intervals['block'].colnames} "
        f"did not match expected columns {expected_block_columns}"
    )
    assert len(nwbfile.intervals["block"].start_time) == 3 # there are 3 blocks in this session
    # Check block data
    assert nwbfile.intervals["block"].num_trials[0] == 68
    assert nwbfile.intervals["block"].num_trials[1] == 66
    assert nwbfile.intervals["block"].num_trials[2] == 54
    assert nwbfile.intervals["block"].maze_configuration[0] == "11,12,14,15,18,22,29,31,35,45"
    assert nwbfile.intervals["block"].maze_configuration[1] == "11,12,14,15,18,20,22,29,31,45"
    assert nwbfile.intervals["block"].maze_configuration[2] == "11,12,14,15,20,29,31,36,45"
    assert nwbfile.intervals["block"].pA[0] == 10
    assert nwbfile.intervals["block"].pB[0] == 50
    assert nwbfile.intervals["block"].pC[0] == 90

    # Test that expected trial data has been added to the nwbfile
    assert nwbfile.trials is not None
    expected_trial_columns = {
        'start_time', 'stop_time', 'epoch', 'block', 'trial_within_block', 
        'trial_within_epoch', 'start_port', 'end_port', 'reward', 
        'opto_condition', 'duration', 'poke_in', 'poke_out'}
    assert set(nwbfile.trials.colnames) == expected_trial_columns, (
        f"Trial columns {nwbfile.trials.colnames} "
        f"did not match expected columns {expected_trial_columns}"
    )
    assert len(nwbfile.trials.start_time) == 188 # there are 188 trials in this session

    # Check that block boundaries are correct: 
    # block 1 = trials 1-68, block 2 = 69-134, block 3 = 135-188
    trial_blocks = nwbfile.trials["block"].data[:]
    assert trial_blocks[67] == 1  # trial 68 is block 1
    assert trial_blocks[68] == 2  # trial 69 is block 2
    assert trial_blocks[133] == 2  # trial 134 is block 2
    assert trial_blocks[134] == 3  # trial 135 is block 3


def test_barrier_shift_trial_counts(dummy_logger):
    """
    Test that barrier_shift_trial_counts in metadata correctly re-assigns block boundaries.

    The test session has blocks with 68/66/54 trials (188 trials total).
    We specify [69, 64, 55] as the user-specified trial counts per block (must sum to 188).
    """
    metadata = {}
    metadata["behavior"] = {}
    metadata["behavior"]["arduino_text_file_path"] = "tests/test_data/behavior/arduinoraw0.txt"
    metadata["behavior"]["arduino_timestamps_file_path"] = "tests/test_data/behavior/ArduinoStamps0.csv"
    metadata["behavior"]["maze_configuration_file_path"] = "tests/test_data/behavior/barriers.txt"
    metadata["behavior"]["barrier_shift_trial_counts"] = [69, 64, 55]

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_behavior(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Total trials should be unchanged
    assert len(nwbfile.trials.start_time) == 188

    # Actual block counts: 69 / 64 / 55
    assert nwbfile.intervals["block"].num_trials[0] == 69   # trials 1-69
    assert nwbfile.intervals["block"].num_trials[1] == 64   # trials 70-133
    assert nwbfile.intervals["block"].num_trials[2] == 55   # trials 134-188

    # Maze configurations should still be in the original block order from the barriers file
    assert nwbfile.intervals["block"].maze_configuration[0] == "11,12,14,15,18,22,29,31,35,45"
    assert nwbfile.intervals["block"].maze_configuration[1] == "11,12,14,15,18,20,22,29,31,45"
    assert nwbfile.intervals["block"].maze_configuration[2] == "11,12,14,15,20,29,31,36,45"

    # Reward probabilities should be the same for all blocks (barrier change session)
    for i in range(3):
        assert nwbfile.intervals["block"].pA[i] == 10
        assert nwbfile.intervals["block"].pB[i] == 50
        assert nwbfile.intervals["block"].pC[i] == 90

    # Check that block boundaries are correct: 
    # block 1 = trials 1-69, block 2 = 70-133, block 3 = 134-188
    trial_blocks = nwbfile.trials["block"].data[:]
    assert trial_blocks[68] == 1   # trial 69 is block 1
    assert trial_blocks[69] == 2   # trial 70 is block 2
    assert trial_blocks[132] == 2  # trial 133 is block 2
    assert trial_blocks[133] == 3  # trial 134 is block 3


def test_barrier_shift_wrong_trial_sum(dummy_logger):
    """If barrier_shift_trial_counts doesn't sum to total trials, raise ValueError."""
    metadata = {}
    metadata["behavior"] = {}
    metadata["behavior"]["arduino_text_file_path"] = "tests/test_data/behavior/arduinoraw0.txt"
    metadata["behavior"]["arduino_timestamps_file_path"] = "tests/test_data/behavior/ArduinoStamps0.csv"
    metadata["behavior"]["maze_configuration_file_path"] = "tests/test_data/behavior/barriers.txt"
    metadata["behavior"]["barrier_shift_trial_counts"] = [70, 64, 55]  # sums to 189, not 188

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Should error for wrong number of total trials
    with pytest.raises(ValueError):
        add_behavior(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)


def test_barrier_shift_wrong_maze_count(dummy_logger, tmp_path):
    """If maze config file has the wrong number of mazes for the user-specified block count, raise ValueError."""

    metadata = {}
    metadata["behavior"] = {}
    metadata["behavior"]["arduino_text_file_path"] = "tests/test_data/behavior/arduinoraw0.txt"
    metadata["behavior"]["arduino_timestamps_file_path"] = "tests/test_data/behavior/ArduinoStamps0.csv"
    metadata["behavior"]["maze_configuration_file_path"] = "tests/test_data/behavior/barriers.txt"
    metadata["behavior"]["barrier_shift_trial_counts"] = [69, 64, 30, 25]  # expects 4 mazes, file has 3

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Should error for wrong number of maze configs
    with pytest.raises(ValueError):
        add_behavior(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)