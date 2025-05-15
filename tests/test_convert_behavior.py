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
    assert nwbfile.intervals["block"].maze_configuration[0] == "11,12,14,15,18,22,29,31,35,45", (
        f"Maze configuration {nwbfile.intervals['block'].maze_configuration[0]} "
        f"did not match expected '11,12,14,15,18,22,29,31,35,45'"
    )
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
