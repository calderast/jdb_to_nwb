import csv
import re
import itertools
import warnings
import numpy as np
from pathlib import Path
from pynwb import NWBFile
from hdmf.common.table import DynamicTable, VectorData
from ndx_franklab_novela import AssociatedFiles
from .timestamps_alignment import trim_sync_pulses, handle_timestamps_reset
from .plotting.plot_behavior import plot_maze_configurations, plot_trial_time_histogram


def load_maze_configurations(maze_configuration_file_path: Path):
    """Load maze configurations from the text file into a list of sets."""
    maze_sequence = []
    try:
        # Read from the text file and append each set to the list
        with open(maze_configuration_file_path, "r") as maze_configuration_file:
            for line in maze_configuration_file:
                # Ignore any empty lines or lines starting with '#'
                if line.strip().startswith("#") or not line.strip():
                    continue
                # Strip any whitespace and split by commas to create a set of ints
                maze = set(map(int, line.strip().split(",")))
                maze_sequence.append(maze)
            return maze_sequence
    except Exception as e:
        warnings.warn(f"Could not load maze configurations from {maze_configuration_file_path}: {e}")
        return []


def adjust_arduino_timestamps(arduino_timestamps: list, logger):
    """
    Convert arduino timestamps to seconds and make photometry start at time 0.
    If needed, detect and handle timestamps resetting to 0 when recording passes 12:00pm.
    """
    # Check for and handle potential timestamps reset
    arduino_timestamps = handle_timestamps_reset(timestamps=arduino_timestamps, logger=logger)

    # The photometry start time is always the second timestamp in arduino_timestamps
    photometry_start_in_arduino_ms = arduino_timestamps[1]

    # Adjust all arduino timestamps so the photometry starts at time zero
    arduino_timestamps = np.subtract(arduino_timestamps, photometry_start_in_arduino_ms)

    # Convert timestamps from ms to seconds
    arduino_timestamps = arduino_timestamps / 1000

    return arduino_timestamps, photometry_start_in_arduino_ms


def determine_session_type(block_data: list):
    """Determine the session type ("barrier change" or "probability change") based on block data."""

    # This case is rare/hopefully nonexistent - we always expect to have more than one block per session
    if len(block_data) == 1:
        return "single block"

    # Get the reward probabilities at each port for each block in the session
    reward_probabilities = []
    for block in block_data:
        reward_probabilities.append([block["pA"], block["pB"], block["pC"]])

    # If the reward contingencies change with each block, this is a probability change session
    if reward_probabilities[0] != reward_probabilities[1]:
        return "probability change"
    # Otherwise, this must be a barrier change session
    else:
        return "barrier change"


def parse_arduino_text(arduino_text: list, arduino_timestamps: list, logger):
    """
    Parse the arduino text output and corresponding timestamps into lists
    containing information about trials and blocks in this session.

    We have some interesting time alignment issues here between trials and blocks.
    Beam breaks span multiple indices in the text file (the beam stays broken the
    entire time the rat's head is in the port, and will continuously print "beam break").
    Trials end when the rat removes their head from the reward port (end of beam break)
    In the arduino text file, a new block is triggered by the beam break of the last trial in a block.
    However, we don't want to start the block at the exact moment the "Block" text appears
    because this is still in the middle of the beam break. We want the new block to start
    once the beam break has ended, so it aligns with the end of the last trial in the block.

    Note that for the first trial and block, the start time is set to 3s before the first beam break 
    (or 0, to avoid negative start times) to exclude pre-maze times.
    """
    trial_data = []
    block_data = []
    current_trial = {}
    previous_trial = {}
    current_block = {}
    previous_block = {}
    trial_within_block = 1  # tracks the trials in each block
    trial_within_session = 1  # tracks the total trials in this session

    for i, line in enumerate(arduino_text):
        # Detect block starts
        block_start = re.match(r"Block: (\d+)", line)
        if block_start:
            # Get block metadata: reward probabilities are always on the next 3 lines
            current_block = {
                "block": int(block_start.group(1)) + 1,
                "pA": int(arduino_text[i + 1].split(":")[1].strip()),
                "pB": int(arduino_text[i + 2].split(":")[1].strip()),
                "pC": int(arduino_text[i + 3].split(":")[1].strip()),
                "start_time": None,  # Set this once we find the end of the beam break that triggered this block
                "end_time": None,  # Set this as the start time of the next block
                "num_trials": None,  # Set this once we find the last trial in this block
            }
            logger.debug(f"Found new block: {current_block}")

            # If this is the first block, wait to set the start time. It will be set to 3s before the first beam break
            if not previous_block:
                logger.debug("This is the first block.")
                previous_block = current_block

        # Detect beam breaks
        beam_break = re.match(r"beam break at port (\w)", line)
        if beam_break:
            port = beam_break.group(1)

            # If this is the start of a beam break at a new port, create the trial ending at this port
            if not current_trial and port != previous_trial.get("end_port", None):
                # Set the start time of the first trial to 3s before the first beam break to exclude pre-maze times
                if not previous_trial:
                    first_trial_start_time = max(float(arduino_timestamps[i]) - 3, 0) # don't allow start time < 0
                    logger.debug("This is the first trial. Setting start time to 3s before the beam break start "
                                 "(or 0 to avoid a negative start time).")
                    logger.debug(f"First beam break start: {float(arduino_timestamps[i])}")
                    logger.debug(f"Start of first trial/block: {first_trial_start_time}")
                    current_trial["start_time"] = first_trial_start_time
                    current_block["start_time"] = first_trial_start_time
                # Subsequent trials start at previous trial end
                else:
                    current_trial["start_time"] = float(previous_trial.get("end_time"))
                current_trial["beam_break_start"] = float(arduino_timestamps[i])
                current_trial["start_port"] = previous_trial.get("end_port", "None")
                current_trial["end_port"] = port
                current_trial["trial_within_block"] = trial_within_block
                current_trial["trial_within_session"] = trial_within_session
                current_trial["block"] = current_block.get("block")

                # The line immediately following the beam break start contains reward information
                current_trial["reward"] = (
                    1
                    if re.search(rf"rwd delivered at port {port}", arduino_text[i + 1])
                    else 0 if re.search(rf"no Reward port {port}", arduino_text[i + 1]) else None
                )

            # If we are in the middle of a beam break, update the end times until we reach the end of the beam break
            if current_trial:

                # If we are in the middle of a trial that ends at a different port, end that trial first!
                # This only happens during ultra-short beam breaks, e.g. this arduino snippet below.
                # See Github issue https://github.com/calderast/jdb_to_nwb/issues/163
                # ...
                # beam break at port A; 134084 trial 7 of 61
                # beam break at port A; 134130 trial 7 of 61
                # beam break at port C; 142826 trial 7 of 61 <-- BEAM BREAK
                # no Reward port C; trial 7 of 61            <-- REWARD INFO FOR THIS TRIAL
                # beam break at port B; 148678 trial 8 of 61 <-- NEXT BEAM BREAK IS ALREADY THE NEXT TRIAL! 
                # rwd delivered at port B; 148731
                # beam break at port B; 148732 trial 9 of 61
                # beam break at port B; 148767 trial 9 of 61
                # ...
                if port != current_trial["end_port"]:
                    trial_data.append(current_trial)
                    logger.debug(f"Short beam break! Beam break is over. Adding trial {current_trial}")
                    previous_trial = current_trial
                    current_trial = {}
                    trial_within_session += 1
                    trial_within_block += 1

                    # Start new trial at this port
                    current_trial = {
                        "start_time": float(previous_trial["end_time"]),
                        "beam_break_start": float(arduino_timestamps[i]),
                        "start_port": previous_trial.get("end_port", "None"),
                        "end_port": port,
                        "trial_within_block": trial_within_block,
                        "trial_within_session": trial_within_session,
                        "block": current_block.get("block"),
                    }
                    current_trial["reward"] = (
                        1
                        if re.search(rf"rwd delivered at port {port}", arduino_text[i + 1])
                        else 0 if re.search(rf"no Reward port {port}", arduino_text[i + 1]) else None
                    )

                current_trial["beam_break_end"] = float(arduino_timestamps[i])
                current_trial["end_time"] = float(arduino_timestamps[i])

                # If the next timestamp is far enough away (>1s), the beam break is over, so end the trial.
                # Note this threshold should be at the very least 200ms because with arduino jitter, 
                # if the the threshold is too short (e.g. 100ms), the beam break ends immediately and 
                # it sets beam_break_start = beam_break_end. This ends up causing issues for us downstream 
                # (with proper block/trial alignment). 1 second is also a good threshold to properly capture the 
                # time the animal spends at the port (for SWR, etc.) and is the same threshold as Frank Lab.
                beam_break_time_thresh = 1 # seconds
                if (i < len(arduino_timestamps) - 1) and (
                    arduino_timestamps[i + 1] - current_trial["beam_break_end"]
                ) >= beam_break_time_thresh:
                    trial_data.append(current_trial)
                    logger.debug(f"Beam break is over. Adding trial {current_trial}")
                    # Reset trial data
                    previous_trial = current_trial
                    current_trial = {}
                    trial_within_session += 1
                    trial_within_block += 1
                    # If this trial triggered a new block, the start time of the block = the end of the beam break
                    if not current_block["start_time"]:
                        current_block["start_time"] = float(arduino_timestamps[i])
                        previous_block["end_time"] = float(arduino_timestamps[i])
                        previous_block["num_trials"] = previous_trial.get("trial_within_block")
                        logger.debug("This trial triggered a new block.")
                        logger.debug(f"Adding previous block: {previous_block}")
                        block_data.append(previous_block)
                        previous_block = current_block
                        trial_within_block = 1
                # If we have reached the last timestamp of the file while in the middle of a beam break,
                # make the trial end time the last timestamp
                elif i == len(arduino_timestamps)-1:
                    logger.debug("Reached the last timestamp of the file in the middle of a beam break.")
                    # Add the last trial
                    trial_data.append(current_trial)
                    logger.debug(f"Adding final trial: {current_trial}")
                    # Reset trial data (make current_trial = None) so we don't add it twice
                    previous_trial = current_trial
                    current_trial = {}

                    # We sometimes have the case where the last trial in the session triggers a new block
                    # (if we choose to stop the recording after the rat has completed a full block)
                    # This new block does not actually have any trials, so don't add it.
                    # But make sure to add the previous block
                    if not current_block["start_time"]:
                        # Add the previous block (the block that ends at the end of this session)
                        previous_block["end_time"] = float(arduino_timestamps[i])
                        previous_block["num_trials"] = previous_trial.get("trial_within_block")
                        block_data.append(previous_block)
                        logger.debug(f"Adding final block: {previous_block}")
                        # Make the current block (the new block that this trial started) empty
                        # so we don't add it. 
                        current_block = {}
    
    # Append the last trial if it exists
    if current_trial:
        trial_data.append(current_trial)
        logger.debug(f"Adding last trial: {current_trial}")
        previous_trial = current_trial
        
    # If the last trial in the session triggered a new block,
    # but there are not actually any trials in that block, do not add it.
    # This is the case if the start time of our block is the end time of our last trial.
    if previous_trial.get("end_time") == current_block.get("start_time"):
        current_block = {}

    # Append the last block if it exists
    if current_block:
        current_block["end_time"] = float(previous_trial["end_time"])
        current_block["num_trials"] = previous_trial.get("trial_within_block")
        block_data.append(current_block)
        logger.debug(f"Adding last block: {current_block}")

    return trial_data, block_data


def align_data_to_visits(trial_data, block_data, metadata, logger):
    """
    Align arduino trial and block data to ground truth visit times.
    Ground truth is photometry if it exists, otherwise ephys. 
    """

    # Get ground truth visit times (ground truth is photometry if it exists, otherwise ephys)
    ground_truth_visit_times = metadata.get("ground_truth_visit_times")
    ground_truth_time_source = metadata.get("ground_truth_time_source")

    # If we have no ground truth visits to align to, keep trial and block data as-is
    if ground_truth_visit_times is None:
        logger.info("No photometry or ephys visits to align to, keeping trial and block timestamps as-is.")
        return trial_data, block_data

    logger.info("Aligning trial and block data to ground truth port visit times...")
    logger.info(f"Using {ground_truth_time_source} time as ground truth.")
    logger.info(f"There are {len(trial_data)} trials and {len(ground_truth_visit_times)} visit times")

    # If we have more ground truth visit times (pulses from photometry/ephys) than trials,
    # trim the extra pulses so we can do alignment.
    if len(ground_truth_visit_times) > len(trial_data):
        logger.info(f"We have more {ground_truth_time_source} pulses than arduino visits. "
                    "Trimming pulses to match the number of visits to do time alignment.")
        # Port visits in arduino time are the beam break start for each trial
        arduino_visits = [trial['beam_break_start'] for trial in trial_data]
        ground_truth_visit_times, arduino_visits = trim_sync_pulses(ground_truth_visit_times, arduino_visits, logger)

    # We should never have more trials than photometry/ephys visits.
    # If we do, error so we can figure out why this happened and handle it accordingly.
    elif len(trial_data) > len(ground_truth_visit_times):
        logger.critical(f"Found more trials recorded by arduino ({len(trial_data)}) "
                        f"than {ground_truth_time_source} visits ({len(ground_truth_visit_times)})!!!")
        logger.critical("This should never happen!!! Skipping alignment of trial/block data.")
        return trial_data, block_data

    # Now that we have the correct number of ground truth visit times, replace arduino times with ground truth times
    for trial, visit_time in zip(trial_data, ground_truth_visit_times):
        time_diff = visit_time - trial['beam_break_start']
        logger.debug(f"Replacing arduino beam break time {trial['beam_break_start']} with {visit_time}"
                     f" (diff={time_diff:.3f})")
        # Update beam_break_start to be the ground truth port visit time
        trial['beam_break_start'] = visit_time
        # Set beam_break_end based on the difference between the original arduino time and the ground truth time
        trial['beam_break_end'] = trial['beam_break_end'] + time_diff
        trial['end_time'] = trial['beam_break_end']

    # The new start time of each trial is the updated end time of the previous trial
    for prev_trial, trial in zip(trial_data, trial_data[1:]):
        trial['start_time'] = prev_trial['end_time']

    # Update block start times (block start time = start time of the first trial in the block)
    for block in block_data:
        block["start_time"] = next(trial["start_time"] for trial in trial_data if trial["block"] == block["block"])

    # Update block end times (block end time = start time of the next block)
    for block_num, block in enumerate(block_data[:-1]):
        block["end_time"] = block_data[block_num+1]["start_time"]
    # The end time of the last block is the end time of the last trial
    block_data[-1]["end_time"] = trial_data[-1]["end_time"]

    return trial_data, block_data


def validate_trial_and_block_data(trial_data: list, block_data: list, logger):
    """Run basic tests to check that trial and block data is valid."""

    # The number of the last trial/block must match the number of trials/blocks
    assert len(trial_data) == trial_data[-1].get("trial_within_session"), (
        f"Found {len(trial_data)} trials, but the last trial number is {trial_data[-1].get("trial_within_session")}"
    )
    logger.debug(f"The last trial number {trial_data[-1].get("trial_within_session")} "
                 f"matches the total number of trials {len(trial_data)}")
    
    assert len(block_data) == block_data[-1].get("block"), (
        f"Found {len(block_data)} blocks, but the last block number is {block_data[-1].get("block")}"
    )
    logger.debug(f"The last block number {block_data[-1].get("block")} "
                 f"matches the total number of blocks {len(block_data)}")

    # All trial numbers must be unique and match the range 1 to [num trials in session]
    trial_numbers = {trial.get("trial_within_session") for trial in trial_data}
    assert trial_numbers == set(range(1, len(trial_data) + 1)), (
        "Trial numbers are not unique and/or do not match the range 1 to [num trials in session]\n"
        f"Expected trial numbers: {set(range(1, len(trial_data) + 1))}\n"
        f"Got trial numbers {trial_numbers}"
    )
    logger.debug(f"All trial numbers are unique and match the range 1 to {len(trial_data)}")

    # All block numbers must be unique and match the range 1 to [num blocks in session]
    block_numbers = {block.get("block") for block in block_data}
    assert block_numbers == set(range(1, len(block_data) + 1)), (
        "Block numbers are not unique and/or do not match the range 1 to [num blocks in session]\n"
        f"Expected block numbers: {set(range(1, len(block_data) + 1))}\n"
        f"Got block numbers {block_numbers}"
    )
    logger.debug(f"All block numbers are unique and match the range 1 to {len(block_data)}")
    
    # The end time of each trial must be the start time of the next trial
    for t1, t2 in zip(trial_data, trial_data[1:]):
        assert t1.get("end_time") == t2.get("start_time"), (
            f"Trial {t1.get('trial_within_session')} end_time {t1.get('end_time')} "
            f"does not match trial {t2.get('trial_within_session')} start_time {t2.get('start_time')}"
        )
    logger.debug("The end time of each trial matches the start time of the next trial")

    # There must be a legitimate reward value (1 or 0) for all trials (instead of default None)
    assert all(trial.get("reward") in {0, 1} for trial in trial_data), (
        "Not all trials have a legitimate reward value (1 or 0)"
    )
    logger.debug("All trials have a legitimate reward value (1 or 0)")

    # There must be a legitimate p(reward) value for each block at ports A, B, and C
    assert all(0 <= block.get(key) <= 100 for block in block_data for key in ["pA", "pB", "pC"]), (
        "Not all blocks have a legitimate p(reward) value for pA, pB, or pC (must be in range 0-100)"
    )
    logger.debug("All blocks have a legitimate value for pA, pB, or pC (value in range 0-100)")

    # There must be a not-null maze_configuration for each block
    assert all(block.get("maze_configuration") not in ([], None) for block in block_data), (
        "Not all blocks have a non-null maze configuration"
    )
    logger.debug("All blocks have a non-null maze configuration")

    # There must be a valid task type for each block
    valid_task_types = ["probability change", "barrier change", "single block"]
    assert all(block.get("task_type") in valid_task_types for block in block_data), (
        f"Not all blocks have a valid task_type. Must be one of: {valid_task_types}"
    )
    # The task type must be the same for all blocks
    assert len({block["task_type"] for block in block_data}) == 1, (
        f"All blocks must have the same task_type. Must be one of: {valid_task_types}"
    )
    logger.debug(f"All blocks have the same task_type ({block_data[0]["task_type"]})")

    # In a probability change session, reward probabilities vary and maze configs do not
    if block_data[0]["task_type"] == "probability change":
        # All maze configs should be the same
        assert len({block["maze_configuration"] for block in block_data}) == 1, (
            "All maze configurations must be the same for a probability change session"
        )
        # Reward probabilities should vary.
        # We check for any changes instead of all because of cases like where we have only 2 
        # blocks and only 2 of the probabilities change (e.g. pA and pB switch, pC stays the same)
        assert any([
            len({block["pA"] for block in block_data}) > 1,
            len({block["pB"] for block in block_data}) > 1,
            len({block["pC"] for block in block_data}) > 1
        ]), "pA, pB, or pC must vary in a probability change session"
        logger.debug("All maze configurations are the same and all reward probabilities vary across blocks")

    # In a barrier change session, maze configs vary and reward probabilities do not
    elif block_data[0]["task_type"] == "barrier change":
        # Maze configurations should be different for each block
        # We choose to log at ERROR level instead of failing an assert because we have 
        # at least one barrier change session (cough cough Jose) where a maze configuration repeats
        unique_mazes = {block["maze_configuration"] for block in block_data}
        if len(unique_mazes) != len(block_data):
            logger.error("Maze configurations must differ for each block in a barrier change session!")
            logger.error(f"Got {len(unique_mazes)} unique maze configs for {len(block_data)} blocks!")
        else:
            logger.debug(f"Found {len(unique_mazes)} unique maze configs for {len(block_data)} blocks.")
        # All reward probabilities should be the same for all blocks
        assert len({block["pA"] for block in block_data}) == 1, "pA should not vary in a barrier change session"
        assert len({block["pB"] for block in block_data}) == 1, "pB should not vary in a barrier change session"
        assert len({block["pC"] for block in block_data}) == 1, "pC should not vary in a barrier change session"
        logger.debug("All reward probabilities stay the same across blocks")

    summed_trials = 0
    # Check trials within each block
    for block in block_data:
        trials_in_block = [trial for trial in trial_data if trial.get("block") == block.get("block")]
        trial_numbers = [trial.get("trial_within_block") for trial in trials_in_block]

        # All trial numbers in the block must be unique and match the range 1 to [num trials in block]
        num_trials_expected = block.get("num_trials")
        assert len(set(trial_numbers)) == len(trial_numbers) == num_trials_expected, (
            f"Expected {num_trials_expected} trials in this block from the block data, got {len(trial_numbers)}"
        )
        assert set(trial_numbers) == set(range(1, num_trials_expected + 1)), (
            "Trial numbers within this block are not unique and/or do not match the range 1 to [num trials in block]\n"
            f"Expected trial numbers: {set(range(1, num_trials_expected + 1))}\n"
            f"Got trial numbers {set(trial_numbers)}"
        )
        logger.debug(f"All trial numbers in this block are unique and match the range 1 to {num_trials_expected}")

        # Check time alignment between trials and blocks
        first_trial = min(trials_in_block, key=lambda t: t.get("trial_within_block"))
        last_trial = max(trials_in_block, key=lambda t: t.get("trial_within_block"))
        block_start = block.get("start_time")
        block_end = block.get("end_time")

        # The start of the first trial and end of the last trial must exactly match the start and end of the block
        assert first_trial.get("start_time") == block_start, (
            f"First trial start {first_trial.get('start_time')} does not match block start {block_start}"
        )
        logger.debug(f"The start time of the first trial in the block matches the block start {block_start}")
        assert last_trial.get("end_time") == block_end, (
            f"Last trial end {last_trial.get('end_time')} does not match block end {block_end}"
        )
        logger.debug(f"The end time of the last trial in the block matches the block end {block_end}")

        # The start and end time of each trial in the block must be within the block time bounds
        for trial in trials_in_block:
            assert block_start <= trial.get("start_time") <= block_end, (
                f"Trial {trial.get('trial_within_block')} start_time {trial.get('start_time')} "
                f"is outside block bounds ({block_start} to {block_end})"
            )
            assert block_start <= trial.get("end_time") <= block_end, (
                f"Trial {trial.get('trial_within_block')} end_time {trial.get('end_time')} "
                f"is outside block bounds ({block_start} to {block_end})"
            )
        logger.debug("The start and end time of each trial in the block is within the block time bounds")

        summed_trials += block.get("num_trials")

    # The summed number of trials in each block must match the total number of trials
    assert summed_trials == len(trial_data), (
        f"The summed number of trials in each block {summed_trials} "
        f"does not match the total number of trials {len(trial_data)}"
    )
    logger.debug(f"The number of trials in each block sums to the total number of trials {len(trial_data)}")


def add_behavior(nwbfile: NWBFile, metadata: dict, logger, fig_dir=None):
    """Add trial and block data to the nwbfile"""
    print("Adding behavior...")
    logger.info("Adding behavior...")

    # Get file paths for behavior from metadata file
    arduino_text_file_path = metadata["behavior"]["arduino_text_file_path"]
    arduino_timestamps_file_path = metadata["behavior"]["arduino_timestamps_file_path"]
    maze_configuration_file_path = metadata["behavior"]["maze_configuration_file_path"]

    # Read arduino text file into a list of strings to use for parsing
    with open(arduino_text_file_path, "r") as arduino_text_file:
        arduino_text = arduino_text_file.read().splitlines()

    # Read arduino timestamps from the CSV into a list of floats to use for parsing
    with open(arduino_timestamps_file_path, "r") as arduino_timestamps_file:
        arduino_timestamps = list(map(float, itertools.chain.from_iterable(csv.reader(arduino_timestamps_file))))

    # Make sure arduino text and arduino timestamps are the same length
    if len(arduino_text) != len(arduino_timestamps):
        logger.error(
            f"Mismatch in list lengths: arduino text has {len(arduino_text)} entries, "
            f"but timestamps have {len(arduino_timestamps)} entries."
        )
        raise ValueError(
            f"Mismatch in list lengths: arduino text has {len(arduino_text)} entries, "
            f"but timestamps have {len(arduino_timestamps)} entries."
        )

    # Convert arduino timestamps to seconds and make photometry start at time 0
    # If needed, handle reset to 0 that happens when the recording passes 12:00pm
    arduino_timestamps, photometry_start_in_arduino_time = adjust_arduino_timestamps(arduino_timestamps, logger)
    logger.debug(f"Photometry start in arduino time: {photometry_start_in_arduino_time}")

    # Read through the arduino text and timestamps to get trial and block data
    logger.debug("Parsing arduino text file...")
    trial_data, block_data = parse_arduino_text(arduino_text, arduino_timestamps, logger)
    logger.info(f"There are {len(block_data)} blocks and {len(trial_data)} trials")

    # Use block data to determine if this is a probability change or barrier change session
    session_type = determine_session_type(block_data)
    for block in block_data:
        block["task_type"] = session_type
    logger.info(f"This is a {session_type} session")

    # Load maze configurations for each block from the maze configuration file
    maze_configurations = load_maze_configurations(maze_configuration_file_path)
    logger.debug(f"Found {len(maze_configurations)} maze(s) in the maze configuration file")

    # Make sure the number of blocks matches the number of loaded maze configurations
    if len(block_data) != len(maze_configurations):
        # If this is a probability change session, we may have a single maze configuration
        # to be used for all blocks. If so, duplicate it so we have one maze per block.
        if len(maze_configurations) == 1 and session_type == "probability change":
            maze_configurations = maze_configurations * len(block_data)
        else:
            logger.error(
                f"There are {len(block_data)} blocks in the arduino text file, "
                f"but {len(maze_configurations)} mazes in the maze configuration file. "
            )
            raise ValueError(
                f"There are {len(block_data)} blocks in the arduino text file, "
                f"but {len(maze_configurations)} mazes in the maze configuration file. "
                "There should be exactly one maze configuration per block, "
                "or a single maze configuration if this is a probability change session."
            )

    # Convert each maze config from a set to a sorted, comma separated string 
    # for compatibility with NWB and spyglass
    def barrier_set_to_string(set):
        return ",".join(map(str, sorted(set)))

    # Add the maze configuration to the metadata for each block
    for i, (block, maze) in enumerate(zip(block_data, maze_configurations), start=1):
        logger.debug(f"Block {i} maze: {barrier_set_to_string(maze)}")
        block["maze_configuration"] = barrier_set_to_string(maze)

    # Plot maze configurations for each block
    plot_maze_configurations(block_data=block_data, fig_dir=fig_dir)

    # Save original arduino visit times for alignment before we re-align to photometry/ephys  
    arduino_visit_times = [trial['beam_break_start'] for trial in trial_data]

    # Align visit times to photometry/ephys
    trial_data, block_data = align_data_to_visits(trial_data, block_data, metadata, logger)

    # Do some checks on trial and block data before adding to the NWB
    logger.debug("Validating trial and block data...")
    validate_trial_and_block_data(trial_data, block_data, logger)
    logger.debug("All validation checks passed.")

    # Plot histogram of trial times
    plot_trial_time_histogram(trial_data=trial_data, fig_dir=fig_dir)

    # Add columns for block data to the NWB file
    block_table = nwbfile.create_time_intervals(
        name="block",
        description="The block within a session. "
        "Each block is defined by a maze configuration and set of reward probabilities.",
    )
    block_table.add_column(name="epoch", 
        description="The epoch (session) this block is in, for consistency with Frank Lab. "
        "For Berke Lab, there is currently only one session per day.")
    block_table.add_column(name="block", description="The block number within the session")
    block_table.add_column(
        name="maze_configuration",
        description="The maze configuration for each block, "
        "defined by the set of hexes in the maze where barriers are placed.",
    )
    block_table.add_column(name="pA", description="The probability of reward at port A")
    block_table.add_column(name="pB", description="The probability of reward at port B")
    block_table.add_column(name="pC", description="The probability of reward at port C")
    block_table.add_column(name="num_trials", description="The number of trials in this block")
    block_table.add_column(name="task_type", description="The session type ('barrier change' or 'probability change'")

    # Add columns for trial data to the NWB file
    nwbfile.add_trial_column(name="epoch", 
        description="The epoch (session) this trial is in, for consistency with Frank Lab. "
        "For Berke Lab, there is currently only one session per day.")
    nwbfile.add_trial_column(name="block", description="The block this trial is in")
    nwbfile.add_trial_column(name="trial_within_block", description="The trial number within the block")
    nwbfile.add_trial_column(name="trial_within_epoch", description="The trial number within the epoch (session)")
    nwbfile.add_trial_column(name="start_port", description="The reward port the rat started at (A, B, or C)")
    nwbfile.add_trial_column(name="end_port", description="The reward port the rat ended at (A, B, or C)")
    nwbfile.add_trial_column(name="reward", description="If the rat got a reward at the port (1 or 0)")
    nwbfile.add_trial_column(name="opto_condition", description="Description of the opto condition, if any")
    nwbfile.add_trial_column(name="duration", description="The duration of the trial")
    nwbfile.add_trial_column(name="poke_in", description="The time the rat entered the reward port")
    nwbfile.add_trial_column(name="poke_out", description="The time the rat exited the reward port")

    # Overwrite session description with the session type, number of blocks, and number of trials
    nwbfile.fields["session_description"] = (
        f"{session_type} session for the hex maze task with {len(block_data)} blocks and {len(trial_data)} trials."
    )

    # Add each block to the block table in the NWB
    logger.debug("Adding each block to the block table in the NWB")
    for block in block_data:
        block_table.add_row(
            epoch=0, # Berke Lab only has one epoch (session) per day
            block=block["block"],
            maze_configuration=block["maze_configuration"],
            pA=block["pA"],
            pB=block["pB"],
            pC=block["pC"],
            num_trials=block["num_trials"],
            task_type=block["task_type"],
            start_time=block["start_time"],
            stop_time=block["end_time"],
        )

    # Add each trial to the NWB
    logger.debug("Adding each trial to the trial table in the NWB")
    nwbfile.intervals.add(nwbfile.trials)
    for trial in trial_data:
        nwbfile.add_trial(
            epoch=0, # Berke Lab only has one epoch (session) per day
            block=trial["block"],
            trial_within_block=trial["trial_within_block"],
            trial_within_epoch=trial["trial_within_session"],
            start_port=trial["start_port"],
            end_port=trial["end_port"],
            reward=trial["reward"],
            opto_condition="None", # For now, Berke Lab has no opto
            duration=(trial["end_time"]-trial["start_time"]),
            poke_in=trial["beam_break_start"],
            poke_out=trial["beam_break_end"],
            start_time=trial["start_time"],
            stop_time=trial["end_time"],
        )

    # Add a single epoch to the NWB for this session
    session_start = block_data[0]["start_time"]
    session_end = block_data[-1]["end_time"]
    epoch_tag = "00_r1" # This is epoch 0 and run session 1
    nwbfile.add_epoch(start_time=session_start, stop_time=session_end, tags=epoch_tag)

    # Add tasks processing module for compatibility with Spyglass
    # Many of these fields are repetitive but exist to match Frank Lab
    nwbfile.create_processing_module(
        name="tasks", description="Contains all tasks information"
    )
    task_name = VectorData(
        name="task_name",
        description="the name of the task",
        data=["Hex maze"],
    )
    task_description = VectorData(
        name="task_description",
        description="a description of the task",
        data=["Hex maze"],
    )
    task_epochs = VectorData(
        name="task_epochs",
        description="the temporal epochs where the animal was exposed to this task",
        data=[[0]],
    )
    task_environment = VectorData(
        name="task_environment",
        description="the environment in which the animal performed the task",
        data=["hexmaze"],
    )
    camera_id = VectorData(
        name="camera_id",
        description="the ID number of the camera used for video",
        data=[[1]],
    )
    task = DynamicTable(
        name="task_0",
        description="",
        columns=[
            task_name,
            task_description,
            task_epochs,
            task_environment,
            camera_id,
        ],
    )
    nwbfile.processing["tasks"].add(task)

    # Save the raw arduino text and timestamps as strings to be used to create AssociatedFiles objects
    logger.debug("Saving the arduino text file and arduino timestamps file as AssociatedFiles objects")
    with open(arduino_text_file_path, "r") as arduino_text_file:
        raw_arduino_text = arduino_text_file.read()
    with open(arduino_timestamps_file_path, "r") as arduino_timestamps_file:
        raw_arduino_timestamps = arduino_timestamps_file.read()

    raw_arduino_text_file = AssociatedFiles(
        name="arduino_text",
        description="Raw arduino text",
        content=raw_arduino_text,
        task_epochs="0",  # Berke Lab only has one epoch (session) per day
    )
    raw_arduino_timestamps_file = AssociatedFiles(
        name="arduino_timestamps",
        description="Raw arduino timestamps",
        content=raw_arduino_timestamps,
        task_epochs="0",  # Berke Lab only has one epoch (session) per day
    )

    # If it doesn't exist already, make a processing module for associated files
    if "associated_files" not in nwbfile.processing:
        logger.debug("Creating nwb processing module for associated files")
        nwbfile.create_processing_module(name="associated_files", description="Contains all associated files")

    # Add arduino text and timestamps to the NWB as associated files
    nwbfile.processing["associated_files"].add(raw_arduino_text_file)
    nwbfile.processing["associated_files"].add(raw_arduino_timestamps_file)

    # Return photometry start in arduino time for video/DLC and behavioral alignment with photometry
    return {'photometry_start_in_arduino_time': photometry_start_in_arduino_time, 'port_visits': arduino_visit_times}
