import csv
import re
import itertools
import warnings
import numpy as np
from pathlib import Path
from pynwb import NWBFile
from ndx_franklab_novela import AssociatedFiles


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


def adjust_arduino_timestamps(arduino_timestamps: list):
    """Convert arduino timestamps to seconds and make photometry start at time 0"""
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


def parse_arduino_text(arduino_text: list, arduino_timestamps: list):
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
        block_start = re.match(r"Block: (\d)", line)
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

            # If this is the first block, we can use the current time as the start time
            if not previous_block:
                current_block["start_time"] = float(arduino_timestamps[i])
                previous_block = current_block

        # Detect beam breaks
        beam_break = re.match(r"beam break at port (\w)", line)
        if beam_break:
            port = beam_break.group(1)

            # If this is the start of a new trial, create the trial
            if not current_trial and port != previous_trial.get("end_port", None):
                # The first trial starts at the first block start, subsequent trials start at previous trial end
                current_trial["start_time"] = float(previous_trial.get("end_time", current_block.get("start_time")))
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

            # If we are in a trial, update the end times until we reach the end of the beam break
            if current_trial:
                current_trial["beam_break_end"] = float(arduino_timestamps[i])
                current_trial["end_time"] = float(arduino_timestamps[i])

                # If the next timestamp is far enough away (>100ms), the beam break is over, so end the trial
                beam_break_time_thresh = 0.1 # seconds
                if (i < len(arduino_timestamps) - 1) and (
                    arduino_timestamps[i + 1] - current_trial["beam_break_end"]
                ) >= beam_break_time_thresh:
                    trial_data.append(current_trial)
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
                        block_data.append(previous_block)
                        previous_block = current_block
                        trial_within_block = 1
                # If we have reached the last timestamp of the file while in the middle of a beam break,
                # make the trial end time the last timestamp
                elif i == len(arduino_timestamps)-1:
                    # Add the last trial
                    trial_data.append(current_trial)
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
                        # Make the current block (the new block that this trial started) empty
                        # so we don't add it. 
                        current_block = {}
    
    # Append the last trial if it exists
    if current_trial:
        trial_data.append(current_trial)
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
        assert len({block["maze_configuration"] for block in block_data}) == len(block_data), (
            "Maze configurations must be different for each block in a barrier change session"
        )
        # All reward probabilities should be the same for all blocks
        assert len({block["pA"] for block in block_data}) == 1, "pA should not vary in a barrier change session"
        assert len({block["pB"] for block in block_data}) == 1, "pB should not vary in a barrier change session"
        assert len({block["pC"] for block in block_data}) == 1, "pC should not vary in a barrier change session"
        logger.debug("All maze configurations are different "
                     "and all reward probabilities stay the same across blocks")

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


def add_behavior(nwbfile: NWBFile, metadata: dict, logger):
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
    arduino_timestamps, photometry_start_in_arduino_time = adjust_arduino_timestamps(arduino_timestamps)
    logger.debug(f"Photometry start in arduino time: {photometry_start_in_arduino_time}")

    # Read through the arduino text and timestamps to get trial and block data
    logger.debug("Parsing arduino text file...")
    trial_data, block_data = parse_arduino_text(arduino_text, arduino_timestamps)
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

    # Do some checks on trial and block data before adding to the NWB
    logger.debug("Validating trial and block data...")
    validate_trial_and_block_data(trial_data, block_data, logger)
    logger.debug("All validation checks passed.")

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
            epoch=1, # Berke Lab only has one epoch (session) per day
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
    for trial in trial_data:
        nwbfile.add_trial(
            epoch=1, # Berke Lab only has one epoch (session) per day
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
        task_epochs="",  # Required but unused
    )
    raw_arduino_timestamps_file = AssociatedFiles(
        name="arduino_timestamps",
        description="Raw arduino timestamps",
        content=raw_arduino_timestamps,
        task_epochs="",  # Required but unused
    )

    # Add arduino text and timestamps to the NWB as associated files
    nwbfile.create_processing_module(
        name="associated_files", description="Contains all associated files for behavioral data"
    )
    nwbfile.processing["associated_files"].add(raw_arduino_text_file)
    nwbfile.processing["associated_files"].add(raw_arduino_timestamps_file)
    
    # Return photometry start in arduino time for video/DLC and behavioral alignment with photometry
    return photometry_start_in_arduino_time
