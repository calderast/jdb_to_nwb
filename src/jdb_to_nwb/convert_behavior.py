import csv
import re
import itertools
import numpy as np
from pathlib import Path
from pynwb import NWBFile


def load_maze_configurations(maze_configuration_file_path: Path):
    """ Load maze configurations from the text file into a list of sets. """
    maze_sequence = []
    try:
        # Read from the text file and append each set to the list
        with open(maze_configuration_file_path, 'r') as maze_configuration_file:
            for line in maze_configuration_file:
               # Ignore any empty lines or lines starting with '#'
               if line.strip().startswith('#') or not line.strip():
                   continue
               # Strip any whitespace and split by commas to create a set of ints
               maze = set(map(int, line.strip().split(',')))
               maze_sequence.append(maze)
            return maze_sequence
    except:
       print(f"Could not load maze configurations from {maze_configuration_file_path}")
       return []
    

def adjust_arduino_timestamps(arduino_timestamps: list):
    """ Convert arduino timestamps to corresponding photometry sample number. """
    # The photometry start time is always the second timestamp in arduino_timestamps
    photometry_start = arduino_timestamps[1]

    # Adjust all arduino timestamps so the photometry starts at time zero
    arduino_timestamps = np.subtract(arduino_timestamps, photometry_start)

    # Convert arduino timestamps to corresponding photometry sample number. 
    # Target rate = 250 samples/sec from 10 KHz initial sample rate (same as photometry)
    # (This is named 'ardstamps' in the original conversion code)
    photometry_sample_for_arduino_event = np.round(arduino_timestamps*(250/1000)).astype(int)
    return photometry_sample_for_arduino_event
    

def parse_arduino_text(arduino_text: list, arduino_timestamps: list):
    """
    Parse the arduino text output and corresponding timestamps into lists
    containing information about trials and blocks in this session. 

    We have some interesting time alignment issues here between trials and blocks.
    Beam breaks span multiple indices in the text file (the beam stays broken the 
    entire time the rat's head is in the port, and will continuously print "beam break").
    Trials end when the rat removes their head from the reward port (end of beam break)
    In the arduino text file, a block is triggered by the beam break of the last trial in a block.
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
    trial_within_block = 1 # tracks the trials in each block
    trial_within_session = 1 # tracks the total trials in this session

    for i, line in enumerate(arduino_text):
        # Detect block starts
        block_start = re.match(r'Block: (\d)', line)
        if block_start:
            # Get block metadata: reward probabilities are always on the next 3 lines
            current_block = {
                'block': int(block_start.group(1)) + 1,
                'pA': int(arduino_text[i+1].split(":")[1].strip()),
                'pB': int(arduino_text[i+1].split(":")[1].strip()),
                'pC': int(arduino_text[i+3].split(":")[1].strip()),
                'start_time': None, # Set this once we find the end of the beam break that triggered this block
                'end_time': None  # Set this as the start time of the next block
            }

            # If this is the first block, we can use the current time as the start time 
            if not previous_block:
                current_block['start_time'] = float(arduino_timestamps[i])
                previous_block = current_block

        # Detect beam breaks
        beam_break = re.match(r'beam break at port (\w)', line)
        if beam_break:
            port = beam_break.group(1)

            # If this is the start of a new trial, create the trial 
            if not current_trial and port != previous_trial.get('end_port', None):
                # The first trial starts at the first block start, subsequent trials start at previous trial end
                current_trial['start_time'] = float(previous_trial.get('end_time', current_block.get('start_time')))
                current_trial['beam_break_start'] = float(arduino_timestamps[i])
                current_trial['start_port'] = previous_trial.get('end_port', 'None')
                current_trial['end_port'] = port
                current_trial['trial'] = trial_within_block
                current_trial['trial_within_session'] = trial_within_session

                # The line immediately following the beam break start contains reward information
                current_trial['reward'] = (
                    1 if re.search(fr"rwd delivered at port {port}", arduino_text[i+1]) 
                    else 0 if re.search(fr"no Reward port {port}", arduino_text[i+1]) 
                    else None
                )

            # If we are in a trial, update the end times until we reach the end of the beam break
            if current_trial:
                current_trial['beam_break_end'] = float(arduino_timestamps[i])
                current_trial['end_time'] = float(arduino_timestamps[i])

                # If the next timestamp is far enough away (>100ms), the beam break is over, so end the trial
                if (i < len(arduino_timestamps)-1) and (arduino_timestamps[i+1] - current_trial['beam_break_end']) >= 100:
                    trial_data.append(current_trial)
                    # Reset trial data
                    previous_trial = current_trial
                    current_trial = {}
                    trial_within_session += 1
                    trial_within_block += 1
                    # If this trial triggered a new block, the start time of the block = the end of the beam break
                    if not current_block['start_time']:
                        current_block['start_time'] = float(arduino_timestamps[i])
                        previous_block['end_time'] = float(arduino_timestamps[i])
                        block_data.append(previous_block)
                        previous_block = current_block
                        trial_within_block = 1

    # Append the last trial if it exists
    if current_trial:
        trial_data.append(current_trial)
        previous_trial = current_trial
    
    # Append the last block
    current_block['end_time'] = float(previous_trial['end_time'])
    block_data.append(current_block)

    return trial_data, block_data


def add_behavior(nwbfile: NWBFile, metadata: dict):
    print("Adding behavior...")

    # Get file paths for behavior from metadata file
    arduino_text_file_path = metadata["behavior"]["arduino_text_file_path"]
    arduino_timestamps_file_path = metadata["behavior"]["arduino_timestamps_file_path"]
    maze_configuration_file_path = metadata["behavior"]["maze_configuration_file_path"]

    # Read arduino text file into a list of strings
    with open(arduino_text_file_path, 'r') as arduino_text_file:
        arduino_text = arduino_text_file.read().splitlines()

    # Read arduino timestamps from the CSV into a list of floats
    with open(arduino_timestamps_file_path, 'r') as arduino_timestamps_file:
        arduino_timestamps = list(map(float, itertools.chain.from_iterable(csv.reader(arduino_timestamps_file))))

    # Make sure arduino text and arduino timestamps are the same length
    if len(arduino_text) != len(arduino_timestamps):
        raise ValueError(f"Mismatch in list lengths: arduino text has {len(arduino_text)} entries, "
                         f"but timestamps have {len(arduino_timestamps)} entries.")
    
    # Convert arduino timestamps to corresponding photometry sample number
    arduino_timestamps = adjust_arduino_timestamps(arduino_timestamps)
    
    # Read through the arduino text and timestamps to get trial and block data
    trial_data, block_data = parse_arduino_text(arduino_text, arduino_timestamps)

    # Load maze configurations for each block from the maze configuration file  
    maze_configurations = load_maze_configurations(maze_configuration_file_path)

    # Make sure the number of blocks matches the number of loaded maze configurations
    if len(block_data) != len(maze_configurations):
        raise ValueError(f"There are {len(block_data)} in the arduino text file, "
                         f"but {len(maze_configurations)} in the maze configuration file."
                         "There should be one maze configuration per block.")
    
    # Add maze_configuration to each block in block_metadata
    for block, maze in zip(block_data, maze_configurations):
        block['maze_configuration'] = maze

    # Add columns for block and trial data to the NWB file
    nwbfile.add_epoch_column(name="block", description="The block number within the session")
    nwbfile.add_epoch_column(name="maze_configuration", description="The maze configuration for each block")
    nwbfile.add_epoch_column(name="prob_A", description="The probability of reward at port A for each block")
    nwbfile.add_epoch_column(name="prob_B", description="The probability of reward at port B for each block")
    nwbfile.add_epoch_column(name="prob_C", description="The probability of reward at port C for each block")
    nwbfile.add_trial_column(name="trial_num", description="The trial number within the block")
    nwbfile.add_trial_column(name="trial_within_session", description="The trial number within the session")
    nwbfile.add_trial_column(name="start_port", description="The reward port the rat started at (A, B, or C)")
    nwbfile.add_trial_column(name="end_port", description="The reward port the rat ended at (A, B, or C)")
    nwbfile.add_trial_column(name="reward", description="If the rat got a reward at the port (1 or 0)")
    nwbfile.add_trial_column(name="beam_break_start", description="The time the rat entered the reward port")
    nwbfile.add_trial_column(name="beam_break_end", description="The time the rat exited the reward port")

    # Add each block to the NWB as a behavioral epoch
    for block in block_data:
        nwbfile.add_epoch(
            block=block['block'],
            start_time=block['start_time'],
            stop_time=block['end_time'], 
            maze_configuration=block['maze_configuration'],
            prob_A=block['pA'],
            prob_B=block['pB'],
            prob_C=block['pC'],
        )

    # Add each trial to the NWB
    for trial in trial_data:
        nwbfile.add_trial(
            start_time=trial['start_time'],
            stop_time=trial['end_time'],
            trial_num=trial['trial'], 
            trial_within_session=trial['trial_within_session'],
            start_port=trial['start_port'],
            end_port=trial['end_port'],
            reward=trial['reward'],
            beam_break_start=trial['beam_break_start'],
            beam_break_end=trial['beam_break_end']
        )

    # NOTE: the start/end times are in photometry samples, but NWB wants seconds relative to the start of the recording
    # NOTE: first trial currently starts at block 1 start and has start_port=None, should I change things so time before the first nosepoke is not included in any trial
    # NOTE: time after the last nosepoke is not included in any trial