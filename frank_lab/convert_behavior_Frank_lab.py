import re
import os
import ast
import h5py
import logging
import pandas as pd
from . import __version__
from pathlib import Path
from collections import Counter
from hdmf.common import DynamicTable
from pynwb import NWBFile, NWBHDF5IO

# Define regex for parsing stateScriptLog
poke_in_regex = re.compile(r"^(\d+)\sUP\s(\d+)")  # matches: timestamp UP port_num
poke_out_regex = re.compile(r"^(\d+)\sDOWN\s(\d+)")  # matches: timestamp DOWN port_num
behavior_data_regex = re.compile(
    r"(\d+)\s+(contingency|trialThresh|totalPokes|totalRewards|ifDelay|"
    r"countPokes[1-3]|countRewards[1-3]|portProbs[1-3])\s*=\s*(\d+)"
)
block_end_regex = re.compile(r"(\d+)\s+This block is over!")
session_end_regex = re.compile(r"(\d+)\s+This session is complete!")


def setup_logger(log_name, path_logfile_info, path_logfile_warn, path_logfile_debug) -> logging.Logger:
    """
    Sets up a logger that outputs to 3 different files:
    - File for all general logs (log level INFO and above).
    - File for warnings and errors (log level WARNING and above).
    - File for detailed debug output (log level DEBUG and above)

    Args:
    log_name: Name of the logfile (for logger identification)
    path_logfile_info: Path to the logfile for info messages
    path_logfile_warn: Path to the logfile for warning and error messages
    path_logfile_debug: Path to the logfile for debug messages

    Returns:
    logging.Logger
    """

    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels (DEBUG and above)

    # Define format for log messages
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%d-%b-%y %H:%M:%S")

    # Handler for logging messages INFO and above to a file
    fileHandler_info = logging.FileHandler(path_logfile_info, mode="w")
    fileHandler_info.setFormatter(formatter)
    fileHandler_info.setLevel(logging.INFO)

    # Handler for logging messages WARNING and above to a file
    fileHandler_warn = logging.FileHandler(path_logfile_warn, mode="w")
    fileHandler_warn.setFormatter(formatter)
    fileHandler_warn.setLevel(logging.WARNING)

    # Handler for logging messages DEBUG and above to a file
    fileHandler_debug = logging.FileHandler(path_logfile_debug, mode="w")
    fileHandler_debug.setFormatter(formatter)
    fileHandler_debug.setLevel(logging.DEBUG)

    # Add handlers to the logger
    logger.addHandler(fileHandler_info)
    logger.addHandler(fileHandler_warn)
    logger.addHandler(fileHandler_debug)

    return logger


def parse_ifDelay_events_for_legacy_statescriptlog(behavior_data, logger):
    """
    Current statescriptlogs print a full set of info for each trial:
    (contingency|trialThresh|totalPokes|totalRewards|ifDelay|countPokes[1-3]|countRewards[1-3]|portProbs[1-3])

    However, earlier statescriptlogs only printed the ifDelay value when the value changed, 
    instead of with the trial information. For these cases, the most recent printed ifDelay 
    value before the trial info is the correct value for this trial. 

    This function processes behavior_data parsed from a legacy statescriptlog 
    (ifDelay printed when the value changes) into the expected structure matching 
    current statescriptlogs (ifDelay info for each trial).

    Args:
    behavior_data: list of dicts each with keys {'timestamp', 'name', 'value'}, where the 'name' \
        field is one of those specified above (contingency, etc). For a legacy statescriptlog, there \
        should be the same number of entries with each name (one per trial), except for ifDelay
    logger: Logger to track progress

    Returns:
    behavior_data: same as above, but with the same number of entries for each trial, including \
    ifDelay. Note the timestamp of ifDelay has been artificially modified to match the trial
    """

    logger.info("Parsing ifDelay events from legacy stateScriptLog and assigning correct value to each trial")

    # Separate out 'ifDelay' events from behavior_data
    ifdelay_events = [event for event in behavior_data if event["name"] == "ifDelay"]
    behavior_data = [event for event in behavior_data if event["name"] != "ifDelay"]

    # Make sure we have the complete set of information for each trial (excluding 'ifDelay')
    variable_counts = Counter(event["name"] for event in behavior_data)
    if len(set(variable_counts.values())) != 1:
        logger.error(f"Mismatch in the amount of information for each trial: {variable_counts}!")
        raise Exception(f"Mismatch in the amount of information for each trial: {variable_counts}!")

    logger.debug(f"Amount of info for each trial field (excluding ifDelay): {variable_counts}")
    logger.debug("The amount of info should be equal to the number of trials.")
    logger.debug(f"The value for ifDelay changes {len(ifdelay_events)} times.")

    # Group behavioral events by trial
    info_rows_per_trial = len(variable_counts)
    trial_info = [behavior_data[i : i + info_rows_per_trial] for i in range(0, len(behavior_data), info_rows_per_trial)]

    # Extra check to ensure each trial has all of the required info
    required_info = {event["name"] for event in behavior_data}
    for trial in trial_info:
        info_present = {event["name"] for event in trial}
        assert info_present == required_info, f"Trial is missing events: {required_info - info_present}!"

    # Loop through trials, assigning the most recent 'ifDelay' value to each trial
    # The 'ifDelay' value is 0 for trials before the first 'ifDelay' was printed
    latest_ifdelay = {"timestamp": 0, "name": "ifDelay", "value": 0}
    for trial in trial_info:
        # Find the time this trial info was printed
        trial_timestamp = min(event["timestamp"] for event in trial)

        # Find the most recent 'ifDelay' value before this trial
        for ifdelay in ifdelay_events:
            if ifdelay["timestamp"] < trial_timestamp:
                # Update the most recent valid ifDelay
                latest_ifdelay = ifdelay
            else:
                # Stop checking once we pass the trial's timestamp
                break

        # Add the most recent ifDelay value to this trial's info.
        # We give the ifDelay a timestamp = (min trial timestamp+2) to match the
        # current statescriptlogs when ifDelay was printed alongside trial info.
        # We insert the ifDelay event right after totalRewards to match current statescriptlogs.
        total_rewards_idx = next((i for i, event in enumerate(trial) if event["name"] == "totalRewards"), None)
        trial.insert(
            total_rewards_idx + 1,
            {"timestamp": trial_timestamp + 2, "name": "ifDelay", "value": latest_ifdelay["value"]},
        )

    # Flatten the grouped data to create a new behavior_data with ifDelay info for each trial
    behavior_data = [event for trial in trial_info for event in trial]

    # Make sure we now have the complete set of information for each trial (now including 'ifDelay')
    variable_counts = Counter(event["name"] for event in behavior_data)
    if len(set(variable_counts.values())) != 1:
        logger.error(f"Mismatch in the amount of information for each trial: {variable_counts}!")
        raise Exception(f"Warning: Mismatch in the amount of information for each trial: {variable_counts}")
    logger.debug(f"Amount of info for each trial field (now including ifDelay): {variable_counts}")

    # Return the updated behavior_data
    return behavior_data


def parse_trial_and_block_data(behavior_data, block_ends, logger):
    """
    Parse behavioral data from the stateScriptLog into dataframes of trial-level and block-level data

    Args:
    behavior_data: list of dicts of behavioral event data from the statescriptlog
    block_ends: list of timestamps of block ends found in the statescriptlog
    logger: Logger to track progress

    Returns:
    trial_df: Dataframe of trial information for this epoch
    block_df: Dataframe of block information for this epoch
    """

    logger.info("Parsing behavioral data from the stateScriptLog into dataframes of trial and block data")

    # Convert our list of block end timestamps to a dictionary of block: timestamp
    block_ends_dict = {index + 1: item["timestamp"] for index, item in enumerate(block_ends)}
    logger.debug(f"Found block ends: {block_ends_dict}")
    # Set the default block end time as a big number that will
    # definitely be larger than all timestamps in the stateScriptLog.
    # This will be used if we don't have a recorded block end time
    # and overwritten by the real timestamp later.
    default_block_end_time = 100_000_000

    # Make sure we have the complete set of information for each trial
    variable_counts = Counter(item["name"] for item in behavior_data)
    info_rows_per_trial = len(variable_counts)
    if len(set(variable_counts.values())) != 1:
        logger.error(f"Mismatch in the amount of information for each trial: {variable_counts}")
        raise Exception(f"Warning: Mismatch in the amount of information for each trial: {variable_counts}")
    logger.debug(f"Found the correct amount of information for each trial: {variable_counts}")

    # Initialize variables
    trial_data = []
    block_data = []
    current_trial = {}
    previous_trial = {}
    current_block = {}
    previous_block = {}
    trial_within_block = 1  # tracks the trials in each block
    trial_within_session = 1  # tracks the total trials in this session
    block = 1

    port_visit_counts = {1: 0, 2: 0, 3: 0}
    total_rewards = 0

    # Group our behavioral data into trials
    for row in range(0, len(behavior_data), info_rows_per_trial):
        # Grab the data for this trial
        trial_dict = {
            item["name"]: {"timestamp": item["timestamp"], "value": item["value"]}
            for item in behavior_data[row : row + info_rows_per_trial]
        }

        # Start the first block
        if trial_within_session == 1:
            current_block = {
                "block": block,
                "pA": trial_dict["portProbs1"]["value"],
                "pB": trial_dict["portProbs2"]["value"],
                "pC": trial_dict["portProbs3"]["value"],
                "statescript_end_timestamp": block_ends_dict.get(block, default_block_end_time),
                "start_trial": 1,
                "end_trial": None,
                # This may be updated later if the rat does not complete all trials in this block
                "num_trials": trial_dict["trialThresh"]["value"],
            }
        # Or move to the next block if it's time
        elif trial_dict["contingency"]["timestamp"] >= current_block["statescript_end_timestamp"]:
            # Update the number of trials in the block because we may not have reached the trial threshold
            current_block["num_trials"] = trial_within_block - 1
            current_block["end_trial"] = current_trial.get("trial_within_session")
            # The current block is now the previous block
            previous_block = current_block
            logger.debug(f"Adding block: {previous_block}")
            block_data.append(previous_block)
            block += 1
            # Set up the new current block
            current_block = {
                "block": block,
                "pA": trial_dict["portProbs1"]["value"],
                "pB": trial_dict["portProbs2"]["value"],
                "pC": trial_dict["portProbs3"]["value"],
                "statescript_end_timestamp": block_ends_dict.get(block, default_block_end_time),
                "start_trial": previous_block.get("end_trial") + 1,
                "end_trial": None,
                "num_trials": trial_dict["trialThresh"]["value"],
            }
            # Reset port visit counts and reward info for the new block
            port_visit_counts = {1: 0, 2: 0, 3: 0}
            total_rewards = 0
            trial_within_block = 1

        # Get the end port for this trial by checking which poke count increased
        current_port_visit_counts = {port_num: trial_dict[f"countPokes{port_num}"]["value"] for port_num in [1, 2, 3]}
        end_port = next((i for i in [1, 2, 3] if current_port_visit_counts[i] == port_visit_counts[i] + 1), None)
        if end_port is None:
            logger.error(
                f"No end port detected for trial: {trial_within_block} in block {current_block.get('block_num')}"
            )
            raise Exception(
                f"Warning: No end port detected for trial: {trial_within_block} "
                f"in block {current_block.get('block_num')}"
            )

        # Only record the delay value if this was a rewarded trial
        reward = 1 if trial_dict["totalRewards"]["value"] == total_rewards + 1 else 0

        # We may not have delay info for all trial types:
        # If trial dict does not include key "ifDelay", create a default dict that makes delay value "N/A"
        delay_dict = trial_dict.get("ifDelay", {"value": "N/A"})
        delay = delay_dict.get("value") if reward else "N/A"

        # Add the information for this trial
        current_trial = {
            "trial_within_block": trial_within_block,
            "trial_within_session": trial_within_session,
            "block": current_block.get("block"),
            "start_port": previous_trial.get("end_port", -1),
            "end_port": end_port,
            "reward": reward,
            "delay": delay,
            "statescript_reference_timestamp": trial_dict["contingency"]["timestamp"],
        }
        logger.debug(f"Adding trial: {current_trial}")
        trial_data.append(current_trial)

        # Update for the next trial
        previous_trial = current_trial
        trial_within_block += 1
        trial_within_session += 1
        port_visit_counts = current_port_visit_counts
        total_rewards = trial_dict["totalRewards"]["value"]

    # Update the number of trials in the final block because we may not have reached the trial threshold
    current_block["num_trials"] = trial_within_block - 1
    current_block["end_trial"] = current_trial.get("trial_within_session")
    # Update the end time of the final block
    if current_block["statescript_end_timestamp"] == default_block_end_time:
        current_block["statescript_end_timestamp"] = previous_trial.get("statescript_reference_timestamp")
    # Append the final block
    logger.debug(f"Adding block: {current_block}")
    block_data.append(current_block)

    # Sanity check that we got data for the expected number of trials
    total_trials = set(variable_counts.values()).pop()
    if len(trial_data) != total_trials:
        logger.error(f"Expected data for {total_trials} trials, got data for {len(trial_data)}")
        raise Exception(f"Warning: Expected data for {total_trials} trials, got data for {len(trial_data)}")
    logger.debug(f"Got data for the expected number of trials ({total_trials})")

    # Map ports 1, 2, 3 to A, B, C (mapping -1 to "None" for the first start_port)
    trial_df = pd.DataFrame(trial_data)
    trial_df["start_port"] = trial_df["start_port"].map({-1: "None", 1: "A", 2: "B", 3: "C"})
    trial_df["end_port"] = trial_df["end_port"].map({1: "A", 2: "B", 3: "C"})

    return trial_df, pd.DataFrame(block_data)


def parse_nosepoke_events(nosepoke_events, nosepoke_DIOs, logger, poke_time_threshold=1):
    """
    Given a all nosepoke events from the statescript and all nosepoke DIOs, 
    ensure all events are valid and match the statescript and DIO nosepokes.
    Return a dataframe including only nosepoke events at a new port.

    The rat must stop breaking the beam for at least poke_time_threshold
    for that poke to be considered over (multiple consecutive pokes at the same 
    port in a short period of time are considered one poke).
    Instead of recording the poke_out directly following the poke_in at a new port,
    we record the last poke_out after which the rat didn't immediately poke back in again
    (immediately = poke_out and next poke_in are less than poke_time_threshold apart).

    Args:
    nosepoke_events: list of dicts, where each dict contains key-value pairs \
        describing each nosepoke event from the statescriptlog: \
            'timestamp': statescript timestamp, \
            'event_name': 'poke_in' or 'poke_out', \
            'port': 1, 2, or 4 (referring to ports A, B, and C) 
    nosepoke_DIOs: dict with keys wellA_poke, wellB_poke, and wellC_poke corresponding \
        to DIO events, and values (data, timestamps) where data is 1/0 DIO high/low \
        and timestamps are the DIO timestamps of these events
    logger: Logger to track progress

    Returns:
    Dataframe including only nosepoke events at new ports, with columns \
    event_name, port, timestamp_DIO, timestamp_statescript
    """

    logger.info("Matching nosepoke events from statescript with nosepokes from DIOs")

    # Make sure we have the same number of poke_in and poke_out events from the statescriptlog
    # NOTE: We later check that each poke_in is followed by a poke_out at the same port, so maybe overkill.
    # Keeping for now, probably delete later.
    event_counts = Counter(event["event_name"] for event in nosepoke_events)
    if event_counts["poke_in"] != event_counts["poke_out"]:
        logger.error(
            f"Got {event_counts['poke_in']} poke_in events "
            f"but {event_counts['poke_out']} poke_out events in the statescript!"
        )
        raise Exception(
            f"Warning: {event_counts['poke_in']} poke_in events "
            f"but {event_counts['poke_out']} poke_out events in the statescript!"
        )

    # Convert statescript pokes from list of dicts to a dataframe (mapping DIO 1, 2, 4 to ports A, B, C)
    statescript_nosepoke_df = pd.DataFrame(nosepoke_events)
    statescript_nosepoke_df["port"] = statescript_nosepoke_df["port"].map({1: "A", 2: "B", 4: "C"})

    # Create a dataframe of DIO pokes that matches the dataframe from the statescript
    port_map = {"wellA_poke": "A", "wellB_poke": "B", "wellC_poke": "C"}
    DIO_nosepoke_df = pd.DataFrame(
        [
            {"timestamp": ts, "event_name": "poke_in" if d == 1 else "poke_out", "port": port_map[k]}
            for k, (data_list, timestamps) in nosepoke_DIOs.items()
            for d, ts in zip(data_list, timestamps)
        ]
    )
    DIO_nosepoke_df = DIO_nosepoke_df.sort_values(by="timestamp").reset_index(drop=True)

    logger.info(f"{len(DIO_nosepoke_df)} nosepokes from the DIOs: {DIO_nosepoke_df['port'].value_counts().to_dict()}")
    logger.info(
        f"{len(statescript_nosepoke_df)} nosepokes from the statescript: "
        f"{statescript_nosepoke_df['port'].value_counts().to_dict()}"
    )

    # Make sure each poke_in is followed by a poke_out at the same port (statescript)
    for row in range(0, len(statescript_nosepoke_df) - 1, 2):
        event1 = statescript_nosepoke_df.iloc[row]
        event2 = statescript_nosepoke_df.iloc[row + 1]
        if not (
            event1["event_name"] == "poke_in"
            and event2["event_name"] == "poke_out"
            and event1["port"] == event2["port"]
        ):
            logger.error(
                "Warning: Invalid nosepoke pair from statescript at "
                f"timestamps {event1['timestamp']} and {event2['timestamp']}!"
            )
            raise Exception(
                "Warning: Invalid nosepoke pair from statescript at "
                f"timestamps {event1['timestamp']} and {event2['timestamp']}!"
            )
    logger.debug("Each poke_in is followed by a poke_out at the same port (statescript)")

    # Make sure each poke_in is followed by a poke_out at the same port (DIO)
    for row in range(0, len(DIO_nosepoke_df) - 1, 2):
        event1 = DIO_nosepoke_df.iloc[row]
        event2 = DIO_nosepoke_df.iloc[row + 1]
        if not (
            event1["event_name"] == "poke_in"
            and event2["event_name"] == "poke_out"
            and event1["port"] == event2["port"]
        ):
            logger.error(
                "Warning: Invalid nosepoke pair from DIOs at "
                f"timestamps {event1['timestamp']} and {event2['timestamp']}!"
            )
            raise Exception(
                "Warning: Invalid nosepoke pair from DIOs at "
                f"timestamps {event1['timestamp']} and {event2['timestamp']}!"
            )
    logger.debug("Each poke_in is followed by a poke_out at the same port (DIOs)")

    # Make sure the number of DIO pokes matches the number of pokes from the statescriptlog.
    # Note that the DIO may have more pokes because it keeps recording
    # after the statescript has been stopped (this is ok).
    # Warn the user about it anyway for the sake of providing all of the info.
    if len(DIO_nosepoke_df) > len(statescript_nosepoke_df):
        logger.warning(
            f"Length mismatch: {len(DIO_nosepoke_df)} nosepokes from DIOs, "
            f"but only {len(statescript_nosepoke_df)} nosepokes from statescript.\n"
            "The DIO may have more pokes because it keeps recording "
            "after the statescript has been stopped (this is ok)."
        )
    # The statescript should never have more pokes than the DIOs - break if this happens so we can figure out why.
    elif len(statescript_nosepoke_df) > len(DIO_nosepoke_df):
        logger.error(
            f"Length mismatch: {len(statescript_nosepoke_df)} nosepokes from statescript "
            f"but {len(DIO_nosepoke_df)} nosepokes from DIOs!"
        )
        raise Exception(
            f"Length mismatch: {len(statescript_nosepoke_df)} nosepokes from statescript "
            f"but {len(DIO_nosepoke_df)} nosepokes from DIOs!"
        )

    # Match statescript and DIO pokes.
    # For each event_name and port combination, add an index column enumerating which one it is.
    # This will allow us to merge the DIO and statescript dfs while matching the correct instances of each event
    DIO_nosepoke_df["index"] = DIO_nosepoke_df.groupby(["event_name", "port"]).cumcount()
    statescript_nosepoke_df["index"] = statescript_nosepoke_df.groupby(["event_name", "port"]).cumcount()

    # Merge based on matching event_name, port, and index (created above)
    merged_nosepokes = pd.merge(
        DIO_nosepoke_df,
        statescript_nosepoke_df,
        on=["event_name", "port", "index"],
        how="inner",
        suffixes=("_DIO", "_statescript"),
    )

    # Also do an outer merge that keeps all rows so we can log info about which rows (if any) do not match.
    # This is for info/debugging purposes only.
    merged_nosepokes_outer = pd.merge(
        DIO_nosepoke_df,
        statescript_nosepoke_df,
        on=["event_name", "port", "index"],
        how="outer",
        suffixes=("_DIO", "_statescript"),
    )
    DIO_statescript_mismatches = merged_nosepokes_outer[
        merged_nosepokes_outer["timestamp_DIO"].isna() | merged_nosepokes_outer["timestamp_statescript"].isna()
    ]

    if not DIO_statescript_mismatches.empty:
        logger.warning("Mismatched nosepokes between statescript and DIOs:")
        logger.warning(DIO_statescript_mismatches)
    else:
        logger.info("All DIO and statescript nosepokes were matched successfully!")

    # Iterate through pairs of rows in the dataframe, keeping only rows
    # that represent poke_in and poke_out events at a new port.
    # The rat must stop breaking the beam for at least poke_time_threshold
    # for that poke to be considered over (multiple consecutive pokes at the same
    # port in a short period of time are considered one poke).
    # Instead of recording the poke_out directly following the poke_in at a new port,
    # we record the last poke_out after which the rat didn't immediately poke back in again
    # (immediately = poke_out and next poke_in are less than poke_time_threshold apart).
    nosepokes_at_new_ports = []
    current_port = None
    potential_poke_out = None

    # Iterate through poke_in / poke_out pairs
    for row in range(0, len(merged_nosepokes) - 1, 2):
        poke_in = merged_nosepokes.iloc[row]
        poke_out = merged_nosepokes.iloc[row + 1]
        # Sanity check for merged statescript/DIO events:
        # make sure each poke_in is followed by a poke_out at the same port
        if not (
            poke_in["event_name"] == "poke_in"
            and poke_out["event_name"] == "poke_out"
            and poke_in["port"] == poke_out["port"]
        ):
            logger.warning(
                f"Invalid nosepoke pair at timestamps {poke_in['timestamp_DIO']} and {poke_out['timestamp_DIO']}!"
            )
            raise Exception(
                f"Invalid nosepoke pair at timestamps {poke_in['timestamp_DIO']} and {poke_out['timestamp_DIO']}!"
            )

        # If we have a poke_in at a new port, record it!
        if poke_in["port"] != current_port:
            # Record the last poke_out for the previous port if we haven't already
            if potential_poke_out is not None:
                nosepokes_at_new_ports.append(potential_poke_out)
            # Add the poke_in event
            nosepokes_at_new_ports.append(poke_in)
            # Save the poke_out as the potential poke end
            # This will likely be overwritten by the "true" poke_out, defined as the time
            # the rat pokes out and then does not immediately poke back in again
            potential_poke_out = poke_out
            # Update the current port so we can search for the "true" poke_out end
            current_port = poke_in["port"]

        # Or if we have another poke_in at the current port, and we are searching for the "true" poke_out,
        # check if the poke has already ended or if this is a continuation of the same poke event.
        elif potential_poke_out is not None:
            # If the poke_in is close enough in time to the previous poke_out, it counts as the same poke
            if (poke_in["timestamp_DIO"] - potential_poke_out["timestamp_DIO"]) <= poke_time_threshold:
                # Update the poke_out as the potential poke end
                potential_poke_out = poke_out
            # Otherwise, the poke_in is far enough in time from the previous poke_out, so the poke has ended.
            else:
                # The previous potential_poke_out is the true poke_out, so record it
                nosepokes_at_new_ports.append(potential_poke_out)
                # Indicate the poke has ended and we are no longer searching for the poke_out
                potential_poke_out = None

        # Otherwise, if we reach here, it means that we have another poke_in at the current port,
        # but we have already determined that the poke event has ended. Ignore these pokes.
        else:
            # NOTE: While this code is still in development, it may be helpful to log how often we reach
            # this case. It may also be helpful to log how far the poke_in was from the previous
            # poke_out as feedback on if we have chosen a good poke_time_threshold or if it should be adjusted.
            continue

    # Add the last poke_out if we missed it
    if potential_poke_out is not None:
        nosepokes_at_new_ports.append(potential_poke_out)

    logger.debug("All nosepokes at new ports:")
    logger.debug(pd.DataFrame(nosepokes_at_new_ports).drop(columns="index"))

    # Return a dataframe of nosepokes including only nosepokes at new ports
    return pd.DataFrame(nosepokes_at_new_ports).drop(columns="index")


def combine_nosepoke_and_trial_data(nosepoke_df, trial_df, session_end, logger):
    """
    Check that nosepoke data matches trial data and add nosepoke data to the trial dataframe

    Args:
    nosepoke_df: Dataframe of nosepoke events at new ports with columns \
        event_name, port, timestamp_DIO, timestamp_statescript
    trial_df: Dataframe of trial information
    session_end: Timestamp of session end (in statescript time), \
        or None if no session_end was recorded in the statescript
    logger: Logger to track progress

    Returns:
    trial_df: Dataframe of trial information with added columns \
        for poke_in and poke_out times (both DIO time and statescript time)
    """

    logger.info("Checking that nosepoke data matches trial data and combining these datastreams")
    logger.info(
        "Note that occasionally there are delays in statescript printing, "
        "which may cause some timing mismatches when comparing statescript times as a sanity check."
    )

    # Check that we have the right lengths for one poke_in and one poke_out per trial
    if len(nosepoke_df) != 2 * len(trial_df):
        if session_end is None:
            logger.error(f"Expected {2*len(trial_df)} nosepokes for {len(trial_df)} trials, got {len(nosepoke_df)}")
            raise Exception(f"Expected {2*len(trial_df)} nosepokes for {len(trial_df)} trials, got {len(nosepoke_df)}")
        else:
            # We may have more nosepoke pairs than trials if the rat kept running after the session end.
            # If we have a recorded session_end time, ignore all poke_in after this time.
            logger.debug(f"Expected {2*len(trial_df)} nosepokes for {len(trial_df)} trials, got {len(nosepoke_df)}!")
            logger.debug("We may have more nosepoke pairs than trials if the rat kept running after the session end.")
            logger.debug("Removing all poke pairs after the session end time.")

            nosepoke_df = nosepoke_df.reset_index()
            pokes_before_session_end = nosepoke_df[nosepoke_df["timestamp_statescript"] <= session_end].copy()

            # Note that if the last event before the session end is a poke_in, make sure to keep the poke_out!
            # The poke_out likely happened after the session end time was printed
            # (as the session end print is triggered by poke_in and not poke_out).

            last_poke_index = 0
            # If the last event before the session end was a poke_in, add 1 to the index to keep its poke_out
            if pokes_before_session_end.iloc[-1]["event_name"] == "poke_in":
                logger.debug("The last event before the session end was a poke_in, keeping its corresponding poke_out")
                last_poke_index = pokes_before_session_end.index[-1] + 1
            # If the last event before the session end is a poke_out, no adjustment needed!
            elif pokes_before_session_end.iloc[-1]["event_name"] == "poke_out":
                logger.debug("The last event before the session end was a poke_out")
                last_poke_index = pokes_before_session_end.index[-1]
            else:
                logger.error(
                    "event_name must be either 'poke_in' or 'poke_out'! "
                    f"Got event_name={pokes_before_session_end.iloc[-1]['event_name']}"
                )
                raise Exception("event_name must be either 'poke_in' or 'poke_out'!!")

            # Filter dataframe to remove all extra pokes
            nosepoke_df = nosepoke_df[nosepoke_df.index <= last_poke_index]

            # Check again after removing nosepokes after session end
            if len(nosepoke_df) != 2 * len(trial_df):
                logger.error(
                    "After removing nosepokes after the session end, "
                    f"expected {2*len(trial_df)} nosepokes for {len(trial_df)} trials, got {len(nosepoke_df)}"
                )
                raise Exception(
                    f"After removing nosepokes after the session end, "
                    f"expected {2*len(trial_df)} nosepokes for {len(trial_df)} trials, got {len(nosepoke_df)}"
                )
            logger.debug(
                "After removing nosepokes after the session end, got the expected "
                f"{len(nosepoke_df)} nosepokes for {len(trial_df)} trials!"
            )

    # Create columns to add poke_in and poke_out data to the trial_df
    trial_df["poke_in_time_statescript"] = None
    trial_df["poke_out_time_statescript"] = None
    trial_df["poke_in_time"] = None  # DIO time
    trial_df["poke_out_time"] = None  # DIO time

    # Iterate through the trial df and find corresponding poke_in and poke_out times
    for i, trial_row in trial_df.iterrows():
        # Find the nosepoke timestamps for the current trial and add them to the trial df
        poke_in_row = nosepoke_df.loc[nosepoke_df["event_name"] == "poke_in"].iloc[i]
        poke_out_row = nosepoke_df.loc[nosepoke_df["event_name"] == "poke_out"].iloc[i]
        trial_df.at[i, "poke_in_time_statescript"] = poke_in_row["timestamp_statescript"]
        trial_df.at[i, "poke_out_time_statescript"] = poke_out_row["timestamp_statescript"]
        trial_df.at[i, "poke_in_time"] = poke_in_row["timestamp_DIO"]
        trial_df.at[i, "poke_out_time"] = poke_out_row["timestamp_DIO"]

        # Sanity check that poke_in timestamp is close enough to the time
        # the trial info was printed to ensure these are matched correctly.
        # NOTE: It seems the trial info is printed after the first poke_out following
        # the poke_in (which is not always the recorded poke_out - see parse_nosepoke_events).
        # This check worked better in an earlier version of the code where we checked against that poke_out,
        # which we no longer record. We probably want to suppress output or set this to a debug
        # log level in the future, as even correctly matched pokes can trigger this warning
        # if the poke was long (causing trial info to be printed >5s after initial poke_in).
        # Keeping it for now - it is still a useful warning as we have not encountered all bug-causing cases.
        time_diff = trial_row["statescript_reference_timestamp"] - poke_in_row["timestamp_statescript"]
        if abs(time_diff) > 5000:
            logger.warning(
                f"Trial {trial_row['trial_within_session']} (block {trial_row['block']} "
                f"trial {trial_row['trial_within_block']}), port {trial_row['end_port']}:"
            )
            logger.warning(
                f"Poke in at statescript time {poke_in_row['timestamp_statescript']} "
                f"may not match trial printed at {trial_row['statescript_reference_timestamp']} "
                f"(Time diff = {time_diff/1000}s). This can also happen due to a long poke."
            )

        # Sanity check to ensure the poke in and poke out match the end_port for this trial
        if not ((trial_row["end_port"] == poke_in_row["port"]) and (trial_row["end_port"] == poke_out_row["port"])):
            logger.error(
                f"Trial ending at port {trial_row['end_port']} does not match "
                f"poke in at port {poke_in_row['port']} and poke out at port {poke_out_row['port']}"
            )
            raise Exception(
                f"Trial ending at port {trial_row['end_port']} does not match "
                f"poke in at port {poke_in_row['port']} and poke out at port {poke_out_row['port']}"
            )

        # Add start and end times based on DIO poke times (trials are poke_out to poke_out)
        trial_df["start_time"] = trial_df["poke_out_time"].shift(1)
        trial_df["end_time"] = trial_df["poke_out_time"]

        # Set the start time of the first trial to 3 seconds before the first poke_in.
        # This handles cases where the epoch start button was pressed and then the rat
        # was placed in the maze, so using epoch start time would be too early.
        # This may be overwritten by the epoch start time later, if the recorded epoch start
        # is after this time (which could happen in the case where 2 people were present
        # so the epoch start button was pressed at the same time the rat was placed in the maze).
        trial_df.at[0, "start_time"] = trial_df.at[0, "poke_in_time"] - 3

        # Add trial duration as a column
        trial_df["duration"] = trial_df["end_time"] - trial_df["start_time"]

    return trial_df


def combine_reward_and_trial_data(trial_df, reward_DIOs, logger):
    """
    Check that reward data from the statescript matches reward data
    from the DIOs, and add reward DIO times to the trial dataframe.

    Args:
    trial_df: Dataframe of information for each trial including column 'reward'
    reward_DIOs: tuple of (1/0 data, timestamps) for reward DIOs 'wellA_pump', 'wellB_pump', 'wellC_pump'
    logger: Logger to track progress

    Returns:
    trial_df: Dataframe of information for each trial with added columns 'pump_on_time' and 'pump_off_time'
    """
    logger.info(
        "Checking that reward data from the statescript matches reward data from the DIOs, "
        "and combining these datastreams"
    )

    # Create a dataframe of reward pump times from the DIO data
    reward_pump_times = []
    port_map = {"wellA_pump": "A", "wellB_pump": "B", "wellC_pump": "C"}
    for key, (data, timestamps) in reward_DIOs.items():
        for i in range(0, len(data), 2):
            # Make sure the data matches structure pump_on, pump_off
            if not (data[i] == 1 and data[i + 1] == 0):
                logger.error(
                    f"Data mismatch at index {i} for key {key}: expected [1, 0], got [{data[i]}, {data[i + 1]}]"
                )
                raise Exception(
                    f"Data mismatch at index {i} for key {key}: expected [1, 0], got [{data[i]}, {data[i + 1]}]"
                )

            # Make sure the pump_on and pump_off times are close together (<1s) to check they are matched correctly
            if not (abs(timestamps[i] - timestamps[i + 1]) < 1):
                logger.error(
                    "Expected timestamps to be within 1s, "
                    f"got pump_on_time {timestamps[i]}, pump_off_time {timestamps[i+1]}]"
                )
                raise Exception(
                    "Expected timestamps to be within 1s, "
                    f"got pump_on_time {timestamps[i]}, pump_off_time {timestamps[i+1]}]"
                )

            # Combine the pump_on and pump_off events into a single row
            reward_pump_times.append(
                {"port": port_map[key], "pump_on_time": timestamps[i], "pump_off_time": timestamps[i + 1]}
            )
    logger.debug("Check passed: reward data matches structure pump_on, pump_off")
    logger.debug("Check passed: all pump_on and pump_off times are close together (<1s)")

    # Make sure pump events end up in the same order regardless of if we sort by pump_on_time or pump_off_time
    reward_pump_df = pd.DataFrame(reward_pump_times).sort_values(by="pump_on_time").reset_index(drop=True)
    if not reward_pump_df.equals(
        pd.DataFrame(reward_pump_times).sort_values(by="pump_off_time").reset_index(drop=True)
    ):
        logger.error("DataFrames do not match when sorted by pump_on_time vs. pump_off_time")
        raise ValueError("DataFrames do not match when sorted by pump_on_time vs. pump_off_time")
    logger.debug(
        "Check passed: Pump events end up in the same order regardless of if we sort by pump_on_time or pump_off_time"
    )

    # Make sure each pump_on_time occurs before its corresponding pump_off_time
    if not (reward_pump_df["pump_on_time"] < reward_pump_df["pump_off_time"]).all():
        logger.error("Timing mismatch: not every pump_on_time is correctly matched to its pump_off_time")
        raise ValueError("Timing mismatch: not every pump_on_time is correctly matched to its pump_off_time")
    logger.debug("Check passed: every pump_on_time occurs before its corresponding pump_off_time")

    # Ensure we have one reward pump on/off DIO per rewarded trial
    rewarded_trial_df = trial_df[trial_df["reward"] == 1]

    if len(reward_pump_df) != len(rewarded_trial_df):
        logger.warning(
            f"Expected {len(rewarded_trial_df)} reward DIO events "
            f"for {len(rewarded_trial_df)} rewarded trials, got {len(reward_pump_df)}"
        )
    else:
        logger.debug(
            f"Check passed: got {len(reward_pump_df)} reward DIO events "
            f"for {len(rewarded_trial_df)} rewarded trials"
        )

    # Create columns to add reward pump times to the trial_df
    trial_df["pump_on_time"] = "N/A"
    trial_df["pump_off_time"] = "N/A"

    # Iterate through the rewarded trials and their corresponding DIO events
    for trial_row, DIO_times in zip(rewarded_trial_df.itertuples(index=True), reward_pump_df.itertuples(index=False)):
        # The end_port of this rewarded trial must match the reward pump port
        if trial_row.end_port != DIO_times.port:
            logger.error(
                f"Mismatch: trial end_port {trial_row.end_port} does not match reward pump port {DIO_times.port}"
            )
            raise Exception(
                f"Mismatch: trial end_port {trial_row.end_port} does not match reward pump port {DIO_times.port}"
            )

        # Ensure the reward pump turns on within a second of the poke
        if abs(trial_row.poke_in_time - DIO_times.pump_on_time) > 1:
            logger.error(f"Pump on at time {DIO_times.pump_on_time} may not match nosepoke at {trial_row.poke_in_time}")
            raise Exception(
                f"Pump on at time {DIO_times.pump_on_time} may not match nosepoke at {trial_row.poke_in_time}"
            )

        # Update the original trial_df with pump_on_time and pump_off_time
        trial_df.loc[trial_row.Index, "pump_on_time"] = DIO_times.pump_on_time
        trial_df.loc[trial_row.Index, "pump_off_time"] = DIO_times.pump_off_time

    return trial_df


def determine_session_type(block_data):
    """Determine the session type ("barrier change" or "probability change") based on block data."""

    # This case is rare/hopefully nonexistent - we always expect to have more than one block per session
    if len(block_data) == 1:
        return "single block"

    # Get the reward probabilities at each port for each block in the session
    reward_probabilities = []
    for _, block in block_data.iterrows():
        reward_probabilities.append([block["pA"], block["pB"], block["pC"]])

    # If the reward probabilities change with each block, this is a probability change session
    if reward_probabilities[0] != reward_probabilities[1]:
        return "probability change"
    # Otherwise, this must be a barrier change session
    else:
        return "barrier change"


def adjust_block_start_trials(trial_data, block_data, DIO_events, excel_data, logger):
    """
    Adjust the block start trials based on barrier_shift DIO events (if they exist)
    or data from the experimental notes excel sheet
    
    Args:
    trial_data: Dataframe of information for each trial in this epoch
    block_data: Dataframe of information for each block in this epoch
    DIO_events: dict of event_name: (data, timestamps) for each named DIO event, \
        including "barrier_shift" event if we have data for it
    excel_data: Dataframe of info for this epoch, with column "barrier shift trial ID"
    logger: Logger to track progress
    
    Returns:
    trial_data: Dataframe of trial info reflecting updated block boundaries
    block_data: Dataframe of block info reflecting updated block boundaries
    """

    logger.info(
        "Adjusting the block start trials based on barrier_shift DIO events (if they exist)"
        "or data from the experimental notes excel sheet"
    )

    barrier_shift_trials_DIO = None
    barrier_shift_trials_excel = None

    # If barrier_shift DIOs exist, use those as the ground truth
    if "barrier_shift" in DIO_events:
        logger.info("Found event 'barrier_shift' in DIO events.")
        logger.info("Getting barrier shift trials based on barrier_shift DIOs")

        barrier_shift_DIOs = DIO_events.get("barrier_shift")
        # The barrier_shift_DIOs are a pair of lists: (1/0 events, timestamps)
        # Take every other timestamp to get the times of the "1" (DIO button press) events
        # (We have already checked each 1 has a corresponding 0 so just taking every other is fine)
        barrier_shift_times = barrier_shift_DIOs[1][0::2]
        logger.debug(f"Barrier shift times: {barrier_shift_times}")

        # We've found some cases of DIO jitter or where the button seems to have been pressed multiple times.
        # BraveLu_20240516 is an example of this: DIOs give barrier shift trials [70, 139, 139, 139]
        # due to multiple DIO pulses in short succession, when we should just record barrier shifts at [70, 139]
        # To handle this, we remove all barrier shift DIO times within 10 seconds of the previous DIO
        # Monitor to see how this issue manifests in other sessions, or if it really just happened in old ones.
        barrier_shift_DIO_time_thresh = 10
        valid_barrier_shift_times = [barrier_shift_times[0]]
        for time in barrier_shift_times[1:]:
            # Keep only barrier shift DIOs that happen at least 10 seconds after the last valid DIO
            time_from_last_valid_DIO = time - valid_barrier_shift_times[-1]
            if time_from_last_valid_DIO >= barrier_shift_DIO_time_thresh:
                valid_barrier_shift_times.append(time)
            # Warn about removing DIOs too close to the previous DIO
            else:
                logger.warning(
                    "Detected jitter in barrier shift DIOs! "
                    f"Removing barrier shift at {time} because it is too close to last valid barrier shift "
                    f"at time {valid_barrier_shift_times[-1]} (time diff = {time_from_last_valid_DIO:.2f}s)"
                )

        barrier_shift_trials_DIO = []
        for barrier_shift_time in valid_barrier_shift_times:
            # Find the closest poke_in time just before the barrier shift time
            # Barrier shifts happen when the rat is at a port (just after poke_in)
            # The next trial (that begins on poke_out) is the first trial of the new block
            trials_pre_shift = trial_data.index[trial_data["poke_in_time"] <= barrier_shift_time]
            closest_idx = pd.to_numeric(
                ((trial_data.loc[trials_pre_shift, "poke_in_time"] - barrier_shift_time).abs())
            ).idxmin()
            barrier_shift_trial = trial_data.loc[closest_idx, "trial_within_session"]

            # Sanity check: get the time from trial start to barrier shift, and shift to next poke
            barrier_shift_time_from_poke = barrier_shift_time - trial_data.loc[closest_idx, "poke_in_time"]
            time_to_next_poke = trial_data.loc[closest_idx + 1, "poke_in_time"] - barrier_shift_time
            logger.info(
                f"Barrier shift DIO pressed {barrier_shift_time_from_poke:.2f}s "
                f"after poke_in of trial {barrier_shift_trial}."
            )
            logger.info(f"Next poke_in was {time_to_next_poke:.2f}s after barrier shift DIO pressed.")
            logger.info(f"Trial {barrier_shift_trial+1} is the first trial of the new block.")

            barrier_shift_trials_DIO.append(barrier_shift_trial)
        # Convert np.int(64) to int
        barrier_shift_trials_DIO = [int(x) for x in barrier_shift_trials_DIO]

        # Final check for DIO jitter - we shouldn't have duplicate barrier shift trials!
        if len(barrier_shift_trials_DIO) != len(set(barrier_shift_trials_DIO)):
            logger.warning(
                f"Got duplicate barrier shift trials from the DIOs due to jitter: {barrier_shift_trials_DIO}"
            )
            barrier_shift_trials_DIO = sorted(set(barrier_shift_trials_DIO))
            logger.warning(f"Removed duplicates. Barrier shift trials are now: {barrier_shift_trials_DIO}")

    # If the excel sheet has barrier shift info, use that also
    if "barrier shift trial ID" in excel_data.columns:
        logger.info("Found column 'barrier shift trial ID' in excel sheet.")
        logger.info("Getting barrier shift trials based on excel sheet")
        # Read barrier shift trials as a comma-separated string, and convert to a list
        barrier_shift_trials_str = excel_data["barrier shift trial ID"].iloc[0]
        barrier_shift_trials_excel = list(map(int, barrier_shift_trials_str.split(", ")))

    # If we have barrier shift info from both DIOs and excel sheet, check if they match
    if barrier_shift_trials_DIO is not None and barrier_shift_trials_excel is not None:
        # If there is a mismatch between DIO and excel, DIO wins
        if barrier_shift_trials_DIO != barrier_shift_trials_excel:
            logger.warning(
                "Mismatch in barrier shift info between barrier_shift DIOs and data from excel sheet!\n"
                f"DIO has barrier shift trials {barrier_shift_trials_DIO}, "
                f"excel sheet has {barrier_shift_trials_excel}!"
            )
            logger.warning("Using DIOs as true barrier shift trials.")
        else:
            logger.info(
                "Barrier_shift DIOs match data from excel sheet, "
                f"with barrier shifts at trials {barrier_shift_trials_DIO}!"
            )
        barrier_shift_trials = barrier_shift_trials_DIO

    # If only DIOs, use that
    elif barrier_shift_trials_DIO is not None:
        logger.debug("No data from excel sheet, using 'barrier_shift' DIO event.")
        barrier_shift_trials = barrier_shift_trials_DIO

    # Or if only excel, use that
    elif barrier_shift_trials_excel is not None:
        logger.debug("No 'barrier_shift' in DIO events, using data from excel sheet.")
        barrier_shift_trials = barrier_shift_trials_excel

    else:
        logger.error(
            "No 'barrier_shift' DIO event or 'barrier shift trial ID' from excel found\n"
            "when trying to adjust block start trials for a barrier change session."
        )
        raise ValueError(
            "No 'barrier_shift' DIO event or 'barrier shift trial ID' from excel found\n"
            "when trying to adjust block start trials for a barrier change session."
        )

    # Sanity check: make sure the last barrier shift is before the last trial
    last_trial = block_data["end_trial"].iloc[-1]
    if barrier_shift_trials[-1] > last_trial:
        # This has never happened (and it never should), but DIOs can be odd.
        # Complain and break if it does so we can evaluate what caused it and how to handle it then.
        logger.error("Something went wrong! Last barrier shift is after the last trial!")
        raise ValueError("Something went wrong! Last barrier shift is after the last trial!")

    # Set up the start and end trials of the blocks based on the barrier shifts
    # Add trial 1 as the start of the first block and the last trial as the end of the last block
    block_start_trials = [1] + [t + 1 for t in barrier_shift_trials]
    block_end_trials = barrier_shift_trials + [int(last_trial)]

    # Get pA, pB, pC (same for all blocks because this is a barrier change session)
    pA, pB, pC = block_data.iloc[0][["pA", "pB", "pC"]]

    # Create new block dataframe using new block start/end trials
    # statescript_end_timestamp is now N/A because statescript timestamps
    # no longer correspond to barrier changes
    new_block_data = pd.DataFrame(
        {
            "block": range(1, len(block_start_trials) + 1),
            "pA": [pA] * len(block_start_trials),
            "pB": [pB] * len(block_start_trials),
            "pC": [pC] * len(block_start_trials),
            "statescript_end_timestamp": "N/A",
            "start_trial": block_start_trials,
            "end_trial": block_end_trials,
            "num_trials": [end - start + 1 for end, start in zip(block_end_trials, block_start_trials)],
            "task_type": "barrier change",
        }
    )

    logger.debug("Creating new block dataframe using new block start/end trials")
    logger.debug(
        "statescript_end_timestamp is now N/A because statescript timestamps no longer correspond to barrier changes"
    )
    logger.debug("Updating 'block' and 'trial' columns in trial_data to reflect the updated block boundaries")

    # Update 'block' and 'trial' columns in trial_data to reflect the updated block boundaries
    for _, row in new_block_data.iterrows():
        trials_in_block = (trial_data["trial_within_session"] >= row["start_trial"]) & (
            trial_data["trial_within_session"] <= row["end_trial"]
        )
        trial_data.loc[trials_in_block, "block"] = row["block"]
        trial_data.loc[trials_in_block, "trial_within_block"] = range(1, trials_in_block.sum() + 1)

    return trial_data, new_block_data


def add_block_start_end_times(trial_data, block_data):
    """
    Add the DIO start and end times to the blocks

    Args:
    trial_data: Dataframe of trial information with columns 'start_time' and 'end_time'
    block_data: Dataframe of block information

    Returns:
    block_data: Dataframe of block information with columns 'start_time' and 'end_time' added
    """

    # The start time of a block is the start time of the first trial in the block
    block_data["start_time"] = block_data["start_trial"].map(lambda x: trial_data.loc[x - 1, "start_time"])
    # The end time of a block is the end time of the last trial in the block
    block_data["end_time"] = block_data["end_trial"].map(lambda x: trial_data.loc[x - 1, "end_time"])

    return block_data


def validate_trial_and_block_data(trial_data, block_data, logger):
    """Run basic tests to check that trial and block data is valid."""

    logger.info("Running basic checks to check that trial and block data is valid...")

    # The number of the last trial/block must match the number of trials/blocks
    assert len(trial_data) == trial_data["trial_within_session"].max(), (
        f"The last trial number {trial_data['trial_within_session'].max()} "
        f"does not match the total number of trials {len(trial_data)}"
    )
    assert (
        len(block_data) == block_data["block"].max()
    ), f"The last block number {block_data['block'].max()} does not match the total number of blocks {len(block_data)}"
    logger.debug(
        f"Check passed: The last trial number {trial_data['trial_within_session'].max()} "
        f"matches the total number of trials {len(trial_data)}"
    )
    logger.debug(
        f"Check passed: The last block number {block_data['block'].max()} "
        f"matches the total number of blocks {len(block_data)}"
    )

    # All trial numbers must be unique and match the range 1 to [num trials in session]
    assert set(trial_data["trial_within_session"]) == set(
        range(1, len(trial_data) + 1)
    ), f"Trial numbers {trial_data['trial_within_session']} do not match expected {range(1, len(trial_data) + 1)}"
    logger.debug(f"Check passed: all trial numbers are unique and match the range 1 to {len(trial_data)}")

    # All block numbers must be unique and match the range 1 to [num blocks in session]
    assert set(block_data["block"]) == set(
        range(1, len(block_data) + 1)
    ), f"Block numbers {block_data['block']} do not match expected {range(1, len(block_data) + 1)}"
    logger.debug(f"Check passed: all block numbers are unique and match the range 1 to {len(block_data)}")

    # There must be a legitimate reward value (1 or 0) for all trials
    assert set(trial_data["reward"]).issubset(
        {0, 1}
    ), f"Not all trials have a legitimate reward value (1 or 0)! Got values: {set(trial_data['reward'])}"
    logger.debug("Check passed: All trials have a legitimate reward value (1 or 0)")

    # There must be a legitimate p(reward) value for each block at ports A, B, and C
    assert block_data[["pA", "pB", "pC"]].apply(lambda col: col.between(0, 100)).all().all(), (
        "Not all blocks have a legitimate value for pA, pB, or pC (value in range 0-100)! "
        f"Got {block_data[['pA', 'pB', 'pC']]}"
    )
    logger.debug("Check passed: All blocks have a legitimate value for pA, pB, or pC (value in range 0-100)")

    # There must be a not-null maze_configuration for each block
    assert (
        not block_data["maze_configuration"].isnull().any()
    ), f"Not all blocks have a maze configuration! Got {block_data['maze_configuration']}"
    logger.debug("Check passed: All blocks have a non-null maze configuration")

    # There must be a valid task type for each block
    assert (
        block_data["task_type"].isin(["probability change", "barrier change"]).all()
    ), f"Task types must be either 'probability change' or 'barrier change', got {block_data['task_type']}"
    # The task type must be the same for all blocks in the epoch
    assert (
        block_data["task_type"].nunique() == 1
    ), f"All blocks in an epoch must have the same task type! Got {block_data['task_type']}"
    logger.debug(f"Check passed: All blocks have the same task_type ({block_data['task_type'][0]})")

    # In a probability change session, reward probabilities vary and maze configs do not
    if block_data["task_type"].iloc[0] == "probability change":
        # All maze configs should be the same
        assert block_data["maze_configuration"].nunique() == 1, (
            "All maze configurations must be the same for a probability change session! "
            f"Got {block_data['maze_configuration']}"
        )
        # Reward probabilities should vary
        # We check for any changes instead of all because of cases like where we have only 2
        # blocks and only 2 of the probabilities change (e.g. pA and pB switch, pC stays the same)
        assert any([block_data["pA"].nunique() > 1, block_data["pB"].nunique() > 1, block_data["pC"].nunique() > 1]), (
            "pA, pB, or pC must vary in a probability change session. "
            f"Got pA={block_data['pA'].tolist()}, pB={block_data['pB'].tolist()}, pC={block_data['pC'].tolist()}"
        )
        logger.debug(
            "Check passed: All maze configurations are the same and all reward probabilities vary across blocks"
        )
    # In a barrier change session, maze configs vary and reward probabilities do not
    elif block_data["task_type"].iloc[0] == "barrier change":
        # All reward probabilities should be the same for all blocks
        assert (
            block_data["pA"].nunique() == 1
        ), f"pA must not vary in a barrier change session, but got pA={block_data['pA'].tolist()}"
        assert (
            block_data["pB"].nunique() == 1
        ), f"pB must not vary in a barrier change session, but got pB={block_data['pB'].tolist()}"
        assert (
            block_data["pC"].nunique() == 1
        ), f"pC must not vary in a barrier change session, but got pC={block_data['pC'].tolist()}"
        # Maze configurations should be different for each block
        assert block_data["maze_configuration"].nunique() == len(block_data), (
            f"Expected {len(block_data)} maze configurations for {len(block_data)} blocks, "
            f"got {block_data['maze_configuration'].nunique()} configs: {block_data['maze_configuration']}"
        )
        logger.debug(
            "Check passed: All maze configurations differ and all reward probabilities stay the same across blocks"
        )

    summed_trials = 0
    # Check trials within each block
    logger.debug("Checking trials within each block...")
    for _, block in block_data.iterrows():
        block_trials = trial_data[trial_data["block"] == block["block"]]
        trial_numbers = block_trials["trial_within_block"]

        # All trial numbers in the block must be unique and match the range 1 to [num trials in block]
        num_trials_expected = block["num_trials"]
        num_trials_expected_2 = block["end_trial"] - block["start_trial"] + 1
        assert len(trial_numbers.unique()) == num_trials_expected == num_trials_expected_2, (
            f"Expected num of trials in block ({num_trials_expected}) "
            f"= block end trial-start trial ({num_trials_expected_2}) "
            f"= number of unique trial_within_block numbers ({len(trial_numbers.unique())})"
        )
        assert set(trial_numbers) == set(range(1, int(num_trials_expected) + 1)), (
            f"Trial numbers in this block {trial_numbers} "
            f"did not match expected {range(1, int(num_trials_expected) + 1)}!"
        )
        logger.debug(f"All trial numbers in this block are unique and match the range 1 to {num_trials_expected}")

        # Check time alignment between trials and blocks
        first_trial = block_trials.loc[block_trials["trial_within_block"].idxmin()]
        last_trial = block_trials.loc[block_trials["trial_within_block"].idxmax()]
        block_start = block["start_time"]
        block_end = block["end_time"]

        assert (
            first_trial["start_time"] == block_start
        ), f"First trial start {first_trial['start_time']} does not match block start {block_start}"
        assert (
            last_trial["end_time"] == block_end
        ), f"Last trial end {last_trial['end_time']} does not match block end {block_end}"
        logger.debug(f"The start time of the first trial in the block matches the block start {block_start}")
        logger.debug(f"The end time of the last trial in the block matches the block end {block_end}")

        # Ensure trial times are within block bounds
        assert (
            block_trials["start_time"].between(block_start, block_end).all()
        ), f"Some trial start_times are outside block bounds ({block_start} to {block_end})"
        assert (
            block_trials["end_time"].between(block_start, block_end).all()
        ), f"Some trial end_times are outside block bounds ({block_start} to {block_end})"
        logger.debug("The start and end time of each trial in the block is within the block time bounds")

        # Ensure poke_in_time and poke_out_time are within trial bounds
        assert (
            block_trials["poke_in_time"].between(block_trials["start_time"], block_trials["end_time"]).all()
        ), "Some poke_in_times are outside trial bounds (start_time to end_time)"
        assert (
            block_trials["poke_out_time"] == block_trials["end_time"]
        ).all(), "Some poke_out_times do not match the trial end_time"
        logger.debug("The poke_in and poke_out time for each trial in the block is within the trial time bounds")

        summed_trials += num_trials_expected

    # The summed number of trials in each block must match the total number of trials
    assert summed_trials == len(trial_data), (
        f"Expected the summed number of trials in each block ({summed_trials}) "
        f"to match the total number of trials ({len(trial_data)})"
    )
    logger.debug(f"The number of trials in each block sums to the total number of trials {len(trial_data)}")
    logger.info("All checks passed!")


def validate_poke_timestamps(trial_data, logger):
    """
    Validate that the DIO poke_in_time and poke_out_time matches the statescript
    poke_in_time and poke_out_time for each trial, after converting units.
    """

    logger.info(
        "Checking that the DIO poke_in_time and poke_out_time matches the statescript "
        "poke_in_time and poke_out_time for each trial, after converting units."
    )

    # Get the time of the first poke_in so we can convert all other timestamps to be relative to this
    first_poke_in_DIO = trial_data.loc[0, "poke_in_time"]
    first_poke_in_statescript = trial_data.loc[0, "poke_in_time_statescript"]

    logger.debug(f"First DIO poke time: {first_poke_in_DIO}")
    logger.debug(f"First statescript poke time: {first_poke_in_statescript}")

    # Get relative DIO poke_in and poke_out times, convert to ms to match statescript times
    DIO_poke_in_times = (trial_data["poke_in_time"] - first_poke_in_DIO) * 1000
    DIO_poke_out_times = (trial_data["poke_out_time"] - first_poke_in_DIO) * 1000

    # Get relative statescript poke_in and poke_out times
    statescript_poke_in_times = trial_data["poke_in_time_statescript"] - first_poke_in_statescript
    statescript_poke_out_times = trial_data["poke_out_time_statescript"] - first_poke_in_statescript

    # Make sure DIO and statescript times are close (enough) together.

    # It is expected for the timestamps to drift apart over the course
    # of a session (drifting by roughly 0 to 0.5 ms per trial).
    # By the end of a session, the DIO and statescript timestamps may be up to ~70ms apart.
    # Reduce warning_tol_ms to a lower value to watch this happen.
    # Because of this, warning_tol_ms is currently set to 100ms, which should
    # be high enough to only warn about variations larger than this expected drift.
    warning_tol_ms = 100

    # The ValueError for diff > error tolerance is commented out because
    # we have some weird stuff going on in BraveLu20240519 epoch 3 and
    # I want to complain about it but not break.
    # Apparently trodes crashed during this session. Investigate further.
    error_tol_ms = 1000

    logger.debug("Printing relative differences in DIO and statescript timestamps.")
    logger.debug(
        "It is expected for the timestamps to drift apart over the course of a session "
        "(drifting by roughly 0 to 0.5 ms per trial, up to ~70ms apart by end of session)."
    )
    logger.debug("Differences bigger than this expected drift should be investigated.")

    # Check poke_in times
    for i, (DIO_poke, ss_poke, port) in enumerate(
        zip(DIO_poke_in_times, statescript_poke_in_times, trial_data["end_port"]), start=1
    ):
        diff = abs(DIO_poke - ss_poke)
        if diff > error_tol_ms:
            # Log big differences at error level! These are definitely concerning
            logger.error(
                f"Trial {i}, port {port}: "
                f"DIO poke_in at {DIO_poke:.1f} and statescript poke_in at {ss_poke} "
                f"are {diff:.1f} ms apart, exceeds error tolerance of {error_tol_ms} ms"
            )
            # raise ValueError(f"Trial {i}: DIO poke_in at {DIO_poke:.1f} and statescript poke_in at {ss_poke} "
            #                  f"are {diff:.1f} ms apart, exceeds error tolerance of {error_tol_ms} ms")
        elif diff > warning_tol_ms:
            # Log medium differences at warning level! These should be investigated further
            logger.warning(
                f"Trial {i}, port {port}: "
                f"DIO poke_in at {DIO_poke:.1f} and statescript poke_in at {ss_poke} "
                f"are {diff:.1f} ms apart, exceeds warning tolerance of {warning_tol_ms} ms"
            )
        else:
            # Log little differences at debug level, these are expected due to drift
            logger.debug(
                f"Trial {i}, port {port}: "
                f"DIO poke_in at {DIO_poke:.1f} and statescript poke_in at {ss_poke} "
                f"are {diff:.1f} ms apart. "
            )

    # Check poke_out times
    for i, (DIO_poke, ss_poke, port) in enumerate(
        zip(DIO_poke_out_times, statescript_poke_out_times, trial_data["end_port"]), start=1
    ):
        diff = abs(DIO_poke - ss_poke)
        if diff > error_tol_ms:
            # Log big differences at error level! These are definitely concerning
            logger.error(
                f"Trial {i}, port {port}: "
                f"DIO poke_out at {DIO_poke:.1f} and statescript poke_out at {ss_poke} "
                f"are {diff:.1f} ms apart, exceeds error tolerance of {error_tol_ms} ms"
            )
            # raise ValueError(f"Trial {i}: DIO poke_out at {DIO_poke:.1f} and statescript poke_out at {ss_poke} "
            #                 f"are {diff:.1f} ms apart, exceeds error tolerance of {error_tol_ms} ms")
        elif diff > warning_tol_ms:
            # Log medium differences at warning level! These should be investigated further
            logger.warning(
                f"Trial {i}, port {port}: "
                f"DIO poke_out at {DIO_poke:.1f} and statescript poke_out at {ss_poke} "
                f"are {diff:.1f} ms apart, exceeds warning tolerance of {warning_tol_ms} ms"
            )
        else:
            # Log little differences at debug level, these are expected due to drift
            logger.debug(
                f"Trial {i}, port {port}: "
                f"DIO poke_out at {DIO_poke:.1f} and statescript poke_out at {ss_poke} "
                f"are {diff:.1f} ms apart. "
            )


def get_barrier_locations_from_excel(excel_data):
    """
    Load barrier locations from all rows in excel_data,
    where each row is a session.

    Args:
    excel_data: dataframe of info for this experiment \
    (originally read from the excel sheet of experimental notes)

    If there are multiple sessions, return a list of lists of sets:
    where the first sub-list is for each run session that day, 
    second sub-list is for each block of the session.
    
    If there is a single session, return a lists of sets:
    where each set is for each block of the session.
    """

    # Helper to read the sets of barrier locations from the excel sheet
    def extract_sets_from_string(value):
        if isinstance(value, str):
            # Regular expression to find all the sets in the string
            sets = re.findall(r"\{.*?\}", value)
            return [ast.literal_eval(s) for s in sets]
        return None

    # Get barrier locations as a list of lists of sets
    list_of_barrier_sets = excel_data["barrier location"].apply(extract_sets_from_string).tolist()

    # If excel_data is for only one session, remove the outer list
    if excel_data.shape[0] == 1 and len(list_of_barrier_sets) == 1:
        return list_of_barrier_sets[0]
    # Else return a list of lists, where each outer list is for a different session
    else:
        return list_of_barrier_sets


def add_barrier_locations_to_block_data(block_data, excel_data, session_type, logger):
    """
    Add "maze_configuration" column to block_data.

    Args:
    block_data: Dataframe of information for each block in this epoch
    excel_data: Dataframe of info for this experiment, with column "barrier_location"
    session_type: "barrier change" or "probability change"
    logger: Logger to track progress

    Returns:
    block_data with added "maze_configuration" column
    """

    logger.info("Adding a 'maze_configuration' column to block data")

    def barrier_set_to_string(set):
        """
        Helper to convert a set of ints to a sorted, comma-separated string.
        Used for going from a set of barrier locations to a string
        maze configuration that plays nicely with NWB and spyglass.
        """
        return ",".join(map(str, sorted(set)))

    # Read barrier locations from excel data
    maze_configs = get_barrier_locations_from_excel(excel_data)

    # Make sure the number of blocks matches the number of loaded maze configurations
    if len(block_data) != len(maze_configs):
        # If this is a probability change session, we have a single maze configuration
        # to be used for all blocks. If so, duplicate it so we have one maze per block.
        if len(maze_configs) == 1 and session_type == "probability change":
            logger.debug("Found a single maze configuration for this session.")
            logger.debug("This is a probability change session, so assigning it to all blocks.")
            maze_configs = maze_configs * len(block_data)
        else:
            logger.error(
                f"There are {len(block_data)} blocks, but {len(maze_configs)} maze configurations "
                "From the excel data. There should be exactly one maze configuration per block, "
                "or a single maze configuration if this is a probability change session."
            )
            raise ValueError(
                f"There are {len(block_data)} blocks, but {len(maze_configs)} maze configurations "
                "From the excel data. There should be exactly one maze configuration per block, "
                "or a single maze configuration if this is a probability change session."
            )
    else:
        logger.debug(f"Found {len(maze_configs)} maze configs for {len(block_data)} blocks!")

    # Convert each maze config from a set to a sorted, comma separated string for compatibility
    maze_configs = [barrier_set_to_string(maze) for maze in maze_configs]

    for block_num, maze in enumerate(maze_configs, start=1):
        logger.info(f"Block {block_num} maze: {maze}")

    # Add the maze configuration for each block
    block_data["maze_configuration"] = maze_configs
    return block_data


def parse_state_script_log(statescriptlog, DIO_events, excel_data_for_epoch, logger):
    """
    Read and parse the stateScriptLog file and align it to DIO events
    for a given behavioral epoch. Get barrier locations and
    other info (if needed) from excel data.

    Args:
    statescriptlog: tuple where statescriptlog[0] is a big string containing the log, \
        statescriptlog[1] is the AssociatedFiles object (unused)
    DIO_events: dict of event_name: (data, timestamps) for each named DIO event
    excel_data_for_epoch: dataframe of info for this epoch \
        (originally read from the excel sheet of experimental notes)
    logger: Logger to track progress

    Returns:
    trial_df: Dataframe of information for each trial in this epoch
    block_data: Dataframe of information for each block in this epoch
    """

    logger.info("Parsing stateScriptLog...")

    nosepoke_events = []
    behavior_data = []
    block_ends = []
    session_end = None

    # Read the statescriptlog line by line
    for line in str(statescriptlog).splitlines():
        # Ignore lines starting with '#'
        if line.startswith("#"):
            continue

        # Find all poke_in and poke_out events
        for match in poke_in_regex.finditer(line):
            nosepoke_events.append(
                {"timestamp": int(match.group(1)), "event_name": "poke_in", "port": int(match.group(2))}
            )
        for match in poke_out_regex.finditer(line):
            nosepoke_events.append(
                {"timestamp": int(match.group(1)), "event_name": "poke_out", "port": int(match.group(2))}
            )
        # Find behavioral data and reward info
        for match in behavior_data_regex.finditer(line):
            behavior_data.append(
                {"timestamp": int(match.group(1)), "name": match.group(2), "value": int(match.group(3))}
            )
        # Check for block or session end timestamps
        for match in block_end_regex.finditer(line):
            block_ends.append({"timestamp": int(match.group(1))})
        for match in session_end_regex.finditer(line):
            session_end = int(match.group(1))

    # Make sure we have the complete set of information for each trial
    variable_counts = Counter(event["name"] for event in behavior_data)

    # If we don't have the same amount of information for each trial, this is probably
    # a legacy stateScriptLog where 'ifDelay' was printed only when the value changed,
    # instead of with the trial information. So we do some processing to create the expected structure.
    if len(set(variable_counts.values())) != 1:
        logger.warning("Mismatch in the amount of information for each trial!")
        logger.warning("Assuming this is a legacy stateScriptLog (ifDelay printed separately)...")
        behavior_data = parse_ifDelay_events_for_legacy_statescriptlog(behavior_data, logger)
        logger.info("Assigned the correct ifDelay value to each trial.")

    # Create dataframes of trial and block data based on the stateScriptLog
    trial_data, block_data = parse_trial_and_block_data(behavior_data, block_ends, logger)

    # Align statescript and DIO nosepokes and create nosepoke dataframe including only nosepoke events at a new port
    # with both statescript timestamps (trodes time) and DIO timestamps (unix time)
    nosepoke_DIOs = {
        key: value for key, value in DIO_events.items() if key in ["wellA_poke", "wellB_poke", "wellC_poke"]
    }
    nosepoke_df = parse_nosepoke_events(nosepoke_events, nosepoke_DIOs, logger)

    # Add combined nosepoke timestamps (both statescript and DIO) to the trial dataframe
    trial_df = combine_nosepoke_and_trial_data(nosepoke_df, trial_data, session_end, logger)
    
    # Check the ifDelays!!

    # Add reward pump timestamps from DIOs to the combined dataframe
    reward_DIOs = {key: value for key, value in DIO_events.items() if key in ["wellA_pump", "wellB_pump", "wellC_pump"]}
    trial_df = combine_reward_and_trial_data(trial_df, reward_DIOs, logger)

    # Use block data to determine if this is a probability change or barrier change session
    session_type = determine_session_type(block_data)
    logger.info(f"This is a {session_type} session.")
    # Add the session type as a column to the block data
    block_data["task_type"] = session_type

    # If this is a barrier change session, the statescript does not accurately reflect block changes
    # Instead, a DIO ("barrier_shift") is pressed to mark the trial in which the barrier is shifted
    # For early sessions, the "barrier_shift" DIO didn't exist yet so this is recorded in the
    # "experimental notes" excel sheet
    if session_type == "barrier change":
        trial_data, block_data = adjust_block_start_trials(
            trial_df, block_data, DIO_events, excel_data_for_epoch, logger
        )

    # Now that we have the correct start/end trial for each block, add the block start/end times
    block_data = add_block_start_end_times(trial_data, block_data)

    # Add maze configs from the excel data to the block dataframe
    block_data = add_barrier_locations_to_block_data(block_data, excel_data_for_epoch, session_type, logger)

    # Do even more basic checks to make sure trial and block data seems reasonable
    validate_trial_and_block_data(trial_data, block_data, logger)
    validate_poke_timestamps(trial_data, logger)

    return trial_df, block_data


def get_DIO_event_data(nwbfile, behavioral_event_name):
    """
    Get DIO data and timestamps from the nwbfile for a given behavioral event

    Args:
    nwbfile: NWB file containing behavioral_event DIOs in the behavioral processing module
    behavioral_event_name: named behavioral_event to access

    Returns:
    data: 1/0 data corresponding to DIO high/low for this event
    timestamps: timestamps for each data point (in unix time)
    """

    data = nwbfile.processing["behavior"]["behavioral_events"][behavioral_event_name].data[:]
    timestamps = nwbfile.processing["behavior"]["behavioral_events"][behavioral_event_name].timestamps[:]
    return data, timestamps


def parse_DIOs(behavioral_event_data, logger):
    """
    Parse behavioral event DIOs and timestamps into DIO pulses for actual events vs epoch starts
    
    Epoch starts are marked by a shared "0" data point and timestamp across all DIO events.
    Remove this event and timestamp from all DIOs so the data and timestamps for
    each DIO reflects the actual behavioral event of interest.

    Args:
    behavioral_event_data: dict of event_name: (data, timestamps) for each named DIO event, \
    including DIO data/timestamps for both "real" events and epoch starts
    logger: Logger to track progress

    Returns:
    behavioral_event_data: dict of DIO_event: (data, timestamps) for that event 
    (with DIO data/timestamps for epoch starts removed)
    epoch_start_timestamps: List of timestamps marking epoch starts
    """

    logger.info("Parsing behavioral event DIOs and timestamps into DIO pulses for actual events vs epoch starts")

    # Get timestamps shared among all behavioral events (triggered by an epoch start)
    epoch_start_timestamps = set.intersection(*[set(ts) for _, ts in behavioral_event_data.values()])
    logger.debug(f"Found epoch start timestamps: {sorted(epoch_start_timestamps)}")

    # Remove epoch start data/timestamps so we are left with only DIOs triggered by real behavioral events
    behavioral_event_data = {
        key: (
            [d for d, ts in zip(data, timestamps) if ts not in epoch_start_timestamps],
            [ts for ts in timestamps if ts not in epoch_start_timestamps],
        )
        for key, (data, timestamps) in behavioral_event_data.items()
    }

    # After removing extra 0s for epoch starts, check that each 1 has a corresponding 0
    for key, (data, timestamps) in behavioral_event_data.items():
        for i in range(len(data) - 1):
            if not ((data[i] == 1 and data[i + 1] == 0) or (data[i] == 0 and data[i + 1] == 1)):
                # For now, just warn about it - it may end up being ok
                # It may be due to a session timeout that cut off a poke_out - we can deal with that elsewhere
                logger.warning(
                    f"{key} has mismatched DIO {data[i], data[i+1]} at timestamps {timestamps[i], timestamps[i+1]}"
                )
                logger.warning("This may be due to a session timeout that cut off a poke_out")

    return behavioral_event_data, sorted(epoch_start_timestamps)


def get_data_from_excel_sheet(excel_path, date, sheet_name="Daily configs and notes_Bandit+"):
    """
    Read the excel sheet of experimental notes and return a dataframe
    of relevant rows for this recording date.
    """

    # Read the excel sheet into a dataframe and filter for run sessions on our target date
    df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=1)
    return df[(df["date"].astype(str) == str(date)) & (df["barrier location"].notna())].reset_index(drop=True)


def add_block_and_trial_data_to_nwb(nwbfile: NWBFile, trial_data, block_data, logger, overwrite=False):
    """
    Add trial and block data to the nwbfile as timeintervals.
    If "block" and "trials" already exist in the nwbfile, complain and
    return without modifying the nwb unless overwrite=True.
    
    Args:
    nwbfile (NWBFile): The nwbfile
    trial_data: Dataframe of trial data for all run epochs in the nwbfile
    block_data: Dataframe of block data for all run epochs in the nwbfile
    logger: Logger to track progress
    overwrite (bool, optional): If we should overwrite existing trial \
    and block data in the nwbfile. Defaults to False
    """

    logger.info("Adding trial and block data to the nwbfile as timeintervals")

    # Check if a block or trials table already exists in the nwbfile
    if not overwrite and ("block" in nwbfile.intervals or "trials" in nwbfile.intervals):
        logger.error("A block or trials table already exists in the nwbfile.")
        logger.error(
            "Stopping. Run again with overwrite=True if you wish to overwrite the original block and trials table."
        )
        print("Stopping. Run again with overwrite=True if you wish to overwrite the original block and trials table.")
        return

    def get_opto_condition(delay):
        """Helper to get opto condition as a string based on delay"""
        return {1: "delay", 0: "no_delay"}.get(delay, "None")

    # Add columns for block data to the NWB file
    block_table = nwbfile.create_time_intervals(
        name="block",
        description="The block within a session. "
        "Each block is defined by a maze configuration and set of reward probabilities.",
    )
    block_table.add_column(name="epoch", description="The epoch (session) this block is in")
    block_table.add_column(name="block", description="The block number within the session")
    block_table.add_column(
        name="maze_configuration",
        description="The maze configuration for each block, "
        "defined by the set of hexes in the maze where barriers are placed.",
    )
    block_table.add_column(name="pA", description="The probability of reward at port A")
    block_table.add_column(name="pB", description="The probability of reward at port B")
    block_table.add_column(name="pC", description="The probability of reward at port C")
    block_table.add_column(name="start_trial", description="The first trial in this block")
    block_table.add_column(name="end_trial", description="The last trial in this block")
    block_table.add_column(name="num_trials", description="The number of trials in this block")
    block_table.add_column(name="task_type", description="The session type ('barrier change' or 'probability change'")

    # Add columns for trial data to the NWB file
    nwbfile.add_trial_column(name="epoch", description="The epoch (session) this trial is in")
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

    # Add each block to the block table in the NWB
    for idx, block in block_data.iterrows():
        block_table.add_row(
            epoch=block["epoch"],
            block=block["block"],
            maze_configuration=block["maze_configuration"],
            pA=block["pA"],
            pB=block["pB"],
            pC=block["pC"],
            start_trial=block["start_trial"],
            end_trial=block["end_trial"],
            num_trials=block["num_trials"],
            task_type=block["task_type"],
            start_time=block["start_time"],
            stop_time=block["end_time"],
        )

    # Add each trial to the NWB
    for idx, trial in trial_data.iterrows():
        nwbfile.add_trial(
            epoch=trial["epoch"],
            block=trial["block"],
            trial_within_block=trial["trial_within_block"],
            trial_within_epoch=trial["trial_within_session"],
            start_port=trial["start_port"],
            end_port=trial["end_port"],
            reward=trial["reward"],
            opto_condition=get_opto_condition(trial["delay"]),
            duration=trial["duration"],
            poke_in=trial["poke_in_time"],
            poke_out=trial["poke_out_time"],
            start_time=trial["start_time"],
            stop_time=trial["end_time"],
        )


def add_hex_centroids_to_nwb(nwbfile: NWBFile, hex_centroids_file_path, logger):
    """
    Read hex centroids from a csv file with columns hex, x, y (and optionally x_meters and y_meters)
    and add them to the nwbfile in the behavior processing module.
    """

    # Try to load centroids and complain if we can't find the file
    try:
        hex_centroids = pd.read_csv(hex_centroids_file_path)
    except FileNotFoundError:
        logger.error(f"The file '{hex_centroids_file_path}' was not found! Skipping adding hex centroids.")
        return

    logger.info("Adding hex centroids to the nwbfile in the behavior processing module")

    # Check that the hex centroids file is in the format we expect
    num_hexes = 49
    if len(hex_centroids) != num_hexes:
        logger.error(f"Expected {num_hexes} centroids in the hex centroids file, got {len(hex_centroids)}!!!")
    else:
        logger.debug(f"Found the expected number of hex centroids in the centroids file ({num_hexes})")

    # Make sure the file includes columns 'hex', 'x', 'y'
    required_columns = {"hex", "x", "y"}
    actual_columns = set(hex_centroids.columns)
    if not required_columns.issubset(actual_columns):
        logger.error(f"Expected {required_columns} columns in the hex centroids file, got {actual_columns}!!")
        logger.error("Skipping adding centroids to the nwb")
        return
    else:
        logger.debug(f"Found expected columns {required_columns} in the hex centroids file")

    # Set up the hex centroids table with columns 'hex', 'x', 'y'
    centroids_table = DynamicTable(
        name="hex_centroids", description="Centroids of each hex in the maze (in video pixel coordinates)"
    )
    centroids_table.add_column(name="hex", description="The ID of the hex in the maze (1-49)")
    centroids_table.add_column(
        name="x", description="The x coordinate of the center of the hex (in video pixel coordinates)"
    )
    centroids_table.add_column(
        name="y", description="The y coordinate of the center of the hex (in video pixel coordinates)"
    )

    # If we also have columns for 'x_meters' and 'y_meters', add those too
    extra_columns = {"x_meters", "y_meters"}
    if extra_columns.issubset(actual_columns):
        logger.debug(f"Found extra hex centroids columns {extra_columns}. Adding those also.")
        centroids_table.add_column(name="x_meters", description="The x coordinate of the center of the hex (in meters)")
        centroids_table.add_column(name="y_meters", description="The y coordinate of the center of the hex (in meters)")

    # Add the hex centroids (make sure we keep only desired columns and in the correct order)
    desired_column_order = ["hex", "x", "y", "x_meters", "y_meters"]
    columns_to_keep = list(filter(lambda col: col in hex_centroids.columns, desired_column_order))
    hex_centroids = hex_centroids[columns_to_keep]
    for _, row in hex_centroids.iterrows():
        centroids_table.add_row(**row.to_dict())

    # If it doesn't exist already, make a processing module for behavior and add to the nwbfile
    if "behavior" not in nwbfile.processing:
        logger.debug("Creating nwb behavior processing module for position data")
        nwbfile.create_processing_module(name="behavior", description="Contains all behavior-related data")

    nwbfile.processing["behavior"].add(centroids_table)


######## Functions the user will actually call ####################


def add_behavioral_data_to_nwb(
    nwb_path,
    excel_path,
    sheet_name="Daily configs and notes_Bandit+",
    hex_centroids_file_path=None,
    save_type=None,
    overwrite=False,
):
    """
    Given an nwbfile, parse behavioral data into trial and block 
    dataframes for each run epoch and save them for future use.
    Also adds the hex centroids to the nwbfile in the behavior processing module.
    
    Args:
    nwb_path: Path to a Frank Lab nwbfile with statescriptlogs saved as AssociatedFiles \
    objects and behavioral event DIOs in the behavior processing module
    excel_path: Path to an excel sheet of behavioral notes for the experiment, \
    including 'date' and 'barrier location' column
    sheet_name: Name of sheet to read from in excel. \
    Defaults to 'Daily configs and notes_Bandit+' if not specified.
    hex_centroids_file_path: Path to csv file with hex ID and centroids in video pixel coordinates, \
    with columns 'hex', 'x', and 'y'.
    save_type: "nwb", "pickle", or "csv". \
    "nwb" to save the data as timeintervals in the nwbfile. \
    "pickle" or "csv" to save the trial and block dataframes as .pkl or .csv files. " \
    Any combination of save_types is allowed, e.g. save_type="pickle,nwb"
    overwrite: If we should overwrite existing trial and block data in the nwbfile, \
    if it exists. Applies only to save_type="nwb". Defaults to False
    """

    # Get the session ID from this nwbfile so we can name our log files correctly
    with NWBHDF5IO(nwb_path, mode="r") as io:
        nwbfile = io.read()
        session_id = nwbfile.session_id

    # Create directory for conversion log files
    log_dir = f"{session_id}_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Then set up logging with paths to log files
    info_log_file = Path(log_dir) / f"{session_id}_info_log.log"
    warning_log_file = Path(log_dir) / f"{session_id}_warning_logs.log"
    debug_log_file = Path(log_dir) / f"{session_id}_debug_log.log"
    logger = setup_logger("conversion_log", info_log_file, warning_log_file, debug_log_file)

    # Hack to remove trials/block table and centroids if we need to overwrite them
    removed_old_data_from_nwb = False
    if overwrite and "nwb" in save_type:
        with h5py.File(nwb_path, "r+") as f:
            if "block" in f["intervals"]:
                logger.info("A block table already exists in the nwbfile!")
                logger.info("The original block table in the nwb will be deleted and overwritten.")
                del f["intervals/block"]
            if "trials" in f["intervals"]:
                logger.info("A trials table already exists in the nwbfile!")
                logger.info("The original trials table in the nwb will be deleted and overwritten.")
                del f["intervals/trials"]
            if "hex_centroids" in f["processing/behavior"]:
                logger.info("A hex centroids table already exists in the nwbfile!")
                logger.info("Deleting hex centroids table from the nwbfile.")
                del f["processing/behavior/hex_centroids"]
            removed_old_data_from_nwb = True

    # Now actually open the file in append mode to do behavior parsing
    with NWBHDF5IO(nwb_path, mode="r+") as io:
        nwbfile = io.read()
        logger.info(f"Parsing behavior for {nwbfile.session_id} ...")
        logger.info(f"Using source script jdb_to_nwb {__version__}")

        # Get session date assuming session ID is in format rat_date
        session_date = nwbfile.session_id.split("_")[-1]
        logger.debug(f"Session date: {session_date}")

        # Read rows from excel sheets for run sessions on this date
        excel_data = get_data_from_excel_sheet(excel_path, session_date, sheet_name)

        # Get epoch table defining session boundaries with columns "start_time", "end_time", and "tags"
        epoch_table = nwbfile.intervals["epochs"][:]

        # Filter epochs to include only run sessions (should include "r" in the tags)
        run_epochs = epoch_table[epoch_table["tags"].apply(lambda x: "r" in x[0])]
        # Filtering epochs for run sessions should be the same as taking every other epoch
        assert run_epochs.equals(epoch_table.iloc[1::2]), "Run epochs should be every other epoch!"
        logger.debug(f"Run epochs: {run_epochs}")

        # Get all stateScriptLogs from run sessions (ignoring logs from sleep sessions)
        module = nwbfile.get_processing_module("associated_files")
        run_statescript_logs = {
            name: log for name, log in module.data_interfaces.items() if name.startswith("statescript r")
        }
        assert len(run_statescript_logs) == len(run_epochs) == len(excel_data), (
            f"Found {len(run_statescript_logs)} stateScriptLogs, {len(run_epochs)} run epochs, "
            f"and {len(excel_data)} run sessions from the excel sheet. \n"
            "Expected all of these to be the same length"
        )
        logger.debug(f"Found {len(run_statescript_logs)} statescriptlogs for {len(run_epochs)} run epochs")

        # Get behavioral events from the nwbfile as a dict of (data, timestamps) for each named behavioral event
        behavioral_events = [
            "barrier_shift",
            "wellA_poke",
            "wellA_pump",
            "wellB_poke",
            "wellB_pump",
            "wellC_poke",
            "wellC_pump",
        ]
        behavioral_event_data = {event: get_DIO_event_data(nwbfile, event) for event in behavioral_events}

        # Separate DIOs into those for actual behavioral events vs epoch starts
        behavioral_event_data, epoch_start_timestamps = parse_DIOs(behavioral_event_data, logger)

        # Check that we have the expected amount of epoch starts
        # NOTE: epoch_start_timestamps from the DIO pulses lag the timestamps in the epoch_table
        # by ~1 second to ~1 minute - check where this discrepancy comes from and which one to use!
        assert len(epoch_start_timestamps) == len(epoch_table), (
            f"Found {len(epoch_start_timestamps)} epoch start timestamps for {len(epoch_table)} epochs. "
            "Expected these to match!"
        )
        logger.debug(f"Found {len(epoch_start_timestamps)} epoch start timestamps for {len(epoch_table)} epochs")

        # Set up lists to store block and trial data for each epoch
        block_dataframes = []
        trial_dataframes = []

        # Parse behavioral data for each epoch using the statescriptlog and align to DIOs
        run_session_num = 0
        for idx, epoch in run_epochs.iterrows():

            # We log the epoch breaks at 'WARNING' level so they show up in all logs
            logger.warning(f"\n\n---------------------------- EPOCH {epoch.name} LOGS ----------------------------\n")
            logger.info(f"Parsing statescript for epoch {epoch.name} ...")
            # Get the statescriptlog for this epoch
            statescriptlog = list(run_statescript_logs.items())[run_session_num]

            # Filter DIOs to only include those in this epoch
            # NOTE: maybe replace epoch.start_time and epoch.stop_time with DIO times
            # (see above comment for reasoning, and commented line below for how to make this switch)
            # So far it does not seem to make a difference in our results, but something to consider.
            DIO_events_in_epoch = {
                event: (list(filtered_data), list(filtered_timestamps))
                for event, (data, timestamps) in behavioral_event_data.items()
                if (
                    filtered := [
                        (d, ts) for d, ts in zip(data, timestamps) if epoch.start_time <= ts <= epoch.stop_time
                    ]
                )
                # if (filtered := [(d, ts) for d, ts in zip(data, timestamps)
                #               if epoch_start_timestamps[idx] < ts < epoch_start_timestamps[idx+1]])
                for filtered_data, filtered_timestamps in [zip(*filtered)]
            }

            # Filter excel data for this epoch
            excel_data_for_epoch = excel_data.iloc[[run_session_num]]

            # Parse statescriptlog and DIO events for this epoch into tables of trial and block data
            trial_data, block_data = parse_state_script_log(
                statescriptlog, DIO_events_in_epoch, excel_data_for_epoch, logger
            )

            # Adjustment for start time of first trial/block
            # If the epoch start is after the start time, set the start time to the epoch start.
            if epoch.start_time > trial_data.loc[0, "start_time"]:
                logger.info(
                    f"Setting start time of the first block/trial to epoch start time {epoch.start_time}, "
                    f"was previously {trial_data.loc[0, 'start_time']}"
                )
                trial_data.loc[0, "start_time"] = epoch.start_time
                block_data.loc[0, "start_time"] = epoch.start_time
                trial_data.loc[0, "duration"] = trial_data.loc[0, "end_time"] - trial_data.loc[0, "start_time"]

            # Add epoch column to the dataframes
            trial_data["epoch"] = epoch.name
            block_data["epoch"] = epoch.name

            # Reorder columns so epoch comes first
            trial_data = trial_data[["epoch"] + [col for col in trial_data.columns if col != "epoch"]]
            block_data = block_data[["epoch"] + [col for col in block_data.columns if col != "epoch"]]

            # Append the dataframes for this epoch
            trial_dataframes.append(trial_data)
            block_dataframes.append(block_data)

            logger.debug(f"Trial and block data for epoch {epoch.name}:")
            logger.debug("\n" + trial_data.to_string(index=False))
            logger.debug("\n" + block_data.to_string(index=False))

            run_session_num += 1

        trial_data_all_epochs = pd.concat(trial_dataframes, ignore_index=True)
        block_data_all_epochs = pd.concat(block_dataframes, ignore_index=True)

        if "pickle" in save_type:
            trial_data_all_epochs.to_pickle(f"{nwbfile.session_id}_trial_data.pkl")
            block_data_all_epochs.to_pickle(f"{nwbfile.session_id}_block_data.pkl")
        if "csv" in save_type:
            trial_data_all_epochs.to_csv(f"{nwbfile.session_id}_trial_data.csv", index=False)
            block_data_all_epochs.to_csv(f"{nwbfile.session_id}_block_data.csv", index=False)
        if "nwb" in save_type:
            # Add the trial and block tables to the original nwbfile
            add_block_and_trial_data_to_nwb(nwbfile, trial_data_all_epochs, block_data_all_epochs, logger, overwrite)
            if hex_centroids_file_path is not None:
                add_hex_centroids_to_nwb(nwbfile, hex_centroids_file_path, logger)
            # Write to the nwb if either overwrite=True,
            # or overwrite=False but there was no existing block and trial data (so we are adding not overwriting)
            if overwrite or not removed_old_data_from_nwb:
                io.write(nwbfile)


def delete_blocks_and_trials_from_nwb(nwb_path):
    """
    Delete block and trials tables (stored as TimeIntervals) from an nwbfile if they exist.
    Modifies the file in-place. Note that this will not actually reduce the file size
    due to limitations in the HDF5 format
    """
    with h5py.File(nwb_path, "r+") as f:
        if "block" in f["intervals"]:
            print("Deleting block tale from the nwbfile")
            del f["intervals/block"]
        else:
            print("No block table to delete.")
        if "trials" in f["intervals"]:
            print("Deleting trials table from the nwbfile")
            del f["intervals/trials"]
        else:
            print("No trials table to delete.")


def delete_hex_centroids_from_nwb(nwb_path):
    """
    Delete hex centroids tables (stored as a DynamicTable in the behavior processing module)
    from an nwbfile if it exists. Modifies the file in-place. Note that this will
    not actually reduce the file size due to limitations in the HDF5 format.
    """
    with h5py.File(nwb_path, "r+") as f:
        if "hex_centroids" in f["processing/behavior"]:
            print("Deleting hex centroids table from the nwbfile.")
            del f["processing/behavior/hex_centroids"]
        else:
            print("No hex centroids table to delete.")
