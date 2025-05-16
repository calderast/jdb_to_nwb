import numpy as np
from scipy.interpolate import interp1d


def handle_timestamps_reset(timestamps, logger):
    """
    In our setup, arduino and video timestamps reset to 0 after 86,400,000 
    (aka 24:00:00 in ms), which happens at 12pm recording computer time. 

    Given a list of timestamps (in ms), check if the timestamps reset to 0 
    and if so, unwrap them such that they are consistently increasing.

    We expect maximum 1 reset. If we encounter any time drops that do not
    fit this expectation, error so we can figure out what went wrong.
    """
    # Helper to convert timestamp in ms to HH:MM:SS.mmm for logging
    def ms_to_hhmmss(ms):
        return f"{ms // 3600000:02}:{(ms // 60000) % 60:02}:{(ms // 1000) % 60:02}.{ms % 1000:03}"

    reset_threshold = 86_400_000 # 24:00:00 in ms

    # If all timestamps are already increasing (no reset), return as-is
    if all(t2 >= t1 for t1, t2 in zip(timestamps, timestamps[1:])):
        logger.debug("Check passed: All timestamps are increasing.")
        return timestamps

    # Otherwise, find indices where timestamps drop (potential resets)
    logger.info("Detected a potential timestamp reset. This is expected when the recording passes 12:00pm.")
    time_drops = [i for i in range(1, len(timestamps)) if timestamps[i] < timestamps[i - 1]]

    # Timestamps should only reset once. Break if >1 so we can figure out what happened
    if len(time_drops) != 1:
        logger.error(f"Expected at most one timestamp reset, found {len(time_drops)}!!")
        raise ValueError(f"Expected at most one timestamp reset, found {len(time_drops)}!!")

    reset_idx = time_drops[0]
    time_pre_reset = timestamps[reset_idx - 1]
    time_post_reset = timestamps[reset_idx]

    # Make sure the time drop matches a 24h reset
    drop = time_pre_reset - time_post_reset
    if drop < 0.9 * reset_threshold:
        logger.error(f"Timestamp drop at index {reset_idx} too small to be a 24h reset!")
        logger.error(f"Timestamp before reset={time_pre_reset} ({ms_to_hhmmss(time_pre_reset)}), "
                     f"timestamp post reset={time_post_reset} ({ms_to_hhmmss(time_post_reset)}), "
                     f"drop={drop}ms ({ms_to_hhmmss(drop)})")
        raise ValueError("Drop in timestamps too small to be a valid reset!")

    logger.debug(f"Timestamps reset at index {reset_idx}.")
    logger.info(f"Timestamp before reset: {time_pre_reset} ({ms_to_hhmmss(time_pre_reset)})")
    logger.info(f"Timestamp after reset: {time_post_reset} ({ms_to_hhmmss(time_post_reset)})")
    logger.info("Adjusting timestamps to account for reset...")

    # Adjust all timestamps after the reset so they are increasing
    timestamps = np.array(timestamps)
    adjusted_timestamps = np.concatenate([
        timestamps[:reset_idx],
        timestamps[reset_idx:] + reset_threshold
    ]).tolist()

    # Final sanity check that the adjusted timestamps are increasing
    if not all(t2 >= t1 for t1, t2 in zip(adjusted_timestamps, adjusted_timestamps[1:])):
        logger.error("Timestamps are not increasing after adjustment for 24h reset!!")
        raise AssertionError("Timestamps are not increasing after adjustment for 24h reset!")

    return adjusted_timestamps


def trim_sync_pulses(ground_truth_visits, unaligned_visits, logger):
    """
    We may have an unequal number of sync pulses (port visit times) recorded by different datastreams
    (photometry vs ephys vs arduino), if one datastream was started (or ended) before (or after) another 
    and port visits occurred during this time. Typically, this occurs when photometry was started before
    ephys/behavior (e.g. IM-1478, 07/19/2022). If this is the case, trim the longer list of port visits
    so both lists are the same length and can be used for timestamps alignment. We auto-detect if the 
    visits to be removed are at the start or the end of the longer list by matching the relative spacing 
    between port visit times for each datastream.

    Args:
    ground_truth_visits (list or np.array): List of ground truth port visit times
    unaligned_visits (list or np.array): List of unaligned port visit times

    Returns:
    tuple of arrays:
    ground_truth_visits: ground truth visit times trimmed if it was the longer list
    unaligned_visits: unaligned visit times trimmed if it was the longer list
    """

    # Ensure both lists are arrays
    visits_1, visits_2 = np.array(ground_truth_visits), np.array(unaligned_visits)
    logger.info(f"Initial number of port visits: ground truth={len(visits_1)}, unaligned={len(visits_2)}")

    # If they are already the same length, just return
    if len(visits_1) == len(visits_2):
        logger.info("Port visit lists are already the same length!")
        return visits_1, visits_2

    # Determine which list is longer
    if len(visits_1) > len(visits_2):
        long_list, short_list = visits_1, visits_2
        logger.debug("List of ground truth visits is longer; trimming it.")
    else:
        long_list, short_list = visits_2, visits_1
        logger.debug("List of unaligned visits is longer; trimming it.")

    # Get the spacing between pulses for each list
    long_diffs = np.diff(long_list)
    short_diffs = np.diff(short_list)

    # Find the best start index for the long list.
    # The best starting index is the one that best matches the relative spacing of pulses for each list.
    min_error = float("inf")
    best_start = 0

    logger.info("Finding best alignment between port visits by minimizing error in pulse spacing.")

    for i in range(len(long_diffs) - len(short_diffs) + 1):
        error = np.sum(np.abs(long_diffs[i:i+len(short_diffs)] - short_diffs))
        logger.debug(f"Trimming {i} samples from longer list. Sum of differences in pulse spacing={error}")
        if error < min_error:
            min_error = error
            best_start = i

    logger.info(f"Best alignment is found by trimming {best_start} samples from the longer list (error={min_error})")

    # We generally expect the error to be minimized by exclusively trimming pulses 
    # from the beginning of the longer list (because one datastream was started earlier). 
    # Warn if this isn't the case.
    expected_best_start = len(long_diffs) - len(short_diffs)
    if best_start != expected_best_start:
        logger.warning(f"Expected best alignment to be found by removing {expected_best_start} samples "
                       f"from the start of the longer list, but got best alignment removing {best_start} samples!")
        logger.warning("Check differences in pulse spacing in the DEBUG log to make sure this is correct, "
                       "and/or confirm this matches known experimental choices \n"
                       "(e.g. this would be the case if a datastream was stopped earlier instead of started later)")

    # Trim the longer list to match the length of the shorter list
    trimmed_list = long_list[best_start:best_start + len(short_list)]

    # Ensure correct order of return values
    return (trimmed_list, short_list) if len(visits_1) > len(visits_2) else (short_list, trimmed_list)


def align_via_interpolation(unaligned_timestamps, unaligned_visit_times, ground_truth_visit_times, logger):
    """
    Align timestamps to ground truth timestamps via interpolation using port visit times as sync pulses.
    Timestamps that fall before the first port visit or after the last port visit will be aligned via extrapolation.
    
    Automatically handles cases where the lengths of unaligned_visit_times and ground_truth_visit_times
    do not match by trimming the longer list in a way that maximizes alignment between the 2 lists.
    (see timestamps_alignment.trim_sync_pulses for more info)

    Args:
    unaligned_timestamps (list or np.array): List of timestamps to align
    unaligned_visit_times (list or np.array): List of port visit times (in the same time base as unaligned timestamps) 
    ground_truth_visit_times (list or np.array): List of ground truth port visit times (to be aligned to)
    logger: Logger to record timestamp alignment

    Returns:
    aligned_timestamps (list or np.array): Timestamps aligned to the ground_truth_visit_times
    """
    
    # Check that we have the same number of ground truth visit times and unaligned visit times
    # If we don't, warn the user and fix it so we can proceed with alignment
    if len(ground_truth_visit_times) != len(unaligned_visit_times):
        logger.warning(f"There are {len(ground_truth_visit_times)} ground truth visit times but "
                       f"{len(unaligned_visit_times)} unaligned visit times!")
        logger.warning("The longer list will be trimmed by matching relative spacing between visits.")
        ground_truth_visit_times, unaligned_visit_times = trim_sync_pulses(ground_truth_visit_times, 
                                                                           unaligned_visit_times, logger)
    else:
        logger.debug(f"There are {len(unaligned_visit_times)} visit times from both datastreams "
                     "to be used for alignment.")

    # Create an interpolation function to go from unaligned visit times to ground truth visit times
    # Extrapolate timestamps that fall before the first visit or after the last visit
    logger.info("Aligning timestamps to ground truth port visit times via linear interpolation")
    interp_func = interp1d(unaligned_visit_times, ground_truth_visit_times, kind='linear', fill_value='extrapolate')
    aligned_timestamps = interp_func(unaligned_timestamps)

    return aligned_timestamps
