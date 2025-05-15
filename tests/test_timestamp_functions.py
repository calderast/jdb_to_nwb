import pytest
import numpy as np
from jdb_to_nwb.timestamps_alignment import handle_timestamps_reset, trim_sync_pulses


def test_handle_timestamps_with_single_reset(dummy_logger):
    """ 
    Test handle_timestamps_reset properly adjusts timestamps that approach 24:00:00 then reset to 0
    """
    # Simulate timestamps before reset: start at 86_100_000 (23:55:00)
    pre_reset_times = [86_100_000 + i * 10 for i in range(10)]
    # Simulate timestamps after reset: reset to near 0
    post_reset_times = [(i+1) * 1000 for i in range(10)]
    timestamps_with_reset = pre_reset_times + post_reset_times
    
    # Adjust timestamps with handle_timestamps_reset
    adjusted_timestamps = handle_timestamps_reset(timestamps=timestamps_with_reset, logger=dummy_logger)

    # We expect 86,400,000 to be added to all timestamps post-reset
    reset_threshold = 86_400_000  # 24:00:00 in ms
    expected_adjusted_timestamps = pre_reset_times + [t + reset_threshold for t in post_reset_times]

    assert adjusted_timestamps == expected_adjusted_timestamps, (
        f"Adjusted timestamps {adjusted_timestamps} did not match expected {expected_adjusted_timestamps}"
    )


def test_handle_timestamps_with_invalid_resets(dummy_logger):
    """ 
    Test handle_timestamps_reset raises the correct errors if our timestamps are weird
    """
    # Timestamps starting from 23:55:00
    times_high = [86_100_000 + i * 10 for i in range(5)]
    # Timestamps starting from close to 0
    times_low = [(i+1) * 10 for i in range(5)]

    # Normal reset (close to 24h then close to 0), but then times rise and drop again!!
    timestamps_multiple_resets = times_high + times_low + times_high + times_low

    # Invalid reset (small negative jump in timestamps, not close to a -24h jump)
    timestamps_invalid_reset = times_low + times_low

    # We expect a ValueError for multiple resets
    with pytest.raises(ValueError, match="Expected at most one timestamp reset"):
        handle_timestamps_reset(timestamps=timestamps_multiple_resets, logger=dummy_logger)

    # We expect an error for a backwards time jump too small to be a reset
    # (this should never happen and could indicate a problem with the recording)
    with pytest.raises(ValueError, match="Drop in timestamps too small to be a valid reset"):
        handle_timestamps_reset(timestamps=timestamps_invalid_reset, logger=dummy_logger)


def test_trim_sync_pulses_trim_front(dummy_logger):
    """
    Test that trim_sync_pulses correctly trims from the start of the longer list
    when the longer list has extra pulses in the front.
    """
    # Unaligned has an extra pulse at the start
    ground_truth = np.array([100.04, 200.05, 300.06, 400.07])
    unaligned = np.array([50.0, 100.0, 200.0, 300.0, 400.0])

    # trim_sync_pulses should trim the first element from unaligned
    ground_truth_trimmed, unaligned_trimmed = trim_sync_pulses(
        ground_truth_visits=ground_truth, 
        unaligned_visits=unaligned, 
        logger=dummy_logger
    )

    assert len(ground_truth_trimmed) == len(unaligned_trimmed), "The 2 lists should have equal length after trimming"
    assert np.array_equal(ground_truth, ground_truth_trimmed), "The shorter visits list should be unchanged"
    assert np.array_equal(unaligned[1:], unaligned_trimmed), "The longer visits list should be trimmed at the start"


def test_trim_sync_pulses_trim_end(dummy_logger):
    """
    Test that trim_sync_pulses correctly trims from the end of the longer list
    when the longer list has extra pulses at the end.
    """
    # Unaligned has an extra pulse at the end
    ground_truth = np.array([100.04, 200.05, 300.06, 400.07])
    unaligned = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    # trim_sync_pulses should trim the last element from unaligned
    ground_truth_trimmed, unaligned_trimmed = trim_sync_pulses(
        ground_truth_visits=ground_truth,
        unaligned_visits=unaligned,
        logger=dummy_logger
    )

    assert len(ground_truth_trimmed) == len(unaligned_trimmed), "The 2 lists should have equal length after trimming"
    assert np.array_equal(ground_truth_trimmed, ground_truth), "The shorter visits list should be unchanged"
    assert np.array_equal(unaligned_trimmed, unaligned[:-1]), "The longer visits list should be trimmed at the end"


# TODO add tests for align_via_interpolation - probably as a part of a test of a full conversion