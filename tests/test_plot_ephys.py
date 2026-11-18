"""Tests for the ephys figures (plotting.plot_ephys)."""

import pandas as pd

from jdb_to_nwb.plotting.plot_ephys import _find_contiguous_channel_groups_npx


def _channels(probe, shank, y_positions):
    """One row per recorded channel, shaped like the electrodes table after plot_ephys renames it."""
    return pd.DataFrame({"probe": probe, "shank": shank, "y_um": y_positions,
                         "channel_num": range(len(y_positions)),
                         "data_col": range(len(y_positions))})


def test_npx_channel_groups_keep_the_probes_apart():
    """
    Bilateral probes are the same model, so their shanks have the same numbers AND the same site
    positions. Grouping by shank alone would stack both probes' traces in one figure at identical
    y, with nothing in the figure to say which probe a trace came from.
    """
    both_probes = pd.concat([_channels("ProbeA", 0, [0.0, 15.0, 30.0]),
                             _channels("ProbeB", 0, [0.0, 15.0, 30.0])], ignore_index=True)

    groups = _find_contiguous_channel_groups_npx(both_probes)

    assert [(probe, shank, run, len(run_df)) for probe, shank, run, run_df in groups] == [
        ("ProbeA", 0, 0, 3),
        ("ProbeB", 0, 0, 3),
    ]


def test_npx_channel_groups_split_a_shank_on_a_real_gap():
    """
    A gap bigger than two electrode rows is a true break in coverage (e.g. a different bank), so
    it starts a new run. Within a run the channels are contiguous, which is what makes stacking
    them in one figure meaningful.
    """
    one_shank = _channels("ProbeA", 0, [0.0, 15.0, 30.0, 3000.0, 3015.0])

    groups = _find_contiguous_channel_groups_npx(one_shank)

    assert [(run, list(run_df["y_um"])) for _probe, _shank, run, run_df in groups] == [
        (0, [0.0, 15.0, 30.0]),
        (1, [3000.0, 3015.0]),
    ]
