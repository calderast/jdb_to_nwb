from datetime import datetime
from dateutil import tz
from pynwb import NWBFile

from jdb_to_nwb.convert_spikes import add_spikes


def test_add_spikes():
    """
    Test the add_spikes function.

    File `tests/test_data/processed_ephys/firings.mda` must be created first by running
    `python tests/test_data/create_spike_test_data.py`.
    """
    metadata = {}
    metadata["ephys"]["mountain_sort_output_file_path"] = "tests/test_data/processed_ephys/firings.mda"
    metadata["ephys"]["sampling_frequency"] = 30_000

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_spikes(nwbfile, metadata)

    assert len(nwbfile.units) == 63
    assert nwbfile.units["spike_times"] is not None
    assert nwbfile.units["unit_name"] is not None
    assert len(nwbfile.units["spike_times"][0]) == 22
    assert nwbfile.units["unit_name"][0] == "15"
    assert len(nwbfile.units.spike_times.data) == 462  # Check total number of spikes
