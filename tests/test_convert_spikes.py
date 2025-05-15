from datetime import datetime
from dateutil import tz
from pynwb import NWBFile

from jdb_to_nwb.convert_spikes import add_spikes


def test_add_spikes(dummy_logger):
    """
    Test the add_spikes function.

    File `tests/test_data/processed_ephys/firings.mda` must be created first by running
    `python tests/test_data/create_spike_test_data.py`.
    """
    metadata = {}
    metadata["ephys"] = {}
    metadata["ephys"]["mountain_sort_output_file_path"] = "tests/test_data/processed_ephys/firings.mda"
    metadata["ephys"]["sampling_frequency"] = 30_000

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    assert len(nwbfile.units) == 17
    assert nwbfile.units["spike_times"] is not None
    assert nwbfile.units["unit_name"] is not None
    assert len(nwbfile.units["spike_times"][0]) == 3
    assert nwbfile.units["unit_name"][0] == "15"
    assert len(nwbfile.units.spike_times.data) == 30  # Check total number of spikes


def test_add_spikes_with_incomplete_metadata(dummy_logger, capsys):
    """
    Test that the add_spikes function responds appropriately to missing or incomplete metadata.
    
    If no 'ephys' key is in the metadata dictionary, it should move on without raising any errors.
    
    If there is a 'ephys' key in the metadata dict but the required spike subfields
    are not present, print that we are skipping spike conversion and move on with no errors.
    (It is ok for a user to specify raw ephys data but not to have done spike sorting).
    """

    # Create a test metadata dictionary with no ephys key
    metadata = {}

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # If we call the add_spikes function with no 'ephys' key in metadata,
    # it should skip ephys conversion and return with no errors
    # (No output because we have already printed that we are skipping ephys in add_raw_ephys)
    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Create a test metadata dictionary with an ephys field but no spike data
    metadata["ephys"] = {}

    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)
    captured = capsys.readouterr() # capture stdout

    # Check that the correct message was printed to stdout
    assert "No spike sorting metadata found for this session. Skipping spike conversion." in captured.out
