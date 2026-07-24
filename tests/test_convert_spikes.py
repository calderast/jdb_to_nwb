from datetime import datetime
from pathlib import Path
from dateutil import tz
import pytest
from pynwb import NWBFile

from jdb_to_nwb.convert_spikes import add_spikes

MOUNTAINSORT_MDA_PATH = Path("tests/test_data/processed_ephys/firings.mda")
KILOSORT_ANALYZER_PATH = Path("tests/test_data/downloaded/IM-1971/20260619/kilosort4_1_bombcell/analyzer.zarr")


def test_add_spikes_mountainsort(dummy_logger):
    """
    Test adding MountainSort output (.mda) to the NWB.

    File `tests/test_data/processed_ephys/firings.mda` must be created first by running
    `python tests/test_data/create_spike_test_data.py`.
    """
    metadata = {
        "ephys": {
            "mountain_sort_output_file_path": MOUNTAINSORT_MDA_PATH,
            "sampling_frequency": 30_000,
        }
    }
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    units = nwbfile.units
    assert len(units) == 17
    assert units["spike_times"] is not None
    assert units["unit_name"] is not None
    assert len(units["spike_times"][0]) == 3
    assert units["unit_name"][0] == "15"
    assert len(units.spike_times.data) == 30  # Check total number of spikes


@pytest.mark.slow
def test_add_kilosort_bombcell_spikes(dummy_logger):
    """
    Test adding Kilosort4 + BombCell spike sorting output (a SpikeInterface SortingAnalyzer) to the NWB.

    All units should be written (not just 'good' ones), with curation labels and quality metrics as
    columns and a peak-channel waveform_mean per unit.
    """
    metadata = {
        "ephys": {
            "sorting_analyzer_path": KILOSORT_ANALYZER_PATH
        }
    }
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    units = nwbfile.units
    # All 888 units in the analyzer should be present
    assert len(units) == 888

    # Curation labels, SpikeInterface quality metrics, and BombCell/cluster_info metrics should all be columns
    for column in ("bc_unitType", "ks_label", "phy_group", "original_cluster_id", "peak_channel_id",
                   "waveform_mean",
                   "snr", "firing_rate", "isi_violations_ratio",  # SpikeInterface quality_metrics
                   "nPeaks", "signalToNoiseRatio", "waveformDuration_peakTrough"):  # BombCell/cluster_info metrics
        assert column in units.colnames

    # Each unit should have spike times and a 1D waveform_mean (peak channel over samples)
    assert len(units["spike_times"][0]) > 0
    assert units["waveform_mean"][0].ndim == 1

    # bc_unitType should only contain the expected BombCell classifications
    assert set(units["bc_unitType"][:]).issubset({"GOOD", "MUA", "NON-SOMA", "NOISE"})


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
