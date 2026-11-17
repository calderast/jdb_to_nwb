from datetime import datetime
from pathlib import Path
from dateutil import tz
import numpy as np
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
    for column in ("bc_unitType", "ks_label", "phy_group", "original_cluster_id", "peak_electrode",
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


def _two_probe_sample_mapping(fs=30000.0, probe_offset_samples=45, probeB_extra_behavior=181):
    """
    The sample mapping a two-probe, two-recording conversion produces, shaped like the real one.

    ProbeB's recording started `probe_offset_samples` earlier, so the conversion aligned both to
    ProbeA's later start - a FRONT trim. ProbeB also recorded `probeB_extra_behavior` more samples
    in the behavior session than was written, an END trim. Those two trims are exactly what a
    single `sample_index / fs - bonsai_start_time` cannot account for, so both are present here.
    """
    behavior_samples = 1000
    opto_samples = 300
    streams = {}
    for probe_key, extra in (("probeA", 0), ("probeB", probeB_extra_behavior)):
        behavior_end = behavior_samples + extra
        streams[probe_key] = {
            "fs": fs, "n_chan": 4, "n_samples_total": behavior_end + opto_samples,
            "sessions": [
                {"session_id": "behavior", "sample_start": 0,
                 "sample_end": behavior_end, "n_samples": behavior_end},
                {"session_id": "optotagging", "sample_start": behavior_end,
                 "sample_end": behavior_end + opto_samples, "n_samples": opto_samples},
            ],
        }
    return {
        "session_boundaries": {"session_name": "TEST",
                               "concatenate_order": ["behavior", "optotagging"],
                               "streams": streams},
        "probe_start_times": {
            "ProbeA": {"behavior": 0.0, "optotagging": 500.0},
            "ProbeB": {"behavior": -probe_offset_samples / fs,
                       "optotagging": 500.0 - probe_offset_samples / fs},
        },
        # The conversion aligned both probes to ProbeA's later start, and the gap between the two
        # recordings put optotagging at 452s on the nwb clock rather than its hardware 500s
        "session_hardware_starts": {"behavior": 0.0, "optotagging": 500.0},
        "session_nwb_starts": {"behavior": 0.0, "optotagging": 452.0},
        "session_epochs": [
            {"session": "behavior", "start_time": 0.0, "stop_time": behavior_samples / fs},
            {"session": "optotagging", "start_time": 452.0,
             "stop_time": 452.0 + opto_samples / fs},
        ],
    }


def _spike_trains_at(samples, probe_label, sample_mapping, logger, fs=30000.0):
    """Run one spike per unit through the alignment, so each returned train is one spike."""
    from spikeinterface.core import NumpySorting
    from jdb_to_nwb.convert_spikes import aligned_spike_trains_by_unit

    sorting = NumpySorting.from_samples_and_labels(
        samples_list=[np.asarray(samples, dtype=np.int64)],
        labels_list=[np.arange(len(samples))],
        sampling_frequency=fs,
    )
    return aligned_spike_trains_by_unit(
        sorting=sorting, sampling_frequency=fs,
        metadata={"ephys_sample_mapping": sample_mapping}, logger=logger,
        probe_label=probe_label,
    )


def test_spike_samples_map_onto_the_nwb_clock_on_both_probes_and_sessions(dummy_logger):
    """
    The same physical moment, recorded by two probes with different trims, must land at the same
    nwb time in BOTH recordings.

    sample_index / fs - bonsai_start_time cannot do this: it misses the gap between recordings, the
    front trim that aligned the probes, and the end trim that shortens one probe's behavior
    segment. This checks a known moment through the real mapping on each probe in each session,
    the same way test_add_laser_events_maps_pulses_onto_the_nwb_clock checks the laser pulses.
    """
    fs = 30000.0
    offset, extra = 45, 181
    mapping = _two_probe_sample_mapping(fs=fs, probe_offset_samples=offset,
                                        probeB_extra_behavior=extra)

    # Two moments on the hardware clock: 10 samples into behavior, 30 samples into optotagging
    # (both measured from ProbeA's start, which is where the conversion aligned everything)
    behavior_a, opto_a = 10, 1000 + 30
    # The same two moments on ProbeB's own axis: it started `offset` samples earlier, so each
    # moment sits that much further along, and its optotagging begins `extra` samples later
    behavior_b, opto_b = 10 + offset, (1000 + extra) + 30 + offset

    trains_a = _spike_trains_at([behavior_a, opto_a], "ProbeA", mapping, dummy_logger, fs)
    trains_b = _spike_trains_at([behavior_b, opto_b], "ProbeB", mapping, dummy_logger, fs)

    # Behavior: 10 samples after the nwb clock's zero
    assert trains_a[0][0] == pytest.approx(10 / fs)
    assert trains_b[0][0] == pytest.approx(10 / fs)
    # Optotagging: 30 samples into the epoch that starts at 452s, NOT at its hardware 500s
    assert trains_a[1][0] == pytest.approx(452.0 + 30 / fs)
    assert trains_b[1][0] == pytest.approx(452.0 + 30 / fs)

    # sample / fs is the time on the probe's own axis, which is not a time on the nwb clock
    assert trains_a[1][0] != pytest.approx(opto_a / fs)
    assert trains_b[1][0] != pytest.approx(opto_b / fs)


def test_spikes_far_outside_their_session_epoch_raise(dummy_logger):
    """
    Make sure the guard actually catches a mapping error.

    It fires when the session start times and the epochs disagree about where a recording sits on
    the nwb clock. Here the epoch says optotagging starts at 600s while the start times put its
    samples at 452s.
    """
    fs = 30000.0
    mapping = _two_probe_sample_mapping(fs=fs)
    opto_epoch = next(e for e in mapping["session_epochs"] if e["session"] == "optotagging")
    opto_epoch["start_time"], opto_epoch["stop_time"] = 600.0, 600.0 + 300 / fs

    with pytest.raises(ValueError, match="trimmed samples could explain"):
        _spike_trains_at([1000 + 30], "ProbeA", mapping, dummy_logger, fs)


def test_a_single_recording_is_mapped_exactly_as_the_bonsai_shift_does(dummy_logger):
    """
    One probe, one recording must come out byte-identical to the bonsai shift, which is what
    every single-probe file is converted with.

    For a single recording the front trim IS the bonsai trim, so mapping through the session's
    recorded starts and subtracting a bonsai offset are the same arithmetic.
    """
    fs = 30000.0
    bonsai_trim_samples = 285530   # the real IM-1994 trim
    total_samples = bonsai_trim_samples + 1000
    mapping = {
        "session_boundaries": {"streams": {"probeA": {
            "fs": fs, "n_chan": 4, "n_samples_total": total_samples,
            "sessions": [{"session_id": "behavior", "sample_start": 0,
                          "sample_end": total_samples, "n_samples": total_samples}]}}},
        "probe_start_times": {"ProbeA": {"behavior": 0.0}},
        # The conversion trimmed to bonsai start, so the first converted sample is that far in
        "session_hardware_starts": {"behavior": bonsai_trim_samples / fs},
        "session_nwb_starts": {"behavior": 0.0},
        "session_epochs": [{"session": "behavior", "start_time": 0.0, "stop_time": 1000 / fs}],
    }
    spike_samples = [bonsai_trim_samples + 1, bonsai_trim_samples + 500]

    mapped = _spike_trains_at(spike_samples, "ProbeA", mapping, dummy_logger, fs)

    bonsai_start_time = bonsai_trim_samples / fs
    for train, sample in zip(mapped, spike_samples):
        assert train[0] == pytest.approx(sample / fs - bonsai_start_time)
