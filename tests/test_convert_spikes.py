import logging
from datetime import datetime
from pathlib import Path
from dateutil import tz
import numpy as np
import pytest
from pynwb import NWBFile

from jdb_to_nwb.convert_spikes import add_spikes, check_unit_columns_match, create_units_table

MOUNTAINSORT_MDA_PATH = Path("tests/test_data/processed_ephys/firings.mda")


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
    # Spike times come from sample indices, so one sample is as fine as they get
    assert units.resolution == 1 / 30_000


def _kilosort_bombcell_fixture(tmp_path, fs=30_000.0, num_channels=4):
    """
    A SortingAnalyzer and nwb shaped like what the Kilosort4 + BombCell path really gets handed.

    The real analyzer is a 5.6 GB directory of a whole session's sorting, which cannot live in the
    test suite, so this builds the same STRUCTURE from generated data: the per-unit properties
    BombCell leaves on the sorting, the extensions the conversion reads, Phy's cluster_group.tsv
    sitting next to the analyzer, and the electrodes table the raw ephys conversion writes first
    (which is what a unit's peak electrode is a row in).

    Returns (metadata, nwbfile).
    """
    import spikeinterface.core as sc

    recording, sorting = sc.generate_ground_truth_recording(
        durations=[1.0], sampling_frequency=fs, num_channels=num_channels, num_units=3, seed=0)
    # Kilosort numbers its clusters, and so does the real analyzer; SpikeInterface's generated
    # sorting names them "0".."2" instead, which is not a cluster id our pipeline ever produces
    sorting = sorting.rename_units(np.arange(len(sorting.unit_ids)))
    # The properties our pipeline leaves on the sorting: BombCell's class, Kilosort's auto label,
    # the source Phy cluster id, and one of BombCell's numeric cluster_info metrics
    sorting.set_property("bc_unitType", ["GOOD", "MUA", "NOISE"])
    sorting.set_property("KSLabel", ["good", "mua", "mua"])
    sorting.set_property("original_cluster_id", [10, 11, 12])
    sorting.set_property("signalToNoiseRatio", [5.0, 2.0, 1.0])

    analyzer_path = tmp_path / "analyzer.zarr"
    analyzer = sc.create_sorting_analyzer(sorting, recording, format="zarr", folder=analyzer_path)
    analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels"])
    analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])

    # Phy's MANUAL curation, which lives beside the analyzer and is matched by original_cluster_id
    (tmp_path / "cluster_group.tsv").write_text("cluster_id\tgroup\n10\tgood\n11\tmua\n12\tnoise\n")

    metadata = {
        "ephys": {"probes": [{"stream": "ProbeA", "sorting_analyzer_path": analyzer_path}]},
        # What add_raw_ephys reports back for each probe it converted
        "ephys_probe_results": [{"label": "ProbeA", "num_samples": recording.get_num_samples()}],
    }

    # The electrodes table add_raw_ephys writes: one row per recording channel, on two shanks,
    # carrying the probe each row belongs to
    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))
    device = nwbfile.create_device(name="probe")
    shanks = [nwbfile.create_electrode_group(name=f"ProbeA_{shank}", description="shank",
                                             location="NAc", device=device) for shank in range(2)]
    nwbfile.add_electrode_column(name="intan_channel_number", description="recording channel")
    nwbfile.add_electrode_column(name="probe", description="probe this electrode is on")
    for channel in range(num_channels):
        nwbfile.add_electrode(group=shanks[channel // (num_channels // 2)], location="NAc",
                              intan_channel_number=channel, probe="ProbeA")
    return metadata, nwbfile


def test_add_kilosort_bombcell_spikes(dummy_logger, tmp_path):
    """
    Every piece of the Kilosort4 + BombCell path on one sorting: ALL units written (not just
    'good' ones), BombCell and Phy labels, quality metrics, a peak-channel waveform_mean, and the
    unit's peak electrode as both a row number and an NWB electrodes reference.
    """
    metadata, nwbfile = _kilosort_bombcell_fixture(tmp_path)

    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    units = nwbfile.units
    # Every unit, including the one BombCell called NOISE, so they can be filtered downstream
    assert len(units) == 3
    assert set(units["bc_unitType"][:]) == {"GOOD", "MUA", "NOISE"}
    assert list(units["ks_label"][:]) == ["good", "mua", "mua"]
    # Phy's manual curation, matched to units by original_cluster_id
    assert list(units["phy_group"][:]) == ["good", "mua", "noise"]
    assert list(units["probe"][:]) == ["ProbeA"] * 3
    for column in ("snr", "firing_rate",            # SpikeInterface quality_metrics
                   "signalToNoiseRatio",            # BombCell/cluster_info metric
                   "sorter_unit_id", "peak_electrode", "waveform_mean"):
        assert column in units.colnames

    # Spike times are on the nwb clock, and the table says they are good to one sample
    assert len(units["spike_times"][0]) > 0
    assert units.resolution == 1 / 30_000

    # waveform_mean is the mean on the unit's peak channel: 1D over samples
    assert units["waveform_mean"][0].ndim == 1

    # The peak electrode, as a row number and as the reference NWB defines. Both must point at
    # the same electrode, and the group must be the shank that electrode is on.
    units_df = units.to_dataframe()
    peak_row = units_df["peak_electrode"][0]
    assert list(units_df["electrodes"][0].index) == [peak_row]
    assert units_df["electrode_group"][0] is nwbfile.electrodes["group"][peak_row]


def test_add_kilosort_bombcell_spikes_rejects_a_sorting_of_another_recording(dummy_logger, tmp_path):
    """
    A sorting done on a different (or cropped) recording would have its spike samples indexing
    something else, so aligning them to this session's clock would be wrong with nothing to show
    it. The recorded sample count is what catches it.
    """
    metadata, nwbfile = _kilosort_bombcell_fixture(tmp_path)
    metadata["ephys_probe_results"][0]["num_samples"] += 1000

    with pytest.raises(ValueError, match="does not match the full raw ephys recording"):
        add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)


def test_a_probe_sorted_at_another_rate_is_reported(dummy_logger, caplog):
    """
    The Units table carries one resolution, written by the first probe converted, so a second
    probe sorted at a different rate has its spike times described by the wrong number. The table
    cannot hold two, so the only honest thing is to say so.
    """
    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))
    create_units_table(nwbfile, 30_000, "Units spike sorted with Kilosort4", dummy_logger)

    with caplog.at_level(logging.WARNING):
        create_units_table(nwbfile, 25_000, "Units spike sorted with Kilosort4", dummy_logger)

    assert nwbfile.units.resolution == 1 / 30_000   # the first probe's, unchanged
    assert [record for record in caplog.records
            if "25000.0 Hz" in record.message and record.levelno >= logging.WARNING]


def test_units_from_a_second_probe_must_write_the_same_columns(dummy_logger, tmp_path):
    """
    Every probe's units go in the one Units table, so a second probe that cannot fill a column the
    first one wrote - an analyzer computed without the 'templates' extension, say - has to be
    caught here. add_unit would otherwise fail inside hdmf with "'NoneType' object is not
    iterable", naming neither the probe nor the column, part way through a conversion.
    """
    metadata, nwbfile = _kilosort_bombcell_fixture(tmp_path)
    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # What the second probe would offer if its analyzer had no templates computed
    from_templates = {"waveform_mean", "peak_electrode", "electrodes", "electrode_group"}
    without_templates = [column for column in nwbfile.units.colnames
                         if column != "spike_times" and column not in from_templates]

    with pytest.raises(ValueError, match="peak_electrode"):
        check_unit_columns_match(nwbfile, "ProbeB", without_templates, dummy_logger)

    # The same columns are fine, which is what every real two-probe session has
    check_unit_columns_match(nwbfile, "ProbeB",
                             [c for c in nwbfile.units.colnames if c != "spike_times"], dummy_logger)


def test_spikes_from_several_raw_recordings_are_refused(dummy_logger, tmp_path):
    """
    Spike times taken straight from the raw recordings cannot span more than one of them.

    A spike's sample index counts samples that were recorded; the dead time between two recordings
    was not recorded, so shifting by the bonsai start alone places every spike after the first
    recording early by the length of the gap. Nothing about the result looks wrong - the sorting
    even passes the length check, because a probe's length is summed over the recordings it was
    sorted across - so this has to be refused rather than written.
    """
    metadata, nwbfile = _kilosort_bombcell_fixture(tmp_path)
    # What add_raw_ephys reports after joining two recordings on the RAW path: no sample mapping,
    # which only the preprocessed path produces
    metadata["ephys_num_recordings"] = 2
    metadata["ephys_sample_mapping"] = None

    with pytest.raises(ValueError, match="joins 2 Open Ephys recordings"):
        add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # One recording is the case this path is for, and still converts
    metadata["ephys_num_recordings"] = 1
    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)
    assert len(nwbfile.units) == 3


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
