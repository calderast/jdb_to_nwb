"""
Tests for converting PREPROCESSED ephys traces (the optonpx pipeline output) into the nwb.

Tests for the raw Open Ephys conversion live in test_convert_raw_ephys.py.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from dateutil import tz
from pynwb import NWBFile

from jdb_to_nwb.convert_raw_ephys import add_raw_ephys
from jdb_to_nwb.convert_processed_ephys import (
    add_laser_events,
    open_processed_probe_sessions,
    read_session_boundaries,
    verify_processed_matches_raw,
)


def _make_processed_fixture(tmp_path, probe_samples):
    """
    Build a preprocessed session directory shaped like the optonpx output: one lower-camel-case
    folder per probe holding a concatenated int16 continuous.dat, plus session_boundaries.json.

    probe_samples maps "probeA"/"probeB" -> (behavior_samples, optotagging_samples), mirroring the
    real recordings where the probes differ by a few samples in each session.
    """
    import json

    num_channels = 4
    streams = {}
    for probe_key, (n_behavior, n_opto) in probe_samples.items():
        probe_dir = tmp_path / probe_key
        probe_dir.mkdir()
        total = n_behavior + n_opto
        # Each sample's value encodes its index, so a slice can be checked against where it came from
        data = np.tile(np.arange(total, dtype=np.int16).reshape(-1, 1), (1, num_channels))
        data.tofile(probe_dir / "continuous.dat")
        streams[probe_key] = {
            "fs": 30000.0,
            "n_chan": num_channels,
            "n_samples_total": total,
            "sessions": [
                {"session_id": "behavior", "sample_start": 0,
                 "sample_end": n_behavior, "n_samples": n_behavior},
                {"session_id": "optotagging", "sample_start": n_behavior,
                 "sample_end": total, "n_samples": n_opto},
            ],
        }
    (tmp_path / "session_boundaries.json").write_text(json.dumps(
        {"session_name": "TEST", "concatenate_order": ["behavior", "optotagging"], "streams": streams}
    ))
    return tmp_path

def _laser_fixture(tmp_path, fs=30000.0, behavior_samples=1000):
    """
    A preprocessed session directory with a two-probe laser_events.csv, shaped like the real one.

    The probes start 45 samples apart (the measured IM-1979 2026-09-11 offset), so the SAME
    physical pulse sits at different sample indices on each probe's axis - which is exactly how
    that offset was originally measured.
    """
    import json

    probe_offset_samples = 45
    streams = {}
    for probe_key, extra in (("probeA", 0), ("probeB", probe_offset_samples)):
        total = behavior_samples + 300
        streams[probe_key] = {
            "fs": fs, "n_chan": 4, "n_samples_total": total,
            "sessions": [
                {"session_id": "behavior", "sample_start": 0,
                 "sample_end": behavior_samples, "n_samples": behavior_samples},
                {"session_id": "optotagging", "sample_start": behavior_samples,
                 "sample_end": total, "n_samples": 300},
            ],
        }
    (tmp_path / "session_boundaries.json").write_text(json.dumps({
        "session_name": "TEST", "concatenate_order": ["behavior", "optotagging"], "streams": streams}))

    # Two pulses in the optotagging session. probeB sees them 45 samples later on its own axis,
    # because probeB's recording started 45 samples earlier in hardware time.
    rows = ["probe,session,color,on_sample_concat,off_sample_concat"]
    for pulse_sample in (behavior_samples + 30, behavior_samples + 120):
        rows.append(f"probeA,optotagging,blue_laser,{pulse_sample},{pulse_sample + 10}")
        rows.append(f"probeB,optotagging,blue_laser,{pulse_sample + probe_offset_samples},"
                    f"{pulse_sample + probe_offset_samples + 10}")
    (tmp_path / "laser_events.csv").write_text("\n".join(rows) + "\n")
    return tmp_path, probe_offset_samples

def test_open_processed_probe_sessions_splits_at_the_boundary(dummy_logger, tmp_path):
    """
    Each session must come back as its own recording, cut at the right samples and carrying its own
    start time. The boundary sits at a different sample index for each probe, which is exactly why
    the sessions are opened separately rather than sliced from one recording.
    """
    processed = _make_processed_fixture(tmp_path, {"probeA": (1000, 300), "probeB": (995, 299)})
    boundaries = read_session_boundaries(processed, dummy_logger)

    sessions = open_processed_probe_sessions(
        processed_ephys_path=processed,
        probe_label="ProbeA",
        probe_boundaries=boundaries["streams"]["probeA"],
        conversion_factor_uv=0.195,
        session_start_times={"behavior": 0.0, "optotagging": 500.0},
        logger=dummy_logger,
    )

    assert set(sessions) == {"behavior", "optotagging"}
    assert sessions["behavior"].get_num_samples() == 1000
    assert sessions["optotagging"].get_num_samples() == 300
    # Each session carries its own start time, so they can be aligned per session
    assert sessions["behavior"].sample_index_to_time(0) == 0.0
    assert sessions["optotagging"].sample_index_to_time(0) == 500.0
    # Sample values encode their index in the concatenated file, so this proves the right bytes
    assert sessions["behavior"].get_traces(start_frame=0, end_frame=1)[0][0] == 0
    assert sessions["optotagging"].get_traces(start_frame=0, end_frame=1)[0][0] == 1000
    # Conversion factor comes from the raw oebin, since the binary file carries none
    np.testing.assert_allclose(sessions["behavior"].get_channel_gains()[0], 0.195)

def test_open_processed_probe_sessions_rejects_wrong_file_length(dummy_logger, tmp_path):
    """
    If the .dat and session_boundaries.json disagree, every sample index taken from the boundaries
    file points at the wrong data, so this must fail loudly rather than convert silently.
    """
    processed = _make_processed_fixture(tmp_path, {"probeA": (1000, 300)})
    boundaries = read_session_boundaries(processed, dummy_logger)
    boundaries["streams"]["probeA"]["n_samples_total"] = 9999  # lie about the length

    with pytest.raises(ValueError, match="session_boundaries.json says"):
        open_processed_probe_sessions(
            processed_ephys_path=processed,
            probe_label="ProbeA",
            probe_boundaries=boundaries["streams"]["probeA"],
            conversion_factor_uv=0.195,
            session_start_times={},
            logger=dummy_logger,
        )

def test_add_raw_ephys_writes_preprocessed_traces(dummy_logger, tmp_path):
    """
    With 'processed_ephys_path' set, the ElectricalSeries holds the PREPROCESSED traces while the
    timeline, channel map and conversion factor still come from the raw recordings. This is the
    shape needed to re-sort in Spyglass on the same data Kilosort saw.

    The preprocessed file here is the raw fixture's own traces copied into the flat concatenated
    layout the preprocessing pipeline writes, so the two can be compared sample for sample.
    """
    import json
    import shutil

    fixture = Path("tests/test_data/raw_ephys/2022-07-25_15-30-00")
    raw_folder = tmp_path / "raw" / "2022-07-25_15-30-00"
    raw_folder.parent.mkdir()
    shutil.copytree(fixture, raw_folder)

    # Build the preprocessed layout: one folder per probe holding the concatenated continuous.dat.
    # The preprocessed file holds only the NEURAL channels - the real Neuropixels one is 384
    # channels with no ADC - so the ADC channels are dropped here, as the pipeline drops them.
    # (If they were left in, the channel count would not match the electrode table and the
    # conversion fails loudly, which is the behavior we want.)
    raw_dat = raw_folder / "experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat"
    num_channels, num_samples = 256, 3_000
    raw_traces = np.memmap(raw_dat, dtype="int16", mode="r").reshape(num_samples, 264)
    processed = tmp_path / "processed"
    probe_dir = processed / "rhythm_fpga-100.0"
    probe_dir.mkdir(parents=True)
    np.ascontiguousarray(raw_traces[:, :num_channels]).tofile(probe_dir / "continuous.dat")
    (processed / "session_boundaries.json").write_text(json.dumps({
        "session_name": "TEST",
        "concatenate_order": ["behavior"],
        "streams": {"rhythm_fpga-100.0": {
            "fs": 30000.0, "n_chan": num_channels, "n_samples_total": num_samples,
            "sessions": [{"session_id": "behavior", "sample_start": 0,
                          "sample_end": num_samples, "n_samples": num_samples}],
        }},
    }))
    (processed / "preprocessing.json").write_text(json.dumps({"engine": "test", "probes": {}}))

    metadata = {"ephys": {
        "openephys_folder_paths": [{"path": str(raw_folder), "session": "behavior"}],
        "processed_ephys_path": str(processed),
        "impedance_file_path": "tests/test_data/processed_ephys/impedance.csv",
        "probes": [
            {"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
             "electrodes_location": "Hippocampus CA1",
             "targeted_x": 2.2, "targeted_y": 4.5, "targeted_z": -2.0},
        ],
    }}
    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))

    add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    eseries = nwbfile.acquisition["ElectricalSeries"]
    # One channel per electrode, matching the electrode table built from the raw settings.xml
    assert eseries.data.maxshape[1] == num_channels == len(nwbfile.electrodes)

    # The series must say the traces are preprocessed rather than claiming to be raw
    assert "PREPROCESSED" in eseries.description
    assert "interpolation" in eseries.description

    # How the traces were made travels with the file
    associated = nwbfile.processing["associated_files"].data_interfaces
    assert "preprocessing" in associated
    assert "session_boundaries" in associated

    # The timeline still comes from raw: one epoch for the recording
    assert len(nwbfile.epochs.to_dataframe()) == 1

def test_add_laser_events_maps_pulses_onto_the_nwb_clock(dummy_logger, tmp_path):
    """
    A pulse's sample index on a probe's own axis must land at the right time on the nwb clock, and
    the same pulse seen by both probes must land at the same time - which independently checks the
    probe alignment against a signal both probes recorded.
    """
    import json

    fs = 30000.0
    processed, probe_offset_samples = _laser_fixture(tmp_path, fs=fs, behavior_samples=1000)
    boundaries = json.loads((processed / "session_boundaries.json").read_text())

    # probeB's recording started 45 samples EARLIER, so the same moment sits later on its axis
    probe_start_times = {
        "ProbeA": {"behavior": 0.0, "optotagging": 500.0},
        "ProbeB": {"behavior": -probe_offset_samples / fs, "optotagging": 500.0 - probe_offset_samples / fs},
    }
    # The conversion aligned both probes to the later start (ProbeA) and placed optotagging at 452s
    session_hardware_starts = {"behavior": 0.0, "optotagging": 500.0}
    session_nwb_starts = {"behavior": 0.0, "optotagging": 452.0}

    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))

    add_laser_events(
        nwbfile=nwbfile,
        processed_ephys_path=processed,
        session_boundaries=boundaries,
        probe_start_times=probe_start_times,
        session_hardware_starts=session_hardware_starts,
        session_nwb_starts=session_nwb_starts,
        logger=dummy_logger,
    )

    laser_events = nwbfile.intervals["laser_events"].to_dataframe()
    # Two physical pulses, not four: one row per pulse, no per-probe duplicates
    assert len(laser_events) == 2
    assert list(laser_events["color"]) == ["blue_laser", "blue_laser"]
    assert list(laser_events["session"]) == ["optotagging", "optotagging"]

    # The first pulse is 30 samples into the optotagging session, which starts at 452s on the nwb clock
    assert np.isclose(laser_events["start_time"][0], 452.0 + 30 / fs)
    assert np.isclose(laser_events["stop_time"][0], 452.0 + 40 / fs)
    assert np.isclose(laser_events["start_time"][1], 452.0 + 120 / fs)

def test_add_laser_events_skips_sessions_not_converted(dummy_logger, tmp_path):
    """Pulses from a recording this conversion did not write must be left out, not misplaced."""
    import json

    processed, _offset = _laser_fixture(tmp_path)
    boundaries = json.loads((processed / "session_boundaries.json").read_text())
    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))

    # Only the behavior session was converted, and every pulse is in optotagging
    add_laser_events(
        nwbfile=nwbfile,
        processed_ephys_path=processed,
        session_boundaries=boundaries,
        probe_start_times={"ProbeA": {"behavior": 0.0}, "ProbeB": {"behavior": 0.0}},
        session_hardware_starts={"behavior": 0.0},
        session_nwb_starts={"behavior": 0.0},
        logger=dummy_logger,
    )

    assert "laser_events" not in (nwbfile.intervals or {})

def test_add_laser_events_detects_a_probe_alignment_error(dummy_logger, tmp_path, caplog):
    """
    The cross-probe agreement check has to actually catch a misalignment, or it is worthless. Here
    probeB's start time is wrong by 45 samples, so the same pulse maps to two different nwb times
    and the check must say so.
    """
    import json
    import logging

    fs = 30000.0
    processed, probe_offset_samples = _laser_fixture(tmp_path, fs=fs, behavior_samples=1000)
    boundaries = json.loads((processed / "session_boundaries.json").read_text())

    # The correct value is -45/fs (as in the passing test); 0.0 is wrong by exactly 45 samples
    probe_start_times = {
        "ProbeA": {"optotagging": 500.0},
        "ProbeB": {"optotagging": 500.0},   # WRONG: ignores probeB's earlier start
    }
    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))

    with caplog.at_level(logging.WARNING):
        add_laser_events(
            nwbfile=nwbfile,
            processed_ephys_path=processed,
            session_boundaries=boundaries,
            probe_start_times=probe_start_times,
            session_hardware_starts={"optotagging": 500.0},
            session_nwb_starts={"optotagging": 452.0},
            logger=dummy_logger,
        )

    disagreement_warnings = [record for record in caplog.records
                             if "Laser pulses agree" in record.message and record.levelno >= logging.WARNING]
    assert disagreement_warnings, "a 45-sample misalignment must be reported"
    assert "45.0 samples" in disagreement_warnings[0].message

def test_verify_processed_matches_raw_accepts_a_phase_shift(dummy_logger, caplog):
    """
    Preprocessing legitimately changes the samples: the NP2.0 phase shift moves each channel by a
    FRACTION of a sample. Real ephys is band-limited, so a fractional shift barely changes a
    channel's correlation with its raw counterpart - and the check must not flag it.

    (A full-sample shift of white noise would decorrelate completely, which is why this uses a
    band-limited signal and a sub-sample shift, as the real phase shift does.)
    """
    import logging
    import spikeinterface.core as si_core

    fs, num_samples, num_channels = 30000.0, 10_000, 8
    times = np.arange(num_samples) / fs
    # Band-limited signal, a different frequency per channel
    raw_traces = np.stack([
        np.sin(2 * np.pi * (300 + 50 * channel) * times) * 1000
        for channel in range(num_channels)
    ], axis=1)
    # Each channel shifted by a fraction of a sample, as the phase shift does
    shifted_traces = np.stack([
        np.interp(np.arange(num_samples) + 0.3, np.arange(num_samples), raw_traces[:, channel])
        for channel in range(num_channels)
    ], axis=1)

    raw = si_core.NumpyRecording([raw_traces.astype("float32")], sampling_frequency=fs)
    processed = si_core.NumpyRecording([shifted_traces.astype("float32")], sampling_frequency=fs)

    with caplog.at_level(logging.WARNING):
        verify_processed_matches_raw(processed_recording=processed, raw_recording=raw,
                                     logger=dummy_logger, num_samples=num_samples)

    assert not [record for record in caplog.records if "different order" in record.message]

def test_verify_processed_matches_raw_catches_reordered_channels(dummy_logger, caplog):
    """
    A reordering is the failure that nothing else would notice: every electrode would be mapped to
    the wrong trace and the conversion would otherwise succeed. The check must catch it.
    """
    import logging
    import spikeinterface.core as si_core

    raw = si_core.generate_recording(num_channels=8, durations=[1.0], sampling_frequency=30000.0, seed=4)
    # Same data, channels in a different order - exactly what must not go unnoticed
    reordered = raw.select_channels(channel_ids=list(raw.channel_ids)[::-1])
    reordered = reordered.rename_channels(list(raw.channel_ids))

    with caplog.at_level(logging.WARNING):
        verify_processed_matches_raw(processed_recording=reordered, raw_recording=raw,
                                     logger=dummy_logger, num_samples=10_000)

    warnings = [record for record in caplog.records if "different order" in record.message]
    assert warnings, "a channel reordering must be reported"
