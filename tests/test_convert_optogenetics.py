"""Tests for the optotagging laser pulses (convert_optogenetics)."""

import json
import logging
from datetime import datetime

import numpy as np
import pytest
from dateutil import tz
from pynwb import NWBFile

from jdb_to_nwb.convert_optogenetics import add_laser_events

# How far apart the two probes in the fixture started, in samples
PROBE_OFFSET_SAMPLES = 45


def _laser_fixture(tmp_path, fs=30000.0, behavior_samples=1000):
    """
    A preprocessed session directory with a two-probe laser_events.csv, shaped like the real one.

    The probes start PROBE_OFFSET_SAMPLES apart (the measured IM-1979 2026-09-11 offset), so the
    SAME physical pulse sits at different sample indices on each probe's axis - which is exactly
    how that offset was originally measured.
    """
    streams = {}
    for probe_key in ("probeA", "probeB"):
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

    # Two pulses in the optotagging session. probeB's recording started earlier, so the same
    # pulse sits that much further along its axis.
    rows = ["probe,session,color,on_sample_concat,off_sample_concat"]
    for pulse_sample in (behavior_samples + 30, behavior_samples + 120):
        probeB_sample = pulse_sample + PROBE_OFFSET_SAMPLES
        rows.append(f"probeA,optotagging,blue_laser,{pulse_sample},{pulse_sample + 10}")
        rows.append(f"probeB,optotagging,blue_laser,{probeB_sample},{probeB_sample + 10}")
    (tmp_path / "laser_events.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_add_laser_events_maps_pulses_onto_the_nwb_clock(dummy_logger, tmp_path):
    """
    A pulse's sample index on a probe's own axis must land at the right time on the nwb clock, and
    the same pulse seen by both probes must land at the same time - which independently checks the
    probe alignment against a signal both probes recorded.
    """
    fs = 30000.0
    processed = _laser_fixture(tmp_path, fs=fs)
    boundaries = json.loads((processed / "session_boundaries.json").read_text())

    # probeB's recording started 45 samples EARLIER, so the same moment sits later on its axis
    offset = PROBE_OFFSET_SAMPLES / fs
    probe_start_times = {
        "ProbeA": {"behavior": 0.0, "optotagging": 500.0},
        "ProbeB": {"behavior": -offset, "optotagging": 500.0 - offset},
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
    processed = _laser_fixture(tmp_path)
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
    Make sure the cross-probe agreement check actually catches a misalignment. Here
    probeB's start time is wrong by 45 samples, so the same pulse maps to two different nwb times
    and the check must say so.
    """
    processed = _laser_fixture(tmp_path)
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
                             if "Laser pulses from" in record.message and record.levelno >= logging.WARNING]
    assert disagreement_warnings, "a 45-sample misalignment must be reported"
    # Whole samples, and the message says how far outside the tolerance it is
    assert "within 45 sample(s)" in disagreement_warnings[0].message
    assert "exceeds the 2-sample tolerance" in disagreement_warnings[0].message


def _write_preprocessing_json(tmp_path, conditions_by_probe, masked_on_probeB=None):
    """A preprocessing.json holding the optotagging conditions, shaped like the real one."""
    probes = {}
    for probe_key, conditions in conditions_by_probe.items():
        probes[probe_key] = {
            "optotagging_conditions": conditions,
            "lag_ms": {"blue_laser": 0.0333333},
            # Only set on a probe where preprocessing threw a whole condition away
            "noise_decisions": ({"conditions": [{"laser": "blue_laser",
                                                 "condition": masked_on_probeB,
                                                 "mask_whole_condition": True}]}
                                if masked_on_probeB and probe_key == "probeB" else None),
        }
    (tmp_path / "preprocessing.json").write_text(json.dumps({"probes": probes}))


def _two_blue_conditions(current_for_second=475.0):
    """One condition per pulse, so the two pulses in the fixture get different powers."""
    return [
        {"laser": "blue_laser", "condition": 1, "power": "1 mW", "current": 410.0,
         "first_pulse": 0, "last_pulse": 0, "pulses_with_events": 0},
        {"laser": "blue_laser", "condition": 2, "power": "5 mW", "current": current_for_second,
         "first_pulse": 1, "last_pulse": 1, "pulses_with_events": 0},
    ]


def _add_events(tmp_path, dummy_logger, fs=30000.0):
    """Run the fixture through add_laser_events with correct start times, and return the table."""
    offset = PROBE_OFFSET_SAMPLES / fs
    nwbfile = NWBFile(session_description="Mock", identifier="mock",
                      session_start_time=datetime.now(tz.tzlocal()))
    add_laser_events(
        nwbfile=nwbfile, processed_ephys_path=tmp_path,
        session_boundaries=json.loads((tmp_path / "session_boundaries.json").read_text()),
        probe_start_times={"ProbeA": {"behavior": 0.0, "optotagging": 500.0},
                           "ProbeB": {"behavior": -offset, "optotagging": 500.0 - offset}},
        session_hardware_starts={"behavior": 0.0, "optotagging": 500.0},
        session_nwb_starts={"behavior": 0.0, "optotagging": 452.0},
        logger=dummy_logger,
    )
    return nwbfile.intervals["laser_events"].to_dataframe()


def test_add_laser_events_labels_each_pulse_with_its_condition(dummy_logger, tmp_path):
    """
    Every pulse looks the same without this, so the table can only say that a pulse happened.
    The condition gives it a power and a current, and masked_on_probes says when preprocessing
    threw a whole condition away on one probe.
    """
    _laser_fixture(tmp_path)
    _write_preprocessing_json(tmp_path, {"probeA": _two_blue_conditions(),
                                         "probeB": _two_blue_conditions()},
                              masked_on_probeB=2)

    table = _add_events(tmp_path, dummy_logger)

    assert list(table["condition"]) == [1, 2]
    assert list(table["power"]) == [1.0, 5.0]
    assert list(table["current"]) == [410.0, 475.0]
    assert list(table["pulse_index"]) == [0, 1]
    assert list(table["lag_ms"]) == [0.0333333, 0.0333333]
    # The second condition was masked out on probeB only
    assert list(table["masked_on_probes"]) == ["", "ProbeB"]


def test_add_laser_events_rejects_conditions_that_differ_between_probes(dummy_logger, tmp_path):
    """
    Both probes recorded the same session, so if they disagree about what a condition was, a
    pulse's power depends on which probe we happened to read. The per-probe event COUNTS do
    differ, so only the fields that define a condition are compared.
    """
    _laser_fixture(tmp_path)
    different_current = _two_blue_conditions(current_for_second=999.0)
    _write_preprocessing_json(tmp_path, {"probeA": _two_blue_conditions(),
                                         "probeB": different_current})

    with pytest.raises(ValueError, match="different optotagging conditions"):
        _add_events(tmp_path, dummy_logger)


def test_add_laser_events_accepts_probes_that_saw_different_noise_counts(dummy_logger, tmp_path):
    """
    pulses_with_events counts what each probe saw, so it is different on every probe. This is the
    real IM-1994 case, and it must not be mistaken for a disagreement about the experiment.
    """
    _laser_fixture(tmp_path)
    probeB_conditions = _two_blue_conditions()
    probeB_conditions[0]["pulses_with_events"] = 100
    _write_preprocessing_json(tmp_path, {"probeA": _two_blue_conditions(),
                                         "probeB": probeB_conditions})

    table = _add_events(tmp_path, dummy_logger)

    assert list(table["condition"]) == [1, 2]


def test_add_laser_events_rejects_a_pulse_no_condition_covers(dummy_logger, tmp_path):
    """A pulse outside every condition range would be written with no power or current."""
    _laser_fixture(tmp_path)
    only_first_pulse = _two_blue_conditions()[:1]
    _write_preprocessing_json(tmp_path, {"probeA": only_first_pulse, "probeB": only_first_pulse})

    with pytest.raises(ValueError, match="falls in 0 optotagging conditions"):
        _add_events(tmp_path, dummy_logger)


def test_add_laser_events_rejects_a_power_that_is_not_in_mw(dummy_logger, tmp_path):
    """The nwb column is written in mW, so a power in other units would be mislabelled."""
    _laser_fixture(tmp_path)
    watts = _two_blue_conditions()
    watts[0]["power"] = "5 W"
    _write_preprocessing_json(tmp_path, {"probeA": watts, "probeB": watts})

    with pytest.raises(ValueError, match="is not in mW"):
        _add_events(tmp_path, dummy_logger)


def test_laser_events_are_moved_onto_the_photometry_clock(dummy_logger, tmp_path):
    """
    When photometry is the ground truth the pulses must make the same move the traces make.

    The ElectricalSeries timestamps and the spike times are interpolated from the ephys clock onto
    the photometry clock, using the port visits both streams recorded. Leaving the laser pulses
    behind would drift them away from the very traces they were recorded alongside, while the
    table goes on saying it is on the series' clock.
    """
    fs = 30000.0
    processed = _laser_fixture(tmp_path, fs=fs)
    boundaries = json.loads((processed / "session_boundaries.json").read_text())
    offset = PROBE_OFFSET_SAMPLES / fs

    # Port visits as the two streams recorded them: photometry reads 2s later throughout, so the
    # interpolation between them is a straight 2s shift and the expected times are exact
    metadata = {
        "ground_truth_time_source": "photometry",
        "ground_truth_visit_times": [102.0, 202.0, 302.0],
    }
    ephys_visit_times = [100.0, 200.0, 300.0]

    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))
    add_laser_events(
        nwbfile=nwbfile,
        processed_ephys_path=processed,
        session_boundaries=boundaries,
        probe_start_times={"ProbeA": {"behavior": 0.0, "optotagging": 500.0},
                           "ProbeB": {"behavior": -offset, "optotagging": 500.0 - offset}},
        session_hardware_starts={"behavior": 0.0, "optotagging": 500.0},
        session_nwb_starts={"behavior": 0.0, "optotagging": 452.0},
        logger=dummy_logger,
        metadata=metadata,
        ephys_visit_times=ephys_visit_times,
    )

    laser_events = nwbfile.intervals["laser_events"].to_dataframe()
    # The same two pulses as on the ephys clock (452.0 + 30/fs and 452.0 + 120/fs), each 2s later
    assert laser_events["start_time"][0] == pytest.approx(454.0 + 30 / fs)
    assert laser_events["start_time"][1] == pytest.approx(454.0 + 120 / fs)
    # A pulse keeps its 10-sample length: both edges go through the same interpolation
    assert laser_events["stop_time"][0] - laser_events["start_time"][0] == pytest.approx(10 / fs)


def test_laser_events_stay_on_the_ephys_clock_without_photometry(dummy_logger, tmp_path):
    """
    With no photometry, ephys IS the ground truth and the pulses are already on its clock, so
    nothing should move them. This is the case every optotagging session so far has been.
    """
    fs = 30000.0
    processed = _laser_fixture(tmp_path, fs=fs)
    boundaries = json.loads((processed / "session_boundaries.json").read_text())
    offset = PROBE_OFFSET_SAMPLES / fs

    nwbfile = NWBFile(session_description="Mock session", identifier="mock_session",
                      session_start_time=datetime.now(tz.tzlocal()))
    add_laser_events(
        nwbfile=nwbfile,
        processed_ephys_path=processed,
        session_boundaries=boundaries,
        probe_start_times={"ProbeA": {"behavior": 0.0, "optotagging": 500.0},
                           "ProbeB": {"behavior": -offset, "optotagging": 500.0 - offset}},
        session_hardware_starts={"behavior": 0.0, "optotagging": 500.0},
        session_nwb_starts={"behavior": 0.0, "optotagging": 452.0},
        logger=dummy_logger,
        metadata={"ground_truth_time_source": "ephys",
                  "ground_truth_visit_times": [100.0, 200.0, 300.0]},
        ephys_visit_times=[100.0, 200.0, 300.0],
    )

    laser_events = nwbfile.intervals["laser_events"].to_dataframe()
    assert laser_events["start_time"][0] == pytest.approx(452.0 + 30 / fs)
