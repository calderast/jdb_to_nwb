# Conversion of OPTOGENETICS data into the nwb: the optotagging laser pulses.
#
# The pulses are detected by Jifu's preprocessing pipeline, which writes laser_events.csv in the
# preprocessed session directory. Each pulse is recorded as a sample index on a probe's own
# concatenated axis, so putting them on the nwb clock uses the same mapping the spike times use
# (concat_sample_nwb_time in convert_processed_ephys).
#
# add_raw_ephys calls this after it writes the ElectricalSeries, since the pulses go on the clock
# the series established.

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from pynwb import NWBFile
from pynwb.epoch import TimeIntervals

from .utils import log_and_print
from .convert_raw_ephys import probe_name_token
from .convert_processed_ephys import (
    concat_sample_nwb_time,
    match_probe_key,
    read_preprocessing_json,
)


# The fields that DEFINE a condition. The rest of a condition entry (pulses_with_events,
# laser_period_events) counts what each probe saw, so those differ between probes.
CONDITION_FIELDS = ("laser", "condition", "power", "current", "first_pulse", "last_pulse")

# Two probes recording the same pulse map to within this many samples of each other. Measured on
# IM-1994: 728 pulses at 0 samples, 69 at 1, 3 at 2.
LASER_AGREEMENT_TOLERANCE_SAMPLES = 2


def _power_in_mw(power, logger):
    """The number out of a power string like '5 mW'. The nwb column is in mW, so the unit must be."""
    text = str(power).strip()
    match = re.fullmatch(r"([\d.]+)\s*mW", text, flags=re.IGNORECASE)
    if match is None:
        logger.error(f"Laser power '{text}' is not in mW")
        raise ValueError(f"Laser power '{text}' in preprocessing.json is not in mW, but the nwb "
                         "'power' column is written in mW. Convert it or add the unit.")
    return float(match.group(1))


def read_optotagging_conditions(processed_ephys_path: Path, probe_labels: list, logger) -> dict | None:
    """
    Read what each optotagging pulse was from preprocessing.json.

    The pipeline steps the laser through conditions - 100 pulses at 1 mW, then 100 at 5 mW, and so
    on per laser - and records each condition's pulse range, power and drive current. Without this
    every pulse in the nwb looks the same, so the laser_events table cannot be used for anything
    except "a pulse happened here".

    Returns a dict with 'conditions' (the condition list), 'lag_ms' (laser -> the ADC-to-probe lag
    already corrected for in the sample indices) and 'masked_conditions' ((laser, condition) ->
    the probes whose preprocessing masked that whole condition out), or None if there is no
    preprocessing.json.

    Parameters:
        processed_ephys_path (Path):
            The preprocessed session directory, holding preprocessing.json
        probe_labels (list):
            The probes being converted, used to read each one's conditions and lags
        logger (Logger):
            Logger to track conversion progress
    """
    preprocessing = read_preprocessing_json(processed_ephys_path, logger)
    if preprocessing is None:
        logger.info("Writing laser events without their condition, power and current.")
        return None

    probes = preprocessing.get("probes") or {}

    conditions_by_probe = {}
    lag_by_probe = {}
    masked_conditions = {}
    for probe_label in probe_labels:
        probe_key = match_probe_key(probes, probe_label)
        probe_info = probes[probe_key]
        conditions_by_probe[probe_label] = probe_info.get("optotagging_conditions") or []
        lag_by_probe[probe_label] = probe_info.get("lag_ms") or {}

        # A condition can be dropped entirely on one probe if too much of it was masked as noise.
        # noise_decisions is null when nothing was decided, which is the usual case.
        noise_decisions = probe_info.get("noise_decisions") or {}
        masked = noise_decisions.get("conditions") or []
        if not isinstance(masked, list):
            logger.warning(f"'{probe_label}' noise_decisions.conditions is a {type(masked).__name__}, "
                           "not a list; not recording which conditions were masked out.")
            masked = []
        for entry in masked:
            if isinstance(entry, dict) and entry.get("mask_whole_condition"):
                key = (entry.get("laser"), entry.get("condition"))
                masked_conditions.setdefault(key, []).append(probe_label)

    # Every probe must describe the same experiment, or a pulse's condition depends on which probe
    # we happened to read it from
    reference_label = probe_labels[0]
    reference = [{field: condition.get(field) for field in CONDITION_FIELDS}
                 for condition in conditions_by_probe[reference_label]]
    for probe_label in probe_labels[1:]:
        other = [{field: condition.get(field) for field in CONDITION_FIELDS}
                 for condition in conditions_by_probe[probe_label]]
        if other != reference:
            logger.error(f"'{reference_label}' and '{probe_label}' describe different optotagging "
                         f"conditions: {reference} vs {other}")
            raise ValueError(
                f"'{reference_label}' and '{probe_label}' describe different optotagging conditions "
                "in preprocessing.json. They recorded the same session, so the conditions must "
                "match for a pulse's condition to mean anything."
            )

    conditions = [
        {"laser": condition["laser"], "condition": int(condition["condition"]),
         "power_mw": _power_in_mw(condition["power"], logger), "current": float(condition["current"]),
         "first_pulse": int(condition["first_pulse"]), "last_pulse": int(condition["last_pulse"])}
        for condition in conditions_by_probe[reference_label]
    ]
    logger.info(f"Read {len(conditions)} optotagging condition(s) from preprocessing.json: "
                + ", ".join(f"{c['laser']} condition {c['condition']} pulses "
                            f"{c['first_pulse']}-{c['last_pulse']} at {c['power_mw']}mW "
                            f"({c['current']}mA)" for c in conditions))
    if masked_conditions:
        logger.warning("Conditions masked out of preprocessing entirely: "
                       + ", ".join(f"{laser} condition {condition} on {probes_masked}"
                                   for (laser, condition), probes_masked in masked_conditions.items()))
    return {"conditions": conditions, "lag_ms": lag_by_probe[reference_label],
            "masked_conditions": masked_conditions}


def condition_for_pulse(conditions: list, laser: str, pulse_index: int, logger) -> dict:
    """
    The condition a pulse belongs to, by its index within its own laser's pulses.

    Each laser steps through its own conditions, so the ranges restart per laser: blue pulses
    0-99 are blue condition 1 and red pulses 0-99 are red condition 1.
    """
    matches = [condition for condition in conditions
               if condition["laser"] == laser
               and condition["first_pulse"] <= pulse_index <= condition["last_pulse"]]
    if len(matches) != 1:
        logger.error(f"{laser} pulse {pulse_index} matches {len(matches)} conditions")
        raise ValueError(
            f"{laser} pulse {pulse_index} falls in {len(matches)} optotagging conditions, not 1. "
            "The conditions in preprocessing.json must cover every detected pulse exactly once, or "
            "the pulses in the nwb would be labelled with the wrong power and current."
        )
    return matches[0]


def add_laser_events(
    nwbfile: NWBFile,
    processed_ephys_path: Path,
    session_boundaries: dict,
    probe_start_times: dict,
    session_hardware_starts: dict,
    session_nwb_starts: dict,
    logger,
) -> None:
    """
    Write the optotagging laser pulses to the nwb as a TimeIntervals table, on the nwb clock.

    laser_events.csv records every pulse as sample indices on each probe's own concatenated axis,
    already corrected for the ADC-to-probe lag. Converting those to nwb times is exact arithmetic:
    the sample index gives the time on the recording's hardware clock, and each session's hardware
    start and nwb start (both recorded by the conversion) place it on the nwb clock. A pulse stays
    fixed to the sample it was detected on, so the ~1s uncertainty in the gap BETWEEN recordings
    does not move it.

    The same physical pulse appears once per probe, at different sample indices - that difference
    is how the probes' start offset was measured in the first place. After mapping they must agree,
    so the worst disagreement is logged: a check on the probe alignment that does not come from the
    raw timestamps the alignment itself came from. One row per physical pulse is written, from the
    first probe.

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled
        processed_ephys_path (Path):
            The preprocessed session directory, holding laser_events.csv
        session_boundaries (dict):
            The parsed session_boundaries.json (gives each probe's session ranges and sample rate)
        probe_start_times (dict):
            probe label -> {session label: start time on the hardware clock}
        session_hardware_starts (dict):
            session label -> the hardware-clock time of that session's first converted sample
        session_nwb_starts (dict):
            session label -> the nwb-clock time of that session's first converted sample
        logger (Logger):
            Logger to track conversion progress
    """
    laser_events_path = Path(processed_ephys_path) / "laser_events.csv"
    if not laser_events_path.exists():
        logger.info(f"No laser_events.csv at {laser_events_path}; no laser events to add.")
        return

    laser_events = pd.read_csv(laser_events_path)
    required_columns = {"probe", "session", "color", "on_sample_concat", "off_sample_concat"}
    missing_columns = required_columns - set(laser_events.columns)
    if missing_columns:
        logger.error(f"laser_events.csv is missing columns {sorted(missing_columns)}")
        raise ValueError(f"laser_events.csv is missing columns {sorted(missing_columns)}; "
                         f"it has {list(laser_events.columns)}.")
    logger.info(f"Read {len(laser_events)} laser event(s) from {laser_events_path}")

    streams = session_boundaries["streams"]

    def to_nwb_time(probe_label, session_label, concat_sample):
        """One pulse's sample index on a probe's concatenated axis -> its time on the nwb clock."""
        return concat_sample_nwb_time(
            probe_label=probe_label, session_label=session_label, concat_sample=concat_sample,
            session_boundaries=session_boundaries, probe_start_times=probe_start_times,
            session_hardware_starts=session_hardware_starts, session_nwb_starts=session_nwb_starts,
        )

    # Map every row, keeping only the probes and sessions this conversion actually wrote
    mapped_by_probe = {}
    skipped_sessions = set()
    for _row_index, row in laser_events.iterrows():
        probe_label = next((label for label in probe_start_times
                            if probe_name_token(label) == probe_name_token(row["probe"])), None)
        if probe_label is None:
            continue
        if row["session"] not in session_nwb_starts:
            skipped_sessions.add(row["session"])
            continue
        on_time = to_nwb_time(probe_label, row["session"], int(row["on_sample_concat"]))
        off_time = to_nwb_time(probe_label, row["session"], int(row["off_sample_concat"]))
        if on_time is None or off_time is None:
            skipped_sessions.add(row["session"])
            continue
        mapped_by_probe.setdefault(probe_label, []).append(
            {"session": row["session"], "color": row["color"], "start_time": on_time, "stop_time": off_time}
        )

    if skipped_sessions:
        logger.warning(f"Skipped laser events from session(s) {sorted(skipped_sessions)}, which this "
                       "conversion did not write.")
    if not mapped_by_probe:
        logger.warning("No laser events matched the probes and sessions being converted.")
        return

    # The same pulse seen by each probe must land at the same nwb time. This checks the probe
    # alignment against a signal both probes recorded, independently of the raw timestamps.
    probe_labels = list(mapped_by_probe)
    reference_label = probe_labels[0]
    reference_events = sorted(mapped_by_probe[reference_label], key=lambda event: event["start_time"])
    for other_label in probe_labels[1:]:
        other_events = sorted(mapped_by_probe[other_label], key=lambda event: event["start_time"])
        if len(other_events) != len(reference_events):
            logger.warning(f"'{reference_label}' detected {len(reference_events)} laser pulses but "
                           f"'{other_label}' detected {len(other_events)}; not comparing their times.")
            continue
        differences = [abs(a["start_time"] - b["start_time"])
                       for a, b in zip(reference_events, other_events)]
        sampling_rate = float(streams[match_probe_key(streams, reference_label)]["fs"])
        # Whole samples: a pulse is detected on a sample, so the two probes can only disagree by
        # a whole number of them, and comparing the raw float makes 2 samples look like 2.000001
        samples_apart = [round(difference * sampling_rate) for difference in differences]
        worst_samples = max(samples_apart)
        distribution = ", ".join(f"{samples_apart.count(n)} pulse(s) at {n}"
                                 for n in sorted(set(samples_apart)))
        message = (f"Laser pulses from '{reference_label}' and '{other_label}' are within "
                   f"{worst_samples} sample(s) ({max(differences) * 1000:.3f} ms) of each other: "
                   f"{distribution}. This independently checks the probe time alignment.")
        if worst_samples <= LASER_AGREEMENT_TOLERANCE_SAMPLES:
            log_and_print(logger, message, level="info")
        else:
            log_and_print(logger, message + f" That exceeds the {LASER_AGREEMENT_TOLERANCE_SAMPLES}"
                          "-sample tolerance, so the probes may not be aligned correctly.",
                          level="warning")

    # What each pulse was: its condition, power and current. Without this every pulse in the
    # table looks identical, so the only usable fact is that a pulse happened.
    condition_info = read_optotagging_conditions(processed_ephys_path, probe_labels, logger)
    if condition_info:
        # Each laser numbers its own pulses from 0, in time order, which is what the conditions'
        # first_pulse/last_pulse ranges count
        pulses_so_far = Counter()
        for event in reference_events:
            laser = str(event["color"])
            event["pulse_index"] = pulses_so_far[laser]
            pulses_so_far[laser] += 1
            condition = condition_for_pulse(condition_info["conditions"], laser,
                                            event["pulse_index"], logger)
            event["condition"] = condition["condition"]
            event["power"] = condition["power_mw"]
            event["current"] = condition["current"]
            # NaN when preprocessing recorded no lag for this laser: unknown, not zero
            event["lag_ms"] = float(condition_info["lag_ms"].get(laser, float("nan")))
            event["masked_on_probes"] = ",".join(
                condition_info["masked_conditions"].get((laser, condition["condition"]), [])
            )

    # One row per physical pulse, from the first probe
    laser_events_table = TimeIntervals(
        name="laser_events",
        description=("Optotagging laser pulses, on the same clock as the ElectricalSeries. Mapped "
                     "from laser_events.csv, which records each pulse as a sample index on a "
                     "probe's concatenated axis (already corrected for the ADC-to-probe lag)."),
    )
    columns = {
        "color": "Which laser produced the pulse",
        "session": "The recording the pulse occurred in",
    }
    if condition_info:
        columns.update({
            "condition": "The optotagging condition this pulse belongs to, numbered per laser",
            "power": "The laser power this condition was run at, in mW",
            "current": "The laser drive current this condition was run at, in mA",
            "pulse_index": "This pulse's index within its own laser's pulses, in time order",
            "lag_ms": ("The ADC-to-probe lag in ms that preprocessing already corrected for when "
                       "it wrote this pulse's sample index"),
            "masked_on_probes": ("The probes whose preprocessing masked this pulse's whole "
                                 "condition out of the traces, comma separated. Empty if none."),
        })
    for name, description in columns.items():
        laser_events_table.add_column(name=name, description=description)
    for event in reference_events:
        laser_events_table.add_row(start_time=float(event["start_time"]),
                                   stop_time=float(event["stop_time"]),
                                   **{name: event[name] for name in columns})

    nwbfile.add_time_intervals(laser_events_table)
    pulses_by_color = pd.Series([event["color"] for event in reference_events]).value_counts().to_dict()
    log_and_print(logger, f"Added {len(reference_events)} laser events to the nwb "
                          f"(from '{reference_label}'): {pulses_by_color}", level="info")
