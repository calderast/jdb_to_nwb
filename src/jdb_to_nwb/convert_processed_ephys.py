# Conversion of PREPROCESSED ephys traces (the output of the optonpx pipeline) into the nwb.
#
# The raw Open Ephys conversion lives in convert_raw_ephys.py. This module handles the case where
# the ElectricalSeries holds Jifu's preprocessed traces instead of the raw ones: laser artifacts
# and common-mode noise interpolated over, a Neuropixels 2.0 phase shift applied, and the source
# recordings already concatenated. Those are the traces Kilosort was run on, so a re-sort in
# Spyglass is comparable to the existing sorting.
#
# The raw recordings are still required and still drive the conversion. Everything structural
# comes from them - the ADC (port visits and time zero), settings.xml (channel map and geometry),
# structure.oebin (conversion factor, sample rate, channel count), and the per-stream timestamps
# (each probe's start time, which is what the probes are aligned by). Only the traces themselves,
# and the laser events, come from the preprocessed directory.
#
# ============================================================================================
# ASSUMPTIONS specific to preprocessed traces
# (see convert_raw_ephys.py for the assumptions about the recordings, the nwb and Spyglass)
#
#   1. Preprocessing preserved sample indices and channel ORDER from the raw recording, and did
#      not rescale: bit_volts from the raw structure.oebin is applied to them.
#      Checked: verify_processed_matches_raw correlates one second of the first probe's
#      preprocessed traces against the raw ones and warns if the channels do not correspond
#      (a reordering would otherwise map every electrode to the wrong trace, unnoticed). It
#      compares by correlation because the phase shift and the masking legitimately change the
#      samples. A warning there must be investigated before the file is used.
#   2. The preprocessed file holds exactly the neural channels (no ADC), and
#      session_boundaries.json describes it exactly. Checked: a length mismatch raises, and the
#      channel count must match the electrode table.
#   3. The 'session' labels in the metadata are the ones the preprocessing used. Checked: raises.
#   4. laser_events.csv sample indices are on each probe's own concatenated axis and are already
#      corrected for the ADC-to-probe lag (the pipeline does this). Checked indirectly: the same
#      pulse mapped from each probe must land at the same nwb time, logged every conversion.
# ============================================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBFile
from pynwb.epoch import TimeIntervals

from .utils import log_and_print
from .convert_raw_ephys import combine_probe_recordings, probe_name_token

try:
    from spikeinterface.extractors.extractor_classes import BinaryRecordingExtractor
except ImportError:  # spikeinterface < 0.103
    from spikeinterface.extractors import BinaryRecordingExtractor


# The preprocessing pipeline writes its per-probe folders in lower camel case ("probeA") while
# Open Ephys names the streams "ProbeA", so probe names are matched as probe tokens.


def match_probe_key(available_keys, probe_label: str) -> str:
    """
    Find the key naming this probe among keys written by the preprocessing pipeline.

    The pipeline writes "probeA"/"probeB" (folders and session_boundaries.json keys) while Open
    Ephys calls the same probe "ProbeA"/"ProbeB", so names are compared as probe tokens.

    Parameters:
        available_keys: the keys/folder names to search
        probe_label (str): the Open Ephys probe label, e.g. "ProbeA"

    Returns:
        str: the matching key, as written by the preprocessing pipeline
    """
    matches = [key for key in available_keys if probe_name_token(key) == probe_name_token(probe_label)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one entry for probe '{probe_label}', found {matches} "
                         f"among {list(available_keys)}")
    return matches[0]

def read_session_boundaries(processed_ephys_path: Path, logger) -> dict:
    """
    Read session_boundaries.json, which records how the preprocessing pipeline concatenated the
    source recordings into one continuous.dat per probe.

    We need it because the concatenated file has no time axis of its own: it is the only record of
    where one source session ends and the next begins, and the boundary sits at a DIFFERENT sample
    index for each probe (the probes do not record exactly the same number of samples).

    Parameters:
        processed_ephys_path (Path):
            The preprocessed session directory (holding session_boundaries.json and the probe folders)
        logger (Logger):
            Logger to track conversion progress

    Returns:
        dict: the parsed session_boundaries.json ("streams" maps each probe to its sessions)
    """
    boundaries_path = Path(processed_ephys_path) / "session_boundaries.json"
    if not boundaries_path.exists():
        logger.error(f"No session_boundaries.json found at {boundaries_path}")
        raise FileNotFoundError(f"No session_boundaries.json found at {boundaries_path}")

    with open(boundaries_path, "r") as boundaries_file:
        boundaries = json.load(boundaries_file)

    logger.info(f"Read session boundaries for '{boundaries.get('session_name')}' "
                f"(concatenate order {boundaries.get('concatenate_order')})")
    for probe_key, stream in boundaries["streams"].items():
        session_summary = {s["session_id"]: s["n_samples"] for s in stream["sessions"]}
        logger.debug(f"  {probe_key}: {stream['n_samples_total']} samples total at "
                     f"{stream['fs']} Hz, {stream['n_chan']} channels, sessions {session_summary}")
    return boundaries

def open_processed_probe_sessions(
    processed_ephys_path: Path,
    probe_label: str,
    probe_boundaries: dict,
    conversion_factor_uv: float,
    session_start_times: dict,
    logger,
) -> dict:
    """
    Open each source session of one probe's preprocessed, concatenated continuous.dat as its own
    SpikeInterface recording.

    The sessions are opened separately rather than sliced out of one recording because each probe
    starts a different few samples into each session, so every session needs its own start time.
    Opening at a byte offset lets each one carry the right t_start, which is what lets the probes
    be aligned to each other per session further downstream.

    The preprocessed file is a flat int16 binary with no metadata of its own, so sample rate,
    channel count and conversion factor all come from the RAW recording's structure.oebin, and the
    start times come from the raw per-stream timestamps. Channels are named CH0..CHn to match the
    raw stream so the electrode table maps onto them identically.

    Parameters:
        processed_ephys_path (Path):
            The preprocessed session directory
        probe_label (str):
            Open Ephys probe label, e.g. "ProbeA"
        probe_boundaries (dict):
            This probe's entry from session_boundaries.json (fs, n_chan, n_samples_total, sessions)
        conversion_factor_uv (float):
            bit_volts from the raw structure.oebin (preprocessing preserves the scale)
        session_start_times (dict):
            session id -> start time (seconds) of that session for THIS probe, on the shared
            hardware clock, taken from the raw recording's per-stream timestamps
        logger (Logger):
            Logger to track conversion progress

    Returns:
        dict: session id (e.g. "behavior") -> the recording for that session
    """
    probe_folder = match_probe_key(
        [path.name for path in Path(processed_ephys_path).iterdir() if path.is_dir()], probe_label
    )
    dat_path = Path(processed_ephys_path) / probe_folder / "continuous.dat"
    if not dat_path.exists():
        logger.error(f"No preprocessed continuous.dat for probe '{probe_label}' at {dat_path}")
        raise FileNotFoundError(f"No preprocessed continuous.dat for probe '{probe_label}' at {dat_path}")

    num_channels = int(probe_boundaries["n_chan"])
    sampling_frequency = float(probe_boundaries["fs"])
    bytes_per_sample = num_channels * np.dtype("int16").itemsize

    # The file must hold exactly what session_boundaries.json says, or every sample index we take
    # from it (session ranges, laser event times) points at the wrong data
    expected_num_samples = int(probe_boundaries["n_samples_total"])
    actual_num_samples = dat_path.stat().st_size // bytes_per_sample
    if actual_num_samples != expected_num_samples:
        logger.error(f"'{probe_label}' continuous.dat holds {actual_num_samples} samples but "
                     f"session_boundaries.json says {expected_num_samples}")
        raise ValueError(f"'{probe_label}' continuous.dat holds {actual_num_samples} samples but "
                         f"session_boundaries.json says {expected_num_samples}. The preprocessed "
                         "file and its boundaries file disagree.")

    session_recordings = {}
    for session in probe_boundaries["sessions"]:
        session_id = session["session_id"]
        sample_start, sample_end = int(session["sample_start"]), int(session["sample_end"])
        if session_id not in session_start_times:
            logger.error(f"No start time for '{probe_label}' session '{session_id}'")
            raise ValueError(f"No start time for '{probe_label}' session '{session_id}'. The probes "
                             "are aligned to each other by their raw start times, so a missing one "
                             "would misalign them without any sign that it had.")
        t_start = float(session_start_times[session_id])

        # Open at this session's byte offset so it can carry its own start time, then cut it to
        # the session's length (opening at an offset runs to the end of the file)
        recording = BinaryRecordingExtractor(
            file_paths=[str(dat_path)],
            sampling_frequency=sampling_frequency,
            dtype="int16",
            num_channels=num_channels,
            channel_ids=[f"CH{channel}" for channel in range(num_channels)],
            gain_to_uV=conversion_factor_uv,
            file_offset=sample_start * bytes_per_sample,
            t_starts=[t_start],
            is_filtered=True,  # these traces are preprocessed, not raw
        )
        session_recordings[session_id] = recording.frame_slice(
            start_frame=0, end_frame=sample_end - sample_start
        )
        logger.debug(f"  '{probe_label}' session '{session_id}': samples "
                     f"[{sample_start}, {sample_end}) = {sample_end - sample_start} samples, "
                     f"t_start {t_start}s")

    logger.info(f"Opened preprocessed '{probe_label}': {expected_num_samples} samples x "
                f"{num_channels} channels at {sampling_frequency} Hz, "
                f"sessions {list(session_recordings)}")
    return session_recordings

def get_processed_session_recordings(
    processed_ephys_path: Path,
    probe_streams_by_label: dict,
    session_labels: list,
    probe_start_times: dict,
    conversion_factor_uv: float,
    samples_to_remove: int,
    logger,
) -> tuple:
    """
    Build one combined recording per session from the preprocessed, concatenated files.

    The preprocessing pipeline concatenated the recordings already, so the sessions are split back
    out at the boundaries it recorded, aligned per session (the probes start a different few
    samples apart in each one), and combined across probes. The traces are the preprocessed ones -
    laser artifacts and common-mode noise interpolated over, phase shift applied - which is what
    Kilosort saw, so a re-sort is comparable to the existing sorting.

    Parameters:
        processed_ephys_path (Path):
            The preprocessed session directory
        probe_streams_by_label (dict):
            Short probe label -> full Open Ephys stream name, in ElectricalSeries channel order
        session_labels (list):
            The session names, in concatenation order, as given in the metadata
        probe_start_times (dict):
            probe label -> {session label: start time}, from the raw recordings
        conversion_factor_uv (float):
            bit_volts from the raw structure.oebin
        samples_to_remove (int):
            Samples to trim from the start of the FIRST session (data before bonsai start)
        logger (Logger):
            Logger to track conversion progress

    Returns:
        session_recordings (list): one combined recording per session, in order
        session_timestamps (list): each session's own timestamps
        num_samples_by_probe (dict): each probe's full concatenated length, for the sorting guard
    """
    boundaries = read_session_boundaries(processed_ephys_path=processed_ephys_path, logger=logger)
    streams = boundaries["streams"]

    # The metadata's session names have to be the ones the preprocessing pipeline used, or the
    # recordings would be matched to the wrong segments of the concatenated file
    for probe_label in probe_streams_by_label:
        probe_key = match_probe_key(streams, probe_label)
        available = [session["session_id"] for session in streams[probe_key]["sessions"]]
        missing = [label for label in session_labels if label not in available]
        if missing:
            logger.error(f"'{probe_label}' has no preprocessed session(s) {missing}; it has {available}")
            raise ValueError(f"'{probe_label}' has no preprocessed session(s) {missing}. The "
                             f"preprocessed file holds {available}. The 'session' names in the "
                             "metadata must match the ones the preprocessing used.")

    sessions_by_probe = {}
    num_samples_by_probe = {}
    for probe_label in probe_streams_by_label:
        probe_key = match_probe_key(streams, probe_label)
        sessions_by_probe[probe_label] = open_processed_probe_sessions(
            processed_ephys_path=processed_ephys_path,
            probe_label=probe_label,
            probe_boundaries=streams[probe_key],
            conversion_factor_uv=conversion_factor_uv,
            session_start_times=probe_start_times[probe_label],
            logger=logger,
        )
        # The sorting was run on the whole concatenated file, so that is the length its analyzer
        # must match
        num_samples_by_probe[probe_label] = int(streams[probe_key]["n_samples_total"])

    session_recordings = []
    session_timestamps = []
    for session_index, session_label in enumerate(session_labels):
        recordings_this_session = {
            probe_label: sessions_by_probe[probe_label][session_label]
            for probe_label in probe_streams_by_label
        }
        # Only the first session is trimmed to bonsai start
        if session_index == 0 and samples_to_remove:
            recordings_this_session = {
                probe_label: recording.frame_slice(start_frame=samples_to_remove, end_frame=None)
                for probe_label, recording in recordings_this_session.items()
            }
            logger.info(f"Trimmed {samples_to_remove} samples from the start of '{session_label}' "
                        "to drop data before bonsai start")

        combined_recording, timestamps = combine_probe_recordings(recordings_this_session, logger)
        session_recordings.append(combined_recording)
        session_timestamps.append(timestamps)

    return session_recordings, session_timestamps, num_samples_by_probe

def verify_processed_matches_raw(processed_recording, raw_recording, logger,
                                 num_samples: int = 30_000, min_correlation: float = 0.9) -> None:
    """
    Check that the preprocessed traces line up channel-for-channel with the raw ones.

    This is the assumption with the worst failure mode: if preprocessing reordered the channels,
    every electrode would be mapped to the wrong trace and nothing else in the conversion would
    notice. Nothing in the preprocessed file asserts the order, so it is checked here against the
    raw recording it came from.

    Compared by correlation rather than equality, because preprocessing legitimately changes the
    samples: the phase shift moves each channel by a fraction of a sample, and masked windows are
    replaced by interpolation. Neither changes which channel is which, so a channel that still
    corresponds to its raw counterpart stays highly correlated with it, while a reordered one does
    not. Only a short window is read, from the start of the recording.

    Logs the result and warns on a low correlation; it does not raise, because a low value can also
    mean the window happened to land on a masked stretch. A warning here should be investigated
    before the conversion is trusted.

    Parameters:
        processed_recording:
            The preprocessed recording for one probe
        raw_recording:
            The raw recording for the same probe, with the same channels in the same order
        logger (Logger):
            Logger to track conversion progress
        num_samples (int):
            How many samples to compare (default 30000, one second at 30 kHz)
        min_correlation (float):
            Warn if the median per-channel correlation is below this
    """
    comparable_samples = min(num_samples, processed_recording.get_num_samples(),
                             raw_recording.get_num_samples())
    if comparable_samples < 2:
        logger.warning("Too few samples to compare the preprocessed traces against the raw ones.")
        return

    if processed_recording.get_num_channels() != raw_recording.get_num_channels():
        logger.warning(f"Preprocessed recording has {processed_recording.get_num_channels()} channels "
                       f"but the raw one has {raw_recording.get_num_channels()}; not comparing them.")
        return

    processed_traces = processed_recording.get_traces(start_frame=0, end_frame=comparable_samples)
    raw_traces = raw_recording.get_traces(start_frame=0, end_frame=comparable_samples)

    correlations = []
    for channel_index in range(processed_traces.shape[1]):
        processed_channel = processed_traces[:, channel_index].astype("float64")
        raw_channel = raw_traces[:, channel_index].astype("float64")
        # A flat channel has no correlation to measure, so it is skipped rather than counted as bad
        if processed_channel.std() == 0 or raw_channel.std() == 0:
            continue
        correlations.append(float(np.corrcoef(processed_channel, raw_channel)[0, 1]))

    if not correlations:
        logger.warning("Every compared channel was flat; cannot check the preprocessed traces "
                       "against the raw ones.")
        return

    median_correlation = float(np.median(correlations))
    worst_correlation = float(np.min(correlations))
    message = (f"Preprocessed traces vs raw over {comparable_samples} samples: median per-channel "
               f"correlation {median_correlation:.4f}, worst {worst_correlation:.4f} "
               f"({len(correlations)} channels compared)")
    if median_correlation < min_correlation:
        log_and_print(
            logger,
            message + f". This is below {min_correlation}, which can mean the preprocessed channels "
            "are in a different order than the raw ones - in which case every electrode is mapped "
            "to the wrong trace. Investigate before using this file. (It can also mean the compared "
            "window landed on a masked stretch.)",
            level="warning",
        )
    else:
        log_and_print(logger, message + ". The preprocessed channels correspond to the raw ones.",
                      level="info")

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
    start and nwb start (both recorded by the conversion) place it on the nwb clock. Nothing is
    estimated, so this is unaffected by the ~1s uncertainty in the gap BETWEEN recordings - the
    pulses stay fixed to the samples they were detected on.

    The same physical pulse appears once per probe, at different sample indices (that difference is
    how the probes' start offset was originally measured). After mapping they must agree, so the
    worst disagreement is logged as a check on the probe alignment, independent of the raw
    timestamps the alignment itself came from. One row per physical pulse is written, using the
    first probe's mapped times.

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
        probe_key = match_probe_key(streams, probe_label)
        probe_stream = streams[probe_key]
        sampling_rate = float(probe_stream["fs"])
        session_entry = next((session for session in probe_stream["sessions"]
                              if session["session_id"] == session_label), None)
        if session_entry is None:
            return None
        # Time on the recording's hardware clock, then placed on the nwb clock
        hardware_time = (probe_start_times[probe_label][session_label]
                         + (concat_sample - int(session_entry["sample_start"])) / sampling_rate)
        return (hardware_time - session_hardware_starts[session_label]
                + session_nwb_starts[session_label])

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
        worst_samples = max(differences) * sampling_rate
        log_and_print(
            logger,
            f"Laser pulses agree between '{reference_label}' and '{other_label}' to within "
            f"{max(differences) * 1000:.3f} ms ({worst_samples:.1f} samples). This independently "
            "checks the probe time alignment.",
            level="info" if worst_samples <= 2 else "warning",
        )

    # One row per physical pulse, from the first probe
    laser_events_table = TimeIntervals(
        name="laser_events",
        description=("Optotagging laser pulses, on the same clock as the ElectricalSeries. Mapped "
                     "from laser_events.csv, which records each pulse as a sample index on a "
                     "probe's concatenated axis (already corrected for the ADC-to-probe lag)."),
    )
    laser_events_table.add_column(name="color", description="Which laser produced the pulse")
    laser_events_table.add_column(name="session", description="The recording the pulse occurred in")
    for event in reference_events:
        laser_events_table.add_row(start_time=float(event["start_time"]),
                                   stop_time=float(event["stop_time"]),
                                   color=str(event["color"]), session=str(event["session"]))

    nwbfile.add_time_intervals(laser_events_table)
    pulses_by_color = pd.Series([event["color"] for event in reference_events]).value_counts().to_dict()
    log_and_print(logger, f"Added {len(reference_events)} laser events to the nwb "
                          f"(from '{reference_label}'): {pulses_by_color}", level="info")
