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

import json
from pathlib import Path

import numpy as np

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


def read_preprocessing_json(processed_ephys_path: Path, logger) -> dict | None:
    """
    The preprocessing pipeline's record of how it made the traces, or None if it is not there.

    Holds the masking parameters, the optotagging conditions, the laser lags and the Kilosort
    parameters, per probe.

    Parameters:
        processed_ephys_path (Path):
            The preprocessed session directory
        logger (Logger):
            Logger to track conversion progress
    """
    preprocessing_path = Path(processed_ephys_path) / "preprocessing.json"
    if not preprocessing_path.exists():
        logger.info(f"No preprocessing.json at {preprocessing_path}.")
        return None
    return json.loads(preprocessing_path.read_text())


def read_kilosort_bad_channels(processed_ephys_path: Path, probe_labels: list, logger) -> dict:
    """
    The channels Kilosort was told to ignore, per probe.

    Each probe numbers its own channels from 0, so these are channel numbers within a probe, not
    rows of the electrodes table. Spyglass drops bad channels from its sort groups, so marking
    them here is what carries the preprocessing pipeline's decision into a re-sort.

    Returns probe label -> set of channel numbers, empty for a probe that lists none.

    Parameters:
        processed_ephys_path (Path):
            The preprocessed session directory, holding preprocessing.json
        probe_labels (list):
            The probes being converted
        logger (Logger):
            Logger to track conversion progress
    """
    preprocessing = read_preprocessing_json(processed_ephys_path, logger)
    if preprocessing is None:
        logger.info("No preprocessing.json, so no channels are marked bad.")
        return {probe_label: set() for probe_label in probe_labels}

    probes = preprocessing.get("probes") or {}
    bad_channels_by_probe = {}
    for probe_label in probe_labels:
        probe_key = match_probe_key(probes, probe_label)
        kilosort_params = probes[probe_key].get("kilosort_params") or {}
        bad_channels = kilosort_params.get("bad_channels") or []
        bad_channels_by_probe[probe_label] = {int(channel) for channel in bad_channels}
    logger.info("Bad channels from preprocessing.json (Kilosort ignored these): "
                + ", ".join(f"'{label}': {sorted(channels) if channels else 'none'}"
                            for label, channels in bad_channels_by_probe.items()))
    return bad_channels_by_probe


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

    # The preprocessed file holds exactly the neural channels and no ADC, so n_chan is the row
    # length of the binary and must also match the electrode table built from the raw channel map
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
    session_boundaries: dict,
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
        session_boundaries (dict):
            The already-parsed session_boundaries.json (from read_session_boundaries)
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
        sessions_by_probe (dict): probe label -> {session label: that probe's recording}, so a
            caller can check one against the raw traces without opening it again
    """
    streams = session_boundaries["streams"]

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

    return session_recordings, session_timestamps, num_samples_by_probe, sessions_by_probe

def verify_processed_matches_raw(processed_recording, raw_recording, logger, label: str,
                                 num_samples: int = 30_000, min_correlation: float = 0.9) -> None:
    """
    Check that one probe's preprocessed recording still corresponds to the raw one it came from.

    Nothing in the preprocessed file states its sample count, channel count or channel order, and
    the conversion relies on all three: the electrode table is built from the raw channel map, and
    a spike's sample index is an index into this recording. Three things are checked:

      - The sample count, which must match exactly. A length change puts every spike after it on
        the wrong sample. Raises.
      - The channel count, which must match too, or the traces and the electrode table describe
        different channels. Raises.
      - The channel order, over one second of traces from the start. Each preprocessed channel is
        correlated against EVERY raw channel, and the best match must be its own. This catches a
        swap of two channels as well as every channel being out of order. Warns.

    Order is checked by correlation, not equality, because preprocessing is supposed to change
    the samples: the phase shift moves each channel by a fraction of a sample, and masked windows
    are replaced by interpolation. Neither of those changes which channel is which, so a channel
    still correlates best with its own raw channel.

    Parameters:
        processed_recording:
            The preprocessed recording for one probe and one session
        raw_recording:
            The raw recording for the same probe and session, untrimmed, so that sample 0 is the
            same sample in both
        logger (Logger):
            Logger to track conversion progress
        label (str):
            Which probe and session this is, for the log messages
        num_samples (int):
            How many samples to correlate (default 30000, one second at 30 kHz)
        min_correlation (float):
            Warn if the median channel's correlation with its own raw counterpart is below this
    """
    if processed_recording.get_num_samples() != raw_recording.get_num_samples():
        logger.error(f"{label}: preprocessed recording has {processed_recording.get_num_samples()} "
                     f"samples but the raw one has {raw_recording.get_num_samples()}")
        raise ValueError(
            f"{label}: the preprocessed recording has {processed_recording.get_num_samples()} "
            f"samples but the raw recording it came from has {raw_recording.get_num_samples()}. "
            "Preprocessing writes one sample per raw sample, and spike sample indices are indices "
            "into this recording, so a different length puts every spike after it on the wrong "
            "sample. Check that the preprocessed directory was built from these recordings."
        )
    if processed_recording.get_num_channels() != raw_recording.get_num_channels():
        logger.error(f"{label}: preprocessed recording has {processed_recording.get_num_channels()} "
                     f"channels but the raw one has {raw_recording.get_num_channels()}")
        raise ValueError(
            f"{label}: the preprocessed recording has {processed_recording.get_num_channels()} "
            f"channels but the raw recording has {raw_recording.get_num_channels()}. The electrode "
            "table is built from the raw channel map, so the two must describe the same channels."
        )

    comparable_samples = min(num_samples, processed_recording.get_num_samples())
    if comparable_samples < 2:
        logger.warning(f"{label}: too few samples to compare the preprocessed traces against the raw ones.")
        return

    processed_traces = processed_recording.get_traces(start_frame=0, end_frame=comparable_samples)
    raw_traces = raw_recording.get_traces(start_frame=0, end_frame=comparable_samples)

    # Correlate every preprocessed channel against every raw channel in one matmul: center each
    # column and divide by its norm, so the product of two columns IS their correlation.
    def centered_and_scaled(traces):
        centered = traces.astype("float32") - traces.mean(axis=0, dtype="float32")
        norms = np.linalg.norm(centered, axis=0)
        # A flat channel has nothing to correlate. Its norm is 0, so leave the column alone and
        # exclude it below instead of dividing by zero.
        return centered / np.where(norms == 0, 1.0, norms), norms

    processed_scaled, processed_norms = centered_and_scaled(processed_traces)
    raw_scaled, raw_norms = centered_and_scaled(raw_traces)
    correlations = processed_scaled.T @ raw_scaled

    comparable = (processed_norms > 0) & (raw_norms > 0)
    if not comparable.any():
        logger.warning(f"{label}: every compared channel was flat; cannot check the preprocessed "
                       "traces against the raw ones.")
        return

    own_correlations = np.diagonal(correlations)[comparable]
    best_match = np.argmax(correlations, axis=1)
    mismatched = np.flatnonzero(comparable & (best_match != np.arange(correlations.shape[0])))

    median_correlation = float(np.median(own_correlations))
    message = (f"{label}: preprocessed vs raw over {comparable_samples} samples, median channel "
               f"correlation with its own raw channel {median_correlation:.4f}, worst "
               f"{float(np.min(own_correlations)):.4f} ({int(comparable.sum())} channels compared)")

    if len(mismatched):
        examples = ", ".join(f"{index}->{best_match[index]}" for index in mismatched[:5])
        log_and_print(
            logger,
            message + f". {len(mismatched)} channel(s) correlate better with a DIFFERENT raw "
            f"channel than with their own ({examples}), which means the preprocessed channels are "
            "not in the raw order and every electrode is mapped to the wrong trace. Do not use "
            "this file until this is explained.",
            level="warning",
        )
    elif median_correlation < min_correlation:
        log_and_print(
            logger,
            message + f". Every channel still matches its own raw channel best, but the median is "
            f"below {min_correlation}, so the traces differ more than the phase shift and masking "
            "explain. (It can also mean the compared window landed on a masked stretch.)",
            level="warning",
        )
    else:
        log_and_print(logger, message + ". Every channel matches its own raw channel best.",
                      level="info")

def _session_entry(probe_stream: dict, session_label: str) -> dict | None:
    """That session's entry in a probe's stream, or None if the probe has no such session."""
    return next((session for session in probe_stream["sessions"]
                 if session["session_id"] == session_label), None)


def concat_sample_nwb_time(
    probe_label: str,
    session_label: str,
    concat_sample: int,
    session_boundaries: dict,
    probe_start_times: dict,
    session_hardware_starts: dict,
    session_nwb_starts: dict,
) -> float | None:
    """
    One sample index on a probe's concatenated preprocessed axis -> its time on the nwb clock.

    This is exact arithmetic, and it is per session: the sample index gives a time on the
    recording's hardware clock, and that session's hardware start and nwb start (both recorded by
    the conversion) place it on the nwb clock. Because each session is mapped through its OWN
    starts, the gap between recordings and any samples trimmed off either end of a session are
    already accounted for.

    Returns None if the probe has no such session in session_boundaries.

    Parameters:
        probe_label (str):
            Which probe's axis the sample index is on
        session_label (str):
            Which recording the sample is in (the caller already knows, e.g. from a csv column)
        concat_sample (int):
            The sample index on that probe's concatenated axis
        session_boundaries (dict):
            The parsed session_boundaries.json (gives each probe's session ranges and sample rate)
        probe_start_times (dict):
            probe label -> {session label: start time on the hardware clock}
        session_hardware_starts (dict):
            session label -> the hardware-clock time of that session's first converted sample
        session_nwb_starts (dict):
            session label -> the nwb-clock time of that session's first converted sample
    """
    streams = session_boundaries["streams"]
    probe_stream = streams[match_probe_key(streams, probe_label)]
    session_entry = _session_entry(probe_stream, session_label)
    if session_entry is None:
        return None
    sampling_rate = float(probe_stream["fs"])
    hardware_time = (probe_start_times[probe_label][session_label]
                     + (concat_sample - int(session_entry["sample_start"])) / sampling_rate)
    return (hardware_time - session_hardware_starts[session_label]
            + session_nwb_starts[session_label])


def concat_samples_to_nwb_times(
    probe_label: str,
    concat_samples,
    session_boundaries: dict,
    probe_start_times: dict,
    session_hardware_starts: dict,
    session_nwb_starts: dict,
):
    """
    Map many sample indices on one probe's concatenated axis onto the nwb clock, in bulk.

    Same arithmetic as concat_sample_nwb_time, but the session is worked out FROM the sample index
    rather than given, so this suits data that only knows its sample index - spike trains, where
    every spike carries a sample index on its probe's own concatenated preprocessed axis and
    nothing else.

    Returns (nwb_times, session_labels), both arrays as long as concat_samples. A sample in a
    session this conversion did not write gets NaN and "".

    Parameters:
        probe_label (str):
            Which probe's axis the sample indices are on
        concat_samples (array-like):
            Sample indices on that probe's concatenated axis
        session_boundaries, probe_start_times, session_hardware_starts, session_nwb_starts:
            As in concat_sample_nwb_time
    """
    samples = np.asarray(concat_samples, dtype=np.int64)
    nwb_times = np.full(samples.shape, np.nan, dtype=float)
    session_labels = np.full(samples.shape, "", dtype=object)

    streams = session_boundaries["streams"]
    probe_stream = streams[match_probe_key(streams, probe_label)]
    sampling_rate = float(probe_stream["fs"])

    for session_entry in probe_stream["sessions"]:
        session_label = session_entry["session_id"]
        # Only the sessions this conversion actually wrote can be placed on the nwb clock
        if (session_label not in session_nwb_starts or session_label not in session_hardware_starts
                or session_label not in (probe_start_times.get(probe_label) or {})):
            continue
        sample_start = int(session_entry["sample_start"])
        in_session = (samples >= sample_start) & (samples < int(session_entry["sample_end"]))
        if not in_session.any():
            continue
        hardware_times = (probe_start_times[probe_label][session_label]
                          + (samples[in_session] - sample_start) / sampling_rate)
        nwb_times[in_session] = (hardware_times - session_hardware_starts[session_label]
                                 + session_nwb_starts[session_label])
        session_labels[in_session] = session_label

    return nwb_times, session_labels
