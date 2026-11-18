import numpy as np
import pandas as pd
from math import isclose
from pathlib import Path

from pynwb import NWBFile
from pynwb.misc import Units

from .mdasortinginterface import MdaSortingInterface
from .utils import log_and_print
from .timestamps_alignment import align_via_interpolation
from .convert_processed_ephys import concat_samples_to_nwb_times, match_probe_key


def add_spikes(nwbfile: NWBFile, metadata: dict, logger):
    """
    Add spike sorting output to the NWB file as a Units table.

    Dispatches based on which spike sorting metadata is present:
      - 'sorting_analyzer_path': a SpikeInterface SortingAnalyzer (e.g. Kilosort4 + BombCell,
        saved as analyzer.zarr) -> add_kilosort_bombcell_spikes
      - 'mountain_sort_output_file_path' + 'sampling_frequency': MountainSort .mda -> add_mountainsort_spikes

    If neither is present, spike conversion is skipped.
    """
    if "ephys" not in metadata:
        return

    # The ephys metadata is in its current shape by now: create_nwbs normalizes an older file
    # (where 'sorting_analyzer_path' sat directly under 'ephys') before any converter runs.
    ephys = metadata["ephys"]

    # Each probe is sorted on its own channels, so each has its own analyzer
    probe_entries = ephys.get("probes") or []
    probes_with_sorting = [
        (probe_index, probe) for probe_index, probe in enumerate(probe_entries)
        if probe.get("sorting_analyzer_path")
    ]
    if probes_with_sorting:
        for probe_index, probe in probes_with_sorting:
            add_kilosort_bombcell_spikes(
                nwbfile=nwbfile,
                metadata=metadata,
                probe=probe,
                probe_index=probe_index,
                logger=logger,
            )
    elif "mountain_sort_output_file_path" in ephys and "sampling_frequency" in ephys:
        add_mountainsort_spikes(nwbfile=nwbfile, metadata=metadata, logger=logger)
    else:
        log_and_print(logger, "No spike sorting metadata found for this session. Skipping spike conversion.",
                      level="info")


def align_spike_times(spike_times_s: np.ndarray, metadata: dict, logger) -> np.ndarray:
    """
    Alignment STEP 2: put already-bonsai-shifted spike times (in seconds) onto the session's
    "ground truth" clock (that every data stream in the NWB is aligned to)

    Which stream is the ground truth is decided earlier in the pipeline (metadata["ground_truth_time_source"]):

      - "photometry": photometry was recorded, so it is the ground truth. The spikes are in ephys time,
        so we interpolate ephys -> photometry using the port-visit pulses that both streams recorded as
        shared sync pulses. This is the exact same interpolation add_raw_ephys applies to the raw ephys
        ElectricalSeries timestamps, so spikes and raw ephys stay lined up.

      - "ephys" (or anything else / None): there is no photometry, so ephys itself is the ground truth.
        The spikes are sorted from the ephys recording, so they are already on the ground truth clock -
        just return them unchanged (no interpolation).

    NOTE: interpolation only happens in the "photometry" branch, and that branch always returns. So if
    execution reaches the end of this function, the spike times were NOT interpolated.
    """
    ground_truth_time_source = metadata.get("ground_truth_time_source")

    # Ground truth is photometry: interpolate ephys spike times onto the photometry clock
    if ground_truth_time_source == "photometry":
        # Interpolation needs BOTH sets of sync pulses: the ephys port visits (times in the spikes' own
        # clock) and the photometry port visits (the matching times in the ground truth clock).
        ephys_visit_times = metadata.get("ephys_visit_times")
        ground_truth_visit_times = metadata.get("ground_truth_visit_times")

        # If either set is missing or empty (e.g. no raw ephys was converted, or no port visit pulses
        # were detected), we can't build the interpolation. Warn if this ever happens. 
        # (I think this should never happen, but I am allowing it to continue without throwing an error
        # on the off chance we want sorting but no raw ephys because raw ephys would be too big? idk)
        if not ephys_visit_times or not ground_truth_visit_times:
            logger.warning("Ground truth is photometry, but the ephys and/or photometry port-visit "
                          "times needed to interpolate are missing or empty. Leaving spike times "
                          "bonsai-shifted only (NOT aligned to photometry).")
            return spike_times_s

        logger.info("Ground truth is photometry: interpolating spike times onto the photometry "
                    "clock using port visits.")
        return np.asarray(align_via_interpolation(
            unaligned_timestamps=spike_times_s,
            unaligned_visit_times=ephys_visit_times,
            ground_truth_visit_times=ground_truth_visit_times,
            logger=logger,
            label="spike",
        ))

    # Ground truth is ephys (no photometry): spikes are already on the ground truth clock
    logger.info(f"Ground truth is '{ground_truth_time_source}' (not photometry): spikes are sorted "
                  "from the ephys recording, so they are already on the ground truth clock. No interpolation "
                  "needed.")
    return spike_times_s


def check_spikes_within_session_epochs(spike_times_s, spike_sessions, sample_mapping: dict,
                                      probe_label: str, logger) -> None:
    """
    Check mapped spike times land inside the epoch of the recording they came from.

    This is the spike equivalent of the laser-pulse agreement check: it tests the mapping against
    something known independently, namely where the conversion actually placed each recording.

    A spike can sit a little outside its epoch, because a probe's own recording is longer than
    what was written for it - samples before the bonsai start are trimmed off the front, and
    samples past the shortest probe are trimmed off the end. Anything further out than those
    trims explain means the mapping is wrong. That raises instead of warning, because wrong spike
    times look just like real ones once they are in the nwb.
    """
    epochs_by_session = {epoch["session"]: epoch for epoch in sample_mapping["session_epochs"]}
    streams = sample_mapping["session_boundaries"]["streams"]
    probe_stream = streams[match_probe_key(streams, probe_label)]
    sampling_rate = float(probe_stream["fs"])

    for session_entry in probe_stream["sessions"]:
        session_label = session_entry["session_id"]
        epoch = epochs_by_session.get(session_label)
        if epoch is None:
            continue
        in_session = spike_sessions == session_label
        if not in_session.any():
            continue
        session_spikes = spike_times_s[in_session]

        # Each bound gets its own trim. A spike can sit earlier than the epoch by the front
        # trim (the bonsai trim on the first session, plus whatever aligned this probe to the
        # others) and later by the end trim (whatever ran past the written length).
        own_duration = int(session_entry["n_samples"]) / sampling_rate
        epoch_duration = epoch["stop_time"] - epoch["start_time"]
        front_trim_s = max(0.0, sample_mapping["session_hardware_starts"][session_label]
                           - sample_mapping["probe_start_times"][probe_label][session_label])
        end_trim_s = max(0.0, own_duration - front_trim_s - epoch_duration)
        one_sample = 1 / sampling_rate
        earliest_allowed = epoch["start_time"] - front_trim_s - one_sample
        latest_allowed = epoch["stop_time"] + end_trim_s + one_sample

        outside_epoch = ((session_spikes < epoch["start_time"])
                         | (session_spikes > epoch["stop_time"])).sum()
        too_far = (session_spikes < earliest_allowed) | (session_spikes > latest_allowed)
        logger.info(
            f"'{probe_label}' session '{session_label}': {len(session_spikes)} spikes, "
            f"{outside_epoch} outside the epoch ({epoch['start_time']:.6f}s to "
            f"{epoch['stop_time']:.6f}s). Expected, from this probe's trims: up to "
            f"{round(front_trim_s * sampling_rate)} samples before it (trimmed off the front) and "
            f"{round(end_trim_s * sampling_rate)} after it (trimmed off the end)."
        )
        if too_far.any():
            worst_s = max(earliest_allowed - session_spikes.min(),
                          session_spikes.max() - latest_allowed)
            logger.error(f"{int(too_far.sum())} spikes from '{probe_label}' session "
                         f"'{session_label}' are outside its epoch by more than the trims explain.")
            raise ValueError(
                f"{int(too_far.sum())} of {len(session_spikes)} spikes from '{probe_label}' session "
                f"'{session_label}' fall outside that session's epoch "
                f"({epoch['start_time']:.6f}s to {epoch['stop_time']:.6f}s) by more than the "
                f"trimmed samples could explain (worst is {worst_s:.3f}s past the allowed range, "
                f"{round(worst_s * sampling_rate)} samples). Either the spike sample indices are "
                "not on the concatenated axis this mapping assumes, or the session start times and "
                "the epochs disagree about where this recording sits on the nwb clock."
            )


def aligned_spike_trains_by_unit(sorting, sampling_frequency: float, metadata: dict, logger,
                                 probe_label: str = None) -> list:
    """
    Take a sorter's spike trains and return them on the NWB clock, grouped one array per unit.

    A sorter reports each spike as a sample index into the raw recording. We put those spikes onto the
    same clock as the rest of the NWB using the SAME two steps add_raw_ephys applies to the raw ephys
    ElectricalSeries timestamps (so spikes and raw ephys line up):

      STEP 1 (bonsai shift): samples -> seconds, then subtract the bonsai start time so that the bonsai
                             start becomes time 0.
      STEP 2 (ground truth alignment): interpolate onto the photometry clock if photometry is the ground
                             truth, otherwise leave as-is (ephys is the ground truth). See align_spike_times.

    Shared by the MountainSort and Kilosort paths so both sorters are aligned identically.
    Returns a list of numpy arrays, one per unit in sorting.unit_ids order (each still time-ordered).
    """
    # to_spike_vector() returns every spike as ('sample_index', 'unit_index'), sorted by time
    spike_vector = sorting.to_spike_vector()
    all_unit_index = spike_vector["unit_index"]
    sample_index = spike_vector["sample_index"]
    logger.debug(f"Aligning: {len(spike_vector)} spikes | raw sample_index range "
                 f"[{int(sample_index.min())}, {int(sample_index.max())}] @ {sampling_frequency} Hz")

    # STEP 1: sample indices -> seconds on the nwb clock
    sample_mapping = metadata.get("ephys_sample_mapping")
    if sample_mapping and probe_label:
        # The traces came from the preprocessed files, so a spike's sample index is on this probe's
        # OWN concatenated axis: one axis spanning every recording, trimmed differently per probe.
        # Mapping each spike through its own session's recorded start times carries the gap between
        # recordings and this probe's trims. See concat_samples_to_nwb_times.
        logger.info(f"Mapping '{probe_label}' spike sample indices onto the nwb clock per session "
                    "(preprocessed traces), the same way the laser pulses are mapped.")
        all_spike_times_s, spike_sessions = concat_samples_to_nwb_times(
            probe_label=probe_label,
            concat_samples=sample_index,
            session_boundaries=sample_mapping["session_boundaries"],
            probe_start_times=sample_mapping["probe_start_times"],
            session_hardware_starts=sample_mapping["session_hardware_starts"],
            session_nwb_starts=sample_mapping["session_nwb_starts"],
        )
        # A spike in a recording this nwb does not contain has no place on the nwb clock, so we
        # drop it. This happens when the preprocessed directory holds more recordings than the
        # metadata lists: the sorting covers all of them, and the sample-count check above passes
        # because it compares the analyzer against the whole preprocessed file.
        unmapped = np.isnan(all_spike_times_s)
        if unmapped.any():
            logger.warning(f"Dropped {int(unmapped.sum())} of {len(all_spike_times_s)} '{probe_label}' "
                           "spikes: they fall in a recording this conversion did not write. The "
                           "sorting covers every recording in the preprocessed directory, so list "
                           "them all in 'openephys_folder_paths' to keep these spikes.")
            all_spike_times_s = all_spike_times_s[~unmapped]
            spike_sessions = spike_sessions[~unmapped]
            all_unit_index = all_unit_index[~unmapped]
        check_spikes_within_session_epochs(all_spike_times_s, spike_sessions, sample_mapping,
                                           probe_label, logger)
    else:
        # The traces came from the raw files: one recording, so the bonsai start is the only shift.
        # bonsai_start_time (seconds after ephys start that bonsai started) comes from
        # add_raw_ephys, which sets it to time 0 for the raw ephys. If there was no raw ephys this
        # session it is None, so we can't shift - warn and leave spike times relative to the raw
        # recording start (bonsai_start_time = 0). But that should never really happen (if we have
        # spikes we should also have raw ephys).
        bonsai_start_time = metadata.get("ephys_bonsai_start_time")
        if bonsai_start_time is None:
            logger.warning("No 'ephys_bonsai_start_time' (no raw ephys this session??): leaving spike times "
                          "relative to the raw recording start, NOT shifted to bonsai start.")
            bonsai_start_time = 0.0
        logger.debug(f"Bonsai shift = -{bonsai_start_time}s")
        # Spikes before the bonsai start keep their negative times - they were really recorded
        all_spike_times_s = sample_index / sampling_frequency - bonsai_start_time

    # STEP 2: ground truth alignment to account for drift between clocks (applied to the whole array at once)
    all_spike_times_s = align_spike_times(all_spike_times_s, metadata, logger)

    # Group the flat, aligned spike-time array back into one train per unit. Sorting by unit index with a
    # STABLE sort preserves each unit's time ordering; searchsorted then gives each unit's slice boundaries.
    num_units = len(sorting.unit_ids)
    order = np.argsort(all_unit_index, kind="stable")
    sorted_unit_index = all_unit_index[order]
    sorted_spike_times = all_spike_times_s[order]
    unit_boundaries = np.searchsorted(sorted_unit_index, np.arange(num_units + 1))
    return [sorted_spike_times[unit_boundaries[i]:unit_boundaries[i + 1]] for i in range(num_units)]


def create_units_table(nwbfile: NWBFile, sampling_rate: float, description: str, logger) -> None:
    """
    Create the nwb's Units table, so it can say what its spike times' resolution is.

    NWBFile.add_unit makes the table on demand, but only with a name and a description
    ("Autogenerated by NWBFile"); resolution, waveform_rate and electrode_table can only be set
    when the table is created. Spike times come to us as sample indices, so one sample IS the
    resolution, and the waveform means are averages of samples at that same rate.

    Only the first probe makes the table, since this runs once per probe and every probe's units
    go in the one table; a later probe just has its sampling rate checked against it. Call it
    after the electrodes table is written, which is what a unit's 'electrodes' reference points
    into.
    """
    # float because the metadata may give the rate as an int, which pynwb rejects
    sampling_rate = float(sampling_rate)
    if nwbfile.units is not None:
        # The table carries ONE resolution, already written by the first probe, so a probe whose
        # sorting ran at a different rate would have its spike times described by the wrong number
        if isclose(nwbfile.units.waveform_rate, sampling_rate, rel_tol=1e-6):
            logger.debug("Units table already exists (an earlier probe made it); not remaking it.")
        else:
            logger.warning(f"The Units table says {nwbfile.units.waveform_rate} Hz, from the first probe "
                           f"converted, but this probe's sorting is {sampling_rate} Hz, so its spike times "
                           "are described by the wrong resolution. Was each probe sorted from the same "
                           "recording?")
        return

    logger.debug(f"Creating the Units table at {sampling_rate} Hz "
                 f"(resolution {1 / sampling_rate:.3e}s, one sample)")
    nwbfile.units = Units(
        name="units",
        description=description,
        resolution=1 / sampling_rate,
        waveform_rate=sampling_rate,
        # The table the 'electrodes' column points into. Units resolves this through the nwbfile
        # anyway, but only once a unit with electrodes has been added, with a warning if it can't.
        electrode_table=nwbfile.electrodes,
    )


def check_unit_columns_match(nwbfile: NWBFile, probe_label: str, column_names: list, logger) -> None:
    """
    Check that this probe writes the same unit columns as the probes already in the Units table.

    Every probe's units go in the one table, so a column one probe fills and another cannot is not
    just a hole: add_unit has no value to put there and fails inside hdmf with "'NoneType' object
    is not iterable", naming nothing. Probes come out this way when their analyzers were computed
    differently - one without the 'templates' extension, or without a quality metric - so say
    which columns differ while we still know.
    """
    if len(nwbfile.units) == 0:
        return

    already_written = set(nwbfile.units.colnames) - {"spike_times"}
    ours = set(column_names)
    if already_written == ours:
        return

    cannot_fill = sorted(already_written - ours)
    nowhere_to_put = sorted(ours - already_written)
    logger.error(f"Unit columns for {probe_label} do not match the table: missing {cannot_fill}, "
                 f"extra {nowhere_to_put}")
    raise ValueError(
        f"The units of {probe_label} do not have the same columns as the units already in this nwb. "
        f"Written by an earlier probe but empty for {probe_label}: {cannot_fill}. Present for "
        f"{probe_label} but empty for the earlier probes: {nowhere_to_put}. Every probe's units share "
        "one table, so every probe must carry the same per-unit information. Compute the same "
        "extensions and quality metrics for every probe's sorting analyzer, then convert again."
    )


def add_mountainsort_spikes(nwbfile: NWBFile, metadata: dict, logger):
    """
    Add MountainSort output (.mda) to the NWB file as a Units table.

    Reads the firings.mda via NeuroConv's MdaSortingInterface and aligns the spike times to the NWB
    clock (bonsai shift + photometry interpolation) exactly like the Kilosort path, so both sorters end
    up on the same time base as the rest of the file. The firings.mda holds only spike times and unit
    labels, so we just carry the original sorter unit id as a 'unit_name' column.
    """
    log_and_print(logger, "Adding MountainSort spikes...", level="info")
    mountain_sort_output_file_path = metadata["ephys"]["mountain_sort_output_file_path"]
    sampling_frequency = metadata["ephys"]["sampling_frequency"]

    interface = MdaSortingInterface(mountain_sort_output_file_path, sampling_frequency=sampling_frequency)
    sorting = interface.sorting_extractor
    unit_ids = sorting.unit_ids
    log_and_print(logger, f"Loaded MountainSort sorting with {len(unit_ids)} units at {sampling_frequency} Hz",
                  level="info")

    # Align spike times for each unit to the nwb's ground truth clock
    # (make bonsai start time 0 and align to photometry clock if photometry exists)
    aligned_spike_trains = aligned_spike_trains_by_unit(sorting, sampling_frequency, metadata, logger)

    create_units_table(nwbfile, sampling_frequency, "Units spike sorted with MountainSort", logger)
    nwbfile.add_unit_column(name="unit_name", description="The MountainSort unit id")
    for unit_id, spike_times in zip(unit_ids, aligned_spike_trains):
        nwbfile.add_unit(spike_times=spike_times, unit_name=str(unit_id))

    log_and_print(logger, f"Added {len(unit_ids)} units to the NWB Units table.", level="info")


def verify_analyzer_phy_correspondence(sorting, analyzer_path: Path, phy_group_by_unit: list, logger) -> None:
    """
    Sanity-check (logged at DEBUG) that the analyzer's units still correspond to the Phy curation the
    phy_group labels come from. Called on every Kilosort/BombCell conversion.

    We label each unit's phy_group by looking up its `original_cluster_id` in cluster_group.tsv. That
    is only meaningful if the analyzer's units are actually the same clusters as the Phy output. Phy
    merges/splits done AFTER the analyzer was built can break this: a merged cluster gets a new id and
    its old id disappears from the curation, so a unit still carrying that old original_cluster_id no
    longer matches any Phy cluster (it gets 'unknown'). This logs that.

    Two levels of check:
      1. Always: how many units matched a Phy group vs got 'unknown' (uses phy_group_by_unit only).
      2. If the Phy per-spike files (spike_times.npy + spike_clusters.npy) are present next to the
         analyzer: verify each matched unit's spike train is byte-for-byte identical to the Phy cluster
         its original_cluster_id points to. Skipped (DEBUG note) if those files aren't there, e.g. a
         trimmed upload that only kept analyzer.zarr + cluster_group.tsv.

    A genuine spike-train mismatch (a matched id whose spikes differ) is logged at WARNING, since that
    means the labels are actively wrong rather than just stale.
    """
    num_units = len(sorting.unit_ids)
    n_unknown = sum(1 for group in phy_group_by_unit if group == "unknown")
    logger.debug(f"phy_group correspondence: {num_units - n_unknown}/{num_units} units matched a Phy cluster "
                 f"group, {n_unknown} got 'unknown'.")
    if n_unknown:
        logger.debug(f"{n_unknown}/{num_units} units have an original_cluster_id not in cluster_group.tsv. "
                     "(They could not be matched to a Phy curation group and were set to phy_group='unknown'. "
                     "This is expected from manual curation if you did splits/merges in Phy after building "
                     "the analyzer.zarr .")

    # Deeper spike-train check needs the Phy per-spike files. Skip if they aren't present.
    spike_times_path = analyzer_path.parent / "spike_times.npy"
    spike_clusters_path = analyzer_path.parent / "spike_clusters.npy"
    logger.debug(f"Looking for Phy per-spike files next to the analyzer: {spike_times_path} "
                 f"(exists={spike_times_path.exists()}), {spike_clusters_path} (exists={spike_clusters_path.exists()})")
    if not (spike_times_path.exists() and spike_clusters_path.exists()):
        logger.debug("spike_times.npy / spike_clusters.npy not found next to the analyzer; skipping the "
                     "spike-train correspondence check.")
        return

    spike_frames = np.load(spike_times_path).ravel()
    spike_clusters = np.load(spike_clusters_path).ravel()
    logger.debug(f"Loaded spike_times.npy {spike_frames.shape} + spike_clusters.npy "
                 f"{spike_clusters.shape}; {len(np.unique(spike_clusters))} distinct Phy clusters")

    # Group the Phy spike frames by cluster id with a single stable sort (preserves time order within
    # each cluster), then slice out each cluster's frames.
    phy_order = np.argsort(spike_clusters, kind="stable")
    phy_sorted_clusters = spike_clusters[phy_order]
    phy_sorted_frames = spike_frames[phy_order]
    unique_cluster_ids, cluster_starts = np.unique(phy_sorted_clusters, return_index=True)
    cluster_ends = np.append(cluster_starts[1:], len(phy_sorted_clusters))
    phy_frames_by_cluster = {int(cid): phy_sorted_frames[start:end]
                             for cid, start, end in zip(unique_cluster_ids, cluster_starts, cluster_ends)}

    # Group the analyzer's own spike frames by unit the same way. to_spike_vector() is cached from the
    # earlier alignment step, so this is cheap (avoids 100s of per-unit get_unit_spike_train calls).
    spike_vector = sorting.to_spike_vector()
    unit_order = np.argsort(spike_vector["unit_index"], kind="stable")
    sorted_unit_index = spike_vector["unit_index"][unit_order]
    sorted_unit_frames = spike_vector["sample_index"][unit_order]
    unit_boundaries = np.searchsorted(sorted_unit_index, np.arange(num_units + 1))

    original_cluster_id = np.asarray(sorting.get_property("original_cluster_id"))
    n_exact_match = n_mismatch = n_missing = 0
    for i, cluster_id in enumerate(original_cluster_id):
        phy_frames = phy_frames_by_cluster.get(int(cluster_id))
        if phy_frames is None:
            n_missing += 1  # this unit's cluster isn't in the Phy output (the 'unknown' units)
            continue
        unit_frames = sorted_unit_frames[unit_boundaries[i]:unit_boundaries[i + 1]]
        if len(unit_frames) == len(phy_frames) and np.array_equal(np.sort(unit_frames), np.sort(phy_frames)):
            n_exact_match += 1
        else:
            n_mismatch += 1

    logger.debug(f"phy spike-train check ({num_units} units): {n_exact_match} exact spike-train matches, "
                 f"{n_mismatch} mismatched, {n_missing} clusters not in the Phy output.")
    if n_mismatch:
        logger.warning(f"{n_mismatch} unit(s) matched a Phy cluster id but have DIFFERENT spikes than that "
                       "cluster - the analyzer's original_cluster_id does not line up with cluster_group.tsv, "
                       "so phy_group labels may be wrong. Check that the analyzer was built from this Phy output.")


def electrode_rows_by_channel(nwbfile, probe_label: str, logger) -> dict:
    """
    This probe's channel number -> the row it sits on in the nwb electrodes table.

    Every Neuropixels probe numbers its channels CH0..CH383, so a channel number alone does not
    identify an electrode in a file with two probes. The electrodes table carries the probe each
    row belongs to, so the row is looked up by probe AND channel. Call this after the raw ephys
    conversion has written that table; without it there is nothing to look up and it raises.
    """
    if nwbfile.electrodes is None:
        logger.error("No electrodes table to look up peak electrodes in.")
        raise ValueError("No electrodes table in the nwb, so a unit's peak electrode cannot be "
                         "located. Raw ephys must be converted before spike sorting.")

    electrodes = nwbfile.electrodes.to_dataframe()
    # A single-probe file (Berke lab) has no 'probe' column, and every row is that one probe
    if "probe" in electrodes.columns:
        on_this_probe = electrodes["probe"] == probe_label
    else:
        on_this_probe = pd.Series(True, index=electrodes.index)

    # A row is identified by its POSITION in the table, which is what a reference into electrodes
    # uses. to_dataframe() indexes by the electrode id, so enumerate instead of using the index.
    rows_by_channel = {
        int(channel): position
        for position, (channel, is_this_probe)
        in enumerate(zip(electrodes["intan_channel_number"], on_this_probe))
        if is_this_probe
    }

    if not rows_by_channel:
        found = (sorted(set(electrodes["probe"])) if "probe" in electrodes.columns
                 else "no probe column")
        logger.error(f"No electrodes table rows for probe '{probe_label}'; the table has {found}")
        raise ValueError(f"No electrodes in this nwb belong to probe '{probe_label}', so its units "
                         "have no electrode to point at. Check that the probe labels in the "
                         "metadata match the ones the raw ephys conversion wrote.")
    logger.debug(f"'{probe_label}' has {len(rows_by_channel)} electrodes table rows "
                 f"(channels {min(rows_by_channel)}..{max(rows_by_channel)} -> rows "
                 f"{min(rows_by_channel.values())}..{max(rows_by_channel.values())})")
    return rows_by_channel


def add_kilosort_bombcell_spikes(nwbfile: NWBFile, metadata: dict, probe: dict,
                                 probe_index: int, logger):
    """
    Add Kilosort4 + BombCell spike sorting output to the NWB file as a Units table.

    Expected input (how this data is produced in our pipeline):
      We spike sort with Kilosort4, classify units automatically with BombCell, and (optionally) curate
      manually in Phy. The result is saved as a SpikeInterface SortingAnalyzer directory (analyzer.zarr),
      pointed to by metadata["ephys"]["sorting_analyzer_path"]. From it we read:
        - the sorting: unit ids, spike trains (sample indices into the raw recording), and per-unit
          properties - 'bc_unitType' (BombCell class), 'KSLabel' (Kilosort auto label), 'original_cluster_id'
          (the source Phy/Kilosort cluster id), plus the numeric cluster_info metrics BombCell computed.
        - the 'quality_metrics' extension (SpikeInterface per-unit metrics).
        - the 'templates' extension (per-unit average waveform) -> peak-channel waveform + peak_electrode.
      Phy's MANUAL curation labels are NOT stored in the analyzer. They live in a 'cluster_group.tsv'
      file sitting next to analyzer.zarr, mapped to units by original_cluster_id -> 'phy_group'.

    We write ALL units (not just 'good' ones), carrying the labels as columns so units can be filtered
    downstream while keeping the NWB self-describing. Every piece of enrichment above is OPTIONAL: if the
    analyzer/pipeline did not produce it, we log a warning and skip that column rather than failing - the
    only thing always written is spike_times (+ the NWB unit id).

    Spike times are sample indices relative to the start of the raw recording. We convert them to
    seconds and shift by the bonsai start time (so bonsai start = time 0, matching the raw ephys
    ElectricalSeries), then align to the ground truth clock (photometry, if present) exactly as the
    raw ephys timestamps are aligned. See aligned_spike_trains_by_unit.
    """
    import spikeinterface as si  # local import; heavy dependency only needed when spikes are present
    from spikeinterface.core import get_template_extremum_channel

    # The label and recording length come from the raw ephys conversion, matched by this probe's
    # position in the metadata 'probes' list, since a probe may be named by any of the forms its
    # Open Ephys stream goes by. Raw ephys is always converted first - it is the only source of the
    # channel map, the port visits and the probe start times - so these are always here.
    probe_results = (metadata.get("ephys_probe_results") or [])
    if probe_index >= len(probe_results):
        logger.error(f"No raw ephys result for probe {probe_index}; got {len(probe_results)}")
        raise ValueError(f"No raw ephys was converted for probe {probe_index + 1} of the metadata "
                         "'probes' list, so there is nothing to tell which probe these units are "
                         "from or to check the sorting against. Convert the raw ephys too.")
    probe_label = probe_results[probe_index]["label"]
    recording_num_samples = probe_results[probe_index]["num_samples"]
    log_and_print(logger, f"Adding Kilosort4 spikes for '{probe_label}'...", level="info")
    analyzer_path = Path(probe["sorting_analyzer_path"])
    logger.debug(f"Found Kilosort/BombCell spikes from SortingAnalyzer at {analyzer_path}")

    analyzer = si.load_sorting_analyzer(analyzer_path)
    sorting = analyzer.sorting
    fs = analyzer.sampling_frequency
    unit_ids = sorting.unit_ids
    num_units = len(unit_ids)
    log_and_print(logger, f"Loaded SortingAnalyzer with {num_units} units at {fs} Hz", level="info")

    # Log what we found in the analyzer
    logger.debug(f"unit_ids dtype={unit_ids.dtype}, unit id range {unit_ids.min()}..{unit_ids.max()}")
    logger.debug(f"{len(analyzer.channel_ids)} channels | channel_ids={[int(c) for c in analyzer.channel_ids]}")
    saved_extensions = analyzer.get_saved_extension_names()
    logger.debug(f"Saved extensions ({len(saved_extensions)}): {saved_extensions}")
    logger.debug(f"Sorting property keys ({len(sorting.get_property_keys())}): "
                 f"{list(sorting.get_property_keys())}")

    # Verify the sorting was done on this session's full recording before we do anything with it.
    # The analyzer retains its recording's total sample count (even though it is recordingless),
    # which must match the raw ephys we converted. If they differ, the spike sample indices index a
    # different (or cropped) recording, so aligning them to this session's clock would be wrong
    # with nothing to show it.
    analyzer_num_samples = analyzer.get_num_samples()
    logger.debug(f"analyzer={analyzer_num_samples} samples vs full raw ephys={recording_num_samples} samples "
                 f"(diff={analyzer_num_samples - recording_num_samples})")
    if analyzer_num_samples != recording_num_samples:
        raise ValueError(
            "Spike sorting does not match the full raw ephys recording for this session! The SortingAnalyzer "
            f"at {analyzer_path} was computed on a recording of {analyzer_num_samples} samples "
            f"({analyzer_num_samples / fs:.1f}s), but this session's raw ephys recording has "
            f"{recording_num_samples} samples ({recording_num_samples / fs:.1f}s). The spike times would "
            f"be aligned to the wrong recording. Check that the 'sorting_analyzer_path' for "
            f"'{probe_label}' points to the sorting for this session's recording."
        )
    log_and_print(logger, f"Verified spike sorting matches this session's recording "
                  f"({recording_num_samples} samples).", level="info")

    # Align spike times for each unit to the nwb's ground truth clock
    # (make bonsai start time 0 and align to photometry clock if photometry exists)
    aligned_spike_trains = aligned_spike_trains_by_unit(sorting, fs, metadata, logger,
                                                       probe_label=probe_label)

    # Spike-train stats after alignment
    spike_counts = [len(train) for train in aligned_spike_trains]
    total_spikes = sum(spike_counts)
    logger.debug(f"Found {total_spikes} total spikes across {num_units} units")
    logger.debug("Per-unit spike count min/median/max = "
                 f"{min(spike_counts)}/{int(np.median(spike_counts))}/{max(spike_counts)}")
    nonempty = [train for train in aligned_spike_trains if len(train)]
    if nonempty:
        logger.debug(f"Aligned spike time range across units: "
                     f"[{min(train[0] for train in nonempty):.3f}, {max(train[-1] for train in nonempty):.3f}] s")

    # Gather the per-unit columns to attach. Every piece is OPTIONAL: if the analyzer/pipeline did not
    # produce it, we warn and skip that column rather than failing the whole conversion. The only thing
    # we always write is spike_times (+ the NWB unit id). Each entry is column_name -> (description,
    # values) where values is a sequence in unit_ids order.
    unit_columns = {}
    property_keys = set(sorting.get_property_keys())

    # Curation labels / source cluster id from the sorting properties (each optional)
    if "original_cluster_id" in property_keys:
        original_cluster_id = np.asarray(sorting.get_property("original_cluster_id"))
        unit_columns["original_cluster_id"] = ("Original Kilosort/Phy cluster id this unit corresponds to",
                                               [int(x) for x in original_cluster_id])
    else:
        original_cluster_id = None
        logger.warning("Analyzer sorting has no 'original_cluster_id' property; skipping that column and the "
                       "'phy_group' labels (phy_group is mapped to units via original_cluster_id).")
    if "bc_unitType" in property_keys:
        unit_columns["bc_unitType"] = ("BombCell unit classification (GOOD, MUA, NON-SOMA, NOISE)",
                                       [str(x) for x in sorting.get_property("bc_unitType")])
        logger.debug(f"bc_unitType distribution: {pd.Series(unit_columns['bc_unitType'][1]).value_counts().to_dict()}")
    else:
        logger.warning("Analyzer sorting has no 'bc_unitType' property; skipping the BombCell classification "
                       "column. (Was BombCell run and attached to this analyzer?)")
    if "KSLabel" in property_keys:
        unit_columns["ks_label"] = ("Kilosort automated label (good or mua)",
                                    [str(x) for x in sorting.get_property("KSLabel")])
        logger.debug(f"ks_label distribution: {pd.Series(unit_columns['ks_label'][1]).value_counts().to_dict()}")
    else:
        logger.warning("Analyzer sorting has no 'KSLabel' property; skipping the Kilosort label column.")

    # Phy MANUAL curation 'group' lives in cluster_group.tsv next to the analyzer (not in the analyzer
    # itself). Map it via original_cluster_id; units without a match (e.g. Phy re-merged clusters) get
    # 'unknown'. Needs original_cluster_id, so skip the whole phy_group column if that property is absent.
    if original_cluster_id is not None:
        phy_group_by_unit = ["unknown"] * num_units
        phy_group_tsv = analyzer_path.parent / "cluster_group.tsv"
        logger.debug(f"Looking for Phy manual curation at {phy_group_tsv} (exists={phy_group_tsv.exists()})")
        if phy_group_tsv.exists():
            group_df = pd.read_csv(phy_group_tsv, sep="\t")
            group_lookup = dict(zip(group_df["cluster_id"], group_df["group"]))
            logger.debug(f"cluster_group.tsv: {len(group_df)} rows | cluster_id range "
                         f"{group_df['cluster_id'].min()}..{group_df['cluster_id'].max()} | "
                         f"group value counts {group_df['group'].value_counts().to_dict()}")
            phy_group_by_unit = [str(group_lookup.get(int(cid), "unknown")) for cid in original_cluster_id]
            logger.debug(f"Phy group mapped onto units: {pd.Series(phy_group_by_unit).value_counts().to_dict()}")
        else:
            log_and_print(logger, f"No cluster_group.tsv found at {phy_group_tsv}; 'phy_group' set to 'unknown'.",
                          level="warning")
        unit_columns["phy_group"] = ("Phy manual curation group (good, mua, noise, or unknown if unlabeled)",
                                     phy_group_by_unit)
        # Sanity-check (logs at DEBUG) that the analyzer's units actually correspond to the Phy curation
        # we just labeled them from - catches an analyzer out of sync with the final Phy curation.
        verify_analyzer_phy_correspondence(sorting, analyzer_path, phy_group_by_unit, logger)

    # SpikeInterface quality metrics (optional 'quality_metrics' extension). Coerce to float so missing
    # values become np.nan (NWB stores floats; some metrics are NA for some units), and reindex to
    # unit_ids order (its index is assumed to be the unit ids).
    if analyzer.has_extension("quality_metrics"):
        quality_metrics = analyzer.get_extension("quality_metrics").get_data()
        quality_metrics = quality_metrics.apply(pd.to_numeric, errors="coerce").astype("float64").reindex(unit_ids)
        logger.debug(f"Found {len(quality_metrics.columns)} SpikeInterface metrics: "
                     f"{list(quality_metrics.columns)}")
        for metric in quality_metrics.columns:
            unit_columns[metric] = (f"SpikeInterface quality metric: {metric}", quality_metrics[metric].to_numpy())
    else:
        logger.warning("Analyzer has no 'quality_metrics' extension; skipping SpikeInterface quality metric "
                       "columns. (Was compute('quality_metrics') run on this analyzer?)")

    # BombCell/Kilosort per-unit metrics carried from the remaining (numeric) sorting properties (nPeaks,
    # waveformDuration_peakTrough, signalToNoiseRatio, presenceRatio, etc.). These complement the
    # SpikeInterface quality_metrics. Skip the label/id properties handled above (KSLabel_repeat is a
    # duplicate of ks_label). Iterate the ordered property list (not the set) for deterministic column order.
    handled_props = {"bc_unitType", "KSLabel", "KSLabel_repeat", "original_cluster_id"}
    cluster_info_added, cluster_info_skipped = [], []
    for prop in sorting.get_property_keys():
        if prop in handled_props:
            continue
        # We only turn 1-D, scalar-per-unit properties into metric columns. Skip anything multi-dimensional
        # (e.g. a (num_units, 3) unit_locations property), which isn't a single per-unit metric value.
        prop_values = np.asarray(sorting.get_property(prop))
        if prop_values.ndim != 1:
            logger.debug(f"Skipping property '{prop}' as a metric column: not scalar-per-unit "
                         f"(shape {prop_values.shape}).")
            cluster_info_skipped.append(prop)
            continue
        # Some cluster_info properties are text, not numbers ('quality' holds "good" / "mua").
        # Coercing those to a number makes the whole column NaN and loses the labels, so a
        # property whose every value is non-numeric is written as text instead.
        prop_series = pd.Series(prop_values)
        numeric = pd.to_numeric(prop_series, errors="coerce")
        not_a_number = int((numeric.isna() & prop_series.notna()).sum())
        if not_a_number == len(prop_series) and len(prop_series):
            values = prop_series.astype(str).to_numpy()
            unit_columns[prop] = (f"BombCell/Kilosort per-unit label (from cluster_info): {prop}", values)
        else:
            if not_a_number:
                logger.warning(f"{not_a_number} of {len(prop_series)} values of '{prop}' are not "
                               "numbers; they are NaN in that column.")
            values = numeric.to_numpy(dtype="float64")
            unit_columns[prop] = (f"BombCell/Kilosort per-unit metric (from cluster_info): {prop}", values)
        cluster_info_added.append(prop)
    logger.debug(f"Added {len(cluster_info_added)} BombCell/Kilosort metric columns: {cluster_info_added}")
    if cluster_info_skipped:
        logger.debug(f"Skipped {len(cluster_info_skipped)} non-scalar properties: {cluster_info_skipped}")

    # Peak-channel waveform + the electrode it was on (optional 'templates' extension)
    if analyzer.has_extension("templates"):
        templates = analyzer.get_extension("templates").get_data()  # (num_units, num_samples, num_channels)
        peak_channel_index = get_template_extremum_channel(analyzer, outputs="index")  # unit_id -> channel index
        channel_ids = analyzer.channel_ids
        electrode_rows = electrode_rows_by_channel(nwbfile, probe_label, logger)
        # channel_ids[index] is the channel NUMBER; the index is its position in the analyzer,
        # which is not the same thing because Kilosort drops bad channels from the sorting.
        peak_channels = [int(channel_ids[peak_channel_index[uid]]) for uid in unit_ids]
        peak_electrode_rows = [electrode_rows[channel] for channel in peak_channels]
        unit_columns["peak_electrode"] = (
            "Row in this nwb's electrodes table for the electrode with the largest-amplitude "
            "template for this unit. Found by this unit's probe and peak channel number, so it "
            "points at that probe's own electrodes.",
            peak_electrode_rows,
        )
        logger.debug(f"Templates extension shape {templates.shape} (units, samples, channels); peak channel "
                     f"numbers {min(peak_channels)}..{max(peak_channels)} across "
                     f"{len(set(peak_channels))} distinct channels -> electrode rows "
                     f"{min(peak_electrode_rows)}..{max(peak_electrode_rows)}")
    else:
        templates = None
        logger.warning("Analyzer has no 'templates' extension; skipping 'waveform_mean' and 'peak_electrode'. "
                       "(Was compute('templates') run on this analyzer?)")

    # Which probe each unit came from, and its id in that probe's own sorting. The nwb assigns the
    # row ids: both probes have Kilosort cluster ids starting at 0, so the sorter's id cannot be the
    # row id without colliding.
    unit_columns["probe"] = ("The probe this unit was sorted from", [probe_label] * num_units)
    unit_columns["sorter_unit_id"] = ("This unit's id in its own probe's sorting (Kilosort cluster id)",
                                      [int(unit_id) for unit_id in unit_ids])

    # Register all the present columns, then add each unit
    waveform_column_names = ["waveform_mean", "electrodes", "electrode_group"] if templates is not None else []
    all_column_names = list(unit_columns.keys()) + waveform_column_names
    logger.debug(f"Writing {len(all_column_names)} metadata columns (+ spike_times) to nwb units table: "
                 f"{all_column_names}")
    create_units_table(nwbfile, fs, "Units spike sorted with Kilosort4 and classified by BombCell", logger)
    check_unit_columns_match(nwbfile, probe_label, all_column_names, logger)
    # This runs once per probe, so only declare each column the first time
    existing_unit_columns = set(nwbfile.units.colnames)
    for name, (description, _values) in unit_columns.items():
        if name not in existing_unit_columns:
            nwbfile.add_unit_column(name=name, description=description)

    for i, unit_id in enumerate(unit_ids):
        add_unit_kwargs = {name: values[i] for name, (_description, values) in unit_columns.items()}
        if templates is not None:
            # Waveform mean on the unit's peak channel (1D over samples)
            peak_idx = peak_channel_index[unit_id]
            add_unit_kwargs["waveform_mean"] = templates[i, :, peak_idx].astype("float64")
            # The peak electrode again, in the form NWB defines for it: 'electrodes' is a real
            # reference into the electrodes table and 'electrode_group' is that electrode's shank.
            # This is also what says which electrode waveform_mean was measured on.
            add_unit_kwargs["electrodes"] = [peak_electrode_rows[i]]
            add_unit_kwargs["electrode_group"] = nwbfile.electrodes["group"][peak_electrode_rows[i]]
        nwbfile.add_unit(spike_times=aligned_spike_trains[i], **add_unit_kwargs)

    log_and_print(logger, f"Added {num_units} units to the NWB Units table "
                  f"({len(all_column_names)} metadata columns + spike_times).", level="info")
