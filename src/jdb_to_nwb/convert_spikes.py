import numpy as np
import pandas as pd
from pathlib import Path

from pynwb import NWBFile

from .mdasortinginterface import MdaSortingInterface
from .utils import log_and_print
from .timestamps_alignment import align_via_interpolation


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

    ephys = metadata["ephys"]

    if "sorting_analyzer_path" in ephys:
        add_kilosort_bombcell_spikes(nwbfile=nwbfile, metadata=metadata, logger=logger)
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
        ))

    # Ground truth is ephys (no photometry): spikes are already on the ground truth clock
    logger.info(f"Ground truth is '{ground_truth_time_source}' (not photometry): spikes are sorted "
                  "from the ephys recording, so they are already on the ground truth clock. No interpolation "
                  "needed.")
    return spike_times_s


def aligned_spike_trains_by_unit(sorting, sampling_frequency: float, metadata: dict, logger) -> list:
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
    # STEP 1: shift so bonsai start is time 0
    # bonsai_start_time (seconds after ephys start that bonsai started) comes from add_raw_ephys, which
    # sets it to time 0 for the raw ephys. If there was no raw ephys this session it is None, so we can't
    # shift - warn and leave spike times relative to the raw recording start (bonsai_start_time = 0).
    # But that should never really happen (if we have spikes we should also have raw ephys). 
    bonsai_start_time = metadata.get("ephys_bonsai_start_time")
    if bonsai_start_time is None:
        logger.warning("No 'ephys_bonsai_start_time' (no raw ephys this session??): leaving spike times "
                      "relative to the raw recording start, NOT shifted to bonsai start.")
        bonsai_start_time = 0.0

    # to_spike_vector() returns every spike as ('sample_index', 'unit_index'), sorted by time
    spike_vector = sorting.to_spike_vector()
    logger.debug(f"Aligning: {len(spike_vector)} spikes | raw sample_index range "
                 f"[{int(spike_vector['sample_index'].min())}, {int(spike_vector['sample_index'].max())}] "
                 f"| bonsai shift = -{bonsai_start_time}s @ {sampling_frequency} Hz")
    # Convert all spikes to seconds and apply the shift so bonsai start is time 0
    all_spike_times_s = spike_vector["sample_index"] / sampling_frequency - bonsai_start_time

    # STEP 2: ground truth alignment to account for drift between clocks (applied to the whole array at once)
    all_spike_times_s = align_spike_times(all_spike_times_s, metadata, logger)

    # Group the flat, aligned spike-time array back into one train per unit. Sorting by unit index with a
    # STABLE sort preserves each unit's time ordering; searchsorted then gives each unit's slice boundaries.
    all_unit_index = spike_vector["unit_index"]
    num_units = len(sorting.unit_ids)
    order = np.argsort(all_unit_index, kind="stable")
    sorted_unit_index = all_unit_index[order]
    sorted_spike_times = all_spike_times_s[order]
    unit_boundaries = np.searchsorted(sorted_unit_index, np.arange(num_units + 1))
    return [sorted_spike_times[unit_boundaries[i]:unit_boundaries[i + 1]] for i in range(num_units)]


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


def add_kilosort_bombcell_spikes(nwbfile: NWBFile, metadata: dict, logger):
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
        - the 'templates' extension (per-unit average waveform) -> peak-channel waveform + peak_channel_id.
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

    log_and_print(logger, "Adding Kilosort4 spikes...", level="info")
    analyzer_path = Path(metadata["ephys"]["sorting_analyzer_path"])
    logger.debug(f"Found Kilosort/BombCell spikes from SortingAnalyzer at {analyzer_path}")

    analyzer = si.load_sorting_analyzer(analyzer_path)
    sorting = analyzer.sorting
    fs = analyzer.sampling_frequency
    unit_ids = sorting.unit_ids
    num_units = len(unit_ids)
    log_and_print(logger, f"Loaded SortingAnalyzer with {num_units} units at {fs} Hz", level="info")

    # Log what we found in the analyzer
    logger.debug(f"unit_ids dtype={unit_ids.dtype}, unit id range {unit_ids.min()}..{unit_ids.max()}")
    logger.debug(f"{len(analyzer.channel_ids)} channels | channel_ids[:8]={list(analyzer.channel_ids[:8])}"
                 f"{' ...' if len(analyzer.channel_ids) > 8 else ''}")
    saved_extensions = analyzer.get_saved_extension_names()
    logger.debug(f"Saved extensions ({len(saved_extensions)}): {saved_extensions}")
    logger.debug(f"Sorting property keys ({len(sorting.get_property_keys())}): "
                 f"{list(sorting.get_property_keys())}")

    # Verify the sorting was actually done on this session's full recording before we do anything with it.
    # The analyzer retains its recording's total sample count (even though it is recordingless), which
    # must match the raw ephys we converted for this session. If they differ, the spike sample indices
    # index a different (or cropped) recording, so aligning them to this session's clock would be silently wrong.
    # ephys_num_samples is set by add_raw_ephys; if it is absent (no raw ephys this session) 
    # we can't do this check, so we warn and continue.
    recording_num_samples = metadata.get("ephys_num_samples")
    if recording_num_samples is None:
        logger.warning("No 'ephys_num_samples' in metadata (raw ephys not converted this session?); cannot "
                       "verify the spike sorting was done on this session's full recording.")
    else:
        analyzer_num_samples = analyzer.get_num_samples()
        logger.debug(f"analyzer={analyzer_num_samples} samples vs full raw ephys={recording_num_samples} samples "
                     f"(diff={analyzer_num_samples - recording_num_samples})")
        if analyzer_num_samples != recording_num_samples:
            raise ValueError(
                "Spike sorting does not match the full raw ephys recording for this session! The SortingAnalyzer "
                f"at {analyzer_path} was computed on a recording of {analyzer_num_samples} samples "
                f"({analyzer_num_samples / fs:.1f}s), but this session's raw ephys recording has "
                f"{recording_num_samples} samples ({recording_num_samples / fs:.1f}s). The spike times would "
                "be aligned to the wrong recording. Check that 'sorting_analyzer_path' points to the sorting "
                "for this session's 'openephys_folder_path'."
            )
        log_and_print(logger, f"Verified spike sorting matches this session's recording "
                      f"({recording_num_samples} samples).", level="info")

    # Align spike times for each unit to the nwb's ground truth clock
    # (make bonsai start time 0 and align to photometry clock if photometry exists)
    aligned_spike_trains = aligned_spike_trains_by_unit(sorting, fs, metadata, logger)

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
        values = pd.to_numeric(pd.Series(prop_values), errors="coerce").to_numpy(dtype="float64")
        unit_columns[prop] = (f"BombCell/Kilosort per-unit metric (from cluster_info): {prop}", values)
        cluster_info_added.append(prop)
    logger.debug(f"Added {len(cluster_info_added)} BombCell/Kilosort metric columns: {cluster_info_added}")
    if cluster_info_skipped:
        logger.debug(f"Skipped {len(cluster_info_skipped)} non-scalar properties: {cluster_info_skipped}")

    # Peak-channel waveform + channel id (optional 'templates' extension)
    if analyzer.has_extension("templates"):
        templates = analyzer.get_extension("templates").get_data()  # (num_units, num_samples, num_channels)
        peak_channel_index = get_template_extremum_channel(analyzer, outputs="index")  # unit_id -> channel index
        channel_ids = analyzer.channel_ids
        unit_columns["peak_channel_id"] = ("Recording channel id with the largest-amplitude template for this unit",
                                           [str(channel_ids[peak_channel_index[uid]]) for uid in unit_ids])
        peak_idxs = [int(peak_channel_index[uid]) for uid in unit_ids]
        logger.debug(f"Templates extension shape {templates.shape} (units, samples, channels); peak channel "
                     f"index range {min(peak_idxs)}..{max(peak_idxs)} across {len(set(peak_idxs))} distinct channels")
    else:
        templates = None
        logger.warning("Analyzer has no 'templates' extension; skipping 'waveform_mean' and 'peak_channel_id'. "
                       "(Was compute('templates') run on this analyzer?)")

    # Register all the present columns, then add each unit
    all_column_names = list(unit_columns.keys()) + (["waveform_mean"] if templates is not None else [])
    logger.debug(f"Writing {len(all_column_names)} metadata columns (+ spike_times) to nwb units table: "
                 f"{all_column_names}")
    for name, (description, _values) in unit_columns.items():
        nwbfile.add_unit_column(name=name, description=description)

    for i, unit_id in enumerate(unit_ids):
        add_unit_kwargs = {name: values[i] for name, (_description, values) in unit_columns.items()}
        if templates is not None:
            # Waveform mean on the unit's peak channel (1D over samples)
            peak_idx = peak_channel_index[unit_id]
            add_unit_kwargs["waveform_mean"] = templates[i, :, peak_idx].astype("float64")
        nwbfile.add_unit(id=int(unit_id), spike_times=aligned_spike_trains[i], **add_unit_kwargs)

    log_and_print(logger, f"Added {num_units} units to the NWB Units table "
                  f"({len(all_column_names)} metadata columns + spike_times).", level="info")
