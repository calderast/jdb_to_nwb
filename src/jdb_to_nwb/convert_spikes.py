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


def add_kilosort_bombcell_spikes(nwbfile: NWBFile, metadata: dict, logger):
    """
    Add Kilosort4 + BombCell spike sorting output to the NWB file as a Units table.

    Reads a SpikeInterface SortingAnalyzer (analyzer.zarr) pointed to by
    metadata["ephys"]["sorting_analyzer_path"]. The analyzer carries the sorting plus per-unit
    quality metrics, templates, and curation labels (BombCell 'bc_unitType', Kilosort 'KSLabel').
    We write ALL units (not just 'good' ones), carrying the curation labels as columns so units can
    be filtered downstream while keeping the NWB self-describing.

    Spike times are sample indices relative to the start of the raw recording. We convert them to
    seconds and shift by the bonsai start time (so bonsai start = time 0, matching the raw ephys
    ElectricalSeries), then align to the ground truth clock (photometry, if present) exactly as the
    raw ephys timestamps are aligned.
    """
    import spikeinterface as si  # local import; heavy dependency only needed when spikes are present
    from spikeinterface.core import get_template_extremum_channel

    log_and_print(logger, "Adding Kilosort4 spikes...", level="info")
    analyzer_path = Path(metadata["ephys"]["sorting_analyzer_path"])
    logger.info(f"Found Kilosort/BombCell spikes from SortingAnalyzer at {analyzer_path}")

    analyzer = si.load_sorting_analyzer(analyzer_path)
    sorting = analyzer.sorting
    fs = analyzer.sampling_frequency
    unit_ids = sorting.unit_ids
    num_units = len(unit_ids)
    log_and_print(logger, f"Loaded SortingAnalyzer with {num_units} units at {fs} Hz", level="info")

    # Align spike times for each unit to the nwb's ground truth clock
    # (make bonsai start time 0 and align to photometry clock if photometry exists)
    aligned_spike_trains = aligned_spike_trains_by_unit(sorting, fs, metadata, logger)

    # Gather per-unit metadata to attach as Units table columns
    quality_metrics = analyzer.get_extension("quality_metrics").get_data()  # DataFrame indexed by unit_id
    # Coerce to float so missing values become np.nan (NWB stores floats; some metrics are NA for some units),
    # then reindex to unit_ids order and pull out numpy columns (much faster than per-unit .loc lookups)
    quality_metrics = quality_metrics.apply(pd.to_numeric, errors="coerce").astype("float64").reindex(unit_ids)
    quality_metric_arrays = {metric: quality_metrics[metric].to_numpy() for metric in quality_metrics.columns}
    templates = analyzer.get_extension("templates").get_data()  # (num_units, num_samples, num_channels)
    peak_channel_index = get_template_extremum_channel(analyzer, outputs="index")  # unit_id -> channel index
    channel_ids = analyzer.channel_ids

    # Curation labels straight from the analyzer's sorting properties (complete for all units)
    bc_unit_type = sorting.get_property("bc_unitType")
    ks_label = sorting.get_property("KSLabel")
    original_cluster_id = np.asarray(sorting.get_property("original_cluster_id"))

    # Phy manual curation 'group' lives in cluster_group.tsv next to the analyzer's Phy output.
    # Map it via original_cluster_id; units without a match (e.g. Phy re-merged clusters) get 'unknown'.
    phy_group_by_unit = ["unknown"] * num_units
    phy_group_tsv = analyzer_path.parent / "cluster_group.tsv"
    if phy_group_tsv.exists():
        group_df = pd.read_csv(phy_group_tsv, sep="\t")
        group_lookup = dict(zip(group_df["cluster_id"], group_df["group"]))
        phy_group_by_unit = [str(group_lookup.get(int(cid), "unknown")) for cid in original_cluster_id]
    else:
        log_and_print(logger, f"No cluster_group.tsv found at {phy_group_tsv}; 'phy_group' set to 'unknown'.",
                      level="warning")

    # BombCell/Kilosort per-unit metrics from cluster_info (the metrics BombCell used to classify units:
    # nPeaks, waveformDuration_peakTrough, signalToNoiseRatio, presenceRatio, etc.). These are carried
    # straight from the analyzer's sorting properties and complement the SpikeInterface quality_metrics.
    # Skip the label/id properties we handle explicitly above (KSLabel_repeat is a duplicate of ks_label).
    handled_props = {"bc_unitType", "KSLabel", "KSLabel_repeat", "original_cluster_id"}
    cluster_info_metric_arrays = {
        prop: pd.to_numeric(pd.Series(sorting.get_property(prop)), errors="coerce").to_numpy(dtype="float64")
        for prop in sorting.get_property_keys() if prop not in handled_props
    }

    # Define the Units table columns
    nwbfile.add_unit_column(name="original_cluster_id",
                            description="Original Kilosort/Phy cluster id this unit corresponds to")
    nwbfile.add_unit_column(name="bc_unitType",
                            description="BombCell unit classification (GOOD, MUA, NON-SOMA, NOISE)")
    nwbfile.add_unit_column(name="ks_label",
                            description="Kilosort automated label (good or mua)")
    nwbfile.add_unit_column(name="phy_group",
                            description="Phy manual curation group (good, mua, noise, or unknown if unlabeled)")
    nwbfile.add_unit_column(name="peak_channel_id",
                            description="Recording channel id with the largest-amplitude template for this unit")
    # Add columns for SpikeInterface quality metrics
    quality_metric_columns = list(quality_metric_arrays.keys())
    for metric in quality_metric_columns:
        nwbfile.add_unit_column(name=metric, 
                                description=f"SpikeInterface quality metric: {metric}")
    # Add columns for BombCell/Kilosort metrics
    cluster_info_metric_columns = list(cluster_info_metric_arrays.keys())
    for metric in cluster_info_metric_columns:
        nwbfile.add_unit_column(name=metric, 
                                description=f"BombCell/Kilosort per-unit metric (from cluster_info): {metric}")

    # Add each unit
    for i, unit_id in enumerate(unit_ids):
        spike_times = aligned_spike_trains[i]

        # Waveform mean on the unit's peak channel (1D over samples)
        peak_idx = peak_channel_index[unit_id]
        waveform_mean = templates[i, :, peak_idx].astype("float64")

        per_unit_metric_values = {
            metric: float(quality_metric_arrays[metric][i]) for metric in quality_metric_columns
        }
        per_unit_metric_values.update(
            {metric: float(cluster_info_metric_arrays[metric][i]) for metric in cluster_info_metric_columns}
        )

        nwbfile.add_unit(
            id=int(unit_id),
            spike_times=spike_times,
            waveform_mean=waveform_mean,
            original_cluster_id=int(original_cluster_id[i]),
            bc_unitType=str(bc_unit_type[i]),
            ks_label=str(ks_label[i]),
            phy_group=phy_group_by_unit[i],
            peak_channel_id=str(channel_ids[peak_idx]),
            **per_unit_metric_values,
        )

    log_and_print(logger, f"Added {num_units} units to the NWB Units table.", level="info")
