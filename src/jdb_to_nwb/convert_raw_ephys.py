# NOTE: We could have used NeuroConv to convert the ephys data but we want to use the Frank Lab's 
# ndx-franklab-novela extension to store the Probe information for maximal integration with Spyglass, 
# so we are doing the conversion manually using PyNWB.
#
# ============================================================================================
# ASSUMPTIONS
#
# These are the things this module takes to be true and does NOT verify, or verifies only
# indirectly. If a conversion produces something surprising, check these first.
#
# About the recordings
#   1. All probes being converted are the same model, and every recording holds every probe.
#      Checked: a mixed probe_type raises, and a missing stream in a later recording raises.
#   2. Probes share one hardware clock, so samples can be combined by index without resampling.
#      Measured on IM-1979 via laser artifacts seen by both probes: drift < 0.01 ppm over 457s.
#      Probes still START a few samples apart, which IS corrected (by time, per session).
#   3. Recordings are listed in the order they were recorded, and the gap between them is real
#      dead time. Checked: out-of-order folder times raise, as does an overlap.
#   4. A recording's wall-clock start is the datetime in its Open Ephys folder name, so the gap
#      between recordings is only accurate to ~1 second. Fine for sorting (the gap just has to
#      exist); NOT a sync, so cross-recording timing finer than a second is not supported.
#   5. Each probe's own timestamps are a usable clock within a recording. They are used as-is.
#   6. 'experiment_number' (when given) is the same for every recording.
#
# About preprocessed traces
#   7. When processed_ephys_path is set the ElectricalSeries holds preprocessed traces instead of
#      raw ones. That path and its own assumptions live in convert_processed_ephys.py.
#
# About the nwb and Spyglass
#  11. Spyglass reads raw ephys with nwbfile.get_acquisition(), which requires EXACTLY ONE
#      acquisition object - which is why every probe and every recording share one
#      ElectricalSeries rather than getting one each.
#  12. Spyglass turns each nwb epoch's FIRST tag into an interval list of that name, so the
#      'session' labels become the names a spike sort is restricted to.
#  13. Spyglass casts Shank and ShanksElectrode names to int, so those stay numeric. Electrode
#      GROUP names are varchar, so they can carry the probe prefix.
#  14. Spyglass keys Probe on probe_type, so two physical probes of the same model resolve to one
#      Spyglass Probe entry even though they are two nwb devices.
# ============================================================================================

import os
import re
import glob
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import xml.etree.ElementTree as ET
from importlib.resources import files
from hdmf.backends.hdf5 import H5DataIO

from neuroconv.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
    SpikeInterfaceRecordingDataChunkIterator,
)
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
# spikeinterface >= 0.103 exports only the read_* functions from spikeinterface.extractors and moved
# the extractor classes to spikeinterface.extractors.extractor_classes (the old location stops working
# entirely in 0.105). We need the class rather than the read_openephys function because we call
# OpenEphysBinaryRecordingExtractor.get_streams() to list streams before creating a recording.
try:
    from spikeinterface.extractors.extractor_classes import OpenEphysBinaryRecordingExtractor
except ImportError:  # spikeinterface < 0.103
    from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor

from .utils import log_and_print, add_associated_file
from .timestamps_alignment import align_via_interpolation
from .plotting.plot_ephys import plot_channel_map, plot_channel_impedances, plot_neuropixels
from ndx_franklab_novela import Probe, NwbElectrodeGroup, Shank, ShanksElectrode

VOLTS_PER_MICROVOLT = 1e-6

MIN_IMPEDANCE_OHMS = 1e5
MAX_IMPEDANCE_OHMS = 3e6

# Neuropixels port visits are recorded on the separate OneBox-ADC stream on ADC2 (in volts)
# (ADC0 carries a 1 Hz sync clock, ADC1 is used for future optotagging)
# Both the channel and the threshold are overridable via metadata["ephys"]
NPX_DEFAULT_PORT_VISITS_ADC_CHANNEL = 2
NPX_PORT_VISITS_THRESHOLD_VOLTS = 4.0

# When combining multiple probes into one recording, SpikeInterface requires every channel location
# to be unique, but all Neuropixels probes have sites at the same relative positions. We shift each
# probe's SpikeInterface geometry apart by this much (um) to keep the locations distinct. This never
# reaches the nwb: the rel_x/rel_y we write come from our own electrode coordinates table.
PROBE_LOCATION_OFFSET_UM = 10_000

# Get the location of the resources directory when the package is installed from pypi
__location_of_this_file = Path(files(__name__))
RESOURCES_DIR = __location_of_this_file / "resources" / "electrophysiology"

# If the resources directory does not exist, we are probably running the code from the source directory
if not RESOURCES_DIR.exists():
    RESOURCES_DIR = __location_of_this_file.parent.parent / "resources" / "electrophysiology"

BERKE_256CH_PROBE_CHANNEL_MAP_PATH = RESOURCES_DIR / "256ch_silicon_probe_channel_map.csv"
BERKE_252CH_PROBE_CHANNEL_MAP_PATH = RESOURCES_DIR  / "252ch_silicon_probe_channel_map.csv"
ELECTRODE_COORDS_PATH_256CH_3MM_PROBE = RESOURCES_DIR  / "256ch_probe_3mm_length_66um_pitch_coords.csv"
ELECTRODE_COORDS_PATH_256CH_6MM_PROBE = RESOURCES_DIR / "256ch_probe_6mm_length_80um_pitch_coords.csv"
ELECTRODE_COORDS_PATH_252CH_4MM_PROBE = RESOURCES_DIR / "252ch_probe_4mm_length_80um_pitch_coords.csv"
ELECTRODE_COORDS_PATH_252CH_10MM_PROBE = RESOURCES_DIR / "252ch_probe_10mm_length_100um_pitch_coords.csv"
ELECTRODE_COORDS_PATH_NPX_MULTISHANK = RESOURCES_DIR / "neuropixels_2.0_multishank_electrode_coords.csv"
DEVICES_PATH = RESOURCES_DIR / "ephys_devices.yaml"

# Names of custom Berke Lab probes
BERKE_LAB_PROBES = {
    "256-ch Silicon Probe, 3mm length, 66um pitch": {
        "channel_map": BERKE_256CH_PROBE_CHANNEL_MAP_PATH,
        "electrode_coords": ELECTRODE_COORDS_PATH_256CH_3MM_PROBE
    },
    "256-ch Silicon Probe, 6mm length, 80um pitch": {
        "channel_map": BERKE_256CH_PROBE_CHANNEL_MAP_PATH,
        "electrode_coords": ELECTRODE_COORDS_PATH_256CH_6MM_PROBE
    },
    "252-ch Silicon Probe, 4mm length, 80um pitch": {
        "channel_map": BERKE_252CH_PROBE_CHANNEL_MAP_PATH,
        "electrode_coords": ELECTRODE_COORDS_PATH_252CH_4MM_PROBE
    },
    "252-ch Silicon Probe, 10mm length, 100um pitch": {
        "channel_map": BERKE_252CH_PROBE_CHANNEL_MAP_PATH,
        "electrode_coords": ELECTRODE_COORDS_PATH_252CH_10MM_PROBE
    },
}

class MicrovoltsSpikeInterfaceRecordingDataChunkIterator(SpikeInterfaceRecordingDataChunkIterator):
    def __init__(self, iterator: SpikeInterfaceRecordingDataChunkIterator, conversion_factor_uv):
        self.iterator = iterator
        self.conversion_factor_uv = conversion_factor_uv
        super().__init__(recording=iterator.recording)

    def _get_default_chunk_shape(self, chunk_mb: float = 10.0) -> tuple[int, int]:
        return self.iterator._get_default_chunk_shape(chunk_mb)

    def _get_data(self, selection: tuple[slice]):
        data = self.iterator._get_data(selection)
        return (data * self.conversion_factor_uv).astype("int16")

    def _get_dtype(self):
        return np.dtype("int16")

    def _get_maxshape(self):
        return self.iterator._get_maxshape()


def find_open_ephys_paths(open_ephys_folder_path, experiment_number=None, logger=None) -> dict:
    """
    Given the Open Ephys folder path, find the relevant settings.xml file and all associated continuous.dat files.

    For custom Berke Lab probes, we expect a single probe folder with a single continuous.dat. For example:
    open_ephys_folder_path/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat

    For Neuropixels, we may have multiple "probe name" folders, each with a different continuous.dat file
    This will be the case for the ADC, and also if we implant 2 probes bilaterally. For example:
    open_ephys_folder_path/Record Node 101/experiment2/recording1/continuous/OneBox-100.OneBox-ADC/continuous.dat
    open_ephys_folder_path/Record Node 101/experiment2/recording1/continuous/OneBox-100.ProbeA/continuous.dat

    For both cases, the settings.xml file lives at the same level as the experiment{number} folder.
    The suffix for settings.xml matches the number of the experiment if >=2, or no suffix for experiment1
    Example paths:
    open_ephys_folder_path/settings.xml (Berke Lab probe)
    open_ephys_folder_path/Record Node 101/settings_2.xml (Neuropixels)

    Parameters:
        open_ephys_folder_path:
            Path to the Open Ephys output folder (the auto-generated one with the
            recording start time in the folder name)
        experiment_number (int):
            Optional. If multiple recordings in the same output folder (which get
            auto-named experiment1, experiment2, etc), which one to use. If None (the default),
            the experiment number is auto-detected from the settings*.xml file(s) present.
        logger (Logger):
            Optional. Logger to track conversion progress (used to report the auto-detected number)

    Returns:
        dict:
            "settings_file": path to settings.xml file
            "recording_files": dict with probe_name: path to the continuous.dat for that probe
    """
    # Auto-detect the experiment number if not specified
    if experiment_number is None:
        # Look at whatever settings*.xml files exist
        settings_candidates = glob.glob(os.path.join(open_ephys_folder_path, "**", "settings*.xml"), recursive=True)
        experiment_numbers = set()
        for candidate in settings_candidates:
            # experiment1 has settings.xml, experiment{N>1} has settings_N.xml
            match = re.fullmatch(r"settings(?:_(\d+))?\.xml", os.path.basename(candidate))
            if match:
                experiment_numbers.add(int(match.group(1)) if match.group(1) else 1)
        if not experiment_numbers:
            raise FileNotFoundError(
                f"No settings.xml found anywhere in {open_ephys_folder_path}"
            )
        if len(experiment_numbers) > 1:
            raise ValueError(
                f"Found multiple Open Ephys experiments ({sorted(experiment_numbers)}) in "
                f"{open_ephys_folder_path}. Specify which one to convert by setting 'experiment_number' "
                "in the ephys metadata."
            )
        experiment_number = experiment_numbers.pop()
        if logger is not None:
            logger.info(f"Auto-detected Open Ephys experiment number {experiment_number}")

    # Determine the suffix for settings.xml (empty for experiment1, "_2" for experiment2, etc.)
    suffix = "" if experiment_number == 1 else f"_{experiment_number}"

    # Look for the settings.xml file
    settings_file_pattern = os.path.join(open_ephys_folder_path, f"**/settings{suffix}.xml")
    settings_files = glob.glob(settings_file_pattern, recursive=True)

    # Make sure there is a single settings.xml file
    if not settings_files:
        raise FileNotFoundError(f"No settings.xml found for experiment {experiment_number} in {open_ephys_folder_path}")
    if len(settings_files) > 1:
        raise ValueError(
            f"Found {len(settings_files)} settings.xml files for experiment {experiment_number} in "
            f"{open_ephys_folder_path}, expected 1: {settings_files}"
            )
    settings_xml_file = settings_files[0]

    # Look for the continuous.dat files
    experiment_folder = f"experiment{experiment_number}"
    continuous_base = os.path.join(open_ephys_folder_path, "**", experiment_folder, "recording*", "continuous")
    probe_folders = glob.glob(os.path.join(continuous_base, "*"), recursive=True)

    # Inside each "probe name" folder, there should be a single continuous.dat
    continuous_dat_files = {}
    for probe_folder in probe_folders:
        if not os.path.isdir(probe_folder):
            continue
        probe_name = os.path.basename(probe_folder)
        dat_files = glob.glob(os.path.join(probe_folder, "continuous.dat"))
        if len(dat_files) == 0:
            print(f"No continuous.dat found in {probe_folder}!")
            continue
        if len(dat_files) > 1:
            raise ValueError(f"Multiple continuous.dat files found in {probe_folder}: {dat_files}")
        continuous_dat_files[probe_name] = dat_files[0]

    return {"settings_file": settings_xml_file, "recording_files": continuous_dat_files}


def read_oebin_params(continuous_dat_file_path: Path, logger) -> dict:
    """
    Read per-stream recording parameters from the structure.oebin file.

    The structure.oebin is a JSON file at the recording level (the same folder that contains the
    'continuous' folder). It describes every continuous stream (probe streams and the ADC stream),
    including the number of channels, sampling rate, and the per-channel bit_volts conversion factor.
    This works the same for Berke Lab probes and Neuropixels.

    Note: structure.oebin does NOT contain the bandpass filtering info (LowCut/HighCut) -
    as far as I know that lives only in settings.xml, so filtering info still comes from
    read_open_ephys_settings_xml.

    Parameters:
        continuous_dat_file_path (Path):
            Path to a 'continuous.dat' file. The structure.oebin is found relative to it
            (.../recording{n}/continuous/<stream>/continuous.dat -> .../recording{n}/structure.oebin)
        logger (Logger):
            Logger to track conversion progress

    Returns:
        dict:
            Params for the stream matching this continuous.dat:
            "channel_count" (int), "sample_rate" (float), "bit_volts" (float, the uV conversion
            factor, shared across channels), "channel_names" (list[str]), and "oebin_path" (Path to
            the structure.oebin file itself, so we can save it as an AssociatedFile)
    """
    continuous_dat_file_path = Path(continuous_dat_file_path)
    # .../recording{n}/continuous/<stream folder>/continuous.dat -> stream folder name and oebin path
    stream_folder_name = continuous_dat_file_path.parent.name
    oebin_path = continuous_dat_file_path.parent.parent.parent / "structure.oebin"
    if not oebin_path.exists():
        raise FileNotFoundError(f"Could not find structure.oebin at expected location {oebin_path}")

    with open(oebin_path, "r") as f:
        oebin = json.load(f)

    # Log a summary of every continuous stream in the oebin 
    logger.debug(f"structure.oebin at {oebin_path} (Open Ephys GUI version {oebin.get('GUI version', '?')})")
    for s in oebin["continuous"]:
        logger.debug(f"  continuous stream: folder_name={s['folder_name']!r} "
                     f"num_channels={s['num_channels']} sample_rate={s['sample_rate']}")

    # Find the continuous stream whose folder_name matches this continuous.dat's stream folder
    # (oebin folder_name has a trailing slash, e.g. "Rhythm_FPGA-100.0/", so strip it to compare)
    for stream in oebin["continuous"]:
        if stream["folder_name"].rstrip("/") == stream_folder_name:
            break
    else:
        raise ValueError(
            f"Could not find a continuous stream matching '{stream_folder_name}' in {oebin_path}. "
            f"Available streams: {[s['folder_name'] for s in oebin['continuous']]}"
        )

    # bit_volts (the uV conversion factor) should be the same for every channel in the stream
    bit_volts_per_channel = {ch["bit_volts"] for ch in stream["channels"]}
    if len(bit_volts_per_channel) != 1:
        logger.warning(f"Stream '{stream_folder_name}' has multiple bit_volts values: {bit_volts_per_channel}")
    bit_volts = float(stream["channels"][0]["bit_volts"])

    params = {
        "channel_count": int(stream["num_channels"]),
        "sample_rate": float(stream["sample_rate"]),
        "bit_volts": bit_volts,
        "channel_names": [ch["channel_name"] for ch in stream["channels"]],
        "oebin_path": oebin_path,  # so we can save structure.oebin as an AssociatedFile
    }
    logger.info(f"Read oebin params for stream '{stream_folder_name}': "
                f"{params['channel_count']} channels at {params['sample_rate']} Hz, bit_volts={bit_volts}")
    # List every channel name found in this stream
    logger.debug(f"structure.oebin channel names for '{stream_folder_name}': {params['channel_names']}")
    return params


def validate_oebin_channel_names(channel_names: list[str], expected_channel_names: set[str], logger) -> None:
    """
    Sanity-check the channel names read from structure.oebin against what we expect for this probe.

    We source this from structure.oebin (which lists each channel's name) rather than settings.xml,
    since oebin is format-independent and we already read it. Logs warnings but does NOT raise on a
    mismatch: a few missing/unexpected channels usually just get excluded downstream by impedance
    anyway (e.g. IM-1875 is missing CH1-CH8, replaced by duplicated ADC1-ADC8).

    Parameters:
        channel_names (list[str]):
            Ordered channel names from structure.oebin (from read_oebin_params)
        expected_channel_names (set[str]):
            The channel names we expect for this probe (e.g. CH1-CH256 + ADC1-ADC8 for a Berke probe)
        logger (Logger):
            Logger to track conversion progress
    """
    found_channel_names = set(channel_names)

    # Check for missing or unexpected channel names
    missing_channel_names = expected_channel_names - found_channel_names
    unexpected_channel_names = found_channel_names - expected_channel_names
    if missing_channel_names or unexpected_channel_names:
        logger.warning("Channel names in structure.oebin do not match expectations!!!!")
        if missing_channel_names:
            logger.warning(f"Missing channel names: {sorted(missing_channel_names)}")
        if unexpected_channel_names:
            logger.warning(f"Unexpected channel names: {sorted(unexpected_channel_names)}")
    else:
        logger.info(f"All {len(expected_channel_names)} expected channel names found in structure.oebin")

    # Check for duplicate channel names
    duplicate_names = [name for name, count in Counter(channel_names).items() if count > 1]
    if duplicate_names:
        logger.warning("Duplicate channel names found in structure.oebin!!!!")
        logger.warning(f"Duplicate channel names: {sorted(duplicate_names)}")


def get_port_visits(continuous_dat_file_path: Path,
                    total_channels: int,
                    port_visits_channel_num: int,
                    sample_rate: float,
                    logger,
                    pulse_high_threshold: float = 10_000) -> tuple[list[float], float]:
    """
    Extract port visit times from an OpenEphys continuous.dat file.

    Works for both Berke Lab custom probes and Neuropixels. The difference is just which file/channel
    the port visits live on, plus the sample rate and threshold:
      - Berke Lab probes: port visits are on ADC1 (channel 256) of the probe's own continuous.dat
        (264 channels, 30000 Hz, raw int16 threshold ~10000).
      - Neuropixels (OneBox): port visits are on the SEPARATE OneBox-ADC continuous.dat (ADC2,
        12 channels, 30300.5 Hz). ADC data is stored as raw int16 like everything else, so pass a
        threshold in raw int16 units (volts / bit_volts).
        
    NOTE: We return the bonsai start time (in seconds, not a sample count) because the Neuropixels port
    visit file (the OneBox-ADC at 30300.5 Hz) has a different sample rate than the probe file (30000 Hz).
    For Berke Lab probes the sample rates are the same (ADC1 lives in the same continuous.dat as the
    neural data), but returning a time keeps this function correct for both cases

    Parameters:
        continuous_dat_file_path (Path):
            Path to the 'continuous.dat' OpenEphys binary recording that holds the port visit channel
        total_channels (int):
            The total number of channels in that continuous.dat file
        port_visits_channel_num (int):
            The number of the channel (zero-indexed) that port visits are recorded on
        sample_rate (float):
            The sample rate (Hz) of that continuous.dat file (from structure.oebin)
        logger (Logger):
            Logger to track conversion progress
        pulse_high_threshold (float):
            Raw int16 value above which the port visit channel is considered "high". Defaults to 10000

    Returns:
        port_visits (list[float]):
            List of port visit times in seconds (relative to the bonsai start pulse)
        bonsai_start_time (float):
            Time (seconds) of the start pulse marking bonsai start, relative to the start of this
            continuous.dat. The caller converts this to a sample count in the neural recording's
            sample rate to trim data before bonsai start
    """
    # Downsample to ~1000 Hz before searching for pulses. 1000ish is low enough to speed things up
    # (this takes ~2mins on my machine) but high enough that we definitely don't miss any port visit pulses.
    # For reference, a typical port visit keeps the channel high for ~10ms, so 10 samples at 1000 Hz.
    # Sample rates aren't always a clean multiple of 1000 (the OneBox ADC is 30300.5 Hz), so compute the
    # integer decimation factor and the actual downsampled rate it gives us (used for samples->seconds).
    downsampled_fs_target = 1000
    decimation = max(1, round(sample_rate / downsampled_fs_target))
    downsampled_fs = sample_rate / decimation

    # Memory-map the large .dat file to avoid loading everything into memory. Reshape to (samples, channels)
    data_for_all_channels = np.memmap(continuous_dat_file_path, dtype='int16', mode='c').reshape(-1, total_channels)

    logger.debug(f"Reading Open Ephys port visits from channel {port_visits_channel_num}")
    logger.debug(f"Downsampling data from {sample_rate} Hz to {downsampled_fs} Hz so it isn't huge")

    # Extract the channel that records port visits and downsample so the data isn't massive
    # Force a contiguous copy of just the one channel up front (much faster than slicing the memmap column)
    visits_channel_data = np.array(data_for_all_channels[:, port_visits_channel_num])  # copies into RAM contiguously
    visits_channel_data = visits_channel_data[::decimation]

    # Find indices where the visits channel is high (aka port visit times)
    logger.debug(f"Recording times the channel is high (>{pulse_high_threshold})")
    pulse_above_threshold = np.where(visits_channel_data > pulse_high_threshold)[0]

    # If no port visits were found, return an empty list of visits and bonsai start time 0
    if pulse_above_threshold.size == 0:
        return [], 0.0

    # Find pulse boundaries (breaks in the sequence of threshold crossings)
    breaks = np.where(np.diff(pulse_above_threshold) != 1)[0] + 1

    # Get the durations of each pulse (each contiguous block where the visits channel is high)
    pulse_starts = np.insert(pulse_above_threshold[breaks], 0, pulse_above_threshold[0])
    pulse_ends = np.append(pulse_above_threshold[breaks - 1], pulse_above_threshold[-1])
    pulse_durations = pulse_ends - pulse_starts + 1

    logger.debug(f"Found {len(pulse_starts)} visit pulses.")

    # Only keep pulses at least 5ms long just in case (typical port visit keeps the channel high for ~10ms)
    min_pulse_duration = int(downsampled_fs * 0.005)
    pulse_starts = pulse_starts[pulse_durations > min_pulse_duration]

    logger.debug(f"Only keeping pulses at least 5ms long ({min_pulse_duration} samples at {downsampled_fs} Hz).")
    logger.debug(f"Kept {len(pulse_starts)} pulses.")
    logger.debug(f"Pulse durations: {pulse_durations}")

    # Set first pulse (which marks bonsai start time) as time=0 and convert to seconds
    # The duration of the first pulse is longer than a normal pulse (~500ms instead of ~10ms)
    if pulse_durations[0] < 300:
        logger.warning("Expected the first pulse marking the start time to have duration >= 300! "
                       f"Got pulse duration {pulse_durations[0]} - this may indicate a problem with the start pulse!")

    # Bonsai start time (seconds) relative to the start of this continuous.dat
    bonsai_start_time = float(pulse_starts[0] / downsampled_fs)
    logger.info(f"Bonsai start time occurred {bonsai_start_time}s after ephys was started.")
    logger.info("This will be set to time=0 and samples before this will be removed.")

    # Convert port visit times to seconds relative to bonsai start pulse
    port_visits = pulse_starts[1:] - pulse_starts[0]
    port_visits = [float(visit / downsampled_fs) for visit in port_visits]

    return port_visits, bonsai_start_time


def probe_name_token(name: str) -> str:
    """
    Reduce any of the names a probe goes by to the one token used to MATCH them.

    The same probe is written differently depending on where the name came from: the metadata says
    "ProbeA", the Open Ephys continuous folder is "OneBox-100.ProbeA", SpikeInterface reports
    "Record Node 101#OneBox-100.ProbeA", and the preprocessing pipeline writes "probeA". Comparing
    tokens lets all of those match, and keeps one definition of probe identity in one place.

    This is for matching ONLY, not for naming. It is the last dot-separated component, which is
    the probe name for a OneBox stream but is the stream index for others ("Rhythm_FPGA-100.0"
    reduces to "0"), so it makes a poor label. probe_label_for_stream does the naming.

    Two streams differing only before the last dot reduce to the same token. That does not pick
    the wrong one silently: every caller requires exactly one match and raises otherwise.

    Parameters:
        name (str): a probe name in any of those forms

    Returns:
        str: the matching token, lowercased (e.g. "probea")
    """
    return str(name).split("#")[-1].split(".")[-1].lower()


def probe_label_for_stream(requested_stream: str, resolved_stream: str) -> str:
    """
    The name a probe is called by in the nwb: its electrode groups, its 'probe' electrode column,
    and the 'probe' column on its units.

    This is what the user called it in the metadata, when they named one. Otherwise it is the
    recording's own stream name, with the Record Node prefix removed. It is never a parsed
    fragment of the stream name, which would label a Berke probe "0".

    Parameters:
        requested_stream (str): the 'stream' from the probe's metadata entry, or None
        resolved_stream (str): the full Open Ephys stream name it resolved to

    Returns:
        str: the label for this probe
    """
    if requested_stream:
        return str(requested_stream).split("#")[-1]
    return str(resolved_stream).split("#")[-1]


def resolve_probe_stream(folder_path: Path, logger, stream_name: str = None) -> str:
    """
    Find the full name of the probe (neural data) stream to convert from an OpenEphys recording.

    The ADC stream carries sync pulses, port visits and laser events rather than neural data, so it
    is never a candidate here (it is read separately by get_port_visits).

    Parameters:
        folder_path (Path):
            Path to the OpenEphys output folder
        logger (Logger):
            Logger to track conversion progress
        stream_name (str):
            Optional. The probe stream to convert, either the full Open Ephys stream name
            ("OneBox-100.ProbeA") or just the probe part ("ProbeA", as written in metadata).
            If None, there must be exactly one probe stream in the recording.

    Returns:
        str: the full Open Ephys stream name to pass to OpenEphysBinaryRecordingExtractor
    """
    streams, _ = OpenEphysBinaryRecordingExtractor.get_streams(folder_path=folder_path)
    logger.debug(f"Found streams in the OpenEphys binary data: {streams}")

    probe_streams = [s for s in streams if not s.upper().endswith("ADC")]
    if not probe_streams:
        logger.error(f"No probe (non-ADC) streams found in {folder_path}. Streams: {streams}")
        raise ValueError(f"No probe (non-ADC) streams found in {folder_path}. Streams: {streams}")

    # If no particular stream was requested, there must be exactly one probe stream to convert
    if stream_name is None:
        if len(probe_streams) != 1:
            logger.error(f"Found {len(probe_streams)} probe streams {probe_streams} but no stream was "
                         "requested. Specify which probe(s) to convert in the ephys metadata.")
            raise ValueError(f"Found {len(probe_streams)} probe streams {probe_streams} but no stream "
                             "was requested. Specify which probe(s) to convert in the ephys metadata.")
        return probe_streams[0]

    # Match on the shared probe token so every form of the name resolves to the same stream
    matches = [s for s in probe_streams if probe_name_token(s) == probe_name_token(stream_name)]
    if len(matches) != 1:
        logger.error(f"Expected exactly one probe stream matching '{stream_name}', found {matches} "
                     f"among {probe_streams}")
        raise ValueError(f"Expected exactly one probe stream matching '{stream_name}', found {matches} "
                         f"among {probe_streams}")
    logger.debug(f"Resolved requested probe stream '{stream_name}' to Open Ephys stream '{matches[0]}'")
    return matches[0]


def combine_probe_recordings(recordings_by_stream: dict, logger):
    """
    Combine one recording per probe into a single recording holding every probe's channels, so that
    multi-probe (e.g. bilateral Neuropixels) data can be written as a single ElectricalSeries.

    Spyglass reads raw ephys with nwbfile.get_acquisition(), which requires exactly ONE acquisition
    object, so all probes must share one ElectricalSeries rather than getting one each.

    Channels are kept in probe order: all of the first probe's channels, then all of the second's.
    The returned channel ids are prefixed with the stream name ("ProbeA_CH0") because every
    Neuropixels probe uses the same CH0..CH383 channel names, which would otherwise collide.

    Probes are hardware-synced to the same clock (verified on IM-1979: laser artifacts seen by both
    probes drift by <0.01 ppm over a session), so samples can be combined by index without
    resampling. Probes can still START a few samples apart, so recordings are trimmed to the
    shortest length and any difference is logged.

    Parameters:
        recordings_by_stream (dict):
            stream name -> SpikeInterface recording for that probe, already channel-sliced
            and trimmed to the conversion start time
        logger (Logger):
            Logger to track conversion progress

    Returns:
        combined_recording:
            The combined SpikeInterface recording (all probes' channels, common sample count)
        timestamps (np.ndarray):
            Timestamps (seconds) for the combined recording, taken from the first probe after
            alignment. All probes share this clock by construction.
    """
    from spikeinterface.core import aggregate_channels

    stream_names = list(recordings_by_stream)
    if len(stream_names) == 1:
        logger.debug(f"Only one probe stream ('{stream_names[0]}'), no combining needed")
        single_recording = recordings_by_stream[stream_names[0]]
        return single_recording, single_recording.get_times()

    # All probes must share a sampling rate for a combined ElectricalSeries to be meaningful
    sampling_rates = {name: rec.get_sampling_frequency() for name, rec in recordings_by_stream.items()}
    if len(set(sampling_rates.values())) != 1:
        logger.error(f"Probe streams do not share a sampling rate: {sampling_rates}")
        raise ValueError(f"Probe streams do not share a sampling rate: {sampling_rates}")

    # Align the probes in TIME before combining. They share a hardware clock, but can start a few
    # samples apart (measured on IM-1979 via laser artifacts seen by both probes: ~0 samples on
    # 2026-09-04, ~45 samples / 1.5 ms on 2026-09-11). Without this, one ElectricalSeries row would
    # hold samples that were actually recorded at slightly different times.
    # sample_index_to_time avoids materializing the whole (multi-GB) timestamp vector.
    sampling_rate = next(iter(sampling_rates.values()))
    start_times = {name: rec.sample_index_to_time(0) for name, rec in recordings_by_stream.items()}
    if max(start_times.values()) != min(start_times.values()):
        log_and_print(
            logger,
            f"Probe streams do not start at the same time: {start_times}. Trimming the front of the "
            "earlier stream(s) so all probes begin at the same time in the combined ElectricalSeries.",
            level="warning",
        )
    common_start_time = max(start_times.values())
    time_aligned_recordings = {}
    for name, recording in recordings_by_stream.items():
        samples_to_align = int(round((common_start_time - start_times[name]) * sampling_rate))
        if samples_to_align > 0:
            logger.info(f"Trimming {samples_to_align} sample(s) from the start of '{name}' to align probes")
            recording = recording.frame_slice(start_frame=samples_to_align, end_frame=None)
        time_aligned_recordings[name] = recording
    recordings_by_stream = time_aligned_recordings

    # Trim all probes to the shortest length. Probes are synced to one clock but can start/stop a
    # few samples apart, so this drops a handful of samples off the end of the longer stream(s).
    sample_counts = {name: rec.get_num_samples() for name, rec in recordings_by_stream.items()}
    common_num_samples = min(sample_counts.values())
    logger.info(f"Combining {len(stream_names)} probe streams: samples per stream {sample_counts}")
    if len(set(sample_counts.values())) != 1:
        log_and_print(
            logger,
            f"Probe streams have different numbers of samples {sample_counts}. Trimming all probes to the "
            f"shortest ({common_num_samples} samples) so they can share one ElectricalSeries. "
            f"At most {max(sample_counts.values()) - common_num_samples} samples are dropped from the end "
            "of the longer stream(s).",
            level="warning",
        )

    combined_channel_ids = []
    aligned_recordings = []
    for probe_index, (stream, recording) in enumerate(recordings_by_stream.items()):
        # Always slice (even when the length already matches) so we work on a new recording object.
        # We clear the time bookkeeping below, and that mutates in place - without this, we would be
        # modifying the caller's recording.
        recording = recording.frame_slice(start_frame=0, end_frame=common_num_samples)

        # SpikeInterface refuses to aggregate recordings whose channel locations overlap, and every
        # Neuropixels probe has sites at the same relative positions. Shift each probe's geometry
        # apart so the locations are unique. This only affects the SpikeInterface object used to read
        # traces - the rel_x/rel_y written to the nwb come from our own coordinates table.
        if probe_index > 0:
            try:
                probe = recording.get_probe()
                probe.move([probe_index * PROBE_LOCATION_OFFSET_UM, 0.0])
                recording = recording.set_probe(probe)
            except Exception as error:
                logger.debug(f"Could not offset probe geometry for '{stream}' ({type(error).__name__}: "
                             f"{error}). Continuing - this only matters if locations collide.")

        aligned_recordings.append(recording)
        combined_channel_ids.extend(f"{stream}_{channel_id}" for channel_id in recording.channel_ids)

    # Capture the real timestamps from the (now aligned and length-matched) first probe BEFORE
    # clearing the per-recording time bookkeeping below. These are the timestamps written to the
    # ElectricalSeries - all probes are aligned to this same clock by construction.
    timestamps = aligned_recordings[0].get_times()

    # SpikeInterface refuses to aggregate recordings whose time bookkeeping disagrees ("All segments
    # should have the same t_start"), and slicing the front of one probe leaves them inconsistent.
    # Clear it on every recording so they agree: we already captured the timestamps we need, and the
    # combined recording is only used to read traces.
    for recording in aligned_recordings:
        recording.reset_times()

    combined_recording = aggregate_channels(aligned_recordings, renamed_channel_ids=combined_channel_ids)
    logger.info(f"Combined recording is {combined_recording.get_num_samples()} samples x "
                f"{combined_recording.get_num_channels()} channels across probes {stream_names}")
    return combined_recording, timestamps


def get_raw_probe_start_times(session_folders: list, probe_streams_by_label: dict, logger) -> dict:
    """
    Read each probe's start time in each recording, from the raw Open Ephys timestamps.

    Only needed when the traces come from the preprocessed files, which are flat binaries with no
    time axis of their own. The probes share a hardware clock but start a few samples apart, and by
    a different amount in each recording, so these start times are what lets the probes be aligned
    to each other per session.

    Only the small per-stream timestamp files are touched - the raw continuous.dat is never read.

    Parameters:
        session_folders (list):
            The recordings, in order, each with "path" and "session"
        probe_streams_by_label (dict):
            Short probe label -> full Open Ephys stream name
        logger (Logger):
            Logger to track conversion progress

    Returns:
        dict: probe label -> {session label: start time in seconds}
    """
    start_times = {probe_label: {} for probe_label in probe_streams_by_label}
    for session_folder in session_folders:
        for probe_label, stream in probe_streams_by_label.items():
            recording, _conversion_factor = open_probe_stream_recording(
                folder_path=session_folder["path"], logger=logger, stream_name=stream,
                samples_to_remove=0,
            )
            start_times[probe_label][session_folder["session"]] = float(recording.sample_index_to_time(0))

    for probe_label, by_session in start_times.items():
        logger.info(f"Raw start times for '{probe_label}': "
                    f"{ {session: round(t, 6) for session, t in by_session.items()} }")
    return start_times


def get_raw_session_recordings(
    session_folders: list,
    probe_streams_by_label: dict,
    exclude_channel_names: list,
    samples_to_remove: int,
    logger,
) -> tuple:
    """
    Build one combined recording per session, straight from the raw Open Ephys recordings.

    Each recording is read on its own and its probes aligned and combined, so that the sessions can
    then be concatenated into a single ElectricalSeries.

    Parameters:
        session_folders (list):
            The recordings, in concatenation order, each with "path" and "session"
        probe_streams_by_label (dict):
            Short probe label -> full Open Ephys stream name, in ElectricalSeries channel order
        exclude_channel_names (list):
            Channel names to drop (the extra channels on the 252-channel Berke probes)
        samples_to_remove (int):
            Samples to trim from the start of the FIRST recording (data before bonsai start)
        logger (Logger):
            Logger to track conversion progress

    Returns:
        session_recordings (list): one combined recording per session, in order
        session_timestamps (list): each session's own timestamps, as recorded
        num_samples_by_probe (dict): each probe's full length across every recording
        channel_conversion_factor_uv (float): the conversion factor from raw data to uV
    """
    session_recordings = []
    session_timestamps = []
    num_samples_by_probe = {}
    channel_conversion_factor_uv = None
    conversion_factors_by_session = {}

    for session_index, session_folder in enumerate(session_folders):
        # Only the first recording is trimmed: bonsai start is a time within that recording, and
        # the later recordings are separate acquisitions that happen after it
        session_samples_to_remove = samples_to_remove if session_index == 0 else 0
        logger.info(f"Reading recording '{session_folder['session']}' from {session_folder['path']}")
        (
            session_recording,
            channel_conversion_factor_uv,
            session_recorded_timestamps,
            session_num_samples_by_probe,
        ) = get_combined_raw_ephys_data(folder_path=session_folder["path"],
                                        probe_streams_by_label=probe_streams_by_label,
                                        logger=logger,
                                        exclude_channels=exclude_channel_names,
                                        samples_to_remove=session_samples_to_remove,
                                        )
        # An ElectricalSeries has ONE conversion factor, so every recording must share it.
        # Overwriting it per recording would silently scale part of the data differently.
        if conversion_factors_by_session and channel_conversion_factor_uv not in conversion_factors_by_session.values():
            conversion_factors_by_session[session_folder["session"]] = channel_conversion_factor_uv
            logger.error(f"Recordings have different conversion factors: {conversion_factors_by_session}")
            raise ValueError(f"Recordings have different conversion factors: "
                             f"{conversion_factors_by_session}. They cannot share one ElectricalSeries.")
        conversion_factors_by_session[session_folder["session"]] = channel_conversion_factor_uv

        session_recordings.append(session_recording)
        session_timestamps.append(session_recorded_timestamps)
        # A probe's full length is the sum over the recordings it was sorted across
        for probe_label, probe_num_samples in session_num_samples_by_probe.items():
            num_samples_by_probe[probe_label] = num_samples_by_probe.get(probe_label, 0) + probe_num_samples

    return session_recordings, session_timestamps, num_samples_by_probe, channel_conversion_factor_uv
def get_session_time_offsets(session_folders: list, logger) -> list:
    """
    Work out when each recording started, relative to the first one.

    Separate Open Ephys recordings are separate acquisitions with no shared sample numbering, so
    the only link between them is wall-clock time, which Open Ephys puts in the folder name
    ("2026-09-04_14-20-43"). The gap between recordings is real dead time - on IM-1979 2026-09-04
    the behavior session runs 7,883s and the optotagging session starts 8,335s later, so about
    452s of nothing. Keeping that gap in the timestamps is what makes Spyglass derive one valid
    interval per session instead of treating the file as continuous.

    Folder names carry whole seconds, so these offsets are only good to ~1s. That is fine for
    sorting (the gap just has to exist) but is not a sync, and anything needing sample-accurate
    cross-session timing needs a real sync signal instead.

    Parameters:
        session_folders (list):
            The recordings, in concatenation order, each with "path" and "session" (from
            get_session_folders)
        logger (Logger):
            Logger to track conversion progress

    Returns:
        list[float]: seconds from the first recording's start to each recording's start
        (the first is always 0.0)
    """
    session_start_times = []
    for session_folder in session_folders:
        folder_name = str(session_folder["path"]).replace("\\", "/").rstrip("/").split("/")[-1]
        match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", folder_name)
        if not match:
            logger.error(f"No date and time in the Open Ephys folder name '{folder_name}'")
            raise ValueError(f"No date and time in the Open Ephys folder name '{folder_name}'. "
                             "The recordings are placed in time by their folder names.")
        session_start_times.append(datetime.strptime(match.group(0), "%Y-%m-%d_%H-%M-%S"))

    first_start = session_start_times[0]
    offsets = [(start - first_start).total_seconds() for start in session_start_times]

    if any(offset < 0 for offset in offsets):
        ordering = {folder["session"]: str(start) for folder, start in zip(session_folders, session_start_times)}
        logger.error(f"Recordings are not in chronological order: {ordering}")
        raise ValueError(f"Recordings are not in chronological order: {ordering}. List them in the "
                         "order they were recorded.")

    for session_folder, start, offset in zip(session_folders, session_start_times, offsets):
        logger.info(f"Recording '{session_folder['session']}' started {start} (+{offset:.0f}s)")
    return offsets


def concatenate_session_recordings(session_recordings: list, session_timestamps: list,
                                   session_offsets: list, session_labels: list, logger) -> tuple:
    """
    Join each session's recording into ONE recording, with timestamps that keep the real gaps.

    The traces are contiguous (the gap holds no samples), but the timestamps are not: sample i
    keeps its true time, so a reader sees two stretches of data separated by dead time. Spyglass
    splits on that gap and derives one valid interval per session, which is what lets a sort be
    restricted to one session or run across both.

    Parameters:
        session_recordings (list):
            One combined recording per session, in order. Every session must have the same
            channels, since they become one ElectricalSeries.
        session_timestamps (list):
            Each session's own timestamps, as recorded. They are shifted to start at that session's
            offset, which keeps any real irregularity inside a session rather than assuming a
            perfectly uniform clock.
        session_offsets (list):
            Each session's start time on the nwb clock, in seconds (the first is usually 0.0 after
            trimming to bonsai start)
        session_labels (list):
            Each session's name, used for the nwb epochs
        logger (Logger):
            Logger to track conversion progress

    Returns:
        concatenated_recording:
            All sessions end to end, as one recording
        timestamps (np.ndarray):
            Timestamp per sample, carrying the real gaps between sessions
        epochs (list[dict]):
            One entry per session with "session", "start_time" and "stop_time", for the nwb
            epochs table
    """
    from spikeinterface.core import concatenate_recordings

    if len(session_recordings) == 1:
        logger.debug("Only one recording, no concatenation needed")

    channel_counts = {label: rec.get_num_channels() for label, rec in zip(session_labels, session_recordings)}
    if len(set(channel_counts.values())) != 1:
        logger.error(f"Recordings have different channel counts: {channel_counts}")
        raise ValueError(f"Recordings have different channel counts: {channel_counts}. They cannot "
                         "share one ElectricalSeries.")

    sampling_rates = {label: rec.get_sampling_frequency()
                      for label, rec in zip(session_labels, session_recordings)}
    if len(set(sampling_rates.values())) != 1:
        logger.error(f"Recordings have different sampling rates: {sampling_rates}")
        raise ValueError(f"Recordings have different sampling rates: {sampling_rates}.")

    timestamps = []
    epochs = []
    for label, recording, recorded_timestamps, offset in zip(
        session_labels, session_recordings, session_timestamps, session_offsets
    ):
        num_samples = recording.get_num_samples()
        if len(recorded_timestamps) != num_samples:
            logger.error(f"Recording '{label}' has {num_samples} samples but "
                         f"{len(recorded_timestamps)} timestamps")
            raise ValueError(f"Recording '{label}' has {num_samples} samples but "
                             f"{len(recorded_timestamps)} timestamps.")
        # Keep the session's own timing, moved to where the session sits on the nwb clock
        placed_timestamps = np.asarray(recorded_timestamps) - recorded_timestamps[0] + offset
        timestamps.append(placed_timestamps)
        epochs.append({
            "session": label,
            "start_time": float(placed_timestamps[0]),
            "stop_time": float(placed_timestamps[-1]),
        })
        logger.info(f"Recording '{label}': {num_samples} samples, "
                    f"{placed_timestamps[0]:.3f}s to {placed_timestamps[-1]:.3f}s")

    # Each gap is real dead time between two separate acquisitions
    for earlier, later in zip(epochs, epochs[1:]):
        gap = later["start_time"] - earlier["stop_time"]
        if gap <= 0:
            logger.error(f"Recording '{later['session']}' starts before '{earlier['session']}' ends "
                         f"(overlap of {-gap:.3f}s)")
            raise ValueError(f"Recording '{later['session']}' starts before '{earlier['session']}' "
                             f"ends (overlap of {-gap:.3f}s). Check the recording order and folder names.")
        logger.info(f"{gap:.1f}s gap between '{earlier['session']}' and '{later['session']}'")

    # concatenate_recordings ignores the recordings' own times, which is what we want: the
    # timestamps above are the authority, and they carry the gaps the samples do not
    concatenated_recording = (session_recordings[0] if len(session_recordings) == 1
                              else concatenate_recordings(session_recordings))
    timestamps = np.concatenate(timestamps)

    logger.info(f"Concatenated {len(session_recordings)} recording(s): "
                f"{concatenated_recording.get_num_samples()} samples x "
                f"{concatenated_recording.get_num_channels()} channels, "
                f"spanning {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s")
    return concatenated_recording, timestamps, epochs


def get_combined_raw_ephys_data(
    folder_path: Path,
    probe_streams_by_label: dict,
    logger,
    exclude_channels: list = (),
    samples_to_remove: int = 0,
) -> tuple[SpikeInterfaceRecordingDataChunkIterator, float, np.ndarray]:
    """
    Get the raw ephys data for one or more probes as a SINGLE recording, so that multi-probe
    (e.g. bilateral Neuropixels) data can be written as one ElectricalSeries.

    Each probe stream is opened and trimmed the same way as the single-probe path, then the probes
    are aligned in time and combined channel-wise (see combine_probe_recordings).

    Parameters:
        folder_path (Path):
            Path to the Open Ephys output folder
        probe_streams_by_label (dict):
            Short probe label -> full Open Ephys stream name, in the order the probes' channels
            should appear in the ElectricalSeries (e.g. {"ProbeA": "OneBox-100.ProbeA", ...}).
            The short label prefixes the combined channel ids.
        logger (Logger):
            Logger to track conversion progress
        exclude_channels (list):
            Channel names to exclude, if any
        samples_to_remove (int):
            Number of samples to trim off the start of each probe (to drop data before bonsai start)

    Returns:
        combined_recording:
            All probes' channels as one recording, for this one session
        channel_conversion_factor_uv (float):
            The conversion factor from the raw data to uV (shared by all probes)
        original_timestamps (np.ndarray):
            This session's own timestamps, as recorded (the caller places the session in time)
        num_samples_by_probe (dict):
            probe label -> that probe's own full (untrimmed) recording length in samples. Spike
            sorting uses it to confirm each analyzer was computed on that probe's recording.
    """
    recordings_by_stream = {}
    conversion_factors = {}
    num_samples_by_probe = {}
    for probe_label, stream in probe_streams_by_label.items():
        recording, conversion_factor = open_probe_stream_recording(
            folder_path=folder_path,
            logger=logger,
            stream_name=stream,
            exclude_channels=exclude_channels,
            samples_to_remove=samples_to_remove,
        )
        recordings_by_stream[probe_label] = recording
        conversion_factors[probe_label] = conversion_factor
        # This probe's own full length, before any alignment trim. The spike sorting guard compares
        # it against the analyzer, and each probe was sorted on its OWN recording - which can be a
        # few samples longer or shorter than the others.
        num_samples_by_probe[probe_label] = recording.get_num_samples() + samples_to_remove

    # An ElectricalSeries has a single conversion factor, so every probe must share one
    if len(set(conversion_factors.values())) != 1:
        logger.error(f"Probe streams have different conversion factors: {conversion_factors}")
        raise ValueError(f"Probe streams have different conversion factors: {conversion_factors}. "
                         "They cannot share a single ElectricalSeries.")
    channel_conversion_factor_uv = next(iter(conversion_factors.values()))

    combined_recording, original_timestamps = combine_probe_recordings(recordings_by_stream, logger)
    return (
        combined_recording,
        channel_conversion_factor_uv,
        original_timestamps,
        num_samples_by_probe,
    )


def open_probe_stream_recording(
    folder_path: Path,
    logger,
    stream_name: str = None,
    exclude_channels: list = (),
    samples_to_remove: int = 0,
):
    """
    Open a single probe stream of an Open Ephys recording as a SpikeInterface recording, keeping
    only the neural "CH" channels and trimming samples before the conversion start time.

    Used by get_combined_raw_ephys_data, once per probe, so every probe is treated identically.

    Parameters:
        folder_path (Path):
            Path to the Open Ephys output folder
        logger (Logger):
            Logger to track conversion progress
        stream_name (str):
            Optional. Which probe stream to open. If None, the recording must contain exactly one.
        exclude_channels (list):
            Channel names to exclude, if any
        samples_to_remove (int):
            Number of samples to trim off the start of the recording

    Returns:
        recording_sliced:
            The channel-sliced, time-trimmed SpikeInterface recording
        channel_conversion_factor_uv (float):
            The conversion factor from the raw data to uV
    """
    stream_to_convert = resolve_probe_stream(folder_path=folder_path, stream_name=stream_name, logger=logger)

    # Ignore the "ADC" channels
    recording = OpenEphysBinaryRecordingExtractor(folder_path=folder_path, stream_name=stream_to_convert)
    # Exclude extra channels
    if exclude_channels:
        logger.debug(f"Excluding {len(exclude_channels)} channels: {exclude_channels}")
    channel_ids_to_convert = [
        ch for ch in recording.channel_ids if ch.startswith("CH") and ch not in exclude_channels
    ]
    recording_sliced = recording.select_channels(channel_ids=channel_ids_to_convert)

    # Confirm all channel names start with "CH"
    assert all([ch.startswith("CH") for ch in recording_sliced.channel_ids]), \
        (f"Some channels do not start with 'CH': {recording_sliced.channel_ids}")

    logger.debug(
        f"Found {len(recording_sliced.channel_ids)} Open Ephys channels to convert: "
        f"{recording_sliced.channel_ids}"
    )

    # Get the channel conversion factor
    channel_conversion_factors_uv = recording_sliced.get_channel_gains()
    # Warn if the channel conversion factors are not the same for all channels
    if not all(channel_conversion_factors_uv == channel_conversion_factors_uv[0]):
        log_and_print(
            logger,
            "The channel conversion factors are not the same for all channels. "
            "This is unexpected and may indicate a problem with the conversion factors.",
            level="warning",
        )
    # Just grab the first one, because it should be the same for all channels
    channel_conversion_factor_uv = channel_conversion_factors_uv[0]
    logger.debug(f"Channel conversion factor in uV: {channel_conversion_factor_uv}")

    # NOTE channel offsets should be 0 for all channels in openephys data
    channel_conversion_offsets = recording_sliced.get_channel_offsets()
    assert all(channel_conversion_offsets == 0), "Channel conversion offsets are not all 0."

    fs = recording_sliced.get_sampling_frequency()
    logger.debug(f"Open Ephys sampling frequency: {fs}")

    # Remove ephys samples before bonsai start time
    recording_sliced = recording_sliced.frame_slice(start_frame=samples_to_remove, end_frame=None)
    logger.debug(f"Trimmed {samples_to_remove} samples ({float(samples_to_remove/fs)}s) from the start of "
                 "ephys data to exclude data before bonsai start.")

    return recording_sliced, channel_conversion_factor_uv


def add_probe_info(nwbfile: NWBFile, probe_type: str, logger, probe_label: str = None, probe_id: int = 0):
    """
    Add the Probe as a device to the nwbfile

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled.
        probe_type (str):
            The probe model, matching a name in resources/electrophysiology/ephys_devices.yaml
        logger (Logger):
            Logger to track conversion progress

    Returns:
        probe_metadata (dict):
            Metadata for this Probe read from resources/ephys_devices.yaml
        probe_obj (Probe):
            The Probe object added to the nwbfile
    """
    # Load probe metadata
    with open(DEVICES_PATH, "r") as f:
        devices = yaml.safe_load(f)

    # metadata lists the probe type(s) used. Multiple physical probes of the same type share one
    # entry, so a single-entry list is normal even for bilateral recordings.
    probe_name = probe_type
    logger.info(f"Probe type is: {probe_name}")
    # Find the matching device by name in the devices list
    for device in devices["probe"]:
        if device["name"] == probe_name:
            probe_metadata = device
            break
    else:
        logger.error(f"Probe '{probe_name}' not found in resources/ephys_devices.yaml")
        raise ValueError(f"Probe '{probe_name}' not found in resources/ephys_devices.yaml")

    logger.debug("Probe metadata:")
    for key in probe_metadata:
        logger.debug(f"{key}: {probe_metadata[key]}")

    required_probe_keys = {
        "name",
        "description",
        "manufacturer",
        "contact_side_numbering",
        "contact_size",
        "units",
        "shanks",
    }
    assert required_probe_keys.issubset(probe_metadata.keys()), (
        f"Probe is missing required keys {required_probe_keys - probe_metadata.keys()}"
    )
    # Create the Probe device and add it to the nwb.
    # Device names must be unique within an nwbfile, so when there is more than one physical probe
    # (e.g. bilateral Neuropixels, which are the same model) the stream name is appended to make the
    # device name distinct. probe_type stays the model name, which is what Spyglass keys Probe on,
    # so both physical probes still resolve to the same Spyglass Probe entry.
    device_name = f"{probe_metadata['name']} {probe_label}" if probe_label else probe_metadata["name"]
    logger.info(f"Adding Probe device '{device_name}' (probe_type '{probe_metadata['name']}')")
    probe_obj = Probe(
        id=probe_id,  # int
        name=device_name,  # str
        probe_type=probe_metadata["name"],  # str
        probe_description=probe_metadata["description"],  # str
        manufacturer=probe_metadata["manufacturer"],  # str
        contact_side_numbering=probe_metadata["contact_side_numbering"],  # bool
        contact_size=float(probe_metadata["contact_size"]),  # float
        units=probe_metadata["units"],  # str (um or mm)
    )
    nwbfile.add_device(probe_obj)
    return probe_metadata, probe_obj


### Functions for Berke Lab custom probes

def read_open_ephys_settings_xml(settings_file_path: Path, logger) -> str:
    """
    Read the OpenEphys settings.xml file to get the filtering applied across all channels.

    Supports two settings.xml formats:

    Old format (Open Ephys < 1.0):
        <PROCESSOR name="Sources/Rhythm FPGA" ...>
          <CHANNEL_INFO>
            <CHANNEL name="CH1" number="0" gain="0.19499999284744263"/>
            ...
          </CHANNEL_INFO>
          <EDITOR ... LowCut="..." HighCut="..." />
        </PROCESSOR>

    Note that only the information in the "Sources/Rhythm FPGA" is relevant -
    the other processors ("Filters/Channel Map", "Filters/Bandpass Filter", etc.)
    reflect OpenEphys display settings only

    New format (Open Ephys >= 1.0):
        <PROCESSOR name="Acquisition Board" ...>
          <STREAM name="acquisition_board" ... channel_count="264"/>
          <EDITOR ... LowCut="..." HighCut="..." />
        </PROCESSOR>

    NOTE: All we actually use from settings.xml is the filtering info (a single string stored on
    every electrode). We used to also validate the channel numbers/names against the expected
    256 CH + 8 ADC layout here, but that validation now happens from structure.oebin instead
    (see validate_oebin_channel_names), which is format-independent and works across Open Ephys
    versions and both Berke Lab and Neuropixels probes. 
    That validation warns but doesn't error: some of our rats (e.g. IM-1875) have incorrect channels
    due to an old open ephys bug (missing CH1-CH8, replaced by a duplicated ADC1-ADC8) that we knowingly
    ignore because they end up being excluded by impedance criteria anyway

    Parameters:
        settings_file_path (Path):
            Path to the OpenEphys settings.xml file
        logger (Logger):
            Logger to track conversion progress

    Returns:
        filtering_info (str):
            Filtering applied to all channels
    """
    settings_tree = ET.parse(settings_file_path)
    settings_root = settings_tree.getroot()

    # The EDITOR (with the filtering info) lives inside the acquisition processor, whose name
    # differs between the old format ("Sources/Rhythm FPGA") and the new format ("Acquisition Board")
    
    # Try old format first: "Sources/Rhythm FPGA" processor
    acq_processor = settings_root.find(".//PROCESSOR[@name='Sources/Rhythm FPGA']")
    if acq_processor is None:
        # Try new format: "Acquisition Board" processor
        acq_processor = settings_root.find(".//PROCESSOR[@name='Acquisition Board']")
    if acq_processor is None:
        logger.error(
            "Could not find 'Sources/Rhythm FPGA' or 'Acquisition Board' processor in the settings.xml file."
        )
        raise ValueError(
            "Could not find 'Sources/Rhythm FPGA' or 'Acquisition Board' processor in the settings.xml file."
        )

    # Get the filtering info from the EDITOR element (same location in both formats)
    editor = acq_processor.find("EDITOR")
    if editor is not None and "LowCut" in editor.attrib and "HighCut" in editor.attrib:
        lowcut = float(editor.attrib.get("LowCut"))
        highcut = float(editor.attrib.get("HighCut"))
        filtering_info = f"Filter with highcut={highcut} Hz, lowcut={lowcut} Hz"
        logger.info(f"Filtering info from settings.xml: {filtering_info}")
    else:
        logger.warning("EDITOR tag with LowCut/HighCut not found in settings.xml, no filtering info for channels!")
        filtering_info = "Unknown"

    return filtering_info


def get_electrode_info(metadata: dict, probe_type: str, logger, fig_dir: Path = None) -> pd.DataFrame:
    """
    Create a DataFrame of complete electrode information for a Berke Lab silicon probe.
    Combines channel map, electrode coordinates, and impedance data.

    Parameters:
        metadata (dict):
            Full metadata dictionary (from user-specified yaml)
        logger (Logger):
            Logger to track conversion progress
        fig_dir (Path):
            Optional. The directory to save the figure. If None, the figure will not be saved.

    Returns:
        pd.DataFrame:
            DataFrame of electrode information including electrode name, shank, electrode number,
            intan channel, relative x and y coordinates, impedance info, and 'bad channel' tag
    """

    # Get info for this probe
    probe_name = probe_type

    try:
        probe_info = BERKE_LAB_PROBES[probe_name]
    except KeyError:
        logger.error(f"Unknown probe '{probe_name}'. Valid probes are: {list(BERKE_LAB_PROBES)}")
        raise ValueError(f"Unknown probe '{probe_name}'. Valid probes are: {list(BERKE_LAB_PROBES)}")

    # Combine channel map and electrode coords into a single dataframe
    channel_map_df = pd.read_csv(probe_info["channel_map"])
    electrode_coords_df = pd.read_csv(probe_info["electrode_coords"])
    channel_coords_df = pd.merge(channel_map_df, electrode_coords_df, on=["shank", "electrode"], how="outer")

    # Make sure we have exactly 256 rows (one per Intan channel)
    if len(channel_coords_df) != 256:
        logger.error(f"Expected 256 rows in merged probe info, but got {len(channel_coords_df)}")
        raise ValueError(f"Expected 256 rows in merged probe info, but got {len(channel_coords_df)}")

    # Choose the correct channel map based on how the rat was plugged in (assume "chip_first" if none specified)
    plug_order = metadata["ephys"].get("plug_order", "chip_first")
    logger.info(f"Plug order is: {plug_order}")

    # Rename the correct channel map column to 'intan_channel' and remove the unused channel map column
    if plug_order == "chip_first":
        channel_coords_df = channel_coords_df.drop(columns=["cable_first"])
        channel_coords_df = channel_coords_df.rename(columns={"chip_first": "intan_channel"})
    elif plug_order == "cable_first":
        channel_coords_df = channel_coords_df.drop(columns=["chip_first"])
        channel_coords_df = channel_coords_df.rename(columns={"cable_first": "intan_channel"})
    else:
        logger.error(f"Unknown plug order: {plug_order}, expected 'chip_first' or 'cable_first'")
        raise ValueError(f"Unknown plug order: {plug_order}, expected 'chip_first' or 'cable_first'")

    # Add another channel name column based on intan_channel to match Open Ephys 1-based 'CH##' channel names
    channel_coords_df["open_ephys_channel_string"] = channel_coords_df["intan_channel"].apply(lambda i: f"CH{i + 1}")

    # Plot the channel coordinates
    plot_channel_map(probe_name=probe_name, channel_coords=channel_coords_df, fig_dir=fig_dir)

    # Now load impedance data
    impedance_file_path = metadata["ephys"]["impedance_file_path"]
    impedance_data = pd.read_csv(impedance_file_path)

    # Check that the expected columns are present in order and no extra columns are present
    expected_columns = [
        "Channel Number",
        "Channel Name",
        "Port",
        "Enabled",
        "Impedance Magnitude at 1000 Hz (ohms)",
        "Impedance Phase at 1000 Hz (degrees)",
        "Series RC equivalent R (Ohms)",
        "Series RC equivalent C (Farads)",
    ]

    assert impedance_data.columns.tolist() == expected_columns, (
        f"Impedance file has columns {impedance_data.columns.tolist()}, "
        f"does not match expected columns {expected_columns}"
    )
    logger.debug(f"Impedance file has expected columns {expected_columns}")

    # Drop the first column which should be the same as the second column
    assert (impedance_data["Channel Number"] == impedance_data["Channel Name"]).all(), (
        "First column is not the same as the second column."
    )
    impedance_data.drop(columns=["Channel Number"], inplace=True)

    # Make sure the channel coords dataframe and impedance dataframe have the same length (256)
    assert len(channel_coords_df) == len(impedance_data), (
        "Mismatch in lengths: "
        f"channel coordinates ({len(channel_coords_df)}), "
        f"impedance data ({len(impedance_data)})"
    )
    logger.debug(f"Channel coordinates and impedance data have the same length ({len(channel_coords_df)})")

    # Create a column 'intan_channel' (zero-indexed) in the impedance df so we can merge with channel coords df
    impedance_data["intan_channel"] = range(len(impedance_data))
    full_electrode_info_df = channel_coords_df.merge(impedance_data, on="intan_channel", how="left")
    assert len(full_electrode_info_df) == len(channel_coords_df) == 256, (
        "Full electrode info dataframe does not have 256 rows"
    )

    # Sanity check that the channel numbers from the "Channel Name" column match the "intan_channel" column
    # We expect that the impedance file contains either Port B and Port C, or Port A and Port D
    port_prefixes = sorted(full_electrode_info_df["Channel Name"].str.split("-").str[0].unique())
    assert port_prefixes in [["A", "D"], ["B", "C"]], f"Unexpected port prefixes: {port_prefixes}"

    # Get the channel number based on the "Channel Name" column
    # channel nums = the num for Port A/B channels (e.g. B-001 is 1) and num+128 for Port C/D channels (C-001 is 129)
    expected_channel_nums = full_electrode_info_df["Channel Name"].apply(
        lambda s: int(s.split("-")[1]) + (128 if s.startswith(("C-", "D-")) else 0)
    )

    mismatched_channel_nums = full_electrode_info_df["intan_channel"] != expected_channel_nums
    if mismatched_channel_nums.any():
        logger.warning(f"{mismatched_channel_nums.sum()} rows have mismatched intan_channel values")
        logger.warning(full_electrode_info_df[mismatched_channel_nums])
    else:
        logger.debug("All intan channel numbers from coordinates file match channel names from the impedance file!")

    # Mark electrodes with impedances outside the allowed range as "bad channel"
    # The default impedance range is between >=0.1 MOhms or <=3.0 MOhms
    min_impedance = float(metadata["ephys"].get("min_impedance_ohms", MIN_IMPEDANCE_OHMS))
    max_impedance = float(metadata["ephys"].get("max_impedance_ohms", MAX_IMPEDANCE_OHMS))

    logger.info(f"Marking channels with impedance > {max_impedance} Ohms or < {min_impedance} Ohms as 'bad_channel'")
    full_electrode_info_df["bad_channel"] = (
        full_electrode_info_df["Impedance Magnitude at 1000 Hz (ohms)"].lt(min_impedance) |
        full_electrode_info_df["Impedance Magnitude at 1000 Hz (ohms)"].gt(max_impedance)
    ).astype(int)

    # Log the number of good and bad channels
    bad_channel_count = full_electrode_info_df["bad_channel"].sum()
    good_channel_count = len(full_electrode_info_df) - bad_channel_count
    logger.info(f"There are {good_channel_count} good channels and {bad_channel_count} bad channels based on impedance")

    # Plot channel impedances
    plot_channel_impedances(probe_name=probe_name, electrode_info=full_electrode_info_df, 
                            min_impedance=min_impedance, max_impedance=max_impedance, fig_dir=fig_dir)

    return full_electrode_info_df


def add_electrode_data_berke_probe(
    *,
    nwbfile: NWBFile,
    filtering_info: str,
    metadata: dict,
    probe_metadata: dict,
    probe_obj,
    probe_location: dict,
    logger,
    fig_dir: Path = None,
) -> list[pd.Series]:
    """
    Add the electrode data from the impedance and channel geometry files.
    Specific to Berke Lab custom probes

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled.
        filtering_info (str):
            The filtering applied to all channels.
        metadata (dict):
            Full metadata dictionary (from user-specified yaml)
        probe_metadata (dict):
            Metadata for this Probe read from resources/ephys_devices.yaml
        probe_obj (Probe):
            The Probe object added to the nwbfile
        logger (Logger):
            Logger to track conversion progress
        fig_dir (Path):
            Optional. The directory to save the figure. If None, the figure will not be saved.
    Returns:
        list[pd.Series]:
            List of information about channels excluded from the electrode table, if any.
            For now, we exclude the extra 4 channels that can be connected to ECoG screws in the 252-channel probes
    """

    # Get dataframe of electrode information (channel map, x/y coordinates, and impedance info)
    electrode_info = get_electrode_info(metadata=metadata, probe_type=probe_metadata["name"],
                                        logger=logger, fig_dir=fig_dir)

    # Save electrode info as an AssociatedFiles object in the NWB (and save a copy to the log directory)
    add_associated_file(
        nwbfile,
        name="electrode_info",
        description="Electrode channel map and impedance information (CSV)",
        content=electrode_info.to_csv(index=False),
        logger=logger,
        log_filename="electrode_info.csv",
    )

    # Get general metadata for the Probe
    # Axis convention: targeted_x is ML, targeted_y is AP, targeted_z is DV (see metadata_fully_explained.yaml).
    # ML must be in targeted_x because Spyglass derives the target hemisphere from its sign (>=0 Right, <0 Left).
    # These pass straight through to NwbElectrodeGroup below. We do not set the electrodes table's
    # absolute x/y/z columns (only per-electrode rel_x/rel_y on the probe), so Spyglass has no per-
    # electrode position and the target hemisphere depends solely on the sign of targeted_x.
    # Each probe has its OWN location - bilateral probes sit in opposite hemispheres.
    electrodes_location = probe_location["electrodes_location"]
    targeted_x = probe_location["targeted_x"]
    targeted_y = probe_location["targeted_y"]
    targeted_z = probe_location["targeted_z"]

    electrode_to_shank_map = {}
    electrode_groups_by_shank = {}

    # Process each Shank on the Probe
    for shank_dict in probe_metadata["shanks"]:
        for shank_index, shank_electrode_indices in shank_dict.items():
            shank = Shank(name=str(shank_index))
            logger.debug(f"Adding shank {shank_index} with electrodes {shank_electrode_indices}")

            # Make an ElectrodeGroup for this Shank and add it to the nwb
            electrode_group = NwbElectrodeGroup(
                name=str(shank_index),
                description=f"Electrodes on shank {shank_index}",
                location=electrodes_location,
                targeted_location=electrodes_location,
                targeted_x=float(targeted_x), # ML
                targeted_y=float(targeted_y), # AP
                targeted_z=float(targeted_z), # DV
                units="mm",
                device=probe_obj,
            )
            nwbfile.add_electrode_group(electrode_group)
            # Store the group so we can reference it when adding electrodes to the electrodes table
            electrode_groups_by_shank[shank_index] = electrode_group

            # Add each Shank and its ShanksElectrodes to the Probe 
            for electrode_index in shank_electrode_indices:
                # NOTE: We follow the ndx-franklab-novela/trodes-to-nwb usage of ShanksElectrode
                # even though it is redundant with NWB's electrode table fields, for consistency
                # with Frank Lab when using Spyglass
                shank.add_shanks_electrode(
                    ShanksElectrode(  
                        name=str(electrode_index),
                        rel_x=float(electrode_info.iloc[electrode_index]['x_um']),
                        rel_y=float(electrode_info.iloc[electrode_index]['y_um']),
                        rel_z=0.0,
                    )
                )
                electrode_to_shank_map[electrode_index] = shank_index
            probe_obj.add_shank(shank)

    # Add the electrode data to the NWB file, one column at a time
    nwbfile.add_electrode_column(
        name="electrode_name",
        description="The name of the electrode, in 'S(shank number)E(electrode number)' format",
    )
    nwbfile.add_electrode_column(
        name="intan_channel_number",
        description="The intan channel number (0-indexed) for the electrode",
    )
    nwbfile.add_electrode_column(
        name="open_ephys_channel_str",
        description="The Open Ephys channel name (1-based) for the electrode",
    )
    nwbfile.add_electrode_column(
        name="imp_file_channel_name",
        description="The name of the channel from the impedance file",
    )
    nwbfile.add_electrode_column(
        name="port",
        description="The port of the electrode from the impedance file",
    )
    nwbfile.add_electrode_column(
        name="imp",
        description="The impedance of the electrode (Impedance Magnitude at 1000 Hz (ohms))",
    )
    nwbfile.add_electrode_column(
        name="imp_phase",
        description="The phase of the impedance of the electrode (Impedance Phase at 1000 Hz (degrees))",
    )
    nwbfile.add_electrode_column(
        name="series_resistance_in_ohms",
        description="The series resistance of the electrode (Series RC equivalent R (Ohms))",
    )
    nwbfile.add_electrode_column(
        name="series_capacitance_in_farads",
        description="The series capacitance of the electrode (Series RC equivalent C (Farads))",
    )
    nwbfile.add_electrode_column(
        name="bad_channel",
        description="Whether the channel is a bad channel based on too low or too high impedance",
    )
    nwbfile.add_electrode_column(
        name="rel_x",
        description="The relative x coordinate of the electrode (um)",
    )
    nwbfile.add_electrode_column(
        name="rel_y",
        description="The relative y coordinate of the electrode (um)",
    )
    nwbfile.add_electrode_column(
        name="filtering",
        description="The filtering applied to the electrode",
    )
    nwbfile.add_electrode_column(
        name="probe_electrode",
        description= (
            "The index of the electrode on the probe (0-indexed). Equivalent to electrode number-1"
        ),
    )
    nwbfile.add_electrode_column(
        name="probe_shank",
        description="The index of the shank this electrode is on",
    )
    nwbfile.add_electrode_column(
        name="ref_elect_id",
        description=(
            "The index of the reference electrode in this table. "
            "-1 if not set. Used by Spyglass."
        ),
    )

    # TODO: Figure out an elegant(ish) way to handle ECoG screws.
    # For now I am excluding them entirely.

    # # Make an ElectrodeGroup for unconnected Intan channels (potentially used for ECoG screws)
    # # This is only needed for the 252-channel probes. 
    # # We may re-evaluate this later (they should maybe have a separate Probe object? This is a hack for now.)
    # extra_electrode_group = NwbElectrodeGroup(
    #     name="Other",
    #     description="Intan channels not connected to probe electrodes (may be fully unconnected or ECoG screws)",
    #     location="unknown",
    #     targeted_location="unknown",
    #     targeted_x=0.0,
    #     targeted_y=0.0,
    #     targeted_z=0.0,
    #     units="mm",
    #     device=probe_obj,
    # )
    # added_extra_electrode_group = False

    # Keep track of which channels we are excluding, if any.
    # For now, we exclude channels that have no ElectrodeGroup (aka the extra 4 channels on our 252ch probes)
    # Every row in the ElectricalSeries needs an associated electrode, 
    # so we need to keep track of the channels we exclude to exclude those rows later as well.
    excluded_channels = []

    for i, row in electrode_info.iterrows():
        # Get the Shank and ElectrodeGroup for this electrode
        shank_index = electrode_to_shank_map.get(i, -1)
        electrode_group = electrode_groups_by_shank.get(shank_index, None)

        # TODO: See above note on ECoG screws. Keeping this here for now:

        # # If we have no electrode_group, this is one of the 4 extra channels on the 252-ch probes.
        # # So add the extra_electrode_group to the nwb if we haven't already and assign this electrode to it
        # if electrode_group is None:
        #     if not added_extra_electrode_group:
        #         nwbfile.add_electrode_group(extra_electrode_group)
        #         added_extra_electrode_group = True
        #         logger.debug("Adding extra electrode group for unconnected Intan channels")
        #     electrode_group = extra_electrode_group

        if electrode_group is not None:
            nwbfile.add_electrode(
                electrode_name=row["electrode_name"],
                intan_channel_number=row["intan_channel"],
                open_ephys_channel_str=row["open_ephys_channel_string"],
                imp_file_channel_name=row["Channel Name"],
                port=row["Port"],
                imp=row["Impedance Magnitude at 1000 Hz (ohms)"],
                imp_phase=row["Impedance Phase at 1000 Hz (degrees)"],
                series_resistance_in_ohms=row["Series RC equivalent R (Ohms)"],
                series_capacitance_in_farads=row["Series RC equivalent C (Farads)"],
                bad_channel=row["bad_channel"],  # used by Spyglass
                rel_x=float(row["x_um"]),
                rel_y=float(row["y_um"]),
                group=electrode_group,
                location=electrodes_location, # same for all electrodes
                filtering=filtering_info, # same for all electrodes
                probe_electrode=i,  # used by Spyglass
                probe_shank=shank_index,  # used by Spyglass
                ref_elect_id=-1,  # used by Spyglass
            )
        else:
            logger.debug(f"Excluding '{row["electrode_name"]}' from electrode table:")
            logger.debug(row)
            excluded_channels.append(row)

    return excluded_channels


### Functions for Neuropixels

# Neuropixels 2.0 has 4 shanks with 1280 electrodes each, for 5120 total recording sites.
# The global electrode index (0..5119) encodes the shank: shank = electrode_index // 1280.
NPX_ELECTRODES_PER_SHANK = 1280


def _parse_np_probe_element(probe, tag: str, logger) -> dict:
    """
    Parse one per-channel element of an NP_PROBE (e.g. CHANNELS, ELECTRODE_INDEX, ELECTRODE_XPOS).

    These elements store one attribute per recording channel, keyed by channel name ("CH0", "CH1", ...).
    Returns a dict of {channel_num (int): value (str)} for the "CH#" attributes.
    """
    element = probe.find(tag)
    if element is None:
        logger.error(f"Could not find {tag} in the NP_PROBE in the settings.xml file.")
        raise ValueError(f"Could not find {tag} in the NP_PROBE in the settings.xml file.")
    return {int(name[2:]): value for name, value in element.attrib.items() if name.startswith("CH")}


def get_channel_map_neuropixels(settings_file_path, logger, fig_dir=None, probe_stream_names=None) -> dict:
    """
    Read the OpenEphys settings.xml for a Neuropixels 2.0 multishank probe and build a channel map:
    which physical electrode each recording channel (CH0..CH383) is wired to, plus its geometry.

    Each NP_PROBE element in settings.xml stores, per channel:
        CHANNELS        : "bank:shank" (we take the shank)
        ELECTRODE_INDEX : global electrode index (0..5119) = shank * 1280 + local_electrode
        ELECTRODE_XPOS / ELECTRODE_YPOS : physical position on the probe (um)

    We treat ELECTRODE_INDEX as the authoritative identity and merge it against our canonical coordinate
    table (neuropixels_2.0_multishank_electrode_coords.csv, one row per global electrode) to get the
    shank, shank_column, shank_row, and canonical x_um/y_um. We merge on the global electrode index rather
    than on x/y because the index is the authoritative identity - matching on float coordinates can break
    depending on the x or y value they are referenced to. (Our coords should matche the settings.xml 
    x/y positions exactly, which we check below)

    Parameters:
        settings_file_path (Path):
            Path to the OpenEphys settings.xml file
        logger (Logger):
            Logger to track conversion progress
        fig_dir (Path):
            Optional. The directory to save associated figures. If None, the figures will not be saved

    Returns:
        dict:
            probe_id -> DataFrame with one row per recording channel and columns:
            channel_num, channel_name, bank, shank, electrode, shank_column, shank_row, x_um, y_um
    """
    # Load canonical coordinates for all 5120 potential Neuropixels recording sites
    all_neuropixels_electrodes_info = pd.read_csv(ELECTRODE_COORDS_PATH_NPX_MULTISHANK)

    settings_root = ET.parse(settings_file_path).getroot()
    probes = settings_root.findall(".//NP_PROBE")
    if not probes:
        logger.error("Could not find any NP_PROBE elements in the settings.xml file.")
        raise ValueError("Could not find any NP_PROBE elements in the settings.xml file.")

    # Each NP_PROBE describes one recorded probe stream (e.g. "ProbeA", "ProbeB" for bilateral
    # recordings). Open Ephys lists the NP_PROBEs in hardware order and names the probe streams
    # alphabetically in that same order, so sort both and zip them to label each NP_PROBE with the
    # stream its data actually lands in. Without stream names (or if the counts disagree, which
    # would mean we misunderstand the recording) fall back to positional probe_0, probe_1, ... keys.
    probes = sorted(probes, key=lambda p: tuple(int(p.get(attr) or 0) for attr in ("slot", "port", "dock")))
    if probe_stream_names is not None and len(probe_stream_names) == len(probes):
        probe_labels = sorted(probe_stream_names)
        logger.info(f"Mapping {len(probes)} NP_PROBE(s) in settings.xml to probe streams {probe_labels}")
    else:
        if probe_stream_names is not None:
            logger.warning(f"Found {len(probes)} NP_PROBE(s) in settings.xml but {len(probe_stream_names)} "
                           f"probe stream(s) {probe_stream_names}! Falling back to positional probe names.")
        probe_labels = [f"probe_{i}" for i in range(len(probes))]

    # Parse each probe (there will be more than one for bilateral recordings)
    neuropixels_channel_data = {}
    for i, (probe, probe_id) in enumerate(zip(probes, probe_labels)):
        logger.debug(f"NP_PROBE {i} -> '{probe_id}': serial={probe.get('probe_serial_number')} "
                     f"slot={probe.get('slot')} port={probe.get('port')} dock={probe.get('dock')} "
                     f"electrodeConfigurationPreset='{probe.get('electrodeConfigurationPreset')}'")

        # Parse the per-channel elements, each keyed by channel number
        bank_shank = _parse_np_probe_element(probe, "CHANNELS", logger)  # "bank:shank"
        electrode_index = _parse_np_probe_element(probe, "ELECTRODE_INDEX", logger)
        xpos = _parse_np_probe_element(probe, "ELECTRODE_XPOS", logger)
        ypos = _parse_np_probe_element(probe, "ELECTRODE_YPOS", logger)

        channel_map = {}
        for channel_num, bank_shank_str in bank_shank.items():
            bank_str, shank_str = bank_shank_str.split(":")
            channel_map[channel_num] = {
                "channel_name": f"CH{channel_num}",
                "bank": int(bank_str),
                "shank": int(shank_str),
                "electrode": int(electrode_index[channel_num]),
                "x_um_settings": int(xpos[channel_num]),
                "y_um_settings": int(ypos[channel_num]),
            }

        channel_df = pd.DataFrame.from_dict(channel_map, orient="index")
        channel_df["channel_num"] = channel_df.index.astype(int)
        channel_df = channel_df.sort_values("channel_num").reset_index(drop=True)

        # Cross-check: the shank from CHANNELS should match the shank implied by the electrode index
        shank_from_electrode = channel_df["electrode"] // NPX_ELECTRODES_PER_SHANK
        if not (channel_df["shank"] == shank_from_electrode).all():
            logger.warning(f"{probe_id}: shank from CHANNELS disagrees with electrode_index // "
                           f"{NPX_ELECTRODES_PER_SHANK} for some channels!")

        # Merge with the canonical coordinate table on the global electrode index to validate and enrich.
        # This pulls in shank_column, shank_row, and canonical x_um/y_um (the 'shank' also comes from here)
        logger.debug(f"Validating electrode indices for Neuropixels {probe_id} against the coords table...")
        merged = channel_df.drop(columns=["shank"]).merge(
            all_neuropixels_electrodes_info[["electrode", "shank", "shank_column", "shank_row", "x_um", "y_um"]],
            on="electrode", how="left", validate="one_to_one",
        )

        # Check for electrode indices we couldn't match to a known recording site
        unmatched = merged[merged["shank_row"].isnull()]
        if len(unmatched) > 0:
            logger.error(f"{probe_id}: {len(unmatched)} channels have electrode indices not found in the "
                         f"coords table:\n{unmatched[['channel_num', 'electrode']]}")
            raise ValueError(f"{probe_id}: electrode indices not found in coords table: "
                             f"{sorted(unmatched['electrode'])}")

        # Sanity-check our canonical coords against what OpenEphys reports in settings.xml. 
        # They should match exactly, warn if a future probe config ever disagrees
        # Then drop the now-redundant settings columns to keep the channel info clean
        x_mismatch = (merged["x_um"] != merged["x_um_settings"]).sum()
        y_mismatch = (merged["y_um"] != merged["y_um_settings"]).sum()
        if x_mismatch or y_mismatch:
            logger.warning(f"{probe_id}: coords table disagrees with settings.xml for "
                           f"{x_mismatch} x and {y_mismatch} y positions - check the coords file!")
        merged = merged.drop(columns=["x_um_settings", "y_um_settings"])

        assert len(merged) == 384, f"Expected 384 channels, got {len(merged)}"
        logger.info(f"Neuropixels {probe_id}: all {len(merged)} channels matched to electrodes successfully!")
        logger.debug(f"{probe_id}: recording channels per shank: {merged.groupby('shank').size().to_dict()}")
        logger.debug(f"{probe_id}: electrode index range {int(merged['electrode'].min())}-"
                     f"{int(merged['electrode'].max())}, y range {int(merged['y_um'].min())}-"
                     f"{int(merged['y_um'].max())} um")
        neuropixels_channel_data[probe_id] = merged

        # Plot the electrode layout and channel info for this probe
        plot_neuropixels(all_neuropixels_electrodes=all_neuropixels_electrodes_info,
                         channel_info=merged,
                         probe_name=probe_id,
                         fig_dir=fig_dir)

    return neuropixels_channel_data


def get_session_folders(metadata: dict, logger) -> list:
    """
    Resolve the raw Open Ephys folder(s) to convert, in the order they should be concatenated.

    One recording is the common case. Two or more (e.g. a behavior session followed by an
    optotagging session) are concatenated into one ElectricalSeries, which is what lets them be
    spike sorted together. Each folder is labelled, and the labels become the nwb epoch tags -
    Spyglass turns each epoch tag into an interval list of that name, so these are the names you
    later restrict a sort to.

    Parameters:
        metadata (dict):
            Full metadata dictionary (from user-specified yaml)
        logger (Logger):
            Logger to track conversion progress

    Returns:
        list[dict]: one entry per recording, in concatenation order, with keys "path" and "session"
    """
    ephys_metadata = metadata["ephys"]

    if "openephys_folder_path" in ephys_metadata:
        logger.error("Found 'openephys_folder_path' in the ephys metadata.")
        raise ValueError("Found 'openephys_folder_path' in the ephys metadata. This became "
                         "'openephys_folder_paths', a list of recordings to concatenate, each with "
                         "a 'path' and a 'session' label.")

    session_folders = ephys_metadata.get("openephys_folder_paths")
    if not session_folders:
        logger.error("No 'openephys_folder_paths' in the ephys metadata.")
        raise ValueError("No 'openephys_folder_paths' in the ephys metadata. List each Open Ephys "
                         "recording to convert, with its 'path' and 'session' label.")

    resolved_folders = []
    for session_folder in session_folders:
        missing = [field for field in ("path", "session") if not session_folder.get(field)]
        if missing:
            logger.error(f"Open Ephys recording entry is missing {missing}: {session_folder}")
            raise ValueError(f"Open Ephys recording entry is missing {missing}. Every recording "
                             "needs a 'path' and a 'session' label.")
        resolved_folders.append({"path": session_folder["path"], "session": session_folder["session"]})

    session_labels = [folder["session"] for folder in resolved_folders]
    if len(set(session_labels)) != len(session_labels):
        logger.error(f"Duplicate session labels {session_labels}")
        raise ValueError(f"Duplicate session labels {session_labels}. Each recording needs a "
                         "distinct label, because the labels name the nwb epochs.")

    logger.info(f"Converting {len(resolved_folders)} Open Ephys recording(s) in order: {session_labels}")
    return resolved_folders


def get_probe_configs(metadata: dict, probe_file_paths: dict, logger) -> list:
    """
    Resolve which probes to convert and where each one is implanted.

    metadata["ephys"]["probes"] is the only place probes are specified: listing a probe both
    selects it for conversion and gives its implant location. There is no fallback - a probe that
    is not listed is not converted, and a listed probe must carry its own location.

    Each probe needs its own location because Spyglass derives the target hemisphere from the SIGN
    of targeted_x, so bilateral probes sharing one location would land in the same hemisphere.

    'stream' names the Open Ephys probe stream (e.g. "ProbeA"). It is required when the recording
    holds more than one probe, and may be omitted when there is only one.

    Parameters:
        metadata (dict):
            Full metadata dictionary (from user-specified yaml)
        probe_file_paths (dict):
            Open Ephys probe stream name -> continuous.dat path, for the streams in the recording
        logger (Logger):
            Logger to track conversion progress

    Returns:
        list[dict]: one entry per probe to convert, in the order given, with keys
        "stream" (the full Open Ephys stream name), "label" (the short name, e.g. "ProbeA"),
        "electrodes_location", "targeted_x", "targeted_y", "targeted_z"
    """
    ephys_metadata = metadata["ephys"]

    # The location used to live directly under "ephys". Fail loudly on an un-migrated file rather
    # than converting it with no location at all.
    outdated_fields = [
        field for field in ("electrodes_location", "targeted_x", "targeted_y", "targeted_z", "probe")
        if field in ephys_metadata
    ]
    if outdated_fields:
        logger.error(f"Found {outdated_fields} directly under 'ephys'. These moved into the "
                     "'probes' list, where each probe carries its own location.")
        raise ValueError(f"Found {outdated_fields} directly under 'ephys' in the metadata. These "
                         "moved into the 'probes' list - give each probe its own entry with "
                         "electrodes_location and targeted_x/y/z.")

    probe_entries = ephys_metadata.get("probes")
    if not probe_entries:
        logger.error("No 'probes' in the ephys metadata.")
        raise ValueError("No 'probes' in the ephys metadata. List each probe to convert, with its "
                         "electrodes_location and targeted_x/y/z.")

    # 'stream' may only be omitted when the recording holds exactly one probe
    streams_given = ["stream" in entry for entry in probe_entries]
    if len(probe_entries) > 1 and not all(streams_given):
        logger.error(f"{len(probe_entries)} probes listed but not all give a 'stream'.")
        raise ValueError(f"{len(probe_entries)} probes are listed in the metadata, so each one must "
                         f"name its 'stream' (one of {sorted(probe_file_paths)}).")

    probe_configs = []
    for probe_entry in probe_entries:
        requested_stream = probe_entry.get("stream")
        if requested_stream is None:
            if len(probe_file_paths) != 1:
                logger.error(f"No 'stream' given but the recording holds {len(probe_file_paths)} "
                             f"probe streams: {sorted(probe_file_paths)}")
                raise ValueError(f"The recording holds {len(probe_file_paths)} probe streams "
                                 f"{sorted(probe_file_paths)}, so the probe must name its 'stream'.")
            stream = next(iter(probe_file_paths))
        else:
            matches = [name for name in probe_file_paths
                       if probe_name_token(name) == probe_name_token(requested_stream)]
            if len(matches) != 1:
                logger.error(f"Expected one probe stream matching '{requested_stream}', found "
                             f"{matches} among {sorted(probe_file_paths)}")
                raise ValueError(f"Expected one probe stream matching '{requested_stream}', found "
                                 f"{matches} among {sorted(probe_file_paths)}.")
            stream = matches[0]

        missing_fields = [
            field for field in
            ("probe_type", "electrodes_location", "targeted_x", "targeted_y", "targeted_z")
            if probe_entry.get(field) is None
        ]
        if missing_fields:
            logger.error(f"Probe '{stream}' is missing {missing_fields}")
            raise ValueError(f"Probe '{stream}' is missing {missing_fields}. Every probe carries its "
                             "own type and location, because bilateral probes are in opposite hemispheres.")

        probe_configs.append({
            "stream": stream,
            "label": probe_label_for_stream(requested_stream=requested_stream, resolved_stream=stream),
            "probe_type": probe_entry["probe_type"],
            "electrodes_location": probe_entry["electrodes_location"],
            "targeted_x": probe_entry["targeted_x"],
            "targeted_y": probe_entry["targeted_y"],
            "targeted_z": probe_entry["targeted_z"],
        })

    # One conversion handles one kind of probe: the Berke and Neuropixels paths read different
    # files and find port visits in different places, so a mix of types cannot be converted together
    probe_types = {config["probe_type"] for config in probe_configs}
    if len(probe_types) > 1:
        logger.error(f"Probes have different types: {sorted(probe_types)}")
        raise ValueError(f"Probes have different types ({sorted(probe_types)}). A single conversion "
                         "handles one probe type.")

    for probe_config in probe_configs:
        logger.info(f"Converting probe '{probe_config['label']}' ({probe_config['probe_type']}, stream "
                    f"'{probe_config['stream']}') at "
                    f"{probe_config['electrodes_location']}, ML={probe_config['targeted_x']} "
                    f"AP={probe_config['targeted_y']} DV={probe_config['targeted_z']}")
    return probe_configs


def add_electrode_data_neuropixels(
    *,
    nwbfile: NWBFile,
    channel_info: pd.DataFrame,
    filtering_info: str,
    probe_obj,
    probe_location: dict,
    probe_label: str,
    logger,
    prefix_names: bool = False,
) -> None:
    """
    Add electrode groups and the electrode table for a Neuropixels probe.

    Analogous to add_electrode_data_berke_probe, but simpler: Neuropixels has no per-session impedance
    file, so we don't mark bad channels here (all channels are treated as good). We create one
    ElectrodeGroup per shank that has recording sites, and add one electrode per recording channel.

    We register the FULL probe geometry (all 5120 sites) on the Probe's Shank/ShanksElectrode objects,
    even though only 384 are recorded this session. Spyglass builds Probe.Electrode from these under a
    single shared probe_id, so if we only registered this session's 384 sites, a later session recording
    a different 384-subset would fail the Electrode -> Probe.Electrode foreign key. The electrodes table
    itself still gets ONLY the 384 recorded channels. Non-recorded sites live only on the Probe object.

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled.
        channel_info (pd.DataFrame):
            One row per recording channel (from get_channel_map_neuropixels), with columns
            channel_num, shank, electrode, shank_column, shank_row, x_um, y_um
        filtering_info (str):
            The filtering applied to all channels (Neuropixels applies hardware filtering)
        probe_obj (Probe):
            The Probe object added to the nwbfile
        logger (Logger):
            Logger to track conversion progress
    """
    # Save electrode info as an AssociatedFiles object in the NWB (and save a copy to the log directory).
    # The name includes the probe because this runs once per probe and AssociatedFiles names must be unique.
    electrode_info_name = f"electrode_info_{probe_label}" if prefix_names else "electrode_info"
    add_associated_file(
        nwbfile,
        name=electrode_info_name,
        description=(f"Neuropixels channel map for probe {probe_label} "
                     "(recording channel -> electrode geometry) (CSV)" if prefix_names else
                     "Neuropixels channel map (recording channel -> electrode geometry) (CSV)"),
        content=channel_info.to_csv(index=False),
        logger=logger,
        log_filename=f"{electrode_info_name}.csv",
    )

    # General metadata for the electrode groups
    # Axis convention: targeted_x is ML, targeted_y is AP, targeted_z is DV (see metadata_fully_explained.yaml).
    # ML must be in targeted_x because Spyglass derives the target hemisphere from its sign (>=0 Right, <0 Left).
    # These pass straight through to NwbElectrodeGroup below. We do not set the electrodes table's
    # absolute x/y/z columns (only per-electrode rel_x/rel_y on the probe), so Spyglass has no per-
    # electrode position and the target hemisphere depends solely on the sign of targeted_x.
    # Each probe has its OWN location - bilateral probes sit in opposite hemispheres.
    electrodes_location = probe_location["electrodes_location"]
    targeted_x = probe_location["targeted_x"]
    targeted_y = probe_location["targeted_y"]
    targeted_z = probe_location["targeted_z"]

    # Make an ElectrodeGroup for each shank that actually has recording sites. Only recorded electrodes
    # reference an ElectrodeGroup; non-recorded sites live only on the Probe geometry (registered below).
    # NOTE: we group one ElectrodeGroup per shank (mirrors the Berke probe per-shank structure).
    # We might later change to an electrode group per contiguous y block.
    # With more than one probe the shank numbers repeat, so prefix the group name with the probe
    # stream ("ProbeA_0"). Spyglass stores electrode_group_name as a varchar, so a non-numeric name
    # is fine here (unlike Shank/ShanksElectrode names, which Spyglass casts to int).
    electrode_groups_by_shank = {}
    for shank_index in sorted(channel_info["shank"].unique()):
        group_name = f"{probe_label}_{shank_index}" if prefix_names else str(shank_index)
        group_description = (f"Electrodes on shank {shank_index} of probe {probe_label}"
                             if prefix_names else f"Electrodes on shank {shank_index}")
        electrode_group = NwbElectrodeGroup(
            name=group_name,
            description=group_description,
            location=electrodes_location,
            targeted_location=electrodes_location,
            targeted_x=float(targeted_x), # ML
            targeted_y=float(targeted_y), # AP
            targeted_z=float(targeted_z), # DV
            units="mm",
            device=probe_obj,
        )
        nwbfile.add_electrode_group(electrode_group)
        electrode_groups_by_shank[shank_index] = electrode_group
        logger.debug(f"Adding electrode group '{group_name}'")

    # Register the FULL probe geometry (all 5120 sites), not just this session's 384 recorded channels,
    # on the Probe's Shank / ShanksElectrode objects (see docstring).
    # ShanksElectrode.name is the GLOBAL electrode index (0..5119), matching probe_electrode on the
    # electrodes table so Spyglass's Electrode -> Probe.Electrode join holds for every possible channel map.
    all_probe_sites = pd.read_csv(ELECTRODE_COORDS_PATH_NPX_MULTISHANK)
    for shank_index in sorted(all_probe_sites["shank"].unique()):
        shank = Shank(name=str(shank_index))
        shank_sites = all_probe_sites[all_probe_sites["shank"] == shank_index].sort_values("electrode")
        for _, site in shank_sites.iterrows():
            shank.add_shanks_electrode(
                ShanksElectrode(
                    name=str(int(site["electrode"])),  # GLOBAL index (0..5119), matches probe_electrode
                    rel_x=float(site["x_um"]),
                    rel_y=float(site["y_um"]),
                    rel_z=0.0,
                )
            )
        probe_obj.add_shank(shank)
    logger.info(f"Registered full Neuropixels probe geometry on the Probe: {len(all_probe_sites)} sites "
                f"across {all_probe_sites['shank'].nunique()} shanks "
                f"({len(channel_info)} of them recorded this session).")

    # Add the electrode data columns (mirrors the Berke table, minus the impedance columns).
    # This function runs once per probe, so only add each column the first time -
    # pynwb raises if the same column is added twice.
    existing_electrode_columns = set(nwbfile.electrodes.colnames) if nwbfile.electrodes is not None else set()

    def add_electrode_column_once(name, description):
        if name not in existing_electrode_columns:
            nwbfile.add_electrode_column(name=name, description=description)

    add_electrode_column_once(
        name="electrode_name",
        description="The name of the electrode, in 'S(shank number)E(electrode number)' format",
    )
    add_electrode_column_once(
        name="intan_channel_number",
        description="The recording channel number (0-indexed) for the electrode (CH0..CH383 -> 0..383)",
    )
    add_electrode_column_once(
        name="electrode_index",
        description="The global Neuropixels electrode index (0-indexed, 0..5119)",
    )
    add_electrode_column_once(
        name="bad_channel",
        description="Whether the channel is a bad channel. Always False for Neuropixels (no impedance file)",
    )
    add_electrode_column_once(
        name="rel_x",
        description="The relative x coordinate of the electrode (um)",
    )
    add_electrode_column_once(
        name="rel_y",
        description="The relative y coordinate of the electrode (um)",
    )
    add_electrode_column_once(
        name="filtering",
        description="The filtering applied to the electrode",
    )
    add_electrode_column_once(
        name="probe_electrode",
        description="The global index of the electrode on the probe (0-indexed)",
    )
    add_electrode_column_once(
        name="probe_shank",
        description="The index of the shank this electrode is on",
    )
    add_electrode_column_once(
        name="probe",
        description="The probe (Open Ephys stream name) this electrode belongs to",
    )
    add_electrode_column_once(
        name="ref_elect_id",
        description="The index of the reference electrode in this table. -1 if not set. Used by Spyglass.",
    )

    # Add the electrodes table rows for ONLY the 384 recorded channels (the full probe geometry, including
    # non-recorded sites, was already registered on the Probe above). Add them in channel_num order (0..383)
    # so the electrode table row -> raw data row mapping (built downstream from intan_channel_number) is correct.
    for _, row in channel_info.sort_values("channel_num").iterrows():
        shank_index = int(row["shank"])
        electrode_index = int(row["electrode"])
        local_electrode = electrode_index % NPX_ELECTRODES_PER_SHANK

        nwbfile.add_electrode(
            electrode_name=f"S{shank_index:02d}E{local_electrode}",
            probe=probe_label,
            intan_channel_number=int(row["channel_num"]),
            electrode_index=electrode_index,
            bad_channel=False,  # no impedance file for Neuropixels, so no channels are marked bad
            rel_x=float(row["x_um"]),
            rel_y=float(row["y_um"]),
            group=electrode_groups_by_shank[shank_index],
            location=electrodes_location,  # same for all electrodes
            filtering=filtering_info,  # same for all electrodes
            probe_electrode=electrode_index,  # GLOBAL index (0..5119), matches ShanksElectrode.name. Used by Spyglass
            probe_shank=shank_index,  # used by Spyglass
            ref_elect_id=-1,  # used by Spyglass
        )

    logger.info(f"Added {len(channel_info)} Neuropixels electrodes across {len(electrode_groups_by_shank)}"
                f"shank ElectrodeGroup(s): {sorted(electrode_groups_by_shank)}")


### Functions for preprocessed (optonpx) ephys data

# The preprocessing pipeline writes its per-probe folders in lower camel case ("probeA") while
# Open Ephys names the streams "ProbeA", so probe names are matched case-insensitively.
def add_raw_ephys(
    *,
    nwbfile: NWBFile,
    metadata: dict,
    logger,
    fig_dir: Path = None,
) -> dict:
    """Add the raw ephys data to a NWB file.

    Returns a dict with the Open Ephys start time and detected port visits (empty dict if there is no
    ephys metadata for this session, or if the required ephys metadata subfields are missing).

    Parameters:
        nwbfile (NWBFile):
            The NWB file being assembled.
        metadata (dict):
            Metadata dictionary
        logger (Logger):
            Logger to track conversion progress
        fig_dir (Path):
            Optional. The directory to save associated figures. If None, the figures will not be saved.
    """

    if "ephys" not in metadata:
        log_and_print(logger, "No ephys metadata found for this session. Skipping ephys conversion.", level="info")
        return {}

    # If we do have "ephys" in metadata, check for the required keys.
    # (impedance_file_path is required for Berke Lab probes but not Neuropixels, so it is checked
    # in the Berke branch below rather than here.)
    required_ephys_keys = {"openephys_folder_paths", "probes"}
    missing_keys = required_ephys_keys - metadata["ephys"].keys()
    if missing_keys:
        log_and_print(
            logger,
            "The required ephys subfields do not exist in the metadata dictionary.\n"
            "Remove the 'ephys' field from metadata if you do not have ephys data "
            f"for this session, \nor specify the following missing subfields: {missing_keys}",
            level="warning",
        )
        return {}

    log_and_print(logger, "Adding raw ephys...", level="info")

    # The recording(s) to convert, in concatenation order
    session_folders = get_session_folders(metadata=metadata, logger=logger)
    session_time_offsets = get_session_time_offsets(session_folders=session_folders, logger=logger)
    # The first recording defines the session: its settings.xml gives the channel map, its ADC
    # gives the port visits, and its bonsai start is time zero
    openephys_folder_path = session_folders[0]["path"]

    # Get Open Ephys start time as datetime object based on the time specified in the folder name
    openephys_folder_name = openephys_folder_path.split('/')[-1]
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", openephys_folder_name)
    if match:
        datetime_str = match.group(0)
        open_ephys_start = datetime.strptime(datetime_str, "%Y-%m-%d_%H-%M-%S")
    else:
        raise ValueError(f"No valid datetime string found in path: {openephys_folder_path}")

    open_ephys_start = open_ephys_start.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    logger.info(f"Open Ephys start time: {open_ephys_start}")

    # Get paths to the settings.xml file and the continuous.dat files.
    # The experiment number is auto-detected by default, but can be overridden via metadata for the
    # rare case of multiple experiments recorded into the same Open Ephys output folder.
    # This happens when you start and stop the ephys recording multiple times.
    experiment_number = metadata["ephys"].get("experiment_number", None)
    # Paths for every recording, so a disagreement between them is caught before anything is written
    session_paths = [
        find_open_ephys_paths(open_ephys_folder_path=session_folder["path"],
                              experiment_number=experiment_number, logger=logger)
        for session_folder in session_folders
    ]
    open_ephys_paths_dict = find_open_ephys_paths(open_ephys_folder_path=openephys_folder_path,
                                                  experiment_number=experiment_number, logger=logger)
    settings_file_path = open_ephys_paths_dict["settings_file"]
    recording_file_paths_dict = open_ephys_paths_dict["recording_files"]

    # Separate the neural data stream(s) from the ADC stream. For Berke Lab probes, the ADC channels live
    # inside the single probe continuous.dat (so there is no separate ADC stream). For Neuropixels (OneBox),
    # the ADC is its own stream/folder (e.g. "OneBox-100.OneBox-ADC") separate from the probe ("...ProbeA").
    adc_file_paths = {name: path for name, path in recording_file_paths_dict.items() if "ADC" in name.upper()}
    probe_file_paths = {name: path for name, path in recording_file_paths_dict.items() if "ADC" not in name.upper()}

    if not probe_file_paths:
        logger.error(f"No probe (non-ADC) continuous.dat found in {openephys_folder_path}")
        raise ValueError(f"No probe (non-ADC) continuous.dat found in {openephys_folder_path}")

    # Which probes to convert, and where each is implanted, both come from the 'probes' metadata.
    # Multiple probes are combined into ONE ElectricalSeries (Spyglass reads a single acquisition).
    probe_configs = get_probe_configs(metadata=metadata, probe_file_paths=probe_file_paths, logger=logger)
    probe_streams_to_convert = {config["stream"]: probe_file_paths[config["stream"]] for config in probe_configs}
    probe_labels = [config["label"] for config in probe_configs]
    probe_locations_by_label = {config["label"]: config for config in probe_configs}
    multiple_probes = len(probe_configs) > 1

    # Labels for EVERY probe in the recording, not just the ones being converted. settings.xml
    # describes all of them, so the NP_PROBE -> stream mapping is only correct if we hand over the
    # full set. Converting a subset (e.g. just ProbeB of a bilateral recording) selects from it.
    all_probe_labels = sorted(name.split(".")[-1] for name in probe_file_paths)

    for stream_name, continuous_dat_file_path in probe_streams_to_convert.items():
        logger.info(f"Found probe continuous.dat for '{stream_name}' at {continuous_dat_file_path}")

    logger.info("Adding probe(s)...")
    probe_objs_by_label = {}
    for probe_index, probe_label in enumerate(probe_labels):
        # Only suffix the device name when there is more than one probe, so single-probe nwb files
        # keep exactly the device and electrode group names they had before
        probe_metadata, probe_obj = add_probe_info(
            nwbfile=nwbfile,
            probe_type=probe_locations_by_label[probe_label]["probe_type"],
            logger=logger,
            probe_label=probe_label if multiple_probes else None,
            probe_id=probe_index,
        )
        probe_objs_by_label[probe_label] = probe_obj

    # Read recording params (channel count, sample rate, conversion factor) from structure.oebin.
    # This works the same for Berke lab probes and Neuropixels. Every probe sharing one
    # ElectricalSeries must agree on these, so check rather than assume.
    oebin_params_per_probe = {
        probe_label: read_oebin_params(continuous_dat_file_path=path, logger=logger)
        for probe_label, path in zip(probe_labels, probe_streams_to_convert.values())
    }
    for param in ("channel_count", "sample_rate", "bit_volts"):
        values = {label: params[param] for label, params in oebin_params_per_probe.items()}
        if len(set(values.values())) != 1:
            logger.error(f"Probe streams disagree on {param}: {values}")
            raise ValueError(f"Probe streams disagree on {param}: {values}. "
                             "They cannot share a single ElectricalSeries.")
    # Every recording must agree too: the series has one channel count, one sample rate and one
    # conversion factor, so a mismatch between recordings would silently misdescribe part of it.
    for session_folder, paths in list(zip(session_folders, session_paths))[1:]:
        for probe_config in probe_configs:
            stream = probe_config["stream"]
            if stream not in paths["recording_files"]:
                logger.error(f"Recording '{session_folder['session']}' has no probe stream '{stream}'")
                raise ValueError(f"Recording '{session_folder['session']}' has no probe stream "
                                 f"'{stream}'. Every recording must hold every probe being converted.")
            other_params = read_oebin_params(
                continuous_dat_file_path=paths["recording_files"][stream], logger=logger
            )
            for param in ("channel_count", "sample_rate", "bit_volts"):
                first = oebin_params_per_probe[probe_config["label"]][param]
                if other_params[param] != first:
                    logger.error(f"'{probe_config['label']}' {param} differs between recordings: "
                                 f"{session_folders[0]['session']}={first}, "
                                 f"{session_folder['session']}={other_params[param]}")
                    raise ValueError(
                        f"'{probe_config['label']}' {param} differs between recordings "
                        f"({session_folders[0]['session']}={first}, "
                        f"{session_folder['session']}={other_params[param]}). They cannot share "
                        "one ElectricalSeries."
                    )

    oebin_params = next(iter(oebin_params_per_probe.values()))
    total_channels = oebin_params["channel_count"]

    # Berke Lab probes and Neuropixels store their electrode info and port visits differently, so each
    # branch sets up its electrode table and tells us where to find the port visit pulses:
    #   port_visits_* describe the continuous.dat file/channel/rate that the port visit pulses live on
    #   pulse_high_threshold is the raw int16 value above which the port visit channel is considered "high"
    #   exclude_channel_names are neural channels to drop from the ElectricalSeries (Berke ECoG screws)
    if probe_metadata["name"] in BERKE_LAB_PROBES:
        log_and_print(logger, f"Processing '{probe_metadata['name']}' as a Berke Lab custom probe.", level="info")

        # Berke Lab probes require a per-session impedance file (used to mark bad channels)
        if "impedance_file_path" not in metadata["ephys"]:
            raise ValueError("Berke Lab probes require 'impedance_file_path' in the ephys metadata.")

        # Sanity-check the oebin channels against the expected 256 "CH" + 8 "ADC" layout
        expected_channel_names = {f"CH{i}" for i in range(1, 257)} | {f"ADC{i}" for i in range(1, 9)}
        validate_oebin_channel_names(oebin_params["channel_names"], expected_channel_names, logger)

        # For Berke Lab probes, port visits are recorded on ADC1, which lives in the same continuous.dat
        # as the neural data channels (256 "CH" + 8 "ADC" = 264 total channels)
        port_visits_dat_file_path = continuous_dat_file_path
        port_visits_total_channels = total_channels
        port_visits_sample_rate = oebin_params["sample_rate"]
        port_visits_channel_num = 256  # Port visits are recorded on ADC1, aka channel 256 (zero-indexed)
        pulse_high_threshold = 10_000  # raw int16 threshold

        # Read the settings.xml file to get filtering info
        filtering_info = read_open_ephys_settings_xml(settings_file_path=settings_file_path, logger=logger)

        # Create electrode groups and add electrode data to the NWB file
        excluded_channels = add_electrode_data_berke_probe(
                                    nwbfile=nwbfile,
                                    filtering_info=filtering_info,
                                    metadata=metadata,
                                    probe_metadata=probe_metadata,
                                    probe_obj=next(iter(probe_objs_by_label.values())),
                                    probe_location=probe_configs[0],
                                    logger=logger,
                                    fig_dir=fig_dir,
                                )

        # Get Open Ephys names of channels to exclude for use by OpenEphysBinaryRecordingExtractor
        exclude_channel_names = [row["open_ephys_channel_string"] for row in excluded_channels]

    else:
        log_and_print(logger, f"Processing '{probe_metadata['name']}' as a Neuropixels probe.", level="info")

        # For Neuropixels (OneBox), port visits are recorded on a SEPARATE OneBox-ADC stream (its own
        # continuous.dat), not in the probe's continuous.dat. Find that ADC stream and read its params.
        if not adc_file_paths:
            raise ValueError("No OneBox-ADC stream found for Neuropixels; cannot read port visits.")
        if len(adc_file_paths) > 1:
            raise NotImplementedError(f"Expected exactly one ADC stream, found: {adc_file_paths}")
        adc_stream_name, adc_dat_file_path = next(iter(adc_file_paths.items()))
        logger.info(f"Found OneBox-ADC continuous.dat for '{adc_stream_name}' at {adc_dat_file_path}")
        adc_oebin_params = read_oebin_params(continuous_dat_file_path=adc_dat_file_path, logger=logger)

        # ADC data is stored as raw int16 like everything else, so convert our desired volts threshold 
        # to raw int16 using the ADC bit_volts. 
        # Port visits default to ADC2 (ADC0 is a 1 Hz sync clock, and ADC1 will be used for future optotagging)
        # Both the channel and threshold are overridable via metadata
        port_visits_dat_file_path = adc_dat_file_path
        port_visits_total_channels = adc_oebin_params["channel_count"]
        port_visits_sample_rate = adc_oebin_params["sample_rate"]
        port_visits_channel_num = metadata["ephys"].get("port_visits_adc_channel", NPX_DEFAULT_PORT_VISITS_ADC_CHANNEL)
        threshold_volts = metadata["ephys"].get("port_visits_threshold_volts", NPX_PORT_VISITS_THRESHOLD_VOLTS)
        pulse_high_threshold = threshold_volts / adc_oebin_params["bit_volts"]
        logger.info(f"Reading Neuropixels port visits from ADC channel {port_visits_channel_num} "
                    f"with threshold {threshold_volts} V ({pulse_high_threshold:.0f} raw int16)")

        # Neuropixels applies hardware filtering (not recorded in settings.xml like the Berke bandpass)
        filtering_info = "Neuropixels 2.0 hardware filtering (see IMEC Neuropixels documentation)"

        # Build the channel map for each probe, then create electrode groups and add electrode data.
        channel_map = get_channel_map_neuropixels(settings_file_path=settings_file_path,
                                                  logger=logger, fig_dir=fig_dir,
                                                  probe_stream_names=all_probe_labels)
        missing_channel_maps = [label for label in probe_labels if label not in channel_map]
        if missing_channel_maps:
            logger.error(f"No channel map for probe stream(s) {missing_channel_maps}. "
                         f"settings.xml gave channel maps for {list(channel_map)}.")
            raise ValueError(f"No channel map for probe stream(s) {missing_channel_maps}. "
                             f"settings.xml gave channel maps for {list(channel_map)}.")

        # Add electrodes probe by probe, in the SAME order the probes are combined into the
        # ElectricalSeries below, so that electrode table row i is channel i of the series.
        for probe_label in probe_labels:
            add_electrode_data_neuropixels(
                nwbfile=nwbfile,
                channel_info=channel_map[probe_label],
                filtering_info=filtering_info,
                probe_obj=probe_objs_by_label[probe_label],
                probe_location=probe_locations_by_label[probe_label],
                probe_label=probe_label,
                logger=logger,
                # Keep single-probe files naming their groups "0".."3" as they always have
                prefix_names=multiple_probes,
            )

        # All Neuropixels recording channels are kept (no ECoG-screw channels to exclude)
        exclude_channel_names = []

    # Get port visits recorded by Open Ephys for timestamp alignment
    logger.info("Getting port visits recorded by Open Ephys...")
    logger.debug(f"Port visit source: channel {port_visits_channel_num} of {port_visits_dat_file_path} "
                 f"({port_visits_total_channels} channels @ {port_visits_sample_rate} Hz, "
                 f"threshold {pulse_high_threshold:.0f} raw int16)")
    ephys_visit_times, bonsai_start_time = get_port_visits(continuous_dat_file_path=port_visits_dat_file_path,
                                                           total_channels=port_visits_total_channels,
                                                           port_visits_channel_num=port_visits_channel_num,
                                                           sample_rate=port_visits_sample_rate,
                                                           logger=logger,
                                                           pulse_high_threshold=pulse_high_threshold)
    log_and_print(logger, f"Open Ephys recorded {len(ephys_visit_times)} port visits.", level="info")
    logger.debug(f"Open Ephys port visits: {ephys_visit_times}")

    # Convert the bonsai start time (seconds) to a number of samples in the probe's sample rate.
    # This matters for Neuropixels: port visits come from the OneBox-ADC (30300.5 Hz) but we trim the
    # probe data (30000 Hz), so we need to convert
    probe_sample_rate = oebin_params["sample_rate"]
    samples_to_remove = round(bonsai_start_time * probe_sample_rate)
    logger.debug(f"Trimming {samples_to_remove} samples ({bonsai_start_time}s at {probe_sample_rate} Hz) "
                 "from the start of the probe data")

    # Build each recording, then join them end to end. Within a session the probes are aligned and
    # combined channel-wise; across sessions the recordings are concatenated with the real gap kept
    # in the timestamps. Everything ends up in ONE ElectricalSeries, which is what Spyglass reads.
    probe_streams_by_label = dict(zip(probe_labels, probe_streams_to_convert))
    session_labels = [session_folder["session"] for session_folder in session_folders]

    # The traces can come from the preprocessed files instead of the raw ones. Everything else
    # still comes from raw: the ADC gives the port visits, settings.xml the channel map, and the
    # per-stream timestamps the start times the probes are aligned by.
    processed_ephys_path = metadata["ephys"].get("processed_ephys_path")
    session_recordings = []
    session_timestamps = []
    num_samples_by_probe = {}

    if processed_ephys_path:
        # Imported here, not at the top: convert_processed_ephys imports helpers from this module,
        # so a module-level import either way round would be circular.
        from .convert_processed_ephys import (
            get_processed_session_recordings,
            read_session_boundaries,
            verify_processed_matches_raw,
        )

        log_and_print(logger, f"Writing PREPROCESSED traces from {processed_ephys_path} "
                              "(laser artifacts and common-mode noise interpolated over, phase "
                              "shift applied) instead of the raw traces.", level="info")
        probe_start_times = get_raw_probe_start_times(
            session_folders=session_folders, probe_streams_by_label=probe_streams_by_label, logger=logger
        )
        # Nothing in the preprocessed file says its channels are in the raw order, and if they are
        # not then every electrode maps to the wrong trace. Check it against the raw recording.
        first_probe_label, first_probe_stream = next(iter(probe_streams_by_label.items()))
        first_raw_recording, _factor = open_probe_stream_recording(
            folder_path=session_folders[0]["path"], logger=logger,
            stream_name=first_probe_stream, samples_to_remove=0,
        )
        # The preprocessed binary carries no scale of its own; preprocessing preserves the raw one
        channel_conversion_factor_uv = oebin_params["bit_volts"]
        processed_session_boundaries = read_session_boundaries(
            processed_ephys_path=processed_ephys_path, logger=logger
        )
        processed_session_recordings_result = get_processed_session_recordings(
            session_boundaries=processed_session_boundaries,
            processed_ephys_path=processed_ephys_path,
            probe_streams_by_label=probe_streams_by_label,
            session_labels=session_labels,
            probe_start_times=probe_start_times,
            conversion_factor_uv=channel_conversion_factor_uv,
            samples_to_remove=samples_to_remove,
            logger=logger,
        )
        (session_recordings, session_timestamps, num_samples_by_probe,
         processed_sessions_by_probe) = processed_session_recordings_result

        # Compare the first recording's first probe against the raw traces it came from, reusing
        # the recording that was already opened above
        verify_processed_matches_raw(
            processed_recording=processed_sessions_by_probe[first_probe_label][session_labels[0]],
            raw_recording=first_raw_recording,
            logger=logger,
        )
    else:
        processed_session_boundaries = None
        probe_start_times = None
        (session_recordings, session_timestamps, num_samples_by_probe,
         raw_conversion_factor_uv) = get_raw_session_recordings(
            session_folders=session_folders,
            probe_streams_by_label=probe_streams_by_label,
            exclude_channel_names=exclude_channel_names,
            samples_to_remove=samples_to_remove,
            logger=logger,
        )
        channel_conversion_factor_uv = raw_conversion_factor_uv

    combined_recording, ephys_timestamps_unaligned, session_epochs = concatenate_session_recordings(
        session_recordings=session_recordings,
        session_timestamps=session_timestamps,
        session_offsets=session_time_offsets,
        session_labels=[folder["session"] for folder in session_folders],
        logger=logger,
    )
    traces_as_iterator = SpikeInterfaceRecordingDataChunkIterator(recording=combined_recording)
    original_timestamps = ephys_timestamps_unaligned
    num_samples, num_channels = traces_as_iterator.maxshape

    # Check that the number of electrodes in the NWB file is the same as the number of channels in traces_as_iterator
    assert (len(nwbfile.electrodes) == num_channels), (
        f"Number of electrodes in NWB file ({len(nwbfile.electrodes)}) does not match number of channels "
        f"in traces_as_iterator ({num_channels})."
    )
    logger.debug(f"There are {len(nwbfile.electrodes)} electrodes in the nwbfile "
                 "(same as number of channels in traces_as_iterator)")

    # Get electrodes table (added to nwb in `add_electrode_data_berke_probe`)
    electrodes = nwbfile.electrodes
    if multiple_probes:
        # With more than one probe, intan_channel_number repeats (every Neuropixels probe uses
        # CH0..CH383), so it cannot order the rows. Electrodes were added probe by probe in the same
        # order the probes were combined, and each probe's electrodes are in channel order, so row i
        # of the ElectricalSeries is already row i of the electrode table.
        raw_data_row_to_electrode_table_row = list(range(len(electrodes)))
    else:
        # Get the mapping of electrode to channel number (0-indexed)
        electrode_table_row_to_raw_data_row = electrodes['intan_channel_number'].data[:]
        # Get the reverse mapping of row in the raw ElectricalSeries to row in the electrode table
        raw_data_row_to_electrode_table_row = np.argsort(electrode_table_row_to_raw_data_row)

    # Create the electrode table region encompassing all electrodes
    electrode_table_region = nwbfile.create_electrode_table_region(
        region=list(raw_data_row_to_electrode_table_row),
        description="Electrodes used in raw ElectricalSeries recording",
    )
    logger.info(f"Raw ephys data is {num_samples} samples x {num_channels} channels "
                f"(conversion factor {channel_conversion_factor_uv} uV/bit)")

    # Convert to uV without loading the whole thing at once
    uv_traces_as_iterator = MicrovoltsSpikeInterfaceRecordingDataChunkIterator(traces_as_iterator,
                                                                               channel_conversion_factor_uv)

    # A chunk of shape (81920, 64) and dtype int16 (2 bytes) is ~10 MB, the recommended chunk size by the
    # NWB team. Compress with blosc-zstd (via the hdf5plugin library), also recommended by the NWB team: 
    # it is much faster to write than gzip and compresses better, which matters a lot for these large recordings. 
    # Fall back to gzip (always available) if hdf5plugin is not installed.
    # NOTE: reading a blosc-zstd-compressed NWB requires the blosc HDF5 filter (hdf5plugin) to be installed.
    # We include hdf5plugin as a dependency, so anything using jdb_to_nwb (or Spyglass) can read these files,
    # but a bare-HDF5 reader without the filter cannot. Use gzip instead if you need maximum portability.
    chunks = (min(num_samples, 81920), min(num_channels, 64))
    try:
        import hdf5plugin
        blosc_zstd = dict(hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
        data_data_io = H5DataIO(
            data=uv_traces_as_iterator,
            chunks=chunks,
            compression=blosc_zstd["compression"],
            compression_opts=blosc_zstd["compression_opts"],
            allow_plugin_filters=True,
        )
        compression_desc = "blosc-zstd (clevel 5)"
    except ImportError:
        data_data_io = H5DataIO(
            data=uv_traces_as_iterator,
            chunks=chunks,
            compression="gzip",
        )
        compression_desc = "gzip (install hdf5plugin for faster, smaller blosc-zstd compression)"
    logger.info(f"ElectricalSeries data will be written with chunks {chunks} and {compression_desc} compression")

    # If we have ground truth port visit times (photometry), align timestamps to that
    ground_truth_time_source = metadata.get("ground_truth_time_source")
    if ground_truth_time_source is not None:
        logger.info(f"Aligning ephys visit times to ground truth ({ground_truth_time_source})")
        ground_truth_visit_times = metadata.get("ground_truth_visit_times")
        ephys_timestamps = align_via_interpolation(unaligned_timestamps=original_timestamps,
                                                   unaligned_visit_times=ephys_visit_times,
                                                   ground_truth_visit_times=ground_truth_visit_times,
                                                   logger=logger)
    else:
        # If we don't have photometry, keep the original timestamps
        ephys_timestamps = original_timestamps
        logger.info("No ground truth visit times (photometry) found. "
                    "Ephys visit times are now our ground truth visit times.")

        # Ephys visit times are now our ground truth visit times
        metadata["ground_truth_time_source"] = "ephys"
        metadata["ground_truth_visit_times"] = ephys_visit_times

    # Say what the traces actually are. Preprocessed traces hold samples that were never recorded
    # (laser artifacts and common-mode noise are interpolated over), so the series must not claim
    # to be raw acquisition data.
    if processed_ephys_path:
        electrical_series_description = (
            "PREPROCESSED ephys data, in uV (multiply by conversion factor to get data in V). "
            "Laser artifacts and common-mode noise events have been replaced by linear "
            "interpolation and a Neuropixels 2.0 phase shift has been applied; no common average "
            "reference was applied (Kilosort applies its own). These are the traces the spike "
            f"sorting was run on, read from {processed_ephys_path}. See the 'preprocessing' "
            "associated file for the exact parameters."
        )
    else:
        electrical_series_description = (
            "Raw ephys data from OpenEphys recording, in uV (multiply by conversion factor to get data in V)."
        )

    # Create the ElectricalSeries
    # For now, we do not chunk or compress the timestamps, which are relatively small
    eseries = ElectricalSeries(
        name="ElectricalSeries",
        description=electrical_series_description,
        data=data_data_io,
        timestamps=ephys_timestamps,
        electrodes=electrode_table_region,
        conversion=VOLTS_PER_MICROVOLT, # conversion from uV to V
    )

    # Add the ElectricalSeries to the NWBFile
    logger.info("Adding raw ephys to the nwbfile as an ElectricalSeries")
    nwbfile.add_acquisition(eseries)

    # Record where each source recording sits in the file. Spyglass turns every epoch tag into an
    # interval list of that name, so these tags are what a spike sort is later restricted to -
    # one session, or both. The epoch times come from the timestamps, so they already account for
    # the bonsai trim and the gaps between recordings.
    for epoch in session_epochs:
        nwbfile.add_epoch(start_time=epoch["start_time"], stop_time=epoch["stop_time"],
                          tags=[epoch["session"]])
        logger.info(f"Added epoch '{epoch['session']}': {epoch['start_time']:.3f}s to "
                    f"{epoch['stop_time']:.3f}s")

    # Laser pulses, placed on the same clock as the ElectricalSeries. The conversion from a
    # recording's hardware clock to the nwb clock is taken from what this conversion actually did -
    # each session's first converted sample, in both clocks - rather than re-derived from the trims.
    if processed_ephys_path:
        from .convert_processed_ephys import add_laser_events

        add_laser_events(
            nwbfile=nwbfile,
            processed_ephys_path=processed_ephys_path,
            session_boundaries=processed_session_boundaries,
            probe_start_times=probe_start_times,
            session_hardware_starts={
                label: float(timestamps[0])
                for label, timestamps in zip(session_labels, session_timestamps)
            },
            session_nwb_starts=dict(zip(session_labels, session_time_offsets)),
            logger=logger,
        )

    # Save the raw Open Ephys settings.xml and structure.oebin as AssociatedFiles
    with open(settings_file_path, "r") as settings_file:
        raw_settings_xml = settings_file.read()
    add_associated_file(
        nwbfile,
        name="open_ephys_settings_xml",
        description="Raw settings.xml file from OpenEphys",
        content=raw_settings_xml,
        logger=logger,
        log_filename="settings.xml",
    )

    with open(oebin_params["oebin_path"], "r") as oebin_file:
        raw_structure_oebin = oebin_file.read()
    add_associated_file(
        nwbfile,
        name="open_ephys_structure_oebin",
        description="Raw structure.oebin file from OpenEphys (recording metadata: streams, channels, "
                    "bit_volts, sample rates)",
        content=raw_structure_oebin,
        logger=logger,
        log_filename="structure.oebin",
    )

    # When the traces are preprocessed, the file has to carry how they were made - the masking
    # parameters, the laser lags, and where each source recording sits in the concatenated data.
    if processed_ephys_path:
        provenance_files = ["preprocessing.json", "session_boundaries.json",
                            "laser_events.csv", "noise_events.csv"]
        # The laser protocol file is named after the rig, so find it rather than assuming the name
        provenance_files += [path.name for path in
                             sorted(Path(processed_ephys_path).glob("laser_control_protocol*.csv"))]
        for provenance_file_name in provenance_files:
            provenance_name = Path(provenance_file_name).stem
            provenance_path = Path(processed_ephys_path) / provenance_file_name
            if not provenance_path.exists():
                logger.info(f"No {provenance_file_name} at {provenance_path}; not recording it.")
                continue
            with open(provenance_path, "r") as provenance_file:
                add_associated_file(
                    nwbfile,
                    name=provenance_name,
                    description=f"{provenance_file_name} from the preprocessing pipeline "
                                "(how the traces in the ElectricalSeries were produced)",
                    content=provenance_file.read(),
                    logger=logger,
                    log_filename=provenance_file_name,
                )

    log_and_print(logger, "Finished adding raw ephys to the nwb.", level="info")
    # Each probe's own full (untrimmed) recording length. Spike sorting uses it to confirm each
    # analyzer was computed on that probe's recording - the probes can differ by a few samples.
    return {
        "ephys_start": open_ephys_start,
        "port_visits": ephys_visit_times,
        # bonsai_start_time (seconds) and probe sample rate let downstream spike sorting align
        # its sample-indexed spike times to the same "bonsai start = time 0" clock used here
        "bonsai_start_time": bonsai_start_time,
        "sample_rate": probe_sample_rate,
        # One entry per probe, in the same order as metadata["ephys"]["probes"]. Spike sorting
        # joins on that position: it cannot re-derive the label, because the metadata may name a
        # probe in any of the forms a stream goes by.
        "probe_results": [
            {"label": config["label"], "num_samples": num_samples_by_probe[config["label"]]}
            for config in probe_configs
        ],
    }
