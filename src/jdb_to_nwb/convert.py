import logging
import argparse
from pathlib import Path
import yaml
import uuid
import os
import shutil

from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from datetime import datetime
from dateutil import tz

from . import __version__
from .convert_video import add_video
from .convert_dlc import add_dlc
from .convert_raw_ephys import add_raw_ephys
from .convert_spikes import add_spikes
from .convert_behavior import add_behavior
from .convert_photometry import add_photometry


def setup_logger(log_name, path_logfile_info, path_logfile_warn, path_logfile_debug) -> logging.Logger:
    """
    Sets up a logger that outputs to 3 different files:
    - File for all general logs (log level INFO and above).
    - File for warnings and errors (log level WARNING and above).
    - File for detailed debug output (log level DEBUG and above)

    Args:
    log_name: Name of the logfile (for logger identification)
    path_logfile_info: Path to the logfile for info messages
    path_logfile_warn: Path to the logfile for warning and error messages
    path_logfile_debug: Path to the logfile for debug messages

    Returns:
    logging.Logger
    """

    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels (DEBUG and above)

    # Define format for log messages
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%d-%b-%y %H:%M:%S")

    # Handler for logging messages INFO and above to file
    fileHandler_info = logging.FileHandler(path_logfile_info, mode="w")
    fileHandler_info.setFormatter(formatter)
    fileHandler_info.setLevel(logging.INFO)

    # Handler for logging messages WARNING and above to file
    fileHandler_warn = logging.FileHandler(path_logfile_warn, mode="w")
    fileHandler_warn.setFormatter(formatter)
    fileHandler_warn.setLevel(logging.WARNING)

    # Handler for logging messages DEBUG and above to a file
    fileHandler_debug = logging.FileHandler(path_logfile_debug, mode="w")
    fileHandler_debug.setFormatter(formatter)
    fileHandler_debug.setLevel(logging.DEBUG)

    # Add handlers to the logger
    logger.addHandler(fileHandler_info)
    logger.addHandler(fileHandler_warn)
    logger.addHandler(fileHandler_debug)

    return logger


def create_nwbs(metadata_file_path: Path, output_nwb_dir: Path):
    # Read metadata
    with open(metadata_file_path, "r") as f:
        metadata = yaml.safe_load(f)

    # TODO: Add checks for required metadata?

    # Parse subject metadata
    subject = Subject(**metadata["subject"])

    # Create session_id in rat_date format
    session_id = f"{metadata.get("animal_name")}_{metadata.get("date")}"

    # Create directory for associated figures
    fig_dir = Path(output_nwb_dir) / f"{session_id}_figures"
    os.makedirs(fig_dir, exist_ok=True)

    # Create directory for conversion log files
    log_dir = Path(output_nwb_dir) / f"{session_id}_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logger with paths to log files
    info_log_file = Path(log_dir) / f"{session_id}_info_logs.log"
    warning_log_file = Path(log_dir) / f"{session_id}_warning_logs.log"
    debug_log_file = Path(log_dir) / f"{session_id}_debug_logs.log"
    logger = setup_logger("conversion_log", info_log_file, warning_log_file, debug_log_file)

    logger.info(f"Starting conversion for session_id {session_id}")

    # Save a copy of the metadata file to the logging directory
    metadata_copy_file_path = Path(log_dir) / f"{session_id}_metadata.yaml"
    shutil.copy(metadata_file_path, metadata_copy_file_path)

    logger.info(f"Original metadata file path was {metadata_file_path}")
    logger.info(f"Saved a copy of metadata to {metadata_copy_file_path}")

    nwbfile = NWBFile(
        session_description="Placeholder description",  # Placeholder: updated in add_behavior
        session_start_time=datetime.now(tz.tzlocal()),  # Placeholder: updated as the start of the earliest datastream
        identifier=str(uuid.uuid4()),
        institution=metadata.get("institution"),
        lab=metadata.get("lab"),
        experimenter=metadata.get("experimenter"),
        experiment_description=metadata.get("experiment_description"),
        session_id=session_id,
        subject=subject,
        surgery=metadata.get("surgery"),
        virus=metadata.get("virus"),
        notes=metadata.get("notes"),
        keywords=metadata.get("keywords"),
        source_script="jdb_to_nwb " + __version__,
        source_script_file_name="convert.py",
    )

    photometry_data_dict = add_photometry(nwbfile=nwbfile, metadata=metadata, fig_dir=fig_dir, logger=logger)
    add_behavior(nwbfile=nwbfile, metadata=metadata, logger=logger)

    output_video_path = Path(output_nwb_dir) / f"{session_id}_video.mp4"
    add_video(nwbfile=nwbfile, metadata=metadata, output_video_path=output_video_path, logger=logger)
    add_dlc(nwbfile=nwbfile, metadata=metadata, logger=logger)

    add_raw_ephys(nwbfile=nwbfile, metadata=metadata, fig_dir=fig_dir)
    add_spikes(nwbfile=nwbfile, metadata=metadata)

    # TODO: Time alignment? Or just assign the same time=0 and let NWB do the rest?
    # If photometry is present, timestamps should be aligned to the photometry
    # otherwise ephys, otherwise behavior
    #
    # For this alignment, add_photometry returns a photometry_data_dict with keys:
    # - sampling_rate: int (Hz)
    # - port visits: list of port visits in photometry time
    # - photometry_start: datetime object marking the start time of photometry recording
    # and add_behavior returns: photometry_start_in_arduino_time
    # For now, ignore that these functions return values because we don't use them yet
    
    # DLC / spatial series currently start at photometry start, so we want to subtract that out!
    
    # If we have a recorded photometry start time, use that as the session start time
    if photometry_data_dict.get('photometry_start') is not None:
        nwbfile.fields["session_start_time"] = photometry_data_dict.get('photometry_start')
    else:
        # Placeholder
        nwbfile.fields["session_start_time"] = datetime.now(tz.tzlocal())

    print("Writing file...")
    output_nwb_file_path = Path(output_nwb_dir) / f"{session_id}.nwb"
    with NWBHDF5IO(output_nwb_file_path, mode="w") as io:
        io.write(nwbfile)

    print(f"NWB file created successfully at {output_nwb_file_path}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file_path", type=Path, help="Path to the metadata YAML file.")
    parser.add_argument("output_nwb_dir", type=Path, help="Path to the output NWB directory.")
    args = parser.parse_args()

    create_nwbs(args.metadata_file_path, args.output_nwb_dir)
