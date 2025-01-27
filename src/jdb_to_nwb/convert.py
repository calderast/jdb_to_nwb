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


def setup_logger(log_name: str, path_logfile_all: str, path_logfile_warn: str) -> logging.Logger:
    """
    Sets up a logger that outputs to two different files:
    - One file for all logs (INFO and above).
    - Another file for warnings and above (WARNING and above).

    Args:
    log_name: Name of the logfile (for logger identification)
    path_logfile_all: Path to the logfile for all messages (INFO and above).
    path_logfile_warn: Path to the logfile for warning and above messages (WARNING and above).

    Returns:
    logging.Logger
    """
    # Create logger
    logger = logging.getLogger(log_name)

    # Define format for log messages
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S")

    # Handler for logging all messages (INFO and above) to file
    fileHandler_all = logging.FileHandler(path_logfile_all, mode="w")
    fileHandler_all.setFormatter(formatter)

    # Handler for logging only warnings and above to file
    fileHandler_warn = logging.FileHandler(path_logfile_warn, mode="w")
    fileHandler_warn.setFormatter(formatter)

    # Set different logging levels for handlers
    fileHandler_all.setLevel(logging.INFO)  # Logs all messages (INFO, WARNING, ERROR, CRITICAL)
    fileHandler_warn.setLevel(logging.WARNING)  # Logs warnings and above (WARNING, ERROR, CRITICAL)

    # Set the logger level to INFO (this controls the global level)
    logger.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(fileHandler_all)
    logger.addHandler(fileHandler_warn)

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
    logger = setup_logger("conversion_log", info_log_file, warning_log_file)

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

    add_photometry(nwbfile=nwbfile, metadata=metadata, fig_dir=fig_dir, logger=logger)
    add_behavior(nwbfile=nwbfile, metadata=metadata)

    output_video_path = Path(output_nwb_dir) / f"{session_id}_video.mp4"
    add_video(nwbfile=nwbfile, metadata=metadata, output_video_path=output_video_path)
    add_dlc(nwbfile=nwbfile, metadata=metadata)

    add_raw_ephys(nwbfile=nwbfile, metadata=metadata, fig_dir=fig_dir)
    add_spikes(nwbfile=nwbfile, metadata=metadata)

    # TODO: Time alignment? Or just assign the same time=0 and let NWB do the rest?
    # If photometry is present, timestamps should be aligned to the photometry
    # otherwise ephys, otherwise behavior
    # For this alignment, add_photometry returns: phot_sampling_rate, port_visits
    # and add_behavior returns: photometry_start_in_arduino_time
    # For now, ignore that these functions return values because we don't use them yet

    # TODO: Reset the session start time to the earliest of the data streams
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
