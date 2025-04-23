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
from zoneinfo import ZoneInfo

from . import __version__
from .convert_video import add_video
from .convert_position import add_position
from .convert_raw_ephys import add_raw_ephys
from .convert_spikes import add_spikes
from .convert_behavior import add_behavior
from .convert_photometry import add_photometry


def to_datetime(date_str):
    """
    Return a datetime object from a date string in MMDDYYYY or YYYYMMDD format.
    Set the HH:MM:SS time to 00:00:00 Pacific Time.
    """
    # If it's already a datetime, we're good to go
    if isinstance(date_str, datetime):
        return date_str

    # Try ISO format first (e.g. "2024-01-22T00:00:00-08:00")
    if "T" in str(date_str):
        try:
            dt = datetime.fromisoformat(str(date_str))
            return dt.astimezone(ZoneInfo("America/Los_Angeles"))
        except (ValueError, TypeError, AttributeError):
            pass  # If ISO parsing fails, continue with other formats

    # Remove slashes and dashes so we can handle formats like MM/DD/YYYY, MM-DD-YYYY, etc
    date_str = str(date_str).replace("/", "").replace("-", "").strip()

    # Add a leading 0 if needed (if date_str was specified as an int, leading 0s get clipped)
    date_str = date_str.zfill(8)

    if len(date_str) != 8:
        raise ValueError("Date string must be exactly 8 characters long. "
                         f"Got date = {date_str} ({len(date_str)} characters)")

    # Auto-detect the date format: if date starts with "20", it must be YYYYMMDD format
    if date_str.startswith("20"):
        date_format = "%Y%m%d"
    # Otherwise, assume MMDDYYYY
    else:
        date_format = "%m%d%Y"

    # Convert to datetime and set timezone to Pacific 
    dt = datetime.strptime(date_str, date_format)
    dt = dt.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    return dt


def check_required_metadata(metadata):
    """
    Make sure all required basic metadata fields are present before doing conversion to nwb.
    """

    # Break if these fields are missing!!
    assert "experimenter" in metadata, (
        "Required field 'experimenter' not found in metadata!! Add this field and re-run conversion."
    )
    assert "date" in metadata, (
        "Required field 'date' not found in metadata!! Add this field and re-run conversion."
    )
    assert "subject" in metadata, (
        "Required field 'subject' not found in metadata!! Add this field and re-run conversion."
    )
    # Make sure subject has the required subfields
    subject_fields = set(metadata.get("subject").keys())
    required_subject_fields = {"subject_id", "species", "genotype", "sex", "description"}
    missing_subject_fields = required_subject_fields - subject_fields
    assert not missing_subject_fields, (
        f"Required subfields {missing_subject_fields} not found in subject metadata! "
        "Add these fielda and re-run conversion"
    )
    assert "age" in subject_fields or "date_of_birth" in subject_fields, (
        "Required subfield 'age' or 'date_of_birth' not found in subject metadata! "
        "Add one of these fields and re-run conversion. ('date_of_birth' is strongly preferred)"
    )
    # If date_of_birth is specified as a string, convert to a datetime object to make nwb happy
    # They are working on fixing this, so this will eventually be removed
    if "date_of_birth" in subject_fields:
        metadata["subject"]["date_of_birth"] = to_datetime(metadata.get("subject").get("date_of_birth"))

    # If animal_name not set in metadata, set it to subject_id.
    # We do this here instead of in set_default_metadata because animal_name needed
    # to name the file and set up logging
    if "animal_name" not in metadata:
        metadata["animal_name"] = metadata["subject"]["subject_id"]


def set_default_metadata(metadata, logger):
    """
    Set default values for some metadata fields if they are not specified in the metadata file.
    Log this at WARNING level.
    """
    # Set defaults if these fields are missing
    if "institution" not in metadata:
        metadata["institution"] = "University of California, San Francisco"
        logger.warning("No 'institution' found in metadata, "
                       "setting to default 'University of California, San Francisco'")
        print("No 'institution' found in metadata, setting to default 'University of California, San Francisco'")
    if "lab" not in metadata:
        metadata["lab"] = "Berke Lab"
        logger.warning("No 'lab' found in metadata, setting to default 'Berke Lab'")
        print("No 'lab' found in metadata, setting to default 'Berke Lab'")
    if "experiment_description" not in metadata:
        metadata["experiment_description"] = "Hex maze task"
        logger.warning("No 'experiment_description' found in metadata, setting to default 'Hex maze task'")
        print("No 'experiment_description' found in metadata, setting to default 'Hex maze task'")


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

    # Handler for logging messages INFO and above to a file
    fileHandler_info = logging.FileHandler(path_logfile_info, mode="w")
    fileHandler_info.setFormatter(formatter)
    fileHandler_info.setLevel(logging.INFO)

    # Handler for logging messages WARNING and above to a file
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

    # Check that required metadata fields are present before doing conversion
    check_required_metadata(metadata)

    # Convert date in metadata to datetime object
    metadata["datetime"] = to_datetime(metadata.get("date"))

    # Create session_id in {rat}_{date} format where date is YYYYMMDD
    session_id = f"{metadata.get('animal_name')}_{metadata.get('datetime').strftime('%Y%m%d')}"

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
    logger.info(f"Using source script jdb_to_nwb {__version__}")
    print(f"Starting conversion for session_id {session_id}")
    print(f"Using source script jdb_to_nwb {__version__}")

    # Save a copy of the metadata file to the logging directory
    metadata_copy_file_path = Path(log_dir) / f"{session_id}_metadata.yaml"
    shutil.copy(metadata_file_path, metadata_copy_file_path)

    logger.info(f"Original metadata file path was {metadata_file_path}")
    logger.info(f"Saved a copy of metadata to {metadata_copy_file_path}")

    # Set default lab, institution, and experiment_description if unspecified
    set_default_metadata(metadata, logger)

    # Parse subject metadata
    subject = Subject(**metadata["subject"])
    logger.info(f"Created subject: {subject}")

    nwbfile = NWBFile(
        session_description="Placeholder description",  # Placeholder: updated in add_behavior
        session_start_time=metadata.get("datetime"),    # Placeholder: updated as the start of the earliest datastream
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

    # Add photometry. Returns a dict with 'photometry_start' and 'port_visits' for alignment
    photometry_data_dict = add_photometry(nwbfile=nwbfile, metadata=metadata, logger=logger, fig_dir=fig_dir)
    metadata["photometry_visit_times"] = photometry_data_dict.get("port_visits")
    photometry_start = photometry_data_dict.get('photometry_start')

    # Add ephys. Returns a dict with 'ephys_start' and 'port_visits' for alignment
    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=logger, fig_dir=fig_dir)
    metadata["ephys_visit_times"] = ephys_data_dict.get("port_visits")
    ephys_start = ephys_data_dict.get('ephys_start')

    # Add behavior. Aligns port visits to photometry (if it exists) or ephys (if it exists and photometry doesn't)
    behavior_data_dict = add_behavior(nwbfile=nwbfile, metadata=metadata, logger=logger)
    metadata["photometry_start_in_arduino_ms"] = behavior_data_dict.get("photometry_start_in_arduino_time")
    metadata['arduino_visit_times'] = behavior_data_dict.get("port_visits")

    # Add video and DLC (position tracking). 
    # Aligns timestamps to photometry (if it exists) or ephys (if it exists and photometry doesn't)
    output_video_path = Path(output_nwb_dir) / f"{session_id}_video.mp4"
    add_video(nwbfile=nwbfile, metadata=metadata, output_video_path=output_video_path, logger=logger)
    add_position(nwbfile=nwbfile, metadata=metadata, logger=logger)

    # Add spikes
    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=logger)

    # If we have an exact photometry start time, use that as the session start time
    if photometry_start is not None:
        nwbfile.fields["session_start_time"] = photometry_start
        logger.info(f"Setting session_start_time to photometry start: {photometry_start}")
    # Otherwise if we have an exact Open Ephys start time, use that
    elif ephys_start is not None:
        nwbfile.fields["session_start_time"] = ephys_start
        logger.info(f"No photometry start time found, so setting session_start_time to ephys start: {ephys_start}")
    # Otherwise keep the default start time (00:00:00 Pacific Time on the session date)
    else:
        logger.warning("No photometry or ephys start time found, \nso session_start_time is the default start time: "
                       f"{nwbfile.fields['session_start_time']} (00:00:00 Pacific Time on the session date)")

    # Set the timestamps reference time equal to the session start time
    nwbfile.fields["timestamps_reference_time"] = nwbfile.fields["session_start_time"]

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
