import argparse
from pathlib import Path
import yaml
import uuid
import os
import shutil

from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject

from . import __version__
from .utils import to_datetime, setup_logger
from .convert_video import add_video
from .convert_position import add_position
from .convert_raw_ephys import add_raw_ephys
from .convert_spikes import add_spikes
from .convert_behavior import add_behavior
from .convert_photometry import add_photometry

from .plotting.plot_combined import plot_photometry_signal_aligned_to_port_entry


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

    # TODO: If single phot and box file, do phot first and have it be ground truth.
    # If we have multiple that need to be stitched together, do the other stuff first and stitch based on
    # whatever we have determined the ground truth timestamps to be (ephys first, or behavior if no ephys)

    # Add photometry. Returns a dict with 'photometry_start' and 'port_visits' for alignment
    photometry_data_dict = add_photometry(nwbfile=nwbfile, metadata=metadata, logger=logger, fig_dir=fig_dir)
    metadata["photometry_visit_times"] = photometry_data_dict.get("port_visits")
    photometry_start = photometry_data_dict.get('photometry_start')

    # Add ephys. Returns a dict with 'ephys_start' and 'port_visits' for alignment
    ephys_data_dict = add_raw_ephys(nwbfile=nwbfile, metadata=metadata, logger=logger, fig_dir=fig_dir)
    metadata["ephys_visit_times"] = ephys_data_dict.get("port_visits")
    ephys_start = ephys_data_dict.get('ephys_start')

    # Add behavior. Aligns port visits to photometry (if it exists) or ephys (if it exists and photometry doesn't)
    behavior_data_dict = add_behavior(nwbfile=nwbfile, metadata=metadata, logger=logger, fig_dir=fig_dir)
    metadata["photometry_start_in_arduino_ms"] = behavior_data_dict.get("photometry_start_in_arduino_time")
    metadata['arduino_visit_times'] = behavior_data_dict.get("port_visits")

    # Add video and DLC (position tracking). 
    # Aligns timestamps to photometry (if it exists) or ephys (if it exists and photometry doesn't)
    output_video_path = Path(output_nwb_dir) / f"{session_id}_video.mp4"
    add_video(nwbfile=nwbfile, metadata=metadata, output_video_path=output_video_path, logger=logger)
    add_position(nwbfile=nwbfile, metadata=metadata, logger=logger, fig_dir=fig_dir)

    # Add spikes
    add_spikes(nwbfile=nwbfile, metadata=metadata, logger=logger)

    # Now that we have added both photometry and behavior, plot photometry signals aligned to port entry
    if photometry_data_dict.get('signals_to_plot') is not None:
        for signal in photometry_data_dict.get('signals_to_plot'):
            plot_photometry_signal_aligned_to_port_entry(nwbfile=nwbfile, signal_name=signal, fig_dir=fig_dir)

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
    logger.info("Writing file...")
    output_nwb_file_path = Path(output_nwb_dir) / f"{session_id}.nwb"
    with NWBHDF5IO(output_nwb_file_path, mode="w") as io:
        io.write(nwbfile)

    print(f"NWB file created successfully at {output_nwb_file_path}")
    logger.info(f"NWB file created successfully at {output_nwb_file_path}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file_path", type=Path, help="Path to the metadata YAML file.")
    parser.add_argument("output_nwb_dir", type=Path, help="Path to the output NWB directory.")
    args = parser.parse_args()

    create_nwbs(args.metadata_file_path, args.output_nwb_dir)
