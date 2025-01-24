import argparse
from pathlib import Path
import yaml
import uuid
import os

from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from datetime import datetime
from dateutil import tz

from . import __version__
from .convert_raw_ephys import add_raw_ephys
from .convert_spikes import add_spikes
from .convert_behavior import add_behavior
from .convert_photometry import add_photometry


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

    nwbfile = NWBFile(
        session_description="Mock session",  # TODO: replace this from behavior data
        session_start_time=datetime.now(tz.tzlocal()),  # Will be updated later 
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

    # If photometry is present, timestamps should be aligned to the photometry
    add_photometry(nwbfile=nwbfile, metadata=metadata, fig_dir=fig_dir)
    photometry_start_in_arduino_time = add_behavior(nwbfile=nwbfile, metadata=metadata)

    add_raw_ephys(nwbfile=nwbfile, metadata=metadata)
    add_spikes(nwbfile=nwbfile, metadata=metadata)

    # TODO: time alignment

    # TODO: Reset the session start time to the earliest of the data streams
    nwbfile.fields["session_start_time"] = datetime.now(tz.tzlocal())

    print("Writing file, including iterative read from raw ephys data...")
    output_nwb_file_path = Path(output_nwb_dir) / f"{session_id}.nwb"
    with NWBHDF5IO(output_nwb_file_path, mode="w") as io:
        io.write(nwbfile)

    print(f"NWB file created successfully at {output_nwb_file_path}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file_path", type=Path, help="Path to the metadata YAML file.")
    parser.add_argument("output_nwb_dir", type=Path, help="Path to the output NWB directory.")
    args = parser.parse_args()

    create_nwbs(args.metadata_file_path, args.output_nwb_file_path)
