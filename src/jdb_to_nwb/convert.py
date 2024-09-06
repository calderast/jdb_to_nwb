import argparse
from pathlib import Path
import yaml

from pynwb import NWBFile, NWBHDF5IO
from datetime import datetime
from dateutil import tz

from .convert_raw_ephys import add_raw_ephys
from .convert_spikes import add_spikes


def create_nwbs(
    metadata_file_path: Path,
    output_nwb_file_path: Path,
):
    with open(metadata_file_path, "r") as f:
        metadata = yaml.safe_load(f)

    # TODO: read these from metadata
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_raw_ephys(nwbfile=nwbfile, metadata=metadata)
    add_spikes(nwbfile=nwbfile, metadata=metadata)

    print(f"Writing file, including iterative read from raw ephys data...")

    with NWBHDF5IO(output_nwb_file_path, mode="w") as io:
        io.write(nwbfile)

    print(f"NWB file created successfully at {output_nwb_file_path}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file_path", type=Path, help="Path to the metadata YAML file.")
    parser.add_argument("output_nwb_file_path", type=Path, help="Path to the output NWB file.")
    args = parser.parse_args()

    create_nwbs(args.metadata_file_path, args.output_nwb_file_path)