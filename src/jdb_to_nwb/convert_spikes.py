from .mdasortinginterface import MdaSortingInterface
from pynwb import NWBFile


def add_spikes(nwbfile: NWBFile, metadata: dict):
    print("Adding spikes...")
    mountain_sort_output_file_path = metadata["ephys"]["mountain_sort_output_file_path"]
    sampling_frequency = metadata["ephys"]["sampling_frequency"]

    interface = MdaSortingInterface(mountain_sort_output_file_path, sampling_frequency=sampling_frequency)
    metadata = interface.get_metadata()

    # TODO re-align timestamps to the photometry timestamps if they exist

    # Append to existing in-memory NWBFile
    interface.add_to_nwbfile(nwbfile, metadata)
