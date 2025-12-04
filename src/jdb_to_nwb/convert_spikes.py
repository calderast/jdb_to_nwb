from .mdasortinginterface import MdaSortingInterface
from pynwb import NWBFile


def add_spikes(nwbfile: NWBFile, metadata: dict, logger):
    
    if "ephys" not in metadata:
        return
    
    if not ("mountain_sort_output_file_path" in metadata["ephys"] and "sampling_frequency" in metadata["ephys"]):
        print("No spike sorting metadata found for this session. Skipping spike conversion.")
        logger.info("No spike sorting metadata found for this session. Skipping spike conversion.")
        return

    print("Adding spikes...")
    logger.info("Adding spikes...")
    mountain_sort_output_file_path = metadata["ephys"]["mountain_sort_output_file_path"]
    sampling_frequency = metadata["ephys"]["sampling_frequency"]

    interface = MdaSortingInterface(mountain_sort_output_file_path, sampling_frequency=sampling_frequency)
    metadata = interface.get_metadata()

    # Append to existing in-memory NWBFile
    interface.add_to_nwbfile(nwbfile, metadata)