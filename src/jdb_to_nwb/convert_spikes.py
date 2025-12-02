from pynwb import NWBFile
from spikeinterface.extractors import read_mda_sorting
from nwb_conversion_tools import SortingInterface


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

    # Load MountainSort output via SpikeInterface
    sorting = read_mda_sorting(
        folder_path=mountain_sort_output_file_path,
        sampling_frequency=sampling_frequency
    )

    # Wrap it in an NWB converter interface
    interface = SortingInterface(sorting=sorting)

    # Get metadata for NWB
    sorting_metadata = interface.get_metadata()

    # Add to NWBFile
    interface.add_to_nwbfile(nwbfile=nwbfile, metadata=sorting_metadata)