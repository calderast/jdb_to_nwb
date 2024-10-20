from pynwb import NWBFile


def add_behavior(nwbfile: NWBFile, metadata: dict):
    print("Adding behavior...")

    # get metadata for behavior from metadata file
    arduino_text_file_path = metadata["behavior"]["arduino_text_file_path"]
    arudino_timestamps_file_path = metadata["behavior"]["arduino_timestamps_file_path"]
    maze_configuration_file_path = metadata["behavior"]["maze_configuration_file_path"]

    # TODO: extract behavior data

    # TODO: add to NWB file
    # TODO: replace placeholders with actual data
    nwbfile.add_epoch_column(name="maze_configuration", description="The maze configuration for each epoch")
    nwbfile.add_epoch_column(name="probability_A", description="The probability of A for each epoch")
    nwbfile.add_epoch_column(name="probability_B", description="The probability of B for each epoch")
    nwbfile.add_epoch_column(name="probability_C", description="The probability of C for each epoch")
    nwbfile.add_epoch(
        start_time=0,  # in seconds relative to the start of the recording
        stop_time=100,
        maze_configuration="12312131313",
        probability_A=0.5,
        probability_B=0.5,
        probability_C=0.5,
    )

    nwbfile.add_trial_column(name="custom_column", description="???")
    nwbfile.add_trial(
        start_time=0,  # in seconds relative to the start of the recording
        stop_time=1,
        custom_column="12312131313",
    )
    nwbfile.add_trial(
        start_time=1,  # in seconds relative to the start of the recording
        stop_time=2,
        custom_column="12312131313",
    )

    # NOTE: time before the first nosepoke is not included in any trial
    # NOTE: time after the last nosepoke is not included in any trial
