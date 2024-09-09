from pynwb import NWBFile



def add_photometry(nwbfile: NWBFile, metadata: dict):
    print("Adding photometry...")
    # get metadata for photometry from metadata file
    signals_mat_file_path = metadata["photometry"]["signals_mat_file_path"]
    photometry_sampling_rate_in_hz = metadata["photometry"]["sampling_rate"]

    # TODO: extract photometry signals

    # Create ndx-fiber-photometry objects

    # TODO: extract nosepoke times

    # if photometry exists, it serves as the main clock, so we do not need to realign these timestamps

    # TODO: add to NWB file


    