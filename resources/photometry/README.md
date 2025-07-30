# Photometry resources

The `photometry_devices.yaml` file contains a running list of excitation sources (i.e. LEDs and their metadata/wavelengths), optic fibers, and photodetectors used by the Berke Lab for the hex maze task. 

`virus_info.yaml` contains a running list of all indicators used for fiber photometry, as well as any excitatory/inhibitory opsins used for experiments.

`photometry_mappings.yaml` maps each indicator to a list of possible excitation sources.
Our current photometry processing code makes a lot of assumptions about which channel is which signal based on known experimental choices (e.g. we always record dLight in the old maze room, and always do dual recordings of rDA3m and gACh4h in the new maze room). Currently the old maze room has blue (470 nm) and purple (405 nm) LEDs, and the new maze room has blue (470 nm), purple (405 nm), and green (565 nm) LEDs.

See Github issue #28 (https://github.com/calderast/jdb_to_nwb/issues/28) for a complete overview of the current setup and assumptions.

As we move towards adding more photometry metadata and flexibility to our photometry processing, we will allow the user to specify which wavelength was recorded by each channel. Knowing the indicators present and the wavelengths of each channel allows us to flexibly process signals for various recording setups. Each time we use a new indicator, it should be added to both `virus_info.yaml` and `photometry_mappings.yaml`

Note that these lists are currently in-progress. The ndx-fiber-photometry extension for nwb is still in development, so how we specify these values is subject to change.
