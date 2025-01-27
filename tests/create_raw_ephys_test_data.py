# The OpenEphys binary data format is described here:
# https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html

# Each continuous directory contains the following files:

# - `continuous.dat`: A simple binary file containing N channels x M samples 16-bit integers in little-endian format.
# Data is saved as ch1_samp1, ch2_samp1, ... chN_samp1, ch1_samp2, ch2_samp2, ..., chN_sampM. The value of the
# least significant bit needed to convert the 16-bit integers to physical units is specified in the bitVolts
# field of the relevant channel in the structure.oebin JSON file. For “headstage” channels, multiplying by
# bitVolts converts the values to microvolts, whereas for “ADC” channels, bitVolts converts the values to volts.

# - `timestamps.npy`: A numpy array containing M 64-bit integers that represent the index of each sample in the
# .dat file since the start of acquisition.

# We could use SpikeInterface to read this data, but manipulating the data is easier with numpy since the data
# is a flat binary file.

# In Tim's data, the `continuous.dat` file contains 264 channels. The first 256 channels are the headstage (neural)
# channels, and the last 8 channels are the ADC channels.

# The `structure.oebin` JSON file and `settings.xml` contains metadata for the recording.

# To create test data of a reasonable size, we will trim the existing data and timestamps to 3,000 samples
# (one second of data) and 264 channels and save it to a new directory.

# We will manually edit the `structure.oebin` JSON file to remove the events and TTL channels and extra headstage
# and ADC channels. We will also manually edit the `settings.xml` file to remove the events and TTL channels and
# extra headstage and ADC channels.

# To run this script, copy Tim's open ephys data directory for IM-1478/2022-07-25_15-30-00 to your computer
# and adjust the paths in this script to point to the location of the data on your computer.

# Then run this script from the command line from the root of the repo:
#   python tests/test_data/create_raw_ephys_test_data.py

import numpy as np
from pathlib import Path
import shutil

# NOTE: Adjust this path to point to the location of Tim's sorted data for IM-1478/2022-07-25_15-30-00
open_ephys_data_root = Path("/Users/rly/Documents/NWB/berke-lab-to-nwb/data/2022-07-25_15-30-00")

continuous_dat_file_path = open_ephys_data_root / "experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat"
timestamps_file_path = open_ephys_data_root / "experiment1/recording1/continuous/Rhythm_FPGA-100.0/timestamps.npy"

# Set the properties of the source data and parameters for the trimmed data
num_channels = 264
sampling_rate_in_hz = 30_000

# Specify the number of seconds and channels of the original data to keep
num_seconds_to_keep = 0.1
num_channels_to_keep = 264

# Create a new directory to store the trimmed data
new_data_root = Path("./tests/test_data/raw_ephys/2022-07-25_15-30-00")
new_data_root.mkdir(parents=True, exist_ok=True)
new_data_dir = new_data_root / "experiment1/recording1/continuous/Rhythm_FPGA-100.0"
new_data_dir.mkdir(parents=True, exist_ok=True)

# Load the data from the continuous.dat file into a memory-mapped numpy array
data = np.memmap(continuous_dat_file_path, dtype=np.int16, mode="r")
assert len(data) % num_channels == 0, f"Data length is not divisible by num_channels: {num_channels}"
num_samples = len(data) // num_channels
data = data.reshape(num_samples, num_channels)

# Trim the data to desired number of seconds
num_samples_to_keep = int(num_seconds_to_keep * sampling_rate_in_hz)
data = data[0:num_samples_to_keep, 0:num_channels_to_keep]

# Save the trimmed data to a new binary file
trimmed_continuous_dat_file_path = new_data_dir / "continuous.dat"
with open(trimmed_continuous_dat_file_path, "wb") as f:
    f.write(data.tobytes())

# Load the timestamps from the timestamps.npy file into a numpy array
timestamps = np.load(timestamps_file_path)
assert len(timestamps) == num_samples, f"Timestamps length does not match expected length: {num_samples}"
timestamps = timestamps[:num_samples_to_keep]

# Save the trimmed timestamps to a new numpy file
trimmed_timestamps_file_path = new_data_dir / "timestamps.npy"
np.save(trimmed_timestamps_file_path, timestamps)

# Read the new binary file back into a numpy array and confirm the data matches
read_data = np.fromfile(trimmed_continuous_dat_file_path, dtype=np.int16)
assert (
    len(read_data) == num_samples_to_keep * num_channels_to_keep
), f"Data length does not match expected length: {num_samples_to_keep * num_channels_to_keep}"
read_data = read_data.reshape(num_samples_to_keep, num_channels_to_keep)
assert np.allclose(data, read_data), "Data does not match original data"

# If we trimmed the data to the full number of channels, we can copy the original structure.oebin JSON file
# and settings.xml file to the new directory
if num_channels_to_keep == num_channels:
    original_structure_oebin_file_path = open_ephys_data_root / "experiment1/recording1/structure.oebin"
    new_structure_oebin_file_path = new_data_root / "experiment1/recording1/structure.oebin"
    shutil.copy(original_structure_oebin_file_path, new_structure_oebin_file_path)
    
    original_settings_xml_file_path = open_ephys_data_root / "settings.xml"
    new_settings_xml_file_path = new_data_root / "settings.xml"
    shutil.copy(original_settings_xml_file_path, new_settings_xml_file_path)
else:
    print("Data was trimmed to fewer channels than the original data, so we did not copy the structure.oebin "
          "JSON file or settings.xml file to the new directory. Make sure to manually edit these files "
          "to remove the channels that were trimmed.")

print(f"Trimmed raw ephys data saved to: {new_data_root}")
