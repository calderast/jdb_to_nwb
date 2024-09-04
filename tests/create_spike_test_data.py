# The MountainSort MDA file format is described here:
# https://mountainsort.readthedocs.io/en/latest/mda_file_format.html

# SpikeInterface can read this format easily.

# In Tim's data, the MDA file contains 145 units and 3,040,410 spikes.

# To create test data of a reasonable size, we will trim the spike times to only those in the first 30,000 samples.
# This results in 63 units and 462 spikes.

# To run this script, copy Tim's MountainSort output file "firing.mda" to "../data/ephys/mntsort_output/firings.mda"
# or change the paths in this script to point to the location of Tim's data.

# Then run this script from the command line from the root of the repo:
#   python tests/test_data/create_spike_test_data.py

from pathlib import Path

from spikeinterface.extractors import read_mda_sorting, MdaSortingExtractor

# Create a new directory to store the trimmed data
new_data_dir = Path("./tests/test_data/processed_ephys")
new_data_dir.mkdir(parents=True, exist_ok=True)
output_file_path = new_data_dir / "firings.mda"

# Read the .mda file
firings_mda_file_path = Path("../data/ephys/mntsort_output/firings.mda")
sampling_frequency = 30_000
sorting = read_mda_sorting(firings_mda_file_path, sampling_frequency=sampling_frequency)

# Trim the spike times to only those in the first 30,000 samples
sorting_trimmed = sorting.frame_slice(start_frame=0, end_frame=30_000)

# Write the trimmed spike sorting data to a new .mda file
MdaSortingExtractor.write_sorting(sorting=sorting_trimmed, save_path=output_file_path)

print(f"Trimmed spike sorting data saved to: {output_file_path}")
