"""
Example usage of plot_probability_matching function

This script demonstrates how to use the plot_probability_matching function
with trial_data and block_data from parse_arduino_text.
"""

import csv
import itertools
from jdb_to_nwb.convert_behavior import parse_arduino_text, adjust_arduino_timestamps
from jdb_to_nwb.plotting.plot_behavior import plot_probability_matching
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example: Load data from arduino files
arduino_text_file_path = "tests/test_data/behavior/arduinoraw0.txt"
arduino_timestamps_file_path = "tests/test_data/behavior/ArduinoStamps0.csv"

# Read arduino text file
with open(arduino_text_file_path, "r") as arduino_text_file:
    arduino_text = arduino_text_file.read().splitlines()

# Read arduino timestamps
with open(arduino_timestamps_file_path, "r") as arduino_timestamps_file:
    arduino_timestamps = list(map(float, itertools.chain.from_iterable(csv.reader(arduino_timestamps_file))))

# Adjust timestamps
arduino_timestamps, _ = adjust_arduino_timestamps(arduino_timestamps, logger)

# Parse the arduino text to get trial and block data
trial_data, block_data = parse_arduino_text(arduino_text, arduino_timestamps, logger)

logger.info(f"Parsed {len(trial_data)} trials across {len(block_data)} blocks")

# Create the probability matching plot
# Option 1: Save to a file
fig = plot_probability_matching(trial_data, block_data, fig_dir="./output")
logger.info("Plot saved to ./output/probability_matching.png")

# Option 2: Just create the figure without saving
# fig = plot_probability_matching(trial_data, block_data, fig_dir=None)
# You can then display it with fig.show() or save it manually with fig.savefig()
