import numpy as np

path = "/Volumes/Tim/Ephys/IM-1478/07242022/2022-07-24_15-04-42/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat"

total_channels = 264
port_visits_channel_num = 256
downsample_factor = 30

max_samples_read = 100000

# Set up memmap and shape it to (num_samples, total_channels)
data = np.memmap(path, dtype=np.int16, mode="r").reshape(-1, total_channels)

print(data.shape)

# Slice the data to the max_samples_read and downsample the visits channel by the downsample factor
visits_channel_data = data[0:max_samples_read, port_visits_channel_num]
visits_channel_data = visits_channel_data[::downsample_factor]

# Load the visits channel data into memory from the memory mapped file
visits_channel_data = np.array(visits_channel_data)

# Threshold the visits channel data 
# thresholded = np.where(visits_channel_data > 0)[0]

# Print the thresholded data
# print(thresholded)
