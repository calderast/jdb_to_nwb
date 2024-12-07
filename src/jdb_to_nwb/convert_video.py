from pynwb import NWBFile
from scipy.ndimage import gaussian_filter
import csv
import numpy as np

def add_video(nwbfile: NWBFile, metadata: dict):
    print("Adding video..")

    # Get file paths for video from metadata file
    video_file_path = metadata["video"]["arduino_video_file_path"]
    video_timestamps_file_path = metadata["video"]["arduino_video_timestamps_file_path"]

    # TODO: use Steph's returned value
    arduino_timestamps_file_path = metadata["behavior"]["arduino_timestamps_file_path"]
    deeplabcut_file_path = metadata["video"]["deeplabcut_file_path"]

    # TODO: make this optional or automatic?
    # TODO: specifying dlc_algorithm_name?
    # TODO: adding module to automatically return pixelsPerCm

    phot_dlc = metadata["video"]["phot_dlc"] # 'y' or 'n'?
    pixelsPerCm = 3.14 # 3.14 if video is before IM-1594. 2.6 before 1/11/2024. 2.688 after (old maze)

    # Read times of each position point, equal to the time of each camera frame recording
    with open(video_timestamps_file_path, "r") as video_timestamps_file:
        video_timestamps = np.array(list(csv.reader(open(video_timestamps_file, 'r'))), dtype=float).ravel()

    # Read arduino timestamps from the CSV into a list of floats to use for parsing
    with open(arduino_timestamps_file_path, "r") as arduino_timestamps_file:
        arduino_timestamps = list(map(float, itertools.chain.from_iterable(csv.reader(arduino_timestamps_file))))

    # Convert arduino timestamps to corresponding photosmetry sample number
    video_timestamps = adjust_video_timestamps(video_timestamps, arduino_timestamps)

    # Read 
    x, y, vel, acc = read_dlc(deeplabcut_file_path, phot_dlc = phot_dlc, cutoff = 0.9, cam_fps = 15, pixelsPerCm)

    return video_timestamps, x, y, vel, acc

    

def read_dlc(deeplabcut_file_path, phot_dlc = 'n', cutoff = 0.9, cam_fps = 15, pixelsPerCm)

    # Read position data from deeplabcut
    if phot_dlc == 'y':
        position_col = 'cap'
        dlc_position_file = 'Behav_Vid0DLC_resnet50_Triangle_Maze_PhotFeb12shuffle1_800000.h5'
        dlc_position = pd.read_hdf(deeplabcut_file_path + dlc_position_file).DLC_resnet50_Triangle_Maze_PhotFeb12shuffle1_800000
    else:
        position_col = 'cap_back'
        dlc_position_file = 'Behav_Vid0DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000.h5'
        dlc_position = pd.read_hdf(deeplabcut_file_path + dlc_position_file).DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000

    # Remove position with uncertainty
    position = dlc_position[position_col][['x', 'y']].copy()
    position.loc[dlc_position[position_col].likelihood < cutoff, ['x', 'y']] = np.nan

    # Remove the abrupt jump of position bigger than a body of rat (30cm)
    pixelJumpCutoff = 30 * pixelsPerCm
    position.loc[position.x.notnull(),['x','y']] = detect_and_replace_jumps(
        position.loc[position.x.notnull(),['x','y']].values,pixelJumpCutoff)

    # Fill the missing gaps
    position.loc[:,['x','y']] = fill_missing_gaps(position.loc[:,['x','y']].values)

    # Calculate velocity
    vel = calculate_velocity(position['x'].values, position['y'].values,fps=cam_fps, unit_conversion = pixelsPerCm)
    vel = np.append([0],vel)
    
    # Caculate acceleration
    acc = calculate_acceleration(position['x'].values, position['y'].values,fps = cam_fps, pixel_to_cm = pixelsPerCm)

    return position['x'].values, position['y'].values, vel, acc


def detect_and_replace_jumps(coordinates, pixelJumpCutoff):
    n = len(coordinates)
    jumps = []

    # Calculate Euclidean distances between consecutive points
    distances = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    
    # Find positions where the distance exceeds the threshold: pixelJumpCutoff
    jump_indices = np.where(distances > pixelJumpCutoff)[0] + 1
    
    # Mark all points within the jump range
    for idx in jump_indices:
        start = max(0, idx - 1)
        end = min(n, idx + 2)
        jumps.extend(range(start, end))
    
    # Replace points belonging to jumps with NaN
    coordinates[jumps] = np.nan

    return coordinates

def fill_missing_gaps(position_data):

    # Identify missing values as NaNs
    missing_values = np.isnan(position_data[:, 0]) | np.isnan(position_data[:, 1])

    # Compute the cumulative sum of missing values to identify contiguous gaps
    cumulative_sum = np.cumsum(missing_values)
    gap_starts = np.where(np.diff(cumulative_sum) == 1)[0] + 1
    gap_ends = np.where(np.diff(cumulative_sum) == -1)[0]

    # Interpolate the missing values in each gap using linear interpolation
    for gap_start, gap_end in zip(gap_starts, gap_ends):
        if gap_start == 0 or gap_end == len(position_data) - 1:
            continue  # ignore gaps at the beginning or end of the data
        else:
            x = position_data[gap_start - 1:gap_end + 1, 0]
            y = position_data[gap_start - 1:gap_end + 1, 1]
            interp_func = interp1d(x, y, kind='linear')
            position_data[gap_start:gap_end, 0] = np.linspace(x[0], x[-1], gap_end - gap_start + 1)
            position_data[gap_start:gap_end, 1] = interp_func(position_data[gap_start:gap_end, 0])

    return position_data

def adjust_video_timestamps(video_timestamps: list, arduino_timestamps: list):
    """Convert arduino timestamps to corresponding photometry sample number."""
    # The photometry start time is always the second timestamp in arduino_timestamps
    photometry_start = arduino_timestamps[1]

    # Adjust all arduino timestamps so the photometry starts at time zero
    video_timestamps = np.subtract(video_timestamps, photometry_start)


def calculate_velocity(x, y, fps, unit_conversion=1):
    # Convert pixels to cm if required
    x = x * unit_conversion
    y = y * unit_conversion
    
    # Calculate distance
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)

    # Calculate time
    time = 1/fps
    t = np.arange(0, len(x) - 1) * time

    # Calculate velocity
    vel = dist / time

    return vel

def calculate_acceleration(x, y, fps, pixel_to_cm=1):
    # convert pixel to cm
    x_cm = x * pixel_to_cm
    y_cm = y * pixel_to_cm

    # calculate velocity
    velocity_x = np.gradient(x_cm) * fps
    velocity_y = np.gradient(y_cm) * fps
    velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

    # calculate acceleration
    acceleration_x = np.gradient(velocity_x) * fps
    acceleration_y = np.gradient(velocity_y) * fps
    acceleration = np.sqrt(acceleration_x ** 2 + acceleration_y ** 2)


# ##  TODO: alignment with visit indices, This is leftover from tim's code that to be dumpted or used
# def align_pos_to_visits(Fs, visits, datepath, phot_dlc='n',
#                         filecount="0", cam_fps=15,pixelsPerCm=3.14):
#     # Import arduino behavioral data and their timestamps
#     ardtext = open(datepath + f'arduinoraw{filecount}.txt', 'r').read().splitlines()
#     with open(datepath + f'ArduinoStamps{filecount}.csv', 'r') as at:
#         ardtimes = np.array(list(csv.reader(at)), dtype=float).ravel()
#     photStart = ardtimes[1] # time when pulse was sent to R series. 0 index for sig data
#     ardstamps = np.round((ardtimes - photStart) * (Fs / 1000)).astype(int) # convert ardtimes to sample number to match photometry data.

#     # Align with photometry visit indices
#     porttimes = get_port_times(ardtext, ardtimes) - photStart
#     framestamps = []
#     inds = (frametimes<=porttimes[0])
#     framestamps.append(frametimes[inds]*(Fs/1000))
#     for p in range(1,len(porttimes)):
#         inds = (frametimes>porttimes[p-1])&(frametimes<=porttimes[p])
#         framestamps.append((frametimes[inds]-porttimes[p-1])*(Fs/1000) + visits[p-1])
#     inds = (frametimes>=porttimes[-1])
#     framestamps.append((frametimes[inds]-porttimes[-1])*(Fs/1000) + visits[-1])
#     framestamps = np.concatenate(framestamps).astype(int)
#     ##TODO ADD IN CODE TO REMOVE ABERRANT JUMPS AND THEN FILL IN MISSING GAPS