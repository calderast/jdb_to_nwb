import csv
import datetime
import numpy as np
import pandas as pd
from pynwb import NWBFile
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


def add_video(nwbfile: NWBFile, metadata: dict):
    
    if "video" not in metadata:
        print("No video metadata found for this session. Skipping video conversion.")
        return None

    print("Adding video...")

    # Get file paths for video from metadata file
    video_file_path = metadata["video"]["arduino_video_file_path"]
    video_timestamps_file_path = metadata["video"]["arduino_video_timestamps_file_path"]

    # Metadata should include the full path to the dlc h5
    # e.g. Behav_Vid0DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000.h5
    deeplabcut_file_path = metadata["video"]["dlc_path"]

    # If pixels_per_cm exists in metadata, use that value
    if "pixels_per_cm" in metadata["video"]:
        PIXELS_PER_CM = metadata["video"]["pixels_per_cm"]
    # Otherwise, assign it based on the date of the experiment
    else:
        PIXELS_PER_CM = assign_pixels_per_cm(metadata["date"])

    # Read times of each position point, equal to the time of each camera frame recording
    with open(video_timestamps_file_path, "r") as video_timestamps_file:
        video_timestamps = np.array(list(csv.reader(open(video_timestamps_file, 'r'))), dtype=float).ravel()

    # Read x and y position data and calculate velocity and acceleration
    x, y, velocity, acceleration = read_dlc(deeplabcut_file_path, pixels_per_cm=PIXELS_PER_CM, 
                                            cutoff=0.9, cam_fps=15)

    return video_timestamps, x, y, velocity, acceleration


def assign_pixels_per_cm(date_str):
    """
    Assigns constant PIXELS_PER_CM based on the provided date string in MMDDYYYY format.
    PIXELS_PER_CM is 3.14 if video is before IM-1594 (before 01012023), 
    2.3 before (01112024), or 2.688 after (old maze)

    Args:
    - date_str (str): Date string in MMDDYYYY format, e.g., '11122022'.

    Returns:
    - float: The corresponding PIXELS_PER_CM value.
    """
    # Define the date format
    date_format = "%m%d%Y"
    
    try:
        # Parse the input date string into a datetime object
        date = datetime.strptime(date_str, date_format)
    except ValueError:
        raise ValueError(f"Date string '{date_str}' is not in the expected format 'mmddyyyy'.")

    # Define cutoff dates
    cutoff1 = datetime.strptime("12312022", date_format)  # December 31, 2022
    cutoff2 = datetime.strptime("01112024", date_format)  # January 11, 2024

    # Assign pixelsPerCm based on the date
    if date <= cutoff1:
        pixels_per_cm = 3.14
    elif cutoff1 < date <= cutoff2:
        pixels_per_cm = 2.3
    else:
        pixels_per_cm = 2.688  # After January 11, 2024
    return pixels_per_cm


def read_dlc(deeplabcut_file_path, pixels_per_cm, cutoff=0.9, cam_fps=15):
    """
    Read dlc position data from the deeplabcut file that contains algorithm name and position data

    Position data is under the column names: cap_back and cap_front:
    cap_back is the back of the rat implant (red)
    cap_front is the front of the rat implant (green)

    After reading the position data, calculate velocity and acceleration 
    based on the camera fps and pixels_per_cm

    Returns:
    - x: x coordinates of the rat's body parts
    - y: y coordinates of the rat's body parts
    - velocity: velocity of the rat's body parts
    - acceleration: acceleration of the rat's body parts
    """

    # Read deeplabcut file into a dataframe
    dlc_position = pd.read_hdf(deeplabcut_file_path)

    # Remove the multi-level column names so we are left with column names 'x', 'y', 'likelihood'
    dlc_position.columns = [col[-1] for col in dlc_position.columns]
    assert set(dlc_position.columns) == {'x', 'y', 'likelihood'}, (
        f"Expected DLC file columns x, y, and likelihood, got {dlc_position.columns}"
    )

    # Replace x,y coordinates where DLC has low confidence with NaN
    position = dlc_position[['x', 'y']].copy()
    position[dlc_position['likelihood'] < cutoff] = np.nan

    # Remove abrupt jumps of position bigger than a body of rat (30cm)
    pixel_jump_cutoff = 30 * pixels_per_cm
    position.loc[position.x.notnull(),['x','y']] = detect_and_replace_jumps(
        position.loc[position.x.notnull(),['x','y']].values,pixel_jump_cutoff)

    # Fill the missing gaps
    position.loc[:,['x','y']] = fill_missing_gaps(position.loc[:,['x','y']].values)

    # Calculate velocity and acceleration
    velocity, acceleration = calculate_velocity_acceleration(position['x'].values, 
        position['y'].values, fps=cam_fps, pixels_per_cm=pixels_per_cm)

    return position['x'].values, position['y'].values, velocity, acceleration


def detect_and_replace_jumps(coordinates, pixel_jump_cutoff):
    """
    Detect and replace jumps in the position data that are bigger than pixel_jump_cutoff (default 30 cm)
    Jumps are replaced with NaN
    """
    n = len(coordinates)
    jumps = []

    # Calculate Euclidean distances between consecutive points
    distances = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    
    # Find positions where the distance exceeds the threshold pixel_jump_cutoff
    jump_indices = np.where(distances > pixel_jump_cutoff)[0] + 1
    
    # Mark all points within the jump range
    for idx in jump_indices:
        start = max(0, idx - 1)
        end = min(n, idx + 2)
        jumps.extend(range(start, end))
    
    # Replace points belonging to jumps with NaN
    coordinates[jumps] = np.nan

    return coordinates


def fill_missing_gaps(position_data):
    """
    Fill missing values in the position data
    It identifies gaps in the position data and fills them with linear interpolation
    """
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


def calculate_velocity_acceleration(x, y, fps, pixels_per_cm):
    """
    Calculate velocity and acceleration based on the camera fps and pixels_per_cm
    """
    # Convert pixels to cm
    x_cm = x * pixels_per_cm
    y_cm = y * pixels_per_cm

    # Calculate velocity
    velocity_x = np.gradient(x_cm) * fps
    velocity_y = np.gradient(y_cm) * fps
    velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

    # Calculate acceleration
    acceleration_x = np.gradient(velocity_x) * fps
    acceleration_y = np.gradient(velocity_y) * fps
    acceleration = np.sqrt(acceleration_x ** 2 + acceleration_y ** 2)

    return velocity, acceleration