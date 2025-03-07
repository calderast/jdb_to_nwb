import csv
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import Position
from scipy.interpolate import interp1d
from hdmf.common import DynamicTable


def assign_pixels_per_cm(session_date):
    """
    Assigns default PIXELS_PER_CM based on the date of the session.
    PIXELS_PER_CM is 3.14 if video is before IM-1594 (before 01/01/2023), 
    2.3 before (01/11/2024), or 2.688 after (old maze)

    Args:
    session_date (datetime): Datetime object for the date of this session

    Returns:
    float: The corresponding PIXELS_PER_CM value.
    """

    # Define cutoff dates
    cutoff1 = datetime.strptime("12312022", "%m%d%Y").replace(tzinfo=ZoneInfo("America/Los_Angeles"))  # Dec 31, 2022
    cutoff2 = datetime.strptime("01112024", "%m%d%Y").replace(tzinfo=ZoneInfo("America/Los_Angeles"))  # Jan 11, 2024

    # Assign pixels per cm based on the session date
    if session_date <= cutoff1:
        pixels_per_cm = 3.14
    elif cutoff1 < session_date <= cutoff2:
        pixels_per_cm = 2.3
    else:
        pixels_per_cm = 2.688 # After January 11, 2024
    return pixels_per_cm


def read_dlc(deeplabcut_file_path, pixels_per_cm, logger, likelihood_cutoff=0.9, cam_fps=15):
    """
    Read position data from the DeepLabCut file that contains algorithm name and position data

    For ephys, data is under the column names 'cap_back' and 'cap_front':
    cap_back is the back of the rat implant (red)
    cap_front is the front of the rat implant (green)
    For photometry, data is under the single column 'cap'
    The correct column(s) are detected and handled automatically.

    After reading the position data, calculate velocity and acceleration 
    based on the camera fps and pixels_per_cm (currently unused, but keeping for now)

    Returns:
    list of tuples: list of (string, pd.DataFrame) tuples specifying the named bodypart \
        tracked by DeepLabCut and its dataframe with columns x, y, likelihood, velocity, and acceleration
    """

    # Read DeepLabCut file into a dataframe
    dlc_position = pd.read_hdf(deeplabcut_file_path)
    
    # The bodypart names are stored as second-level columns
    body_part_names = set(dlc_position.columns.get_level_values('bodyparts'))
    print(f"Found DeepLabCut bodyparts: {body_part_names}")
    logger.info(f"Found DeepLabCut bodyparts: {body_part_names}")

    body_part_dfs = []
    for body_part in list(body_part_names):
        logger.info(f"Processing DLC bodypart {body_part}...")
    
        # Split the dataframe based on the body part
        bodypart_df = dlc_position.loc[:, dlc_position.columns.get_level_values('bodyparts') == body_part]
        bodypart_df.columns = bodypart_df.columns.get_level_values(-1)
    
        # Make sure we have the correct columns for this bodypart
        assert set(bodypart_df.columns) == {'x', 'y', 'likelihood'}, (
            f"Expected {body_part} columns x, y, and likelihood, got {bodypart_df.columns}"
        )
        logger.debug(f"Found expected columns {set(bodypart_df.columns)} for bodypart {body_part}")

        # Replace x, y coordinates where DLC has low confidence with NaN
        logger.debug(f"Replacing x, y coordinates where DLC has likelihood<{likelihood_cutoff} with NaN")
        position = bodypart_df[['x', 'y', 'likelihood']].copy()
        position.loc[position['likelihood'] < likelihood_cutoff, ['x', 'y']] = np.nan

        # Remove abrupt jumps of position bigger than a body of rat (30cm)
        pixel_jump_cutoff = 30 * pixels_per_cm
        logger.debug(f"Removing abrupt position jumps larger than 30cm ({pixel_jump_cutoff} pixels)")
        position.loc[position.x.notnull(),['x','y']] = detect_and_replace_jumps(
        position.loc[position.x.notnull(),['x','y']].values, pixel_jump_cutoff)

        # Fill the missing gaps
        logger.debug("Filling missing gaps with interpolated position values")
        position.loc[:,['x','y']] = fill_missing_gaps(position.loc[:,['x','y']].values)

        # Calculate velocity and acceleration
        velocity, acceleration = calculate_velocity_acceleration(position['x'].values, 
            position['y'].values, fps=cam_fps, pixels_per_cm=pixels_per_cm)
    
        # Add velocity and acceleration columns to df
        position['velocity'] = velocity
        position['acceleration'] = acceleration
        
        # Add each body part name and its corresponding position dataframe
        body_part_dfs.append((body_part, position))
    
    return body_part_dfs


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


def add_position_to_nwb(nwbfile: NWBFile, position_data: list[tuple], pixels_per_cm, video_timestamps, logger):
    """
    Add position data to the nwbfile as a SpatialSeries in the behavior processing module.
    
    Args:
    nwbfile: the nwb
    position_data: List of (string, pd.DataFrame) tuples. \
        String names tracked bodypart, DataFrame has position tracking columns x, y, and likelihood
    pixels_per_cm: pixels per cm conversion rate of the position data
    video_timestamps: timestamps of each camera frame (aka position datapoint) in seconds
    """
    
    # Convert pixels_per_cm to meters_per_pixel for consistency with Frank Lab
    meters_per_pixel = 0.01 / pixels_per_cm
    logger.debug(f"Meters per pixel: {meters_per_pixel}")

    # Make a processing module for behavior and add to the nwbfile
    if "behavior" not in nwbfile.processing:
        logger.debug("Creating nwb behavior processing module for position data")
        nwbfile.create_processing_module(
            name="behavior", description="Contains all behavior-related data"
        )

    position = Position(name="position")
    
    # Add x,y position to the nwb as a SpatialSeries for each tracked body part
    for body_part_name, body_part_position_df in position_data:
        logger.info(f"Adding position data for {body_part_name} to the nwb...")
        print(f"Adding position data for {body_part_name} to the nwb...")
        position.create_spatial_series(
            name=f"{body_part_name}_position",
            description="xloc, yloc",
            data=np.asarray(body_part_position_df[["x", "y"]]),
            unit="meters",
            conversion=meters_per_pixel,
            reference_frame="Upper left corner of video frame",
            timestamps=video_timestamps,
        )

        # Add DLC position likelihood as a timeseries to the behavior processing module
        # We may want this in the future if we adjust our likelihood threshold
        # It may also be helpful to know which coordinates are "real" and which were interpolated
        logger.debug("Adding DLC position likelihood as a timeseries to the behavior processing module")
        nwbfile.processing["behavior"].add(
            TimeSeries(
                name=f"DLC_likelihood_{body_part_name}",
                description="DeepLabCut position tracking likelihood",
                data=np.asarray(body_part_position_df["likelihood"]),
                unit="fraction",
                comments=f"Likelihood of each x,y coordinate for tracked bodypart '{body_part_name}'. "
                "Coordinates with likelihood <0.9 were interpolated from surrounding coordinates.",
                timestamps=video_timestamps,
            )
        )

    # TODO: The Frank Lab does a lot of checks based on video frame timestamps.
    # Do we need to do this? How often do we see the video frame timestamps skip or repeat?
    # We seem to take camera fps for granted in our current pipeline - is this the right move?
    # -> see https://github.com/LorenFrankLab/trodes_to_nwb/blob/main/src/trodes_to_nwb/convert_position.py

    nwbfile.processing["behavior"].add(position)


def add_hex_centroids(nwbfile: NWBFile, metadata: dict, pixels_per_cm, logger):
    """
    Read hex centroids from a csv file with columns hex, x, y 
    and add them to the nwbfile in the behavior processing module.
    """

    hex_centroids_file = metadata["video"].get("hex_centroids_file_path")
    if hex_centroids_file is None:
        print("No subfield 'hex_centroids_file_path' found in video metadata! Skipping adding hex centroids.")
        logger.warning("No subfield 'hex_centroids_file_path' found in video metadata! Skipping adding hex centroids.")
        return

    try:
        hex_centroids = pd.read_csv(hex_centroids_file)
    except FileNotFoundError:
        logger.error(f"The file '{hex_centroids_file}' was not found! Skipping adding hex centroids.")
        print(f"Error: The file '{hex_centroids_file}' was not found! Skipping adding hex centroids.")
        return

    # Check that the hex centroids file is in the format we expect
    num_hexes = 49
    if len(hex_centroids) != num_hexes:
        logger.error(f"Expected {num_hexes} centroids in the hex centroids file, got {len(hex_centroids)}!!!")
    else:
        logger.debug(f"Found the expected number of hex centroids in the centroids file ({num_hexes})")

    expected_columns = {'hex', 'x', 'y'}
    if set(hex_centroids.columns) != expected_columns:
        logger.error(f"Expected {expected_columns} columns in the hex centroids file, "
                     f"got {set(hex_centroids.columns)}!!")
        logger.error("Skipping adding centroids to the nwb")
        print(f"Expected {expected_columns} columns in the hex centroids file, got {set(hex_centroids.columns)}!")
        print("Skipping adding centroids to the nwb.")
        return
    else:
        logger.debug(f"Found expected columns {expected_columns} in the hex centroids file")

    # Convert pixels per cm to meters per pixel
    meters_per_pixel = 0.01 / pixels_per_cm
    hex_centroids['x_meters'] = hex_centroids['x'] * meters_per_pixel
    hex_centroids['y_meters'] = hex_centroids['y'] * meters_per_pixel

    # Set up the hex centroids table
    centroids_table = DynamicTable(name="hex_centroids", 
            description="Centroids of each hex in the maze (in video pixel coordinates)")
    centroids_table.add_column(name="hex", description="The ID of the hex in the maze (1-49)")
    centroids_table.add_column(name="x",
            description="The x coordinate of the center of the hex (in video pixel coordinates)")
    centroids_table.add_column(name="y",
            description="The y coordinate of the center of the hex (in video pixel coordinates)")
    centroids_table.add_column(name="x_meters", 
            description="The x coordinate of the center of the hex (in meters)")
    centroids_table.add_column(name="y_meters", 
            description="The y coordinate of the center of the hex (in meters)")
    # Add the hex centroids
    for _, row in hex_centroids.iterrows():
        centroids_table.add_row(**row.to_dict())

    # If it doesn't exist already, make a processing module for behavior and add to the nwbfile
    if "behavior" not in nwbfile.processing:
        logger.debug("Creating nwb behavior processing module for position data")
        nwbfile.create_processing_module(
            name="behavior", description="Contains all behavior-related data"
        )

    print("Adding hex centroids to the nwb...")
    logger.info("Adding hex centroids to the behavior processing module in the nwbfile")
    nwbfile.processing["behavior"].add(centroids_table)


def add_position(nwbfile: NWBFile, metadata: dict, logger):

    if "video" not in metadata:
        # Do not print "no video metadata found" message, because we already print that in add_video
        return

    # If pixels_per_cm exists in metadata, use that value
    if "pixels_per_cm" in metadata["video"]:
        PIXELS_PER_CM = metadata["video"]["pixels_per_cm"]
        logger.info(f"Assigning video PIXELS_PER_CM={PIXELS_PER_CM} from metadata.")
    # Otherwise, assign it based on the date of the experiment
    else:
        PIXELS_PER_CM = assign_pixels_per_cm(metadata["datetime"])
        logger.info("No 'pixels_per_cm' value found in video metadata.")
        logger.info(f"Automatically assigned video PIXELS_PER_CM={PIXELS_PER_CM} based on date of experiment.")

    if "hex_centroids_file_path" in metadata["video"]:
        add_hex_centroids(nwbfile=nwbfile, metadata=metadata, pixels_per_cm=PIXELS_PER_CM, logger=logger)
    else:
        print("No subfield 'hex_centroids_file_path' found in video metadata! Skipping adding hex centroids.")
        logger.warning("No subfield 'hex_centroids_file_path' found in video metadata! Skipping adding hex centroids.")

    # It is ok if we have video field in metadata but not DLC data
    # The user may wish to only convert the raw video file and do position tracking later
    if "dlc_path" not in metadata["video"]:
        print("No DeepLabCut (DLC) metadata found for this session. Skipping DLC conversion.")
        logger.info("No DeepLabCut (DLC) metadata found for this session. Skipping DLC conversion.")
        return

    # If we do have dlc_path, we must also have video timestamps for DLC conversion
    if "video_timestamps_file_path" not in metadata["video"]:
        logger.error("Video subfield 'video_timestamps_file_path' not found in metadata. \n"
            "This is required along with 'dlc_path' for DLC position conversion. \n"
            "If you do not wish to convert DeepLabCut data, please remove field 'dlc_path' from metadata."
            )
        print("Video subfield 'video_timestamps_file_path' not found in metadata. \n"
            "This is required along with 'dlc_path' for DLC position conversion. \n"
            "If you do not wish to convert DeepLabCut data, please remove field 'dlc_path' from metadata."
            )
        return
    else:
        # Read timestamps of each camera frame (in ms)
        video_timestamps_file_path = metadata["video"]["video_timestamps_file_path"]
        with open(video_timestamps_file_path, "r") as video_timestamps_file:
            video_timestamps_ms = np.array(list(csv.reader(video_timestamps_file)), dtype=float).ravel()
 
        # Adjust video timestamps so photometry starts at time 0 and convert to seconds to match NWB standard
        video_timestamps_ms = np.subtract(video_timestamps_ms, metadata.get("photometry_start_in_arduino_ms", 0))
        video_timestamps_seconds = video_timestamps_ms / 1000

        # Align video timestamps to photometry/ephys
        ground_truth_visit_times = metadata.get("photometry_visit_times", metadata.get("ephys_visit_times"))
        arduino_visit_times = metadata.get("arduino_visit_times")

        if ground_truth_visit_times is not None:
            logger.info("Aligning DLC timestamps...")
            # Make sure we have the same number of arduino and ground truth visit times for alignment
            assert len(arduino_visit_times) == len(ground_truth_visit_times), (
                f"Expected the same number of port visits recorded by arduino and ephys/photometry! \n"
                f"Got {len(arduino_visit_times)} arduino visits, "
                f"but {len(ground_truth_visit_times)} visits for alignment!"
            )
            # Align video timestamps via interpolation. For timestamps out of visit bounds, 
            # use the ratio of spacing between arduino_visit_times and ground_truth_visit_times
            true_video_timestamps = np.interp(
                x=video_timestamps_seconds,
                xp=arduino_visit_times,
                fp=ground_truth_visit_times,
                left=ground_truth_visit_times[0] + 
                    (video_timestamps_seconds[0] - arduino_visit_times[0]) * 
                    (ground_truth_visit_times[1] - ground_truth_visit_times[0]) / 
                    (arduino_visit_times[1] - arduino_visit_times[0]),
                right=ground_truth_visit_times[-1] + 
                    (video_timestamps_seconds[-1] - arduino_visit_times[-1]) * 
                    (ground_truth_visit_times[-1] - ground_truth_visit_times[-2]) / 
                    (arduino_visit_times[-1] - arduino_visit_times[-2])
            )
        else:
            # If we don't have port visits for alignment, keep the original timestamps
            true_video_timestamps = video_timestamps_seconds

    print("Adding position data from DeepLabCut...")
    logger.info("Adding position data from DeepLabCut...")

    # Metadata should include the full path to the DLC h5 file
    # e.g. Behav_Vid0DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000.h5
    deeplabcut_file_path = metadata["video"]["dlc_path"]

    # Read x, y position data and calculate velocity and acceleration
    position_dfs = read_dlc(deeplabcut_file_path, pixels_per_cm=PIXELS_PER_CM, logger=logger, 
                            likelihood_cutoff=0.9, cam_fps=15)

    # Add x, y position data to the nwbfile
    add_position_to_nwb(nwbfile, position_data=position_dfs, 
                        pixels_per_cm=PIXELS_PER_CM, video_timestamps=true_video_timestamps, logger=logger)