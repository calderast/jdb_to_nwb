import os
import csv
import ffmpeg
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pynwb import NWBFile
from pynwb.image import ImageSeries
from pynwb.behavior import BehavioralEvents
from ndx_franklab_novela import CameraDevice
from hdmf.common import DynamicTable
from .timestamps_alignment import align_via_interpolation, handle_timestamps_reset


def assign_pixels_per_cm(session_date):
    """
    Assigns video pixels_per_cm based on the date of the session.
    pixels_per_cm is 3.14 if video is before IM-1594 (before 01/01/2023), 
    2.3 before 01/11/2024, or 2.688 after (old maze)

    Args:
    session_date (datetime): Datetime object for the date of this session

    Returns:
    float: The corresponding pixels_per_cm value.
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
        pixels_per_cm = 2.688
    return pixels_per_cm


def add_hex_centroids(nwbfile: NWBFile, metadata: dict, logger):
    """
    Read hex centroids from a csv file with columns hex, x, y 
    and add them to the nwbfile in the behavior processing module.
    """

    # If this function is called, it has already been established that 'hex_centroids_file_path' 
    # and 'pixels_per_cm' exist in video metadata, so we don't need any extra checks here
    hex_centroids_file = metadata["video"]["hex_centroids_file_path"]
    pixels_per_cm = metadata["pixels_per_cm"]

    # Try to load centroids and complain if we can't find the file
    try:
        hex_centroids = pd.read_csv(hex_centroids_file)
        logger.debug(f"Found hex centroids file at '{hex_centroids_file}'")
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

    # Make sure the file includes columns 'hex', 'x', 'y'
    required_columns = {"hex", "x", "y"}
    actual_columns = set(hex_centroids.columns)
    if not required_columns.issubset(actual_columns):
        logger.error(f"Expected {required_columns} columns in the hex centroids file, got {actual_columns}!!")
        logger.error("Skipping adding centroids to the nwb")
        return
    else:
        logger.debug(f"Found expected columns {required_columns} in the hex centroids file")

        # Drop any extra columns. (Our centroids code may produce a file with extra columns
        # 'x_meters' and 'y_meters', but we instead calculate those here for consistency)
        extra_columns = actual_columns - required_columns
        if extra_columns:
            logger.debug(f"Dropping extra columns from centroids file: {extra_columns}")
            hex_centroids = hex_centroids[list(required_columns)]  # keep only required columns

    # Convert pixels per cm to meters per pixel
    meters_per_pixel = 0.01 / pixels_per_cm
    hex_centroids['x_meters'] = hex_centroids['x'] * meters_per_pixel
    hex_centroids['y_meters'] = hex_centroids['y'] * meters_per_pixel
    logger.debug(f"Calculating centroids in meters using meters_per_pixel={meters_per_pixel}")

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

    # Add the hex centroids (make sure we keep only desired columns and in the correct order)
    desired_column_order = ["hex", "x", "y", "x_meters", "y_meters"]
    columns_to_keep = list(filter(lambda col: col in hex_centroids.columns, desired_column_order))
    hex_centroids = hex_centroids[columns_to_keep]
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


def compress_avi_to_mp4(input_video_path, output_video_path, logger, crf=23, preset="ultrafast"):
    """
    Compress an AVI video file to MP4 using ffmpeg.

    Args:
    input_video_path: Path to the input AVI file
    output_video_path: Path to save the compressed MP4 file
    crf (int): ffmpeg Constant Rate Factor (controls quality/size trade-off). \
        Lower = better quality. Range: 0 (lossless compression) to 51 (bad quality)
    preset (str): ffmpeg compression speed preset: \
        "ultrafast", "superfast", "fast", "medium", "slow", "slower", "veryslow"
    """
    try:
        (
            ffmpeg.input(str(input_video_path)).output(str(output_video_path), vcodec="libx264", crf=crf, preset=preset)
            .run(overwrite_output=True, capture_stdout=False, capture_stderr=True)
        )
        logger.info(f"Compressed video at {input_video_path} to {output_video_path}")
        print(f"Compressed video at {input_video_path} to {output_video_path}")
    except Exception as e:
        logger.error(f"An error occurred during video compression: {str(e)}")
        print("An error occurred during video compression:")
        print(str(e))


def add_camera(nwbfile: NWBFile, metadata: dict):
    '''
    Adds camera because this is required to create a TaskEpoch in spyglass.
    Camera metadata (lens, model) is currently placeholder values.    
    '''

    pixels_per_cm = metadata["pixels_per_cm"]
    meters_per_pixel = 0.01 / pixels_per_cm
    mpp_rounded = f"{meters_per_pixel:.6f}"
    # In spyglass, the camera_name is used as a primary key. 
    # We add the meters_per_pixel to the camera_name so we get a distinct
    # camera entry in the table each time the meters_per_pixel changes.

    # Note that the name "camera_device 1" is required for spyglass compatibility
    # (it must match format "camera_device {epoch+1}")
    nwbfile.add_device(
        CameraDevice(
            name="camera_device 1",
            meters_per_pixel=meters_per_pixel,
            manufacturer="Logitech",
            model="Brio webcam",
            lens="lens",
            camera_name=f"berke_maze_cam_{mpp_rounded}m_per_pixel",
        )
    )


def add_video(nwbfile: NWBFile, metadata: dict, output_video_path, logger):
    """
    Add video and related (camera, hex centroids) data to the nwbfile.
    """

    # NOTE: We do the pixels_per_cm assignment before the video conversion 
    # so we can add camera data regardless of if we have video data to add.
    # This is a bit odd, but we need the CameraDevice to exist so the tasks
    # processing module can be added to the TaskEpoch table in spyglass
    # (We assign it directly to metadata instead of metadata["video"] so 
    # the omission of the video key can still signal if we don't have video data)

    # If pixels_per_cm exists in metadata, use that value
    if "video" in metadata and "pixels_per_cm" in metadata["video"]:
        pixels_per_cm = metadata["video"]["pixels_per_cm"]
        logger.info(f"Assigning video pixels_per_cm={pixels_per_cm} from metadata.")
    # Otherwise, assign it based on the date of the experiment
    else:
        pixels_per_cm = assign_pixels_per_cm(metadata["datetime"])
        logger.info("No 'pixels_per_cm' value found in video metadata.")
        logger.info(f"Automatically assigned video pixels_per_cm={pixels_per_cm} based on date of experiment.")
    # Add it to metadata so we can use it for position conversion if needed
    metadata["pixels_per_cm"] = pixels_per_cm

    # Add camera once we ensure pixels_per_cm is in metadata
    print("Adding camera...")
    logger.info("Adding camera...")
    add_camera(nwbfile=nwbfile, metadata=metadata)

    # Now check if we actually have video data to add
    if "video" not in metadata:
        print("No video metadata found for this session. Skipping video conversion.")
        logger.warning("No video metadata found for this session. Skipping video conversion.")
        return None

    if "hex_centroids_file_path" in metadata["video"]:
        add_hex_centroids(nwbfile=nwbfile, metadata=metadata, logger=logger)
    else:
        print("No subfield 'hex_centroids_file_path' found in video metadata! Skipping adding hex centroids.")
        logger.warning("No subfield 'hex_centroids_file_path' found in video metadata! Skipping adding hex centroids.")

    if "video_file_path" not in metadata["video"] or "video_timestamps_file_path" not in metadata["video"]:
        print("Skipping video file conversion (requires both 'video_file_path' and 'video_timestamps_file_path')")
        logger.warning("Skipping video file conversion "
                       "(requires both 'video_file_path' and 'video_timestamps_file_path')")
        # Warn but don't raise an error here because it is technically ok for a user to specify the "video"
        # field in metadata but not the actual video data, because DLC (position) also lives under the video field
        return None

    print("Adding video...")
    logger.info("Adding video...")

    # Get file paths for video from metadata file
    video_file_path = metadata["video"]["video_file_path"]
    video_timestamps_file_path = metadata["video"]["video_timestamps_file_path"]

    # Read timestamps of each camera frame (in ms)
    with open(video_timestamps_file_path, "r") as video_timestamps_file:
        video_timestamps_ms = np.array(list(csv.reader(video_timestamps_file)), dtype=float).ravel()

    # Check for and handle potential timestamps reset (happens when the recording passes 12:00pm)
    video_timestamps_ms = handle_timestamps_reset(timestamps=video_timestamps_ms, logger=logger)

    # Adjust video timestamps so photometry starts at time 0 (this is also done to match arduino visit times)
    video_timestamps_ms = np.subtract(video_timestamps_ms, metadata.get("photometry_start_in_arduino_ms", 0))

    # Convert video timestamps to seconds to match NWB standard
    video_timestamps_seconds = video_timestamps_ms / 1000

    # Get port visits in video time (aka arduino time)
    arduino_visit_times = metadata.get("arduino_visit_times")

    # If we have ground truth port visit times, align video timestamps to that
    ground_truth_time_source = metadata.get("ground_truth_time_source")
    if ground_truth_time_source is not None:

        logger.info(f"Aligning video to ground truth ({ground_truth_time_source})")
        ground_truth_visit_times = metadata.get("ground_truth_visit_times")
        true_video_timestamps = align_via_interpolation(unaligned_timestamps=video_timestamps_seconds,
                                                   unaligned_visit_times=arduino_visit_times,
                                                   ground_truth_visit_times=ground_truth_visit_times,
                                                   logger=logger)
    else:
        # If we don't have port visits for alignment, keep the original timestamps
        logger.info("No ground truth port visits found, keeping original video timestamps.")
        true_video_timestamps = video_timestamps_seconds

    logger.debug("Difference between aligned and original video timestamps: "
                 f"{np.array(true_video_timestamps) - np.array(video_timestamps_seconds)}")

    # Convert video from .avi to .mp4 and copy it to the nwb output directory
    print("Compressing video from .avi to .mp4 and copying to nwb output directory...")
    logger.info("Compressing video from .avi to .mp4 and copying to nwb output directory...")
    compress_avi_to_mp4(input_video_path=video_file_path, output_video_path=output_video_path, logger=logger)

    # Create nwb processing module for video files
    nwbfile.create_processing_module(
        name="video_files", description="Contains all associated video files data"
    )
    # Create a BehavioralEvents object to hold videos
    video = BehavioralEvents(name="video")

    # The converted video is saved in the same directory as the nwb, 
    # so the path relative to the nwbfile is just the video file name
    video_file_name = os.path.basename(output_video_path)

    video.add_timeseries(
        ImageSeries(
            device=nwbfile.devices["camera_device 1"],
            name="behavior_video",
            timestamps=true_video_timestamps,
            external_file=[video_file_name],
            format="external",
            starting_frame=[0],
            description="Video of animal behavior in the hex maze",
        )
    )

    nwbfile.processing["video_files"].add(video)
    logger.info("Created nwb processing module for video files and added behavior_video as an nwb ImageSeries")