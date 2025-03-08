import os
import csv
import ffmpeg
import numpy as np
from pynwb import NWBFile
from pynwb.image import ImageSeries
from pynwb.behavior import BehavioralEvents
from ndx_franklab_novela import CameraDevice


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


def add_camera(nwbfile: NWBFile):
    '''
    Adds camera because this is required to create a TaskEpoch in spyglass.
    Data is currently placeholder values.
    '''
    
    pixels_per_cm = 3.14
    meters_per_pixel = 0.01 / pixels_per_cm
    
    nwbfile.add_device(
        CameraDevice(
            name="camera_device 1",
            meters_per_pixel=meters_per_pixel,
            manufacturer="Logitech",
            model="Brio webcam",
            lens="lens",
            camera_name="maze_camera",
        )
    )


def add_video(nwbfile: NWBFile, metadata: dict, output_video_path, logger):

    if "video" not in metadata:
        print("No video metadata found for this session. Skipping video conversion.")
        logger.warning("No video metadata found for this session. Skipping video conversion.")
        return None

    if "video_file_path" not in metadata["video"] or "video_timestamps_file_path" not in metadata["video"]:
        print("Skipping video file conversion (requires both 'video_file_path' and 'video_timestamps_file_path')")
        logger.warning("Skipping video file conversion "
                       "(requires both 'video_file_path' and 'video_timestamps_file_path')")
        # Don't raise an error here because it is technically ok for a user to specify the "video"
        # field in metadata but not the actual video data, because DLC also lives under the video field.
        return None

    print("Adding video...")
    logger.info("Adding video...")
    
    print("Adding camera...")
    logger.info("Adding camera...")
    add_camera(nwbfile)

    # Get file paths for video from metadata file
    video_file_path = metadata["video"]["video_file_path"]
    video_timestamps_file_path = metadata["video"]["video_timestamps_file_path"]

    # Read timestamps of each camera frame (in ms)
    with open(video_timestamps_file_path, "r") as video_timestamps_file:
        video_timestamps_ms = np.array(list(csv.reader(video_timestamps_file)), dtype=float).ravel()

    # Adjust video timestamps so photometry starts at time 0
    video_timestamps_ms = np.subtract(video_timestamps_ms, metadata.get("photometry_start_in_arduino_ms", 0))

    # Convert video timestamps to seconds to match NWB standard
    video_timestamps_seconds = video_timestamps_ms / 1000

    # Align video timestamps to photometry/ephys
    ground_truth_visit_times = metadata.get("photometry_visit_times", metadata.get("ephys_visit_times"))
    arduino_visit_times = metadata.get("arduino_visit_times")

    if ground_truth_visit_times is not None:
        logger.info("Aligning video timestamps...")
        # Make sure we have the same number of arduino and ground truth visit times for alignment
        assert len(arduino_visit_times) == len(ground_truth_visit_times), (
            f"Expected the same number of port visits recorded by arduino and ephys/photometry! \n"
            f"Got {len(arduino_visit_times)} arduino visits, but {len(ground_truth_visit_times)} visits for alignment!"
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
