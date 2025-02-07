import os
import csv
import ffmpeg
import numpy as np
from pynwb import NWBFile
from pynwb.image import ImageSeries
from pynwb.behavior import BehavioralEvents


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

    # Get file paths for video from metadata file
    video_file_path = metadata["video"]["video_file_path"]
    video_timestamps_file_path = metadata["video"]["video_timestamps_file_path"]

    # Read timestamps of each camera frame (in ms)
    with open(video_timestamps_file_path, "r") as video_timestamps_file:
        video_timestamps_ms = np.array(list(csv.reader(video_timestamps_file)), dtype=float).ravel()

    # Convert video timestamps to seconds to match NWB standard
    video_timestamps_seconds = video_timestamps_ms / 1000

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
            timestamps=video_timestamps_seconds,
            external_file=[video_file_name],
            format="external",
            starting_frame=[0],
            description="Video of animal behavior in the hex maze",
        )
    )

    nwbfile.processing["video_files"].add(video)
    logger.info("Created nwb processing module for video files and added behavior_video as an nwb ImageSeries")
