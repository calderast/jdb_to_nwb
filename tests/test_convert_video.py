from datetime import datetime
from dateutil import tz
from pynwb import NWBFile

from jdb_to_nwb.convert_video import add_video


def test_add_video():
    """Test the add_video function."""

    metadata = {}
    metadata["video"] = {}
    metadata["video"]["video_file_path"] = "tests/test_data/downloaded/IM-1478/07252022/Behav_Vid0.avi"
    metadata["video"]["video_timestamps_file_path"] = "tests/test_data/downloaded/IM-1478/07252022/testvidtimes0.csv"

    test_output_video_path = "tests/test_data/downloaded/output/test_video.mp4"

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_video(nwbfile=nwbfile, metadata=metadata, output_video_path=test_output_video_path)

    # Test that the nwbfile has the expected associated video file
    assert "video_files" in nwbfile.processing
    video_module = nwbfile.processing["video_files"]

    assert "video" in video_module.data_interfaces
    behavioral_events = video_module.data_interfaces["video"]

    assert "behavior_video" in behavioral_events.time_series
    behavior_video = behavioral_events.time_series["behavior_video"]

    # Validate video timeseries metadata
    assert behavior_video.external_file == ["test_video.mp4"]
    assert behavior_video.format == "external"
    assert behavior_video.description == "Video of animal behavior in the hex maze"


def test_add_video_with_incomplete_metadata(capsys):
    """
    Test that the add_video function responds appropriately to missing or incomplete metadata.
    
    If no 'video' key is in the metadata dictionary, it should print that we are skipping 
    video conversion and move on without raising any errors.
    
    If there is a 'video' key in the metadata dict but no video subfields, also print that
    we are skipping conversion
    """

    # Create a test metadata dictionary with no video key
    metadata = {}

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Test output video path (doesn't matter, conversion should stop before then)
    test_output_video_path = "tests/test_data/downloaded/output/test_video.mp4"

    # Call the add_video function with no 'video' key in metadata
    add_video(nwbfile=nwbfile, metadata=metadata, output_video_path=test_output_video_path)
    captured = capsys.readouterr() # capture stdout

    # Check that the correct message was printed to stdout
    assert "No video metadata found for this session. Skipping video conversion." in captured.out

    # Create a test metadata dictionary with a video field but no video data
    metadata["video"] = {}

    # Call the add_video function with 'video' key in metadata, but no video subfields
    add_video(nwbfile=nwbfile, metadata=metadata, output_video_path=test_output_video_path)
    captured = capsys.readouterr() # capture stdout

    # Check that the skip message was printed to stdout
    # This should print with no error, because it is ok if we have 'video' but not these subfields, 
    # because maybe we only have DLC and not raw video
    assert "Skipping video file conversion" in captured.out
