from datetime import datetime
from dateutil import tz
from zoneinfo import ZoneInfo
from pynwb import NWBFile
from pathlib import Path
from ndx_franklab_novela import CameraDevice
from hdmf.common.table import DynamicTable, VectorData

from jdb_to_nwb.convert_video import add_video, add_hex_centroids, add_camera


def test_add_hex_centroids(dummy_logger):
    """
    Test the add_hex_centroids function
    """

    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")

    metadata = {}
    metadata["pixels_per_cm"] = 3.14
    metadata["video"] = {}
    metadata["video"]["hex_centroids_file_path"] = test_data_dir / "hex_coordinates_IM-1478_07252022.csv"

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Add centroids table to the nwb
    add_hex_centroids(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Check that behavior processing module has been added
    assert "behavior" in nwbfile.processing

    # Check that the "hex_centroids" table exists in the behavior processing module
    behavior_module = nwbfile.processing["behavior"]
    assert "hex_centroids" in behavior_module.data_interfaces
    hex_centroids_table = behavior_module.data_interfaces["hex_centroids"]

    assert isinstance(hex_centroids_table, DynamicTable)
    assert hex_centroids_table.name == "hex_centroids"
    assert hex_centroids_table.description == "Centroids of each hex in the maze (in video pixel coordinates)"

    # The table should have data for all 49 hex centroids
    assert len(hex_centroids_table) == 49

    # Check that the table contains the correct columns
    for column in hex_centroids_table.columns:
        assert isinstance(column, VectorData)
    expected_columns = {"hex", "x", "y", "x_meters", "y_meters"}
    assert set(hex_centroids_table.colnames) == expected_columns, (
        f"Hex centroid columns {set(hex_centroids_table.colnames)} "
        f"did not match expected {expected_columns}"
    )


def test_add_camera(dummy_logger):
    """
    Test the add_camera function
    """

    metadata = {}
    metadata["video"] = {}
    metadata["pixels_per_cm"] = 3.14

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Add a CameraDevice to the nwb
    add_camera(nwbfile=nwbfile, metadata=metadata)

    # Check that the CameraDevice has been added
    assert "camera_device 1" in nwbfile.devices
    camera_device = nwbfile.devices["camera_device 1"]
    assert isinstance(camera_device, CameraDevice)
    assert camera_device.camera_name == "berke_maze_cam_0.003185m_per_pixel"
    assert camera_device.meters_per_pixel == 0.01 / metadata["pixels_per_cm"]
    assert camera_device.manufacturer == "Logitech"


def test_add_video(dummy_logger):
    """Test the add_video function."""

    metadata = {}
    metadata["video"] = {}
    metadata["video"]["video_file_path"] = "tests/test_data/downloaded/IM-1478/07252022/Behav_Vid0.avi"
    metadata["video"]["video_timestamps_file_path"] = "tests/test_data/downloaded/IM-1478/07252022/testvidtimes0.csv"
    metadata["datetime"] = datetime.strptime("07252022", "%m%d%Y").replace(tzinfo=ZoneInfo("America/Los_Angeles")) 

    test_output_video_path = "tests/test_data/downloaded/output/test_video.mp4"

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_video(nwbfile=nwbfile, metadata=metadata, 
              output_video_path=test_output_video_path, logger=dummy_logger)

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


def test_add_video_with_incomplete_metadata(capsys, dummy_logger):
    """
    Test that the add_video function responds appropriately to missing or incomplete metadata.
    
    If no 'video' key is in the metadata dictionary, it should print that we are skipping 
    video conversion and move on without raising any errors.
    
    If there is a 'video' key in the metadata dict but no video subfields, also print that
    we are skipping conversion
    """

    # Create a test metadata dictionary with no video key
    metadata = {}
    metadata["datetime"] = datetime.strptime("07252022", "%m%d%Y").replace(tzinfo=ZoneInfo("America/Los_Angeles")) 

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Test output video path (doesn't matter, conversion should stop before then)
    test_output_video_path = "tests/test_data/downloaded/output/test_video.mp4"

    # Call the add_video function with no 'video' key in metadata
    add_video(nwbfile=nwbfile, metadata=metadata, 
              output_video_path=test_output_video_path, logger=dummy_logger)
    captured = capsys.readouterr() # capture stdout

    # Check that the correct message was printed to stdout
    assert "No video metadata found for this session. Skipping video conversion." in captured.out

    # Create a test metadata dictionary with a video field but no video data
    metadata["video"] = {}
    metadata["datetime"] = datetime.strptime("07252022", "%m%d%Y").replace(tzinfo=ZoneInfo("America/Los_Angeles")) 
    
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Call the add_video function with 'video' key in metadata, but no video subfields
    add_video(nwbfile=nwbfile, metadata=metadata, 
              output_video_path=test_output_video_path, logger=dummy_logger)
    captured = capsys.readouterr() # capture stdout

    # Check that the skip message was printed to stdout
    # This should print with no error, because it is ok if we have 'video' but not these subfields, 
    # because maybe we only have DLC and not raw video
    assert "Skipping video file conversion" in captured.out
    assert "No subfield 'hex_centroids_file_path' found in video metadata!" in captured.out