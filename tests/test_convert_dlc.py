from datetime import datetime
from dateutil import tz
from pynwb import NWBFile
from pathlib import Path

from jdb_to_nwb.convert_dlc import add_dlc


def test_add_dlc_one_bodypart():
    """
    Test the add_dlc function where the DeepLabCut h5 file has position data 
    for a single bodypart ('cap')
    """

    test_data_dir = Path("tests/test_data/downloaded/IM-1770_corvette/11062024")
    
    metadata = {}
    metadata["date"] = "11062024"
    metadata["video"] = {}
    metadata["video"]["video_file_path"] = test_data_dir / "Behav_Vid0.avi"
    metadata["video"]["video_timestamps_file_path"] = test_data_dir / "testvidtimes0.csv"
    metadata["video"]["dlc_path"] = test_data_dir / "Behav_Vid0DLC_resnet50_Triangle_Maze_PhotFeb12shuffle1_800000.h5"

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_dlc(nwbfile=nwbfile, metadata=metadata)

    # Check that behavior processing module has been added
    assert "behavior" in nwbfile.processing

    # Check that the "position" object exists in the behavior processing module
    behavior_module = nwbfile.processing["behavior"]
    assert "position" in behavior_module.data_interfaces
    position_object = behavior_module.data_interfaces["position"]

    # This file contains a single set of position data
    expected_bodyparts = ["cap"]
    
    # Check that data for all body parts has been added to the nwb
    for body_part in expected_bodyparts:
    
        # Check that the x,y SpatialSeries was added to the "position" object
        spatial_series_name = f"{body_part}_position"
        assert spatial_series_name in position_object.spatial_series, (
            f"Expected SpatialSeries '{spatial_series_name}' is missing."
        )
        spatial_series = position_object.spatial_series[spatial_series_name]

        # Validate some SpatialSeries metadata
        assert spatial_series.unit == "meters"
        assert spatial_series.reference_frame == "Upper left corner of video frame"

        # Check that the DLC likelihood TimeSeries exists
        dlc_likelihood_name = f"DLC_likelihood_{body_part}"
        assert dlc_likelihood_name in behavior_module.data_interfaces
        dlc_likelihood = behavior_module.data_interfaces[dlc_likelihood_name]

        # Validate some DLC likelihood metadata
        assert dlc_likelihood.unit == "fraction"
        expected_comment = f"Likelihood of each x,y coordinate for tracked bodypart '{body_part}'"
        assert dlc_likelihood.comments.startswith(expected_comment)


def test_add_dlc_two_bodyparts():
    """
    Test the add_dlc function where the DeepLabCut h5 file has position data 
    for 2 bodyparts ('cap_front' and 'cap_back')
    """
    
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")

    metadata = {}
    metadata["date"] = "07252022"
    metadata["video"] = {}
    metadata["video"]["video_file_path"] = test_data_dir / "Behav_Vid0.avi"
    metadata["video"]["video_timestamps_file_path"] = test_data_dir / "testvidtimes0.csv"
    metadata["video"]["dlc_path"] = test_data_dir / "Behav_Vid0DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000.h5"

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_dlc(nwbfile=nwbfile, metadata=metadata)

    # Check that behavior processing module has been added
    assert "behavior" in nwbfile.processing

    # Check that the "position" object exists in the behavior processing module
    behavior_module = nwbfile.processing["behavior"]
    assert "position" in behavior_module.data_interfaces
    position_object = behavior_module.data_interfaces["position"]

    # This file contains 2 sets of position data
    expected_bodyparts = ["cap_front", "cap_back"]
    
    # Check that data for all body parts has been added to the nwb
    for body_part in expected_bodyparts:
    
        # Check that the x,y SpatialSeries was added to the "position" object
        spatial_series_name = f"{body_part}_position"
        assert spatial_series_name in position_object.spatial_series, (
            f"Expected SpatialSeries '{spatial_series_name}' is missing."
        )
        spatial_series = position_object.spatial_series[spatial_series_name]

        # Validate some SpatialSeries metadata
        assert spatial_series.unit == "meters"
        assert spatial_series.reference_frame == "Upper left corner of video frame"

        # Check that the DLC likelihood TimeSeries exists
        dlc_likelihood_name = f"DLC_likelihood_{body_part}"
        assert dlc_likelihood_name in behavior_module.data_interfaces
        dlc_likelihood = behavior_module.data_interfaces[dlc_likelihood_name]

        # Validate some DLC likelihood metadata
        assert dlc_likelihood.unit == "fraction"
        expected_comment = f"Likelihood of each x,y coordinate for tracked bodypart '{body_part}'"
        assert dlc_likelihood.comments.startswith(expected_comment)


def test_add_dlc_with_incomplete_metadata(capsys):
    """
    Test that the add_dlc function responds appropriately to missing or incomplete metadata.
    
    If no 'video' key is in the metadata dictionary, it should silently return and 
    skip conversion.
    
    If there is a 'video' key in the metadata dict but no dlc_path subfield, print that
    we are skipping DLC conversion
    
    If there is a 'dlc_path' subfield but no 'video_timestamps_file_path', raise a ValueError
    """
    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # 1. Test with no 'video' key
    metadata = {}

    # Call the add_dlc function with no 'video' key in metadata and see there are no errors
    add_dlc(nwbfile=nwbfile, metadata=metadata)
    
    
    # 2. Test with 'video' key and no 'dlc_path'
    metadata["video"] = {}

    # Call the add_video function with 'video' key in metadata, but no 'dlc_path'
    add_dlc(nwbfile=nwbfile, metadata=metadata)
    captured = capsys.readouterr() # capture stdout

    # Check that the correct message was printed to stdout
    assert "No DeepLabCut (DLC) metadata found for this session. Skipping DLC conversion." in captured.out
    
    
    # 3. Test with 'video' key and 'dlc_path', but no 'video_timestamps_file_path'
    metadata["video"] = {}
    metadata["video"]["dlc_path"] = (
        "tests/test_data/downloaded/IM-1478/07252022/Behav_Vid0DLC_resnet50_Triangle_Maze_EphysDec7shuffle1_800000.h5"
    )

    # Check that add_dlc raises a ValueError about missing fields in the metadata dictionary
    try:
        add_dlc(nwbfile=nwbfile, metadata=metadata)
    except ValueError as e:
        assert str(e).startswith("Video subfield 'video_timestamps_file_path' not found in metadata.")
    else:
        assert False, (
            "Expected ValueError was not raised in response to "
            "missing timestamp subfields in the metadata dict."
        )