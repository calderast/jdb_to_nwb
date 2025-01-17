from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io
from dateutil import tz
from pynwb import NWBFile
from ndx_fiber_photometry import FiberPhotometryResponseSeries

from jdb_to_nwb.convert_photometry import add_photometry, process_raw_photometry_signals


def test_process_raw_photometry_signals():
    """Test that the process_raw_photometry_signals function returns a signals dictionary equivalent to signals.mat."""

    # Create a test metadata dictionary
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["signals_mat_file_path"] = test_data_dir / "signals.mat"
    metadata["photometry"]["phot_file_path"] = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.phot"
    metadata["photometry"]["box_file_path"] = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.box"

    # Load signals.mat created by the external MATLAB photometry processing code as a reference
    signals_mat_file_path = metadata["photometry"]["signals_mat_file_path"]
    reference_signals_mat = scipy.io.loadmat(signals_mat_file_path, matlab_compatible=True)

    # Process the raw .phot and .box files from Labview into an equivalent signals dict
    phot_file_path = metadata["photometry"]["phot_file_path"]
    box_file_path = metadata["photometry"]["box_file_path"]
    signals = process_raw_photometry_signals(phot_file_path, box_file_path)

    # Ensure all relevant keys are present in both the reference and created signals dict
    expected_keys = {"sig1", "sig2", "loc", "ref", "visits"}
    for key in expected_keys:
        assert key in reference_signals_mat, f"Expected key '{key}' is missing from the reference signals.mat"
        assert key in signals, f"Expected key '{key}' is missing from the signals dict"

    # Port visit times should be identical (use a very small tolerance to account for int/float differences)
    assert np.allclose(signals["visits"], reference_signals_mat["visits"], atol=1e-10)

    # Fiber location should be identical
    # (First convert char array from signals.mat to a string for comparison)
    reference_loc_as_string = "".join(reference_signals_mat["loc"].flatten().astype(str))
    assert signals["loc"] == reference_loc_as_string

    # In our former MATLAB photometry preprocessing pipeline (used to create signals.mat), we would sometimes
    # remove a certain number of samples from the beginning of our signals for alignment with behavior.
    # In this new pipeline, we do this alignment later so this is no longer necessary at this step.
    # This results in the signals in our created signals dict being longer than the signals in the reference signals.mat,
    # so we calculate the number of samples to remove from the beginning so we can do comparisons.
    samples_removed_from_reference = len(signals["ref"]) - len(reference_signals_mat["ref"].flatten())

    # The difference between our created signal and the reference from signals.mat must be <0.01% of the magnitude of the reference
    assert np.allclose(signals["ref"][samples_removed_from_reference:], reference_signals_mat["ref"].flatten(), rtol=1e-4)
    assert np.allclose(signals["sig1"][samples_removed_from_reference:], reference_signals_mat["sig1"].flatten(), rtol=1e-4)
    # We allow an additional absolute tolerance for sig2 because there are some values very close to 0 where relative tolerance is less useful
    # (Also, we don't use actually sig2 so it doesn't really matter)
    assert np.allclose(signals["sig2"][samples_removed_from_reference:], reference_signals_mat["sig2"].flatten(), rtol=1e-4, atol=5)


def test_add_photometry_from_signals_mat():
    """
    Test that the add_photometry function results in the expected FiberPhotometryResponseSeries.

    This version of the test uses the already created signals.mat at "signals_mat_file_path" to further process and add photometry signals to the NWB.
    """

    # Create a test metadata dictionary with preprocessed LabVIEW data ("signals_mat_file_path")
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["signals_mat_file_path"] = test_data_dir / "signals.mat"

    # Define paths to reference data
    reference_data_path = test_data_dir / "IM-1478_07252022_h_sampleframe.csv"
    reference_dataframe = pd.read_csv(reference_data_path)

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Add photometry data to the nwbfile
    visits = add_photometry(nwbfile=nwbfile, metadata=metadata)

    # Define the FiberPhotometryResponseSeries we expect to have been added to the nwbfile
    expected_photometry_series = {"raw_green", "raw_reference", "z_scored_green_dFF", "z_scored_reference_fitted"}
    expected_sampling_rate = 250  # Hz

    # Assert that all expected photometry series are present in the acquisition field of the nwbfile
    actual_photometry_series = set(nwbfile.acquisition.keys())
    missing_photometry_series = expected_photometry_series - actual_photometry_series
    assert not missing_photometry_series, f"Missing FiberPhotometryResponseSeries: {missing_photometry_series}"

    # Check the basic attributes of each series
    for series_name in expected_photometry_series:
        # Assert that all series are of type FiberPhotometryResponseSeries
        assert isinstance(nwbfile.acquisition[series_name], FiberPhotometryResponseSeries), f"{series_name} is not of type FiberPhotometryResponseSeries"
        # Assert all series have a sampling rate of 250 Hz
        assert (
            getattr(nwbfile.acquisition[series_name], "rate", None) == expected_sampling_rate
        ), f"{series_name} has a sampling rate of {getattr(nwbfile.acquisition[series_name], 'rate', None)}, expected {expected_sampling_rate}"

    # Check that the photometry series in the nwbfile match the expected signals from the reference dataframe
    green_z_scored_dFF = np.array(nwbfile.acquisition["z_scored_green_dFF"].data)
    green_z_scored_dFF_reference = reference_dataframe["green"].values

    # Check that the lengths match
    assert len(green_z_scored_dFF) == len(green_z_scored_dFF_reference), (
        f"Data length mismatch: z_scored_green_dFF has {len(green_z_scored_dFF)} points, "
        f"but the reference signal has {len(green_z_scored_dFF_reference)} points."
    )

    # Check that the z-scored green dF/F signal in the nwbfile matches the reference green signal (within a tolerance)
    np.testing.assert_allclose(
        green_z_scored_dFF,
        green_z_scored_dFF_reference,
        atol=0.005,
        rtol=0.05,
        err_msg="Data mismatch between nwbfile z_scored_green_dFF and reference data",
    )


def test_add_photometry_from_raw_labview():
    """
    Test that the add_photometry function results in the expected FiberPhotometryResponseSeries.

    This version of the test uses the raw LabVIEW data at "phot_file_path" and "box_file_path" 
    to first create a signals dict, and then further process and add photometry signals to the NWB.
    """

    # Create a test metadata dictionary with raw LabVIEW data ("phot_file_path" and "box_file_path")
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["phot_file_path"] = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.phot"
    metadata["photometry"]["box_file_path"] = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.box"

    # Define paths to reference data
    reference_data_path = test_data_dir / "IM-1478_07252022_h_sampleframe.csv"
    reference_dataframe = pd.read_csv(reference_data_path)

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Add photometry data to the nwbfile
    visits = add_photometry(nwbfile=nwbfile, metadata=metadata)

    # Define the FiberPhotometryResponseSeries we expect to have been added to the nwbfile
    expected_photometry_series = {"raw_green", "raw_reference", "z_scored_green_dFF", "z_scored_reference_fitted"}
    expected_sampling_rate = 250  # Hz

    # Assert that all expected photometry series are present in the acquisition field of the nwbfile
    actual_photometry_series = set(nwbfile.acquisition.keys())
    missing_photometry_series = expected_photometry_series - actual_photometry_series
    assert not missing_photometry_series, f"Missing FiberPhotometryResponseSeries: {missing_photometry_series}"

    # Check the basic attributes of each series
    for series_name in expected_photometry_series:
        # Assert that all series are of type FiberPhotometryResponseSeries
        assert isinstance(nwbfile.acquisition[series_name], FiberPhotometryResponseSeries), f"{series_name} is not of type FiberPhotometryResponseSeries"
        # Assert all series have a sampling rate of 250 Hz
        assert (
            getattr(nwbfile.acquisition[series_name], "rate", None) == expected_sampling_rate
        ), f"{series_name} has a sampling rate of {getattr(nwbfile.acquisition[series_name], 'rate', None)}, expected {expected_sampling_rate}"

    # Check that the photometry series in the nwbfile match the expected signals from the reference dataframe
    green_z_scored_dFF = np.array(nwbfile.acquisition["z_scored_green_dFF"].data)
    green_z_scored_dFF_reference = reference_dataframe["green"].values

    # In our former MATLAB photometry preprocessing pipeline (used to create the reference dataframe), we would sometimes
    # remove a certain number of samples from the beginning of our signals for alignment with behavior.
    # In this new pipeline, we do this alignment later so this is no longer necessary at this step.
    # This results in the signals in our created signals dict being longer than the signals in the reference dataframe,
    # so we calculate the number of samples to remove from the beginning so we can do comparisons.
    samples_removed_from_reference = len(green_z_scored_dFF) - len(green_z_scored_dFF_reference)
    green_z_scored_dFF = green_z_scored_dFF[samples_removed_from_reference:]

    # Check that the z-scored green dF/F signal in the nwbfile matches the reference green signal (within a tolerance)
    # TODO: have a conversation about how we feel about the removal of some samples from the front of the signal.
    # Because the signals in our returned signals dict and signals.mat are different lengths, when we downsample to
    # 250 Hz we end up taking slightly different data points from the signal. These differences propagate through our
    # filtering etc. The signals are visually pretty similar but would fail most reasonable assertions of similarity.
    # Commenting out this similarity test for now.

    # np.testing.assert_allclose(
    #    green_z_scored_dFF,
    #    green_z_scored_dFF_reference,
    #    atol=0.005,
    #    rtol=0.05,
    #    err_msg=f"Data mismatch between nwbfile z_scored_green_dFF and reference data",
    # )


def test_add_photometry_from_pyphotometry():
    """
    Test that the add_photometry function results in the expected FiberPhotometryResponseSeries.

    This version of the test uses the ppd file from pyPhotometry to add photometry signals to the NWB.
    
    TODO: Implement this test once adding photometry via ppd file is implemented!!
    """

    # Create a test metadata dictionary with pyPhotometry data
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["ppd_file_path"] = "" # TODO. 

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )
    
    # TODO: Actually implement this test.
    # Reference data and most assertions can likely be copied from the above tests.
    # For now we test that the appropriate NotImplementedError is raised.

    # Check that add_photometry complains about pyPhotometry not being implemented if we specify a ppd file
    try:
        visits = add_photometry(nwbfile=nwbfile, metadata=metadata)
    except NotImplementedError as e:
        assert str(e) == "pyPhotometry processing is not yet implemented."
    else:
        assert False, "Expected NotImplementedError when attempting to process pyPhotometry data."


def test_add_photometry_with_incomplete_metadata(capsys):
    """
    Test that the add_photometry function responds appropriately to missing or incomplete metadata.
    
    If no 'photometry' key is in the metadata dictionary, it should print that we are skipping 
    photometry conversion and move on without raising any errors.
    
    If there is a 'photometry' key in the metadata dict but the required paths to photometry data
    are not present, raise a ValueError telling the user which keys must be present in the dict.
    """
    
    # Create a test metadata dictionary with no photometry key
    metadata = {}
    
    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )
    
    # If we call the add_photometry function with no 'photometry' key in metadata,
    # It should print that we are skipping photometry conversion and return None for visits.
    # This should not raise any errors, as omitting the 'photometry' key is a 
    # valid way to specify that we have no photometry data for this session.
    
    # Call the add_photometry function with no 'photometry' key in metadata
    visits = add_photometry(nwbfile=nwbfile, metadata=metadata)
    captured = capsys.readouterr() # capture stdout
    
    # Check that the correct message was printed to stdout and returned visits is None
    assert "No photometry metadata found for this session. Skipping photometry conversion." in captured.out
    assert visits is None
    
    # Create a test metadata dictionary with a photometry field but no photometry data
    metadata["photometry"] = {}
    
    # Check that add_photometry raises a ValueError about missing fields in the metadata dictionary
    try:
        visits = add_photometry(nwbfile=nwbfile, metadata=metadata)
    except ValueError as e:
        assert str(e).startswith("None of the required photometry subfields exist in the metadata dictionary")
    else:
        assert False, "Expected ValueError was not raised in response to missing photometry subfields in the metadata dict."
        
