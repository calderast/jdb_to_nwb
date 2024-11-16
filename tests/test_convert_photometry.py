from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import tz
from pynwb import NWBFile
from ndx_fiber_photometry import FiberPhotometryResponseSeries

from jdb_to_nwb.convert_photometry import add_photometry


def test_add_photometry():
    """Test that the add_photometry function results in the expected FiberPhotometryResponseSeries."""

    # Test data (signals.mat) and reference data (sampleframe.csv) are currently in T:\ACh Rats\80B8CE6_ceecee\02222024-L
    # TODO: Modify this to download some test data and reference data from the OneDrive so this can be run from anywhere

    # Create a test metadata dictionary
    metadata = {}
    metadata["photometry"] = {}
    # metadata["photometry"]["signals_mat_file_path"] = 'T:/ACh Rats/80B8CE6_ceecee/02222024-L/signals.mat' # Jose's version
    metadata["photometry"]["signals_mat_file_path"] = "/Volumes/Jose/ACh Rats/80B8CE6_ceecee/02222024-L/signals.mat"  # Steph's version

    # Define paths to reference data
    reference_data_path = "/Volumes/Jose/ACh Rats/80B8CE6_ceecee/02222024-L/80B8CE6 (Ceecee)_02222024-L_h_sampleframe.csv"
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
    expected_sampling_rate = 250 # Hz

    # Assert that all expected photometry series are present in the acquisition field of the nwbfile
    actual_photometry_series = set(nwbfile.acquisition.keys())
    missing_photometry_series = expected_photometry_series - actual_photometry_series
    assert not missing_photometry_series, f"Missing FiberPhotometryResponseSeries: {missing_photometry_series}"

    # Check the basic attributes of each series
    for series_name in expected_photometry_series:
        # Assert that all series are of type FiberPhotometryResponseSeries
        assert isinstance(
            nwbfile.acquisition[series_name], FiberPhotometryResponseSeries
        ), f"{series_name} is not of type FiberPhotometryResponseSeries"
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
        err_msg=f"Data mismatch between nwbfile z_scored_green_dFF and reference data",
    )

    # TODO: add test for "visits" -> make sure we have the correct timestamps for port entries as compared to a reference
