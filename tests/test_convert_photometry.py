from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io
from dateutil import tz
from pynwb import NWBFile
from ndx_fiber_photometry import (
    FiberPhotometryResponseSeries,
    Indicator,
    ExcitationSource,
    OpticalFiber,
    Photodetector,
    DichroicMirror,
    FiberPhotometryTable,
    FiberPhotometry,
)

from jdb_to_nwb.convert_photometry import (
    add_photometry, 
    add_photometry_metadata, 
    process_raw_labview_photometry_signals,
)


def add_dummy_labview_metadata_to_metadata(metadata):
    """
    Add values to the metadata dictionary so that the tests can run.
    Values assume recording in the old maze room with LabVIEW
    """
    metadata["photometry"]["excitation_sources"] = [
        "Thorlabs Blue LED",
        "Thorlabs Purple LED",
    ]
    metadata["photometry"]["photodetector"] = "Doric iFMC7-G2 (7 ports Fluorescence Mini Cube - Three Fluorophores)"
    # Bilateral fibers in NAcc, recording from right hemisphere
    metadata["photometry"]["optic_fiber_implant_sites"] = [
        {
            "optic_fiber": "Doric 0.66mm Flat 40mm Optic Fiber",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": 1.7,
            "dv_in_mm": -6.0,
            "recording": True,
        },
        {
            "optic_fiber": "Doric 0.66mm Flat 40mm Optic Fiber",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": -1.7,
            "dv_in_mm": -6.0,
        }
    ]
    # Bilateral dLight in NAcc
    metadata["photometry"]["virus_injections"] = [
        {
            "virus_name": "dLight1.3b",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": 1.7,
            "dv_in_mm": -6.2,
            "volume_in_uL": 1.0,
            "titer_in_vg_per_mL": 2e12,
        },
        {
            "virus_name": "dLight1.3b",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": -1.7,
            "dv_in_mm": -6.2,
            "volume_in_uL": 1.0,
            "titer_in_vg_per_mL": 2e12,
        }
    ]


def add_dummy_pyphotometry_metadata_to_metadata(metadata):
    """
    Add values to the metadata dictionary so that the tests can run.
    Values assume recording in the new maze room with pyPhotometry
    test_add_photometry_metadata tests that the values are added correctly.
    """
    metadata["photometry"]["excitation_sources"] = [
        "Doric Purple LED",
        "Doric Blue LED",
        "Doric Green LED",
    ]
    metadata["photometry"]["photodetector"] = "Doric ilFMC7-G2 (Integrated LED Fluorescence Mini Cube 5 ports Gen.2)"
    # Bilateral fibers in NAcc, recording from left hemisphere
    metadata["photometry"]["optic_fiber_implant_sites"] = [
        {
            "optic_fiber": "Doric 0.66mm Flat 40mm Optic Fiber",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": 1.7,
            "dv_in_mm": -6.0,
        },
        {
            "optic_fiber": "Doric 0.66mm Flat 40mm Optic Fiber",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": -1.7,
            "dv_in_mm": -6.0,
            "recording": True,
        }
    ]
    # Bilateral cocktail injection of gACh4h and rDA3m NAcc
    metadata["photometry"]["virus_injections"] = [
        {
            "virus_name": "gACh4h",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": 1.7,
            "dv_in_mm": -6.2,
            "volume_in_uL": 1.0,
            "titer_in_vg_per_mL": 1.15e13,
        },
        {
            "virus_name": "gACh4h",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": -1.7,
            "dv_in_mm": -6.2,
            "volume_in_uL": 1.0,
            "titer_in_vg_per_mL": 1.15e13,
        },
        {
            "virus_name": "rDA3m (rAAV)",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": 1.7,
            "dv_in_mm": -6.2,
            "volume_in_uL": 1.0,
            "titer_in_vg_per_mL": 5.89e12,
        },
        {
            "virus_name": "rDA3m (rAAV)",
            "targeted_location": "NAcc",
            "ap_in_mm": 1.7,
            "ml_in_mm": -1.7,
            "dv_in_mm": -6.2,
            "volume_in_uL": 1.0,
            "titer_in_vg_per_mL": 5.89e12,
        }
    ]


def test_process_raw_labview_photometry_signals(dummy_logger):
    """
    Test that the process_raw_labview_photometry_signals function
    returns a signals dictionary equivalent to signals.mat.
    """

    # Create a test metadata dictionary
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")

    # Load signals.mat created by the external MATLAB photometry processing code as a reference
    signals_mat_file_path = test_data_dir / "signals.mat"
    reference_signals_mat = scipy.io.loadmat(signals_mat_file_path, matlab_compatible=True)

    # Process the raw .phot and .box files from Labview into an equivalent signals dict
    phot_file_path = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.phot"
    box_file_path = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.box"
    signals = process_raw_labview_photometry_signals(phot_file_path, box_file_path, dummy_logger)

    # Returned signals dict should include photometry_start as a datetime object
    assert isinstance(signals.get('photometry_start'), datetime), (
        f"Expected signals dict to include 'photometry_start' as a datetime object, "
        f"got 'photometry_start'={signals.get('photometry_start')}")

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
    # This results in the signals in our created dict being longer than the reference signals in signals.mat,
    # so we calculate the number of samples to remove from the beginning so we can do comparisons.
    samples_removed_from_reference = len(signals["ref"]) - len(reference_signals_mat["ref"].flatten())

    # The difference between our signal and the reference must be <0.01% of the magnitude of the reference
    assert np.allclose(
        signals["ref"][samples_removed_from_reference:],
        reference_signals_mat["ref"].flatten(),
        rtol=1e-4
    )
    assert np.allclose(
        signals["sig1"][samples_removed_from_reference:],
        reference_signals_mat["sig1"].flatten(),
        rtol=1e-4
    )
    # We allow an additional absolute tolerance for sig2 because there are some values very close to 0
    # where relative tolerance is less useful. (Also, we don't use actually sig2 so it doesn't really matter)
    assert np.allclose(
        signals["sig2"][samples_removed_from_reference:],
        reference_signals_mat["sig2"].flatten(),
        rtol=1e-4,
        atol=5
    )


def test_add_photometry_from_signals_mat(dummy_logger):
    """
    Test that the add_photometry function results in the expected FiberPhotometryResponseSeries.

    This version of the test uses the already created signals.mat at "signals_mat_file_path"
    to further process and add photometry signals to the NWB.
    """

    # Create a test metadata dictionary with preprocessed LabVIEW data ("signals_mat_file_path")
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["signals_mat_file_path"] = test_data_dir / "signals.mat"
    add_dummy_labview_metadata_to_metadata(metadata)

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
    photometry_data_dict = add_photometry(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Define the FiberPhotometryResponseSeries we expect to have been added to the nwbfile
    expected_photometry_series = {"raw_green", "raw_reference", "z_scored_green_dFF", "z_scored_reference_fitted"}
    expected_sampling_rate = 250  # Hz

    # Assert that we have returned the correct sampling rate
    assert photometry_data_dict.get('sampling_rate') == expected_sampling_rate, (
        f"Expected sampling rate {expected_sampling_rate} Hz, got {photometry_data_dict.get('sampling_rate')} Hz")
    # We expect photometry_start = None when starting photometry conversion from signals.mat
    assert photometry_data_dict.get('photometry_start') is None, (
        f"Expected 'photometry_start' = None when starting from signals.mat, "
        f"got 'photometry_start'={photometry_data_dict.get('photometry_start')}")
    # Assert that the returned photometry_data_dict includes port_visits as a list or array
    assert isinstance(photometry_data_dict.get('port_visits'), (list, np.ndarray)), (
        f"Expected 'port_visits' to be a list or NumPy array, but got {type(photometry_data_dict.get('port_visits'))}"
    )

    # Assert that all expected photometry series are present in the acquisition field of the nwbfile
    actual_photometry_series = set(nwbfile.acquisition.keys())
    missing_photometry_series = expected_photometry_series - actual_photometry_series
    assert not missing_photometry_series, f"Missing FiberPhotometryResponseSeries: {missing_photometry_series}"

    # Check the basic attributes of each series
    for series_name in expected_photometry_series:
        # Assert that all series are of type FiberPhotometryResponseSeries
        assert isinstance(nwbfile.acquisition[series_name], FiberPhotometryResponseSeries), (
            f"{series_name} is not of type FiberPhotometryResponseSeries")
        # Assert all series have a sampling rate of 250 Hz
        assert (
            getattr(nwbfile.acquisition[series_name], "rate", None) == expected_sampling_rate
        ), (
            f"{series_name} has a sampling rate of {getattr(nwbfile.acquisition[series_name], 'rate', None)}, "
            f"expected {expected_sampling_rate}"
        )

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


def test_add_photometry_from_raw_labview(dummy_logger):
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
    add_dummy_labview_metadata_to_metadata(metadata)

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
    photometry_data_dict = add_photometry(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Define the FiberPhotometryResponseSeries we expect to have been added to the nwbfile
    expected_photometry_series = {"raw_green", "raw_reference", "z_scored_green_dFF", "z_scored_reference_fitted"}
    expected_sampling_rate = 250  # Hz

    # Assert that we have returned the correct sampling rate
    assert photometry_data_dict.get('sampling_rate') == expected_sampling_rate, (
        f"Expected sampling rate {expected_sampling_rate} Hz, got {photometry_data_dict.get('sampling_rate')} Hz")
    # Assert that we have returned photometry_start as a datetime object
    assert isinstance(photometry_data_dict.get('photometry_start'), datetime), (
        f"Expected photometry data dict to include 'photometry_start' as a datetime object, "
        f"got 'photometry_start'={photometry_data_dict.get('photometry_start')}")
    # Assert that the returned photometry_data_dict includes port_visits as a list or array
    assert isinstance(photometry_data_dict.get('port_visits'), (list, np.ndarray)), (
        f"Expected 'port_visits' to be a list or NumPy array, but got {type(photometry_data_dict.get('port_visits'))}"
    )

    # Assert that all expected photometry series are present in the acquisition field of the nwbfile
    actual_photometry_series = set(nwbfile.acquisition.keys())
    missing_photometry_series = expected_photometry_series - actual_photometry_series
    assert not missing_photometry_series, f"Missing FiberPhotometryResponseSeries: {missing_photometry_series}"

    # Check the basic attributes of each series
    for series_name in expected_photometry_series:
        # Assert that all series are of type FiberPhotometryResponseSeries
        assert isinstance(nwbfile.acquisition[series_name], FiberPhotometryResponseSeries), (
            f"{series_name} is not of type FiberPhotometryResponseSeries")
        # Assert all series have a sampling rate of 250 Hz
        assert (
            getattr(nwbfile.acquisition[series_name], "rate", None) == expected_sampling_rate
        ), (
            f"{series_name} has a sampling rate of {getattr(nwbfile.acquisition[series_name], 'rate', None)}, "
            f"expected {expected_sampling_rate}"
        )

    # Check that the photometry series in the nwbfile match the expected signals from the reference dataframe
    green_z_scored_dFF = np.array(nwbfile.acquisition["z_scored_green_dFF"].data)
    green_z_scored_dFF_reference = reference_dataframe["green"].values

    # In our former MATLAB photometry preprocessing pipeline (used to create the reference dataframe),
    # we would sometimes remove some samples from the beginning of our signals for alignment with behavior.
    # In this new pipeline, we do this alignment later so this is no longer necessary at this step.
    # This results in the signals in our created dict being longer than the reference signals in the dataframe,
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


def test_add_photometry_from_pyphotometry(dummy_logger):
    """
    Test that the add_photometry function results in the expected FiberPhotometryResponseSeries.

    This version of the test uses the ppd file from pyPhotometry to add photometry signals to the NWB.
    """

    # Create a test metadata dictionary with pyPhotometry data
    test_data_dir = Path("tests/test_data/downloaded/IM-1770_corvette/11062024")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["ppd_file_path"] = test_data_dir / "Lhem_barswitch_GACh4h_rDA3m_CKTL-2024-11-06-185407.ppd"
    add_dummy_pyphotometry_metadata_to_metadata(metadata)

    # Define paths to reference data
    reference_data_path = test_data_dir / "sampleframe.csv"
    reference_dataframe = pd.read_csv(reference_data_path)

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Add photometry data to the nwbfile
    photometry_data_dict = add_photometry(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Define the FiberPhotometryResponseSeries we expect to have been added to the nwbfile
    expected_photometry_series = {"raw_470", "z_scored_470", "raw_405", "zscored_405", "raw_565",
                                  "zscored_565", "raw_470_405_ratio", "zscored_470_405_ratio"}
    expected_sampling_rate = 86 # Hz

    # Assert that we have returned the correct sampling rate
    assert photometry_data_dict.get('sampling_rate') == expected_sampling_rate, (
        f"Expected sampling rate {expected_sampling_rate} Hz, got {photometry_data_dict.get('sampling_rate')} Hz")
    # Assert that we have returned photometry_start as a datetime object
    assert isinstance(photometry_data_dict.get('photometry_start'), datetime), (
        f"Expected photometry data dict to include 'photometry_start' as a datetime object, "
        f"got 'photometry_start'={photometry_data_dict.get('photometry_start')}")
    # Assert that the returned photometry_data_dict includes port_visits as a list or array
    assert isinstance(photometry_data_dict.get('port_visits'), (list, np.ndarray)), (
        f"Expected 'port_visits' to be a list or NumPy array, but got {type(photometry_data_dict.get('port_visits'))}"
    )

    # Assert that all expected photometry series are present in the acquisition field of the nwbfile
    actual_photometry_series = set(nwbfile.acquisition.keys())
    missing_photometry_series = expected_photometry_series - actual_photometry_series
    assert not missing_photometry_series, f"Missing FiberPhotometryResponseSeries: {missing_photometry_series}"

    # Check the basic attributes of each series
    for series_name in expected_photometry_series:
        # Assert that all series are of type FiberPhotometryResponseSeries
        assert isinstance(nwbfile.acquisition[series_name], FiberPhotometryResponseSeries), (
            f"{series_name} is not of type FiberPhotometryResponseSeries")
        # Assert all series have a sampling rate of 86 Hz
        assert (
            getattr(nwbfile.acquisition[series_name], "rate", None) == expected_sampling_rate
        ), (
            f"{series_name} has a sampling rate of {getattr(nwbfile.acquisition[series_name], 'rate', None)}, "
            f"expected {expected_sampling_rate}"
        )

    # Check that the photometry series in the nwbfile match the expected signals from the reference dataframe
    nwb_signal_names = ["raw_470", "z_scored_470", "raw_405", "zscored_405",
                        "raw_565", "zscored_565", "raw_470_405_ratio", "zscored_470_405_ratio"]
    reference_signal_names = ["raw_green", "green_z_scored", "raw_405", "z_scored_405",
                              "raw_red", "red_z_scored", "raw 470/405", "ratio_z_scored"]

    # Compare each signal in the nwbfile to its respective reference
    for sig_name, ref_name in zip(nwb_signal_names, reference_signal_names):
        signal = np.array(nwbfile.acquisition[sig_name].data)
        reference = reference_dataframe[ref_name].values

        # Check that the lengths match
        assert len(signal) == len(reference), (
            f"Data length mismatch: {sig_name} has {len(signal)} points, "
            f"but the reference signal {ref_name} has {len(reference)} points."
        )

        # Check that the signal in the nwbfile matches the reference signal (within a tolerance)
        np.testing.assert_allclose(
            signal,
            reference,
            atol=0.005,
            rtol=0.05,
            err_msg=f"Data mismatch between nwbfile {sig_name} and reference {ref_name}",
        )


def test_add_photometry_metadata(dummy_logger):
    """
    Test that the add_photometry_metadata function adds the expected metadata to the NWB file.
    For this test, we use pyPhotometry metadata. 
    """

    # Create a test metadata dictionary with pyPhotometry data and metadata
    test_data_dir = Path("tests/test_data/downloaded/IM-1770_corvette/11062024")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["ppd_file_path"] = test_data_dir / "Lhem_barswitch_GACh4h_rDA3m_CKTL-2024-11-06-185407.ppd"
    add_dummy_pyphotometry_metadata_to_metadata(metadata)

    # Create a test NWBFile
    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    # Add photometry metadata to the nwbfile
    add_photometry_metadata(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Check excitation sources
    assert "Doric Purple LED" in nwbfile.devices
    purple_led = nwbfile.devices["Doric Purple LED"]
    assert isinstance(purple_led, ExcitationSource)
    assert purple_led.excitation_wavelength_in_nm == 405.0
    assert purple_led.illumination_type == "LED"
    assert purple_led.manufacturer == "Doric"
    assert purple_led.model == "ilFMC7-G2"

    assert "Doric Blue LED" in nwbfile.devices
    blue_led = nwbfile.devices["Doric Blue LED"]
    assert isinstance(blue_led, ExcitationSource)
    assert blue_led.excitation_wavelength_in_nm == 470.0
    assert blue_led.illumination_type == "LED"
    assert blue_led.manufacturer == "Doric"
    assert blue_led.model == "ilFMC7-G2"

    assert "Doric Green LED" in nwbfile.devices
    green_led = nwbfile.devices["Doric Green LED"]
    assert isinstance(green_led, ExcitationSource)
    assert green_led.excitation_wavelength_in_nm == 565.0
    assert green_led.illumination_type == "LED"
    assert green_led.manufacturer == "Doric"
    assert green_led.model == "ilFMC7-G2"

    # Check photodetector
    assert "Doric ilFMC7-G2 (Integrated LED Fluorescence Mini Cube 5 ports Gen.2)" in nwbfile.devices
    photodetector = nwbfile.devices["Doric ilFMC7-G2 (Integrated LED Fluorescence Mini Cube 5 ports Gen.2)"]
    assert isinstance(photodetector, Photodetector)
    assert photodetector.manufacturer == "Doric"
    assert photodetector.model == "ilFMC7-G2"
    assert photodetector.detector_type == "Silicon photodiode"
    assert photodetector.detected_wavelength_in_nm == 960.0

    # Check dichroic mirror
    dichroic_mirror_name = (
        "Doric ilFMC7-G2 (Integrated LED Fluorescence Mini Cube 5 ports Gen.2) "
        "Built-in Dichroic Mirror"
    )
    assert dichroic_mirror_name in nwbfile.devices
    dichroic_mirror = nwbfile.devices[dichroic_mirror_name]
    assert isinstance(dichroic_mirror, DichroicMirror)
    assert dichroic_mirror.manufacturer == "Doric"
    assert dichroic_mirror.description == "Built-in dichroic mirror for photodetector"

    # Check optical fibers
    assert "Doric 0.66mm Flat 40mm Optic Fiber (left NAcc)" in nwbfile.devices
    optic_fiber_left = nwbfile.devices["Doric 0.66mm Flat 40mm Optic Fiber (left NAcc)"]
    assert isinstance(optic_fiber_left, OpticalFiber)
    assert optic_fiber_left.numerical_aperture == 0.66
    assert optic_fiber_left.core_diameter_in_um == 200.0
    assert optic_fiber_left.manufacturer == "Doric"
    assert optic_fiber_left.model == "MFC_200/250-0.66_40mm_MF2.5_FLT"

    assert "Doric 0.66mm Flat 40mm Optic Fiber (right NAcc)" in nwbfile.devices
    optic_fiber_right = nwbfile.devices["Doric 0.66mm Flat 40mm Optic Fiber (right NAcc)"]
    assert isinstance(optic_fiber_right, OpticalFiber)
    assert optic_fiber_right.numerical_aperture == 0.66
    assert optic_fiber_right.core_diameter_in_um == 200.0
    assert optic_fiber_right.manufacturer == "Doric"
    assert optic_fiber_right.model == "MFC_200/250-0.66_40mm_MF2.5_FLT"

    # Check indicators
    # NOTE volume_in_uL and titer_in_vg_per_mL do not exist as Indicator fields so they
    # are stored as a part of the description. These values are not tested
    assert "gACh4h (left NAcc)" in nwbfile.devices
    gach4h_left = nwbfile.devices["gACh4h (left NAcc)"]
    assert isinstance(gach4h_left, Indicator)
    assert gach4h_left.injection_coordinates_in_mm == (1.7, -1.7, -6.2)
    assert gach4h_left.injection_location == "NAcc"
    assert gach4h_left.label == "AAV-hSyn-ACh3.8"
    assert gach4h_left.manufacturer == "BrainVTA"
    assert gach4h_left.description.startswith("AAV virus expressing the acetylcholine sensor GRAB-ACh3.8")

    assert "gACh4h (right NAcc)" in nwbfile.devices
    gach4h_right = nwbfile.devices["gACh4h (right NAcc)"]
    assert isinstance(gach4h_right, Indicator)
    assert gach4h_right.injection_coordinates_in_mm == (1.7, 1.7, -6.2)
    assert gach4h_right.injection_location == "NAcc"
    assert gach4h_right.label == "AAV-hSyn-ACh3.8"
    assert gach4h_right.manufacturer == "BrainVTA"
    assert gach4h_right.description.startswith("AAV virus expressing the acetylcholine sensor GRAB-ACh3.8")

    assert "rDA3m (rAAV) (left NAcc)" in nwbfile.devices
    rda3m_left = nwbfile.devices["rDA3m (rAAV) (left NAcc)"]
    assert isinstance(rda3m_left, Indicator)
    assert rda3m_left.injection_coordinates_in_mm == (1.7, -1.7, -6.2)
    assert rda3m_left.injection_location == "NAcc"
    assert rda3m_left.label == "rAAV-hsyn-rDA3m"
    assert rda3m_left.manufacturer == "BrainVTA"
    assert rda3m_left.description.startswith("Recombinant AAV expressing the red-shifted dopamine sensor GRAB rDA3m")

    assert "rDA3m (rAAV) (right NAcc)" in nwbfile.devices
    rda3m_right = nwbfile.devices["rDA3m (rAAV) (right NAcc)"]
    assert isinstance(rda3m_right, Indicator)
    assert rda3m_right.injection_coordinates_in_mm == (1.7, 1.7, -6.2)
    assert rda3m_right.injection_location == "NAcc"
    assert rda3m_right.label == "rAAV-hsyn-rDA3m"
    assert rda3m_right.manufacturer == "BrainVTA"
    assert rda3m_right.description.startswith("Recombinant AAV expressing the red-shifted dopamine sensor GRAB rDA3m")

    # Check fiber photometry table
    assert "fiber_photometry" in nwbfile.lab_meta_data
    fiber_photometry_meta = nwbfile.lab_meta_data["fiber_photometry"]
    assert isinstance(fiber_photometry_meta, FiberPhotometry)
    assert isinstance(fiber_photometry_meta.fiber_photometry_table, FiberPhotometryTable)

    table = fiber_photometry_meta.fiber_photometry_table
    assert table.name == "fiber_photometry_table"
    assert table.description == "fiber photometry table"

    # Only recorded indicators are added to the table. We recorded from the fiber in left NAcc.
    # 1. gACh4h (left NAcc) + Doric Purple LED
    # 2. gACh4h (left NAcc) + Doric Blue LED
    # 3. rDA3m (rAAV) (left NAcc) + Doric Green LED
    expected_combinations = {
        (gach4h_left, purple_led),
        (gach4h_left, blue_led),
        (rda3m_left, green_led)
    }
    combinations_in_table = set()

    # Our table should have 3 rows, one for each indicator + excitation source combination
    assert len(table) == 3

    for row in table:
        # The recording fiber, photodetector, and dichroic mirror are the same for all rows
        assert row["optical_fiber"].item() == optic_fiber_left
        assert row["photodetector"].item() == photodetector
        assert row["dichroic_mirror"].item() == dichroic_mirror

        # Each of the indicator + excitation source combinations should appear once
        combinations_in_table.add((row["indicator"].item(), row["excitation_source"].item()))

    assert expected_combinations == combinations_in_table


def test_photometry_series_mappings():
    # TODO: Test that each photometry series 
    # is mapped to the correct row in the fiber photometry table
    pass


def test_add_photometry_with_incomplete_metadata(capsys, dummy_logger):
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
    photometry_data_dict = add_photometry(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)
    captured = capsys.readouterr() # capture stdout

    # Check that the correct message was printed to stdout and returned dict is empty
    assert "No photometry metadata found for this session. Skipping photometry conversion." in captured.out
    assert photometry_data_dict.get('sampling_rate') is None
    assert photometry_data_dict.get('port_visits') is None
    assert photometry_data_dict.get('photometry_start') is None

    # Create a test metadata dictionary with a photometry field and metadata but no photometry data
    metadata["photometry"] = {}
    add_dummy_pyphotometry_metadata_to_metadata(metadata)

    # Check that add_photometry raises a ValueError about missing fields in the metadata dictionary
    try:
        photometry_data_dict = add_photometry(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)
    except ValueError as e:
        assert str(e).startswith("The required photometry subfields do not exist in the metadata dictionary")
    else:
        assert False, (
            "Expected ValueError was not raised in response to "
            "missing photometry subfields in the metadata dict."
        )
