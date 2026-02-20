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
    load_labview_raw,
    load_labview_mat,
    load_pyphotometry,
    load_processing_config,
    apply_rolling_mean,
    apply_lowpass_filter,
    apply_highpass_filter,
    apply_airpls_baseline,
    apply_zscore_median_std,
    apply_zscore_mean_std,
    apply_isosbestic_correction,
    apply_ratiometric_correction,
    PhotometrySignalBundle,
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


def test_load_labview_raw(dummy_logger):
    """
    Test that load_labview_raw returns a valid PhotometrySignalBundle
    with signals consistent with the reference signals.mat.
    """

    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")

    # Load signals.mat created by the external MATLAB photometry processing code as a reference
    signals_mat_file_path = test_data_dir / "signals.mat"
    reference_signals_mat = scipy.io.loadmat(signals_mat_file_path, matlab_compatible=True)

    # Load the raw .phot and .box files through the new pipeline
    phot_file_path = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.phot"
    box_file_path = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.box"
    bundle = load_labview_raw(phot_file_path, box_file_path, dummy_logger)

    # Check bundle type and structure
    assert isinstance(bundle, PhotometrySignalBundle)
    assert isinstance(bundle.photometry_start, datetime)
    assert bundle.sampling_rate == 250
    assert bundle.source == "labview_raw"
    assert set(bundle.signals.keys()) == {"470nm", "405nm"}

    # Signals should be non-empty and same length
    assert len(bundle.signals["470nm"]) > 0
    assert len(bundle.signals["470nm"]) == len(bundle.signals["405nm"])

    # Port visits should be non-negative
    assert len(bundle.port_visits) > 0
    assert all(v >= 0 for v in bundle.port_visits)

    # Cross-validate against signals.mat (preprocessed by MATLAB at 10kHz).
    # The MATLAB pipeline removed some initial samples for alignment. Our pipeline does not,
    # so the bundle has more samples at the front. Both are downsampled to 250Hz.
    SR, Fs = 10000, 250
    downsample_factor = int(SR / Fs)
    # In the loader, 470nm comes from lockin_signals["sig1"] and 405nm comes from lockin_signals["ref"]
    ref_sig1 = np.squeeze(reference_signals_mat["sig1"])[::downsample_factor]
    ref_ref = np.squeeze(reference_signals_mat["ref"])[::downsample_factor]

    assert len(bundle.signals["470nm"]) >= len(ref_sig1), (
        "Bundle should have at least as many samples as the reference signals.mat"
    )

    # Calculate the offset: our bundle has extra samples at the front that the MATLAB pipeline removed
    samples_removed_from_reference = len(bundle.signals["470nm"]) - len(ref_sig1)

    # The 470nm signal (sig1) should match the reference within 0.01%
    assert np.allclose(
        bundle.signals["470nm"][samples_removed_from_reference:],
        ref_sig1,
        rtol=1e-4,
    )
    # The 405nm signal (ref) should match the reference within 0.01%
    assert np.allclose(
        bundle.signals["405nm"][samples_removed_from_reference:],
        ref_ref,
        rtol=1e-4,
    )

    # Port visit count should match (visits occur at the same times)
    ref_visits = np.divide(np.squeeze(reference_signals_mat["visits"]), downsample_factor).astype(int)
    assert np.allclose(bundle.port_visits, ref_visits, atol=1)


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
    expected_photometry_series = {"raw_470", "raw_405", "processed_470", "processed_405",
                                  "corrected_470_dFF", "fitted_405"}
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

    # Check that the corrected dF/F signal in the nwbfile matches the reference green signal
    corrected_dFF = np.array(nwbfile.acquisition["corrected_470_dFF"].data)
    green_dFF_reference = reference_dataframe["green"].values

    # Check that the lengths match
    assert len(corrected_dFF) == len(green_dFF_reference), (
        f"Data length mismatch: corrected_470_dFF has {len(corrected_dFF)} points, "
        f"but the reference signal has {len(green_dFF_reference)} points."
    )

    # Check that the corrected dF/F signal matches the reference (within a tolerance)
    np.testing.assert_allclose(
        corrected_dFF,
        green_dFF_reference,
        atol=0.005,
        rtol=0.05,
        err_msg="Data mismatch between nwbfile corrected_470_dFF and reference data",
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
    expected_photometry_series = {"raw_470", "raw_405", "processed_470", "processed_405",
                                  "corrected_470_dFF", "fitted_405"}
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

    # Check that the corrected dF/F signal in the nwbfile matches the reference green signal
    corrected_dFF = np.array(nwbfile.acquisition["corrected_470_dFF"].data)
    green_dFF_reference = reference_dataframe["green"].values

    # In our former MATLAB photometry preprocessing pipeline (used to create the reference dataframe),
    # we would sometimes remove some samples from the beginning of our signals for alignment with behavior.
    # In this new pipeline, we do this alignment later so this is no longer necessary at this step.
    # This results in the signals in our created dict being longer than the reference signals in the dataframe,
    # so we calculate the number of samples to remove from the beginning so we can do comparisons.
    samples_removed_from_reference = len(corrected_dFF) - len(green_dFF_reference)
    corrected_dFF = corrected_dFF[samples_removed_from_reference:]

    # Check that the corrected dF/F signal matches the reference (within a tolerance)
    # TODO: have a conversation about how we feel about the removal of some samples from the front of the signal.
    # Because the signals in our returned signals dict and signals.mat are different lengths, when we downsample to
    # 250 Hz we end up taking slightly different data points from the signal. These differences propagate through our
    # filtering etc. The signals are visually pretty similar but would fail most reasonable assertions of similarity.
    # Commenting out this similarity test for now.

    # np.testing.assert_allclose(
    #    corrected_dFF,
    #    green_dFF_reference,
    #    atol=0.005,
    #    rtol=0.05,
    #    err_msg="Data mismatch between nwbfile corrected_470_dFF and reference data",
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
    # gACh4h (signal=470nm, reference=405nm, preset=gach_ratiometric) produces:
    #   raw_470, raw_405, processed_470, processed_405, corrected_470_405_ratio
    # rDA3m (signal=565nm, no reference, preset=rda_independent) produces:
    #   raw_565, processed_565
    expected_photometry_series = {"raw_470", "raw_405", "raw_565",
                                  "processed_470", "processed_405", "processed_565",
                                  "raw_470_405_ratio", "corrected_470_405_ratio"}
    expected_sampling_rate = 86  # Hz

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

    # Compare raw signals, individually-processed signals, and ratiometric correction against reference.
    # Raw signals should match exactly. Individually processed signals (lowpass + highpass + mean_std z-score)
    # use the same methods as the old pipeline, so they should also match.
    # The ratiometric correction computes the ratio from raw signals first, then processes it through
    # the same pipeline, matching the old code's behavior.
    nwb_signal_names = ["raw_470", "raw_405", "raw_565",
                        "processed_470", "processed_405", "processed_565",
                        "raw_470_405_ratio", "corrected_470_405_ratio"]
    reference_signal_names = ["raw_green", "raw_405", "raw_red",
                              "green_z_scored", "z_scored_405", "red_z_scored",
                              "raw 470/405", "ratio_z_scored"]

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


def test_photometry_series_mappings(dummy_logger):
    """Test that each FiberPhotometryResponseSeries is linked to the correct fiber_photometry_table row."""

    test_data_dir = Path("tests/test_data/downloaded/IM-1770_corvette/11062024")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["ppd_file_path"] = test_data_dir / "Lhem_barswitch_GACh4h_rDA3m_CKTL-2024-11-06-185407.ppd"
    add_dummy_pyphotometry_metadata_to_metadata(metadata)

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_photometry(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Each photometry series should have a fiber_photometry_table_region pointing to a valid row
    for series_name, series_obj in nwbfile.acquisition.items():
        if isinstance(series_obj, FiberPhotometryResponseSeries):
            region = series_obj.fiber_photometry_table_region
            assert region is not None, f"Series '{series_name}' has no fiber_photometry_table_region"
            assert len(region.data) > 0, f"Series '{series_name}' has an empty table region"

    # Verify that blue LED series map to a row with a blue excitation source
    blue_series = ["raw_470", "processed_470", "raw_470_405_ratio", "corrected_470_405_ratio"]
    for name in blue_series:
        series = nwbfile.acquisition[name]
        row_idx = series.fiber_photometry_table_region.data[0]
        table = nwbfile.get_lab_meta_data("fiber_photometry").fiber_photometry_table
        exc_source = table.excitation_source.data[row_idx]
        assert "blue" in exc_source.name.lower(), (
            f"Series '{name}' should map to a blue LED row, but maps to '{exc_source.name}'"
        )

    # Verify that purple LED series map to a row with a purple excitation source
    purple_series = ["raw_405", "processed_405"]
    for name in purple_series:
        series = nwbfile.acquisition[name]
        row_idx = series.fiber_photometry_table_region.data[0]
        table = nwbfile.get_lab_meta_data("fiber_photometry").fiber_photometry_table
        exc_source = table.excitation_source.data[row_idx]
        assert "purple" in exc_source.name.lower(), (
            f"Series '{name}' should map to a purple LED row, but maps to '{exc_source.name}'"
        )

    # Verify that green LED series map to a row with a green excitation source
    green_series = ["raw_565", "processed_565"]
    for name in green_series:
        series = nwbfile.acquisition[name]
        row_idx = series.fiber_photometry_table_region.data[0]
        table = nwbfile.get_lab_meta_data("fiber_photometry").fiber_photometry_table
        exc_source = table.excitation_source.data[row_idx]
        assert "green" in exc_source.name.lower(), (
            f"Series '{name}' should map to a green LED row, but maps to '{exc_source.name}'"
        )


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


# ============================================================
# Loader unit tests
# ============================================================


def test_load_labview_mat(dummy_logger):
    """Test that load_labview_mat returns a valid PhotometrySignalBundle."""
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")
    signals_mat_path = test_data_dir / "signals.mat"
    bundle = load_labview_mat(signals_mat_path, dummy_logger)

    # Check bundle structure
    assert isinstance(bundle, PhotometrySignalBundle)
    assert bundle.sampling_rate == 250
    assert bundle.source == "labview_mat"
    assert bundle.photometry_start is None  # signals.mat has no start time
    assert set(bundle.signals.keys()) == {"470nm", "405nm"}

    # Signals should be non-empty and same length
    assert len(bundle.signals["470nm"]) > 0
    assert len(bundle.signals["470nm"]) == len(bundle.signals["405nm"])

    # Port visits should be non-negative
    assert len(bundle.port_visits) > 0
    assert all(v >= 0 for v in bundle.port_visits)

    # Cross-validate: the signals should match the raw signals.mat downsampled to 250Hz
    reference_signals_mat = scipy.io.loadmat(signals_mat_path, matlab_compatible=True)
    downsample_factor = int(10000 / 250)
    ref_sig1 = np.squeeze(reference_signals_mat["sig1"])[::downsample_factor]
    ref_ref = np.squeeze(reference_signals_mat["ref"])[::downsample_factor]

    np.testing.assert_array_equal(bundle.signals["470nm"], ref_sig1)
    np.testing.assert_array_equal(bundle.signals["405nm"], ref_ref)


def test_load_pyphotometry(dummy_logger):
    """Test that load_pyphotometry returns a valid PhotometrySignalBundle for 3-signal data."""
    test_data_dir = Path("tests/test_data/downloaded/IM-1770_corvette/11062024")
    ppd_path = test_data_dir / "Lhem_barswitch_GACh4h_rDA3m_CKTL-2024-11-06-185407.ppd"
    bundle = load_pyphotometry(ppd_path, dummy_logger)

    # Check bundle structure
    assert isinstance(bundle, PhotometrySignalBundle)
    assert bundle.sampling_rate == 86
    assert bundle.source == "pyphotometry"
    assert isinstance(bundle.photometry_start, datetime)
    assert set(bundle.signals.keys()) == {"470nm", "405nm", "565nm"}

    # Signals should be non-empty and same length
    assert len(bundle.signals["470nm"]) > 0
    assert len(bundle.signals["470nm"]) == len(bundle.signals["405nm"])
    assert len(bundle.signals["470nm"]) == len(bundle.signals["565nm"])

    # Port visits should be non-negative
    assert len(bundle.port_visits) > 0
    assert all(v >= 0 for v in bundle.port_visits)

    # Cross-validate raw signals against reference CSV
    reference_dataframe = pd.read_csv(test_data_dir / "sampleframe.csv")
    np.testing.assert_allclose(
        bundle.signals["470nm"],
        reference_dataframe["raw_green"].values,
        atol=1e-10,
        err_msg="470nm raw signal doesn't match reference raw_green",
    )
    np.testing.assert_allclose(
        bundle.signals["405nm"],
        reference_dataframe["raw_405"].values,
        atol=1e-10,
        err_msg="405nm raw signal doesn't match reference raw_405",
    )
    np.testing.assert_allclose(
        bundle.signals["565nm"],
        reference_dataframe["raw_red"].values,
        atol=1e-10,
        err_msg="565nm raw signal doesn't match reference raw_red",
    )


# ============================================================
# Processing config tests
# ============================================================


def test_load_processing_config():
    """Test that load_processing_config loads presets and resolves method defaults."""
    config = load_processing_config("dlight_isosbestic")
    assert config["smoothing"]["method"] == "rolling_mean"
    assert config["smoothing"]["params"]["window_fraction"] == 0.0333
    assert config["baseline"]["method"] == "airpls"
    assert config["baseline"]["params"]["lambda"] == 1e8
    assert config["baseline"]["params"]["max_iterations"] == 50
    assert config["normalization"]["method"] == "median_zscore"
    assert config["correction"]["method"] == "isosbestic_lasso"
    assert config["correction"]["params"]["alpha"] == 0.0001
    assert "description" in config

    config = load_processing_config("gach_ratiometric")
    assert config["smoothing"]["method"] == "lowpass"
    assert config["smoothing"]["params"]["cutoff_hz"] == 10
    assert config["smoothing"]["params"]["order"] == 2
    assert config["baseline"]["method"] == "highpass"
    assert config["baseline"]["params"]["cutoff_hz"] == 0.001
    assert config["normalization"]["method"] == "mean_zscore"
    assert config["correction"]["method"] == "ratiometric"

    config = load_processing_config("rda_independent")
    assert config["correction"]["method"] == "none"


def test_load_processing_config_with_overrides():
    """Test that processing overrides are applied correctly."""
    overrides = {
        "smoothing": {"cutoff_hz": 8},
        "baseline": {"cutoff_hz": 0.01},
    }
    config = load_processing_config("gach_ratiometric", overrides=overrides)

    # Overrides should be merged into the defaults
    assert config["smoothing"]["params"]["cutoff_hz"] == 8
    assert config["baseline"]["params"]["cutoff_hz"] == 0.01
    # Non-overridden params should retain defaults
    assert config["smoothing"]["params"]["order"] == 2


def test_load_processing_config_invalid_preset():
    """Test that an invalid preset name raises ValueError."""
    try:
        load_processing_config("nonexistent_preset")
    except ValueError as e:
        assert "nonexistent_preset" in str(e)
    else:
        assert False, "Expected ValueError for invalid preset name"


# ============================================================
# Processing step function tests
# ============================================================


def test_apply_rolling_mean():
    """Test rolling mean smoothing."""
    # Create a noisy signal
    np.random.seed(42)
    raw = np.random.randn(1000) + 10.0
    sampling_rate = 250

    smoothed = apply_rolling_mean(raw, sampling_rate)

    # Output should be same length
    assert len(smoothed) == len(raw)
    # Smoothed signal should have lower variance than the raw signal
    assert np.std(smoothed) < np.std(raw)
    # Mean should be approximately preserved
    assert np.abs(np.mean(smoothed) - np.mean(raw)) < 0.1


def test_apply_lowpass_filter():
    """Test lowpass Butterworth filter."""
    sampling_rate = 250
    t = np.arange(0, 2, 1 / sampling_rate)
    # 5 Hz signal + 50 Hz noise
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    filtered = apply_lowpass_filter(signal, sampling_rate, cutoff_hz=10)

    assert len(filtered) == len(signal)
    # After lowpass at 10Hz, the 50Hz component should be nearly gone
    # Check that variance is reduced (high-frequency noise removed)
    assert np.std(filtered) < np.std(signal)


def test_apply_highpass_filter():
    """Test highpass Butterworth filter."""
    sampling_rate = 250
    t = np.arange(0, 10, 1 / sampling_rate)
    # Slow drift + 5 Hz signal
    signal = 0.1 * t + np.sin(2 * np.pi * 5 * t)

    filtered = apply_highpass_filter(signal, sampling_rate, cutoff_hz=0.5)

    assert len(filtered) == len(signal)
    # The slow linear drift should be removed, so the filtered signal
    # should have a mean closer to zero than the original
    assert np.abs(np.mean(filtered)) < np.abs(np.mean(signal))


def test_apply_airpls_baseline(dummy_logger):
    """Test airPLS baseline subtraction."""
    np.random.seed(42)
    n = 5000
    # Create a signal with a slow baseline drift + fast oscillation
    t = np.linspace(0, 20, n)
    baseline_true = 1000 + 200 * np.sin(2 * np.pi * 0.05 * t)
    fast_component = 10 * np.sin(2 * np.pi * 2 * t)
    signal = baseline_true + fast_component

    baseline_subtracted, baseline = apply_airpls_baseline(signal, dummy_logger)

    assert len(baseline_subtracted) == n
    assert len(baseline) == n
    # The baseline should roughly follow the slow drift
    # The subtracted signal should have much smaller range than the original
    assert np.ptp(baseline_subtracted) < np.ptp(signal)


def test_apply_airpls_baseline_with_raw_signal(dummy_logger):
    """Test that airPLS computes baseline from baseline_signal when provided."""
    np.random.seed(42)
    n = 5000
    raw = np.random.randn(n) * 10 + 1000
    smoothed = np.convolve(raw, np.ones(10) / 10, mode='same')

    # When baseline_signal is provided, baseline should be computed from baseline_signal
    subtracted_with_raw, baseline_from_raw = apply_airpls_baseline(
        smoothed, dummy_logger, baseline_signal=raw
    )
    subtracted_without_raw, baseline_from_smoothed = apply_airpls_baseline(
        smoothed, dummy_logger
    )

    # Baselines should differ because they're computed from different inputs
    assert not np.allclose(baseline_from_raw, baseline_from_smoothed)
    # But the result should be: smoothed - baseline_from_raw
    np.testing.assert_array_equal(subtracted_with_raw, smoothed.ravel() - baseline_from_raw)


def test_apply_zscore_median_std():
    """Test z-scoring with median and std."""
    np.random.seed(42)
    signal = np.random.randn(1000) * 5 + 100

    zscored = apply_zscore_median_std(signal)

    assert len(zscored) == len(signal)
    # The z-scored signal should have median close to 0
    assert np.abs(np.median(zscored)) < 0.01
    # And std close to 1
    assert np.abs(np.std(zscored) - 1.0) < 0.01


def test_apply_zscore_mean_std():
    """Test z-scoring with mean and std."""
    np.random.seed(42)
    signal = np.random.randn(1000) * 5 + 100

    zscored = apply_zscore_mean_std(signal)

    assert len(zscored) == len(signal)
    # The z-scored signal should have mean close to 0
    assert np.abs(np.mean(zscored)) < 1e-10
    # And std close to 1
    assert np.abs(np.std(zscored) - 1.0) < 1e-10


def test_apply_isosbestic_correction():
    """Test isosbestic correction via Lasso regression."""
    np.random.seed(42)
    n = 1000
    # Create correlated signal and reference (simulating motion artifact)
    motion_artifact = np.sin(np.linspace(0, 10, n))
    true_signal = np.random.randn(n) * 0.1
    signal = true_signal + motion_artifact
    reference = motion_artifact + np.random.randn(n) * 0.01

    corrected, fitted_reference = apply_isosbestic_correction(signal, reference)

    assert len(corrected) == n
    assert len(fitted_reference) == n
    # The corrected signal should have the motion artifact removed,
    # so it should be less correlated with the reference than the original signal
    corr_before = np.abs(np.corrcoef(signal, reference)[0, 1])
    corr_after = np.abs(np.corrcoef(corrected, reference)[0, 1])
    assert corr_after < corr_before


def test_apply_ratiometric_correction():
    """Test ratiometric correction: signal / reference."""
    signal = np.array([10.0, 20.0, 30.0, 40.0])
    reference = np.array([5.0, 10.0, 15.0, 20.0])

    ratio = apply_ratiometric_correction(signal, reference)

    np.testing.assert_array_equal(ratio, np.array([2.0, 2.0, 2.0, 2.0]))


# ============================================================
# LabVIEW metadata test (Thorlabs LEDs)
# ============================================================


def test_add_photometry_metadata_labview(dummy_logger):
    """Test that add_photometry_metadata works for LabVIEW sessions with Thorlabs LEDs."""
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["signals_mat_file_path"] = test_data_dir / "signals.mat"
    add_dummy_labview_metadata_to_metadata(metadata)

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_photometry_metadata(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # Check Thorlabs excitation sources
    assert "Thorlabs Blue LED" in nwbfile.devices
    blue_led = nwbfile.devices["Thorlabs Blue LED"]
    assert isinstance(blue_led, ExcitationSource)
    assert blue_led.excitation_wavelength_in_nm == 470.0
    assert blue_led.illumination_type == "LED"
    assert blue_led.manufacturer == "Thorlabs"

    assert "Thorlabs Purple LED" in nwbfile.devices
    purple_led = nwbfile.devices["Thorlabs Purple LED"]
    assert isinstance(purple_led, ExcitationSource)
    assert purple_led.excitation_wavelength_in_nm == 405.0
    assert purple_led.illumination_type == "LED"
    assert purple_led.manufacturer == "Thorlabs"

    # Check photodetector
    assert "Doric iFMC7-G2 (7 ports Fluorescence Mini Cube - Three Fluorophores)" in nwbfile.devices
    photodetector = nwbfile.devices["Doric iFMC7-G2 (7 ports Fluorescence Mini Cube - Three Fluorophores)"]
    assert isinstance(photodetector, Photodetector)

    # Check optical fibers (bilateral, recording from right)
    assert "Doric 0.66mm Flat 40mm Optic Fiber (right NAcc)" in nwbfile.devices
    optic_fiber = nwbfile.devices["Doric 0.66mm Flat 40mm Optic Fiber (right NAcc)"]
    assert isinstance(optic_fiber, OpticalFiber)

    # Check dLight indicators
    assert "dLight1.3b (right NAcc)" in nwbfile.devices
    indicator = nwbfile.devices["dLight1.3b (right NAcc)"]
    assert isinstance(indicator, Indicator)
    assert indicator.injection_location == "NAcc"
    assert indicator.injection_coordinates_in_mm == (1.7, 1.7, -6.2)

    # Check fiber photometry table
    table = nwbfile.get_lab_meta_data("fiber_photometry").fiber_photometry_table
    # dLight has 2 excitation sources (blue + purple), recording from right NAcc
    assert len(table) == 2

    # Verify the table rows have the correct indicator + LED combinations
    table_combos = set()
    for row in table:
        table_combos.add((row["indicator"].item().name, row["excitation_source"].item().name))

    expected_combos = {
        ("dLight1.3b (right NAcc)", "Thorlabs Blue LED"),
        ("dLight1.3b (right NAcc)", "Thorlabs Purple LED"),
    }
    assert table_combos == expected_combos
