from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
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
    PhotometrySignalBundle,
    apply_rolling_mean,
    apply_lowpass_filter,
    apply_highpass_filter,
    apply_airpls_baseline,
    apply_zscore_median_std,
    apply_zscore_mean_std,
    apply_isosbestic_correction,
    apply_ratiometric_correction,
    process_single_signal,
)


### Helper functions to add photometry metadata to the metadata dict so the tests can run

def add_dummy_2_signal_dlight_metadata(metadata):
    """
    Add values to the metadata dictionary so that the tests can run.
    Values assume recording 2 signals (dLight at 470nm and 405nm) in the old maze room (Thorlabs LEDs)
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


def add_dummy_3_signal_gach_rda_metadata(metadata):
    """
    Add values to the metadata dictionary so that the tests can run.
    Values assume recording 3 signals (gACh4h at 470nm and 405nm, rDA3m at 565nm) in the new maze room (Doric LEDs)
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
    

######################## Test adding photometry metadata + correct signal mappings ########################

def test_add_photometry_metadata_2_signals(dummy_logger):
    """
    Test that the add_photometry_metadata function adds the expected metadata to the NWB file.
    For this test, we use 2 signal (dLight at 470nm and 405nm) metadata. 
    """
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["signals_mat_file_path"] = test_data_dir / "signals.mat"
    # Add metadata for recording dLight in the old maze room to the metadata dict
    add_dummy_2_signal_dlight_metadata(metadata)
    
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


def test_add_photometry_metadata_3_signals(dummy_logger):
    """
    Test that the add_photometry_metadata function adds the expected metadata to the NWB file.
    For this test, we use 3 signal (gACh4h at 470nm and 405nm, rDA3m at 565nm) metadata. 
    """

    # Create a test metadata dictionary with pyPhotometry data and metadata
    test_data_dir = Path("tests/test_data/downloaded/IM-1770_corvette/11062024")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["ppd_file_path"] = test_data_dir / "Lhem_barswitch_GACh4h_rDA3m_CKTL-2024-11-06-185407.ppd"
    # Add metadata for recording gACh4h and rDA3m in the new maze room to the metadata dict
    add_dummy_3_signal_gach_rda_metadata(metadata)

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
    # Add metadata for recording gACh4h and rDA3m in the new maze room to the metadata dict
    add_dummy_3_signal_gach_rda_metadata(metadata)

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


def test_photometry_series_mappings(dummy_logger):
    """Test that each FiberPhotometryResponseSeries is linked to the correct fiber_photometry_table row."""

    test_data_dir = Path("tests/test_data/downloaded/IM-1770_corvette/11062024")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["ppd_file_path"] = test_data_dir / "Lhem_barswitch_GACh4h_rDA3m_CKTL-2024-11-06-185407.ppd"
    # Add metadata for recording gACh4h and rDA3m in the new maze room to the metadata dict
    add_dummy_3_signal_gach_rda_metadata(metadata)

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


######################## Test loading processing presets and overrides ########################

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


######################## Test loading signals into a PhotometrySignalBundle ########################
# We have 4 cases:
# 1. Load LabVIEW from signals.mat (legacy)
# 2. Load LabVIEW from raw phot and box files
# 3. Load pyPhotometry from ppd file (3 signals)
# 4. Load pyPhotometry from ppd file (2 signals)

def test_load_labview_mat(dummy_logger, labview_mat_ref):
    """Test that load_labview_mat returns a PhotometrySignalBundle matching the reference npz."""
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")
    signals_mat_path = test_data_dir / "signals.mat"
    bundle = load_labview_mat(signals_mat_path, dummy_logger)

    # Check bundle structure
    assert isinstance(bundle, PhotometrySignalBundle)
    assert bundle.sampling_rate == 250
    assert bundle.source == "labview_mat"
    assert bundle.photometry_start is None  # signals.mat has no start time
    assert set(bundle.signals.keys()) == {"470nm", "405nm"}
    assert len(bundle.port_visits) == 188

    # Signals and port visits must match the reference exactly
    np.testing.assert_array_equal(bundle.signals["470nm"], labview_mat_ref["raw_green"])
    np.testing.assert_array_equal(bundle.signals["405nm"], labview_mat_ref["raw_reference"])
    np.testing.assert_array_equal(bundle.port_visits, labview_mat_ref["port_visits"])


def test_load_labview_raw(dummy_logger, labview_dlight_ref):
    """Test that load_labview_raw returns a PhotometrySignalBundle matching the reference npz."""
    test_data_dir = Path("tests/test_data/downloaded/IM-1478/07252022")
    phot_file_path = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.phot"
    box_file_path = test_data_dir / "IM-1478_2022-07-25_15-24-22____Tim_Conditioning.box"
    bundle = load_labview_raw(phot_file_path, box_file_path, dummy_logger)

    # Check bundle structure
    assert isinstance(bundle, PhotometrySignalBundle)
    assert isinstance(bundle.photometry_start, datetime)
    assert bundle.sampling_rate == 250
    assert bundle.source == "labview_raw"
    assert set(bundle.signals.keys()) == {"470nm", "405nm"}
    assert len(bundle.port_visits) == 188

    # Signals must match the reference exactly (same Python lock-in + downsample)
    np.testing.assert_array_equal(bundle.signals["470nm"], labview_dlight_ref["raw_green"])
    np.testing.assert_array_equal(bundle.signals["405nm"], labview_dlight_ref["raw_reference"])
    np.testing.assert_array_equal(bundle.port_visits, labview_dlight_ref["port_visits"])


def test_load_3_signal_pyphotometry(dummy_logger, ppd_gach_rda_ref):
    """Test that load_pyphotometry returns a PhotometrySignalBundle matching the reference npz."""
    test_data_dir = Path("tests/test_data/downloaded/IM-1770_corvette/11062024")
    ppd_path = test_data_dir / "Lhem_barswitch_GACh4h_rDA3m_CKTL-2024-11-06-185407.ppd"
    bundle = load_pyphotometry(ppd_path, dummy_logger)

    # Check bundle structure
    assert isinstance(bundle, PhotometrySignalBundle)
    assert bundle.sampling_rate == 86
    assert bundle.source == "pyphotometry"
    assert isinstance(bundle.photometry_start, datetime)
    assert set(bundle.signals.keys()) == {"470nm", "405nm", "565nm"}
    assert len(bundle.port_visits) == 220

    # Signals and port visits must match the reference exactly
    np.testing.assert_array_equal(bundle.signals["470nm"], ppd_gach_rda_ref["raw_green"])
    np.testing.assert_array_equal(bundle.signals["405nm"], ppd_gach_rda_ref["raw_405"])
    np.testing.assert_array_equal(bundle.signals["565nm"], ppd_gach_rda_ref["raw_red"])
    np.testing.assert_array_equal(bundle.port_visits, ppd_gach_rda_ref["port_visits"])


def test_load_2_signal_pyphotometry(dummy_logger, ppd_dlight_ref):
    """Test that load_pyphotometry returns a PhotometrySignalBundle matching the reference npz for a 2-signal ppd."""
    test_data_dir = Path("tests/test_data/downloaded/IM-1947/20260422")
    ppd_path = test_data_dir / "IM-1947_L-2026-04-22-145706.ppd"
    bundle = load_pyphotometry(ppd_path, dummy_logger)

    # Check bundle structure
    assert isinstance(bundle, PhotometrySignalBundle)
    assert bundle.sampling_rate == 130
    assert bundle.source == "pyphotometry"
    assert isinstance(bundle.photometry_start, datetime)
    assert set(bundle.signals.keys()) == {"470nm", "405nm"}
    assert len(bundle.port_visits) == 373

    # Signals and port visits must match the reference exactly
    np.testing.assert_array_equal(bundle.signals["470nm"], ppd_dlight_ref["raw_green"])
    np.testing.assert_array_equal(bundle.signals["405nm"], ppd_dlight_ref["raw_reference"])
    np.testing.assert_array_equal(bundle.port_visits, ppd_dlight_ref["port_visits"])

    
######################## Test full process of going from raw photometry to signals in nwb ########################
# We have 4 cases:
# 1. Load LabVIEW from signals.mat (legacy)
# 2. Load LabVIEW from raw phot and box files
# 3. Load pyPhotometry from ppd file (3 signals)
# 4. Load pyPhotometry from ppd file (2 signals)


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
    # Add metadata for recording dLight in the old maze room to the metadata dict
    add_dummy_2_signal_dlight_metadata(metadata)

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
    # Add metadata for recording dLight in the old maze room to the metadata dict
    add_dummy_2_signal_dlight_metadata(metadata)

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

    np.testing.assert_allclose(
       corrected_dFF,
       green_dFF_reference,
       atol=0.005,
       rtol=0.05,
       err_msg="Data mismatch between nwbfile corrected_470_dFF and reference data",
    )


def test_add_photometry_from_3_signal_pyphotometry(dummy_logger):
    """
    Test that the add_photometry function results in the expected FiberPhotometryResponseSeries.

    This version of the test uses the ppd file from pyPhotometry to add photometry signals to the NWB.
    """

    # Create a test metadata dictionary with pyPhotometry data
    test_data_dir = Path("tests/test_data/downloaded/IM-1770_corvette/11062024")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["ppd_file_path"] = test_data_dir / "Lhem_barswitch_GACh4h_rDA3m_CKTL-2024-11-06-185407.ppd"
    # Add metadata for recording gACh4h and rDA3m in the new maze room to the metadata dict
    add_dummy_3_signal_gach_rda_metadata(metadata)

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


def test_add_photometry_from_2_signal_pyphotometry(dummy_logger, ppd_dlight_ref):
    """
    Test that the add_photometry function results in the expected FiberPhotometryResponseSeries.
    Here we do not have reference data from sampleframes. We use signals data generated via the pipeline as
    of commit 1c415f98146fea2f0f9e2d9bf39c442569048fa3 (before the big refactor) as our ground truth reference
    to ensure the refactor and subsequent changes don't break things.

    This version uses a 2-signal pyPhotometry ppd file (470nm dLight + 405nm isosbestic reference).
    Compares all NWB signals against the reference npz with tight tolerance.
    """
    test_data_dir = Path("tests/test_data/downloaded/IM-1947/20260422")
    metadata = {}
    metadata["photometry"] = {}
    metadata["photometry"]["ppd_file_path"] = test_data_dir / "IM-1947_L-2026-04-22-145706.ppd"
    # Add metadata for recording dLight in the old maze room to the metadata dict
    add_dummy_2_signal_dlight_metadata(metadata)

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    photometry_data_dict = add_photometry(nwbfile=nwbfile, metadata=metadata, logger=dummy_logger)

    # dLight 2-signal isosbestic pipeline produces these 6 series
    expected_photometry_series = {"raw_470", "raw_405", "processed_470", "processed_405",
                                  "corrected_470_dFF", "fitted_405"}
    expected_sampling_rate = 130  # Hz

    assert photometry_data_dict.get('sampling_rate') == expected_sampling_rate, (
        f"Expected sampling rate {expected_sampling_rate} Hz, got {photometry_data_dict.get('sampling_rate')} Hz")
    assert isinstance(photometry_data_dict.get('photometry_start'), datetime), (
        f"Expected photometry_start as datetime, got {type(photometry_data_dict.get('photometry_start'))}")
    assert isinstance(photometry_data_dict.get('port_visits'), (list, np.ndarray)), (
        f"Expected port_visits as list or array, got {type(photometry_data_dict.get('port_visits'))}")

    actual_photometry_series = set(nwbfile.acquisition.keys())
    missing_photometry_series = expected_photometry_series - actual_photometry_series
    assert not missing_photometry_series, f"Missing FiberPhotometryResponseSeries: {missing_photometry_series}"

    for series_name in expected_photometry_series:
        assert isinstance(nwbfile.acquisition[series_name], FiberPhotometryResponseSeries), (
            f"{series_name} is not of type FiberPhotometryResponseSeries")
        assert getattr(nwbfile.acquisition[series_name], "rate", None) == expected_sampling_rate, (
            f"{series_name} has unexpected sampling rate {getattr(nwbfile.acquisition[series_name], 'rate', None)}, "
            f"expected {expected_sampling_rate}")

    # Compare all signals against reference with tight tolerance (same Python pipeline = exact match)
    nwb_to_ref = [
        ("raw_470",           "raw_green",                 False),
        ("raw_405",           "raw_reference",             False),
        ("processed_470",     "z_scored_green",            True),
        ("processed_405",     "z_scored_reference",        True),
        ("fitted_405",        "z_scored_reference_fitted", True),
        ("corrected_470_dFF", "z_scored_green_dFF",        True),
    ]
    for nwb_name, ref_key, do_flatten in nwb_to_ref:
        signal = np.array(nwbfile.acquisition[nwb_name].data)
        reference = ppd_dlight_ref[ref_key].flatten() if do_flatten else ppd_dlight_ref[ref_key]
        np.testing.assert_allclose(
            signal, reference, atol=1e-10,
            err_msg=f"Mismatch between NWB '{nwb_name}' and reference '{ref_key}'",
        )


######################## Unit tests of individual processing steps ########################

# These tests load the intermediate arrays saved by tests/generate_reference_data.ipynb
# Test that each of our new modular functions (post- photometry refactor) produce the same output
# as each step in the pipeline as of commit 1c415f98146fea2f0f9e2d9bf39c442569048fa3 (pre-refactor).

# Reference data lives in tests/test_data/reference/<main-commit-hash>/*.npz.
# Run the notebook to regenerate if the data is missing or the commit hash changes.
_REFERENCE_DIR = Path("tests/test_data/reference/1c415f98146fea2f0f9e2d9bf39c442569048fa3")


@pytest.fixture(scope="module")
def labview_dlight_ref():
    npz_path = _REFERENCE_DIR / "labview_dlight_intermediates.npz"
    if not npz_path.exists():
        pytest.skip("LabVIEW raw reference data not found. Run tests/generate_reference_data.ipynb first.")
    return np.load(npz_path)


@pytest.fixture(scope="module")
def labview_mat_ref():
    npz_path = _REFERENCE_DIR / "labview_mat_dlight_intermediates.npz"
    if not npz_path.exists():
        pytest.skip("LabVIEW mat reference data not found. Run tests/generate_reference_data.ipynb first.")
    return np.load(npz_path)


@pytest.fixture(scope="module")
def ppd_gach_rda_ref():
    npz_path = _REFERENCE_DIR / "pyphotometry_gach_rda_intermediates.npz"
    if not npz_path.exists():
        pytest.skip("pyPhotometry gACh4h+rDA3m reference data not found. Run tests/generate_reference_data.ipynb first.")
    return np.load(npz_path)


@pytest.fixture(scope="module")
def ppd_dlight_ref():
    npz_path = _REFERENCE_DIR / "pyphotometry_dlight_intermediates.npz"
    if not npz_path.exists():
        pytest.skip("pyPhotometry dLight reference data not found. Run tests/generate_reference_data.ipynb first.")
    return np.load(npz_path)


def test_apply_rolling_mean_dlight(labview_mat_ref):
    """Rolling mean output matches reference for LabVIEW 470nm and 405nm signals."""
    sampling_rate = float(labview_mat_ref["sampling_rate"])
    for signal_key, smoothed_key in [
        ("raw_green", "smoothed_green"),
        ("raw_reference", "smoothed_reference"),
    ]:
        result = apply_rolling_mean(labview_mat_ref[signal_key], sampling_rate=sampling_rate)
        np.testing.assert_allclose(
            result, labview_mat_ref[smoothed_key].flatten(), atol=1e-10,
            err_msg=f"Rolling mean mismatch for {signal_key}",
        )


def test_apply_rolling_mean_ppd2_dlight(ppd_dlight_ref):
    """Rolling mean output matches reference for 2-signal pyPhotometry dLight."""
    sampling_rate = float(ppd_dlight_ref["sampling_rate"])
    for signal_key, smoothed_key in [
        ("raw_green", "smoothed_green"),
        ("raw_reference", "smoothed_reference"),
    ]:
        result = apply_rolling_mean(ppd_dlight_ref[signal_key], sampling_rate=sampling_rate)
        np.testing.assert_allclose(
            result, ppd_dlight_ref[smoothed_key].flatten(), atol=1e-10,
            err_msg=f"Rolling mean mismatch for {signal_key}",
        )


def test_apply_airpls_baseline_dlight(labview_mat_ref, dummy_logger):
    """airPLS baseline and baseline-subtracted signal match reference for LabVIEW dLight."""
    for raw_key, smoothed_key, baseline_key, subtracted_key in [
        ("raw_green", "smoothed_green", "green_baseline", "baseline_subtracted_green"),
        ("raw_reference", "smoothed_reference", "ref_baseline", "baseline_subtracted_ref"),
    ]:
        baseline_subtracted, baseline = apply_airpls_baseline(
            signal=labview_mat_ref[smoothed_key].flatten(),
            logger=dummy_logger,
            baseline_signal=labview_mat_ref[raw_key],
        )
        np.testing.assert_allclose(
            baseline, labview_mat_ref[baseline_key].flatten(), atol=1e-10,
            err_msg=f"airPLS baseline mismatch for {raw_key}",
        )
        np.testing.assert_allclose(
            baseline_subtracted, labview_mat_ref[subtracted_key].flatten(), atol=1e-10,
            err_msg=f"airPLS subtracted mismatch for {raw_key}",
        )


def test_apply_zscore_median_std_dlight(labview_mat_ref):
    """Median z-score matches reference for LabVIEW dLight baseline-subtracted signals."""
    for subtracted_key, zscored_key in [
        ("baseline_subtracted_green", "z_scored_green"),
        ("baseline_subtracted_ref", "z_scored_reference"),
    ]:
        result = apply_zscore_median_std(labview_mat_ref[subtracted_key].flatten())
        np.testing.assert_allclose(
            result, labview_mat_ref[zscored_key].flatten(), atol=1e-10,
            err_msg=f"Median z-score mismatch for {subtracted_key}",
        )


def test_apply_isosbestic_correction_dlight(labview_mat_ref):
    """Isosbestic Lasso correction (fitted reference and dF/F) matches reference for LabVIEW dLight."""
    z_scored_green = labview_mat_ref["z_scored_green"].flatten()
    z_scored_reference = labview_mat_ref["z_scored_reference"].flatten()

    corrected, fitted_reference = apply_isosbestic_correction(z_scored_green, z_scored_reference)

    np.testing.assert_allclose(
        fitted_reference, labview_mat_ref["z_scored_reference_fitted"].flatten(), atol=1e-10,
        err_msg="Lasso fitted reference mismatch",
    )
    np.testing.assert_allclose(
        corrected, labview_mat_ref["z_scored_green_dFF"].flatten(), atol=1e-10,
        err_msg="dF/F mismatch after isosbestic correction",
    )


def test_apply_isosbestic_correction_ppd2_dlight(ppd_dlight_ref):
    """Isosbestic Lasso correction matches reference for 2-signal pyPhotometry dLight."""
    z_scored_green = ppd_dlight_ref["z_scored_green"].flatten()
    z_scored_reference = ppd_dlight_ref["z_scored_reference"].flatten()

    corrected, fitted_reference = apply_isosbestic_correction(z_scored_green, z_scored_reference)

    np.testing.assert_allclose(
        fitted_reference, ppd_dlight_ref["z_scored_reference_fitted"].flatten(), atol=1e-10,
        err_msg="Lasso fitted reference mismatch (pyPhotometry dLight)",
    )
    np.testing.assert_allclose(
        corrected, ppd_dlight_ref["z_scored_green_dFF"].flatten(), atol=1e-10,
        err_msg="dF/F mismatch after isosbestic correction (pyPhotometry dLight)",
    )


def test_apply_ratiometric_correction_gach(ppd_gach_rda_ref):
    """Ratiometric correction (470/405) matches reference raw ratio."""
    result = apply_ratiometric_correction(ppd_gach_rda_ref["raw_green"], ppd_gach_rda_ref["raw_405"])
    np.testing.assert_allclose(
        result, ppd_gach_rda_ref["raw_ratio"], atol=1e-10,
        err_msg="Ratiometric correction mismatch",
    )


def test_apply_lowpass_filter_gach(ppd_gach_rda_ref):
    """Lowpass filter matches reference for pyPhotometry gACh4h signals."""
    sampling_rate = float(ppd_gach_rda_ref["sampling_rate"])
    for raw_key, lowpass_key in [
        ("raw_green", "green_lowpass"),
        ("raw_red", "red_lowpass"),
        ("raw_405", "lowpass_405"),
        ("raw_ratio", "ratio_lowpass"),
    ]:
        result = apply_lowpass_filter(ppd_gach_rda_ref[raw_key], sampling_rate=sampling_rate)
        np.testing.assert_allclose(
            result, ppd_gach_rda_ref[lowpass_key], atol=1e-10,
            err_msg=f"Lowpass filter mismatch for {raw_key}",
        )


def test_apply_highpass_filter_gach(ppd_gach_rda_ref):
    """Highpass filter matches reference for pyPhotometry gACh4h lowpass-filtered signals."""
    sampling_rate = float(ppd_gach_rda_ref["sampling_rate"])
    for lowpass_key, highpass_key in [
        ("green_lowpass", "green_highpass"),
        ("red_lowpass", "red_highpass"),
        ("lowpass_405", "highpass_405"),
        ("ratio_lowpass", "ratio_highpass"),
    ]:
        result = apply_highpass_filter(ppd_gach_rda_ref[lowpass_key], sampling_rate=sampling_rate)
        np.testing.assert_allclose(
            result, ppd_gach_rda_ref[highpass_key], atol=1e-10,
            err_msg=f"Highpass filter mismatch for {lowpass_key}",
        )


def test_apply_zscore_mean_std_gach(ppd_gach_rda_ref):
    """Mean z-score matches reference for pyPhotometry gACh4h highpass-filtered signals."""
    for highpass_key, zscored_key in [
        ("green_highpass", "green_zscored"),
        ("red_highpass", "red_zscored"),
        ("highpass_405", "zscored_405"),
        ("ratio_highpass", "ratio_zscored"),
    ]:
        result = apply_zscore_mean_std(ppd_gach_rda_ref[highpass_key])
        np.testing.assert_allclose(
            result, ppd_gach_rda_ref[zscored_key], atol=1e-10,
            err_msg=f"Mean z-score mismatch for {highpass_key}",
        )


def test_process_single_signal_dlight_isosbestic(labview_mat_ref, dummy_logger):
    """process_single_signal with dlight_isosbestic config matches all reference intermediates."""
    raw_green = labview_mat_ref["raw_green"]
    sampling_rate = float(labview_mat_ref["sampling_rate"])
    config = load_processing_config("dlight_isosbestic")

    result = process_single_signal(raw_green, sampling_rate, config, dummy_logger)

    np.testing.assert_allclose(
        result["smoothed"], labview_mat_ref["smoothed_green"].flatten(), atol=1e-10,
        err_msg="Smoothed signal mismatch",
    )
    np.testing.assert_allclose(
        result["baseline"], labview_mat_ref["green_baseline"].flatten(), atol=1e-10,
        err_msg="airPLS baseline mismatch",
    )
    np.testing.assert_allclose(
        result["baseline_subtracted"], labview_mat_ref["baseline_subtracted_green"].flatten(), atol=1e-10,
        err_msg="Baseline-subtracted signal mismatch",
    )
    np.testing.assert_allclose(
        result["normalized"], labview_mat_ref["z_scored_green"].flatten(), atol=1e-10,
        err_msg="Normalized (median z-scored) signal mismatch",
    )


def test_process_single_signal_gach_ratiometric(ppd_gach_rda_ref, dummy_logger):
    """process_single_signal with gach_ratiometric config matches reference intermediates for the ratio signal."""
    raw_ratio = ppd_gach_rda_ref["raw_ratio"]
    sampling_rate = float(ppd_gach_rda_ref["sampling_rate"])
    config = load_processing_config("gach_ratiometric")

    result = process_single_signal(raw_ratio, sampling_rate, config, dummy_logger)

    np.testing.assert_allclose(
        result["smoothed"], ppd_gach_rda_ref["ratio_lowpass"], atol=1e-10,
        err_msg="Lowpass-smoothed ratio mismatch",
    )
    np.testing.assert_allclose(
        result["baseline_subtracted"], ppd_gach_rda_ref["ratio_highpass"], atol=1e-10,
        err_msg="Highpass baseline-subtracted ratio mismatch",
    )
    np.testing.assert_allclose(
        result["normalized"], ppd_gach_rda_ref["ratio_zscored"], atol=1e-10,
        err_msg="Z-scored ratio mismatch",
    )
