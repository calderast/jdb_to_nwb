import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from jdb_to_nwb.utils import to_datetime
from jdb_to_nwb.convert import check_required_metadata, set_default_metadata


def test_check_required_metadata_with_all_required_metadata():
    """
    Test that check_required_metadata runs without errors and modifies metadata in-place
    when all required fields are present.
    """
    metadata = {
        "experimenter": "Tim Krausz",
        "date": "07/25/2022",
        "subject": {
            "subject_id": "IM-1478",
            "species": "Rattus norvegicus",
            "genotype": "Wildtype",
            "sex": "M",
            "description": "Long Evans Rat",
            "date_of_birth": "1/20/2022"
        }
    }

    # We expect no errors - all required metadata exists!
    check_required_metadata(metadata)

    # Confirm that animal_name was set to subject_id
    assert metadata["animal_name"] == "IM-1478", (
        f"Expected 'animal_name' to be set to 'IM-1478', got {metadata.get('animal_name')}"
    )

    # Confirm date_of_birth was converted to datetime
    assert isinstance(metadata["subject"]["date_of_birth"], datetime), (
        f"Expected 'date_of_birth' to be set to datetime, got {type(metadata['subject']['date_of_birth'])}"
    )


@pytest.mark.parametrize("missing_field", ["experimenter", "date", "subject"])
def test_check_required_metadata_with_missing_metadata(missing_field):
    """
    Test that missing top-level required fields 'experimenter', 'date', or 'subject' raises an AssertionError.
    """
    metadata = {
        "experimenter": "Tim Krausz",
        "date": "07/25/2022",
        "subject": {
            "subject_id": "IM-1478",
            "species": "Rattus norvegicus",
            "genotype": "Wildtype",
            "sex": "M",
            "description": "Long Evans Rat",
            "date_of_birth": "1/20/2022"
        }
    }

    # Remove each required field to test check_required_metadata complains at us
    metadata.pop(missing_field)

    # We expect an error to be raised for each missing field
    with pytest.raises(AssertionError, match=f"Required field '{missing_field}' not found"):
        check_required_metadata(metadata)


@pytest.mark.parametrize("missing_subfield", ["subject_id", "species", "genotype", "sex", "description"])
def test_check_subject_metadata_with_missing_subfields(missing_subfield):
    """
    Test that missing required subfields under subject raises an AssertionError.
    """
    # Check that we raise an error for any missing subject subfield
    metadata = {
        "experimenter": "Tim Krausz",
        "date": "07/25/2022",
        "subject": {
            "subject_id": "IM-1478",
            "species": "Rattus norvegicus",
            "genotype": "Wildtype",
            "sex": "M",
            "description": "Long Evans Rat",
            "date_of_birth": "1/20/2022"
        }
    }

    metadata["subject"].pop(missing_subfield)

    with pytest.raises(AssertionError, match="Required subfields .* not found in subject metadata"):
        check_required_metadata(metadata)
    
    # Check that we raise an error if either age OR date_of_birth is missing
    metadata = {
        "experimenter": "Tim Krausz",
        "date": "07/25/2022",
        "subject": {
            "subject_id": "IM-1478",
            "species": "Rattus norvegicus",
            "genotype": "Wildtype",
            "sex": "M",
            "description": "Long Evans Rat",
        }
    }

    with pytest.raises(AssertionError, match="Required subfield 'age' or 'date_of_birth' not found"):
        check_required_metadata(metadata)


def test_set_default_metadata_sets_missing_fields(dummy_logger):
    """
    Test that set_default_metadata sets expected default values and logs warnings for missing fields.
    """
    metadata = {}

    set_default_metadata(metadata, dummy_logger)

    # Confirm values were set
    assert metadata["institution"] == "University of California, San Francisco", (
        "Expected default institution to be set to 'University of California, San Francisco', "
        f"got {metadata.get('institution')}"
    )
    assert metadata["lab"] == "Berke Lab", (
        f"Expected default lab to be set to 'Berke Lab', got {metadata.get('lab')}"
    )
    assert metadata["experiment_description"] == "Hex maze task", (
        "Expected default experiment_description to be set to 'Hex maze task', "
        f"got {metadata.get('experiment_description')}"
    )


def test_set_default_metadata_does_not_overwrite_existing(dummy_logger):
    """
    Test that set_default_metadata does not overwrite fields already present in metadata.
    """
    metadata = {
        "institution": "UCSF",
        "lab": "Frank Lab",
        "experiment_description": "Triangle maze!"
    }

    # No defaults should be set, because we specified values for institution, lab, and experiment_description
    set_default_metadata(metadata, dummy_logger)

    assert metadata["institution"] == "UCSF"
    assert metadata["lab"] == "Frank Lab"
    assert metadata["experiment_description"] == "Triangle maze!"


def test_to_datetime_all_cases():
    """
    Check that our to_datetime function is happy to handle all cases of MMDDYYYY and YYYYMMDD  
    (including variations MM/DD/YYYY, MM-DD-YYYY, YYYY/MM/DD, YYYY-MM-DD, or when month 
    is a single digit, which happens when the user specifies an int date instead of a string)
    """
    pacific = ZoneInfo("America/Los_Angeles")

    test_cases = [
        # Test so many valid ways of specifying the same date!!!!
        ("01222024", datetime(2024, 1, 22, tzinfo=pacific)),
        ("1222024", datetime(2024, 1, 22, tzinfo=pacific)),
        ("20240122", datetime(2024, 1, 22, tzinfo=pacific)),
        ("2024-01-22", datetime(2024, 1, 22, tzinfo=pacific)),
        ("2024/01/22", datetime(2024, 1, 22, tzinfo=pacific)),
        ("01/22/2024", datetime(2024, 1, 22, tzinfo=pacific)),
        ("01-22-2024", datetime(2024, 1, 22, tzinfo=pacific)),
        ("1/22/2024", datetime(2024, 1, 22, tzinfo=pacific)),
        ("1-22-2024", datetime(2024, 1, 22, tzinfo=pacific)),
        # An int!
        (1222024, datetime(2024, 1, 22, tzinfo=pacific)),
        # Already a datetime!
        ("2024-01-22T00:00:00-08:00", datetime(2024, 1, 22, tzinfo=pacific)),
        (datetime(2024, 1, 22), datetime(2024, 1, 22)),
        # These should error
        ("12224", ValueError),
        ("nope2024", ValueError),
    ]

    for input_val, expected in test_cases:
        if isinstance(expected, datetime):
            result = to_datetime(input_val)
            assert result == expected, f"to_datetime failed for input date: {input_val}"
        elif expected is ValueError:
            with pytest.raises(ValueError):
                to_datetime(input_val)
