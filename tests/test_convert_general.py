import subprocess
from pathlib import Path

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from jdb_to_nwb import utils
from jdb_to_nwb.utils import source_script_version, to_datetime
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


def _git_repo_with_a_tag(repo_root: Path, package_dir: Path) -> None:
    """A git repository with one tagged commit, holding a file where package_dir says it is."""
    package_dir.mkdir(parents=True)
    (package_dir / "utils.py").write_text("# stand-in for this module\n")
    for command in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "-c", "user.name=T", "-c", "user.email=t@t", "commit", "-qm", "initial"],
                    ["git", "tag", "v9.9.9"]):
        subprocess.run(command, cwd=repo_root, check=True, capture_output=True)


def test_source_script_version_uses_git_describe_in_this_project(dummy_logger, tmp_path, monkeypatch):
    """
    A checkout of THIS project is what `git describe` is consulted for: an editable install does
    not rewrite the installed version, so the working copy's own description is what it really is.
    """
    # The repository top level is two directories above the package, as it is in this project
    package_dir = tmp_path / "src" / "jdb_to_nwb"
    _git_repo_with_a_tag(tmp_path, package_dir)
    monkeypatch.setattr(utils, "__file__", str(package_dir / "utils.py"))

    assert source_script_version(dummy_logger) == "v9.9.9"


def test_source_script_version_ignores_an_unrelated_enclosing_repo(dummy_logger, tmp_path, monkeypatch):
    """
    An installed (non-editable) copy sits in site-packages, which may itself be inside somebody
    else's git repository. `git describe` answers for THAT repository, so believing it would record
    an unrelated project's version as the version of the code that did the conversion.
    """
    # The package is somewhere inside the repository, but the repository is not this project
    package_dir = tmp_path / "some_env" / "lib" / "site-packages" / "jdb_to_nwb"
    _git_repo_with_a_tag(tmp_path, package_dir)
    monkeypatch.setattr(utils, "__file__", str(package_dir / "utils.py"))

    assert source_script_version(dummy_logger) == utils.__version__
