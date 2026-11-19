from pathlib import Path

import yaml

from jdb_to_nwb.convert import check_required_metadata
from jdb_to_nwb.make_metadata import (
    FILL_IN,
    RAT_METADATA_FILENAME,
    derive_rat_metadata,
    dump_session_metadata,
    find_rat_dirs,
    find_session_dirs,
    generate_metadata,
)

BEHAVIOR_FILENAMES = ("arduinoraw0.txt", "ArduinoStamps0.csv", "barriers.txt")
VIDEO_FILENAMES = ("Behav_Vid0.avi", "testvidtimes0.csv",
                   "Behav_Vid0DLC_resnet50_Triangle_Maze_PhotMar24shuffle1_800000.h5")

# A filled-in rat_metadata.yaml for a photometry rat, as a user would leave it.
PHOTOMETRY_RAT_METADATA = {
    "experimenter": ["Figueroa, Jose"],
    "experiment_description": "Hex maze task",
    "subject": {"subject_id": "IM-1891", "species": "Rattus norvegicus", "sex": "M",
                "date_of_birth": "2024-10-17", "genotype": "Wildtype", "description": "Long Evans Rat"},
    "maze": "new_maze",
    # No excitation_sources or photodetector: those come from 'maze'.
    "photometry": {
        "optic_fiber_implant_sites": [{"optic_fiber": "Doric 0.66mm Flat 40mm Optic Fiber",
                                       "targeted_location": "NAcc", "ap_in_mm": 1.8,
                                       "ml_in_mm": -1.8, "dv_in_mm": -6.0, "recording": True}],
        "virus_injections": [{"virus_name": "dLight1.3b", "targeted_location": "NAcc", "ap_in_mm": 1.8,
                              "ml_in_mm": -1.8, "dv_in_mm": -6.2, "volume_in_uL": 1.0,
                              "titer_in_vg_per_mL": 2e12}],
    },
    "keywords": ["photometry", "hex maze"],
}

EPHYS_RAT_METADATA = {
    "experimenter": ["Crater, Stephanie"],
    "experiment_description": "Hex maze task with ephys",
    "subject": {"subject_id": "IM-1982", "species": "Rattus norvegicus", "sex": "M",
                "date_of_birth": "2025-06-01", "genotype": "Wildtype", "description": "Long Evans Rat"},
    "ephys": {"probes": [{"probe_type": "256-ch Silicon Probe, 3mm length, 66um pitch",
                          "electrodes_location": "Hippocampus CA1",
                          "targeted_x": -2.2, "targeted_y": 4.5, "targeted_z": -2.0}]},
}


def make_session(rat_dir: Path, date: str, filenames=(), dirnames=()) -> Path:
    """Create a date folder holding the given files and subdirectories."""
    session_dir = rat_dir / date
    session_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (session_dir / filename).write_text("test data")
    for dirname in dirnames:
        (session_dir / dirname).mkdir(exist_ok=True)
    return session_dir


def write_rat_metadata(rat_dir: Path, metadata: dict) -> Path:
    """Write a rat_metadata.yaml, as the user would once they have filled in the stub."""
    rat_dir.mkdir(parents=True, exist_ok=True)
    rat_metadata_path = rat_dir / RAT_METADATA_FILENAME
    rat_metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    return rat_metadata_path


def session_reports(reports) -> dict:
    """Flatten the report tree into {date folder name: (status, message)} for easy assertions."""
    return {session.session_dir.name: (session.status, session.message)
            for report in reports for session in report.sessions}


def generated_yamls(root: Path) -> list[Path]:
    return sorted(root.rglob("metadata_*.yaml"))



def test_a_rat_with_no_metadata_gets_a_stub_and_no_sessions(tmp_path):
    """
    A rat with nothing to go on gets a stub to fill in, and none of its sessions are generated.

    Generating them would mean inventing a subject, so the first run deliberately produces one file
    for the user to fill in rather than a folder full of placeholder yamls.
    """
    rat_dir = tmp_path / "IM-1891_jimmy"
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES + VIDEO_FILENAMES)

    reports = generate_metadata(tmp_path, write=True)

    assert [report.status for report in reports] == ["stub written"]
    stub = (rat_dir / RAT_METADATA_FILENAME).read_text()
    assert FILL_IN in stub
    # The folder name tells us the subject_id, so that much is filled in already.
    assert "subject_id: IM-1891" in stub
    assert generated_yamls(tmp_path) == []


def test_a_stub_that_is_still_unfilled_blocks_generation(tmp_path):
    """Generation is refused while a required field is still FILL IN, naming the field."""
    rat_dir = tmp_path / "IM-1891_jimmy"
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES)
    generate_metadata(tmp_path, write=True)  # writes the stub

    reports = generate_metadata(tmp_path, write=True)

    assert reports[0].status == "skipped"
    # Every unfilled field is named, so the user does not have to re-run to find the next one.
    for expected in ("experimenter[0]", "subject.sex", "subject.date_of_birth", "subject.genotype"):
        assert expected in reports[0].message
    assert generated_yamls(tmp_path) == []


def test_dry_run_writes_nothing(tmp_path):
    """The default run reports what it would do and leaves the disk alone."""
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES + VIDEO_FILENAMES)

    reports = generate_metadata(tmp_path, write=False)

    assert session_reports(reports)["20250723"][0] == "written"
    assert generated_yamls(tmp_path) == []


def test_sections_follow_the_files_that_are_there(tmp_path):
    """
    Each session gets exactly the sections its own folder has data for.

    This is the heart of it: the same rat_metadata.yaml produces a behavior-only training session, a
    photometry session, and an ephys session, purely from what is on disk.
    """
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, {**PHOTOMETRY_RAT_METADATA, **{"ephys": EPHYS_RAT_METADATA["ephys"]}})
    make_session(rat_dir, "20250418", BEHAVIOR_FILENAMES)
    make_session(rat_dir, "20250419", BEHAVIOR_FILENAMES + VIDEO_FILENAMES)
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES + VIDEO_FILENAMES + ("recording.ppd",))
    make_session(rat_dir, "20250724", BEHAVIOR_FILENAMES + VIDEO_FILENAMES + ("impedance.csv",),
                 dirnames=("2025-07-24_14-20-43",))

    generate_metadata(tmp_path, write=True)

    sections = {}
    for path in generated_yamls(tmp_path):
        metadata = yaml.safe_load(path.read_text())
        sections[path.parent.name] = sorted(set(metadata) & {"behavior", "video", "photometry", "ephys"})

    assert sections == {"20250418": ["behavior"],
                        "20250419": ["behavior", "video"],
                        "20250723": ["behavior", "photometry", "video"],
                        "20250724": ["behavior", "ephys", "video"]}


def test_paths_are_absolute_so_the_yaml_works_from_anywhere(tmp_path, monkeypatch):
    """A session yaml is run from wherever the user happens to be, so relative paths would break."""
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES)

    monkeypatch.chdir(tmp_path)
    generate_metadata(Path("."), write=True)  # a relative root

    metadata = yaml.safe_load(generated_yamls(tmp_path)[0].read_text())
    for key, value in metadata["behavior"].items():
        assert Path(value).is_absolute(), f"{key} is relative: {value}"
        assert Path(value).exists()


def test_hex_centroids_are_found_in_the_rat_folder(tmp_path):
    """The maze does not move between sessions, so one centroids file is often kept for the rat."""
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
    (rat_dir / "IM-1891_jimmy_hex_coordinates.csv").write_text("test data")
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES + VIDEO_FILENAMES)

    generate_metadata(tmp_path, write=True)
    metadata = yaml.safe_load(generated_yamls(tmp_path)[0].read_text())

    assert metadata["video"]["hex_centroids_file_path"].endswith("IM-1891_jimmy/IM-1891_jimmy_hex_coordinates.csv")


def test_a_session_missing_a_required_behavior_file_is_skipped(tmp_path):
    """A day still being collected has no barriers.txt yet, so there is nothing to generate."""
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
    make_session(rat_dir, "20250725", ("arduinoraw0.txt", "ArduinoStamps0.csv"))

    reports = generate_metadata(tmp_path, write=True)

    status, message = session_reports(reports)["20250725"]
    assert status == "skipped"
    assert "maze_configuration_file_path" in message
    assert generated_yamls(tmp_path) == []


def test_two_files_matching_one_field_is_reported_not_guessed(tmp_path):
    """Two .ppd files in one folder could be either recording, so the session is left alone."""
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES + ("morning.ppd", "afternoon.ppd"))

    reports = generate_metadata(tmp_path, write=True)

    status, message = session_reports(reports)["20250723"]
    assert status == "skipped"
    assert "2 files" in message and "afternoon.ppd" in message
    assert generated_yamls(tmp_path) == []


def test_half_a_labview_recording_is_skipped_rather_than_dropped(tmp_path):
    """
    A .phot with no .box cannot be converted, so the session is skipped.

    Writing the yaml without a photometry section would be worse: it would never be regenerated, and
    it would convert to an nwb that looks finished and silently has no photometry in it.
    """
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES + ("IM-1891_2025-07-23____Conditioning.phot",))

    reports = generate_metadata(tmp_path, write=True)

    status, message = session_reports(reports)["20250723"]
    assert status == "skipped"
    assert "box_file_path" in message
    assert generated_yamls(tmp_path) == []


def test_a_day_with_two_open_ephys_recordings_is_skipped(tmp_path):
    """
    Each recording after the first needs a 'session' label, and only the experimenter knows them.

    The label becomes the nwb epoch tag and has to match the preprocessing's session_id, so guessing
    one would be worse than leaving the day for the user to write by hand.
    """
    rat_dir = tmp_path / "IM-1982"
    write_rat_metadata(rat_dir, EPHYS_RAT_METADATA)
    make_session(rat_dir, "20261022", BEHAVIOR_FILENAMES + ("impedance.csv",),
                 dirnames=("2026-10-22_14-20-43", "2026-10-22_16-39-38"))

    reports = generate_metadata(tmp_path, write=True)

    status, message = session_reports(reports)["20261022"]
    assert status == "skipped"
    assert "2 Open Ephys recordings" in message and "session" in message
    assert generated_yamls(tmp_path) == []


def test_ephys_data_with_no_probes_block_is_skipped(tmp_path):
    """Which probes are implanted, and where, cannot be read off the recording folder."""
    rat_dir = tmp_path / "IM-1982"
    write_rat_metadata(rat_dir, {k: v for k, v in EPHYS_RAT_METADATA.items() if k != "ephys"})
    make_session(rat_dir, "20261021", BEHAVIOR_FILENAMES, dirnames=("2026-10-21_14-20-43",))

    reports = generate_metadata(tmp_path, write=True)

    status, message = session_reports(reports)["20261021"]
    assert status == "skipped"
    assert "probes" in message


def test_a_second_session_in_one_day_gets_its_own_name(tmp_path):
    """Nwb files are named animal_name + date, so two sessions in a day would otherwise collide."""
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
    make_session(rat_dir, "20250724", BEHAVIOR_FILENAMES)
    make_session(rat_dir, "20250724_2", BEHAVIOR_FILENAMES)

    generate_metadata(tmp_path, write=True)

    names = {path.parent.name: yaml.safe_load(path.read_text())["animal_name"]
             for path in generated_yamls(tmp_path)}
    assert names == {"20250724": "IM-1891_jimmy", "20250724_2": "IM-1891_jimmy_2"}
    assert {path.name for path in generated_yamls(tmp_path)} == {
        "metadata_IM-1891_jimmy_20250724.yaml", "metadata_IM-1891_jimmy_2_20250724.yaml"}


def test_an_existing_metadata_file_is_never_touched(tmp_path):
    """Whatever the user has already written wins, whatever it is called."""
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
    session_dir = make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES)
    existing = session_dir / "my_own_metadata.yaml"
    existing.write_text("hand written, do not touch")

    reports = generate_metadata(tmp_path, write=True)

    assert session_reports(reports)["20250723"] == ("exists", "my_own_metadata.yaml")
    assert existing.read_text() == "hand written, do not touch"
    assert generated_yamls(tmp_path) == []


def test_a_generated_yaml_passes_the_converters_own_check(tmp_path):
    """
    The point of all this is a file the conversion accepts, so check it with the conversion's gate.

    check_required_metadata is what create_nwbs runs before anything else, and it raises on a
    missing field rather than returning, so reaching the assertions below is the test.
    """
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES + VIDEO_FILENAMES + ("recording.ppd",))

    generate_metadata(tmp_path, write=True)
    metadata = yaml.safe_load(generated_yamls(tmp_path)[0].read_text())
    check_required_metadata(metadata)

    assert metadata["animal_name"] == "IM-1891_jimmy"
    assert metadata["subject"]["subject_id"] == "IM-1891"


def test_derive_rat_metadata_drops_the_date_and_every_path(tmp_path):
    """
    A rat that already has a session yaml gets its rat-level file lifted out of it.

    Everything that varies by session has to go, or the next session would inherit the last one's
    file paths.
    """
    session_metadata = {
        "institution": "University of California, San Francisco",
        "experimenter": ["Figueroa, Jose"],
        "animal_name": "IM-1891_jimmy",
        "date": "20250723",
        "subject": {"subject_id": "IM-1891", "sex": "M"},
        "photometry": {**PHOTOMETRY_RAT_METADATA["photometry"],
                       "ppd_file_path": "/data/old/recording.ppd", "phot_end_time_mins": 77.5},
        "ephys": {"probes": [{"probe_type": "Neuropixels 2.0 (4-shank)"}],
                  "openephys_folder_path": "/data/old/2025-07-23_13-12-30",
                  "impedance_file_path": "/data/old/impedance.csv"},
        "behavior": {"arduino_text_file_path": "/data/old/arduinoraw0.txt"},
        "video": {"video_file_path": "/data/old/Behav_Vid0.avi"},
        "surgery": "Implanted 2025-05-03.",
    }

    derived = derive_rat_metadata(session_metadata)

    assert "date" not in derived and "animal_name" not in derived
    assert "behavior" not in derived and "video" not in derived
    assert derived["photometry"] == PHOTOMETRY_RAT_METADATA["photometry"]
    assert derived["ephys"] == {"probes": [{"probe_type": "Neuropixels 2.0 (4-shank)"}]}
    assert derived["surgery"] == "Implanted 2025-05-03."
    assert derived["institution"] == "University of California, San Francisco"


def test_an_unrecognized_field_is_carried_through_rather_than_dropped(tmp_path):
    """
    A field this script has not heard of is more likely new than wrong, so it survives.

    It has to survive both steps: being lifted into the rat file, and being written back out into
    each session yaml.
    """
    rat_dir = tmp_path / "IM-1891_jimmy"
    old_session = make_session(rat_dir, "20250801", BEHAVIOR_FILENAMES)
    (old_session / "metadata_IM-1891_jimmy_20250801.yaml").write_text(yaml.safe_dump(
        {**PHOTOMETRY_RAT_METADATA, "date": "20250801", "animal_name": "IM-1891_jimmy",
         "some_new_field": "keep me", "session_description": "also keep me"}))
    make_session(rat_dir, "20250802", BEHAVIOR_FILENAMES)

    generate_metadata(tmp_path, write=True)   # derives the rat file
    derived = yaml.safe_load((rat_dir / RAT_METADATA_FILENAME).read_text())
    assert derived["some_new_field"] == "keep me"

    generate_metadata(tmp_path, write=True)   # now generates the new session
    new_metadata = yaml.safe_load((rat_dir / "20250802" / "metadata_IM-1891_jimmy_20250802.yaml").read_text())
    assert new_metadata["some_new_field"] == "keep me"
    assert new_metadata["session_description"] == "also keep me"


def test_a_misspelled_maze_is_caught_before_anything_is_generated(tmp_path):
    """
    A typo here would otherwise be copied into every session of this rat.

    The conversion would catch it eventually, but only after the whole rat had been generated with
    it, so it is worth failing on the one file instead.
    """
    rat_dir = tmp_path / "IM-1891_jimmy"
    write_rat_metadata(rat_dir, {**PHOTOMETRY_RAT_METADATA, "maze": "old"})
    make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES)

    reports = generate_metadata(tmp_path, write=True)

    assert reports[0].status == "skipped"
    assert "'maze' is 'old'" in reports[0].message
    assert generated_yamls(tmp_path) == []


def test_folder_scanning(tmp_path):
    """
    What counts as a rat folder and a date folder.

    Date folders are named in any format to_datetime accepts, optionally with the _2 suffix for a
    second session the same day. Anything else is ignored, including a folder that looks like a date
    but is not a real one, and a folder in the root with no date folders under it is not a rat.
    """
    rat_dir = tmp_path / "IM-1891_jimmy"
    for name in ("20250723", "07242025", "2025-07-25", "07-26-2025", "20250723_2"):
        make_session(rat_dir, name)
    for name in ("notes", "histology", "20259999", "DLC_models"):
        make_session(rat_dir, name)
    make_session(tmp_path / "IM-1905", "20250801")
    (tmp_path / "analysis_scripts").mkdir()

    found = {d.name: (date, epoch) for d, date, epoch in find_session_dirs(rat_dir)}

    assert found == {"20250723": ("20250723", ""), "07242025": ("20250724", ""),
                     "2025-07-25": ("20250725", ""), "07-26-2025": ("20250726", ""),
                     "20250723_2": ("20250723", "2")}
    assert [d.name for d in find_rat_dirs(tmp_path)] == ["IM-1891_jimmy", "IM-1905"]


def test_photometry_data_needs_its_rat_level_block_filled_in(tmp_path):
    """
    The fibers and injections cannot be guessed, so a photometry session waits for them.

    Covers both an unfilled field and no photometry block at all, since a rat file written for a
    behavior-only rat cannot describe a photometry session either.
    """
    unfilled = {**PHOTOMETRY_RAT_METADATA,
                "photometry": {**PHOTOMETRY_RAT_METADATA["photometry"],
                               "virus_injections": [{"virus_name": FILL_IN}]}}
    no_block = {key: value for key, value in PHOTOMETRY_RAT_METADATA.items() if key != "photometry"}
    # Different dates, so the two rats' sessions do not collide in the report
    for rat_name, date, rat_metadata in [("IM-1891_jimmy", "20250723", unfilled),
                                         ("IM-1905", "20250724", no_block)]:
        rat_dir = tmp_path / rat_name
        write_rat_metadata(rat_dir, rat_metadata)
        make_session(rat_dir, date, BEHAVIOR_FILENAMES + ("recording.ppd",))

    reports = session_reports(generate_metadata(tmp_path, write=True))

    assert reports["20250723"][0] == "skipped" and "virus_name" in reports["20250723"][1]
    assert reports["20250724"][0] == "skipped" and "no 'photometry:' block" in reports["20250724"][1]
    assert generated_yamls(tmp_path) == []


def test_a_rat_with_an_existing_session_yaml_gets_its_rat_file_derived(tmp_path):
    """
    An existing session yaml becomes the rat file, and the next session is generated from it.

    Two runs, deliberately: the derived file inherits whatever was in the old one, mistakes
    included, so the first run stops to let the user read it.
    """
    rat_dir = tmp_path / "IM-1905"
    old_session = make_session(rat_dir, "20250801", BEHAVIOR_FILENAMES + ("recording.ppd",))
    (old_session / "metadata_IM-1905_20250801.yaml").write_text(yaml.safe_dump(
        {**PHOTOMETRY_RAT_METADATA, "date": "20250801", "animal_name": "IM-1905",
         "behavior": {"arduino_text_file_path": str(old_session / "arduinoraw0.txt")}}))
    make_session(rat_dir, "20250802", BEHAVIOR_FILENAMES + ("recording2.ppd",))

    assert generate_metadata(tmp_path, write=True)[0].status == "derived"
    assert generated_yamls(tmp_path) == [old_session / "metadata_IM-1905_20250801.yaml"]

    reports = session_reports(generate_metadata(tmp_path, write=True))
    assert reports["20250801"][0] == "exists"
    assert reports["20250802"][0] == "written"
    new_metadata = yaml.safe_load((rat_dir / "20250802" / "metadata_IM-1905_20250802.yaml").read_text())
    assert new_metadata["date"] == "20250802"
    assert new_metadata["photometry"]["ppd_file_path"].endswith("20250802/recording2.ppd")


def test_rat_and_output_dir_flags(tmp_path):
    """--rat limits a run to one animal, --output-dir keeps generated files off the data drive."""
    for rat_name in ("IM-1891_jimmy", "IM-1905"):
        rat_dir = tmp_path / "data" / rat_name
        write_rat_metadata(rat_dir, PHOTOMETRY_RAT_METADATA)
        make_session(rat_dir, "20250723", BEHAVIOR_FILENAMES)
    output_dir = tmp_path / "metadata"

    generate_metadata(tmp_path / "data", output_dir=output_dir, write=True, only_rats=["IM-1905"])

    assert [p.name for p in sorted(output_dir.iterdir())] == ["metadata_IM-1905_20250723.yaml"]
    assert list((tmp_path / "data").glob("*/*/metadata_*.yaml")) == []


def test_numbers_survive_the_round_trip_as_numbers():
    """
    A titer written as 2e+12 must not come back as the string "2e+12".

    Yaml only reads an exponent back as a number when the mantissa has a decimal point and the
    exponent is signed, so this is easy to get subtly wrong.
    """
    values = {"titer": 2e12, "other_titer": 1.15e13, "tiny": 1.15e-7,
              "volume": 1.0, "ml": -1.8, "zero": 0.0}

    loaded = yaml.safe_load(dump_session_metadata(values, "# test"))

    assert loaded == values
    assert all(isinstance(value, float) for value in loaded.values())
