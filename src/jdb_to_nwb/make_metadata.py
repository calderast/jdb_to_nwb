"""
Generate the per-session metadata yaml files for a directory of data folders.

Lab data is organized as <root>/<rat>/<date>/<that session's files>, for example
OptoRats/IM-1891_jimmy/20250723/arduinoraw0.txt. Every date folder needs its own metadata yaml
before it can be converted, and nearly everything in that yaml is identical across every session of
a rat. So this module splits the yaml in two:

  rat-level     subject, experimenter, surgery, virus, implant sites, probes. Written ONCE, by hand,
                into <root>/<rat>/rat_metadata.yaml next to the date folders.
  session-level the date, and the path to each of the session's data files. Discovered by looking in
                the date folder.

A rat's sessions are then generated from one reviewed file plus whatever is actually on disk.

Nothing is guessed. A section is written only once its files are found under the patterns below, so
a training day with no photometry gets no photometry section, and no path is ever written for a file
that is not there. Anything ambiguous - two files matching one field, a required file missing, two
Open Ephys recordings in one day - is reported and skipped instead of filled in with something
plausible. Nothing already on disk is overwritten.

Usage:
    jdb_to_nwb-metadata <root>            # report what is missing and what would be written
    jdb_to_nwb-metadata <root> --write    # actually write it
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .utils import to_datetime, VALID_MAZES

RAT_METADATA_FILENAME = "rat_metadata.yaml"

# Left in a generated rat_metadata.yaml wherever a human has to supply something. Generation refuses
# to run for a rat while one of these is still sitting in a required field, so a half-filled template
# cannot quietly become a pile of yamls full of placeholder text.
FILL_IN = "FILL IN"

# A date folder under a rat folder, optionally with the _2 / _3 suffix we use when a rat ran more
# than one session in a day. to_datetime accepts both YYYYMMDD and MMDDYYYY, so both are allowed
# here, as are the dashed forms.
SESSION_DIR_PATTERN = re.compile(r"^(?P<date>\d{8}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})(?:_(?P<epoch>\d+))?$")

# An Open Ephys recording folder is named for the time it started, e.g. "2026-09-04_14-20-43".
OPENEPHYS_DIR_GLOB = "20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]"

# Key order in the generated yaml, matching metadata_fully_explained.yaml so a generated file reads
# the same way as a hand-written one. Keys not listed here are dropped.
KEY_ORDER = ("institution", "lab", "experimenter", "experiment_description", "animal_name", "date",
             "maze", "subject", "ephys", "photometry", "behavior", "video", "surgery", "virus", "notes",
             "keywords")


@dataclass(frozen=True)
class FileField:
    """One path field in the metadata, and the filenames it is found under."""

    name: str
    # Tried in order, so an exact filename comes before the permissive pattern that covers the
    # sessions where it was named slightly differently.
    patterns: tuple[str, ...]
    required: bool = True
    # Hex centroids are reused across sessions when the maze has not moved, so they are often kept
    # once in the rat folder rather than copied into every date folder.
    search_rat_dir: bool = False
    is_dir: bool = False


BEHAVIOR_FILES = (
    FileField("arduino_text_file_path", ("arduinoraw0.txt", "arduinoraw*.txt")),
    FileField("arduino_timestamps_file_path", ("ArduinoStamps0.csv", "ArduinoStamps*.csv")),
    FileField("maze_configuration_file_path", ("barriers.txt",)),
)

VIDEO_FILES = (
    FileField("video_file_path", ("Behav_Vid0.avi", "Behav_Vid*.avi")),
    FileField("video_timestamps_file_path", ("testvidtimes0.csv", "testvidtimes*.csv")),
    # DLC is run after the session, so a recent session legitimately has no DLC output yet.
    FileField("dlc_path", ("Behav_Vid*DLC*.h5",), required=False),
    FileField("hex_centroids_file_path", ("*hex_coordinates*.csv", "*hex_centroids*.csv"),
              required=False, search_rat_dir=True),
)

# The three ways photometry reaches us, in the order they are preferred. Only one can be in a yaml,
# so the first whose files are all present wins. Raw LabVIEW beats signals.mat for the same reason
# the converter does: the .mat has already been through a preprocessing step we would rather redo.
PHOTOMETRY_VARIANTS = (
    ("pyPhotometry", (FileField("ppd_file_path", ("*.ppd",)),)),
    ("LabVIEW", (FileField("phot_file_path", ("*.phot",)), FileField("box_file_path", ("*.box",)))),
    ("LabVIEW signals.mat", (FileField("signals_mat_file_path", ("signals.mat",)),)),
)

EPHYS_FILES = (
    FileField("impedance_file_path", ("impedance.csv", "*mpedance*.csv"), required=False),
)

# Path fields are per-session, so they are stripped out when a rat_metadata.yaml is derived from an
# existing session yaml. Anything else in that yaml is kept.
SESSION_PATH_FIELDS = frozenset(
    [spec.name for spec in BEHAVIOR_FILES + VIDEO_FILES + EPHYS_FILES]
    + [spec.name for _, specs in PHOTOMETRY_VARIANTS for spec in specs]
    + ["openephys_folder_path", "openephys_folder_paths", "processed_ephys_path",
       "sorting_analyzer_path", "mountain_sort_output_file_path", "phot_end_time_mins",
       "barrier_shift_trial_counts"]
)

# Top-level keys that describe one session rather than the rat.
SESSION_ONLY_KEYS = frozenset(["date", "animal_name", "behavior", "video"])

# Required of every session yaml by convert.check_required_metadata. 'date' and 'animal_name' are
# filled in per session, so they are not asked of the rat file.
REQUIRED_SUBJECT_FIELDS = ("subject_id", "species", "genotype", "sex", "description")

RAT_METADATA_STUB = """\
# Rat-level metadata for {animal_name} - everything that is the same for all of this rat's sessions.
#
# jdb_to_nwb-metadata reads this file and writes one session yaml per date folder beside it, adding
# that session's date and the paths to its data files. So this is the only file to maintain by hand.
#
# Fill in every "{fill_in}" below, then DELETE the blocks for modalities this rat never had. A block
# is only used for a session whose data files are actually on disk, so leaving the photometry block
# in place does nothing to a training session that has no photometry recording.
#
# See metadata_fully_explained.yaml in the jdb_to_nwb repo for what each field means.

experimenter:
  - {fill_in}  # "Last, First", e.g. "Figueroa, Jose"

experiment_description: Hex maze task

# Which maze this rat was run on: "old_maze" or "new_maze". The two are physically different sizes,
# so this is what the pixels-per-cm scale is measured against, calculated from each session's own hex
# centroids. It also picks the photometry hardware for that room, so the LEDs and the photodetector
# do not have to be listed below.
maze: {fill_in}

# institution and lab default to UCSF / Berke Lab, so they are only needed if that is wrong.
# institution: University of California, San Francisco
# lab: Berke Lab

subject:
  subject_id: {subject_id}
  species: Rattus norvegicus
  sex: {fill_in}            # M or F
  date_of_birth: {fill_in}  # YYYY-MM-DD
  genotype: {fill_in}       # e.g. Wildtype, D1-Cre
  description: Long Evans Rat

# ---------------------------------------------------------------------------------------------
# PHOTOMETRY - delete this block if this rat was never recorded with photometry.
# Only the devices and the implant/injection sites go here. The recording file itself is found in
# each date folder, so a session with no .ppd / .phot gets no photometry section at all.
# Device names must match resources/photometry/photometry_devices.yaml exactly.
# ---------------------------------------------------------------------------------------------
photometry:
  # The LEDs and photodetector are taken from the "maze" field above. Only add
  # "excitation_sources" or "photodetector" here if this rat used a non-standard setup.
  optic_fiber_implant_sites:
    # One entry per implanted fiber. ONLY ONE may have "recording: true".
    # Negative ml_in_mm is the left hemisphere, positive is the right.
    - optic_fiber: {fill_in}
      targeted_location: {fill_in}
      ap_in_mm: {fill_in}
      ml_in_mm: {fill_in}
      dv_in_mm: {fill_in}
      recording: true
  virus_injections:
    # One entry per injection. Virus names must match resources/photometry/virus_info.yaml.
    - virus_name: {fill_in}
      targeted_location: {fill_in}
      ap_in_mm: {fill_in}
      ml_in_mm: {fill_in}
      dv_in_mm: {fill_in}
      volume_in_uL: {fill_in}
      titer_in_vg_per_mL: {fill_in}

# ---------------------------------------------------------------------------------------------
# EPHYS - delete this block if this rat was never implanted.
# Only the probes go here. The Open Ephys recording folder and the impedance file are found in each
# date folder, so a session recorded before the implant gets no ephys section.
# ---------------------------------------------------------------------------------------------
# ephys:
#   probes:
#     # One entry per implanted probe. probe_type must match resources/electrophysiology/ephys_devices.yaml.
#     # Negative targeted_x is the left hemisphere, positive is the right - Spyglass reads the
#     # hemisphere from its sign, so a wrong sign puts the probe in the wrong half of the brain.
#     - probe_type: "256-ch Silicon Probe, 3mm length, 66um pitch"
#       electrodes_location: "Hippocampus CA1"
#       targeted_x: 0.0   # ML in mm
#       targeted_y: 0.0   # AP in mm
#       targeted_z: 0.0   # DV in mm

# ---------------------------------------------------------------------------------------------
# The rest is free text for whoever reads this data later. Worth writing, not required.
# Give surgery DATES here - sessions both before and after a surgery share this one file, so the
# dates are how a reader tells which sessions had which implant.
# ---------------------------------------------------------------------------------------------
surgery:
  Describe each surgery and the date it happened.

virus:
  Describe the virus injected, its titer, volume, and the date.

notes:
  Anything else worth knowing about this rat.

keywords:
  - behavior
  - hex maze
"""


def load_yaml(yaml_path: Path) -> dict:
    """Read a yaml file into a dict."""
    with open(yaml_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def find_rat_dirs(root: Path) -> list[Path]:
    """Return the subdirectories of root that look like a rat folder, meaning they hold date folders."""
    return sorted(d for d in root.iterdir() if d.is_dir() and find_session_dirs(d))


def find_session_dirs(rat_dir: Path) -> list[tuple[Path, str, str]]:
    """
    Find the date folders in a rat folder.

    Returns a list of (folder, canonical YYYYMMDD date, epoch suffix), sorted by folder name. The
    epoch suffix is "" for the usual one-session day, or "2", "3", ... for a second or third session
    recorded the same day. A folder whose name parses as a date but is not a real one (say 20259999)
    is left out, so it shows up as unmatched rather than as a session with a nonsense date.
    """
    sessions = []
    for session_dir in sorted(p for p in rat_dir.iterdir() if p.is_dir()):
        match = SESSION_DIR_PATTERN.match(session_dir.name)
        if not match:
            continue
        try:
            session_date = to_datetime(match.group("date"))
        except ValueError:
            continue
        sessions.append((session_dir, session_date.strftime("%Y%m%d"), match.group("epoch") or ""))
    return sessions


def resolve_file_field(spec: FileField, session_dir: Path, rat_dir: Path) -> tuple[Path | None, str | None]:
    """
    Find the one file a FileField refers to.

    Returns (path, problem). Patterns are tried in order and the first that matches exactly one file
    wins. Matching more than one file is a problem rather than a match: there is no way to tell which
    was meant, and picking one would be exactly the silent nonsense we are trying to avoid.
    """
    search_dirs = [session_dir, rat_dir] if spec.search_rat_dir else [session_dir]
    for search_dir in search_dirs:
        for pattern in spec.patterns:
            matches = sorted(p for p in search_dir.glob(pattern) if p.is_dir() == spec.is_dir)
            if len(matches) == 1:
                return matches[0], None
            if len(matches) > 1:
                names = ", ".join(m.name for m in matches)
                return None, (f"{spec.name}: {len(matches)} files in {search_dir.name} match "
                              f"'{pattern}' ({names}). Not guessing which - set this one by hand.")
    return None, None


def discover_section(specs, session_dir: Path, rat_dir: Path) -> tuple[dict, list[str], list[str]]:
    """
    Resolve every FileField of one section against a date folder.

    Returns (fields found, problems, names of required fields that were not found).
    """
    found, problems, missing = {}, [], []
    for spec in specs:
        path, problem = resolve_file_field(spec, session_dir, rat_dir)
        if problem:
            problems.append(problem)
        if path is not None:
            found[spec.name] = path.as_posix()
        elif spec.required:
            missing.append(spec.name)
    return found, problems, missing


def discover_photometry(session_dir: Path, rat_dir: Path) -> tuple[dict, list[str], bool]:
    """
    Find this session's photometry recording, whichever of the three forms it was recorded in.

    Returns (fields found, problems, blocked). Finding nothing is the normal case for a training
    session, so it is not a problem on its own. Finding photometry that cannot be written - half of
    a LabVIEW pair, or two .ppd files - sets blocked, because the session has photometry data and a
    yaml without a photometry section would convert to an nwb that silently has none.
    """
    problems = []
    for variant_name, specs in PHOTOMETRY_VARIANTS:
        found, variant_problems, missing = discover_section(specs, session_dir, rat_dir)
        if variant_problems:
            return {}, variant_problems, True
        if found and missing:
            # A .phot with no .box, say. The data is there but unusable, so say so rather than
            # falling through to a form that happens to be complete.
            return {}, [f"found {variant_name} photometry but not {', '.join(missing)}"], True
        if found:
            return found, problems, False
    return {}, problems, False


def discover_ephys(session_dir: Path, rat_dir: Path) -> tuple[dict, list[str], bool]:
    """
    Find this session's Open Ephys recording and impedance file.

    Returns (fields found, problems, blocked). Finding nothing means no Open Ephys recording folder,
    so this session predates the implant (or was behavior only) and rightly gets no ephys section.
    More than one recording sets blocked instead: the labels they need cannot be guessed, and a yaml
    written without them would convert to an nwb with no ephys in it at all.
    """
    recording_dirs = sorted(p for p in session_dir.glob(OPENEPHYS_DIR_GLOB) if p.is_dir())
    if not recording_dirs:
        return {}, [], False
    if len(recording_dirs) > 1:
        # Every recording after the first needs a 'session' label, and that label has to be a real
        # name (it becomes the nwb epoch tag, and has to match the preprocessing's session_id when
        # there is preprocessed data). Only whoever ran the session knows which is which.
        names = ", ".join(p.name for p in recording_dirs)
        return {}, [f"{len(recording_dirs)} Open Ephys recordings in this day ({names}), so each "
                    "needs a 'session' label naming what it was. Write this one's ephys section by "
                    "hand - see metadata_example_Jifu.yaml."], True
    found, problems, _ = discover_section(EPHYS_FILES, session_dir, rat_dir)
    found["openephys_folder_paths"] = [{"path": recording_dirs[0].as_posix()}]
    return found, problems, False


def find_fill_ins(value, path: str = "") -> list[str]:
    """Return the dotted path to every FILL IN left in a metadata structure."""
    if isinstance(value, dict):
        return [p for key, sub in value.items() for p in find_fill_ins(sub, f"{path}.{key}" if path else str(key))]
    if isinstance(value, list):
        return [p for index, sub in enumerate(value) for p in find_fill_ins(sub, f"{path}[{index}]")]
    return [path] if isinstance(value, str) and FILL_IN in value else []


def check_rat_metadata(rat_metadata: dict) -> list[str]:
    """
    Return the reasons a rat_metadata.yaml cannot be used yet, or an empty list if it is ready.

    Only checks what every session needs regardless of modality. The photometry and ephys blocks are
    checked per session instead, because a rat with no photometry should not have to fill them in.
    """
    problems = []
    if not rat_metadata.get("experimenter"):
        problems.append("'experimenter' is missing")
    subject = rat_metadata.get("subject") or {}
    missing = [f for f in REQUIRED_SUBJECT_FIELDS if not subject.get(f)]
    if missing:
        problems.append(f"subject is missing {', '.join(missing)}")
    if not subject.get("date_of_birth") and not subject.get("age"):
        problems.append("subject needs either 'date_of_birth' (preferred) or 'age'")
    # Checked here rather than left to the conversion: a typo caught now costs a re-run of this
    # script, but one caught later means every session of this rat was generated with it.
    maze = rat_metadata.get("maze")
    if maze is not None and FILL_IN not in str(maze) and maze not in VALID_MAZES:
        problems.append(f"'maze' is {maze!r}, which is not one of {list(VALID_MAZES)}")
    required_blocks = {key: value for key, value in rat_metadata.items() if key not in ("photometry", "ephys")}
    for fill_in in find_fill_ins(required_blocks):
        problems.append(f"'{fill_in}' is still {FILL_IN}")
    return problems


SESSION_HEADER = """\
# Metadata for {animal_name}, {date}.
# Generated by jdb_to_nwb-metadata: the rat-level fields come from {rat_file}, the date and the file
# paths from {session_dir}. Edit it freely - an existing file is never regenerated or overwritten.
"""

DERIVED_RAT_HEADER = """\
# Rat-level metadata for {animal_name}, lifted out of {source}.
# Everything that varies by session - the date, and every file path - has been dropped, since
# jdb_to_nwb-metadata fills those in per session from the date folder's contents.
#
# CHECK THIS BEFORE GENERATING. It inherits whatever was in that file, mistakes included, and the
# comments that were in it are gone.
"""


def derive_rat_metadata(session_metadata: dict) -> dict:
    """
    Pull the rat-level fields out of an existing session yaml.

    This is the shortcut for a rat that already has a metadata file somewhere: rather than filling in
    a blank stub, the fields that are the same every session are carried over and only the paths and
    the date are dropped.
    """
    rat_metadata = {}
    for key, value in session_metadata.items():
        # Anything not recognized is carried over rather than dropped: it is more likely a field
        # this script has not heard of than a mistake, and losing it silently would be worse.
        if key in SESSION_ONLY_KEYS:
            continue
        if isinstance(value, dict):
            kept = {k: v for k, v in value.items() if k not in SESSION_PATH_FIELDS}
            if kept:
                rat_metadata[key] = kept
        else:
            rat_metadata[key] = value
    return rat_metadata


def find_existing_session_yaml(rat_dir: Path) -> Path | None:
    """Find a session metadata yaml already written for this rat, to derive its rat-level file from."""
    candidates = sorted(p for p in rat_dir.rglob("*metadata*.y*ml") if p.name != RAT_METADATA_FILENAME)
    for candidate in candidates:
        try:
            contents = load_yaml(candidate)
        except yaml.YAMLError:
            continue
        # A session yaml, as opposed to some other yaml that happens to be named metadata-something.
        if isinstance(contents, dict) and "subject" in contents and "date" in contents:
            return candidate
    return None


def session_yaml_name(animal_name: str, date: str) -> str:
    """Name a session's metadata file the way the ones in the repo are named."""
    return f"metadata_{animal_name}_{date}.yaml"


def existing_session_yamls(session_dir: Path) -> list[Path]:
    """Any metadata yaml already sitting in a date folder, whatever it is called."""
    return sorted(p for p in session_dir.glob("*metadata*.y*ml") if p.name != RAT_METADATA_FILENAME)


def build_session_metadata(rat_metadata: dict, animal_name: str, date: str, sections: dict) -> dict:
    """
    Assemble one session's metadata from the rat-level file and the sections found on disk.

    A modality's rat-level block (photometry devices, ephys probes) is only merged in when that
    session actually has that modality's data files.
    """
    metadata = {key: value for key, value in rat_metadata.items()
                if key not in SESSION_ONLY_KEYS and key not in ("photometry", "ephys")}
    metadata["animal_name"] = animal_name
    metadata["date"] = date
    for section_name, found in sections.items():
        if not found:
            continue
        rat_level = rat_metadata.get(section_name) or {}
        metadata[section_name] = {**rat_level, **found}
    # Known keys in the order metadata_fully_explained.yaml uses, then anything else the rat file
    # carried, so a field this script does not know about still reaches the conversion.
    ordered = {key: metadata[key] for key in KEY_ORDER if key in metadata}
    ordered.update({key: value for key, value in metadata.items() if key not in ordered})
    return ordered


class MetadataDumper(yaml.SafeDumper):
    """
    A SafeDumper that writes metadata the way we write it by hand.

    Lists are indented under their key rather than sitting at the parent's indentation, and a big
    number keeps its exponent, so a titer stays "2e+12" instead of becoming "2000000000000.0".
    """

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, indentless=False)


def represent_float(dumper: yaml.SafeDumper, value: float):
    """Write a very large or very small number in exponent form, the way a titer is written."""
    if not value or 1e-3 <= abs(value) < 1e5:
        return dumper.represent_scalar("tag:yaml.org,2002:float", repr(value))
    # Yaml only reads an exponent back as a number when the mantissa has a decimal point and the
    # exponent is signed. Without both, the emitter gives up and writes !!float '2e+12' instead.
    mantissa, _, exponent = f"{value:.6g}".partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{mantissa}e{exponent}")


MetadataDumper.add_representer(float, represent_float)


def dump_session_metadata(metadata: dict, header: str) -> str:
    """Render a session's metadata as yaml text, with a header saying where it came from."""
    body = yaml.dump(metadata, Dumper=MetadataDumper, sort_keys=False, default_flow_style=False,
                     allow_unicode=True, width=110)
    return f"{header}\n{body}"


@dataclass
class SessionReport:
    """What happened to one date folder."""

    session_dir: Path
    status: str  # "written", "exists", or "skipped"
    message: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class RatReport:
    """What happened to one rat folder."""

    rat_dir: Path
    status: str  # "generated", "stub written", "derived", or "skipped"
    message: str = ""
    sessions: list[SessionReport] = field(default_factory=list)


def process_session(rat_metadata: dict, rat_dir: Path, animal_name: str, session_dir: Path,
                    date: str, epoch: str, output_dir: Path | None, write: bool) -> SessionReport:
    """
    Write one session's metadata yaml, or explain why it was not written.

    A session is skipped whole rather than written incomplete. Writing a yaml with the photometry
    section quietly left out would convert to an nwb that looks finished and silently has no
    photometry in it, which is worse than not converting at all.
    """
    # A second or third session in one day needs its own animal_name, because nwb files are named
    # animal_name + date and would otherwise collide.
    session_animal_name = f"{animal_name}_{epoch}" if epoch else animal_name
    destination = (output_dir or session_dir) / session_yaml_name(session_animal_name, date)

    already_there = existing_session_yamls(session_dir) + ([destination] if destination.exists() else [])
    if already_there:
        return SessionReport(session_dir, "exists", already_there[0].name)

    behavior, behavior_problems, behavior_missing = discover_section(BEHAVIOR_FILES, session_dir, rat_dir)
    if behavior_missing or behavior_problems:
        reason = behavior_problems or [f"no {', '.join(behavior_missing)}"]
        return SessionReport(session_dir, "skipped", f"behavior: {'; '.join(reason)}")

    video, video_problems, video_missing = discover_section(VIDEO_FILES, session_dir, rat_dir)
    photometry, photometry_problems, photometry_blocked = discover_photometry(session_dir, rat_dir)
    ephys, ephys_problems, ephys_blocked = discover_ephys(session_dir, rat_dir)

    # This session has data for a modality that cannot be written down automatically. Writing the
    # yaml without that section would be worse than not writing it: the file would never be
    # regenerated, and it would convert to an nwb that looks finished and silently has no ephys or
    # no photometry in it.
    if photometry_blocked or ephys_blocked:
        return SessionReport(session_dir, "skipped", "; ".join(photometry_problems + ephys_problems))

    notes = video_problems
    if video_missing:
        video = {}
        notes.append(f"no video section: missing {', '.join(video_missing)}")
    elif "dlc_path" not in video:
        notes.append("no DLC output yet, so position will not be converted")

    # A modality's data is here, so its rat-level block has to be filled in before this session can
    # be written. These are only checked now because a rat with no photometry never needs them.
    if photometry:
        photometry_block = rat_metadata.get("photometry")
        if not photometry_block:
            return SessionReport(session_dir, "skipped",
                                 f"has photometry data, but {RAT_METADATA_FILENAME} has no "
                                 "'photometry:' block giving the devices and the implant sites")
        fill_ins = find_fill_ins(photometry_block)
        if fill_ins:
            return SessionReport(session_dir, "skipped",
                                 f"has photometry data, but {RAT_METADATA_FILENAME} still needs "
                                 + ", ".join(f"photometry.{name}" for name in fill_ins))
    if ephys and not (rat_metadata.get("ephys") or {}).get("probes"):
        return SessionReport(session_dir, "skipped",
                             f"has an Open Ephys recording, but {RAT_METADATA_FILENAME} has no "
                             "'ephys: probes:' block saying which probes are implanted")

    sections = {"behavior": behavior, "video": video, "photometry": photometry, "ephys": ephys}
    metadata = build_session_metadata(rat_metadata, session_animal_name, date, sections)
    header = SESSION_HEADER.format(animal_name=session_animal_name, date=date,
                                   rat_file=RAT_METADATA_FILENAME, session_dir=session_dir.name)
    if write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(dump_session_metadata(metadata, header))

    present = [name for name, found in sections.items() if found]
    return SessionReport(session_dir, "written", f"{destination.name} ({' + '.join(present)})", notes)


def process_rat(rat_dir: Path, output_dir: Path | None, write: bool) -> RatReport:
    """
    Generate the missing session yamls for one rat folder.

    A rat with no rat_metadata.yaml gets one started instead: derived from a session yaml it already
    has, or as a blank stub. Either way nothing is generated this run, because the file wants reading
    first - deriving carries over whatever was in the old file, mistakes included.
    """
    rat_metadata_path = rat_dir / RAT_METADATA_FILENAME

    if not rat_metadata_path.exists():
        source = find_existing_session_yaml(rat_dir)
        if source is not None:
            derived = derive_rat_metadata(load_yaml(source))
            header = DERIVED_RAT_HEADER.format(animal_name=rat_dir.name,
                                               source=source.relative_to(rat_dir).as_posix())
            if write:
                rat_metadata_path.write_text(dump_session_metadata(derived, header))
            return RatReport(rat_dir, "derived",
                             f"{RAT_METADATA_FILENAME} from {source.name} - read it, then re-run")
        # The folder name is the animal_name we use for nwb files, and the subject_id is usually its
        # IM-#### part, so both are filled in rather than left as another thing to type.
        subject_match = re.match(r"^(IM-\d+)", rat_dir.name)
        stub = RAT_METADATA_STUB.format(animal_name=rat_dir.name, fill_in=FILL_IN,
                                        subject_id=subject_match.group(1) if subject_match else FILL_IN)
        if write:
            rat_metadata_path.write_text(stub)
        return RatReport(rat_dir, "stub written",
                         f"{RAT_METADATA_FILENAME} - fill in its {FILL_IN} fields, then re-run")

    rat_metadata = load_yaml(rat_metadata_path)
    problems = check_rat_metadata(rat_metadata)
    if problems:
        return RatReport(rat_dir, "skipped", f"{RAT_METADATA_FILENAME}: {'; '.join(problems)}")

    animal_name = rat_metadata.get("animal_name") or rat_dir.name
    sessions = [process_session(rat_metadata, rat_dir, animal_name, session_dir, date, epoch, output_dir, write)
                for session_dir, date, epoch in find_session_dirs(rat_dir)]
    return RatReport(rat_dir, "generated", sessions=sessions)


def generate_metadata(root: Path, output_dir: Path | None = None, write: bool = False,
                      only_rats: list[str] | None = None) -> list[RatReport]:
    """
    Walk a <root>/<rat>/<date>/ data directory and generate every missing session metadata yaml.

    With write=False (the default) nothing is written and the reports say what would have been.
    """
    # The paths written into the yaml come from these, and a session yaml has to work from whatever
    # directory the conversion is run in, so resolve the root rather than copying a relative path in.
    rat_dirs = find_rat_dirs(Path(root).resolve())
    if only_rats:
        wanted = set(only_rats)
        rat_dirs = [d for d in rat_dirs if d.name in wanted]
    return [process_rat(rat_dir, output_dir, write) for rat_dir in rat_dirs]


def print_report(reports: list[RatReport], write: bool) -> None:
    """Print what was done, or what would be done, one rat at a time."""
    if not reports:
        print("No rat folders found. Expected <root>/<rat>/<date>/, e.g. OptoRats/IM-1891_jimmy/20250723/")
        return

    verb = "wrote" if write else "would write"
    written = skipped = exists = rat_files = 0
    for report in reports:
        print(f"\n{report.rat_dir.name}")
        if report.status == "skipped":
            skipped += len(find_session_dirs(report.rat_dir))
            print(f"  skipped: {report.message}")
            continue
        if report.status != "generated":
            rat_files += 1
            skipped += len(find_session_dirs(report.rat_dir))
            print(f"  {verb} {report.message}")
            continue
        for session in report.sessions:
            if session.status == "written":
                written += 1
                print(f"  {session.session_dir.name}  {verb} {session.message}")
            elif session.status == "exists":
                exists += 1
                print(f"  {session.session_dir.name}  has {session.message}, left alone")
            else:
                skipped += 1
                print(f"  {session.session_dir.name}  SKIPPED - {session.message}")
            for note in session.notes:
                print(f"      note: {note}")

    summary = [f"{rat_files} rat file{'s' if rat_files != 1 else ''}"] if rat_files else []
    summary.append(f"{written} session yaml{'s' if written != 1 else ''}")
    print(f"\n{verb.capitalize()} {' and '.join(summary)}. "
          f"{exists} already there, {skipped} not generated.")
    if not write:
        print("Nothing was written. Re-run with --write once the above looks right.")


def cli():
    parser = argparse.ArgumentParser(
        # Pinned so the usage line is right whether this was run as the console script or as
        # python -m jdb_to_nwb.make_metadata.
        prog="jdb_to_nwb-metadata",
        description="Generate the missing per-session metadata yamls for a <root>/<rat>/<date>/ data directory.",
        epilog="Rat-level metadata is read from rat_metadata.yaml in each rat folder, and one is "
               "started for you if it is not there yet.")
    parser.add_argument("root", type=Path, help="Directory holding one folder per rat, e.g. OptoRats/")
    parser.add_argument("--write", action="store_true",
                        help="Actually write the files. Without this, only reports what would be written.")
    parser.add_argument("--rat", action="append", dest="rats", metavar="NAME",
                        help="Only this rat folder. Repeatable.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write the session yamls here instead of into each date folder.")
    args = parser.parse_args()

    reports = generate_metadata(args.root, output_dir=args.output_dir, write=args.write, only_rats=args.rats)
    print_report(reports, args.write)


if __name__ == "__main__":
    # So this runs as `python -m jdb_to_nwb.make_metadata` too, without reinstalling the package to
    # pick up the jdb_to_nwb-metadata console script.
    cli()
