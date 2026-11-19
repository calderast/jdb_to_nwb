# Generating metadata files for a whole data directory

Every session you convert needs its own metadata yaml, and almost all of it is the same for every
session of a rat. `jdb_to_nwb-metadata` splits that in two:

| | what it holds | where it lives | who writes it |
|---|---|---|---|
| **rat-level** | subject, experimenter, maze, surgery, virus, implant sites, probes | `rat_metadata.yaml` in the rat folder | you, once per rat |
| **session-level** | the `date`, and the path to every data file | `metadata_<animal>_<date>.yaml` in the date folder | generated |

You maintain one file per rat. Everything else is found by looking in the date folders.

**Nothing is guessed.** A section appears only when its data files are actually on disk, and no path
is ever written for a file that is not there. Anything ambiguous is reported and skipped rather than
filled in with something plausible. Nothing already on disk is overwritten, ever.

- [Quick start](#quick-start)
- [The directory layout it expects](#the-directory-layout-it-expects)
- [Which files it looks for](#which-files-it-looks-for)
- [Worked example](#worked-example)
- [When a session is skipped](#when-a-session-is-skipped)
- [Flags](#flags)
- [If it does not find your files](#if-it-does-not-find-your-files)

## Quick start

```bash
pip install -e .                                  # once, to get the command

jdb_to_nwb-metadata /path/to/OptoRats             # 1. look. writes nothing
jdb_to_nwb-metadata /path/to/OptoRats --write     # 2. creates a rat_metadata.yaml per rat
                                                  # 3. open each one and fill it in
jdb_to_nwb-metadata /path/to/OptoRats             # 4. look again
jdb_to_nwb-metadata /path/to/OptoRats --write     # 5. writes the session yamls
```

Without `--write` it only reports. Run it again whenever new sessions come in: existing files are
left alone, so it only ever fills gaps.

If you would rather not reinstall, `python -m jdb_to_nwb.make_metadata` works identically everywhere
below.

## The directory layout it expects

```
<root>/<rat>/<date>/<that session's files>
```

**Rat folders** are any folder in `<root>` that contains at least one date folder. A folder with no
date folders in it (`analysis_scripts/`, `histology/`) is not mistaken for a rat. The folder name
becomes `animal_name`, and its leading `IM-####` becomes the default `subject_id`.

**Date folders** are named in any format the conversion already accepts:

| format | example |
|---|---|
| `YYYYMMDD` | `20250723` |
| `MMDDYYYY` | `07242025` |
| `YYYY-MM-DD` | `2025-07-25` |
| `MM-DD-YYYY` | `07-26-2025` |

Optionally with a `_2`, `_3` suffix for a second or third session recorded the same day. Anything
else in a rat folder is ignored, including a folder that looks like a date but is not a real one
(`20259999`).

A same-day session gets its own `animal_name`, because nwb files are auto-named `animal_name_date`
and would otherwise collide:

| folder | `animal_name` | generated file |
|---|---|---|
| `20250724` | `IM-1891_jimmy` | `metadata_IM-1891_jimmy_20250724.yaml` |
| `20250724_2` | `IM-1891_jimmy_2` | `metadata_IM-1891_jimmy_2_20250724.yaml` |

## Which files it looks for

This is the complete list. Patterns are tried **in order**, and the first pattern matching exactly
one file wins — so the exact filename we normally use is preferred, and the looser pattern is the
fallback for sessions that were named slightly differently.

If a pattern matches **more than one** file, that is not a match: there is no way to tell which was
meant, so it is reported and the session is skipped.

### behavior — required, every session

| metadata field | looks for |
|---|---|
| `arduino_text_file_path` | `arduinoraw0.txt`, then `arduinoraw*.txt` |
| `arduino_timestamps_file_path` | `ArduinoStamps0.csv`, then `ArduinoStamps*.csv` |
| `maze_configuration_file_path` | `barriers.txt` |

All three must be there or the session is skipped — `barriers.txt` is the one you write by hand, so
a day you have not got to yet is skipped until you do.

### video — written when both required files are there

| metadata field | looks for | |
|---|---|---|
| `video_file_path` | `Behav_Vid0.avi`, then `Behav_Vid*.avi` | required |
| `video_timestamps_file_path` | `testvidtimes0.csv`, then `testvidtimes*.csv` | required |
| `dlc_path` | `Behav_Vid*DLC*.h5` | optional |
| `hex_centroids_file_path` | `*hex_coordinates*.csv`, then `*hex_centroids*.csv` | optional |

`dlc_path` is optional because DLC is run after the session — a recent session legitimately has none
yet, and you get a note saying position will not be converted.

`hex_centroids_file_path` is looked for **in the date folder first, then in the rat folder**, since
the maze often does not move between sessions and one centroids file is kept for the whole rat.

### photometry — written when a recording is found

Three forms, tried in this order. The first whose files are all present wins.

| form | metadata field(s) | looks for |
|---|---|---|
| pyPhotometry | `ppd_file_path` | `*.ppd` |
| LabVIEW | `phot_file_path` + `box_file_path` | `*.phot` and `*.box` |
| LabVIEW (legacy) | `signals_mat_file_path` | `signals.mat` |

Finding no photometry at all is normal — that is a training session. Finding *half* a recording (a
`.phot` with no `.box`) skips the session; see [below](#when-a-session-is-skipped).

### ephys — written when an Open Ephys recording folder is found

| metadata field | looks for | |
|---|---|---|
| `openephys_folder_paths` | a **folder** named like `2026-10-21_14-20-43` | required |
| `impedance_file_path` | `impedance.csv`, then `*mpedance*.csv` | optional |

The loose impedance pattern deliberately has no leading `i`, so it also catches
`20250718 Impedance pre drive.csv`.

A day with **more than one** Open Ephys folder is skipped — each recording after the first needs a
`session` label, and only you know what each one was.

## Worked example

Two rats. One is a photometry rat with a training day from before its surgery; the other is an ephys
rat whose second day has not been through DLC yet.

### The data on disk

```
OptoRats/
├── IM-1891_jimmy/
│   ├── IM-1891_jimmy_hex_coordinates.csv         <- one centroids file for the whole rat
│   ├── 20250418/                                 <- training day, before the surgery
│   │   ├── arduinoraw0.txt
│   │   ├── ArduinoStamps0.csv
│   │   ├── barriers.txt
│   │   ├── Behav_Vid0.avi
│   │   ├── testvidtimes0.csv
│   │   └── Behav_Vid0DLC_resnet50_Triangle_Maze_PhotMar24shuffle1_800000.h5
│   └── 20250723/                                 <- recording day, after the surgery
│       ├── arduinoraw0.txt
│       ├── ArduinoStamps0.csv
│       ├── barriers.txt
│       ├── Behav_Vid0.avi
│       ├── testvidtimes0.csv
│       ├── Behav_Vid0DLC_resnet50_Triangle_Maze_PhotMar24shuffle1_800000.h5
│       └── RecordingDay17_ProbSwitch_CKTL-2025-07-23-123503.ppd
└── IM-1982/
    ├── 20261021/
    │   ├── arduinoraw0.txt
    │   ├── ArduinoStamps0.csv
    │   ├── barriers.txt
    │   ├── Behav_Vid0.avi
    │   ├── testvidtimes0.csv
    │   ├── Behav_Vid0DLC_resnet50_Triangle_Maze_PhotMar24shuffle1_800000.h5
    │   ├── impedance.csv
    │   └── 2026-10-21_14-20-43/                  <- Open Ephys recording folder
    └── 20261022/
        ├── arduinoraw0.txt
        ├── ArduinoStamps0.csv
        ├── barriers.txt
        ├── Behav_Vid0.avi
        ├── testvidtimes0.csv                     <- note: no DLC output yet
        ├── impedance.csv
        └── 2026-10-22_11-02-15/
```

### Step 1 — the first run

```
$ jdb_to_nwb-metadata OptoRats

IM-1891_jimmy
  would write rat_metadata.yaml - fill in its FILL IN fields, then re-run

IM-1982
  would write rat_metadata.yaml - fill in its FILL IN fields, then re-run

Would write 2 rat files and 0 session yamls. 0 already there, 4 not generated.
Nothing was written. Re-run with --write once the above looks right.
```

Run it with `--write` and each rat gets a `rat_metadata.yaml` template. No sessions are generated
yet — generating them would mean inventing a subject.

> If a rat **already has a session metadata yaml** somewhere, its `rat_metadata.yaml` is lifted out
> of that file instead of being a blank template, and the run reports `derived` rather than
> `stub written`. Read it before generating: it inherits whatever was in the old file, mistakes
> included, and the comments in that file are gone.

### Step 2 — fill in the rat files

`OptoRats/IM-1891_jimmy/rat_metadata.yaml`, with the ephys block deleted:

```yaml
# Rat-level metadata for IM-1891_jimmy.
# The EPHYS block has been deleted: this rat was never implanted.

experimenter:
  - Figueroa, Jose

experiment_description: Hex maze task

maze: new_maze

subject:
  subject_id: IM-1891
  species: Rattus norvegicus
  sex: M
  date_of_birth: 2024-10-17
  genotype: Wildtype
  description: Long Evans Rat

photometry:
  # No excitation_sources or photodetector here: they come from "maze" above.
  optic_fiber_implant_sites:
    - optic_fiber: Doric 0.66mm Flat 40mm Optic Fiber
      targeted_location: NAcc
      ap_in_mm: 1.8
      ml_in_mm: -1.8
      dv_in_mm: -6.0
      recording: true
  virus_injections:
    - virus_name: dLight1.3b
      targeted_location: NAcc
      ap_in_mm: 1.8
      ml_in_mm: -1.8
      dv_in_mm: -6.2
      volume_in_uL: 1.0
      titer_in_vg_per_mL: 2.0e+12

surgery:
  dLight1.3b injected in NAcc and fiber implanted on 2025-05-03. No complications.

virus:
  dLight1.3b (BrainVTA), 1.0 uL at 2e12 vg/mL, injected into left NAcc on 2025-05-03.

notes:
  Reward was 30uL of 10% sucrose at each port. Water deprived before the session.

keywords:
  - photometry
  - dopamine
  - behavior
  - hex maze
```

`OptoRats/IM-1982/rat_metadata.yaml`, with the photometry block deleted:

```yaml
# Rat-level metadata for IM-1982.
# The PHOTOMETRY block has been deleted: this rat is ephys only.

experimenter:
  - Crater, Stephanie

experiment_description: Hex maze task with ephys

maze: old_maze

subject:
  subject_id: IM-1982
  species: Rattus norvegicus
  sex: F
  date_of_birth: 2025-06-01
  genotype: Wildtype
  description: Long Evans Rat

ephys:
  probes:
    - probe_type: "256-ch Silicon Probe, 3mm length, 66um pitch"
      electrodes_location: "Hippocampus CA1"
      targeted_x: -2.2
      targeted_y: 4.5
      targeted_z: -2.0

surgery:
  256-channel silicon probe implanted in left dorsal CA1 on 2026-09-15.

keywords:
  - ephys
  - behavior
  - hex maze
```

### What `maze` does

`maze` is the one field above that does real work beyond being copied through. It is either
`old_maze` or `new_maze`, and because the two mazes are physically different sizes it is what the
conversion scales against when it measures `pixels_per_cm` from that session's own hex centroids —
so position data is converted correctly even if the camera was moved, raised or zoomed between
sessions. It also selects that room's photometry hardware, which is why the photometry rat's block
above lists no LEDs or photodetector: those come from `maze: new_maze`.

A rat is normally run on one maze, which is why this is a rat-level field. A typo is caught before
anything is generated, rather than being copied into every session.

Note that **neither rat file mentions a surgery date in any field the script reads**. Which sections
a session gets is decided purely by the files in its folder, so the training day below comes out
behavior-only without anybody telling the script when the surgery was. Surgery dates belong in
`surgery:` as prose, for whoever reads the data later.

### Step 3 — generate

```
$ jdb_to_nwb-metadata OptoRats --write

IM-1891_jimmy
  20250418  wrote metadata_IM-1891_jimmy_20250418.yaml (behavior + video)
  20250723  wrote metadata_IM-1891_jimmy_20250723.yaml (behavior + video + photometry)

IM-1982
  20261021  wrote metadata_IM-1982_20261021.yaml (behavior + video + ephys)
  20261022  wrote metadata_IM-1982_20261022.yaml (behavior + video + ephys)
      note: no DLC output yet, so position will not be converted

Wrote 4 session yamls. 0 already there, 0 not generated.
```

### The four generated files

Paths are absolute, so a session yaml works from whatever directory you run the conversion in. They
are shown here rooted at `/data`.

**`OptoRats/IM-1891_jimmy/20250418/metadata_IM-1891_jimmy_20250418.yaml`** — the training day. No
photometry section, because there is no `.ppd` in that folder:

```yaml
# Metadata for IM-1891_jimmy, 20250418.
# Generated by jdb_to_nwb-metadata: the rat-level fields come from rat_metadata.yaml, the date and the file
# paths from 20250418. Edit it freely - an existing file is never regenerated or overwritten.

experimenter:
  - Figueroa, Jose
experiment_description: Hex maze task
animal_name: IM-1891_jimmy
date: '20250418'
maze: new_maze
subject:
  subject_id: IM-1891
  species: Rattus norvegicus
  sex: M
  date_of_birth: 2024-10-17
  genotype: Wildtype
  description: Long Evans Rat
behavior:
  arduino_text_file_path: /data/OptoRats/IM-1891_jimmy/20250418/arduinoraw0.txt
  arduino_timestamps_file_path: /data/OptoRats/IM-1891_jimmy/20250418/ArduinoStamps0.csv
  maze_configuration_file_path: /data/OptoRats/IM-1891_jimmy/20250418/barriers.txt
video:
  video_file_path: /data/OptoRats/IM-1891_jimmy/20250418/Behav_Vid0.avi
  video_timestamps_file_path: /data/OptoRats/IM-1891_jimmy/20250418/testvidtimes0.csv
  dlc_path: /data/OptoRats/IM-1891_jimmy/20250418/Behav_Vid0DLC_resnet50_Triangle_Maze_PhotMar24shuffle1_800000.h5
  hex_centroids_file_path: /data/OptoRats/IM-1891_jimmy/IM-1891_jimmy_hex_coordinates.csv
surgery: dLight1.3b injected in NAcc and fiber implanted on 2025-05-03. No complications.
virus: dLight1.3b (BrainVTA), 1.0 uL at 2e12 vg/mL, injected into left NAcc on 2025-05-03.
notes: Reward was 30uL of 10% sucrose at each port. Water deprived before the session.
keywords:
  - photometry
  - dopamine
  - behavior
  - hex maze
```

`hex_centroids_file_path` points up into the rat folder, since that is where the file is.

**`OptoRats/IM-1891_jimmy/20250723/metadata_IM-1891_jimmy_20250723.yaml`** — same rat file, but this
folder has a `.ppd`, so the whole photometry block comes through with the recording path appended:

```yaml
# Metadata for IM-1891_jimmy, 20250723.
# Generated by jdb_to_nwb-metadata: the rat-level fields come from rat_metadata.yaml, the date and the file
# paths from 20250723. Edit it freely - an existing file is never regenerated or overwritten.

experimenter:
  - Figueroa, Jose
experiment_description: Hex maze task
animal_name: IM-1891_jimmy
date: '20250723'
maze: new_maze
subject:
  subject_id: IM-1891
  species: Rattus norvegicus
  sex: M
  date_of_birth: 2024-10-17
  genotype: Wildtype
  description: Long Evans Rat
photometry:
  optic_fiber_implant_sites:
    - optic_fiber: Doric 0.66mm Flat 40mm Optic Fiber
      targeted_location: NAcc
      ap_in_mm: 1.8
      ml_in_mm: -1.8
      dv_in_mm: -6.0
      recording: true
  virus_injections:
    - virus_name: dLight1.3b
      targeted_location: NAcc
      ap_in_mm: 1.8
      ml_in_mm: -1.8
      dv_in_mm: -6.2
      volume_in_uL: 1.0
      titer_in_vg_per_mL: 2.0e+12
  ppd_file_path: /data/OptoRats/IM-1891_jimmy/20250723/RecordingDay17_ProbSwitch_CKTL-2025-07-23-123503.ppd
behavior:
  arduino_text_file_path: /data/OptoRats/IM-1891_jimmy/20250723/arduinoraw0.txt
  arduino_timestamps_file_path: /data/OptoRats/IM-1891_jimmy/20250723/ArduinoStamps0.csv
  maze_configuration_file_path: /data/OptoRats/IM-1891_jimmy/20250723/barriers.txt
video:
  video_file_path: /data/OptoRats/IM-1891_jimmy/20250723/Behav_Vid0.avi
  video_timestamps_file_path: /data/OptoRats/IM-1891_jimmy/20250723/testvidtimes0.csv
  dlc_path: /data/OptoRats/IM-1891_jimmy/20250723/Behav_Vid0DLC_resnet50_Triangle_Maze_PhotMar24shuffle1_800000.h5
  hex_centroids_file_path: /data/OptoRats/IM-1891_jimmy/IM-1891_jimmy_hex_coordinates.csv
surgery: dLight1.3b injected in NAcc and fiber implanted on 2025-05-03. No complications.
virus: dLight1.3b (BrainVTA), 1.0 uL at 2e12 vg/mL, injected into left NAcc on 2025-05-03.
notes: Reward was 30uL of 10% sucrose at each port. Water deprived before the session.
keywords:
  - photometry
  - dopamine
  - behavior
  - hex maze
```

**`OptoRats/IM-1982/20261021/metadata_IM-1982_20261021.yaml`** — the ephys rat. The `probes` list
comes from the rat file; the recording folder and impedance file come from the date folder:

```yaml
# Metadata for IM-1982, 20261021.
# Generated by jdb_to_nwb-metadata: the rat-level fields come from rat_metadata.yaml, the date and the file
# paths from 20261021. Edit it freely - an existing file is never regenerated or overwritten.

experimenter:
  - Crater, Stephanie
experiment_description: Hex maze task with ephys
animal_name: IM-1982
date: '20261021'
maze: old_maze
subject:
  subject_id: IM-1982
  species: Rattus norvegicus
  sex: F
  date_of_birth: 2025-06-01
  genotype: Wildtype
  description: Long Evans Rat
ephys:
  probes:
    - probe_type: 256-ch Silicon Probe, 3mm length, 66um pitch
      electrodes_location: Hippocampus CA1
      targeted_x: -2.2
      targeted_y: 4.5
      targeted_z: -2.0
  impedance_file_path: /data/OptoRats/IM-1982/20261021/impedance.csv
  openephys_folder_paths:
    - path: /data/OptoRats/IM-1982/20261021/2026-10-21_14-20-43
behavior:
  arduino_text_file_path: /data/OptoRats/IM-1982/20261021/arduinoraw0.txt
  arduino_timestamps_file_path: /data/OptoRats/IM-1982/20261021/ArduinoStamps0.csv
  maze_configuration_file_path: /data/OptoRats/IM-1982/20261021/barriers.txt
video:
  video_file_path: /data/OptoRats/IM-1982/20261021/Behav_Vid0.avi
  video_timestamps_file_path: /data/OptoRats/IM-1982/20261021/testvidtimes0.csv
  dlc_path: /data/OptoRats/IM-1982/20261021/Behav_Vid0DLC_resnet50_Triangle_Maze_PhotMar24shuffle1_800000.h5
surgery: 256-channel silicon probe implanted in left dorsal CA1 on 2026-09-15.
keywords:
  - ephys
  - behavior
  - hex maze
```

`openephys_folder_paths` is written as a one-entry list with no `session` label, which is all a
single-recording day needs. A day with two recordings needs a label on each and is skipped instead.

**`OptoRats/IM-1982/20261022/metadata_IM-1982_20261022.yaml`** — identical in shape, but with
**no `dlc_path`**, because DLC has not been run on that day's video yet. That is the one the run
printed a note about. When you do run DLC, re-running the script will not pick it up: this file
already exists, so it is left alone. Add the `dlc_path` line to it by hand.

```yaml
# Metadata for IM-1982, 20261022.
# Generated by jdb_to_nwb-metadata: the rat-level fields come from rat_metadata.yaml, the date and the file
# paths from 20261022. Edit it freely - an existing file is never regenerated or overwritten.

experimenter:
  - Crater, Stephanie
experiment_description: Hex maze task with ephys
animal_name: IM-1982
date: '20261022'
maze: old_maze
subject:
  subject_id: IM-1982
  species: Rattus norvegicus
  sex: F
  date_of_birth: 2025-06-01
  genotype: Wildtype
  description: Long Evans Rat
ephys:
  probes:
    - probe_type: 256-ch Silicon Probe, 3mm length, 66um pitch
      electrodes_location: Hippocampus CA1
      targeted_x: -2.2
      targeted_y: 4.5
      targeted_z: -2.0
  impedance_file_path: /data/OptoRats/IM-1982/20261022/impedance.csv
  openephys_folder_paths:
    - path: /data/OptoRats/IM-1982/20261022/2026-10-22_11-02-15
behavior:
  arduino_text_file_path: /data/OptoRats/IM-1982/20261022/arduinoraw0.txt
  arduino_timestamps_file_path: /data/OptoRats/IM-1982/20261022/ArduinoStamps0.csv
  maze_configuration_file_path: /data/OptoRats/IM-1982/20261022/barriers.txt
video:
  video_file_path: /data/OptoRats/IM-1982/20261022/Behav_Vid0.avi
  video_timestamps_file_path: /data/OptoRats/IM-1982/20261022/testvidtimes0.csv
surgery: 256-channel silicon probe implanted in left dorsal CA1 on 2026-09-15.
keywords:
  - ephys
  - behavior
  - hex maze
```

### Then convert

```bash
jdb_to_nwb OptoRats/IM-1891_jimmy/20250723/metadata_IM-1891_jimmy_20250723.yaml output_dir
```

## When a session is skipped

A session is skipped **whole**, never written with a section quietly left out. A file written
without its photometry section would never be regenerated — it exists, so the next run leaves it
alone — and it would convert to an nwb that looks finished and silently has no photometry in it.
Skipping is louder and recoverable.

| reason | what to do |
|---|---|
| `behavior: no maze_configuration_file_path` | write that session's `barriers.txt` |
| `2 files in <folder> match '*.ppd' (morning.ppd, afternoon.ppd)` | only you know which recording this session is. Write the yaml by hand, or move the other file |
| `found LabVIEW photometry but not box_file_path` | the `.box` is missing. Find it, or move the `.phot` out |
| `2 Open Ephys recordings in this day (...), so each needs a 'session' label` | write that day's `ephys` section by hand — see `metadata_example_Jifu.yaml` |
| `has photometry data, but rat_metadata.yaml still needs photometry.virus_injections[0].virus_name` | finish filling in the rat file's photometry block |
| `has an Open Ephys recording, but rat_metadata.yaml has no 'ephys: probes:' block` | add the probes to the rat file |
| `rat_metadata.yaml: 'subject.sex' is still FILL IN` | finish the rat file. Every unfilled field is listed at once |

A date folder that already has **any** metadata yaml in it, whatever it is called, is reported as
`has <name>, left alone` and never touched.

## Flags

| flag | what it does |
|---|---|
| *(none)* | report only. Nothing is written |
| `--write` | actually write the files |
| `--rat NAME` | only this rat folder. Repeatable |
| `--output-dir DIR` | write the session yamls into `DIR` instead of into each date folder, for a read-only data drive |

## If it does not find your files

The patterns are all in one table at the top of
[`src/jdb_to_nwb/make_metadata.py`](src/jdb_to_nwb/make_metadata.py) — `BEHAVIOR_FILES`,
`VIDEO_FILES`, `PHOTOMETRY_VARIANTS`, `EPHYS_FILES`. Each entry is a metadata field name, the glob
patterns to try in order, whether the section can be written without it, and whether to fall back to
the rat folder.

Adding a filename we have started using is a one-line change there. `tests/test_make_metadata.py`
covers each section, so run `pytest tests/test_make_metadata.py` after changing one.
