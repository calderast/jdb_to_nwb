# jdb_to_nwb

[![PyPI version](https://img.shields.io/pypi/v/jdb-to-nwb.svg)](https://pypi.org/project/jdb-to-nwb/)
[![Test Package Build Status](https://github.com/calderast/jdb_to_nwb/actions/workflows/test_package_build.yml/badge.svg)](https://github.com/calderast/jdb_to_nwb/actions)

Converts electrophysiology, photometry, and behavioral data for the hex maze task used by the Berke Lab at UCSF to NWB format for sharing and analysis.

## Installation

```bash
git clone https://github.com/calderast/jdb_to_nwb.git
cd jdb_to_nwb
pip install -e .
```

## Usage

0. Run an experiment with the hex maze. This generates various files (arduino text, arduino timestamps, behavior video, video timestamps, and potentially ephys or photometry files if you were recording those). You will then need to manually generate some files (record the maze configuration(s) you used in a text file, run DeepLabCut on the behavior video if you want position tracking, get the centroids of each hex in video pixel coordinates using the notebook at `resources/Get_Hex_centroids.ipynb`). See `metadata_fully_explained.yaml` and `FAQs.md` for more description of all of the files you need.

1. Open one of the example metadata files in a text editor (`metadata_example_Jose.yaml` for Jose's experiments, or `metadata_example_Tim.yaml` for Tim / Yang-Sun / Stephanie's experiments). Update the paths to point to your data files for this session and update the metadata for your experiment. See `metadata_fully_explained.yaml` for an explanation of all metadata fields.

2. Run the conversion to generate an NWB file (replace `output_dir` with your desired output directory).
The nwb file will be automatically named based on the animal name and date (i.e. `rat_date.nwb`):
```bash
jdb_to_nwb metadata_example.yaml output_dir
```

3. Sub-directories for associated figures and conversion log files will be created alongside the nwb file in `output_dir`. Check that there are no warnings or errors in the `warning` log file and that all figures look as expected. It's also a good idea to read through the `info` log file to check that everything makes sense based on your experimental setup. If you are not sure if the conversion was successful (or just want to double check), you can send the `debug` log file to Stephanie to make sure.

## Downloading test data (Developers only)

The large test data files are stored in a shared UCSF Box account. To get access to the test data,
please contact the repo maintainers.

Create a new file called `.env` in the root directory of the repository and add your Box credentials:
```bash
BOX_USERNAME=<your_box_username>
BOX_PASSWORD=<your_box_password>
```
Or set the environment variables in your shell:
```bash
export BOX_USERNAME=<your_box_username>
export BOX_PASSWORD=<your_box_password>
```

Then run the download script:
```bash
python tests/download_test_data.py
```

You can pass the `--overwrite` flag to overwrite existing files:
```bash
python tests/download_test_data.py --overwrite
```

Notes:
- Run `python tests/test_data/create_raw_ephys_test_data.py` to re-create the test data for `raw_ephys`.
- Run `python tests/test_data/create_processed_ephys_test_data.py` to re-create the test data for `processed_ephys`.
- `tests/test_data/processed_ephys/impedance.csv` was manually created for testing purposes.
- `tests/test_data/processed_ephys/geom.csv` was manually created for testing purposes.
- Some files (`settings.xml`, `structure.oebin`) nested within `tests/test_data/raw_ephys/2022-07-25_15-30-00` 
  were manually created for testing purposes.

The GitHub Actions workflow (`.github/workflows/test_package_build.yml`) will automatically download the test data and run the tests.


## Versioning

Versioning is handled automatically using [hatch-vcs](https://github.com/ofek/hatch-vcs) using the latest
tag in the git history as the version number. To make a new release, simply tag the current commit and 
push to the repository. Use [semantic versioning](https://semver.org/) to set the version number.
Create a GitHub release using the tag.
