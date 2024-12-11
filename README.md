# jdb_to_nwb
Converts electrophysiology, photometry, and behavioral data for the hex maze task used by the Berke Lab at UCSF to NWB format for sharing and analysis.

## User Note
This repository is currently in development and should not be treated as a final/working version of an NWB conversion pipeline. Expect the code to change significantly before this pipeline is ready for regular use.

## Installation

```bash
git clone https://github.com/calderast/jdb_to_nwb.git
cd jdb_to_nwb
pip install -e .
```

## Current usage

1. Copy `tests/metadata_full.yaml` to the root directory of the repository:
```bash
cp tests/metadata_full.yaml .
```

2. Open `metadata_full.yaml` in a text editor. Update the paths to point to your data and update the metadata for your experiment.

3. Run the conversion to generate an NWB file (replace `out.nwb` with your desired output file name):
```bash
jdb_to_nwb metadata_full.yaml out.nwb
```

## Downloading test data

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
