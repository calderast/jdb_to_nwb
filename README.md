# jdb_to_nwb
Converts electrophysiology, photometry, and behavioral data for the hex maze task used by the Berke Lab at UCSF to NWB format for sharing and analysis.

## Installation

```bash
git clone https://github.com/calderast/jdb_to_nwb.git
cd jdb_to_nwb
pip install -e .
```

## Current usage

1. Copy `tests/test_data/metadata_full.yaml` to the root directory of the repository:
```bash
cp tests/test_data/metadata_full.yaml .
```

2. Open `metadata_full.yaml` in a text editor. Update the paths to point to your data and update the metadata for your experiment.

3. Run the conversion to generate an NWB file (replace `out.nwb` with your desired output file name):
```bash
jdb_to_nwb metadata_full.yaml out.nwb
```

## Versioning

Versioning is handled automatically using [hatch-vcs](https://github.com/ofek/hatch-vcs) using the latest
tag in the git history as the version number. To make a new release, simply tag the current commit and 
push to the repository. Use [semantic versioning](https://semver.org/) to set the version number.
Create a GitHub release using the tag.

