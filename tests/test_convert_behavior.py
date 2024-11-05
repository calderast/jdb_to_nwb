from datetime import datetime
from dateutil import tz
from pynwb import NWBFile

from jdb_to_nwb.convert_behavior import add_behavior

def test_convert_behavior():
    """ Test the add_behavior function. """

    # Test data is copied from /Volumes/Tim/Photometry/IM-1478/07252022/
    metadata = {}
    metadata["behavior"] = {}
    metadata["behavior"]["arduino_text_file_path"] = "tests/test_data/behavior/arduinoraw0.txt"
    metadata["behavior"]["arduino_timestamps_file_path"] = "tests/test_data/behavior/ArduinoStamps0.csv"
    metadata["behavior"]["maze_configuration_file_path"] = "tests/test_data/behavior/barriers.txt"

    nwbfile = NWBFile(
        session_description="Mock session",
        session_start_time=datetime.now(tz.tzlocal()),
        identifier="mock_session",
    )

    add_behavior(nwbfile, metadata)

# TODO: add some asserts to check:
# - correct number of blocks, trials, etc
# - all block start/ends are aligned to a trial start/end
# - loaded metadata is correct (probabilities and maze configs)