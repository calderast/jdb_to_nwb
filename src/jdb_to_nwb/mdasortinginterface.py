# TODO contribute this back to neuroconv with testing data

from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import BaseSortingExtractorInterface
from pydantic import FilePath


class MdaSortingInterface(BaseSortingExtractorInterface):
    """Primary data interface class for converting a MdaSortingInterface (MountainSort) from spikeinterface."""

    display_name = "MountainSort Sorting"
    associated_suffixes = (".mda",)
    info = "Interface for MountainSort sorting data."

    @classmethod
    def get_source_schema(cls) -> dict:
        source_schema = super().get_source_schema()
        source_schema["properties"]["file_path"]["description"] = "Path to the output MDA file"
        return source_schema

    def __init__(
        self,
        file_path: FilePath,
        sampling_frequency: int,
        verbose: bool = True,
    ):
        """
        Load and prepare sorting data for MountainSort

        Parameters
        ----------
        file_path: str or Path
            Path to the output MDA file
        sampling_frequency : int
            The sampling frequency in Hz.
        verbose: bool, default: True
        """
        super().__init__(file_path=file_path, sampling_frequency=sampling_frequency, verbose=verbose)
