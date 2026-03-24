"""File I/O utilities for blink-labelled epoch data.

Handles saving and loading of epoch files that have been split into
blink-present and blink-absent subsets. Files are named with suffixes
indicating whether ASR was enabled and whether the epochs contain
blinks.
"""

from os.path import isdir, isfile
from mne import Epochs, read_epochs
from mne.epochs import EpochsFIF

from utils.utils import get_subject_list


def get_filename(
    output_folder: str, subject_id: str, with_asr: bool, with_blinks: bool
) -> str:
    """Generate a standardized filename for blink-labeled epoch files.

    Produces filenames like:
    ``sub-001_ASR_with_blinks_epo.fif`` or
    ``sub-002_without_blinks_epo.fif``

    Args:
        output_folder (str): Directory containing the epoch files.
        subject_id (str): Zero-padded subject identifier (e.g. "001").
        with_asr (bool): Whether the epochs were processed with ASR.
            Adds an "_ASR" suffix when True.
        with_blinks (bool): Whether the file contains epochs that
            overlap with detected blinks (True) or blink-free epochs
            (False).

    Returns:
        str: Full file path for the epoch file.
    """
    return f"{output_folder}/sub-{subject_id}{"_ASR" if with_asr else ""}_{"with" if with_blinks else "without"}_blinks_epo.fif"


def save_blink_epochs(
    output_folder: str,
    subject_id: str,
    epochs_with_blink: Epochs,
    epochs_without_blink: Epochs,
    with_asr: bool,
) -> None:
    """Save blink-present and blink-absent epoch subsets to disk.

    Writes two FIF files per subject: one containing epochs that
    overlap with detected blinks, and one containing blink-free
    epochs.

    Args:
        output_folder (str): Directory to save the epoch files into.
        subject_id (str): Zero-padded subject identifier (e.g. "001").
        epochs_with_blink (Epochs): Epochs that overlap with at least
            one detected blink.
        epochs_without_blink (Epochs): Epochs with no detected blinks.
        with_asr (bool): Whether the epochs were processed with ASR.
            Affects the output filename.
    """
    file_with_blinks = get_filename(
        output_folder, subject_id, with_asr, with_blinks=True
    )
    file_without_blinks = get_filename(
        output_folder, subject_id, with_asr, with_blinks=False
    )

    epochs_with_blink.save(file_with_blinks, overwrite=True)
    epochs_without_blink.save(file_without_blinks, overwrite=True)


def read_blink_epochs(
    data_folder: str, subject_id: str, with_asr: bool
) -> tuple[EpochsFIF, EpochsFIF]:
    """Load blink-present and blink-absent epoch files for one subject.

    Both files must exist; a FileNotFoundError is raised if either
    is missing.

    Args:
        data_folder (str): Directory containing the blink-labeled
            epoch files.
        subject_id (str): Zero-padded subject identifier (e.g. "001").
        with_asr (bool): Whether to load epochs from the ASR-enabled
            pipeline branch.

    Returns:
        tuple[EpochsFIF, EpochsFIF]: A tuple of:
            - Epochs that overlap with detected blinks.
            - Epochs with no detected blinks.

    Raises:
        FileNotFoundError: If the data folder does not exist or
            either epoch file is missing.
    """
    if not isdir(data_folder):
        raise FileNotFoundError(f"{data_folder} is not a directory")

    file_with_blinks = get_filename(data_folder, subject_id, with_asr, with_blinks=True)
    file_without_blinks = get_filename(
        data_folder, subject_id, with_asr, with_blinks=False
    )

    if isfile(file_with_blinks):
        epochs_with_blinks = read_epochs(file_with_blinks, preload=True)
    else:
        raise FileNotFoundError(f"{data_folder} doesn't have file: {file_with_blinks}")

    if isfile(file_without_blinks):
        epochs_without_blinks = read_epochs(file_without_blinks, preload=True)
    else:
        raise FileNotFoundError(
            f"{data_folder} doesn't have file: {file_without_blinks}"
        )

    return (
        epochs_with_blinks,
        epochs_without_blinks,
    )  # pyright: ignore[reportReturnType]


def load_all_epochs(
    bids_root: str, output_folder: str, with_asr: bool
) -> tuple[dict[str, Epochs], dict[str, Epochs]]:
    """Load blink-labelled epochs for all subjects.

    Iterates over all subjects in the BIDS dataset and loads both
    blink-present and blink-absent epoch files for each.

    Args:
        bids_root (str): Root directory of the BIDS dataset, used
            to discover subject IDs.
        output_folder (str): Directory containing the blink-labeled
            epoch files.
        with_asr (bool): Whether to load epochs from the ASR-enabled
            pipeline branch.

    Returns:
        tuple[dict[int, Epochs], dict[int, Epochs]]: A tuple of:
            - Dictionary mapping subject index to blink-present epochs.
            - Dictionary mapping subject index to blink-absent epochs.

    Raises:
        FileNotFoundError: If any subject's epoch files are missing.
    """
    with_blinks = {}
    without_blinks = {}

    subject_ids = get_subject_list(bids_root)

    for i, subject_id in enumerate(subject_ids):

        epochs_with_blinks, epochs_without_blinks = read_blink_epochs(
            output_folder, subject_id, with_asr
        )

        with_blinks[str(i)] = epochs_with_blinks
        without_blinks[str(i)] = epochs_without_blinks

    return with_blinks, without_blinks
