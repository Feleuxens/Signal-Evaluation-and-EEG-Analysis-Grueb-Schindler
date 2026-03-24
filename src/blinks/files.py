from os.path import isdir, isfile
from mne import Epochs, read_epochs

from utils.utils import get_subject_list


def get_filename(
    output_folder: str, subject_id: str, with_asr: bool, with_blinks: bool
) -> str:
    """
    Generates the filename for epo files based on the given conditions
    """
    return f"{output_folder}/sub-{subject_id}{"_ASR" if with_asr else ""}_{"with" if with_blinks else "without"}_blinks_epo.fif"


def save_blink_epochs(
    output_folder: str,
    subject_id: str,
    epochs_with_blink: Epochs,
    epochs_without_blink: Epochs,
    with_asr: bool,
):
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
) -> tuple[Epochs, Epochs]:
    if not isdir(data_folder):
        raise FileNotFoundError(f"{data_folder} is not a directory")

    epochs_with_blinks = None
    epochs_without_blinks = None

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
) -> tuple[dict[int, Epochs], dict[int, Epochs]]:
    with_blinks = {}
    without_blinks = {}

    subject_ids = get_subject_list(bids_root)

    for i, subject_id in enumerate(subject_ids):

        epochs_with_blinks, epochs_without_blinks = read_blink_epochs(
            output_folder, subject_id, with_asr
        )

        with_blinks[i] = epochs_with_blinks
        without_blinks[i] = epochs_without_blinks

    return (with_blinks, without_blinks)
