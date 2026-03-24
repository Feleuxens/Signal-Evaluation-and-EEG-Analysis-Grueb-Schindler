"""File I/O utilities for saving and loading pipeline outputs.

Handles reading and writing of MNE FIF files (epochs, raw, ICA)
and JSON metadata produced by the preprocessing pipeline.
"""

from glob import glob
from json import dumps, loads
from os.path import isdir, isfile

from mne import Epochs, read_epochs
from mne.io import read_raw_fif, Raw
from mne.io.edf.edf import RawEDF
from mne.preprocessing import ICA, read_ica


def save_data(
    output_folder: str,
    subject_id: str,
    epochs: Epochs,
    raw: RawEDF,
    ica: ICA | None,
    pipeline_stats: dict | None,
) -> None:
    """Save all pipeline outputs for a single subject.

    Writes epochs and raw data as FIF files. Optionally saves the
    ICA decomposition and pipeline statistics (e.g. rejection log)
    if they are available.

    Args:
        output_folder (str): Directory to write output files into.
        subject_id (str): Zero-padded subject identifier (e.g. "001").
            Used to construct filenames like sub-001_epo.fif.
        epochs (Epochs): Epoched EEG data to save.
        raw (RawEDF): Continuous EEG data to save. Saved as FIF
            regardless of the original input format.
        ica (ICA | None): Fitted ICA object, or None if ICA was
            skipped or failed.
        pipeline_stats (dict | None): Rejection statistics and
            metadata, or None if trial rejection was skipped.
    """
    epochs.save(f"{output_folder}/sub-{subject_id}_epo.fif", overwrite=True)
    raw.save(f"{output_folder}/sub-{subject_id}_raw.fif", overwrite=True)
    if ica is not None:
        ica.save(f"{output_folder}/sub-{subject_id}_ica.fif", overwrite=True)

    if pipeline_stats is not None:
        with open(f"{output_folder}/sub-{subject_id}_meta.txt", "w") as f:
            f.write(dumps(pipeline_stats))


def read_data(
    data_folder: str, config_id: int, subject_id: str
) -> tuple[Epochs | None, Raw | None, ICA | None, dict | None]:
    """Load all pipeline outputs for a single subject.

    Each file is loaded only if it exists; missing files result in
    None for that component. This allows partial loads when not all
    steps were run (e.g. ICA was skipped).

    Args:
        data_folder (str): Root directory containing per-config
            subdirectories (e.g. "data/processed").
        config_id (int): Numeric config identifier. Used to locate
            the subdirectory (e.g. "data/processed/1/").
        subject_id (str): Zero-padded subject identifier (e.g. "001").

    Returns:
        tuple[Epochs | None, Raw | None, ICA | None, dict | None]:
            A tuple of (epochs, raw, ica, pipeline_stats), where each
            element is None if the corresponding file was not found.

    Raises:
        FileNotFoundError: If the config subdirectory does not exist.
    """
    path = f"{data_folder}/{config_id}"
    if not isdir(path):
        raise FileNotFoundError(f"{data_folder}/{config_id} is not a directory")

    epochs = None
    raw = None
    ica = None
    pipeline_stats = None

    if isfile(f"{path}/sub-{subject_id}_epo.fif"):
        epochs = read_epochs(f"{path}/sub-{subject_id}_epo.fif", preload=True)
    if isfile(f"{path}/sub-{subject_id}_raw.fif"):
        raw = read_raw_fif(f"{path}/sub-{subject_id}_raw.fif", preload=True)
    if isfile(f"{path}/sub-{subject_id}_ica.fif"):
        ica = read_ica(f"{path}/sub-{subject_id}_ica.fif")
    if isfile(f"{path}/sub-{subject_id}_meta.txt"):
        with open(f"{path}/sub-{subject_id}_meta.txt", "r") as f:
            pipeline_stats = loads(f.read())

    return epochs, raw, ica, pipeline_stats


def read_all_files_per_type(
    data_folder: str, config_id: int, file_type: str
) -> dict[str, Epochs]:
    """Load a specific file type for all subjects in a config directory.

    Scans the config subdirectory for all matching files and returns
    them keyed by subject ID. Currently only supports epoch files.

    Args:
        data_folder (str): Root directory containing per-config
            subdirectories (e.g. "data/processed").
        config_id (int): Numeric config identifier. Used to locate
            the subdirectory.
        file_type (str): Type of file to load. Currently only "epo"
            is supported.

    Returns:
        dict[str, Epochs]: Mapping of subject ID to loaded Epochs
            object, sorted alphabetically by subject ID.

    Raises:
        FileNotFoundError: If the config subdirectory does not exist.
        ValueError: If no files of the requested type are found.
        NotImplementedError: If file_type is not "epo".
    """
    path = f"{data_folder}/{config_id}"
    if not isdir(path):
        raise FileNotFoundError(f"{data_folder}/{config_id} is not a directory")

    data = {}
    if file_type == "epo":
        files = sorted(glob(f"{path}/sub-*_epo.fif"))

        if not files:
            raise ValueError(f"No processed epoch files found in {path}")

        for fpath in files:
            # Extract subject ID from filename: "sub-001_epo.fif" -> "001"
            subject_id = fpath.split("sub-")[-1].split("_epo")[0]
            data[subject_id] = read_epochs(fpath, preload=True)
        return data

    else:
        raise NotImplementedError(f"File type {file_type} not implemented")
