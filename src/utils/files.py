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
):
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

    return epochs, raw, ica, pipeline_stats  # pyright: ignore[reportReturnType]


def read_all_files_per_type(
    data_folder: str, config_id: int, file_type: str
) -> dict[int, Epochs]:
    path = f"{data_folder}/{config_id}"
    if not isdir(path):
        raise FileNotFoundError(f"{data_folder}/{config_id} is not a directory")

    data = {}
    if file_type == "epo":
        files = sorted(glob(f"{path}/sub-*_epo.fif"))

        if not files:
            raise ValueError(f"No processed epoch files found in {path}")

        for fpath in files:
            subject_id = fpath.split("sub-")[-1].split("_epo")[0]
            data[subject_id] = read_epochs(fpath, preload=True)
        return data

    else:
        raise NotImplementedError(f"File type {file_type} not implemented")
