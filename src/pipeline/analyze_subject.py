from os import mkdir
from os.path import isdir
from mne.preprocessing import ICA
from mne_bids import BIDSPath

from pipeline.step01_loading import load_data
from pipeline.step02_badchannels import detect_bad_channels
from pipeline.step03_filtering import filter_data
from pipeline.step04_downsampling import downsample_data
from pipeline.step05_referencing import rereference_data
from pipeline.step06_asr import run_asr
from pipeline.step07_ica import run_ica
from pipeline.step08_interpolation import interpolate_bad_channels
from pipeline.step09_epoching import epoch_data
from pipeline.step10_trialrejection import reject_trials

from utils.config import PipelineConfig
from utils.files import save_data, read_data, read_all_files_per_type
from utils.plots import (
    power_spectral_density_plot,
    ica_topography_plot,
    one_channel_erp_plot,
    butterfly_plot,
    all_channel_erp_plot,
    plot_channel,
    plot_topomap,
)
from utils.utils import pairwise_average, average_channel


def run_pipeline(
    config: PipelineConfig, bids_root: str, config_id: int, subject_id: str
) -> None:
    output_folder = f"{bids_root}/processed/{config_id}"
    if not isdir(output_folder):
        mkdir(output_folder)

    bids_path = BIDSPath(
        subject=subject_id,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
        task="jacobsen",
    )

    print("#############################################")
    print(f"# Config: {config_id} | Subject: {subject_id}")
    print("#")

    print("\nStep 01: Loading data")
    raw = load_data(bids_path)

    if config.bad_channels.enabled:
        print("\nStep 02: Detecting bad channels")
        raw = detect_bad_channels(raw, config.bad_channels)

    if config.filtering.enabled:
        print(f"\nStep 03: Filtering")
        raw = filter_data(raw, config.filtering)

    if config.downsampling.enabled:
        print(f"\nStep 04: Downsampling")
        raw = downsample_data(raw, config.downsampling)

    if config.rereferencing.enabled:
        print(f"\nStep 05: Rereferencing")
        raw = rereference_data(raw, config.rereferencing)

    if config.asr.enabled:
        print(f"\nStep 06: Artifact correction")
        raw, asr = run_asr(raw, config.asr)

    number_excluded_components = None
    ica: ICA | None = None
    if config.ica.enabled:
        print(f"\nStep 07: ICA cleaning")
        raw, ica, number_excluded_components = run_ica(raw, config.ica)

    if config.interpolation.enabled:
        print(f"\nStep 08: Interpolating bad channels")
        raw = interpolate_bad_channels(raw, config.interpolation)

    print(f"\nStep 09: Epoching")
    epochs, events, event_dict = epoch_data(raw, bids_path, config.epoching)

    pipeline_stats: dict | None = None
    if config.trial_rejection.enabled:
        print(f"\nStep 10: Trial rejection")
        epochs, reject_log = reject_trials(epochs, config.trial_rejection)

        pipeline_stats = reject_log
        pipeline_stats["ica_components_excluded"] = number_excluded_components

    save_data(output_folder, subject_id, epochs, raw, ica, pipeline_stats)


def plot_specific_subject(
    config: PipelineConfig, data_folder: str, config_id: int, subject_id: str
) -> None:
    epochs, raw, ica, pipeline_stats = read_data(data_folder, config_id, subject_id)

    output_folder = data_folder.rstrip("/") + "/" + str(config_id)

    if epochs is not None:
        one_channel_erp_plot(
            f"{output_folder}/sub-{subject_id}_one_channel_erp.png",
            raw,
            epochs,
            config.epoching.baseline,
        )
        all_channel_erp_plot(
            f"{output_folder}/sub-{subject_id}_all_channel_erp.png",
            epochs,
            config.epoching.baseline,
        )
        butterfly_plot(f"{output_folder}/sub-{subject_id}_butterfly", epochs)

    if ica is not None:
        ica_topography_plot(f"{output_folder}/sub-{subject_id}_ica", ica, raw)
    power_spectral_density_plot(f"{output_folder}/sub-{subject_id}_psd.png", raw, 0, 64)


def plot_average_data(config: PipelineConfig, data_folder: str, config_id: int) -> None:
    output_folder = data_folder.rstrip("/") + "/" + str(config_id)

    epochs_dict = read_all_files_per_type(data_folder, config_id, "epo")

    data_random_po7, data_regular_po7, times_po7, n_subjects, evoked_diff_po7 = (
        average_channel("PO7", epochs_dict)
    )
    plot_channel(
        f"{output_folder}/fig-average_po7.png",
        "PO7",
        data_random_po7,
        data_regular_po7,
        times_po7,
        n_subjects,
    )
    data_random_po8, data_regular_po8, times_po8, n_subjects, evoked_diff_po8 = (
        average_channel("PO8", epochs_dict)
    )
    plot_channel(
        f"{output_folder}/fig-average_po8.png",
        "PO8",
        data_random_po8,
        data_regular_po8,
        times_po8,
        n_subjects,
    )
    data_random_both = pairwise_average(data_random_po7, data_random_po8)
    data_regular_both = pairwise_average(data_regular_po7, data_regular_po8)
    plot_channel(
        f"{output_folder}/fig-average_po7po8.png",
        "PO7+PO8",
        data_random_both,
        data_regular_both,
        times_po7,
        n_subjects,
    )

    data_random_po7, data_regular_po7, times_po7, n_subjects, evoked_diff_po7 = (
        average_channel("PO7", epochs_dict)
    )
    plot_topomap(f"{output_folder}/fig-topomap_diff_po7.png", evoked_diff_po7)
