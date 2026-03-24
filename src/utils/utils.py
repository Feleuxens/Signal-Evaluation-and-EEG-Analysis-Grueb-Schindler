"""Utility functions for the EEG processing pipeline.

Provides helpers for discovering subjects and configs, computing
grand averages across subjects, and printing pipeline statistics.
"""

import json
from glob import glob
from math import inf
from os import listdir
from os.path import exists, join

import mne
import mne_bids
import numpy as np
from mne import Epochs, Evoked


def get_subject_list(bids_root) -> list[str]:
    """Get list of zero-padded subject IDs from a BIDS dataset.

    Args:
        bids_root (str): Root directory of the BIDS dataset.

    Returns:
        list[str]: Subject IDs zero-padded to 3 digits
            (e.g. ["001", "002", ...]).
    """

    subject_list = mne_bids.get_entity_vals(bids_root, entity_key="subject")
    subject_list = [f"{int(s):03d}" for s in subject_list]  # zero-pad to 3 digits

    return subject_list


def get_config_ids(config_root: str) -> list[int]:
    """Get sorted list of numeric config IDs from a config directory.

    Config files are expected to be named with a leading numeric ID
    separated by an underscore (e.g. "1_default.toml", "2_no_asr.toml").

    Args:
        config_root (str): Directory containing TOML config files.

    Returns:
        list[int]: Sorted list of config IDs.

    Raises:
        AssertionError: If the config directory does not exist.
        ValueError: If duplicate config IDs are found.
    """
    assert exists(config_root)
    ids = []
    for name in listdir(config_root):
        stem = name.rsplit(".", 1)[0] if "." in name else name
        num = stem.split("_", 1)[0]
        ids.append(int(num))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate config ids found")
    return sorted(ids)


def get_config_path(config_root: str, config_id: int) -> str:
    """Resolve a numeric config ID to its full file path.

    Scans the config directory for a file whose name starts with
    the given ID.

    Args:
        config_root (str): Directory containing TOML config files.
        config_id (int): Numeric config identifier to look up.

    Returns:
        str: Full path to the matching config file.

    Raises:
        FileNotFoundError: If no file matches the given config ID.
    """
    for name in listdir(config_root):
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if stem.split("_", 1)[0] == str(config_id):
            return join(config_root, name)
    raise FileNotFoundError(f"No config file for id {config_id}")


def evoke_channels(epochs: Epochs) -> tuple[mne.EvokedArray, mne.EvokedArray]:
    """Compute average evoked responses for random and regular conditions.

    Args:
        epochs (Epochs): Epoched data containing "random" and
            "regular" condition labels.

    Returns:
        tuple[mne.EvokedArray, mne.EvokedArray]: A tuple of
            (evoked_random, evoked_regular), each averaged across
            all trials in the respective condition.
    """
    desc_random = "random"
    desc_regular = "regular"

    # Select epochs by condition name (string key, not integer ID!)
    epochs_random = epochs[desc_random]
    epochs_regular = epochs[desc_regular]

    # Average across trials (mean)
    evoked_random = epochs_random.average()
    evoked_regular = epochs_regular.average()

    return evoked_random, evoked_regular


def average_channel(
    channel, epochs_dict: dict[str, Epochs]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, Evoked]:
    """Compute grand average ERP at a single channel across subjects.

    For each subject, extracts the evoked response at the specified
    channel for both conditions, then averages across all subjects.

    Args:
        channel (str): Channel name to extract (e.g. "PO7", "PO8").
        epochs_dict (dict[str, Epochs]): Mapping of subject ID to
            Epochs object, as returned by read_all_files_per_type().

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, int, Evoked]:
            A tuple of:
            - data_random: Grand average for random condition in µV,
              shape (n_times,).
            - data_regular: Grand average for regular condition in µV,
              shape (n_times,).
            - times: Time points in seconds, shape (n_times,).
            - n_subjects: Number of subjects included.
            - evoked_diff: Difference wave (regular minus random) from
              the last subject, used for topographic plotting.

    Raises:
        RuntimeError: If no subjects have the requested channel.
    """
    evokeds_random: list[int] = []
    evokeds_regular: list[int] = []
    evoked_diff = None
    times = None

    for subject_id, epochs in epochs_dict.items():

        # Check if channel exists
        if channel not in epochs.ch_names:
            print(
                f"    WARNING: {channel} not found in subject {subject_id}, skipping."
            )
            continue

        # Create evoked responses for each condition
        print(f"Evoking subject {subject_id}")
        evoked_random, evoked_regular = evoke_channels(epochs)
        evoked_diff = mne.combine_evoked(
            [evoked_regular, evoked_random], weights=[1, -1]
        )

        # Get channel channel index and extract data
        channel_idx = evoked_random.ch_names.index(channel)

        evokeds_random.append(evoked_random.get_data()[channel_idx, :])
        evokeds_regular.append(evoked_regular.get_data()[channel_idx, :])

        times = evoked_random.times  # Same for all subjects

    if (
        not evokeds_random
        or not evokeds_regular
        or not evoked_diff
        or times is None
        or len(times) == 0
    ):
        raise RuntimeError(f"No valid subjects with {channel} channel found.")

    # Stack and compute mean across subjects
    n_subjects = len(evokeds_random)
    evokeds_random_arr = np.array(evokeds_random)  # shape: (n_subjects, n_times)
    evokeds_regular_arr = np.array(evokeds_regular)

    data_random = np.mean(evokeds_random_arr, axis=0) * 1e6  # Convert to µV
    data_regular = np.mean(evokeds_regular_arr, axis=0) * 1e6

    print(f"\nGrand average computed from {n_subjects} subject(s)")
    print(f"  Time range: {times[0]:.3f} to {times[-1]:.3f} s")
    print(f"  Number of time points: {len(times)}")

    return data_random, data_regular, times, n_subjects, evoked_diff


def pairwise_average(
    arr1: list[int | float] | np.ndarray, arr2: list[int | float] | np.ndarray
) -> np.ndarray:
    """Compute element-wise average of two equal-length lists.

    Used to average ERPs across two channels (e.g. PO7 and PO8)
    to produce a combined bilateral measure.

    Args:
        arr1 (list[int | float]) | np.ndarray: First array of values.
        arr2 (list[int | float]) | np.ndarray: Second array of values, must be
            the same length as arr1.

    Returns:
        list[float]: Element-wise mean of arr1 and arr2.

    Raises:
        AssertionError: If the two lists have different lengths.
    """
    assert len(arr1) == len(arr2)

    result: list[float] = []
    for i in range(0, len(arr1)):
        result.append((arr1[i] + arr2[i]) / 2)
    return np.array(result, dtype=np.float32)


def pipeline_statistics(bids_root: str, config: int) -> None:
    """Print summary statistics for a completed pipeline run.

    Reads all per-subject metadata files and reports trial rejection
    counts (overall, random, regular) and ICA component removal
    statistics across all subjects.

    Args:
        bids_root (str): Root directory of the BIDS dataset. The
            processed outputs are expected under
            ``<bids_root>/processed/<config>/``.
        config (int): Numeric config identifier used to locate the
            output subdirectory.
    """
    processed_dir = f"{bids_root.rstrip('/')}/processed/{config}/"
    meta_files = sorted(glob(f"{processed_dir}sub-*_meta.txt"))

    if not meta_files:
        print(f"No meta files found in {processed_dir}")

    raw_data = []
    for file in meta_files:
        with open(file, "r") as f:
            raw_data.append(json.loads(f.read()))

    trial_rejection_regular_min = inf
    trial_rejection_regular_max = -1
    trial_rejection_regular_sum = 0
    trial_rejection_random_min = inf
    trial_rejection_random_max = -1
    trial_rejection_random_sum = 0
    trial_rejection_overall_min = inf
    trial_rejection_overall_max = -1
    trial_rejection_overall_sum = 0
    number_of_trials = 0
    number_of_trials_regular = 0
    number_of_trials_random = 0
    ica_removed_components_min = inf
    ica_removed_components_max = -1
    ica_removed_components_sum = 0
    participants = len(raw_data)

    for data in raw_data:
        trial_rejection_regular_min = min(
            trial_rejection_regular_min, data["n_rejected_regular"]
        )
        trial_rejection_regular_max = max(
            trial_rejection_regular_max, data["n_rejected_regular"]
        )
        trial_rejection_regular_sum += data["n_rejected_regular"]
        trial_rejection_random_min = min(
            trial_rejection_random_min, data["n_rejected_random"]
        )
        trial_rejection_random_max = max(
            trial_rejection_random_max, data["n_rejected_random"]
        )
        trial_rejection_random_sum += data["n_rejected_random"]
        trial_rejection_overall_min = min(
            trial_rejection_overall_min, data["n_rejected"]
        )
        trial_rejection_overall_max = max(
            trial_rejection_overall_max, data["n_rejected"]
        )
        trial_rejection_overall_sum += data["n_rejected"]
        number_of_trials += data["n_epochs_before"]
        number_of_trials_regular += data["n_epochs_regular_before"]
        number_of_trials_random += data["n_epochs_random_before"]
        ica_removed_components_min = min(
            ica_removed_components_min, data["ica_components_excluded"]
        )
        ica_removed_components_max = max(
            ica_removed_components_max, data["ica_components_excluded"]
        )
        ica_removed_components_sum += data["ica_components_excluded"]

    print(
        f"Number of trials: {number_of_trials}  |  Random: {number_of_trials_random}, Regular: {number_of_trials_regular}"
    )
    print(
        f"Number of trials removed: {trial_rejection_overall_sum} -> {round((trial_rejection_overall_sum/number_of_trials)*100, 2)} %  | Min: {trial_rejection_overall_min}, Max: {trial_rejection_overall_max}"
    )
    print(
        f"Number of random trials removed: {trial_rejection_random_sum} -> {round((trial_rejection_random_sum/number_of_trials)*100, 2)} %  | Min: {trial_rejection_random_min}, Max: {trial_rejection_random_max}"
    )
    print(
        f"Number of regular trials removed: {trial_rejection_regular_sum} -> {round((trial_rejection_regular_sum/number_of_trials)*100, 2)} %  | Min: {trial_rejection_regular_min}, Max: {trial_rejection_regular_max}"
    )
    print(
        f"Ica components removed: Overall {ica_removed_components_sum}, average per participant {round(ica_removed_components_sum/participants, 2)}, min {ica_removed_components_min}, max {ica_removed_components_max}"
    )
