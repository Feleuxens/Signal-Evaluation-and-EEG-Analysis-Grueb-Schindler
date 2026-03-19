import json
from glob import glob
from math import inf
from os import listdir
from os.path import exists, join

import mne
import mne_bids
import numpy as np
from mne import read_epochs, Epochs


def get_subject_list(bids_root) -> list[str]:
    """Get list of subject IDs from BIDS dataset."""

    subject_list = mne_bids.get_entity_vals(bids_root, entity_key="subject")
    subject_list = [f"{int(s):03d}" for s in subject_list]  # zero-pad to 3 digits

    return subject_list


def get_config_ids(config_root: str) -> list[int]:
    """Get list of configs for pipeline execution."""
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
    for name in listdir(config_root):
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if stem.split("_", 1)[0] == str(config_id):
            return join(config_root, name)
    raise FileNotFoundError(f"No config file for id {config_id}")


def evoke_channels(epochs):
    desc_random = "random"
    desc_regular = "regular"

    # Select epochs by condition name (string key, not integer ID!)
    epochs_random = epochs[desc_random]
    epochs_regular = epochs[desc_regular]

    # Average across trials (mean)
    evoked_random = epochs_random.average()
    evoked_regular = epochs_regular.average()

    return evoked_random, evoked_regular


def average_channel(channel, epochs_dict: dict[int, Epochs]):
    """
    Load all processed epoch files and compute the grand average for channel PO7,
    separately for random and regular conditions
    """

    evokeds_random = []
    evokeds_regular = []
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

    if not evokeds_random:
        raise RuntimeError(f"No valid subjects with {channel} channel found.")

    # Stack and compute mean across subjects
    n_subjects = len(evokeds_random)
    evokeds_random = np.array(evokeds_random)  # shape: (n_subjects, n_times)
    evokeds_regular = np.array(evokeds_regular)

    data_random = np.mean(evokeds_random, axis=0) * 1e6  # Convert to µV
    data_regular = np.mean(evokeds_regular, axis=0) * 1e6

    print(f"\nGrand average computed from {n_subjects} subject(s)")
    print(f"  Time range: {times[0]:.3f} to {times[-1]:.3f} s")
    print(f"  Number of time points: {len(times)}")

    return data_random, data_regular, times, n_subjects, evoked_diff


def pairwise_average(arr1, arr2):
    assert len(arr1) == len(arr2)

    result = []
    for i in range(0, len(arr1)):
        result.append((arr1[i] + arr2[i]) / 2)
    return result


def pipeline_statistics(bids_root="../data/"):
    processed_dir = f"{bids_root}processed/"
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
