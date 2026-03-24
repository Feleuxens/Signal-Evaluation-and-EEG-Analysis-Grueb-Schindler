"""Step 10: Amplitude-based trial (epoch) rejection.

Rejects epochs whose peak-to-peak amplitude exceeds a threshold
(likely artifact-contaminated) or falls below a minimum (likely
flat/disconnected channels). Rejection is applied separately for
EEG and EOG channel types, and statistics are logged per condition
for quality reporting.
"""

from mne import Epochs

from utils.config import StepTrialRejection


def reject_trials(
    epochs,
    config: StepTrialRejection,
    verbose=True,
) -> tuple[Epochs, dict]:
    """Reject epochs containing amplitude-based artifacts.

    Creates a copy of the epochs, applies peak-to-peak amplitude
    thresholds and flat-signal detection, then computes rejection
    statistics broken down by condition (random/regular).

    Args:
        epochs (Epochs): Epoched EEG data containing "random" and
            "regular" conditions. Not modified; a cleaned copy is
            returned.
        config (StepTrialRejection): Rejection parameters including
            amplitude thresholds for EEG and EOG channels, and the
            minimum amplitude for flat channel detection.
        verbose (bool): Whether to print rejection statistics to
            stdout. Defaults to True.

    Returns:
        tuple[Epochs, dict]: A tuple of:
            - Cleaned epochs with bad trials dropped.
            - Rejection log dictionary containing:
                - n_epochs_before/after: Total epoch counts.
                - n_epochs_regular/random_before: Per-condition counts.
                - n_rejected, n_rejected_random, n_rejected_regular:
                  Rejection counts overall and per condition.
                - rejection_rate: Percentage of epochs rejected.
                - reject_criteria, flat_criteria: Thresholds used.
                - drop_log: MNE's per-epoch drop log (tuple of tuples
                  listing the channels that caused each rejection).
    """
    n_epochs_before = len(epochs)

    # Build rejection criteria based on available channel types
    reject = {}
    flat = {}

    ch_types = epochs.get_channel_types()

    if "eeg" in ch_types:
        reject["eeg"] = config.eeg_threshold
        flat["eeg"] = config.flat_threshold

    if "eog" in ch_types:
        reject["eog"] = config.eog_threshold

    epochs_random = epochs["random"]
    epochs_regular = epochs["regular"]

    # Apply rejection criteria
    epochs_clean = epochs.copy()
    epochs_clean.drop_bad(reject=reject, flat=flat)
    epochs_random_clean = epochs_clean["random"]
    epochs_regular_clean = epochs_clean["regular"]

    n_epochs_after = len(epochs_clean)
    n_rejected = n_epochs_before - n_epochs_after
    rejection_rate = (n_rejected / n_epochs_before) * 100 if n_epochs_before > 0 else 0

    # Convert drop_log to serializable format for JSON export
    reject_log: dict = {
        "n_epochs_before": n_epochs_before,
        "n_epochs_after": n_epochs_after,
        "n_epochs_regular_before": len(epochs_regular),
        "n_epochs_random_before": len(epochs_random),
        "n_rejected": n_rejected,
        "n_rejected_random": len(epochs_random) - len(epochs_random_clean),
        "n_rejected_regular": len(epochs_regular) - len(epochs_regular_clean),
        "rejection_rate": rejection_rate,
        "reject_criteria": reject,
        "flat_criteria": flat,
        "drop_log": epochs_clean.drop_log,
    }

    if verbose:
        print(f"  Rejection thresholds: EEG={config.eeg_threshold*1e6:.0f} µV", end="")
        if "eog" in reject:
            print(f", EOG={config.eog_threshold*1e6:.0f} µV", end="")
        print(f", Flat<{config.flat_threshold*1e6:.1f} µV")
        print(f"  Epochs before rejection: {n_epochs_before}")
        print(f"  Epochs after rejection:  {n_epochs_after}")
        print(f"  Rejected: {n_rejected} ({rejection_rate:.1f}%)")

        if rejection_rate > 30:
            print(
                "  WARNING: High rejection rate (>30%). Consider reviewing data quality."
            )

    return epochs_clean, reject_log


def get_rejection_summary(reject_log: dict) -> dict[str, list | dict[str, list[int]]]:
    """Generate a detailed breakdown of rejection reasons per epoch.

    Parses MNE's drop_log to categorize each epoch as kept, rejected
    by a specific channel (with the channel name), or user-rejected.

    Args:
        reject_log (dict): Rejection log as returned by reject_trials().

    Returns:
        dict[str, list | dict[str, list[int]]]: Summary with keys:
            - "kept" (list[int]): Indices of retained epochs.
            - "rejected_by_channel" (dict[str, list[int]]): Mapping
              of channel name to the epoch indices it caused to be
              rejected.
            - "user_rejected" (list[int]): Indices of epochs rejected
              manually by the user (if any).
    """
    drop_log = reject_log["drop_log"]

    summary: dict[str, list | dict[str, list[int]]] = {
        "kept": [],
        "rejected_by_channel": {},
        "user_rejected": [],
    }

    for idx, reasons in enumerate(drop_log):
        if len(reasons) == 0:
            assert isinstance(summary["kept"], list)
            summary["kept"].append(idx)
        elif reasons == ("USER",):
            assert isinstance(summary["user_rejected"], list)
            summary["user_rejected"].append(idx)
        else:
            for channel in reasons:
                if channel not in summary["rejected_by_channel"]:
                    summary["rejected_by_channel"][channel] = []
                summary["rejected_by_channel"][channel].append(idx)

    return summary
