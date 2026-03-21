from mne import Epochs

from utils.config import StepTrialRejection


def reject_trials(
    epochs,
    config: StepTrialRejection,
    verbose=True,
) -> tuple[Epochs, dict]:
    """
    Reject epochs containing artifacts based on amplitude criteria.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to clean.
    eeg_threshold : float
        Peak-to-peak amplitude threshold for EEG channels (in Volts).
        Epochs exceeding this threshold will be rejected.
        Default: 150e-6 (150 µV).
    eog_threshold : float
        Peak-to-peak amplitude threshold for EOG channels (in Volts).
        Default: 250e-6 (250 µV).
    flat_threshold : float
        Minimum peak-to-peak amplitude (in Volts). Epochs with signals
        flatter than this will be rejected (likely bad channels/disconnections).
        Default: 1e-6 (1 µV).
    verbose : bool
        Whether to print rejection statistics.

    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs with bad trials rejected.
    reject_log : dict
        Dictionary containing rejection statistics.
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


def get_rejection_summary(reject_log):
    """
    Generate a detailed summary of which epochs were rejected and why.

    Parameters
    ----------
    reject_log : dict
        The rejection log returned by reject_trials().

    Returns
    -------
    summary : dict
        Dictionary with rejection reasons and affected epoch indices.
    """
    drop_log = reject_log["drop_log"]

    summary = {
        "kept": [],
        "rejected_by_channel": {},
        "user_rejected": [],
    }

    for idx, reasons in enumerate(drop_log):
        if len(reasons) == 0:
            summary["kept"].append(idx)
        elif reasons == ("USER",):
            summary["user_rejected"].append(idx)
        else:
            for channel in reasons:
                if channel not in summary["rejected_by_channel"]:
                    summary["rejected_by_channel"][channel] = []
                summary["rejected_by_channel"][channel].append(idx)

    return summary
