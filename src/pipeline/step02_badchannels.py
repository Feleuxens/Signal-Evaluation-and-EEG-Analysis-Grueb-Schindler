"""Step 02: Automatic bad channel detection.

Identifies bad EEG channels using two complementary methods:
variance-based z-score detection (catches high-amplitude or flat
channels) and correlation-based detection (catches channels that
are uncorrelated with their neighbors). Detected channels are
added to raw.info["bads"] for exclusion in subsequent steps.
"""

import numpy as np
from mne.io.edf.edf import RawEDF

from utils.config import StepBadChannels


def detect_bad_channels(raw: RawEDF, config: StepBadChannels) -> RawEDF:
    """Detect and mark bad EEG channels automatically.

    Runs variance z-score and correlation z-score detection on all
    EEG channels, then appends any flagged channels to
    raw.info["bads"]. Non-EEG channels (EOG, misc, status) are
    excluded from the analysis.

    Args:
        raw (RawEDF): Continuous EEG data with channel types already
            set (EXG channels should be marked as eog/misc).
        config (StepBadChannels): Detection parameters, including
            the z-score threshold and channel prefixes to exclude.

    Returns:
        RawEDF: The same raw object with bad channels appended to
            raw.info["bads"]. Modified in place.
    """

    channel_names = raw.ch_names

    # Select only EEG channels, excluding external and status channels
    ch_types = raw.get_channel_types()
    eeg_picks = [
        i
        for i, ch in enumerate(channel_names)
        if not ch.upper().startswith(config.exg_prefix)
        and not ch.startswith(config.status_prefix)
        and ch_types[i] == "eeg"
    ]

    channel_names = [raw.ch_names[i] for i in eeg_picks]

    if not eeg_picks:
        print("No EEG channels found for bad channel detection.")
        return raw

    data = raw.get_data(picks=eeg_picks)

    bad_channels_var = _zscore_bad_channel_detection(
        data, channel_names, config.z_thresh
    )
    bad_channels_corr = _correlation_bad_channel_detection(
        data, channel_names, config.z_thresh
    )

    bad_channels = sorted(set(bad_channels_var + bad_channels_corr))

    # Mark detected bad channels in raw.info['bads']
    raw.info["bads"].extend(bad_channels)

    print(f"Automatically detected bad channels: {raw.info['bads']}")

    return raw


def _zscore_bad_channel_detection(data, channel_names, bad_channel_z_thresh):
    """Detect bad channels via variance z-scores.

    Computes the variance of each channel across time, then
    z-scores these variances across channels. Channels whose
    absolute z-score exceeds the threshold are flagged (catches
    both abnormally noisy and abnormally flat channels).

    Args:
        data (np.ndarray): EEG data matrix of shape
            (n_channels, n_times).
        channel_names (list[str]): Channel names corresponding to
            rows of data.
        bad_channel_z_thresh (float): Absolute z-score threshold
            above which a channel is flagged as bad.

    Returns:
        list[str]: Names of channels flagged by this method.
    """
    variances = np.var(data, axis=1)
    z_var = (variances - variances.mean()) / variances.std()
    bad_channels_var = [
        channel
        for channel, z in zip(channel_names, z_var)
        if np.abs(z) > bad_channel_z_thresh
    ]

    return bad_channels_var


def _correlation_bad_channel_detection(data, channel_names, bad_channel_z_thresh):
    """Detect bad channels via inter-channel correlation.

    Computes the full pairwise correlation matrix, takes the median
    correlation of each channel with all others, then z-scores
    these medians. Channels with a z-score below the negative
    threshold are flagged (catches channels that are disconnected
    or dominated by independent noise).

    Args:
        data (np.ndarray): EEG data matrix of shape
            (n_channels, n_times).
        channel_names (list[str]): Channel names corresponding to
            rows of data.
        bad_channel_z_thresh (float): Z-score threshold. Channels
            with median correlation z-score below the negative of
            this value are flagged.

    Returns:
        list[str]: Names of channels flagged by this method.
    """
    corr_matrix = np.corrcoef(data)
    mean_corr = np.median(corr_matrix, axis=0)
    z_corr = (mean_corr - mean_corr.mean()) / mean_corr.std()
    bad_channels_corr = [
        channel
        for channel, z in zip(channel_names, z_corr)
        if z < -bad_channel_z_thresh
    ]

    return bad_channels_corr
