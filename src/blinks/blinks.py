"""Blink detection and blink-conditioned epoch analysis.

Implements automatic blink detection on EOG channels using a
threshold-based algorithm, and provides utilities to split epochs
into blink-present and blink-absent subsets. This module supports
the investigation of how ASR artifact correction interacts with
blink-contaminated epochs (see Section 5 of the report).
"""

from mne_bids import BIDSPath
import mne
from mne.io.edf.edf import RawEDF
from mne import Epochs
import numpy as np
from scipy import signal

from pipeline.step01_loading import load_data
from pipeline.step02_badchannels import detect_bad_channels
from pipeline.step03_filtering import filter_data
from pipeline.step04_downsampling import downsample_data
from pipeline.step05_referencing import rereference_data
from pipeline.step06_asr import run_asr
from pipeline.step07_ica import run_ica
from pipeline.step08_interpolation import interpolate_bad_channels
from pipeline.step09_epoching import epoch_data

from utils.config import PipelineConfig
from utils.utils import get_subject_list

from blinks.files import save_blink_epochs


def epochs_have_blinks(
    epochs: mne.Epochs, blink_intervals
) -> np.typing.NDArray[np.bool]:
    """Check which epochs overlap with detected blink intervals.

    For each epoch, determines whether its time window overlaps with
    any blink interval. Both epochs and blink intervals are compared
    in absolute time (seconds from recording start).

    Args:
        epochs (Epochs): Epoched EEG data. Event sample indices and
            epoch tmin/tmax are used to compute absolute time windows.
        blink_intervals (list[tuple[float, float]]): List of
            (start_seconds, end_seconds) pairs defining each detected
            blink in absolute recording time.

    Returns:
        NDArray[np.bool_]: Boolean array of shape (n_epochs,) where
            True indicates the epoch overlaps at least one blink.
    """
    sfreq = epochs.info["sfreq"]
    event_samples = epochs.events[:, 0]
    epoch_start_times = event_samples / sfreq + epochs.tmin
    epoch_duration = epochs.times[-1] - epochs.times[0]
    epoch_end_times = epoch_start_times + epoch_duration

    has_blink = np.zeros(len(epoch_start_times), dtype=bool)
    if not blink_intervals:
        return has_blink

    bi = np.array(blink_intervals)
    for i, (s0, e0) in enumerate(zip(epoch_start_times, epoch_end_times)):
        overlaps = np.logical_not((bi[:, 1] <= s0) | (bi[:, 0] >= e0))
        if overlaps.any():
            has_blink[i] = True

    return has_blink


def detect_blinks_on_raw(
    raw: mne.io.edf.edf.RawEDF,
    eog_chs: list[str],
    l_freq=1.0,
    h_freq=15.0,
    envelope_smooth_ms=20.0,
    mad_mult=6.0,
    min_distance_s=0.05,
    merge_gap_s=0.02,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Detect blink intervals on raw EOG data.

    Uses a multistep algorithm:
    1. Bandpass filter EOG channels to isolate blink frequency range.
    2. Compute a smoothed amplitude envelope (max absolute value
       across EOG channels).
    3. Apply a robust threshold (median + mad_mult * MAD) to find
       candidate blink peaks.
    4. Expand each peak bidirectionally until the envelope falls
       below a baseline level (median + 0.5 * MAD).
    5. Merge overlapping or nearby intervals.

    Args:
        raw (RawEDF): Continuous EEG data containing EOG channels.
        eog_chs (list[str]): Names of EOG channels to use for
            detection (e.g. ["EOG5", "EOG6"]).
        l_freq (float): Lower bandpass frequency in Hz for EOG
            filtering. Defaults to 1.0.
        h_freq (float): Upper bandpass frequency in Hz for EOG
            filtering. Defaults to 15.0.
        envelope_smooth_ms (float): Smoothing window width in
            milliseconds for the amplitude envelope. Defaults to 20.0.
        mad_mult (float): Multiplier for the MAD-based peak detection
            threshold. Higher values detect fewer, more prominent
            blinks. Defaults to 6.0.
        min_distance_s (float): Minimum distance in seconds between
            detected peaks, passed to scipy's find_peaks. Defaults
            to 0.05.
        merge_gap_s (float): Maximum gap in seconds between adjacent
            intervals before they are merged into a single blink.
            Defaults to 0.02.

    Returns:
        tuple[list[tuple[float, float]], np.ndarray]: A tuple of:
            - List of (start_seconds, end_seconds) blink intervals
              in absolute recording time, sorted chronologically.
            - Array of blink durations in seconds, shape (n_blinks,).
              Empty array if no blinks were detected.
    """

    sfreq = raw.info["sfreq"]
    eog_idx = mne.pick_channels(raw.ch_names, include=eog_chs)
    eog_data = raw.get_data(picks=eog_idx)

    # Bandpass filter for blink detection
    eog_filtered = mne.filter.filter_data(
        eog_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
    )

    # envelope: max absolute across eog channels
    env = np.max(np.abs(eog_filtered), axis=0)

    # smooth envelope
    win = int(round(envelope_smooth_ms * 1e-3 * sfreq))
    win = max(1, win)
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="same")

    # robust baseline & threshold
    med = np.median(env_smooth)
    mad = np.median(np.abs(env_smooth - med))
    thresh = med + mad_mult * mad
    baseline_level = med + 0.5 * mad

    min_dist = int(round(min_distance_s * sfreq))
    peaks, _ = signal.find_peaks(env_smooth, height=thresh, distance=min_dist)

    intervals = []
    n = env_smooth.size
    for p in peaks:
        s = p
        while s > 0 and env_smooth[s] > baseline_level:
            s -= 1
        e = p
        while e < n - 1 and env_smooth[e] > baseline_level:
            e += 1
        intervals.append((max(0, s), min(n - 1, e)))

    # convert to seconds
    intervals_s = [(s / sfreq, e / sfreq) for s, e in intervals]
    if not intervals_s:
        return [], np.array([])

    # merge nearby/overlapping intervals
    intervals_s.sort()
    merged = []
    cur_s, cur_e = intervals_s[0]
    for s, e in intervals_s[1:]:
        if s <= cur_e + merge_gap_s:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    durations = np.array([e - s for s, e in merged])

    return merged, durations


def precompute_all_epochs(
    bids_root: str, config: PipelineConfig, output_folder: str, with_asr: bool
) -> None:
    """Run the pipeline and blink detection for all subjects, saving results.

    For each subject, runs the preprocessing pipeline (with and without
    ASR), detects blinks on the EOG channels, splits epochs into
    blink-present and blink-absent subsets, and saves both to disk.

    Args:
        bids_root (str): Root directory of the BIDS dataset.
        config (PipelineConfig): Pipeline configuration used for
            all preprocessing steps.
        output_folder (str): Directory to save blink-labelled epoch
            files into.
        with_asr (bool): If True, use epochs from the ASR-enabled
            pipeline branch. If False, use epochs from the branch
            without ASR.
    """

    subject_ids = get_subject_list(bids_root)

    for i, subject_id in enumerate(subject_ids):

        epochs_after, epochs_before, raw_after = process_subject_with_blinkdetection(
            bids_root, subject_id, config
        )

        epochs = epochs_after if with_asr else epochs_before

        # eeg_chs = ['PO7','PO8']
        eog_chs = ["EOG5", "EOG6"]

        blink_intervals, _ = detect_blinks_on_raw(
            raw_after,
            eog_chs=eog_chs,
            l_freq=1.0,
            h_freq=15.0,
            envelope_smooth_ms=20.0,
            mad_mult=6.0,
        )

        has_blink = epochs_have_blinks(epochs, blink_intervals)

        epochs_with_blinks, epochs_without_blinks = filter_blinks(epochs, has_blink)

        save_blink_epochs(
            output_folder,
            subject_id,
            epochs_with_blinks,
            epochs_without_blinks,
            with_asr,
        )


def filter_blinks(
    epochs: Epochs, has_blink: np.typing.NDArray[np.bool]
) -> tuple[Epochs, Epochs]:
    """Split epochs into blink-present and blink-absent subsets.

    Args:
        epochs (Epochs): Full set of epochs to split.
        has_blink (NDArray[np.bool_]): Boolean array of shape
            (n_epochs,) indicating which epochs contain blinks,
            as returned by epochs_have_blinks().

    Returns:
        tuple[Epochs, Epochs]: A tuple of:
            - Epochs that overlap with at least one blink.
            - Epochs with no detected blinks.

    Raises:
        ValueError: If has_blink is not a 1D boolean array matching
            the number of epochs.
    """
    if has_blink.ndim != 1 or has_blink.shape[0] != len(epochs):
        raise ValueError("filter must be a 1D boolean array with length == len(epochs)")

    print(f"hasblinks: {len(has_blink)}, epochs: {len(epochs)}")
    epochs_with_blink = epochs.copy().drop(~has_blink, reason="blink")
    epochs_without_blink = epochs.drop(has_blink, reason="no blink")

    return (epochs_with_blink, epochs_without_blink)


def process_subject_with_blinkdetection(
    bids_root: str, subject_id: str, config: PipelineConfig
) -> tuple[Epochs, Epochs, RawEDF]:
    """Run the preprocessing pipeline twice for one subject: with and without ASR.

    Executes shared steps (loading through re-referencing) once, then
    forks the data into two branches: one with ASR enabled and one
    without. Both branches then proceed through ICA, interpolation,
    and epoching independently.

    Args:
        bids_root (str): Root directory of the BIDS dataset.
        subject_id (str): Zero-padded subject identifier (e.g. "001").
        config (PipelineConfig): Pipeline configuration controlling
            which steps are enabled and their parameters.

    Returns:
        tuple[Epochs, Epochs, RawEDF]: A tuple of:
            - Epochs from the pipeline branch with ASR.
            - Epochs from the pipeline branch without ASR.
            - Raw data after ASR (used for blink detection on
              the processed EOG channels).
    """
    bids_path = BIDSPath(
        subject=subject_id,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
        task="jacobsen",
    )

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

    # Fork into two branches: with ASR and without ASR
    raw_before = raw.copy()
    raw_after = raw

    if config.asr.enabled:
        print(f"\nStep 06: Artifact correction")
        raw_after, _ = run_asr(raw_after, config.asr)

    if config.ica.enabled:
        print(f"\nStep 07: ICA cleaning")
        raw_after, _, _ = run_ica(raw_after, config.ica)
        raw_before, _, _ = run_ica(raw_before, config.ica)

    if config.interpolation.enabled:
        print(f"\nStep 08: Interpolating bad channels")
        raw_after = interpolate_bad_channels(raw_after, config.interpolation)
        raw_before = interpolate_bad_channels(raw_before, config.interpolation)

    print(f"\nStep 09: Epoching")
    epochs_after, _, _ = epoch_data(raw_after, bids_path, config.epoching)
    epochs_before, _, _ = epoch_data(raw_before, bids_path, config.epoching)

    return epochs_after, epochs_before, raw_after
