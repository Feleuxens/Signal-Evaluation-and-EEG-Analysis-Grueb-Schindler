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
    """
    Return a boolean numpy array of length n_epochs where True means the epoch
    overlaps at least one blink interval.
    epochs: mne.Epochs
    blink_intervals: list of (start_s, end_s) in absolute seconds (same reference as epochs.events)
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
):
    """
    Detect blink intervals on raw — one interval per blink:
      - bandpass EOG channels
      - build smoothed envelope = max(abs(channels))
      - threshold via median + mad_mult*MAD to find candidate peaks
      - for each candidate peak expand left/right until envelope <= baseline_level
        (baseline_level = median + 0.5 * MAD), which avoids splitting rise/fall into multiple blinks
      - merge overlapping/nearby intervals
    Returns:
      intervals: list of (start_s, end_s)
      durations: np.array of durations (s)
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
):

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
    if has_blink.ndim != 1 or has_blink.shape[0] != len(epochs):
        raise ValueError("filter must be a 1D boolean array with length == len(epochs)")

    print(f"hasblinks: {len(has_blink)}, epochs: {len(epochs)}")
    epochs_with_blink = epochs.copy().drop(~has_blink, reason="blink")
    epochs_without_blink = epochs.drop(has_blink, reason="no blink")

    return (epochs_with_blink, epochs_without_blink)


def process_subject_with_blinkdetection(
    bids_root: str, subject_id: str, config: PipelineConfig
) -> tuple[Epochs, Epochs, RawEDF]:
    """
    computes all epochs once with ASR and once without.
    First Epochs are with ASR,
    Second Epochs are without ASR
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

    return (epochs_after, epochs_before, raw_after)
