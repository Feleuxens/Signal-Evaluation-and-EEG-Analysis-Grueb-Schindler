from os import getenv
from mne_bids import BIDSPath
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.lines import Line2D

from pipeline.step01_loading import load_data
from pipeline.step02_badchannels import detect_bad_channels
from pipeline.step03_filtering import filter_data
from pipeline.step04_downsampling import downsample_data
from pipeline.step05_referencing import rereference_data
from pipeline.step06_asr import run_asr
from pipeline.step07_ica import run_ica
from pipeline.step08_interpolation import interpolate_bad_channels
from pipeline.step09_epoching import epoch_data

from utils.config import load_config, PipelineConfig
from utils.utils import get_config_path


def main():
    bids_root = getenv("BIDS_ROOT", "../data/")
    bids_root = bids_root.rstrip("/")
    config_root = getenv("CONFIG_ROOT", "../config/")
    config_root = config_root.rstrip("/")

    # load default config
    config_path = get_config_path(config_root, 1) 
    config = load_config(config_path)

    eeg_plus_eog(bids_root, "010", config)


def eeg_plus_eog(bids_root: str, subject_id: str, config: PipelineConfig):
    """
    Plot EOG channels as well as before and after ASR eeg data for each epoch.
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

    # TODO: Can trialrejection be done here?
    # if config.trial_rejection.enabled:
    #     print(f"\nStep 10: Trial rejection")
    #     epochs, reject_log = reject_trials(epochs, config.trial_rejection)

    #     pipeline_stats = reject_log
    #     pipeline_stats["ica_components_excluded"] = number_excluded_components

    eeg_chs = ['PO7','PO8']
    eog_chs = ['EOG5','EOG6']

    blink_intervals, durations = detect_blinks_on_raw(
        raw_before,
        eog_chs=eog_chs,
        l_freq=1.0, h_freq=15.0,
        envelope_smooth_ms=20.0,
        mad_mult=6.0
    )

    print("Detected blinks:", len(blink_intervals))
    print("mean duration of blinks:", np.mean(durations))

    fig, ax = plot_epochs_before_after(
        epochs_after,
        epochs_before,
        blink_intervals,
        picks=eeg_chs,
        eog_picks=eog_chs,
        scale=1e6
    )

def detect_blinks_on_raw(
    raw: mne.io.edf.edf.RawEDF,
    eog_chs: list[str],
    l_freq=1.0,
    h_freq=15.0,
    envelope_smooth_ms=20.0,
    mad_mult=6.0,
    min_distance_s=0.05,
    merge_gap_s=0.02,
) :
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
      (optionally) env_smooth, threshold
    """

    sfreq = raw.info['sfreq']
    eog_idx = mne.pick_channels(raw.ch_names, include=eog_chs)
    eog_data = raw.get_data(picks=eog_idx)

    # Bandpass filter for blink detection
    eog_filtered = mne.filter.filter_data(
        eog_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method='iir', verbose=False
    )

    # envelope: max absolute across eog channels
    env = np.max(np.abs(eog_filtered), axis=0)

    # smooth envelope
    win = int(round(envelope_smooth_ms * 1e-3 * sfreq))
    win = max(1, win)
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode='same')

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

def epochs_have_blinks(epochs: mne.Epochs, blink_intervals) -> np.typing.NDArray[np.bool] :
    """
    Return a boolean numpy array of length n_epochs where True means the epoch
    overlaps at least one blink interval.
    epochs: mne.Epochs
    blink_intervals: list of (start_s, end_s) in absolute seconds (same reference as epochs.events)
    """
    sfreq = epochs.info['sfreq']
    event_samples = epochs.events[:, 0]
    epoch_start_times = event_samples / sfreq + epochs.tmin
    epoch_duration = epochs.times[-1] - epochs.times[0]
    epoch_end_times = epoch_start_times + epoch_duration

    has_blink = np.zeros(len(epoch_start_times), dtype=bool)
    if not blink_intervals:
        return has_blink

    bi = np.array(blink_intervals)  
    for i, (s0, e0) in enumerate(zip(epoch_start_times, epoch_end_times)):
        overlaps = np.logical_not((bi[:,1] <= s0) | (bi[:,0] >= e0))
        if overlaps.any():
            has_blink[i] = True

    return has_blink

def plot_epochs_before_after(
    epochs_after: mne.Epochs,
    epochs_before: mne.Epochs,
    blink_intervals,
    picks: list[str], 
    eog_picks: list[str],
    scale=1e6,
    figsize=(15, 15),
):
    plot_chs = picks + eog_picks
    picks_idx = [epochs_after.ch_names.index(ch) for ch in plot_chs]

    times = epochs_after.times
    n_epochs = len(epochs_after)

    data_after = epochs_after.get_data()[:, picks_idx, :] * scale
    data_before = epochs_before.get_data()[:, picks_idx, :] * scale

    fixed_spacing = 100
    offsets = [0, fixed_spacing, 2*fixed_spacing, 3*fixed_spacing]

    event_samples = epochs_after.events[:, 0]
    sfreq = epochs_after.info['sfreq']
    epoch_start_times_raw = event_samples / sfreq + epochs_after.tmin

    has_blink = epochs_have_blinks(epochs_before, blink_intervals)

    # Plot
    current = 0
    fig, ax = plt.subplots(figsize=figsize)

    legend_handles = [
        Line2D([0], [0], color='tab:blue', linewidth=1, label='After (ASR)'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=1, label='Before'),
    ]

    def redraw(idx):
        ax.cla()
        for i, ch in enumerate(plot_chs):
            ax.plot(times, data_after[idx, i, :] + offsets[i], color='tab:blue', linewidth=0.9)
            if ch.startswith("PO"):
                ax.plot(times, data_before[idx, i, :] + offsets[i], color='orange', linestyle='--', linewidth=0.9)
        
        ax.set_yticks(offsets)
        ax.set_yticklabels(plot_chs)
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Epoch {idx+1} / {n_epochs}, with {"blink" if has_blink[idx] else "no blink"}")
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.legend(handles=legend_handles, loc='upper right')

        # add new blink shading
        t0 = epoch_start_times_raw[idx]
        t_end = t0 + (times[-1] - times[0])
        for s, e in blink_intervals:
            if e <= t0 or s >= t_end:
                continue
            ov_s = max(s, t0); ov_e = min(e, t_end)
            rel_s = ov_s - t0 + times[0]; rel_e = ov_e - t0 + times[0]
            ax.axvspan(rel_s, rel_e, color='red', alpha=0.15)

        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal current
        if event.key in ('right', 'pagedown') and current < n_epochs - 1:
            current += 1; redraw(current)
        elif event.key in ('left', 'pageup') and current > 0:
            current -= 1; redraw(current)
        elif event.key == 'home':
            current = 0; redraw(current)
        elif event.key == 'end':
            current = n_epochs - 1; redraw(current)

    fig.canvas.mpl_connect('key_press_event', on_key)
    redraw(0)
    plt.tight_layout()
    plt.show()

    return fig, ax


if __name__ == "__main__":
    main()
