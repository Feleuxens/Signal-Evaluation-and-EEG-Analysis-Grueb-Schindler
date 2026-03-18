from time import time
from os import mkdir, getenv
from os.path import isdir
from mne_bids import BIDSPath
import mne
from mne.io.edf.edf import RawEDF
from mne.preprocessing import ICA
import numpy as np
from numpy.typing import NDArray
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
from pipeline.step10_trialrejection import reject_trials

from utils.config import load_config, PipelineConfig
from utils.utils import get_config_path, get_subject_list

BASELINE = (-0.25, 0.0)  # baseline correction period



def main():
    bids_root = getenv("BIDS_ROOT", "../data/")
    bids_root = bids_root.rstrip("/")
    config_root = getenv("CONFIG_ROOT", "../config/")
    config_root = config_root.rstrip("/")

    # load default config
    config_path = get_config_path(config_root, 1) 
    config = load_config(config_path)


    # blink_threshold(bids_root, subject_id = "005")
    # plot_eog(bids_root, subject_id = "005")
    # plot_multiple(bids_root, subject_ids = ["005", "006"])
    eeg_plus_eog(bids_root, "010", config)


def eeg_plus_eog(bids_root: str, subject_id: str, config: PipelineConfig):
    """
    Plot eeg channels PO7 and PO8 next to the two EOG channels.
    Mark epochs and detected blinks to be checked for validity.
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

    if config.asr.enabled:
        print(f"\nStep 06: Artifact correction")
        raw, asr = run_asr(raw, config.asr)

    # number_excluded_components = None
    ica: ICA | None = None
    if config.ica.enabled:
        print(f"\nStep 07: ICA cleaning")
        raw, ica, number_excluded_components = run_ica(raw, config.ica)
        raw_before, ica, number_excluded_components = run_ica(raw_before, config.ica)

    if config.interpolation.enabled:
        print(f"\nStep 08: Interpolating bad channels")
        raw = interpolate_bad_channels(raw, config.interpolation)
        raw_before = interpolate_bad_channels(raw_before, config.interpolation)

    print(f"\nStep 09: Epoching")
    epochs, _, _ = epoch_data(raw, bids_path, config.epoching)
    epochs_before, _, _ = epoch_data(raw_before, bids_path, config.epoching)

    # pipeline_stats = None
    # if config.trial_rejection.enabled:
    #     print(f"\nStep 10: Trial rejection")
    #     epochs, reject_log = reject_trials(epochs, config.trial_rejection)

    #     pipeline_stats = reject_log
    #     pipeline_stats["ica_components_excluded"] = number_excluded_components

    # plot_epochs_with_blinks(epochs, ["PO7", "PO8", "EOG5", "EOG6"], ["EOG5", "EOG6"])

    blink_intervals, durations = detect_blinks_on_raw(
        raw,
        eog_chs=['EOG5','EOG6'],
        l_freq=1.0, h_freq=15.0,
        envelope_smooth_ms=20.0,
        mad_mult=6.0
    )
    print("Detected blinks:", len(blink_intervals))
    print("mean duration of blinks:", np.mean(durations))

    # then plot epochs shading detected blink intervals
    fig, ax = plot_epochs_before_after(
        epochs,
        epochs_before,
        blink_intervals,
        picks=['PO7','PO8'],
        eog_picks=['EOG5','EOG6'],
        scale=1e6
    )

def detect_blinks_on_raw(
   raw,
    eog_chs=None,
    l_freq=1.0,
    h_freq=15.0,
    envelope_smooth_ms=20.0,
    mad_mult=6.0,
    min_distance_s=0.05,
    merge_gap_s=0.02,
    return_env=False,
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
      (optionally) env_smooth, threshold
    """


    sfreq = raw.info['sfreq']
    if eog_chs is None:
        eog_chs = [ch for ch in raw.ch_names if ch.upper().startswith('EOG')]
    if not eog_chs:
        raise ValueError("No EOG channels found; pass eog_chs explicitly.")

    eog_idx = mne.pick_channels(raw.ch_names, include=eog_chs)
    eog_data = raw.get_data(picks=eog_idx)  # (n_eog, n_times)

    # bandpass
    eog_filt = mne.filter.filter_data(
        eog_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method='iir', verbose=False
    )

    # envelope: max absolute across eog channels
    env = np.max(np.abs(eog_filt), axis=0)

    # smooth envelope
    win = int(round(envelope_smooth_ms * 1e-3 * sfreq))
    win = max(1, win)
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode='same')

    # robust baseline & threshold
    med = np.median(env_smooth)
    mad = np.median(np.abs(env_smooth - med))
    thresh = med + mad_mult * mad            # peak detection threshold
    baseline_level = med + 0.5 * mad         # return-to-baseline crossing level

    # find peaks above thresh
    min_dist = int(round(min_distance_s * sfreq))
    peaks, props = signal.find_peaks(env_smooth, height=thresh, distance=min_dist)

    intervals = []
    n = env_smooth.size
    for p in peaks:
        # expand left until envelope <= baseline_level
        s = p
        while s > 0 and env_smooth[s] > baseline_level:
            s -= 1
        # expand right until envelope <= baseline_level
        e = p
        while e < n - 1 and env_smooth[e] > baseline_level:
            e += 1
        intervals.append((max(0, s), min(n - 1, e)))

    # convert to seconds and merge nearby/overlapping intervals
    intervals_s = [(s / sfreq, e / sfreq) for s, e in intervals]
    if not intervals_s:

        return [], np.array([])

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

def epochs_have_blinks(epochs, blink_intervals):
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
    epochs_after,          # clean epochs (after ASR)
    epochs_before,         # raw epochs (before ASR)
    blink_intervals,
    picks: list[str],      # EEG channels to plot, e.g., ['PO7','PO8']
    eog_picks: list[str],  # EOG channels to plot, e.g., ['EOG5','EOG6']
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

    # epoch absolute start times (for blink shading)
    event_samples = epochs_after.events[:, 0]
    sfreq = epochs_after.info['sfreq']
    epoch_start_times_raw = event_samples / sfreq + epochs_after.tmin

    has_blink = epochs_have_blinks(epochs_before, blink_intervals)

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

def plot_epochs_scrollable_matplotlib(epochs, picks: list[str], page_duration=None, scale=1.0):

    picks_idx = [epochs.ch_names.index(ch) for ch in picks]
    sfreq = epochs.info['sfreq']
    n_epochs = len(epochs)
    times = epochs.times  # shape (n_times,)

    # prepare data: shape (n_epochs, n_ch, n_times)
    data = epochs.get_data()[:, picks_idx, :] * scale

    data = data * 1e6

    # compute spacing using ptp per channel across all epochs
    ptp = np.ptp(data, axis=2).max(axis=0)  # peak-to-peak per channel (max across epochs)
    min_spacing = np.maximum(ptp * 1.2, np.maximum(ptp.max()*0.1, 1.0))
    offsets = np.cumsum(np.concatenate(([0.0], min_spacing[:-1])))

    current_epoch = 0
    fig, ax = plt.subplots(figsize=(12, 6))
    lines = []
    for i, ch in enumerate(picks):
        ln, = ax.plot(times, data[current_epoch, i, :] + offsets[i], label=ch, linewidth=0.8)
        lines.append(ln)

    ax.set_yticks(offsets)
    ax.set_yticklabels(picks)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Epoch 1 / {n_epochs}")
    ax.grid(True, linewidth=0.3, alpha=0.6)

    def redraw(epoch_idx):
        for i, ln in enumerate(lines):
            ln.set_ydata(data[epoch_idx, i, :] + offsets[i])
        ax.set_title(f"Epoch {epoch_idx+1} / {n_epochs}")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal current_epoch
        if event.key in ('right', 'pagedown'):
            if current_epoch < n_epochs - 1:
                current_epoch += 1
                redraw(current_epoch)
        elif event.key in ('left', 'pageup'):
            if current_epoch > 0:
                current_epoch -= 1
                redraw(current_epoch)
        elif event.key == 'home':
            current_epoch = 0
            redraw(current_epoch)
        elif event.key == 'end':
            current_epoch = n_epochs - 1
            redraw(current_epoch)
        elif event.key == 'd':  # optionally drop DC of current epoch
            # toggle mean subtraction for visible shape
            y = data[current_epoch] - data[current_epoch].mean(axis=1, keepdims=True)
            for i, ln in enumerate(lines):
                ln.set_ydata(y[i] + offsets[i])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_eog(bids_root: str, subject_id: str):
    """
    Plot presumable EOG channels of the subject.
    This is for manual inspection only.
    No further analysis is done here.
    """

    bids_path = BIDSPath(
        subject=subject_id,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
        task="jacobsen",
    )

    raw = load_data(bids_path)
    raw_eog = raw.pick_types(eog=True)

    raw_eog.plot(scalings='auto', duration=10, start=60.0,
                title='EOG channels', show=True, block=True)



def blink_threshold(bids_root, subject_id):
    """
    Plot a wide view of the EOG channels with markings, where the algorithm assumes a blink to be.
    This is for plotting for the report.
    """
    # Parameters for blink detection
    l_freq, h_freq = 1.0, 15.0          # bandpass for blink detection
    threshold_uv = 100.0                # µV threshold for peak detection
    min_distance_s = 0.1                # minimum distance between peaks (s)
    
    bids_path = BIDSPath(
        subject=subject_id,
        root=bids_root,
        datatype="eeg",
        suffix="eeg",
        task="jacobsen",
    )

    raw = load_data(bids_path)
    raw_eog: RawEDF = raw.pick_types(eog=True)

    eog_ch_names = raw_eog.ch_names

    plot_seconds = 100                 
    plot_start_s = 60.0             
    sfreq = raw.info['sfreq']    

    eog_data = raw_eog.get_data()[0]             
    n_samples = eog_data.shape[0]

    eog_filt = mne.filter.filter_data(
        eog_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method='iir'
    )

    # Convert threshold to Volts
    threshold_v = threshold_uv * 1e-6

    abs_eog = np.abs(eog_filt)
    min_distance_samples = int(min_distance_s * sfreq)
    peaks, props = signal.find_peaks(abs_eog, height=threshold_v, distance=min_distance_samples)

    blink_half_width_s = 0.1
    blink_half_width_samples = int(blink_half_width_s * sfreq)

    blink_mask = np.zeros(n_samples, dtype=bool)
    for p in peaks:
        start = max(0, p - blink_half_width_samples)
        end = min(n_samples, p + blink_half_width_samples + 1)
        blink_mask[start:end] = True

    times = np.arange(n_samples) / sfreq  # in seconds

    start_sample = int(round(plot_start_s * sfreq))
    end_sample = start_sample + int(round(plot_seconds * sfreq))

    times_plot = times[start_sample:end_sample]
    eog_plot = eog_filt[start_sample:end_sample]
    mask_plot = blink_mask[start_sample:end_sample]

    eog_ch_name = eog_ch_names[0]

    plt.figure(figsize=(12, 4))
    plt.plot(times_plot, eog_plot * 1e6, color='C0', label=f'{eog_ch_name} (filtered)')  # convert to µV for display

    # Mark detected peak positions within the plotted window
    peak_mask_in_plot = (peaks >= start_sample) & (peaks < end_sample)
    peak_times_plot = peaks[peak_mask_in_plot] / sfreq
    peak_vals_plot = eog_filt[peaks[peak_mask_in_plot]] * 1e6
    plt.plot(peak_times_plot, peak_vals_plot, 'rx', label='Detected blink peaks')

    # Shade blink regions (contiguous True segments in mask_plot)
    in_blink = False
    start_t = None
    for i, val in enumerate(mask_plot):
        if val and not in_blink:
            in_blink = True
            start_t = times_plot[i]
        elif not val and in_blink:
            end_t = times_plot[i]
            plt.axvspan(start_t, end_t, color='red', alpha=0.15)
            in_blink = False
    if in_blink:
        plt.axvspan(start_t, times_plot[-1], color='red', alpha=0.15)

    plt.xlabel('Time (s)')
    plt.ylabel(f'{eog_ch_name} (µV)')
    plt.title(f'EOG blink detection on subject {subject_id} and channel {eog_ch_name} — window {times_plot[0]:.1f}s to {times_plot[-1]:.1f}s')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()






def plot_multiple(bids_root="../data/", subject_ids = ["003", "004"]):
    # === User parameters ===
    eog_ch_name = "EOG5"
    l_freq, h_freq = 1.0, 15.0
    threshold_uv = 100.0
    min_distance_s = 0.25
    blink_half_width_s = 0.1
    poi_channels = ["PO7", "PO8"]
    t_baseline = (None, 0)
    plot_time_window = (-0.2, 0.8)

    # === Helpers ===
    def _detect_blinks_in_raw(raw)-> NDArray[np.bool_]:
        raw_eog: RawEDF = raw.pick_types(eog=True)
        eog_ch_names = raw_eog.ch_names
        eog_ch_name = eog_ch_names[0]

        sfreq = raw_eog.info["sfreq"]
        eog_data = raw_eog.get_data()[0]
        eog_filt = mne.filter.filter_data(eog_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method="iir")
        threshold_v = threshold_uv * 1e-6
        abs_eog = np.abs(eog_filt)
        min_distance_samples = int(min_distance_s * sfreq)
        peaks, _ = signal.find_peaks(abs_eog, height=threshold_v, distance=min_distance_samples)
        n_samples = eog_data.shape[0]
        blink_half_width_samples = int(blink_half_width_s * sfreq)
        blink_mask = np.zeros(n_samples, dtype=bool)
        for p in peaks:
            start = max(0, p - blink_half_width_samples)
            end = min(n_samples, p + blink_half_width_samples + 1)
            blink_mask[start:end] = True

        return blink_mask

    groups = ["reg_blink", "reg_noblink", "rnd_blink", "rnd_noblink"]
    grand_means = {g: [] for g in groups}

    for subj in subject_ids:


        bids_path = BIDSPath(
            subject=subj,
            root=bids_root,
            datatype="eeg",
            suffix="eeg",
            task="jacobsen",
        )

        # 01) Load data
        raw = load_data(bids_path)

        # 02) Automatic detection of bad channels
        print(f"\n\nStep 02: Detecting bad channels")
        raw = detect_bad_channels(raw)

        # 03) Filtering
        print(f"\n\nStep 03: Filtering")
        raw = filter_data(raw)

        # 04) Downsampling
        print(f"\n\nStep 04: Downsampling")
        raw = downsample_data(raw)

        # 05) Rereference
        print(f"\n\nStep 05: Rereferencing")
        raw = rereference_data(raw)

        # 06) Artifact correction
        print(f"\n\nStep 06: Artifact correction")
        raw, asr = run_asr(raw)

        # 07) ICA cleaning
        print(f"\n\nStep 07: ICA cleaning")
        raw, ica, number_excluded_components = run_ica(raw)

        # 08) Interpolate bad channels
        print(f"\n\nStep 08: Interpolating bad channels")
        raw = interpolate_bad_channels(raw)

        # 09) Epoching
        print(f"\n\nStep 09: Epoching")
        epochs, events, event_dict = epoch_data(raw, bids_path, baseline=BASELINE)

        blink_maks= detect_blinks_in_raw(raw)
        blink_samples = np.where(blink_mask)[0]

        sfreq = raw.info["sfreq"]
        epoch_sample_starts = epochs.events[:, 0] + int(round(epochs.tmin * sfreq))
        epoch_sample_ends = epochs.events[:, 0] + int(round(epochs.tmax * sfreq))
        epoch_has_blink = np.array([
            np.any((blink_samples >= start) & (blink_samples <= end))
            for start, end in zip(epoch_sample_starts, epoch_sample_ends)
        ])

        # conditions
        if epochs.metadata is not None and "condition" in epochs.metadata.columns:
            conditions = epochs.metadata["condition"].astype(str).str.lower().values
        else:
            if isinstance(epochs.event_id, dict):
                inv_map = {v: k for k, v in epochs.event_id.items()}
                event_codes = epochs.events[:, 2]
                conditions = np.array([inv_map.get(code, str(code)).lower() for code in event_codes])
            else:
                conditions = np.array(["unknown"] * len(epochs))

        is_regular = np.array(["regular" in c for c in conditions])
        is_random = np.array(["random" in c for c in conditions])
        if not is_regular.any() and not is_random.any():
            n = len(epochs)
            is_regular = np.zeros(n, dtype=bool)
            is_regular[: n // 2] = True
            is_random = ~is_regular

        regular_idx = np.where(is_regular)[0]
        random_idx = np.where(is_random)[0]

        reg_blink_idx = regular_idx[epoch_has_blink[regular_idx]]
        reg_noblink_idx = regular_idx[~epoch_has_blink[regular_idx]]

        rnd_blink_idx = random_idx[epoch_has_blink[random_idx]]
        rnd_noblink_idx = random_idx[~epoch_has_blink[random_idx]]

        # create evokeds and store PO7+PO8 mean time series
        def evoked_poi_mean_from_idx(idxs):
            if len(idxs) == 0:
                return None
            sub = epochs[idxs].copy().pick_channels(poi_channels)
            sub.apply_baseline(t_baseline)
            ev = sub.average()
            return ev.data.mean(axis=0) * 1e6  # µV

        m = evoked_poi_mean_from_idx(reg_blink_idx)
        if m is not None:
            grand_means["reg_blink"].append(m)
        m = evoked_poi_mean_from_idx(reg_noblink_idx)
        if m is not None:
            grand_means["reg_noblink"].append(m)
        m = evoked_poi_mean_from_idx(rnd_blink_idx)
        if m is not None:
            grand_means["rnd_blink"].append(m)
        m = evoked_poi_mean_from_idx(rnd_noblink_idx)
        if m is not None:
            grand_means["rnd_noblink"].append(m)

    # Compute grand average (across subjects) for each group
    times = epochs.times
    grand_avgs = {}
    for g in groups:
        if len(grand_means[g]) == 0:
            grand_avgs[g] = None
        else:
            arr = np.vstack(grand_means[g])
            grand_avgs[g] = arr.mean(axis=0)

    plt.figure(figsize=(8, 5))
    def plot_if_present(ts, label, color, linestyle='-'):
        if ts is None:
            return
        mask = (times >= plot_time_window[0]) & (times <= plot_time_window[1])
        plt.plot(times[mask], ts[mask], label=label, color=color, linestyle=linestyle)

    plot_if_present(grand_avgs["reg_blink"], f"regular + blink {len(grand_avgs["reg_blink"])} samples", "C0")
    plot_if_present(grand_avgs["reg_noblink"], f"regular + no blink {len(grand_avgs["reg_noblink"])} samples", "C0", linestyle='--')
    plot_if_present(grand_avgs["rnd_blink"], f"random + blink {len(grand_avgs["rnd_blink"])} samples", "C1")
    plot_if_present(grand_avgs["rnd_noblink"], f"random + no blink {len(grand_avgs["rnd_noblink"])} samples", "C1", linestyle='--')

    plt.axvline(0, color="k", linewidth=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("PO7+PO8 (µV)")
    plt.title("Grand average PO7+PO8 by condition and blink presence")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
