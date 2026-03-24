"""Plotting utilities for blink detection analysis.

Provides functions for visualizing EOG channels, comparing epochs
before and after ASR with blink overlay, and generating grand
average ERP plots split by blink presence. Supports both
single-subject interactive inspection and multi-subject grand
average comparisons.
"""

from os import mkdir
from os.path import isdir

from mne_bids import BIDSPath
from mne import Epochs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from pipeline.step01_loading import load_data

from utils.config import PipelineConfig
from utils.utils import average_channel, pairwise_average

from blinks.blinks import (
    epochs_have_blinks,
    process_subject_with_blinkdetection,
    detect_blinks_on_raw,
)
from blinks.files import load_all_epochs


def plot_eog(bids_root: str, subject_id: str) -> None:
    """Plot raw EOG channels for manual visual inspection.

    Loads the subject's data, isolates EOG channels, and opens an
    interactive MNE plot starting at t=60s. Used to verify which
    EXG channels correspond to EOG and to visually confirm blink
    morphology.

    Args:
        bids_root (str): Root directory of the BIDS dataset.
        subject_id (str): Zero-padded subject identifier (e.g. "001").
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

    raw_eog.plot(
        scalings="auto",
        duration=10,
        start=60.0,
        title="EOG channels",
        show=True,
        block=True,
    )


def plot_average_data(
    bids_root: str,
    epochs_with_blinks: dict[str, Epochs],
    epochs_without_blinks: dict[str, Epochs],
    with_asr: bool,
) -> None:
    """Generate grand average ERP plots split by blink condition.

    Computes grand averages at PO7 and PO8 separately for
    blink-present and blink-absent epochs, averages across both
    channels, and saves two plots: one for epochs with blinks and
    one for epochs without.

    Args:
        bids_root (str): Root directory of the BIDS dataset. Used
            to derive the output folder path.
        epochs_with_blinks (dict[str, Epochs]): Mapping of subject
            index to epochs that overlap with detected blinks.
        epochs_without_blinks (dict[str, Epochs]): Mapping of subject
            index to blink-free epochs.
        with_asr (bool): Whether the epochs were processed with ASR.
            Affects the output filename and plot title.
    """
    output_folder = bids_root + "/processed_blinkdetection"

    # Grand average for blink-present epochs
    (
        data_random_po7_with_blink,
        data_regular_po7_with_blink,
        times_po7_with_blink,
        n_subjects_with_blink,
        _,
    ) = average_channel("PO7", epochs_with_blinks)
    data_random_po8_with_blink, data_regular_po8_with_blink, _, _, _ = average_channel(
        "PO8", epochs_with_blinks
    )

    # Grand average for blink-absent epochs
    (
        data_random_po7_without_blink,
        data_regular_po7_without_blink,
        times_po7_without_blink,
        n_subjects_without_blink,
        _,
    ) = average_channel("PO7", epochs_without_blinks)
    data_random_po8_without_blink, data_regular_po8_without_blink, _, _, _ = (
        average_channel("PO8", epochs_without_blinks)
    )

    # Bilateral average (PO7 + PO8) / 2
    data_random_with_blink = pairwise_average(
        data_random_po7_with_blink, data_random_po8_with_blink
    )
    data_random_without_blink = pairwise_average(
        data_random_po7_without_blink, data_random_po8_without_blink
    )
    data_regular_with_blink = pairwise_average(
        data_regular_po7_with_blink, data_regular_po8_with_blink
    )
    data_regular_without_blink = pairwise_average(
        data_regular_po7_without_blink, data_regular_po8_without_blink
    )

    # Count total epochs across all subjects
    n_epochs_with_blinks = sum(len(e) for e in epochs_with_blinks.values())
    n_epochs_without_blinks = sum(len(e) for e in epochs_without_blinks.values())

    if not isdir(output_folder):
        mkdir(output_folder)

    file = f"{output_folder}/fig-average_po7po8{"_ASR" if with_asr else ""}_with_blinks.png"

    plot_channel(
        file,
        f"{"with" if with_asr else "no"} ASR and with blinks",
        "PO7+PO8",
        data_random_with_blink,
        data_regular_with_blink,
        times_po7_with_blink,
        n_subjects_with_blink,
        n_epochs_with_blinks,
    )

    file = f"{output_folder}/fig-average_po7po8{"_ASR" if with_asr else ""}_without_blinks.png"

    plot_channel(
        file,
        f"{"with" if with_asr else "no"} ASR and without blinks",
        "PO7+PO8",
        data_random_without_blink,
        data_regular_without_blink,
        times_po7_without_blink,
        n_subjects_without_blink,
        n_epochs_without_blinks,
    )


def plot_channel(
    output_file, title, channel, data_random, data_regular, times, n_subjects, n_epochs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot grand average ERP at a channel with blink condition info.

    Similar to utils.plots.plot_channel but includes epoch count
    in the title for blink-conditioned analysis.

    Args:
        output_file (str): File path to save the plot to.
        title (str): Condition description included in the plot
            title (e.g. "with ASR and with blinks").
        channel (str): Channel name for the plot title
            (e.g. "PO7+PO8").
        data_random (np.ndarray): Grand average amplitude for the
            random condition in µV, shape (n_times,).
        data_regular (np.ndarray): Grand average amplitude for the
            regular condition in µV, shape (n_times,).
        times (np.ndarray): Time points in seconds. Converted to
            milliseconds for display.
        n_subjects (int): Number of subjects in the grand average.
        n_epochs (int): Total number of epochs across all subjects.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The input
            (data_random, data_regular, times) passed through for
            convenience.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(times * 1000, data_random, "r-", linewidth=2, label="Random")
    plt.plot(times * 1000, data_regular, "b-", linewidth=2, label="Regular")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="k", linestyle="--", linewidth=0.5)
    plt.yticks([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.title(
        f"Grand Average ERP for condition: {title} at {channel} (n={n_subjects} subjects, k={n_epochs} epochs)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")

    return data_random, data_regular, times


def plot_epochs_before_after(
    subject_id: str,
    epochs_after: Epochs,
    epochs_before: Epochs,
    blink_intervals,
    picks: list[str],
    eog_picks: list[str],
    scale=1e6,
    figsize=(15, 15),
) -> tuple[Figure, Axes]:
    """Interactive epoch viewer comparing before/after ASR with blink overlay.

    Displays EEG and EOG channels for one epoch at a time, with the
    ASR-processed data in blue and the pre-ASR data in orange (dashed,
    EEG only). Detected blink intervals are shaded in red. Navigate
    epochs with arrow keys, Page Up/Down, Home, and End.

    Args:
        subject_id (str): Zero-padded subject identifier, shown in
            the plot title.
        epochs_after (Epochs): Epochs from the pipeline branch with
            ASR applied.
        epochs_before (Epochs): Epochs from the pipeline branch
            without ASR.
        blink_intervals (list[tuple[float, float]]): Detected blink
            intervals as (start_seconds, end_seconds) pairs in
            absolute recording time.
        picks (list[str]): EEG channel names to display
            (e.g. ["PO7", "PO8"]). Both before and after traces
            are shown.
        eog_picks (list[str]): EOG channel names to display
            (e.g. ["EOG5", "EOG6"]). Only the after-ASR trace is
            shown.
        scale (float): Scaling factor applied to data for display.
            Defaults to 1e6 (Volts to µV).
        figsize (tuple[int, int]): Figure size in inches as
            (width, height). Defaults to (15, 15).

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects
            for further customization if needed.
    """
    plot_chs = picks + eog_picks
    picks_idx = [epochs_after.ch_names.index(ch) for ch in plot_chs]

    times = epochs_after.times
    n_epochs = len(epochs_after)

    data_after = epochs_after.get_data()[:, picks_idx, :] * scale
    data_before = epochs_before.get_data()[:, picks_idx, :] * scale

    fixed_spacing = 100
    offsets = [0, fixed_spacing, 2 * fixed_spacing, 3 * fixed_spacing]

    event_samples = epochs_after.events[:, 0]
    sfreq = epochs_after.info["sfreq"]
    epoch_start_times_raw = event_samples / sfreq + epochs_after.tmin

    has_blink = epochs_have_blinks(epochs_before, blink_intervals)

    # Interactive plot for each epoch to compare with/without ASR
    current = 0
    fig, ax = plt.subplots(figsize=figsize)

    legend_handles = [
        Line2D([0], [0], color="tab:blue", linewidth=1, label="After (ASR)"),
        Line2D([0], [0], color="orange", linestyle="--", linewidth=1, label="Before"),
    ]

    def redraw(idx) -> None:
        """Redraw the plot for the given epoch index."""
        ax.cla()
        for i, ch in enumerate(plot_chs):
            ax.plot(
                times,
                data_after[idx, i, :] + offsets[i],
                color="tab:blue",
                linewidth=0.9,
            )
            # Show before-ASR trace only for EEG channels
            if ch.startswith("PO"):
                ax.plot(
                    times,
                    data_before[idx, i, :] + offsets[i],
                    color="orange",
                    linestyle="--",
                    linewidth=0.9,
                )

        ax.set_yticks(offsets)
        ax.set_yticklabels(plot_chs)
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel("Time (s)")
        ax.set_title(
            f"Epoch {idx+1} / {n_epochs}, with {"blink" if has_blink[idx] else "no blink"} for subject {subject_id}"
        )
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.legend(handles=legend_handles, loc="upper right")

        # Shade blink intervals that overlap with this epoch
        t0 = epoch_start_times_raw[idx]
        t_end = t0 + (times[-1] - times[0])
        for s, e in blink_intervals:
            if e <= t0 or s >= t_end:
                continue
            ov_s = max(s, t0)
            ov_e = min(e, t_end)
            rel_s = ov_s - t0 + times[0]
            rel_e = ov_e - t0 + times[0]
            ax.axvspan(rel_s, rel_e, color="red", alpha=0.15)

        fig.canvas.draw_idle()

    def on_key(event) -> None:
        """Handle keyboard navigation between epochs."""
        nonlocal current
        if event.key in ("right", "pagedown") and current < n_epochs - 1:
            current += 1
            redraw(current)
        elif event.key in ("left", "pageup") and current > 0:
            current -= 1
            redraw(current)
        elif event.key == "home":
            current = 0
            redraw(current)
        elif event.key == "end":
            current = n_epochs - 1
            redraw(current)

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw(0)
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_eeg_plus_eog_one_subject(
    bids_root: str, subject_id: str, config: PipelineConfig
) -> None:
    """Run the pipeline and show interactive before/after blink overlay for one subject.

    Processes the subject through both pipeline branches (with and
    without ASR), detects blinks, and opens an interactive epoch
    viewer showing PO7, PO8, EOG5, and EOG6 with blink regions
    highlighted.

    Args:
        bids_root (str): Root directory of the BIDS dataset.
        subject_id (str): Zero-padded subject identifier (e.g. "001").
        config (PipelineConfig): Pipeline configuration used for
            preprocessing.
    """
    epochs_after, epochs_before, raw_after = process_subject_with_blinkdetection(
        bids_root, subject_id, config
    )

    eeg_chs = ["PO7", "PO8"]
    eog_chs = ["EOG5", "EOG6"]

    blink_intervals, durations = detect_blinks_on_raw(
        raw_after,
        eog_chs=eog_chs,
        l_freq=1.0,
        h_freq=15.0,
        envelope_smooth_ms=20.0,
        mad_mult=6.0,
    )

    print("Detected blinks:", len(blink_intervals))
    print("mean duration of blinks:", np.mean(durations))

    fig, ax = plot_epochs_before_after(
        subject_id,
        epochs_after,
        epochs_before,
        blink_intervals,
        picks=eeg_chs,
        eog_picks=eog_chs,
        scale=1e6,
    )


def all_subjects_plotting(
    bids_root: str, config: PipelineConfig, output_folder: str, with_asr: bool
) -> None:
    """Load precomputed blink-labeled epochs and generate grand average plots.

    Loads blink-present and blink-absent epochs for all subjects from
    disk, then generates grand average ERP plots split by blink
    condition.

    Args:
        bids_root (str): Root directory of the BIDS dataset.
        config (PipelineConfig): Pipeline configuration. Currently
            unused but passed for consistency.
        output_folder (str): Directory containing the precomputed
            blink-labeled epoch files.
        with_asr (bool): Whether to load epochs from the ASR-enabled
            pipeline branch.
    """
    epochs_with_blinks, epochs_without_blinks = load_all_epochs(
        bids_root, output_folder, with_asr
    )
    plot_average_data(bids_root, epochs_with_blinks, epochs_without_blinks, with_asr)
