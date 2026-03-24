"""Plotting utilities for EEG pipeline visualization.

Provides functions for generating ERP plots, PSD plots, ICA
topography maps, butterfly plots, topomaps, and before/after
comparisons of preprocessing steps.
"""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mne import Evoked
import mne
import numpy as np
from mne.io import Raw

from utils.utils import evoke_channels


def power_spectral_density_plot(output_file, raw, fmin, fmax) -> None:
    """Plot sensor-level power spectral density using Welch's method.

    Computes and saves an average PSD across all EEG channels in the
    specified frequency range.

    Args:
        output_file (str): File path to save the plot to.
        raw (mne.io.Raw): Continuous EEG data.
        fmin (float): Lower frequency bound in Hz.
        fmax (float): Upper frequency bound in Hz.
    """
    # Sensor-level PSD (Welch) — full-band and zoomed alpha/beta
    fig_psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=2048).plot(
        average=True, picks="eeg", show=False
    )
    fig_psd.suptitle(f"Sensor PSD ({fmin}-{fmax} Hz)")
    plt.savefig(output_file, bbox_inches="tight")


def ica_topography_plot(output_file, ica, raw) -> None:
    """Plot ICA component topographies.

    Fits ICA on a copy of the raw data with EXG channels removed,
    then plots up to 20 component topographies. Saves one or more
    image files depending on how many components are plotted.

    Args:
        output_file (str): Base file path for saving plots. Page
            indices and ".png" are appended automatically.
        ica (mne.preprocessing.ICA | None): Fitted ICA object, or
            None to skip plotting.
        raw (mne.io.Raw): Continuous EEG data used for fitting.
    """
    if ica is None:
        return

    # drop EXG from raw copy, then fit ICA
    raw_no_exg = raw.copy().drop_channels(
        [c for c in raw.ch_names if c.startswith("EXG")]
    )

    ica.fit(raw_no_exg)

    comp_picks = list(range(min(20, ica.n_components_)))
    figures: list[Figure] | Any = ica.plot_components(
        picks=comp_picks, inst=raw_no_exg, show=False
    )
    if isinstance(figures, list):
        for index, fig in enumerate(figures):
            fig.savefig(output_file + str(index) + ".png", bbox_inches="tight")
    if isinstance(figures, Figure):
        figures.savefig(output_file + ".png", bbox_inches="tight")


def one_channel_erp_plot(output_file, raw, epochs, baseline) -> None:
    """Plot single-channel ERP comparing random vs regular conditions.

    Selects PO7 by default, falling back to other posterior channels
    if unavailable. Plots both conditions overlaid with the baseline
    window shaded.

    Args:
        output_file (str): File path to save the plot to.
        raw (mne.io.Raw): Continuous EEG data (used to check channel
            availability).
        epochs (mne.Epochs): Epoched data containing "random" and
            "regular" conditions.
        baseline (list[float]): Two-element list [start, end] in
            seconds defining the baseline window, shown as a shaded
            region on the plot.
    """
    evoked_random, evoked_regular = evoke_channels(epochs)

    # choose channel
    preferred = "PO7"
    if preferred not in raw.ch_names:
        # pick a posterior channel if available
        for ch in ["POz", "Oz", "P3", "P4", "O1", "O2"]:
            if ch in raw.ch_names:
                preferred = ch
                break
    # get single-channel evoked (random vs regular)
    evoked_ch_random = evoked_random.copy().pick([preferred])
    evoked_ch_regular = evoked_regular.copy().pick([preferred])
    plt.figure()
    plt.plot(
        evoked_ch_random.times,
        evoked_ch_random.data.T * 1e6,
        label=f"Random — {preferred}",
    )
    plt.plot(
        evoked_ch_regular.times,
        evoked_ch_regular.data.T * 1e6,
        label=f"Regular — {preferred}",
    )
    plt.axvspan(baseline[0], baseline[1], color="gray", alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"ERP at {preferred}")
    plt.legend()
    plt.savefig(output_file, bbox_inches="tight")


def all_channel_erp_plot(output_file, epochs, baseline) -> None:
    """Plot mean ERP across all channels for random vs regular conditions.

    Averages amplitude across all channels at each time point and
    plots both conditions overlaid.

    Args:
        output_file (str): File path to save the plot to.
        epochs (mne.Epochs): Epoched data containing "random" and
            "regular" conditions.
        baseline (list[float]): Two-element list [start, end] in
            seconds defining the baseline window, shown as a shaded
            region on the plot.
    """
    evoked_random, evoked_regular = evoke_channels(epochs)

    # For type checking reasons
    if type(evoked_random) is not mne.evoked.EvokedArray:
        raise ValueError("evoked_random is not an instance of mne.evoked.EvokedArray")

    if type(evoked_regular) is not mne.evoked.EvokedArray:
        raise ValueError("evoked_regular is not an instance of mne.evoked.EvokedArray")

    # simple channel-mean time series plot
    times = evoked_random.times
    mean_random = evoked_random.data.mean(axis=0)
    mean_regular = evoked_regular.data.mean(axis=0)

    plt.plot(times, mean_random * 1e6, label="Random")  # convert to µV if data in V
    plt.plot(times, mean_regular * 1e6, label="Regular")
    plt.axvspan(baseline[0], baseline[1], color="gray", alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.title("Mean across channels — Random vs Regular")
    plt.savefig(output_file, bbox_inches="tight")


def unprocessed_vs_processed_plot(raw_unprocessed: Raw, raw: Raw) -> None:
    """Plot normalized overlay of unprocessed vs processed EEG traces.

    Selects up to 10 channels and plots 10 consecutive time windows,
    each 10 seconds long, starting at t=100s. Unprocessed data is
    shown in black, processed in red. Each channel is normalized
    to unit range for visual comparison.

    Args:
        raw_unprocessed (mne.io.Raw): Original continuous data before
            any preprocessing.
        raw (mne.io.Raw): Continuous data after the full pipeline.
    """
    # select up to 10 channels (change list if desired)
    candidates = ["Fp1", "Fp2", "AF7", "AF8", "Fz", "Cz", "Pz", "Oz", "O1", "O2"]
    picked = [c for c in candidates if c in raw.ch_names]
    if len(picked) < 10:
        eeg_chs = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
        for ch in eeg_chs:
            if ch not in picked:
                picked.append(ch)
            if len(picked) == 10:
                break
    picked = picked[:10]

    # time window (s)
    tstart, duration = 100.0, 10.0

    # compute demeaned, unit-range signals per channel
    def norm(x):
        x = x - x.mean(axis=1, keepdims=True)
        rng = x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)
        rng[rng == 0] = 1.0
        return x / rng

    for i in range(10):
        tmin = tstart + i * duration
        sf = raw.info["sfreq"]
        start_idx = int(tmin * sf)
        stop_idx = int((tmin + duration) * sf)
        times = np.linspace(tmin, tmin + duration, stop_idx - start_idx)

        # get data (channels x times)
        data_un = (
            raw_unprocessed.copy().pick(picked).get_data(start=start_idx, stop=stop_idx)
        )

        data_proc = raw.copy().pick(picked).get_data(start=start_idx, stop=stop_idx)

        # scale to µV
        data_un_uV = data_un * 1e6
        data_proc_uV = data_proc * 1e6

        un = norm(data_un_uV)
        pr = norm(data_proc_uV)

        spacing = 1.5  # small spacing after normalization
        offsets = np.arange(len(picked))[::-1] * spacing

        plt.figure(figsize=(12, 6))
        for j, ch in enumerate(picked):
            off = offsets[i]
            plt.plot(times, un[j] + off, color="k", linewidth=0.6)
            plt.plot(times, pr[j] + off, color="r", linewidth=0.8)
            plt.text(times[0] - duration * 0.01, off, ch, va="center", fontsize=9)
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.title("Normalized: unprocessed (black) vs processed (red)")
        plt.show()


def butterfly_plot(output_file, epochs) -> None:
    """Plot butterfly ERP for the random condition and the random-regular difference.

    Saves two plots: one for the random condition alone, and one for
    the difference wave (random minus regular).

    Args:
        output_file (str): Base file path. "_random.png" and
            "_combined.png" are appended for the two plots.
        epochs (mne.Epochs): Epoched data containing "random" and
            "regular" conditions.
    """
    evoked_random, evoked_regular = evoke_channels(epochs)

    # Butterfly plot for evoked difference or single condition
    figure: Figure = evoked_random.plot(
        spatial_colors=False,  # pyright: ignore[reportArgumentType]
        show=False,
        time_unit="s",
        titles=f"Butterfly — Random",
    )
    figure.savefig(output_file + "_random.png", bbox_inches="tight")
    # Alternatively plot difference
    evoked_diff = mne.combine_evoked([evoked_random, evoked_regular], weights=[1, -1])
    figure2: Figure = evoked_diff.plot(
        spatial_colors=False,
        show=False,
        time_unit="s",
        titles=f"Butterfly — Random-Regular diff",
    )
    figure2.savefig(output_file + "_combined.png", bbox_inches="tight")


def plot_channel(
    output_file,
    channel,
    data_random,
    data_regular,
    times,
    n_subjects,
):
    """Plot grand average ERP at a single channel across subjects.

    Shows random and regular conditions overlaid with reference
    lines at zero amplitude and zero time.

    Args:
        output_file (str): File path to save the plot to.
        channel (str): Channel name used in the plot title
            (e.g. "PO7", "PO7+PO8").
        data_random (np.ndarray): Grand average amplitude for the
            random condition in µV, shape (n_times,).
        data_regular (np.ndarray): Grand average amplitude for the
            regular condition in µV, shape (n_times,).
        times (np.ndarray): Time points in seconds. Converted to
            milliseconds for display.
        n_subjects (int): Number of subjects included in the grand
            average, shown in the plot title.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(times * 1000, data_random, "r-", linewidth=2, label="Random")
    plt.plot(times * 1000, data_regular, "b-", linewidth=2, label="Regular")
    plt.axhline(0, color="k", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="k", linestyle="--", linewidth=0.5)
    plt.yticks([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"Grand Average ERP at {channel} (n={n_subjects} subjects)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")


def plot_topomap(output_file, evoked_diff: Evoked) -> None:
    """Plot scalp topography maps for three ERP time windows.

    Shows the spatial distribution of the random-regular difference
    wave at P1 (100–130 ms), N1 (170–200 ms), and SPN (300–1000 ms)
    time windows. EXG channels are dropped before plotting.

    Args:
        output_file (str): File path to save the combined figure to.
        evoked_diff (Evoked): Difference wave (random minus regular)
            evoked object.
    """
    evoked_diff = evoked_diff.copy().drop_channels(
        [c for c in evoked_diff.ch_names if c.startswith("EXG")]
    )  # pyright: ignore[reportAssignmentType]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    time_windows = {
        "100–130 ms": (0.100, 0.130),
        "170–200 ms": (0.170, 0.200),
        "300–1000 ms": (0.300, 1.000),
    }

    for ax, (title, (tmin, tmax)) in zip(axes, time_windows.items()):
        evoked_diff.plot_topomap(
            times=[(tmin + tmax) / 2],  # pyright: ignore[reportArgumentType]
            average=tmax - tmin,
            axes=ax,
            show=False,
            colorbar=False,
            vlim=(3, -3),
            cmap="RdYlBu_r",
        )
        ax.set_title(title)

    fig.colorbar(axes[-1].images[0], ax=axes, shrink=0.6, label="µV")
    plt.savefig(output_file, bbox_inches="tight")
