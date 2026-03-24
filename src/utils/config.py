"""Pipeline configuration dataclasses and TOML loader.

Defines a dataclass per pipeline step, each holding the parameters
for that step along with sensible defaults. The top-level
PipelineConfig groups all steps and can be constructed from a
sectioned TOML file via load_config().

Typical usage:
    config = load_config("configs/default.toml")
    if config.filtering.enabled:
        raw = filter_data(raw, config.filtering)
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from tomllib import load
from typing import Any, Dict


@dataclass
class StepBadChannels:
    """Configuration for automatic bad channel detection (Step 02).

    Attributes:
        enabled (bool): Whether to run this step.
        z_thresh (float): Z-score threshold for marking channels as bad.
            Channels with variance or correlation z-scores exceeding
            this value are flagged.
        exg_prefix (str): Channel name prefix to exclude from detection
            (e.g. external electrode channels).
        status_prefix (str): Channel name prefix to exclude from detection
            (e.g. BioSemi status/trigger channels).
    """

    enabled: bool = True
    z_thresh: float = 3.0
    exg_prefix: str = "EXG"
    status_prefix: str = "Status"


@dataclass
class StepFiltering:
    """Configuration for bandpass and notch filtering (Step 03).

    Attributes:
        enabled (bool): Whether to run this step.
        pass_filter_enabled (bool): Whether to apply the bandpass filter.
        low_pass (float): Low-pass cutoff frequency in Hz.
        high_pass (float): High-pass cutoff frequency in Hz.
        pass_filter_pick (str): Channel type to apply the bandpass
            filter to.
        pass_filter_method (str): FIR design method passed to MNE's
            filter().
        notch_filter_enabled (bool): Whether to apply the notch filter.
        notch_frequencies (list[float]): Frequencies to notch out in Hz
            (typically line noise harmonics).
        notch_filter_pick (str): Channel type to apply the notch
            filter to.
        notch_filter_method (str): Notch filter method passed to MNE's
            notch_filter().
    """

    enabled: bool = True
    pass_filter_enabled: bool = True
    low_pass: float = 40.0
    high_pass: float = 0.1
    pass_filter_pick: str = "eeg"
    pass_filter_method: str = "fir"
    notch_filter_enabled: bool = True
    notch_frequencies: list[float] = field(default_factory=lambda: [50, 100, 150, 200])
    notch_filter_pick: str = "eeg"
    notch_filter_method: str = "spectrum_fit"


@dataclass
class StepDownsampling:
    """Configuration for downsampling (Step 04).

    Attributes:
        enabled (bool): Whether to run this step.
        target_sfreq (float): Target sampling frequency in Hz.
        npad (str): Padding mode passed to MNE's resample(). "auto"
            lets MNE choose an efficient FFT length.
    """

    enabled: bool = True
    target_sfreq: float = 128.0
    npad: str = "auto"


@dataclass
class StepRereferencing:
    """Configuration for re-referencing (Step 05).

    Attributes:
        enabled (bool): Whether to run this step.
        ref_channels (str): Reference channel(s). Use "average" for
            average reference, or a channel name / list for a specific
            reference.
        projection (bool): If True, apply the reference as a projection
            rather than modifying the data in place.
    """

    enabled: bool = True
    ref_channels: str = "average"
    projection: bool = False


@dataclass
class StepASR:
    """Configuration for Artifact Subspace Reconstruction (Step 06).

    Attributes:
        enabled (bool): Whether to run this step.
        cutoff (int): ASR rejection threshold in standard deviations.
            Lower values are more aggressive (typical range: 5–20).
    """

    enabled: bool = True
    cutoff: int = 10


@dataclass
class StepICA:
    """Configuration for Independent Component Analysis (Step 07).

    Attributes:
        enabled (bool): Whether to run this step.
        n_components (float): Number of ICA components. If float < 1,
            interpreted as the fraction of variance to retain in PCA
            before ICA. Overridden at runtime by data rank when
            necessary.
    """

    enabled: bool = True
    n_components: float = 0.99


@dataclass
class StepInterpolation:
    """Configuration for bad channel interpolation (Step 08).

    Attributes:
        enabled (bool): Whether to run this step.
        reset_bads (bool): If True, clear the bads list after
            interpolation.
        mode (str): Interpolation mode passed to MNE's
            interpolate_bads(). "accurate" uses spherical splines;
            "fast" uses a simpler method.
    """

    enabled: bool = True
    reset_bads: bool = False
    mode: str = "accurate"


@dataclass
class StepEpoching:
    """Configuration for epoching and baseline correction (Step 09).

    Attributes:
        epochrange_tmin (float): Start of the epoch relative to stimulus
            onset in seconds.
        epochrange_tmax (float): End of the epoch relative to stimulus
            onset in seconds.
        baseline (list[float]): Time window for baseline correction in
            seconds, as [start, end]. Mean amplitude in this window is
            subtracted from the entire epoch.
    """

    epochrange_tmin: float = -0.5
    epochrange_tmax: float = 1.0
    baseline: list[float] = field(default_factory=lambda: [-0.25, 0.0])


@dataclass
class StepTrialRejection:
    """Configuration for amplitude-based trial rejection (Step 10).

    Attributes:
        enabled (bool): Whether to run this step.
        eeg_threshold (float): Peak-to-peak amplitude threshold for EEG
            channels in Volts. Epochs exceeding this are rejected.
        eog_threshold (float): Peak-to-peak amplitude threshold for EOG
            channels in Volts.
        flat_threshold (float): Minimum peak-to-peak amplitude in Volts.
            Epochs flatter than this are rejected (likely disconnected
            channels).
    """

    enabled: bool = True
    eeg_threshold: float = 150e-6
    eog_threshold: float = 250e-6
    flat_threshold: float = 1e-6


@dataclass
class PipelineConfig:
    """Top-level configuration grouping all pipeline steps.

    Each field holds the config for one step. Steps not present in
    the TOML file keep their default values, allowing partial configs
    that only override what changes.

    Attributes:
        bad_channels (StepBadChannels): Bad channel detection settings.
        filtering (StepFiltering): Bandpass and notch filter settings.
        downsampling (StepDownsampling): Resampling settings.
        rereferencing (StepRereferencing): Re-referencing settings.
        asr (StepASR): Artifact Subspace Reconstruction settings.
        ica (StepICA): ICA artifact removal settings.
        interpolation (StepInterpolation): Bad channel interpolation
            settings.
        epoching (StepEpoching): Epoching and baseline correction
            settings.
        trial_rejection (StepTrialRejection): Amplitude-based epoch
            rejection settings.
    """

    bad_channels: StepBadChannels = field(default_factory=StepBadChannels)
    filtering: StepFiltering = field(default_factory=StepFiltering)
    downsampling: StepDownsampling = field(default_factory=StepDownsampling)
    rereferencing: StepRereferencing = field(default_factory=StepRereferencing)
    asr: StepASR = field(default_factory=StepASR)
    ica: StepICA = field(default_factory=StepICA)
    interpolation: StepInterpolation = field(default_factory=StepInterpolation)
    epoching: StepEpoching = field(default_factory=StepEpoching)
    trial_rejection: StepTrialRejection = field(default_factory=StepTrialRejection)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a sectioned TOML config file into a PipelineConfig.

    Only sections present in the file are overridden; all other steps
    retain their default values. This allows minimal config files that
    specify only what differs from the defaults.

    Args:
        path (str | Path): Path to the TOML configuration file.

    Returns:
        PipelineConfig: A fully populated PipelineConfig instance.

    Raises:
        ValueError: If the TOML file contains an unrecognized section
            name that doesn't match any pipeline step.
        FileNotFoundError: If the config file does not exist.
    """
    with open(path, "rb") as f:
        raw = load(f)

    step_classes: Dict[str, Any] = {f.name: f.type for f in fields(PipelineConfig)}

    built_steps = {}
    for section, values in raw.items():
        if section not in step_classes:
            raise ValueError(f"Unknown config section: [{section}]")
        built_steps[section] = step_classes[section](**values)

    return PipelineConfig(**built_steps)
