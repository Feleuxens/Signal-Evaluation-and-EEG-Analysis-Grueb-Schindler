from dataclasses import dataclass, field, fields
from pathlib import Path
from tomllib import load
from typing import Any, Dict


@dataclass
class StepBadChannels:
    enabled: bool = True
    z_thresh: float = 3.0
    exg_prefix: str = "EXG"
    status_prefix: str = "Status"


@dataclass
class StepFiltering:
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
    enabled: bool = True
    target_sfreq: float = 128.0
    npad: str = "auto"


@dataclass
class StepRereferencing:
    enabled: bool = True
    ref_channels: str = "average"
    projection: bool = False


@dataclass
class StepASR:
    enabled: bool = True
    cutoff: int = 10


@dataclass
class StepICA:
    enabled: bool = True
    n_components: float = 0.99


@dataclass
class StepInterpolation:
    enabled: bool = True
    reset_bads: bool = False
    mode: str = "accurate"


@dataclass
class StepEpoching:
    epochrange_tmin: float = -0.5
    epochrange_tmax: float = 1.0
    baseline: list[float] = field(default_factory=lambda: [-0.25, 0.0])


@dataclass
class StepTrialRejection:
    enabled: bool = True
    eeg_threshold: float = 150e-6
    eog_threshold: float = 250e-6
    flat_threshold: float = 1e-6


@dataclass
class PipelineConfig:
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
    with open(path, "rb") as f:
        raw = load(f)

    step_classes: Dict[str, Any] = {f.name: f.type for f in fields(PipelineConfig)}

    built_steps = {}
    for section, values in raw.items():
        if section not in step_classes:
            raise ValueError(f"Unknown config section: [{section}]")
        built_steps[section] = step_classes[section](**values)

    return PipelineConfig(**built_steps)
