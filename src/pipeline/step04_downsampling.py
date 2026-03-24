"""Step 04: Downsample continuous EEG data.

Reduces the sampling rate to lower memory usage and speed up
subsequent processing steps (particularly ASR and ICA). An
anti-aliasing low-pass filter is applied automatically by MNE
before resampling.
"""

from mne.io.edf.edf import RawEDF

from utils.config import StepDownsampling


def downsample_data(raw: RawEDF, config: StepDownsampling) -> RawEDF:
    """Downsample the raw data to the target sampling frequency.

    Uses MNE's resample(), which applies an anti-aliasing filter
    before decimation. The npad parameter controls FFT padding
    for efficient computation.

    Args:
        raw (RawEDF): Continuous EEG data to resample. Modified
            in place.
        config (StepDownsampling): Resampling parameters including
            the target sampling frequency and padding mode.

    Returns:
        RawEDF: The resampled raw data (same object, modified
            in place).
    """

    raw.resample(config.target_sfreq, npad="auto")

    return raw
