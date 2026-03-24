"""Step 06: Artifact Subspace Reconstruction (ASR).

Applies ASR to identify and reconstruct time segments containing
high-amplitude transient artifacts. ASR computes a clean reference
covariance from calibration data, then reconstructs windows whose
signal subspace deviates beyond the cutoff threshold. Uses the
asrpy package since MNE does not include a built-in ASR
implementation.
"""

import asrpy
from mne.io.edf.edf import RawEDF

from utils.config import StepASR


def run_asr(raw: RawEDF, config: StepASR) -> tuple[RawEDF, asrpy.ASR | None]:
    """Apply ASR artifact correction to the raw data.

    Fits ASR on the continuous data to identify clean calibration
    segments, then reconstructs artifact-contaminated windows.
    If ASR fails (e.g. due to insufficient clean data or numerical
    issues), the pipeline continues with uncorrected data.

    Args:
        raw (RawEDF): Continuous EEG data. Modified in place on
            success.
        config (StepASR): ASR parameters, primarily the cutoff
            threshold (lower = more aggressive correction).

    Returns:
        tuple[RawEDF, asrpy.ASR | None]: A tuple of:
            - The (possibly corrected) raw data.
            - The fitted ASR object, or None if ASR failed.
    """

    asr = None

    try:
        asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=config.cutoff)
        asr.fit(raw)
        raw = asr.transform(raw)

    except Exception as e:
        print(f"ASR failed: {e}. Continuing without ASR.")

    # finally:
    return raw, asr
