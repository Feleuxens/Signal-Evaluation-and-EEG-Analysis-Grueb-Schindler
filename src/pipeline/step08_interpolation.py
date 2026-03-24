"""Step 08: Interpolate bad channels.

Reconstructs bad channels using spherical spline interpolation from
surrounding good channels. A standard 10-20 montage is set if no
digitization points are present, as interpolation requires electrode
positions. This step is placed late in the pipeline so that
interpolation uses the cleanest possible data.
"""

from mne.io.edf.edf import RawEDF

from utils.config import StepInterpolation


def interpolate_bad_channels(raw: RawEDF, config: StepInterpolation) -> RawEDF:
    """Interpolate channels marked as bad in raw.info["bads"].

    If no channels are marked bad, this step is a no-op. If the raw
    data lacks digitization points (electrode positions), a standard
    10-20 montage is applied automatically, as MNE requires electrode
    coordinates for spherical spline interpolation.

    Args:
        raw (RawEDF): Continuous EEG data with bad channels listed
            in raw.info["bads"]. Modified in place on success.
        config (StepInterpolation): Interpolation parameters
            including whether to clear the bads list after
            interpolation and the interpolation mode.

    Returns:
        RawEDF: The raw data with bad channels interpolated (same
            object, modified in place). Unchanged if no bad channels
            were present or interpolation failed.
    """
    if not raw.info["bads"]:
        return raw

        # Save the list before interpolation, since reset_bads=True clears it
    bad_chs = list(raw.info["bads"])

    try:
        # Electrode positions are required for spherical spline interpolation
        if not raw.info.get("dig"):
            raw.set_montage("standard_1020", on_missing="ignore")

        raw.interpolate_bads(reset_bads=config.reset_bads, mode=config.mode)
        print(f"Interpolated bad channels: {bad_chs}")

    except Exception as e:
        print(f"Interpolation failed: {e}")

    return raw
