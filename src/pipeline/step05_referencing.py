"""Step 05: Re-reference EEG data.

Applies a new reference to the EEG channels. Average referencing
(the default) reduces shared noise across all channels and is a
standard choice for ERP analysis. Note that average referencing
reduces the data rank by one, which must be accounted for in
subsequent ICA decomposition.
"""

from mne.io.edf.edf import RawEDF

from utils.config import StepRereferencing


def rereference_data(raw: RawEDF, config: StepRereferencing) -> RawEDF:
    """Re-reference the raw data to the specified reference.

    Args:
        raw (RawEDF): Continuous EEG data to re-reference. Modified
            in place.
        config (StepRereferencing): Re-referencing parameters
            including the reference channel(s) and whether to apply
            as a projection.

    Returns:
        RawEDF: The re-referenced raw data (same object, modified
            in place).
    """

    raw.set_eeg_reference(config.ref_channels, projection=config.projection)

    return raw
