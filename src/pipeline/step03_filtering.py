"""Step 03: Bandpass and notch filtering.

Applies notch filtering to remove line noise harmonics and bandpass
filtering to retain only the frequency range of interest. Both
filters can be independently enabled or disabled via the config.
"""

from array import array
from mne.io.edf.edf import RawEDF

from utils.config import StepFiltering


def filter_data(raw: RawEDF, config: StepFiltering) -> RawEDF:
    """Apply notch and bandpass filtering to the raw data.

    Notch filtering is applied first to remove power line harmonics,
    followed by bandpass filtering to retain the desired frequency
    range. Each filter operates only on the channel type specified
    in the config.

    Args:
        raw (RawEDF): Continuous EEG data to filter. Modified in place.
        config (StepFiltering): Filter parameters including cutoff
            frequencies, notch frequencies, channel picks, and
            filter methods.

    Returns:
        RawEDF: The filtered raw data (same object, modified in place).
    """

    notch_freqs = array("f", config.notch_frequencies)
    if config.notch_filter_enabled:
        raw.notch_filter(
            notch_freqs,
            picks=config.notch_filter_pick,
            method=config.notch_filter_method,
        )
    if config.pass_filter_enabled:
        # MNE's filter() expects (l_freq, h_freq) = (high-pass, low-pass)
        raw.filter(
            config.high_pass,
            config.low_pass,
            picks=config.pass_filter_pick,
            method=config.pass_filter_method,
        )

    return raw
