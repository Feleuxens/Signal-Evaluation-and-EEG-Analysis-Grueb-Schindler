"""Step 07: Independent Component Analysis (ICA) for artifact removal.

Decomposes the EEG signal into statistically independent components
using the Infomax algorithm, then automatically identifies and
removes components corresponding to ocular (EOG) and cardiac (ECG)
artifacts. Components are detected using MNE's find_bads_eog and
find_bads_ecg correlation-based methods.
"""

from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne import pick_types
from mne.io.edf.edf import RawEDF

from utils.config import StepICA


def run_ica(raw: RawEDF, config: StepICA) -> tuple[RawEDF, ICA | None, int]:
    """Fit ICA and remove ocular/cardiac artifact components.

    Fits ICA on all non-bad EEG channels, then uses EOG and ECG
    channels to automatically identify artifact components. Detected
    components are excluded and the ICA is applied to the continuous
    data in place.

    Requires at least two EEG channels. If fewer are available
    (e.g. all marked bad), ICA is skipped.

    Args:
        raw (RawEDF): Continuous EEG data. Modified in place when
            artifact components are removed.
        config (StepICA): ICA parameters, including the number of
            components (or variance threshold).

    Returns:
        tuple[RawEDF, ICA | None, int]: A tuple of:
            - The (possibly cleaned) raw data.
            - The fitted ICA object, or None if ICA was skipped.
            - Number of excluded components (0 if ICA was skipped
              or no artifacts were found).
    """
    # Require at least two EEG channels for decomposition
    picks_eeg = pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    if len(picks_eeg) < 2:
        print("Not enough EEG channels for ICA.")
        return raw, None, 0

    ica = ICA(
        n_components=config.n_components,
        method="infomax",
        random_state=42,
        max_iter="auto",
    )
    ica.fit(raw, picks=picks_eeg)

    # find EOG components using any channels already marked as 'eog' or named EXG*
    ch_types = raw.get_channel_types()
    eog_chs = [ch for ch, t in zip(raw.ch_names, ch_types) if t == "eog"]

    eog_inds = []
    if eog_chs:
        try:
            # use first available EOG/EXG channel to create EOG epochs
            eog_epochs = create_eog_epochs(
                raw, ch_name=eog_chs[0], reject_by_annotation=True, preload=True
            )
            eog_inds, scores = ica.find_bads_eog(eog_epochs)
        except Exception:
            eog_inds = []

    try:
        ecg_epochs = create_ecg_epochs(raw, reject_by_annotation=True, preload=True)
        ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)
    except Exception:
        ecg_inds = []

    to_remove = sorted(set(eog_inds + ecg_inds))
    if to_remove:
        ica.exclude = to_remove
        ica.apply(raw)  # applies ICA to raw in-place

    number_excluded_components = len(getattr(ica, "exclude", []))
    print(
        "ICA cleaning applied. Excluded components:",
        getattr(ica, "exclude", []),
    )

    return raw, ica, number_excluded_components
