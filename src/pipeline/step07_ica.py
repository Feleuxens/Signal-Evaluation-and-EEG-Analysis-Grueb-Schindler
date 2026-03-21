from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne import pick_types
from mne.io.edf.edf import RawEDF

from utils.config import StepICA


def run_ica(raw: RawEDF, config: StepICA) -> tuple[RawEDF, ICA | None, int]:
    # require at least two EEG channels
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
    exg_chs = [ch for ch in raw.ch_names if ch.upper().startswith("EXG")]

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

    ecg_inds = []
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
