from mne_bids import BIDSPath, read_raw_bids
from mne.io.edf.edf import RawEDF


def load_data(bids_path: BIDSPath) -> RawEDF:
    """Load raw EEG data from BIDS path."""

    raw: RawEDF = read_raw_bids(bids_path)
    raw.load_data()

    # rename EXG5 and EXG6 to EOG5 and EOG6
    for ch in ["EXG5", "EXG6"]:
        raw.set_channel_types({ch: "eog"})
        raw.rename_channels({ch: ch.replace('X', 'O')})

    for ch in ["EXG1", "EXG2", "EXG3", "EXG4", "EXG7", "EXG8"]:
        raw.set_channel_types({ch: "misc"})

    return raw
