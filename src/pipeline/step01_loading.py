"""Step 01: Load raw EEG data from a BIDS dataset.

Reads BioSemi EDF files via mne-bids, loads data into memory, and
reclassifies external electrode channels: EXG5/EXG6 are marked as
EOG channels (renamed to EOG5/EOG6), and the remaining EXG channels
are marked as misc to exclude them from EEG-specific processing.
"""

from mne_bids import BIDSPath, read_raw_bids
from mne.io import Raw
from mne.io.edf.edf import RawEDF


def load_data(bids_path: BIDSPath) -> RawEDF:
    """Load and prepare raw EEG data from a BIDS path.

    Reads the EDF file specified by the BIDS path, loads it into
    memory, and reclassifies the external electrode channels:

    - EXG5, EXG6 → EOG (renamed to EOG5, EOG6), used for blink
      and eye movement detection in later pipeline steps.
    - EXG1–EXG4, EXG7, EXG8 → misc, excluded from EEG processing.

    Args:
        bids_path (BIDSPath): BIDS path object pointing to the
            subject's EEG recording. Must include subject, root,
            datatype, suffix, and task fields.

    Returns:
        RawEDF: Loaded and channel-reclassified raw EEG data.

    Raises:
        AssertionError: If the loaded file is not in EDF format.
    """
    raw: Raw = read_raw_bids(bids_path)
    assert isinstance(raw, RawEDF)

    raw.load_data()
    # rename EXG5 and EXG6 to EOG5 and EOG6
    for ch in ["EXG5", "EXG6"]:
        raw.set_channel_types({ch: "eog"})
        raw.rename_channels({ch: ch.replace("X", "O")})

    # Remaining EXG channels are unused external electrodes
    for ch in ["EXG1", "EXG2", "EXG3", "EXG4", "EXG7", "EXG8"]:
        raw.set_channel_types({ch: "misc"})

    return raw
