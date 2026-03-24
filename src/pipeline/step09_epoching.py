"""Step 09: Epoch continuous EEG data around stimulus events.

Loads event information from the BIDS events.tsv file, attaches
them as annotations to the raw data, then segments the continuous
recording into fixed-length epochs around each event. Baseline
correction is applied to remove pre-stimulus DC offset.
"""

from mne_bids import BIDSPath
from mne import Annotations, events_from_annotations, Epochs
from mne.io.edf.edf import RawEDF
import pandas as pd
import numpy as np

from utils.config import StepEpoching


def epoch_data(
    raw: RawEDF, bids_path: BIDSPath, config: StepEpoching
) -> tuple[Epochs, np.ndarray, dict]:
    """Epoch the continuous data and apply baseline correction.

    Loads stimulus events from the BIDS events.tsv sidecar, attaches
    them as annotations, extracts MNE events, creates epochs around
    each stimulus onset, and applies mean baseline subtraction.

    Args:
        raw (RawEDF): Continuous EEG data to epoch. Annotations are
            attached in place.
        bids_path (BIDSPath): BIDS path used to locate the
            corresponding events.tsv file.
        config (StepEpoching): Epoching parameters including the
            time window (tmin/tmax) and baseline correction interval.

    Returns:
        tuple[Epochs, np.ndarray, dict]: A tuple of:
            - Baseline-corrected epochs.
            - Events array of shape (n_events, 3) as returned by
              MNE's events_from_annotations.
            - Event ID dictionary mapping condition names to integer
              codes (e.g. {"random": 1, "regular": 2}).
    """

    raw = _load_and_attach_annotations(bids_path, raw)
    events, event_dict = events_from_annotations(raw)

    epochs = _generate_epochs(
        raw,
        events,
        event_dict,
        config.baseline,
        config.epochrange_tmin,
        config.epochrange_tmax,
    )
    epochs.apply_baseline(config.baseline)

    return epochs, events, event_dict


def _load_and_attach_annotations(bids_path: BIDSPath, raw: RawEDF) -> RawEDF:
    """Load events from BIDS *events.tsv file and attach as annotations to raw."""
    annotations = _load_events_from_bids(bids_path)
    raw.set_annotations(annotations)

    print(f"Found {len(raw.annotations)} annotations.")

    return raw


def _load_events_from_bids(bids_path: BIDSPath) -> Annotations:
    """Parse the BIDS events.tsv file into MNE Annotations.

    Maps integer event codes to human-readable condition labels:
    1 → "regular", 3 → "random". Unrecognized codes are labelled
    "event".

    Args:
        bids_path (BIDSPath): BIDS path used to derive the
            events.tsv file path.

    Returns:
        Annotations: MNE Annotations object with onset times and
            condition labels for each stimulus event.
    """
    events_tsv = bids_path.copy().update(suffix="events", extension=".tsv").fpath
    df = pd.read_csv(events_tsv, sep="\t")

    # BIDS events.tsv has onset (s), duration (s), and trial_type or value
    onsets = df["onset"].astype(float).to_list()
    durations = [0.0] * len(onsets)

    value_mapping = {1: "regular", 3: "random"}
    values = df["value"].astype(int).to_list()
    descriptions = []

    for v in values:
        descriptions.append(value_mapping.get(v, "event"))

    return Annotations(onset=onsets, duration=durations, description=descriptions)


def _generate_epochs(raw, events, event_dict, baseline, tmin, tmax) -> Epochs:
    """Create epochs from raw data and events.

    Segments the continuous recording into fixed-length windows
    around each event. Epochs overlapping with BAD annotations
    are automatically rejected.

    Args:
        raw (RawEDF): Continuous EEG data.
        events (np.ndarray): Events array of shape (n_events, 3).
        event_dict (dict): Mapping of condition names to event codes.
        baseline (list[float]): Two-element list [start, end] in
            seconds for baseline correction.
        tmin (float): Start of the epoch relative to stimulus onset
            in seconds.
        tmax (float): End of the epoch relative to stimulus onset
            in seconds.

    Returns:
        Epochs: Preloaded epochs with annotation-based rejection
            applied.
    """

    epochs = Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=True,
    )

    return epochs
