from mne_bids import BIDSPath
from mne import Annotations, events_from_annotations, Epochs
from mne.io.edf.edf import RawEDF
import pandas as pd
import numpy as np

from utils.config import StepEpoching


def epoch_data(
    raw: RawEDF, bids_path: BIDSPath, config: StepEpoching
) -> tuple[Epochs, np.ndarray, dict]:
    """Epoch the data based on annotations.
    Also applies baseline correction."""

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


def _load_events_from_bids(bids_path: BIDSPath):
    """Load events from BIDS *events.tsv file."""
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


def _generate_epochs(raw, events, event_dict, baseline, tmin, tmax):
    """Generate epochs from raw data and events."""

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
