"""Entry point for blink detection analysis.

Provides a CLI for investigating the impact of eye blinks on ERP
results, particularly in relation to ASR artifact correction.
Supports plotting EOG channels, visualizing blink-epoch overlays
for individual subjects, precomputing blink-labelled epochs across
all subjects, and generating grand average plots split by blink
presence.
"""

from time import time
from os import getenv

from utils.config import load_config
from utils.utils import get_config_path

from blinks.plots import plot_eog, plot_eeg_plus_eog_one_subject, all_subjects_plotting
from blinks.blinks import precompute_all_epochs


def main():
    """Interactive CLI for blink detection and analysis actions.

    Loads the default pipeline config (config ID 1) and prompts
    the user to choose an action: single-subject blink visualization,
    batch precomputation of blink-labelled epochs, grand average
    plotting split by blink condition, or raw EOG channel inspection.
    """
    bids_root = getenv("BIDS_ROOT", "../data/")
    bids_root = bids_root.rstrip("/")
    config_root = getenv("CONFIG_ROOT", "../config/")
    config_root = config_root.rstrip("/")

    # load default config
    config_path = get_config_path(config_root, 1)
    config = load_config(config_path)

    output_folder = bids_root + "/processed_blinkdetection"

    print("Which action do you want to perform?")
    print("1 - Plot blink detection for one subjects")
    print("2 - Process data for for blink comparison (all subjects)")
    print("3 - Plot data for blink comparison (all subjects)")
    print("4 - Plot presumable EOG5 and 6 channels (one subject)")

    i = input(": ")
    if i.lower() == "1":
        s = f"{int(input("Subject ID: ")):03d}"
        plot_eeg_plus_eog_one_subject(bids_root, s, config)

    elif i.lower() == "2":
        with_asr = input("With ASR (y/n)") == "y"
        start_time = time()
        precompute_all_epochs(bids_root, config, output_folder, with_asr)
        total_time = time() - start_time
        print(f"\nElapsed time: {total_time} seconds\n")

    elif i.lower() == "3":
        with_asr = input("With ASR (y/n)") == "y"
        start_time = time()
        all_subjects_plotting(bids_root, config, output_folder, with_asr)
        total_time = time() - start_time
        print(f"\nElapsed time: {total_time} seconds\n")

    elif i.lower() == "4":
        s = f"{int(input("Subject ID: ")):03d}"
        start_time = time()
        plot_eog(bids_root, subject_id=s)
        total_time = time() - start_time
        print(f"\nElapsed time: {total_time} seconds\n")


if __name__ == "__main__":
    main()
