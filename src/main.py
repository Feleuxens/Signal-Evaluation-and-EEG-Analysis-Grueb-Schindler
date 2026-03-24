"""
Entry point for the EEG processing pipeline.

Provides a CLI menu to run preprocessing pipelines, generate plots,
and view statistics. Supports parallel execution across subjects.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

from os import mkdir, getenv
from os.path import isdir

from pipeline.analyze_subject import (
    run_pipeline,
    plot_specific_subject,
    plot_average_data,
)
from utils.utils import (
    get_subject_list,
    get_config_ids,
    get_config_path,
    pipeline_statistics,
)
from utils.config import load_config


def main():
    """Interactive CLI for selecting and running pipeline actions.

    Reads BIDS_ROOT and CONFIG_ROOT from environment variables,
    discovers available subjects and configs, then prompts the user
    to choose an action (processing, plotting, or statistics).
    """
    bids_root = getenv("BIDS_ROOT", "../data/")
    bids_root = bids_root.rstrip("/")
    config_root = getenv("CONFIG_ROOT", "../config/")
    config_root = config_root.rstrip("/")

    if not isdir(bids_root + "/processed"):
        mkdir(bids_root + "/processed")

    subjects = get_subject_list(bids_root)
    print(f"Subjects: {subjects}\n")
    configs = get_config_ids("config")
    print(f"Configs: {configs}\n")

    print("Which action do you want to perform?")
    print("1 - Process data for specific config (all subjects)")
    print("2 - Process data for all configs (all subjects)")
    print("3 - Process data for specific config (specific subject)")
    print("4 - Process data for all config (specific subject)")
    print("5 - Generate plot for specific config (only one subject)")
    print("6 - Generate plot for specific config (combined subjects)")
    print("7 - Generate plot for all configs (combined subjects)")
    print("8 - Print pipeline statistics for specific config")
    i = input(": ")
    if i.lower() == "1":
        c = int(input("Config ID: "))
        path = get_config_path(config_root, c)
        tasks = [(path, bids_root, c, s) for s in subjects]
        run_parallel(tasks)
    elif i.lower() == "2":
        tasks = [
            (get_config_path(config_root, c), bids_root, c, s)
            for c in configs
            for s in subjects
        ]
        run_parallel(tasks)
    elif i.lower() == "3":
        c = int(input("Config ID: "))
        config = load_config(get_config_path(config_root, c))
        s = f"{int(input("Subject ID: ")):03d}"
        start_time = time()
        run_pipeline(config, bids_root, c, s)
        total_time = time() - start_time
        print(f"\nElapsed time: {total_time} seconds\n")
    elif i.lower() == "4":
        s = f"{int(input("Subject ID: ")):03d}"
        tasks = [(get_config_path(config_root, c), bids_root, c, s) for c in configs]
    elif i.lower() == "5":
        c = int(input("Config ID: "))
        config = load_config(get_config_path(config_root, c))
        s = f"{int(input("Subject ID: ")):03d}"
        plot_specific_subject(config, bids_root + "/processed", c, s)
    elif i.lower() == "6":
        c = int(input("Config ID: "))
        config = load_config(get_config_path(config_root, c))
        plot_average_data(config, bids_root + "/processed", c)
    elif i.lower() == "7":
        for c in configs:
            if not isdir(bids_root + "/processed/" + str(c)):
                continue
            config = load_config(get_config_path(config_root, c))
            plot_average_data(config, bids_root + "/processed", c)
    elif i.lower() == "8":
        c = int(input("Config ID: "))
        pipeline_statistics(bids_root, c)
    else:
        print("Invalid input")


def process_subject(config_path: str, bids_root: str, config_id: int, subject_id: str):
    """Run the pipeline for a single subject in a worker process.

    Loads the config from disk inside each worker rather than receiving
    a PipelineConfig object, to avoid pickling issues with complex types.

    Args:
        config_path: Path to the TOML configuration file.
        bids_root: Root directory of the BIDS dataset.
        config_id: Numeric identifier for the configuration (used for
            organizing output directories).
        subject_id: Zero-padded subject identifier (e.g. "001").

    Returns:
        A tuple of (subject_id, config_id, error_message). The error
        message is None on success, or a string description on failure.
    """
    config = load_config(config_path)
    try:
        run_pipeline(config, bids_root, config_id, subject_id)
        return subject_id, config_id, None
    except Exception as e:
        return subject_id, config_id, str(e)


def run_parallel(tasks: list[tuple[str, str, int, str]]):
    """Execute pipeline tasks in parallel using a process pool.

    The number of workers is controlled by the MAX_WORKERS environment
    variable (default: 2). Each task runs in a separate process to
    avoid GIL contention on CPU-bound EEG processing.

    Args:
        tasks: List of (config_path, bids_root, config_id, subject_id)
            tuples. Each tuple represents one independent pipeline run
            for a single subject with a given configuration.
    """
    start_time = time()

    with ProcessPoolExecutor(max_workers=int(getenv("MAX_WORKERS", 2))) as executor:
        futures = {executor.submit(process_subject, *task): task for task in tasks}

        for future in as_completed(futures):
            subject_id, config_id, error = future.result()
            if error:
                print(f"FAILED config={config_id} subject={subject_id}: {error}")
            else:
                print(f"DONE   config={config_id} subject={subject_id}")

    total_time = time() - start_time
    print(
        f"\nElapsed time: {total_time:.1f}s ({len(tasks)} jobs, {int(getenv("MAX_WORKERS", 2))} workers)\n"
    )


if __name__ == "__main__":
    main()
