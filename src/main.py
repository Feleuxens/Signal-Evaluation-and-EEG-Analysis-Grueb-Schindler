from time import time

from os import mkdir, getenv
from os.path import isdir

from pipeline.analyze_subject import (
    run_pipeline,
    plot_specific_subject,
    plot_average_data,
)
from utils.plots import plot_channel, plot_topomap
from utils.utils import (
    get_subject_list,
    average_channel,
    pairwise_average,
    pipeline_statistics,
    get_config_ids,
    get_config_path,
)
from utils.config import load_config


def main():
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
    i = input(": ")
    if i.lower() == "1":
        c = int(input("Config ID: "))
        config = load_config(get_config_path(config_root, c))
        start_time = time()
        for subject in subjects:
            run_pipeline(config, bids_root, c, subject)
        total_time = time() - start_time
        print(f"\nElapsed time: {total_time} seconds\n")
    elif i.lower() == "2":
        start_time = time()
        for c in configs:
            config = load_config(get_config_path(config_root, c))
            for subject in subjects:
                run_pipeline(config, bids_root, c, subject)
        total_time = time() - start_time
        print(f"\nElapsed time: {total_time} seconds\n")
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
        start_time = time()
        for c in configs:
            config = load_config(get_config_path(config_root, c))
            run_pipeline(config, bids_root, c, s)
        total_time = time() - start_time
        print(f"\nElapsed time: {total_time} seconds\n")
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
            config = load_config(get_config_path(config_root, c))
            plot_average_data(config, bids_root + "/processed", c)
    else:
        print("Invalid input")


if __name__ == "__main__":
    main()
