import os
import sys

sys.path.append(os.getcwd())

import shutil
import argparse
from tqdm import tqdm
from vlr.data.processors import Executor
from vlr.data.utils import TaskConfig, SlicingTaskConfig, DenoisingTaskConfig
from vlr.data.utils import CroppingTaskConfig, TranscribingTaskConfig


def parse_args() -> argparse.Namespace:
    """
    Get arguments from command line.
    :return:    Arguments from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Available tasks: slice, crop, denoise, transcribe.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to data directory.",
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        default=None,
        help="A channel name or path to file containing channel names.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--upload-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload to hub after processing.",
    )
    parser.add_argument(
        "--clean-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove all downloaded input files after processing.",
    )
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove all output files except for metadata after processing.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version of the dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join(os.getcwd(), ".cache"),
        help="Cache directory.",
    )
    return parser.parse_args()


def get_task_configs(args: argparse.Namespace) -> TaskConfig:
    """
    Get task config.
    :param args:    Arguments from command line.
    :return:        Task config.
    """
    task_dict = {
        "slice": {
            "config": SlicingTaskConfig,
            "dir": "vietnamese-speaker-clip",
        },
        "crop": {
            "config": CroppingTaskConfig,
            "dir": "vietnamese-speaker-lip-clip",
        },
        "denoise": {
            "config": DenoisingTaskConfig,
            "dir": "denoised-vietnamese-audio",
        },
        "transcribe": {
            "config": TranscribingTaskConfig,
            "dir": "purified-vietnamese-audio",
        },
    }
    task_configs = task_dict[args.task]["config"](
        output_dir=os.path.join(args.output_dir, task_dict[args.task]["dir"]),
        channel_names=args.channel_names,
        overwrite=args.overwrite,
        upload_to_hub=args.upload_to_hub,
        clean_input=args.clean_input,
        clean_output=args.clean_output,
        version=args.version,
        cache_dir=os.path.join(args.cache_dir, task_dict[args.task]["dir"])
    )
    return task_configs


def main(configs: TaskConfig) -> None:
    """
    This function is used to process data.
    :param configs:     Task configs.
    """
    print(f"Initialize executor for {configs.task} task...")
    executor = Executor(configs=configs)

    progress_bar = tqdm(
        executor.available_channels,
        desc="Processing channels",
        total=len(executor.available_channels),
        unit="channel"
    )
    for channel in progress_bar:
        print("-" * 20 + f" Processing {channel} " + "-" * 20)

        if executor.is_skipped(channel):
            print("\nChannel existed on the hub.")
            print("To overwrite, please run again with --overwrite.\n")
            print("-" * (13 + len(channel) + 2 * 20))
            continue

        # Prepare save directory.
        executor.prepare_dir(channel=channel)

        # Get dataset.
        print("Loading dataset...")
        executor = executor.load_dataset(channel=channel)

        # Extract audio and visual.
        print("Processing data...")
        executor = executor.process()
        print()

        # Check number of samples.
        print("Checking number of samples...")
        if executor.dataset.num_rows < 1:
            print("No samples left in this channel.")
            continue
        executor.check_num_samples_in_dir()
        executor.print_num_samples_change()
        executor.print_num_output_samples()

        # Save metadata.
        print("Saving metada...")
        executor.save_metadata(channel)

        # Upload to hub.
        executor.upload_to_hub(channel)

        print("-" * (13 + len(channel) + 2 * 20))


if __name__ == "__main__":
    task_configs = get_task_configs(parse_args())
    main(task_configs)
