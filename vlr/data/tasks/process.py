import os
import sys

sys.path.append(os.getcwd())

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
        "--channel-names-path",
        type=str,
        default=None,
        help="Path to file containing channel names.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="version of the dataset.",
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
    return parser.parse_args()


def get_task_config(args: argparse.Namespace) -> TaskConfig:
    """
    Get task config.
    :param args:    Arguments from command line.
    :return:        Task config.
    """
    task_dict = {
        "slice": SlicingTaskConfig,
        "crop": CroppingTaskConfig,
        "denoise": DenoisingTaskConfig,
        "transcribe": TranscribingTaskConfig,
    }
    task_config = task_dict[args.task](
        output_dir=args.output_dir,
        channel_names_path=args.channel_names_path,
        overwrite=args.overwrite,
        upload_to_hub=args.upload_to_hub,
        clean_input=args.clean_input,
        clean_output=args.clean_output,
        version=args.version,
    )
    return task_config


def main(configs: TaskConfig) -> None:
    """
    This function is used to process data.
    :param configs:     Task configs.
    """
    print(f"Initialize executor for {configs.task} task...")
    executor = Executor(configs=configs)

    for channel in tqdm(
        executor.available_channels,
        desc="Processing channels",
        total=len(executor.available_channels),
        unit="channel"
    ):
        print("-" * 20 + f" Processing {channel} " + "-" * 20)

        # Prepare save directory.
        print("Preparing save directory...")
        executor.prepare_dir(channel=channel)

        # Get dataset.
        print("Loading dataset...")
        executor = executor.load_dataset(channel=channel)

        # Extract audio and visual.
        print("Processing data...")
        executor = executor.process()

        # Check number of samples.
        print("Checking number of samples...")
        executor.check_num_samples_in_dir()
        executor.print_num_samples_change()
        executor.print_num_output_samples()

        # Save metadata.
        print("Saving metada...")
        executor.save_metadata(channel)

        # Upload to hub.
        executor.upload_to_hub(channel)

        # Clean output.
        executor.clean_output()
        print("-" * (13 + len(channel) + 2 * 20))

    # Clean input.
    executor.clean_input()


if __name__ == "__main__":
    main(configs=get_task_config(args=parse_args()))
