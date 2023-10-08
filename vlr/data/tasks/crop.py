import os
import sys

sys.path.append(os.getcwd())

from tqdm import tqdm
from dataclasses import dataclass
from logging import getLogger
from datasets import load_from_disk
from vlr.data.utils.tools import clean_up
from vlr.data.processors.cropper import Cropper


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    # Path to directory of previous stage.
    prev_stage_dir = "/mnt/d/Projects/sandboxes/vlr/transcribing"    # Change this.
    # Path to directory pf current stage.
    cur_stage_dir = "/mnt/d/Projects/sandboxes/vlr/cropping"   # Change path, but keep the dir name.
    # Path to directory containing mouth regions.
    mouth_dir = "/mnt/d/Projects/sandboxes/vlr/mouths"  # Change path, but keep the dir name.
    # Path to file containing channel names.
    channel_names_path = "/mnt/d/Projects/sandboxes/vlr/channels.txt"   # Change this.

    num_proc = 8    # Change this if necessary. -1 means using all available CPUs.
    overwrite = False   # Change this if necessary.

    cropper = Cropper(
        mouth_dir=mouth_dir,
        min_detection_confidence=0.99,
        overwrite=overwrite,
    )


def main(args: Args):
    """
    Main function.
    """
    with open(args.channel_names_path, "r") as f:
        channel_names = f.read().splitlines()

    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        # Prepare save directory.
        logger.info("Cleaning up old directories...")
        clean_up(channel_name, [args.mouth_dir], args.overwrite)

        # Get dataset.
        logger.info("Preparing dataset...")
        prev_stage_dir = os.path.join(args.prev_stage_dir, channel_name)
        if not os.path.exists(prev_stage_dir):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = load_from_disk(prev_stage_dir)

        # Crop mouth regions.
        dataset = dataset.map(
            args.cropper.process_sample,
            fn_kwargs={"channel_name": channel_name},
        )

        # Filter out samples with no mouth detected.
        dataset = dataset.filter(
            lambda sample: sample["visual"]["path"] is not None,
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )

        # Check number of samples.
        assert len(os.listdir(os.path.join(args.mouth_dir, channel_name))) == dataset.num_rows, \
            f"{channel_name} - Number of mouth regions does not match that in dataset."

        # Save dataset.
        logger.info("Saving dataset...")
        dataset.save_to_disk(
            os.path.join(args.cur_stage_dir, channel_name),
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )
        dataset.cleanup_cache_files()


if __name__ == "__main__":
    main(Args())
