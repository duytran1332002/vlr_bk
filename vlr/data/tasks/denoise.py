import os
import sys

sys.path.append(os.getcwd())

from dataclasses import dataclass
from logging import getLogger
from datasets import load_from_disk
from tqdm import tqdm
from vlr.data.processors.denoiser import Denoiser
from vlr.data.utils.tools import clean_up


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    # Path to directory of previous stage.
    prev_stage_dir = "/mnt/d/Projects/sandboxes/vlr/slicing"    # Change this.
    # Path to directory pf current stage.
    cur_stage_dir = "/mnt/d/Projects/sandboxes/vlr/denoising"   # Change path, but keep the dir name.
    # Path to directory containing denoised sound files.
    denoised_dir = "/mnt/d/Projects/sandboxes/vlr/denoised"   # Change path, but keep the dir name.
    # Path to file containing channel names.
    channel_names_path = "/mnt/d/Projects/sandboxes/vlr/channels.txt"   # Change this.

    batch_size = 40   # Change this if necessary.
    num_proc = -1    # Change this if necessary. -1 means using all available CPUs.
    overwrite = False   # Change this if necessary.

    denoiser = Denoiser(
        denoised_dir=denoised_dir,
        sampling_rate=16000,
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
        clean_up(channel_name, [args.denoised_dir], args.overwrite)

        # Get dataset.
        logger.info("Preparing dataset...")
        prev_stage_dir = os.path.join(args.prev_stage_dir, channel_name)
        if not os.path.exists(prev_stage_dir):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = load_from_disk(prev_stage_dir)

        # Denoise audio.
        logger.info("Denoising audio...")
        dataset = dataset.map(
            args.denoiser.process_batch,
            fn_kwargs={"channel_name": channel_name},
            batched=True,
            batch_size=args.batch_size,
        )

        # Check number of samples.
        assert len(os.listdir(os.path.join(args.denoised_dir, channel_name))) == dataset.num_rows, \
            f"{channel_name} - Number of denoised samples does not match that in dataset."

        # Save dataset.
        logger.info("Saving dataset...")
        dataset.save_to_disk(
            os.path.join(args.cur_stage_dir, channel_name),
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )
        dataset.cleanup_cache_files()


if __name__ == "__main__":
    main(Args())
