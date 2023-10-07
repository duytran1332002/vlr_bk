import os
import sys

sys.path.append(os.getcwd())

import glob
from dataclasses import dataclass
from logging import getLogger
from datasets import Dataset
from tqdm import tqdm
from vlr.data.processors.slicer import Slicer
from vlr.data.utils.tools import clean_up


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    # Path to directory of previous stage.
    prev_stage_dir = "/mnt/d/Projects/sandboxes/vlr/raw"    # Change this.
    # Path to directory pf current stage.
    cur_stage_dir = "/mnt/d/Projects/sandboxes/vlr/stage_1"   # Change this.
    # Path to directory containing muted videos.
    visual_dir = "/mnt/d/Projects/sandboxes/vlr/visual"  # Change this.
    # Path to directory containing sound files.
    audio_dir = "/mnt/d/Projects/sandboxes/vlr/audio"   # Change this.
    # Path to file containing channel names.
    channel_names_path = "/mnt/d/Projects/sandboxes/vlr/channels.txt"   # Change this.

    batch_size = 100    # Change this if necessary.
    num_proc = -1    # Change this if necessary. -1 means using all available CPUs.
    overwrite = False   # Change this if necessary.

    separator = Slicer(
        visual_dir=visual_dir,
        audio_dir=audio_dir,
        fps=25,
        duration_threshold=1.0,
        segment_duration=3.0,
        segment_overlap=1.0,
        keep_last_segment=True,
        overwrite=overwrite,
    )


def initial_dataset(raw_dir: str, channel_name: str):
    """
    Initial dataset.
    :param raw_dir:         Path to directory containing channels.
    :param channel_name:    Channel name.
    :return:                Dataset.
    """
    files = []
    channels = []
    channel_dir = os.path.join(raw_dir, channel_name)
    chunk_dirs = glob.glob(os.path.join(channel_dir, "*"))
    for chunk_dir in chunk_dirs:
        video_paths = glob.glob(os.path.join(chunk_dir, "pyactive", "*.avi"))
        for video_path in video_paths:
            files.append(video_path)
            channels.append(channel_name)
    return {
        "file": files,
    }


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
        clean_up(channel_name, [args.visual_dir, args.audio_dir], args.overwrite)

        # Get dataset.
        logger.info("Preparing dataset...")
        prev_stage_dir = os.path.join(args.prev_stage_dir, channel_name)
        if not os.path.exists(prev_stage_dir):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = Dataset.from_dict(initial_dataset(args.prev_stage_dir, channel_name))

        # Extract audio and visual.
        logger.info("Extracting audio and visual from dataset...")
        print("Number of samples before slicing:", dataset.num_rows)
        dataset = dataset.map(
            args.separator.process_batch,
            fn_kwargs={"channel_name": channel_name},
            batched=True, batch_size=args.batch_size,
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
            remove_columns=["file"],
        )
        print("Number of samples after slicing:", dataset.num_rows)

        # Save dataset.
        logger.info("Saving dataset...")
        dataset.save_to_disk(
            os.path.join(args.cur_stage_dir, channel_name),
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )


if __name__ == "__main__":
    main(Args())
