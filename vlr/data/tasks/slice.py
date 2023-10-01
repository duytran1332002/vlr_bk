import os
import sys

sys.path.append(os.getcwd())

from dataclasses import dataclass
from logging import getLogger
from datasets import Dataset
from vlr.data.processors.slicer import Slicer


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    raw_dir = "/mnt/d/Projects/sandboxes/vlr/raw"
    visual_dir = "/mnt/d/Projects/sandboxes/vlr/visual"
    audio_dir = "/mnt/d/Projects/sandboxes/vlr/audio"
    save_dir = "/mnt/d/Projects/sandboxes/vlr/v0.2"
    fps = 25
    duration_threshold = 1.0
    batch_size = 100
    num_proc = 8
    num_shards = 10

    separator = Slicer(
        raw_dir=raw_dir,
        visual_dir=visual_dir,
        audio_dir=audio_dir,
        fps=fps,
        duration_threshold=1.0,
        segment_duration=5.0,
        segment_overlap=1.0,
        keep_last_segment=True,
        overwrite=False,
    )


def main(args: Args):
    """
    Main function.
    """
    # Get dataset.
    logger.info("Preparing dataset...")
    dataset = Dataset.from_dict({"file": os.listdir(args.raw_dir)})

    # Extract audio and visual.
    logger.info("Extract audio and visual from dataset...")
    print("Before:", len(dataset["file"]))
    dataset = dataset.map(
        args.separator.process,
        batched=True, batch_size=args.batch_size,
        num_proc=args.num_proc if args.num_proc <= os.cpu_count() else os.cpu_count(),
    )
    print("After:", len(dataset["file"]))
    for index in range(args.num_shards):
        shard = dataset.shard(args.num_shards, index)
        shard.save_to_disk(os.path.join(args.save_dir, f"shard_{index}"))


if __name__ == "__main__":
    main(Args())
