import os
import sys

sys.path.append(os.getcwd())

import glob
from dataclasses import dataclass
from logging import getLogger
from datasets import load_from_disk, Audio
from vlr.data.processors.denoiser import Denoiser


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    data_dir = "/mnt/d/Projects/sandboxes/vlr/v0.2"
    save_dir = "/mnt/d/Projects/sandboxes/vlr/v0.3"
    sampling_rate = 16000
    batch_size = 100
    num_proc = 8
    overwrite = False

    denoiser = Denoiser()


def format(sample: dict):
    """
    Format sample.
    :param sample:  Sample.
    :return:        Formatted sample.
    """
    sample["sampling_rate"] = sample["audio"]["sampling_rate"]
    sample["audio"] = sample["audio"]["array"]
    return sample


def main(args: Args):
    """
    Main function.
    """
    logger.info("Denoising audio...")
    shard_paths = glob.glob(os.path.join(args.data_dir, "shard_*"))
    for index, shard_path in enumerate(shard_paths):
        save_path = os.path.join(args.save_dir, f"shard_{index}")
        if os.path.exists(save_path) and not args.overwrite:
            continue
        shard = load_from_disk(shard_path)
        shard = shard.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
        shard = shard.map(
            format,
            num_proc=args.num_proc if args.num_proc <= os.cpu_count() else os.cpu_count(),
        )
        shard = shard.map(
            args.denoiser.process,
            batched=True,
            batch_size=args.batch_size,
        )
        shard.save_to_disk(save_path)


if __name__ == "__main__":
    main(Args())
