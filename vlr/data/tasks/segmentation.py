import os
import sys
sys.path.append(os.getcwd())

import glob
from dataclasses import dataclass
from logging import getLogger
from datasets import Audio, Dataset
from vlr.data.processors.slicer import Slicer


logger = getLogger(__name__)


@dataclass
class Args:
    """
    Data processing arguments.
    """
    data_dir = "/mnt/d/Projects/sandboxes/vlr/raw"
    save_dir = "/mnt/d/Projects/sandboxes/vlr/processed"
    num_shards = 1000
    batch_size = 100
    num_proc = 8

    slicer = Slicer(
        segment_duration=10.0,
        segment_overlap=1.0,
        keep_last_segment=True,
    )


def get_dataset(data_dir: str, sampling_rate: int):
    """
    Get dataset from path.
    :param data_dir:        data directory.
    :param sampling_rate:   sampling rate.
    :return:                dataset.
    """
    file_paths = glob.glob(data_dir + "/*") if os.path.isdir(data_dir) else [data_dir]
    dataset = Dataset.from_dict({"audio": file_paths})
    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))


def process(batch: dict, processor: Slicer):
    """
    Process sample.
    :param sample:      sample.
    :param processor:   processor.
    :return:            processed sample.
    """
    batch["file"] = os.path.basename(batch["audio"]["path"])
    batch["sampling_rate"] = batch["audio"]["sampling_rate"]
    batch["audio"] = batch["audio"]["array"]
    batch = processor.process(batch)
    return batch


def main(args: Args):
    # Get dataset.
    logger.info("Preparing dataset...")
    dataset = get_dataset(args.data_dir, args.sampling_rate)

    # Shard dataset.
    logger.info("Process each shard of dataset...")
    for index in range(args.num_shards):
        shard = dataset.shard(num_shards=args.num_shards, index=index)
        shard = shard.map(
            lambda sample: process(sample, args.processors),
            batched=True, batch_size=args.batch_size,
            num_proc=args.num_proc if args.num_proc <= os.cpu_count() else os.cpu_count(),
        )
        shard.save_to_disk(args.save_dir + f"/shard_{index}")


if __name__ == "__main__":
    main(Args())
