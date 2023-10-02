import os
import sys

sys.path.append(os.getcwd())

import glob
from dataclasses import dataclass
from logging import getLogger
from datasets import load_from_disk
from vlr.data.processors.transcriber import Transcriber


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    data_dir = "/mnt/d/Projects/sandboxes/vlr/v0.3"
    save_dir = "/mnt/d/Projects/sandboxes/vlr/v0.4"
    batch_size = 100
    num_proc = 8
    overwrite = True

    transcriber = Transcriber(
        model_path="/mnt/d/Projects/vlr/vlr/data/resources/wav2vec2-base-vietnamese-250h",
        lm_gram_name="vi_lm_4grams.bin.zip",
        device="cuda",
    )


def main(args: Args):
    """
    Main function.
    """
    logger.info("Transcribing audio...")
    shard_paths = glob.glob(os.path.join(args.data_dir, "shard_*"))
    for index, shard_path in enumerate(shard_paths):
        save_path = os.path.join(args.save_dir, f"shard_{index}")
        if os.path.exists(save_path) and not args.overwrite:
            continue
        shard = load_from_disk(shard_path)
        shard = shard.map(
            args.transcriber.process,
            batched=True,
            batch_size=args.batch_size,
        )
        shard.save_to_disk(save_path)


if __name__ == "__main__":
    main(Args())
