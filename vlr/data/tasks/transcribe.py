import os
import sys

sys.path.append(os.getcwd())

from dataclasses import dataclass
from logging import getLogger
from datasets import load_from_disk
from tqdm import tqdm
from vlr.data.utils import clean_up
from vlr.data.processors.transcriber import Transcriber


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    # Path to directory of previous stage.
    prev_stage_dir = "/mnt/d/Projects/sandboxes/vlr/stage_2"
    # Path to directory pf current stage.
    cur_stage_dir = "/mnt/d/Projects/sandboxes/vlr/stage_3"
    # Path to directory containing denoised sound files.
    transcript_dir = "/mnt/d/Projects/sandboxes/vlr/transcripts"
    # Path to file containing channel names.
    channel_names_path = "/mnt/d/Projects/sandboxes/vlr/channels.txt"

    batch_size = 10
    num_proc = 8
    overwrite = False

    transcriber = Transcriber(
        model_path="/mnt/d/Projects/vlr/vlr/data/resources/wav2vec2-base-vietnamese-250h",
        lm_gram_name="vi_lm_4grams.bin.zip",
        device="cuda",
        transcript_dir=transcript_dir,
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
        clean_up(channel_name, [args.transcript_dir], args.overwrite)

        # Get dataset.
        logger.info("Preparing dataset...")
        prev_stage_dir = os.path.join(args.prev_stage_dir, channel_name)
        if not os.path.exists(prev_stage_dir):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = load_from_disk(prev_stage_dir)

        # Transcribe.
        logger.info("Transcribing...")
        dataset = dataset.map(
            args.transcriber.process_sample,
            fn_kwargs={"channel_name": channel_name},
        )

        # Filter out samples with empty transcripts.
        logger.info("Filtering out samples with empty transcripts...")
        dataset = dataset.filter(
            lambda sample: sample["transcript"] is not None,
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )

        # Save dataset.
        logger.info("Saving dataset...")
        dataset.save_to_disk(
            os.path.join(args.cur_stage_dir, channel_name),
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )


if __name__ == "__main__":
    main(Args())
