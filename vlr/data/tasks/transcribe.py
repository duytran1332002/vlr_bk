import os
import sys

sys.path.append(os.getcwd())

from dataclasses import dataclass
from logging import getLogger
from datasets import load_from_disk
from tqdm import tqdm
from vlr.data.processors.transcriber import Transcriber
from vlr.data.utils.tools import clean_up


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    # Path to directory of previous stage.
    prev_stage_dir = "/mnt/d/Projects/sandboxes/vlr/denoising"    # Change this.
    # Path to directory pf current stage.
    cur_stage_dir = "/mnt/d/Projects/sandboxes/vlr/transcribing"  # Change path, but keep the dir name.
    # Path to directory containing denoised sound files.
    transcript_dir = "/mnt/d/Projects/sandboxes/vlr/transcripts"    # Change path, but keep the dir name.
    # Path to file containing channel names.
    channel_names_path = "/mnt/d/Projects/sandboxes/vlr/channels.txt"   # Change this.

    batch_size = 10     # Change this if necessary.
    num_proc = -1    # Change this if necessary. -1 means using all available CPUs.
    overwrite = False   # Change this if necessary.

    transcriber = Transcriber(
        repo_id="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
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

        # Check number of samples.
        assert len(os.listdir(os.path.join(args.transcript_dir, channel_name))) == dataset.num_rows, \
            "Number of transcripts does not match that in dataset."

        # Save dataset.
        logger.info("Saving dataset...")
        dataset.save_to_disk(
            os.path.join(args.cur_stage_dir, channel_name),
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )
        dataset.cleanup_cache_files()


if __name__ == "__main__":
    main(Args())
