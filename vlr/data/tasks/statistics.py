import os
import sys

sys.path.append(os.getcwd())

import polars as pl
from tqdm import tqdm
from dataclasses import dataclass
from logging import getLogger
from datasets import load_from_disk


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    # Path to directory of the final stage.
    final_stage_dir = "/mnt/d/Projects/sandboxes/vlr/stage_4"   # Change this.
    # Path to directory containing transcripts.
    transcript_dir = "/mnt/d/Projects/sandboxes/vlr/transcripts"    # Change this.
    # Path to directory containing statistics.
    statistics_dir = "/mnt/d/Projects/sandboxes/vlr/statistics"   # Change this.
    # Path to file containing channel names.
    channel_names_path = "/mnt/d/Projects/sandboxes/vlr/channels.txt"   # Change this.


def main(args: Args):
    """
    Main function.
    """
    with open(args.channel_names_path, "r") as f:
        channel_names = f.read().splitlines()

    dictionary = {}
    total_duration = 0
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        # Get dataset.
        logger.info("Preparing dataset...")
        prev_stage_dir = os.path.join(args.final_stage_dir, channel_name)
        if not os.path.exists(prev_stage_dir):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = load_from_disk(prev_stage_dir)

        # Do statistics on dataset.
        logger.info("Doing statistics dataset...")
        for sample in tqdm(
            dataset,
            desc="Doing statistics on dataset",
            total=dataset.num_rows,
            unit="sample"
        ):
            transcript_path = os.path.join(args.transcript_dir, channel_name, sample["transcript"])
            with open(transcript_path, "r") as f:
                words = f.read().split()
            for word in words:
                dictionary[word] = dictionary.get(word, 0) + 1
            total_duration += sample["duration"]

    # Save statistics.
    logger.info("Saving statistics...")
    if not os.path.exists(args.statistics_dir):
        os.makedirs(args.statistics_dir)

    words_df = pl.DataFrame({
        "word": list(dictionary.keys()),
        "count": list(dictionary.values()),
    })
    words_df = words_df.sort("count", descending=True)
    words_df.write_csv(os.path.join(args.statistics_dir, "words.csv"))

    with open(os.path.join(args.statistics_dir, "statistics.txt"), 'w') as f:
        print(f"Number of vocabularies: {len(dictionary)}", file=f)
        print(f"Total number of words: {sum(dictionary.values())}", file=f)
        print(f"Total duration: {total_duration}s", file=f)


if __name__ == "__main__":
    main(Args())
