import os
import sys

sys.path.append(os.getcwd())

import argparse
from tqdm import tqdm
from dataclasses import dataclass
from logging import getLogger
from huggingface_hub import HfApi
from vlr.data.utils.tools import zip_dir


logger = getLogger()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    # Path to directory of the final stage.
    final_stage_dir = "/mnt/d/Projects/sandboxes/vlr/stage_4"   # Change this.
    # Path to directory containing visual features.
    visual_dir = "/mnt/d/Projects/sandboxes/vlr/visual"   # Change this.
    # Path to directory containing transcripts.
    audio_dir = "/mnt/d/Projects/sandboxes/vlr/audio"   # Change this.
    # Path to directory containing denoised sound files.
    denoised_dir = "/mnt/d/Projects/sandboxes/vlr/denoised"    # Change this.
    # Path to directory containing transcripts.
    transcript_dir = "/mnt/d/Projects/sandboxes/vlr/transcripts"    # Change this.
    # Path to directory containing mouths.
    mouth_dir = "/mnt/d/Projects/sandboxes/vlr/mouths"  # Change this.
    # Path to file containing channel names.
    channel_names_path = "/mnt/d/Projects/sandboxes/vlr/channels.txt"   # Change this.

    data_dirs = [
        final_stage_dir,
        visual_dir,
        audio_dir,
        denoised_dir,
        transcript_dir,
        mouth_dir,
    ]


def main(args: Args):
    """
    Main function.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--token", type=str, required=True, help="HuggingFace token."
    )
    token = argparser.parse_args().token

    with open(args.channel_names_path, "r") as f:
        channel_names = f.read().splitlines()

    api = HfApi()
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        for data_dir in tqdm(
            args.data_dirs,
            desc="Processing data directories",
            total=len(args.data_dirs),
            unit="directory",
            leave=False,
        ):
            channel_dir = os.path.join(data_dir, channel_name)
            if not os.path.exists(channel_dir):
                print(f"Channel {channel_name} does not exist in {data_dir}.")
                continue

            zip_dir(channel_dir)
            api.upload_file(
                path_or_fileobj=channel_dir + ".zip",
                path_in_repo=os.path.join(os.path.basename(data_dir), channel_name + ".zip"),
                repo_id="fptu/vlr",
                repo_type="dataset",
                commit_message=f"chore: update {os.path.basename(data_dir)} directory",
                commit_description=f"Add {channel_name}",
                token=token,
            )


if __name__ == "__main__":
    main(Args())
