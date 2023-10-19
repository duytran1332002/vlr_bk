import os
import sys

sys.path.append(os.getcwd())

import argparse
from tqdm import tqdm
from logging import getLogger
from huggingface_hub import HfApi
from vlr.data.utils import zip_dir


logger = getLogger()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace access token.",
    )
    argparser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to file or directory to upload.",
    )
    argparser.add_argument(
        "--channels",
        type=str,
        help="Path to file containing channel names to upload.",
    )
    argparser.add_argument(
        "--auto-zip",
        type=bool,
        default=True,
        help="Whether to automatically zip the directory.",
    )
    argparser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Whether to overwrite existing zip files.",
    )
    argparser.add_argument(
        "--clean-up",
        type=bool,
        default=False,
        help="Whether to clean up zip files after uploading.",
    )


def get_relative_data_dirs(src_dir: str):
    """
    Get data directories.
    :param src:     Path to source directory.
    :return:        Relative paths to data directories.
    """
    relative_data_dirs = []
    for stage_name in os.listdir(src_dir):
        relative_data_dir = stage_name
        stage_dir = os.path.join(src_dir, stage_name)
        if not os.path.isdir(stage_dir):
            continue
        for data_name in os.listdir(stage_dir):
            relative_data_dir = os.path.join(relative_data_dir, data_name)
            data_dir = os.path.join(stage_dir, data_name)
            if not os.path.isdir(data_dir):
                continue
            relative_data_dirs.append(relative_data_dir)
    return relative_data_dirs


def main(args: argparse.Namespace):
    """
    Main function.
    """
    with open(args.channels, "r") as f:
        channel_names = f.read().splitlines()

    relative_data_dirs = get_relative_data_dirs(args.src)

    api = HfApi()
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        for relative_data_dir in tqdm(
            relative_data_dirs,
            desc="Processing data directories",
            total=len(relative_data_dirs),
            unit="directory",
            leave=False,
        ):
            channel_dir = os.path.join(args.src, relative_data_dir, channel_name)
            if not os.path.exists(channel_dir):
                print(f"Channel {channel_name} does not exist in {relative_data_dir}.")
                continue

            if args.auto_zip:
                zip_dir(channel_dir, overwrite=args.overwrite)

            file_path = channel_dir + ".zip"
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.join(
                    os.path.basename(args.src), relative_data_dir, channel_name + ".zip"
                ),
                repo_id="fptu/vlr",
                repo_type="dataset",
                commit_message=f"chore: update {os.path.basename(relative_data_dir)} directory",
                commit_description=f"Add {channel_name}",
                token=args.token,
            )

            if args.clean_up:
                os.remove(file_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
