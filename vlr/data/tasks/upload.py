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
        "--channel-names-path",
        type=str,
        help="Path to file containing channel names to upload.",
    )
    argparser.add_argument(
        "--zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to automatically zip the directory.",
    )
    argparser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to overwrite existing zip files.",
    )
    argparser.add_argument(
        "--clean-up",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to clean up zip files after uploading.",
    )
    return argparser.parse_args()


def get_relative_data_dirs(src_dir: str):
    """
    Get data directories.
    :param src:     Path to source directory.
    :return:        Relative paths to data directories.
    """
    relative_data_dirs = []
    for data_name in os.listdir(src_dir):
        data_dir = os.path.join(src_dir, data_name)
        if not os.path.isdir(data_dir):
            continue
        relative_data_dirs.append(data_name)
    return relative_data_dirs


def main(args: argparse.Namespace):
    """
    Main function.
    """
    with open(args.channel_names_path, "r") as f:
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
            desc=f"Processing {channel_name}",
            total=len(relative_data_dirs),
            unit="directory",
            leave=False,
        ):
            channel_dir = os.path.join(args.src, relative_data_dir, channel_name)
            if not relative_data_dir.startswith("stage"):
                if not os.path.exists(channel_dir):
                    print(f"Channel {channel_name} does not exist in {relative_data_dir}.")
                    continue

                file_path = channel_dir + ".zip"
                if args.zip or not os.path.exists(file_path):
                    zip_dir(channel_dir, overwrite=args.overwrite)

                path_in_repo = os.path.join(
                    os.path.basename(args.src), relative_data_dir, channel_name + ".zip"
                )
            else:
                file_path = channel_dir + ".json"
                if not os.path.exists(file_path):
                    print(f"Channel {file_path} does not exist in {relative_data_dir}.")
                    continue

                path_in_repo = os.path.join(
                    os.path.basename(args.src), relative_data_dir, channel_name + ".json"
                )

            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id="fptu/vlr",
                repo_type="dataset",
                commit_message=f"chore: update {os.path.basename(relative_data_dir)} directory",
                commit_description=f"Add {channel_name}",
                token=args.token,
            )

            if args.clean_up and file_path.endswith("zip"):
                os.remove(file_path)


if __name__ == "__main__":
    main(parse_args())
