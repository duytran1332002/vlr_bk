import os
import sys

sys.path.append(os.getcwd())

import argparse
from tqdm import tqdm
from datasets import load_dataset, disable_caching
from vlr.data.utils.tools import clean_up
from vlr.data.processors.cropper import Cropper


disable_caching()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory.",
    )
    parser.add_argument(
        "--channel-names-path",
        type=str,
        default=None,
        help="Path to file containing channel names.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=-1,
        help="Number of processes.",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    """
    if not os.path.exists(args.data_dir):
        print(f"Directory {args.data_dir} does not exist.")
        return
    visual_dir = os.path.join(args.data_dir, "visual")
    if not os.path.exists(visual_dir):
        print(f"Directory {visual_dir} does not exist.")
        return
    prev_stage_dir = os.path.join(args.data_dir, "stage_3")
    if not os.path.exists(prev_stage_dir):
        print(f"Directory {prev_stage_dir} does not exist.")
        return

    mouth_dir = os.path.join(args.data_dir, "mouths")
    if not os.path.exists(mouth_dir):
        os.makedirs(mouth_dir)
    cur_stage_dir = os.path.join(args.data_dir, "stage_4")
    if not os.path.exists(cur_stage_dir):
        os.makedirs(cur_stage_dir)

    if args.channel_names_path:
        with open(args.channel_names_path, "r") as f:
            channel_names = f.read().strip().split()
    else:
        channel_names = os.listdir(visual_dir)

    cropper = Cropper(
        visual_dir=visual_dir,
        mouth_dir=mouth_dir,
        min_detection_confidence=0.99,
        overwrite=args.overwrite,
    )

    print("\n" + "#" * 50 + " Cropping " + "#" * 50)
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        print("-" * 20 + f" Processing {channel_name} " + "-" * 20)
        # Prepare save directory.
        clean_up(channel_name, [mouth_dir], args.overwrite)

        # Get dataset.
        print("Preparing dataset...")
        prev_stage_path = os.path.join(prev_stage_dir, channel_name + ".json")
        if not os.path.exists(prev_stage_path):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = load_dataset(
            "json", data_files=prev_stage_path, split="train",
        )

        # Crop mouth regions.
        dataset = dataset.map(
            cropper.process_sample,
        )

        # Filter out samples with no mouth detected.
        dataset = dataset.filter(
            lambda sample: sample["id"] is not None,
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )

        # Check number of samples.
        assert len(os.listdir(os.path.join(mouth_dir, channel_name))) == dataset.num_rows, \
            f"{channel_name} - Number of mouth regions does not match that in dataset."

        # Save dataset.
        print("Saving dataset...")
        dataset.to_pandas().to_json(
            os.path.join(cur_stage_dir, channel_name + ".json"),
            orient="records",
        )
        dataset.cleanup_cache_files()


if __name__ == "__main__":
    main(parse_args())
