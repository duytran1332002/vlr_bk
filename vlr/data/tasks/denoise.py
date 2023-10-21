import os
import sys

sys.path.append(os.getcwd())

import argparse
from datasets import load_dataset, disable_caching
from tqdm import tqdm
from vlr.data.processors.denoiser import Denoiser
from vlr.data.utils.tools import clean_up


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
        "--batch-size",
        type=int,
        default=100,
        help="Batch size.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    """
    Main function.
    """
    if not os.path.exists(args.data_dir):
        print(f"Directory {args.data_dir} does not exist.")
        return
    audio_dir = os.path.join(args.data_dir, "audio")
    if not os.path.exists(audio_dir):
        print(f"Directory {audio_dir} does not exist.")
        return
    prev_stage_dir = os.path.join(args.data_dir, "stage_1")
    if not os.path.exists(prev_stage_dir):
        print(f"Directory {prev_stage_dir} does not exist.")
        return

    denoised_dir = os.path.join(args.data_dir, "denoised")
    if not os.path.exists(denoised_dir):
        os.makedirs(denoised_dir)
    cur_stage_dir = os.path.join(args.data_dir, "stage_2")
    if not os.path.exists(cur_stage_dir):
        os.makedirs(cur_stage_dir)

    if args.channel_names_path:
        with open(args.channel_names_path, "r") as f:
            channel_names = f.read().strip().split()
    else:
        channel_names = os.listdir(audio_dir)

    denoiser = Denoiser(
        audio_dir=audio_dir,
        denoised_dir=denoised_dir,
        sampling_rate=16000,
        overwrite=args.overwrite,
    )

    print("\n#" * 50 + " Denoising " + "#" * 50)
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        print("-" * 20 + f" Processing {channel_name} " + "-" * 20)
        # Prepare save directory.
        clean_up(channel_name, [denoised_dir], args.overwrite)

        # Get dataset.
        print("Preparing dataset...")
        prev_stage_path = os.path.join(prev_stage_dir, channel_name + ".json")
        if not os.path.exists(prev_stage_path):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = load_dataset(
            "json", data_files=prev_stage_path, split="train",
        )

        # Denoise audio.
        print("Denoising audio...")
        dataset = dataset.map(
            denoiser.process_batch,
            batched=True,
            batch_size=args.batch_size,
        )

        # Check number of samples.
        assert len(os.listdir(os.path.join(denoised_dir, channel_name))) == dataset.num_rows, \
            f"{channel_name} - Number of denoised samples does not match that in dataset."

        # Save dataset.
        print("Saving dataset...")
        dataset.to_pandas().to_json(
            os.path.join(cur_stage_dir, channel_name + ".json"),
            orient="records",
        )
        dataset.cleanup_cache_files()


if __name__ == "__main__":
    main(parse_args())
