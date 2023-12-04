import os
import sys

sys.path.append(os.getcwd())

import argparse
from tqdm import tqdm
from datasets import load_dataset, disable_caching
from vlr.data.processors.transcriber import Transcriber
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
    denoised_dir = os.path.join(args.data_dir, "denoised")
    if not os.path.exists(denoised_dir):
        print(f"Directory {denoised_dir} does not exist.")
        return
    prev_stage_dir = os.path.join(args.data_dir, "stage_3")
    if not os.path.exists(prev_stage_dir):
        print(f"Directory {prev_stage_dir} does not exist.")
        return

    transcript_dir = os.path.join(args.data_dir, "transcripts")
    os.makedirs(transcript_dir, exist_ok=True)
    cur_stage_dir = os.path.join(args.data_dir, "stage_4")
    os.makedirs(cur_stage_dir, exist_ok=True)

    if args.channel_names_path:
        with open(args.channel_names_path, "r") as f:
            channel_names = f.read().strip().split()
    else:
        channel_names = os.listdir(denoised_dir)

    transcriber = Transcriber(
        repo_id="nguyenvulebinh/wav2vec2-large-vi-vlsp2020",
        denoised_dir=denoised_dir,
        transcript_dir=transcript_dir,
        device="cuda",
        overwrite=args.overwrite,
    )

    print("\n" + "#" * 50 + " Transcribing " + "#" * 50)
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        print("-" * 20 + f" Processing {channel_name} " + "-" * 20)
        # Prepare save directory.
        clean_up(channel_name, [transcript_dir], args.overwrite)

        # Get dataset.
        print("Preparing dataset...")
        prev_stage_path = os.path.join(prev_stage_dir, channel_name + ".json")
        if not os.path.exists(prev_stage_path):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = load_dataset(
            "json", data_files=prev_stage_path, split="train",
        )
        num_samples_before = dataset.num_rows

        # Transcribe.
        print("Transcribing...")
        dataset = dataset.map(
            transcriber.process_sample,
        )

        # Filter out samples with non-empty transcripts.
        print("Filtering out samples with non-empty transcripts...")
        dataset = dataset.filter(
            lambda sample: sample["id"] is not None,
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )
        num_samples_after = dataset.num_rows

        # Check number of samples.
        assert len(os.listdir(os.path.join(transcript_dir, channel_name))) == dataset.num_rows, \
            f"{channel_name} - Number of transcripts does not match that in dataset."
        print(f"Number of samples lost: {num_samples_before - num_samples_after}")

        # Save dataset.
        print("Saving dataset...")
        dataset.to_pandas().to_json(
            os.path.join(cur_stage_dir, channel_name + ".json"),
            orient="records",
        )
        dataset.cleanup_cache_files()
        print("-" * (13 + len(channel_name) + 2 * 20))


if __name__ == "__main__":
    main(parse_args())
