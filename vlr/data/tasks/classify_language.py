import os
import sys

sys.path.append(os.getcwd())

import argparse
from datasets import load_dataset, disable_caching
from tqdm import tqdm
from vlr.data.processors.language_classifier import LanguageClassifier


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
        "--num-proc",
        type=int,
        default=-1,
        help="Number of processes.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
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
    prev_stage_dir = os.path.join(args.data_dir, "stage_2")
    if not os.path.exists(prev_stage_dir):
        print(f"Directory {prev_stage_dir} does not exist.")
        return

    cur_stage_dir = os.path.join(args.data_dir, "stage_3")
    if not os.path.exists(cur_stage_dir):
        os.makedirs(cur_stage_dir)

    if args.channel_names_path:
        with open(args.channel_names_path, "r") as f:
            channel_names = f.read().strip().split()
    else:
        channel_names = os.listdir(denoised_dir)

    language_classifier = LanguageClassifier(
        audio_dir=denoised_dir,
    )

    print("\n" + "#" * 50 + " Classifying language " + "#" * 50)
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        print("-" * 20 + f" Processing {channel_name} " + "-" * 20)

        # Get dataset.
        print("Preparing dataset...")
        prev_stage_path = os.path.join(prev_stage_dir, channel_name + ".json")
        if not os.path.exists(prev_stage_path):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = load_dataset(
            "json", data_files=prev_stage_path, split="train",
        )

        # Classify language.
        print("Classifying language...")
        print(f"Number of samples before: {dataset.num_rows}")
        dataset = dataset.map(
            language_classifier.process_batch,
            batched=True,
            batch_size=args.batch_size,
        )

        # Filter out samples with Vietnamese language.
        print("Filtering out samples with Vietnamese language...")
        dataset = dataset.filter(
            lambda sample: sample["id"] is not None,
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        )
        print(f"Number of samples after: {dataset.num_rows}")

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
