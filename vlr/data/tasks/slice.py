import os
import sys

sys.path.append(os.getcwd())

import glob
import argparse
from tqdm import tqdm
from datasets import Dataset, disable_caching
from vlr.data.processors.slicer import Slicer
from vlr.data.utils.tools import clean_up


disable_caching()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active-speaker-dir",
        type=str,
        required=True,
        help="Path to active speaker data directory.",
    )
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
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def initial_dataset(channel_dir: str, channel_name: str) -> dict:
    """
    Initial dataset.
    :param channel_dir:             Path to channel directory.
    :param channel_name:            Channel name.
    :return:                        Dataset.
    """
    files = []
    chunk_dirs = glob.glob(os.path.join(channel_dir, "*"))
    for chunk_dir in chunk_dirs:
        video_paths = glob.glob(os.path.join(chunk_dir, "pyactive", "*.avi"))
        for video_path in video_paths:
            files.append(video_path)
    return {
        "file": files,
        "channel": [channel_name] * len(files),
    }


def main(args: argparse.Namespace):
    """
    Main function.
    """
    if not os.path.exists(args.active_speaker_dir):
        print(f"Directory {args.active_speaker_dir} does not exist.")
        return

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    visual_dir = os.path.join(args.data_dir, "visual")
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    audio_dir = os.path.join(args.data_dir, "audio")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    cur_stage_dir = os.path.join(args.data_dir, "stage_1")
    if not os.path.exists(cur_stage_dir):
        os.makedirs(cur_stage_dir)

    if args.channel_names_path:
        with open(args.channel_names_path, "r") as f:
            channel_names = f.read().strip().split()
    else:
        channel_names = os.listdir(args.active_speaker_dir)

    separator = Slicer(
        visual_dir=visual_dir,
        audio_dir=audio_dir,
        fps=25,
        duration_threshold=1.0,
        segment_duration=3.0,
        segment_overlap=1.0,
        keep_last_segment=True,
        overwrite=args.overwrite,
    )

    print("\n" + "#" * 50 + " Slicing " + "#" * 50)
    for channel_name in tqdm(
        channel_names,
        desc="Processing channels",
        total=len(channel_names),
        unit="channel"
    ):
        print("-" * 20 + f" Processing {channel_name} " + "-" * 20)
        # Prepare save directory.
        clean_up(channel_name, [visual_dir, audio_dir], args.overwrite)

        # Get dataset.
        print("Preparing dataset...")
        prev_stage_dir = os.path.join(args.active_speaker_dir, channel_name)
        if not os.path.exists(prev_stage_dir):
            print(f"Channel {channel_name} does not exist.")
            continue
        dataset = Dataset.from_dict(initial_dataset(prev_stage_dir, channel_name))
        num_samples_before = dataset.num_rows

        # Extract audio and visual.
        print("Extracting audio and visual from dataset...")
        dataset = dataset.map(
            separator.process_batch,
            batched=True, batch_size=args.batch_size,
            num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
            remove_columns=["file"],
        )
        num_samples_after = dataset.num_rows

        # Check number of samples.
        print("Checking number of samples...")
        num_visual_samples = len(os.listdir(os.path.join(visual_dir, channel_name)))
        assert num_visual_samples == dataset.num_rows, \
            f"{channel_name} - Number of visual samples does not match that in dataset."
        num_audio_samples = len(os.listdir(os.path.join(audio_dir, channel_name)))
        assert num_audio_samples == dataset.num_rows, \
            f"{channel_name} - Number of audio samples does not match that in dataset."
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
