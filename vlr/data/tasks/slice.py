import os
import sys

sys.path.append(os.getcwd())

import glob
import shutil
from dataclasses import dataclass
from logging import getLogger
from datasets import Dataset, load_from_disk, disable_caching
from tqdm import tqdm
from vlr.data.processors.slicer import Slicer


logger = getLogger()
disable_caching()


@dataclass
class Args:
    """
    Data processing arguments.
    """
    prev_stage_dir = "/mnt/d/Projects/sandboxes/vlr/raw"
    cur_stage_dir = "/mnt/d/Projects/sandboxes/vlr/stage_1"
    channel_names_path = "/mnt/d/Projects/sandboxes/vlr/channels.txt"
    visual_dir = "/mnt/d/Projects/sandboxes/vlr/visual"
    audio_dir = "/mnt/d/Projects/sandboxes/vlr/audio"
    cache_dir = "~/.cache/huggingface/datasets/generator"
    duration_threshold = 1.0
    batch_size = 100
    num_proc = 8
    overwrite = False

    separator = Slicer(
        visual_dir=visual_dir,
        audio_dir=audio_dir,
        fps=25,
        duration_threshold=1.0,
        segment_duration=3.0,
        segment_overlap=1.0,
        keep_last_segment=True,
        overwrite=False,
    )


def get_paths(batch: dict, channel_names_path: str):
    """
    """
    raw_dir = batch["file"][0]
    paths = []
    with open(channel_names_path, "r") as f:
        channel_names = f.read().splitlines()
    for channel_name in channel_names:
        channel_dir = os.path.join(raw_dir, channel_name)
        chunk_dirs = glob.glob(os.path.join(channel_dir, "*"))
        for chunk_dir in chunk_dirs:
            video_paths = glob.glob(os.path.join(chunk_dir, "pyactive", "*.avi"))
            for video_path in video_paths:
                # yield {"file": video_path}
                paths.append(video_path)
    batch["file"] = paths
    return batch


def main(args: Args):
    """
    Main function.
    """
    # Prepare save directory.
    if args.overwrite:
        logger.info("Removing old files...")
        visual_paths = glob.glob(os.path.join(args.visual_dir, "*"))
        progress_bar = tqdm(
            visual_paths,
            total=len(visual_paths),
            desc="Removing old visual files",
            unit="file",
        )
        for file_path in progress_bar:
            os.remove(file_path)
        audio_paths = glob.glob(os.path.join(args.audio_dir, "*"))
        progress_bar = tqdm(
            audio_paths,
            total=len(audio_paths),
            desc="Removing old audio files",
            unit="file",
        )
        for file_path in progress_bar:
            os.remove(file_path)

    # Get dataset.
    logger.info("Preparing dataset...")
    # dataset = Dataset.from_generator(
    #     generator=get_paths,
    #     gen_kwargs={
    #         "raw_dir": args.raw_dir,
    #         "channel_names_path": args.channel_names_path
    #     },
    #     num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
    #     cache_dir=None
    # )
    dataset = Dataset.from_dict({
        "file": [args.prev_stage_dir]
    })
    dataset = dataset.map(
        get_paths,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
        load_from_cache_file=False,
        fn_kwargs={"channel_names_path": args.channel_names_path}
    )

    # Extract audio and visual.
    logger.info("Extract audio and visual from dataset...")
    print("Number of samples before slicing:", len(dataset["file"]))
    dataset = dataset.map(
        args.separator.process,
        batched=True, batch_size=args.batch_size,
        num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
    )
    print("Number of samples after slicing:", len(dataset["file"]))

    # Save dataset.
    cur_stage_versions = sorted(os.listdir(args.cur_stage_dir))
    cur_stage_prev_version = 0 if len(cur_stage_versions) == 0 else int(cur_stage_versions[-1][1])
    if not args.overwrite:
        cur_stage_prev_version_dir = os.path.join(args.cur_stage_dir, f"v{cur_stage_prev_version}")
        old_dataset = load_from_disk(cur_stage_prev_version_dir)
        file_names = old_dataset.unique("file")
        for sample in tqdm(dataset, desc="Removing duplicates", unit="sample", total=len(dataset)):
            if sample["file"] not in file_names:
                old_dataset = old_dataset.add_item(sample)
        dataset = old_dataset
    cur_stage_cur_version = cur_stage_prev_version + 1
    cur_stage_cur_version_dir = os.path.join(args.cur_stage_dir, f"v{cur_stage_cur_version}")
    dataset.save_to_disk(
        cur_stage_cur_version_dir,
        num_proc=args.num_proc if 0 < args.num_proc <= os.cpu_count() else os.cpu_count(),
    )


if __name__ == "__main__":
    main(Args())
