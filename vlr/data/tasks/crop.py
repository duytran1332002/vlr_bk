import os
import sys

sys.path.append(os.getcwd())

import argparse
from tqdm import tqdm
from vlr.data.utils.file_system import prepare_dir
from vlr.data.processors import Executor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
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
        "--version",
        type=int,
        default=1,
        help="version of the dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--upload-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload to hub after processing.",
    )
    parser.add_argument(
        "--clean-up",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clean up all downloaded files after processing.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    This function is used to crop mouth regions from videos.
    :param args:    Arguments from command line.
    """
    print("Initialize executor...")
    executor = Executor(
        processor_name="cropper",
        src_repo_id="fptu/vietnamese-speaker-clip",
        dest_repo_id="fptu/vietnamese-speaker-lip-clip",
        output_dir=args.output_dir,
    )
    executor = executor.load_channels(
        channel_names_to_process_path=args.channel_names_path,
        overwrite=args.overwrite,
    )

    for channel in tqdm(
        executor.available_channels,
        desc="Processing channels",
        total=len(executor.available_channels),
        unit="channel"
    ):
        print("-" * 20 + f" Processing {channel} " + "-" * 20)

        # Prepare save directory.
        print("Preparing save directory...")
        channel_visual_dir = prepare_dir(
            dir=os.path.join(args.output_dir, "visual", channel),
            overwrite=args.overwrite,
        )

        # Get dataset.
        print("Preparing dataset...")
        executor = executor.load_dataset(
            channel=channel,
            remove_columns=["audio"],
        )

        # Crop mouth regions.
        print("Cropping mouth regions...")
        executor = executor.process_sample(
            fn_kwargs={
                "visual_output_dir": channel_visual_dir,
                "padding": 20,
            },
            remove_columns=["visual"],
        )

        # Check number of samples.
        print("Checking number of samples...")
        executor.check_num_samples_in_dir(channel_visual_dir)
        print(f"\tNumber of samples lost: {executor.get_num_samples_change()}")

        # Save dataset.
        print("Saving metada...")
        executor.save_metadata(channel)

        # Upload to hub.
        if args.upload_to_hub:
            print("Uploading to hub...")
            executor.upload_metadata_to_hub(channel=channel)
            executor.zip_and_upload_dir(
                dir_path=channel_visual_dir,
                path_in_repo=os.path.join("visual", channel + ".zip"),
            )

        print("-" * (13 + len(channel) + 2 * 20))

    if args.clean_up:
        print("Cleaning up...")
        executor.clean_cache()


if __name__ == "__main__":
    main(parse_args())
