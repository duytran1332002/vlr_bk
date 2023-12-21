import os
from typing import Union
from datasets import Dataset, load_dataset, get_dataset_config_names
from .processor import Processor
from .slicer import Slicer
from .denoiser import Denoiser
from .language_classifier import LanguageClassifier
from .transcriber import Transcriber
from .cropper import Cropper
from .uploader import Uploader
from vlr.data.utils import prepare_dir, check_num_samples_in_dir


class Executor(Processor):
    PROCESSORS = {
        "slicer": Slicer,
        "denoiser": Denoiser,
        "language_classifier": LanguageClassifier,
        "transcriber": Transcriber,
        "cropper": Cropper,
    }

    def __init__(
        self, processor_name: str,
        src_repo_id: str,
        dest_repo_id: str,
        output_dir: str,
        processor_kwargs: dict = {},
        channel_names_to_process_path: str = None,
    ) -> None:
        self.processor: Processor = self.PROCESSORS[processor_name](**processor_kwargs)
        self.uploader = Uploader()
        self.src_repo_id = src_repo_id
        self.dest_repo_id = dest_repo_id

        self.metadata_dir = prepare_dir(os.path.join(output_dir, "metadata"))
        self.channel_names_path = os.path.join(output_dir, "channel_names.txt")

        self.available_channels, self._existing_channels = self.load_channels(
            channel_names_to_process_path=channel_names_to_process_path,
        )
        self.dataset: Dataset = None

    def _load_channels(self, channel_names_to_process_path: str = None) -> Processor:
        # Get available channel names.
        available_channels = set(get_dataset_config_names(self.src_repo_id))

        # Get existing channel names.
        existing_channels = set(get_dataset_config_names(self.dest_repo_id))

        # Get channel names to process.
        new_channels = set()
        if channel_names_to_process_path:
            with open(channel_names_to_process_path, "r") as f:
                new_channels = set(f.read().split())

        available_channels = available_channels.intersection(new_channels)
        available_channels -= existing_channels
        return list(available_channels), list(existing_channels)

    def load_dataset(self, channel: str) -> Processor:
        self.dataset = load_dataset(
            self.src_repo_id, channel,
            split="train",
        )
        self.num_samples_before = self.dataset.num_rows
        self.num_samples_after = 0
        return self

    def process_sample(
        self, fn_kwargs: dict,
        num_proc: int = None,
        remove_columns: Union[str, list[str]] = None,
    ) -> Processor:
        assert self.dataset is not None, "Dataset is not loaded yet."

        self.dataset = self.dataset.map(
            self.processor.process_sample,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            remove_columns=remove_columns,
        )
        self.dataset = self.dataset.filter(
            lambda sample: sample["id"] is not None,
            num_proc=os.cpu_count(),
        )
        self.num_samples_after = self.dataset.num_rows
        return self

    def process_batch(
        self, fn_kwargs: dict,
        batch_size: int,
        num_proc: int = None,
        remove_columns: Union[str, list[str]] = None,
    ) -> Processor:
        assert self.dataset is not None, "Dataset is not loaded yet."

        self.dataset = self.dataset.map(
            self.processor.process_batch,
            fn_kwargs=fn_kwargs,
            batched=True, batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=remove_columns,
        )
        self.dataset = self.dataset.filter(
            lambda sample: sample["id"] is not None,
            num_proc=os.cpu_count(),
        )
        self.num_samples_after = self.dataset.num_rows
        return self

    def check_num_samples_in_dir(self, dir_path: str) -> None:
        assert self.dataset is not None, "Dataset is not loaded yet."

        check_num_samples_in_dir(
            dir_path=dir_path,
            num_samples=self.num_samples_after,
        )

    def get_num_samples_lost(self) -> int:
        return self.num_samples_before - self.num_samples_after

    def save_metadata(self, channel: str) -> None:
        assert self.dataset is not None, "Dataset is not loaded yet."

        metadata_path = os.path.join(self.metadata_dir, channel + ".parquet")
        self.dataset.to_parquet(metadata_path)

        self._existing_channels.append(channel)
        with open(self.channel_names_path, "w") as f:
            f.write("\n".join(self._existing_channels))

    def upload_metadata_to_hub(self, channel: str):
        metadata_path = os.path.join(self.metadata_dir, channel + ".parquet")
        self.uploader.upload_file(
            file_path=metadata_path,
            repo_id=self.dest_repo_id,
            path_in_repo=os.path.join("metadata", channel + ".parquet"),
        )
        self.uploader.upload_file(
            file_path=self.channel_names_path,
            repo_id=self.dest_repo_id,
            path_in_repo=os.path.join("vietnamese-speaker-clip", "channels.txt"),
        )

    def zip_and_upload_dir(self, dir_path: str, path_in_repo: str) -> None:
        self.uploader.zip_and_upload_dir(
            dir_path=dir_path,
            repo_id=self.dest_repo_id,
            path_in_repo=path_in_repo,
        )
