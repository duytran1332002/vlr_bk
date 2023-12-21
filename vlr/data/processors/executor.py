import os
import shutil
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
    """
    This processor is used to execute other processors.
    """
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
        """
        :param processor_name:                  Name of processor.
        :param src_repo_id:                     Source repository id.
        :param dest_repo_id:                    Destination repository id.
        :param output_dir:                      Output directory.
        :param processor_kwargs:                Keyword arguments for processor.
        :param channel_names_to_process_path:   Path to file containing channel names
                                                to process.
        """
        self.processor: Processor = self.PROCESSORS[processor_name](**processor_kwargs)
        self.uploader = Uploader()
        self.src_repo_id = src_repo_id
        self.dest_repo_id = dest_repo_id

        self.metadata_dir = prepare_dir(os.path.join(output_dir, "metadata"))
        self.channel_names_path = os.path.join(output_dir, "channel_names.txt")

        self.available_channels, self._existing_channels = self._load_channels(
            channel_names_to_process_path=channel_names_to_process_path,
        )
        self.dataset: Dataset = None
        self.cache_dir = os.path.join(os.getcwd(), ".cache")

    def _load_channels(self, channel_names_to_process_path: str = None) -> Processor:
        """
        Load channels to process.
        :param channel_names_to_process_path:   Path to file containing channel names
                                                to process.
        :return:                                Executor.
        """
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
        """
        Load dataset.
        :param channel:     Channel name.
        :return:            Executor.
        """
        self.dataset = load_dataset(
            self.src_repo_id, channel,
            split="train",
            cache_dir=self.cache_dir,
        )
        self.num_samples_before = self.dataset.num_rows
        self.num_samples_after = 0
        return self

    def process_sample(
        self, fn_kwargs: dict,
        num_proc: int = None,
        remove_columns: Union[str, list[str]] = None,
    ) -> Processor:
        """
        Process sample.
        :param fn_kwargs:           Keyword arguments for function.
        :param num_proc:            Number of processes.
        :param remove_columns:      Columns to remove.
        :return:                    Executor.
        """
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
        """
        Process batch.
        :param fn_kwargs:           Keyword arguments for function.
        :param batch_size:          Batch size.
        :param num_proc:            Number of processes.
        :param remove_columns:      Columns to remove.
        """
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
        """
        Check if number of samples in directory matches expected number of samples.
        :param dir_path:    Path to directory.
        """
        assert self.dataset is not None, "Dataset is not loaded yet."

        check_num_samples_in_dir(
            dir_path=dir_path,
            num_samples=self.num_samples_after,
        )

    def get_num_samples_lost(self) -> int:
        """
        Get number of samples lost.
        """
        return self.num_samples_before - self.num_samples_after

    def save_metadata(self, channel: str) -> None:
        """
        Save metadata as parquet file and save channel name to file.
        :param channel:     Channel name.
        """
        assert self.dataset is not None, "Dataset is not loaded yet."

        metadata_path = os.path.join(self.metadata_dir, channel + ".parquet")
        self.dataset.to_parquet(metadata_path)

        self._existing_channels.append(channel)
        with open(self.channel_names_path, "w") as f:
            f.write("\n".join(self._existing_channels))

    def upload_metadata_to_hub(self, channel: str) -> None:
        """
        Upload metadata and channel names to hub.
        :param channel:     Channel name.
        """
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
        """
        Zip directory and upload it to the hub.
        :param dir_path:        Path to directory.
        :param path_in_repo:    Path to directory in repository.
        """
        self.uploader.zip_and_upload_dir(
            dir_path=dir_path,
            repo_id=self.dest_repo_id,
            path_in_repo=path_in_repo,
            overwrite=overwrite,
        )

    def clean_cache(self) -> None:
        """
        Clean cache.
        """
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
