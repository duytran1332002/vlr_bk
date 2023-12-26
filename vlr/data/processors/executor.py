import os
import shutil
from typing import Union
from datasets import Dataset, load_dataset, get_dataset_config_names
from datasets import enable_progress_bar, disable_progress_bar
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
        overwrite: bool = False,
    ) -> None:
        """
        :param processor_name:                  Name of processor.
        :param src_repo_id:                     Source repository id.
        :param dest_repo_id:                    Destination repository id.
        :param output_dir:                      Output directory.
        :param processor_kwargs:                Keyword arguments for processor.
        :param overwrite:                       Whether to overwrite existing channels.
        """
        self.processor: Processor = self.PROCESSORS[processor_name](**processor_kwargs)
        self.uploader = Uploader()
        self.src_repo_id = src_repo_id
        self.dest_repo_id = dest_repo_id

        self.metadata_dir = prepare_dir(os.path.join(output_dir, "metadata"))

        self.dataset: Dataset = None
        self.cache_dir = os.path.join(os.getcwd(), ".cache")
        self.overwrite = overwrite

    def load_channels(
        self, channel_names_to_process_path: str = None,
    ) -> Processor:
        """
        Load channels to process.
        :param channel_names_to_process_path:   Path to file containing channel names
                                                to process.
        :return:                                Executor.
        """
        # Get available channel names.
        self.available_channels = set(get_dataset_config_names(self.src_repo_id)) - {"all"}

        # Get channel names to process.
        new_channels = set()
        if channel_names_to_process_path:
            with open(channel_names_to_process_path, "r") as f:
                new_channels = set(f.read().split())

        self.available_channels = self.available_channels.intersection(new_channels)
        if not self.overwrite:
            existing_channels = set(get_dataset_config_names(self.dest_repo_id)) - {"all"}
            self.available_channels -= existing_channels

        self.available_channels = list(self.available_channels)
        return self

    def load_dataset(
        self, channel: str,
        remove_columns: Union[str, list[str]] = None,
    ) -> Processor:
        """
        Load dataset.
        :param channel:     Channel name.
        :param remove_columns
        :return:            Executor.
        """
        disable_progress_bar()
        self.dataset = load_dataset(
            self.src_repo_id, channel,
            split="train",
            cache_dir=self.cache_dir,
        )
        if remove_columns:
            self.dataset = self.dataset.remove_columns(remove_columns)
        enable_progress_bar()

        self.num_samples_before = self.dataset.num_rows
        self.num_samples_after = 0
        return self

    def process(
        self, fn_kwargs: dict,
        num_proc: int = 1,
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
            self.processor.process,
            fn_kwargs=fn_kwargs,
            batched=True, batch_size=1,
            num_proc=num_proc,
            remove_columns=remove_columns,
            load_from_cache_file=not self.overwrite,
        )
        self.dataset = self.dataset.filter(
            lambda sample: sample["id"] is not None,
            num_proc=os.cpu_count(),
            load_from_cache_file=not self.overwrite,
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

    def get_num_samples_change(self) -> int:
        """
        Get number of samples lost.
        """
        return abs(self.num_samples_after - self.num_samples_before)

    def save_metadata(self, channel: str) -> None:
        """
        Save metadata as parquet file and save channel name to file.
        :param channel:     Channel name.
        """
        assert self.dataset is not None, "Dataset is not loaded yet."

        metadata_path = os.path.join(self.metadata_dir, channel + ".parquet")
        disable_progress_bar()
        self.dataset.to_parquet(metadata_path)
        enable_progress_bar()

    def upload_metadata_to_hub(
        self, channel: str,
        overwrite: bool = True,
    ) -> None:
        """
        Upload metadata and channel names to hub.
        :param channel:     Channel name.
        """
        metadata_path = os.path.join(self.metadata_dir, channel + ".parquet")
        self.uploader.upload_file(
            file_path=metadata_path,
            repo_id=self.dest_repo_id,
            path_in_repo=os.path.join("metadata", channel + ".parquet"),
            overwrite=overwrite,
        )

    def zip_and_upload_dir(
        self, dir_path: str,
        path_in_repo: str,
        overwrite: bool = True,
    ) -> None:
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
