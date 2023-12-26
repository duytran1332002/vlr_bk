import os
import shutil
from .cropper import Cropper
from .denoiser import Denoiser
from .processor import Processor
from .slicer import Slicer
from .transcriber import Transcriber
from .uploader import Uploader
from datasets import (Dataset, disable_progress_bar, enable_progress_bar,
                      get_dataset_config_names, load_dataset)
from vlr.data.utils import TaskConfig, check_num_samples_in_dir, prepare_dir


class Executor(Processor):
    """
    This processor is used to execute other processors.
    """
    PROCESSORS = {
        "slice": Slicer,
        "denoise": Denoiser,
        "transcribe": Transcriber,
        "crop": Cropper,
    }

    def __init__(self, configs: TaskConfig) -> None:
        """
        :param processor_name:                  Name of processor.
        :param src_repo_id:                     Source repository id.
        :param dest_repo_id:                    Destination repository id.
        :param output_dir:                      Output directory.
        :param overwrite:                       Whether to overwrite existing channels.
        """
        self.configs = configs
        self.processor: Processor = self.PROCESSORS[self.configs.task]()
        self.uploader = Uploader()

        self.metadata_dir = prepare_dir(os.path.join(self.configs.output_dir, "metadata"))

        self.dataset: Dataset = None
        self.cache_dir = os.path.join(os.getcwd(), ".cache")

        self.available_channels = self.__load_channels()

    def __load_channels(self) -> list:
        """
        Load channels to process.
        :param channel_names_to_process_path:   Path to file containing channel names
                                                to process.
        :return:                                Executor.
        """
        # Get available channel names.
        available_channels = set(get_dataset_config_names(self.configs.src_repo_id)) - {"all"}

        # Get channel names to process.
        new_channels = set()
        if self.configs.channel_names_path:
            with open(self.configs.channel_names_path, "r") as f:
                new_channels = set(f.read().split())

        available_channels = available_channels.intersection(new_channels)
        if not self.configs.overwrite and not self.configs.upload_to_hub:
            existing_channels = set(get_dataset_config_names(self.configs.dest_repo_id)) - {"all"}
            available_channels -= existing_channels

        return list(available_channels)

    def prepare_dir(self, channel: str) -> None:
        """
        Prepare directory.
        :param channel:     Channel name.
        """
        self.configs = self.configs.prepare_dir(
            channel=channel, overwrite=self.configs.overwrite
        )

    def load_dataset(
        self, channel: str,
    ) -> Processor:
        """
        Load dataset.
        :param channel:     Channel name.
        :return:            Executor.
        """
        disable_progress_bar()
        self.dataset = load_dataset(
            self.configs.src_repo_id, channel,
            split="train",
            cache_dir=self.cache_dir,
        )
        if self.configs.remove_columns_loading:
            self.dataset = self.dataset.remove_columns(
                self.configs.remove_columns_loading
            )
        enable_progress_bar()

        self.num_samples_before = self.dataset.num_rows
        self.num_samples_after = 0
        return self

    def process(self) -> Processor:
        """
        Process sample.
        :param fn_kwargs:           Keyword arguments for function.
        :param num_proc:            Number of processes.
        :param remove_columns:      Columns to remove.
        :return:                    Executor.
        """
        assert self.dataset is not None, "Dataset is not loaded yet."

        task_kwargs = self.configs.get_task_kwargs()
        self.dataset = self.dataset.map(
            self.processor.process,
            fn_kwargs=task_kwargs["fn_kwargs"],
            batched=True, batch_size=1,
            num_proc=task_kwargs["num_proc"],
            remove_columns=task_kwargs["remove_columns"],
            load_from_cache_file=not self.configs.overwrite,
        )
        disable_progress_bar()
        self.dataset = self.dataset.filter(
            lambda sample: sample["id"] is not None,
            num_proc=os.cpu_count(),
            load_from_cache_file=not self.configs.overwrite,
        )
        enable_progress_bar()
        self.num_samples_after = self.dataset.num_rows
        return self

    def check_num_samples_in_dir(self) -> None:
        """
        Check if number of samples in directory matches expected number of samples.
        :param dir_path:    Path to directory.
        """
        assert self.dataset is not None, "Dataset is not loaded yet."

        for data_dir in self.configs.schema_dict.values():
            check_num_samples_in_dir(
                dir_path=data_dir,
                num_samples=self.num_samples_after,
            )

    def print_num_samples_change(self):
        """
        Get number of samples lost.
        """
        self.configs.print_num_samples_change(
            abs(self.num_samples_after - self.num_samples_before)
        )

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

    def upload_to_hub(self, channel: str) -> None:
        """
        Upload to hub.
        :param channel:     Channel name.
        """
        if self.configs.upload_to_hub:
            print("Uploading to hub...")
            self.__upload_metadata_to_hub(channel=channel)
            for schema, data_dir in self.configs.schema_dict.items():
                self.__zip_and_upload_dir(
                    dir_path=data_dir,
                    path_in_repo=os.path.join(schema, channel + ".zip"),
                )

    def __upload_metadata_to_hub(
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
            repo_id=self.configs.dest_repo_id,
            path_in_repo=os.path.join("metadata", channel + ".parquet"),
            overwrite=overwrite,
        )

    def __zip_and_upload_dir(
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
            repo_id=self.configs.dest_repo_id,
            path_in_repo=path_in_repo,
            overwrite=overwrite,
        )

    def clean_cache(self) -> None:
        """
        Clean cache.
        """
        if self.configs.clean_up:
            print("Cleaning up...")
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
