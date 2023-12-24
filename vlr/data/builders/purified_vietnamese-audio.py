# Copyright 2023 Thinh T. Duong
import os
import datasets
from requests import get
from huggingface_hub import HfFolder


logger = datasets.logging.get_logger(__name__)


_CITATION = """

"""
_DESCRIPTION = """
    This dataset contain denoised audio of Vietnamese speakers.
"""
_HOMEPAGE = "https://github.com/duytran1332002/vlr"
_MAIN_REPO = "https://huggingface.co/datasets/fptu/purified-vietnamese-audio/resolve/main"
_AUDIO_REPO = "https://huggingface.co/datasets/fptu/denoised-vietnamese-audio/resolve/main"
_URLS = {
    "channels": f"{_MAIN_REPO}/channels.txt",
    "meta": f"{_MAIN_REPO}/metadata/" + "{channel}.parquet",
    "audio": f"{_AUDIO_REPO}/audio/" + "{channel}.zip",
}
_HEADERS = {
    "Authorization": f"Bearer {HfFolder.get_token()}",
}
_CONFIGS = list(set([
    x.decode("UTF8") for x in get(_URLS["channels"], headers=_HEADERS).iter_lines()
]))
_CONFIGS.append("all")


class PurifiedVietnameseAudioConfig(datasets.BuilderConfig):
    """Vietnamese Purified Audio configuration."""

    def __init__(self, name, **kwargs):
        """
        :param name:    Name of subset.
        :param kwargs:  Arguments.
        """
        super().__init__(
            name=name,
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
            **kwargs,
        )


class PurifiedVietnameseAudio(datasets.GeneratorBasedBuilder):
    """Vietnamese Purified Audio dataset."""

    BUILDER_CONFIGS = [PurifiedVietnameseAudioConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "audio": datasets.Value("binary"),
            "sampling_rate": datasets.Value("int64"),
        })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        """
        Get splits.
        :param dl_manager:  Download manager.
        :return:            Splits.
        """
        config_names = _CONFIGS[:-1] if self.config.name == "all" else [self.config.name]

        metadata_paths = dl_manager.download(
            [_URLS["meta"].format(channel=channel) for channel in config_names]
        )
        audio_dirs = dl_manager.download_and_extract(
            [_URLS["audio"].format(channel=channel) for channel in config_names]
        )

        audio_dict = {
            channel: audio_dir for channel, audio_dir in zip(config_names, audio_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_paths": metadata_paths,
                    "audio_dict": audio_dict,
                },
            ),
        ]

    def _generate_examples(
        self, metadata_paths: list[str],
        audio_dict: dict,
    ) -> tuple[int, dict]:
        """
        Generate examples from metadata.
        :param meta_paths:      Paths to metadata.
        :param audio_dict:      Paths to directory containing audios.
        :yield:                 Example.
        """
        dataset = datasets.load_dataset(
            "parquet",
            data_files=metadata_paths,
            split="train",
            num_proc=os.cpu_count(),
        )
        for i, sample in enumerate(dataset):
            channel = sample["channel"]
            audio_path = os.path.join(
                audio_dict[channel], channel, sample["id"] + ".wav"
            )

            yield i, {
                "id": sample["id"],
                "channel": channel,
                "audio": self.__get_binary_data(audio_path),
                "sampling_rate": sample["sampling_rate"],
            }

    def __get_binary_data(self, path: str) -> bytes:
        """
        Get binary data from path.
        :param path:    Path to file.
        :return:        Binary data.
        """
        with open(path, "rb") as f:
            return f.read()
