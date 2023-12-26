# Copyright 2023 Thinh T. Duong
import os
import datasets
from huggingface_hub import HfFileSystem


logger = datasets.logging.get_logger(__name__)


_CITATION = """

"""
_DESCRIPTION = """

"""
_HOMEPAGE = "https://github.com/duytran1332002/vlr"
_META_REPO_PATH = "datasets/fptu/purified-vietnamese-audio"
_VISUAL_REPO_PATH = "datasets/fptu/vietnamese-speaker-lip-clip"
_AUDIO_REPO_PATH = "datasets/fptu/vietnamese-denoised-audio"
_REPO_URL = "https://huggingface.co/{}/resolve/main"
_URLS = {
    "meta": f"{_REPO_URL}/metadata/".format(_META_REPO_PATH) + "{channel}.parquet",
    "visual": f"{_REPO_URL}/visual/".format(_VISUAL_REPO_PATH) + "{channel}.zip",
    "audio": f"{_REPO_URL}/audio/".format(_AUDIO_REPO_PATH) + "{channel}.zip",
    "transcript": f"{_REPO_URL}/transcript/".format(_META_REPO_PATH) + "{channel}.zip",
}
_CONFIGS = [
    os.path.basename(file_name)[:-8]
    for file_name in HfFileSystem().listdir(_META_REPO_PATH + "/metadata", detail=False)
]
_CONFIGS.append("all")


class VLRConfig(datasets.BuilderConfig):
    """VLR configuration."""

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


class VLR(datasets.GeneratorBasedBuilder):
    """VLR dataset."""

    BUILDER_CONFIGS = [VLRConfig(name) for name in _CONFIGS]

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "visual": datasets.Value("binary"),
            "duration": datasets.Value("float64"),
            "fps": datasets.Value("int8"),
            "audio": datasets.Value("binary"),
            "sampling_rate": datasets.Value("int64"),
            "transcript": datasets.Value("string"),
        })

        return datasets.DatasetInfo(
            description=_DESCRIPTION[self.config.name],
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
        visual_dirs = dl_manager.download_and_extract(
            [_URLS["video"].format(channel=channel) for channel in config_names]
        )

        audio_dirs = dl_manager.download_and_extract(
            [_URLS["audio"].format(channel=channel) for channel in config_names]
        )
        transcript_dirs = dl_manager.download_and_extract(
            [_URLS["transcript"].format(channel=channel) for channel in config_names]
        )

        visual_dict = {
            channel: visual_dir
            for channel, visual_dir in zip(config_names, visual_dirs)
        }
        audio_dict = {
            channel: audio_dir
            for channel, audio_dir in zip(config_names, audio_dirs)
        }
        transcript_dict = {
            channel: transcript_dir
            for channel, transcript_dir in zip(config_names, transcript_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_paths": metadata_paths,
                    "visual_dict": visual_dict,
                    "audio_dict": audio_dict,
                    "transcript_dict": transcript_dict,
                },
            ),
        ]

    def _generate_examples(
        self, metadata_paths: list[str],
        visual_dict: dict,
        audio_dict: dict,
        transcript_dict: dict,
    ) -> tuple[int, dict]:
        """
        Generate examples.
        :param metadata_paths:          Paths to metadata files.
        :param visual_dict:             Paths to directory containing visual files.
        :param audio_dict:              Paths to directory containing audio files.
        :param transcript_dict:         Paths to directory containing transcripts.
        :return:                        Example.
        """
        dataset = datasets.load_dataset(
            "parquet",
            data_files=metadata_paths,
            split="train",
        )
        for i, sample in enumerate(dataset):
            channel = sample["channel"]
            visual_path = os.path.join(
                visual_dict[channel], channel, sample["id"] + ".mp4"
            )
            audio_path = os.path.join(
                audio_dict[channel], channel, sample["id"] + ".wav"
            )
            transcript_path = os.path.join(
                transcript_dict[channel], channel, sample["id"] + ".txt"
            )

            yield i, {
                "id": sample["id"],
                "channel": channel,
                "visual": self.__get_binary_data(visual_path),
                "duration": sample["duration"],
                "fps": sample["fps"],
                "audio": self.__get_binary_data(audio_path),
                "sampling_rate": sample["sampling_rate"],
                "transcript": self.__get_text_data(transcript_path),
            }

    def __get_binary_data(self, path: str) -> bytes:
        """
        Get binary data from path.
        :param path:    Path to file.
        :return:        Binary data.
        """
        with open(path, "rb") as f:
            return f.read()

    def __get_text_data(self, path: str) -> str:
        """
        Get transcript from path.
        :param path:     Path to transcript.
        :return:         Transcript.
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
