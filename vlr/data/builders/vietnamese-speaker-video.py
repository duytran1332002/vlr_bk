# Copyright 2023 Thinh T. Duong
import os
import datasets
from requests import get
from huggingface_hub import HfFolder


logger = datasets.logging.get_logger(__name__)


_CITATION = """

"""
_DESCRIPTION = """
    This dataset contain videos of Vietnamese speakers.
"""
_HOMEPAGE = "https://github.com/duytran1332002/vlr"
_REPO = "https://huggingface.co/datasets/fptu/vietnamese-speaker-video/resolve/main"
_URLS = {
    "channels": f"{_REPO}/channels.txt",
    "meta": f"{_REPO}/metadata/" + "{channel}.json",
    "video": f"{_REPO}/video/" + "{channel}.zip",
}
_HEADERS = {
    "Authorization": f"Bearer {HfFolder.get_token()}",
}
_CONFIGS = list(set([
    x.decode("UTF8") for x in get(_URLS["channels"], headers=_HEADERS).iter_lines()
]))
_CONFIGS.append("all")


class VietnameseSpeakerVideoConfig(datasets.BuilderConfig):
    """Vietnamese Speaker Video configuration."""

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


class VietnameseSpeakerVideo(datasets.GeneratorBasedBuilder):
    """Vietnamese Speaker Video dataset."""

    BUILDER_CONFIGS = [VietnameseSpeakerVideoConfig(name) for name in _CONFIGS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "id": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "video": datasets.Value("string"),
            "fps": datasets.Value("int8"),
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
        video_dirs = dl_manager.download_and_extract(
            [_URLS["video"].format(channel=channel) for channel in config_names]
        )

        video_dict = {
            channel: video_dir for channel, video_dir in zip(config_names, video_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_paths": metadata_paths,
                    "video_dict": video_dict,
                },
            ),
        ]

    def _generate_examples(
        self, metadata_paths: list[str],
        video_dict: dict,
    ) -> tuple[int, dict]:
        """
        Generate examples from metadata.
        :param meta_paths:      Paths to metadata.
        :param video_dict:      Paths to directory containing videos.
        :yield:                 Example.
        """
        dataset = datasets.load_dataset(
            "json",
            data_files=metadata_paths,
            split="train",
            num_proc=os.cpu_count(),
        )
        for i, sample in enumerate(dataset):
            channel = sample["channel"]
            video_path = os.path.join(
                video_dict[channel], channel, sample["id"] + ".avi"
            )

            yield i, {
                "id": sample["id"],
                "channel": channel,
                "video": video_path,
                "fps": sample["fps"],
            }
