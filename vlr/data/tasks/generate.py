import os
import sys
sys.path.append(os.getcwd())

import glob
import argparse
from dataclasses import dataclass
from datasets import IterableDataset, Dataset, Audio, Features
from vlr.data.processors.speech_to_text import SpeechToText
from logging import getLogger


logger = getLogger(__name__)


@dataclass
class Args:
    """
    Data processing arguments.
    """
    parser = argparse.ArgumentParser(description="Data processing arguments.")
    parser.add_argument(
        "--data_dir", type=str,
        default="/mnt/d/Projects/sandboxes/vlr/raw/20220701_1.wav",
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="/mnt/d/Projects/sandboxes/vlr/processed",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    streaming = False
    sampling_rate = 16000

    speech2text = SpeechToText(
        model_path=os.getcwd() + "/vlr/data/resources/wav2vec2-base-vietnamese-250h",
        lm_gram_name="vi_lm_4grams.bin.zip",
        device="cpu",
        mode="sentence",
        segment_duration=10.0,
        segment_overlap=1.0,
        keep_last_segment=True,
    )

    processors = (
        speech2text,
    )


def generator(file_paths: list):
    """
    Generator for streaming mode.
    :param file_paths:  list of file paths.
    """
    for file_path in file_paths:
        yield {"audio": file_path}


def get_dataset(data_dir: str, streaming: bool, sampling_rate: int):
    """
    Get dataset from path.
    :param data_dir:        data directory.
    :param streaming:       streaming mode.
    :param sampling_rate:   sampling rate.
    :return:                dataset.
    """
    file_paths = glob.glob(data_dir + "/*") if os.path.isdir(data_dir) else [data_dir]

    if streaming:
        dataset = IterableDataset.from_generator(
            generator=generator,
            gen_kwargs={"file_paths": file_paths},
            features=Features({
                "audio": Audio(sampling_rate=sampling_rate)
            })
        )
    else:
        dataset = Dataset.from_dict({"audio": file_paths})
    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))


def process(sample: dict, processors: list):
    """
    Process sample.
    :param sample:      sample.
    :param processors:  processors.
    :return:            processed sample.
    """
    for processor in processors:
        sample = processor.process(sample)
    # Remove audio path.
    sample["audio"] = {
        "array": sample["audio"]["array"],
        "sampling_rate": sample["audio"]["sampling_rate"],
    }
    return sample


def main(args: Args):
    """
    Main function.
    :param args:    arguments.
    """
    # Prepare dataset.
    logger.info("Preparing dataset...")
    dataset = get_dataset(args.data_dir, args.streaming, args.sampling_rate)

    # Process dataset.
    logger.info("Processing dataset...")
    dataset = dataset.map(lambda sample: process(sample, args.processors))
    logger.info("Dataset processed. Saving...")
    dataset.save_to_disk(args.save_dir)


if __name__ == "__main__":
    main(Args())

    # @staticmethod
    # def sec_to_timecode(x: float) -> str:
    #     '''
    #     Calculate timecode from second

    #     Parameters:
    #         x: float
    #             Second
    #     '''
    #     hour, x = divmod(x, 3600)
    #     minute, x = divmod(x, 60)
    #     second, x = divmod(x, 1)
    #     millisecond = int(x * 1000.)
    #     return '%.2d:%.2d:%.2d,%.3d' % (hour, minute, second, millisecond)
