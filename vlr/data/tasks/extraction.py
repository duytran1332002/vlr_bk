import os
import sys
sys.path.append(os.getcwd())

import glob
from dataclasses import dataclass
from logging import getLogger
from tqdm import tqdm
from datasets import load_from_disk
from vlr.data.processors.transcriber import Transcriber
from vlr.data.processors.denoiser import Denoiser
from vlr.data.processors.cropper import Cropper


logger = getLogger(__name__)


@dataclass
class Args:
    """
    Data processing arguments.
    """

    data_dir = "/mnt/d/Projects/sandboxes/vlr/raw"
    save_dir = "/mnt/d/Projects/sandboxes/vlr/processed"
    stage = 4
    assert 4 <= stage <= 6, "Stages must be in range [4, 6]."
    batch_size = 100
    num_proc = 8

    denoiser_kwags = {
        
    }
    transcriber_kwags = {
        "model_path": os.getcwd() + "/vlr/data/resources/wav2vec2-base-vietnamese-250h",
        "lm_gram_name": "vi_lm_4grams.bin.zip",
        "device": "cuda",
        "mode": "sentence",
    }
    cropper_kwags = {
        
    }
    processors_kwags_dict = {
        3: denoiser_kwags,
        4: transcriber_kwags,
        5: cropper_kwags,
    }
    processors_dict = {
        3: Denoiser,
        4: Transcriber,
        5: Cropper,
    }

    processor = processors_dict[stage](**processors_kwags_dict[stage])


def process(sample: dict, processors: list):
    """
    Process sample.
    :param batch:       batch of samples.
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
    shard_paths = sorted(glob.glob(args.data_dir + "/*"))
    for index, shard_path in tqdm(enumerate(shard_paths), total=len(shard_paths)):
        shard = load_from_disk(shard_path)
        shard = shard.map(
            lambda sample: process(sample, args.processors),
            batched=True, batch_size=args.batch_size,
            num_proc=args.num_proc if args.num_proc <= os.cpu_count() else os.cpu_count(),
        )
        shard.save_to_disk(args.save_dir + f"/shard_{index}")

        # for sample in dataset:
        #     for sent in sample["transcript"]:
        #         print(sent["text"])
        #         print(sent["start"])
        #         print(sent["end"])
        #         print("")


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
