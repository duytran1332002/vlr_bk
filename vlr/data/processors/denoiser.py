import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from denoiser import pretrained
from denoiser.dsp import convert_audio
from vlr.data.processors.base import Processor


class Denoiser(Processor):
    """
    This class is used to denoise audio array.
    """
    def __init__(
        self, denoised_dir: str = None,
        sampling_rate: int = 16000,
        overwrite: bool = False,
    ):
        """
        :param denoised_dir:    Path to directory containing denoised sound files.
        :param sampling_rate:   Sampling rate.
        :param overwrite:       Overwrite existing files.
        """
        self.model = pretrained.dns64().cuda()
        self.denoised_dir = denoised_dir
        self.sampling_rate = sampling_rate
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.model.sample_rate,
            new_freq=self.sampling_rate,
        )
        self.overwrite = overwrite

    def process_sample(self, sample: dict, channel_name: str):
        """
        Denoise audio array.
        :param sample:          Sample.
        :param channel_name:    Channel name.
        :return:                Sample updated with path to denoised audio array.
        """
        id = sample["id"]
        denoised_path = os.path.join(self.denoised_dir, channel_name, f"{id}-denoised.wav")

        if self.overwrite or not os.path.exists(denoised_path):
            audio_array, sampling_rate = torchaudio.load(sample["audio"])
            audio_array = convert_audio(
                audio_array.cuda(),
                sampling_rate,
                self.model.sample_rate,
                self.model.chin
            )

            with torch.no_grad():
                output = self.model(audio_array[None].float())
            denoised_audio_array = self.resampler(output[0].cpu()).numpy()

            sf.write(
                denoised_path,
                np.ravel(denoised_audio_array),
                self.model.sample_rate,
            )

        sample["audio"] = {
            "path": denoised_path,
            "sampling_rate": self.sampling_rate,
        }
        return sample
