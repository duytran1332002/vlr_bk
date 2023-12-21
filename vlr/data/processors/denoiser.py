import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from denoiser import pretrained
from denoiser.dsp import convert_audio
from vlr.data.processors.processor import Processor


class Denoiser(Processor):
    """
    This class is used to denoise audio array.
    """
    def __init__(
        self, audio_dir: str,
        denoised_dir: str,
        sampling_rate: int = 16000,
        overwrite: bool = False,
    ) -> None:
        """
        :param audio_dir:       Path to directory containing sound files.
        :param denoised_dir:    Path to directory containing denoised sound files.
        :param sampling_rate:   Sampling rate.
        :param overwrite:       Overwrite existing files.
        """
        self.model = pretrained.dns64().cuda()
        self.audio_dir = audio_dir
        self.denoised_dir = denoised_dir
        self.sampling_rate = sampling_rate
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.model.sample_rate,
            new_freq=self.sampling_rate,
        )
        self.overwrite = overwrite

    def process_sample(self, sample: dict) -> dict:
        """
        Denoise audio array.
        :param sample:          Sample.
        :return:                Sample updated with path to denoised audio array.
        """
        audio_path = os.path.join(
            self.audio_dir, sample["channel_name"], sample["id"] + ".wav"
        )
        denoised_path = os.path.join(
            self.denoised_dir, sample["channel_name"], sample["id"] + ".wav"
        )

        if self.overwrite or not os.path.exists(denoised_path):
            audio_array, sampling_rate = torchaudio.load(audio_path)
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
                self.sampling_rate,
            )

        sample["sampling_rate"] = self.sampling_rate
        return sample

    def process_batch(self, batch: dict) -> dict:
        """
        Denoise audio array.
        :param batch:           Batch of samples.
        :return:                Samples updated with path to denoised audio array.
        """
        audio_arrays = []
        indexes = [0]
        for id, channel_name in zip(batch["id"], batch["channel"]):
            audio_path = os.path.join(
                self.audio_dir, channel_name, id + ".wav"
            )
            audio_array, sampling_rate = torchaudio.load(audio_path)
            audio_array = convert_audio(
                audio_array.cuda(),
                sampling_rate,
                self.model.sample_rate,
                self.model.chin
            )
            audio_arrays.append(audio_array)
            indexes.append(indexes[-1] + audio_array.shape[1])
        audio_arrays = torch.cat(audio_arrays, dim=1).cuda()

        with torch.no_grad():
            output = self.model(audio_arrays[None].float())
        denoised_audio_arrays = self.resampler(output[0].cpu()).numpy()

        for i, (id, channel_name) in enumerate(zip(batch["id"], batch["channel"])):
            denoised_path = os.path.join(
                self.denoised_dir, channel_name, id + ".wav"
            )

            if self.overwrite or not os.path.exists(denoised_path):
                denoised_audio_array = denoised_audio_arrays[:, indexes[i]:indexes[i+1]]
                sf.write(
                    denoised_path,
                    np.ravel(denoised_audio_array),
                    self.sampling_rate,
                )

        batch["sampling_rate"] = [self.sampling_rate] * len(batch["id"])
        return batch
