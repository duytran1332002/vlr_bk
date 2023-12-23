import os
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from .processor import Processor


class Denoiser(Processor):
    """
    This class is used to denoise audio array.
    """
    def __init__(self) -> None:
        """
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pretrained.dns64().to(self.device)

    def process_sample(
        self, sample: dict,
        audio_output_dir: str,
        output_sampling_rate: int = 16000,
    ) -> dict:
        """
        Denoise audio array.
        :param sample:                  Sample.
        :param audio_output_dir:        Path to directory containing denoised audio array.
        :param output_sampling_rate:    Sampling rate of denoised audio array.
        :return:                        Sample updated with path to denoised audio array.
        """
        audio_output_path = os.path.join(audio_output_dir, sample["id"] + ".wav")

        if not os.path.exists(audio_output_path):
            audio_array, sampling_rate = torchaudio.load(sample["audio"])
            audio_array = convert_audio(
                audio_array.to(self.device),
                sampling_rate,
                self.model.sample_rate,
                self.model.chin
            )

            with torch.no_grad():
                denoised_audio_array = self.model(audio_array[None].float())[0].cpu()

            torchaudio.save(
                audio_output_path,
                denoised_audio_array,
                output_sampling_rate,
            )

        sample["sampling_rate"] = output_sampling_rate
        return sample

    def process_batch(
        self, batch: dict,
        audio_output_dir: str,
        output_sampling_rate: int = 16000,
    ) -> dict:
        """
        Denoise audio array.
        :param batch:                   Batch of samples.
        :param audio_output_dir:        Path to directory containing denoised audio array.
        :param output_sampling_rate:    Sampling rate of denoised audio array.
        :return:                        Samples updated with path to denoised audio array.
        """
        audio_arrays = []
        indexes = [0]
        for id, audio_path in zip(batch["id"], batch["audio"]):
            audio_output_path = os.path.join(audio_output_dir, id + ".wav")
            if os.path.exists(audio_output_path):
                continue
            audio_array, sampling_rate = torchaudio.load(audio_path)
            audio_array = convert_audio(
                audio_array.to(self.device),
                sampling_rate,
                self.model.sample_rate,
                self.model.chin
            )
            audio_arrays.append(audio_array)
            indexes.append(indexes[-1] + audio_array.shape[1])
        audio_arrays = torch.cat(audio_arrays, dim=1).to(self.device)

        with torch.no_grad():
            denoised_audio_arrays = self.model(audio_array[None].float())[0].cpu()

        for i, id in enumerate(batch["id"]):
            denoised_audio_array = denoised_audio_arrays[:, indexes[i]:indexes[i+1]]
            audio_output_path = os.path.join(audio_output_dir, id + ".wav")
            torchaudio.save(
                audio_output_path,
                denoised_audio_array,
                output_sampling_rate,
            )

        batch["sampling_rate"] = [output_sampling_rate] * len(batch["id"])
        return batch
