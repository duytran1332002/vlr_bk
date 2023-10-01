import torch
import logging
from denoiser import pretrained
from denoiser.dsp import convert_audio
from vlr.data.processors.base import Processor


logger = logging.getLogger(__name__)


class Denoiser(Processor):
    """
    This class is used to denoise audio array.
    """
    def __init__(self):
        self.model = pretrained.dns64().cuda()

    def process(self, batch: dict):
        """
        Denoise audio array.
        :param batch:   Batch with audio array.
        :return:        Samples with denoised audio array.
        """
        for i, audio_array in enumerate(batch["audio"]):
            sampling_rate = batch["sampling_rate"][i]
            audio_array = convert_audio(
                torch.tensor([audio_array]).cuda(),
                sampling_rate,
                self.model.sample_rate,
                self.model.chin
            )
            with torch.no_grad():
                denoised_audio_array = self.model(audio_array[None])[0]
            batch["audio"][i] = denoised_audio_array.data.cpu().numpy()
        return batch
