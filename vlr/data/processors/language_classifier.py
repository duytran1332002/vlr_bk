import os
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from vlr.data.processors.processor import Processor


class LanguageClassifier(Processor):
    """
    This class is used to filter out samples with Vietnamese language.
    """
    def __init__(self) -> None:
        """
        """
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="tmp"
        )
        self.sampling_rate = 16000

    def classify(self, audio_array: torch.Tensor, sampling_rate: int) -> tuple[int, float]:
        """
        Classify language of audio array.
        :param audio_array:     Audio array.
        :return:                Language index and score.
        """
        if sampling_rate != self.sampling_rate:
            audio_array = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=self.sampling_rate,
            )
        _, score, lang_idx, _ = self.model.classify_batch(audio_array.cuda())
        score = score.exp().item()
        lang_idx = lang_idx.item()
        return lang_idx, score

    def is_vietnamese(
        self, audio_array: torch.Tensor,
        sampling_rate: int,
        threshold: float = 0.99,
    ) -> bool:
        """
        Check if language is Vietnamese.
        :param lang_idx:        Language index.
        :param score:           Score.
        :param threshold:       Threshold.
        :return:                Whether language is Vietnamese.
        """
        lang_idx, score = self.classify(audio_array, sampling_rate)
        return lang_idx == 102 and score >= threshold
