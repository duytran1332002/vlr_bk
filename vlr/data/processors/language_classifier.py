import os
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from vlr.data.processors.base import Processor


class LanguageClassifier(Processor):
    """
    This class is used to filter out samples with Vietnamese language.
    """
    def __init__(
        self, audio_dir: str,
    ) -> None:
        """
        :param audio_dir:       Path to directory containing sound files.
        """
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="tmp"
        )
        self.audio_dir = audio_dir

    def process_sample(self, sample: dict) -> dict:
        """
        Filter out samples with Vietnamese language.
        :param sample:          Sample.
        :return:                Sample updated with path to denoised audio array.
        """
        audio_path = os.path.join(
            self.audio_dir, sample["channel"], sample["id"] + ".wav"
        )

        audio_array, sampling_rate = torchaudio.load(audio_path)
        if sampling_rate != 16000:
            audio_array = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=16000,
            )
            sample["sampling_rate"] = 16000

        _, score, lang_idx, _ = self.model.classify_batch(audio_array.cuda())
        score = score.exp().item()
        lang_idx = lang_idx.item()
        if lang_idx != 102 or score < 0.99:
            sample["id"] = None
        return sample

    def process_batch(self, batch: dict) -> dict:
        """
        Filter out samples with Vietnamese language.
        :param batch:           Batch of samples.
        :return:                Samples updated with path to denoised audio array.
        """
        audio_arrays = []
        max_length = 0
        for i in range(len(batch["id"])):
            audio_path = os.path.join(
                self.audio_dir, batch["channel"][i], batch["id"][i] + ".wav"
            )
            audio_array, sampling_rate = torchaudio.load(audio_path)
            if sampling_rate != 16000:
                audio_array = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=16000,
                )
                batch["sampling_rate"][i] = 16000
            audio_arrays.append(audio_array)
            max_length = max(max_length, len(audio_array))

        # Pad audio arrays.
        for i in range(len(batch["id"])):
            audio_arrays[i] = torch.nn.functional.pad(
                audio_arrays[i],
                (0, max_length - len(audio_arrays[i])),
                "constant",
                0,
            )

        # Stack audio arrays.
        audio_arrays = torch.stack(audio_arrays).cuda()

        # Classify audio arrays.
        _, scores, lang_idxes, _ = self.model.classify_batch(audio_array)
        scores = scores.exp().cpu().tolist()
        lang_idxes = lang_idxes.cpu().tolist()

        for i, (score, lang_idx) in enumerate(zip(scores, lang_idxes)):
            if lang_idx != 102 or score < 0.99:
                batch["id"][i] = None
        return batch
