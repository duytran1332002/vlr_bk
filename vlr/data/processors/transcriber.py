import os
import zipfile
import torch
import kenlm
import logging
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from vlr.data.processors.base import Processor


logger = logging.getLogger("__name__")


class Transcriber(Processor):
    """
    This class is used to transcribe audio into text.
    """
    def __init__(
        self,
        model_path: str,
        lm_gram_name: str = "vi_lm_4grams.bin.zip",
        device: str = "cuda",
    ):
        # Load the model and the processor.
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)

        # Prepare device.
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("No cuda is detected. Use CPU instead.")
            device = "cpu"
        self.device = device
        self.model = self.model.to(device)

        # Prepare language model decoder.
        lm_zip_path = os.path.join(model_path, lm_gram_name)
        lm_path = lm_zip_path[:-4]
        if not os.path.exists(lm_path):
            logger.info("Extracting language model...")
            with zipfile.ZipFile(lm_zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_path)
        self.lm = self.get_lm_decoder(lm_path)

    def get_lm_decoder(self, lm_path: str):
        """
        Get language model decoder
        :param lm_path:     language model path.
        :return:            language model decoder.
        """
        vocab_dict = self.processor.tokenizer.get_vocab()
        sort_vocab = sorted((value, key)
                            for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:-2]
        vocab_list = vocab
        # convert ctc blank character representation
        vocab_list[self.processor.tokenizer.pad_token_id] = ""
        # replace special characters
        vocab_list[self.processor.tokenizer.unk_token_id] = ""
        # vocab_list[tokenizer.bos_token_id] = ""
        # vocab_list[tokenizer.eos_token_id] = ""
        # convert space character representation
        vocab_list[self.processor.tokenizer.word_delimiter_token_id] = " "
        # specify ctc blank char index,
        # since conventially it is the last entry of the logit matrix
        alphabet = Alphabet.build_alphabet(
            vocab_list, ctc_token_idx=self.processor.tokenizer.pad_token_id)
        lm_model = kenlm.Model(lm_path)
        decoder = BeamSearchDecoderCTC(alphabet,
                                       language_model=LanguageModel(lm_model))
        return decoder

    def process(self, batch: dict):
        """
        Transcribe and include time offset for each word or sentence.
        :param sample:  audio sample.
        :return:        processed sample.
        """
        transcripts = []
        for audio_array, sampling_rate in zip(batch["audio"], batch["sampling_rate"]):
            audio_array = np.array(audio_array).reshape(-1, 1)
            # Normalize waveform in order to make it comparable with the pretrained model.
            if len(audio_array.shape) == 1:
                audio_array = audio_array[:, np.newaxis]
            audio_array = np.mean(audio_array, axis=1)

            transcripts.append(self.transcribe(
                audio_array=audio_array,
                sampling_rate=sampling_rate,
            ))
        batch["transcript"] = transcripts
        return batch

    def transcribe(
        self, audio_array: np.ndarray,
        sampling_rate: int = 16000,
        beam_width: int = 500,
    ):
        """
        Transcribe audio with time offset from audio array.
        :param audio_array:     audio array.
        :param sampling_rate:   sampling rate.
        :param beam_width:      beam width.
        :param align:           align transcript with audio or not.
        :return:                transcript.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

        input_values = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values.to(self.device)

        logits = self.model(input_values).logits[0]
        return self.lm.decode(
            logits.cpu().detach().numpy(), beam_width=beam_width
        )


if __name__ == "__main__":
    stt = Transcriber()
    stt.process(r"I:\My Drive\north.wav")
