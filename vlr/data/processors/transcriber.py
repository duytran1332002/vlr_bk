import os
import zipfile
import kenlm
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from vlr.data.processors.base import Processor


class Transcriber(Processor):
    """
    This class is used to transcribe audio into text.
    """
    def __init__(
        self,
        repo_id: str,
        device: str = "cuda",
        transcript_dir: str = None,
        overwrite: bool = False,
    ):
        """
        :param model_path:          Path to model.
        :param lm_gram_name:        Language model name.
        :param device:              Device to use.
        :param transcript_dir:      Path to directory containing transcripts.
        :param overwrite:           Overwrite existing files.
        """
        # Load the model and the processor.
        self.processor = Wav2Vec2Processor.from_pretrained(repo_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(repo_id)

        # Prepare device.
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            device = "cpu"
        self.device = device
        self.model = self.model.to(device)

        # Prepare language model decoder.
        lm_zip_path = hf_hub_download(
            repo_id=repo_id,
            filename="vi_lm_4grams.bin.zip",
        )
        lm_path = lm_zip_path[:-4]
        if not os.path.exists(lm_path):
            with zipfile.ZipFile(lm_zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(lm_zip_path))
        self.lm = self.get_lm_decoder(lm_path)

        self.transcript_dir = transcript_dir
        self.overwrite = overwrite

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

    def process_sample(self, sample: dict, channel_name: str):
        """
        Transcribe for a sample.
        :param sample:          Audio sample.
        :param channel_name:    Channel name.
        :return:                Sample with path to transcript.
        """
        id = sample["id"]
        transcript_path = os.path.join(self.transcript_dir, channel_name, f"{id}.txt")

        if self.overwrite or not os.path.exists(transcript_path):
            audio_array, sampling_rate = torchaudio.load(sample["audio"]["path"])

            transcript = self.transcribe(
                audio_array=audio_array.numpy(),
                sampling_rate=sampling_rate,
            )

            if len(transcript) > 0:
                with open(transcript_path, "w") as f:
                    print(transcript, file=f)
            else:
                transcript_path = None

        sample["audio"]["path"] = os.path.basename(sample["audio"]["path"])
        sample["transcript"] = transcript_path
        return sample

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
        :return:                transcript(s).
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

        input_values = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        ).input_values.to(self.device)
        logits = self.model(input_values).logits

        return self.lm.decode(
            logits[0].cpu().detach().numpy(),
            beam_width=beam_width
        )
