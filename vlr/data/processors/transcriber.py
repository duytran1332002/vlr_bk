import os
import kenlm
import torch
import torchaudio
import zipfile
import numpy as np
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from vlr.data.processors.base import Processor


class Transcriber(Processor):
    """
    This class is used to transcribe audio into text.
    """
    def __init__(
        self,
        model_path: str,
        lm_gram_name: str = "vi_lm_4grams.bin.zip",
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
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)

        # Prepare device.
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            device = "cpu"
        self.device = device
        self.model = self.model.to(device)

        # Prepare language model decoder.
        lm_zip_path = os.path.join(model_path, lm_gram_name)
        lm_path = lm_zip_path[:-4]
        if not os.path.exists(lm_path):
            with zipfile.ZipFile(lm_zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_path)
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

    def process_batch(self, batch: dict, channel_name: str):
        """
        Transcribe a batch of samples.
        :param sample:          Batch of audio samples.
        :param channel_name:    Channel name.
        :return:                Processed batch.
        """
        audio_arrays = []
        for audio in batch["audio"]:
            audio_array, sampling_rate = torchaudio.load(audio["path"])
            audio_arrays.append(audio_array.numpy())
        audio_array = np.stack(audio_arrays, axis=0)
        sampling_rate = batch["audio"][0]["sampling_rate"]

        transcripts = self.transcribe(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
        )
        print(transcripts)

        transcript_paths = []
        for i, audio in enumerate(batch["audio"]):
            id = batch["id"][i]
            transcript_path = os.path.join(self.transcript_dir, channel_name, f"{id}.txt")

            if self.overwrite or not os.path.exists(transcript_path):
                with open(transcript_path, "w") as f:
                    print(transcripts[i], file=f)

            transcript_paths.append(transcript_path)
            batch["audio"][i]["path"] = os.path.basename(audio["path"])
        batch["transcript"] = transcript_paths
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

        if logits.shape[0] > 1:
            transcripts = []
            for i in range(logits.shape[0]):
                print(logits.shape)
                print(logits[i].unsqueeze(0).shape)
                transcript = self.lm.decode(
                    logits[i].unsqueeze(0).cpu().detach().numpy(),
                    beam_width=beam_width
                )
                transcripts.append(transcript)
            return transcripts
        return self.lm.decode(
            logits[0].cpu().detach().numpy(),
            beam_width=beam_width
        )


if __name__ == "__main__":
    stt = Transcriber()
    stt.process_batch(r"I:\My Drive\north.wav")
