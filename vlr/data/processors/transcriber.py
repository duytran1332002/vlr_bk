import os
import torch
import torchaudio
from CocCocTokenizer import PyTokenizer
from huggingface_hub import hf_hub_download
from importlib.machinery import SourceFileLoader
from transformers import Wav2Vec2ProcessorWithLM
from vlr.data.processors.base import Processor


class Transcriber(Processor):
    """
    This class is used to transcribe audio into text.
    """
    def __init__(
        self,
        repo_id: str,
        denoised_dir: str,
        transcript_dir: str,
        device: str = "cuda",
        overwrite: bool = False,
    ) -> None:
        """
        :param model_path:          Path to model.
        :param lm_gram_name:        Language model name.
        :param denoised_dir:        Path to directory containing denoised sound files.
        :param transcript_dir:      Path to directory containing transcripts.
        :param device:              Device to use.
        :param overwrite:           Overwrite existing files.
        """
        # Load the model and the processor.
        self.model = (
            SourceFileLoader(
                "model", hf_hub_download(
                    repo_id=repo_id,
                    filename="model_handling.py"
                )
            )
            .load_module()
            .Wav2Vec2ForCTC
            .from_pretrained(repo_id)
        )
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(repo_id)

        # Prepare device.
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            device = "cpu"
        self.device = device
        self.model = self.model.to(device)

        self.denoised_dir = denoised_dir
        self.transcript_dir = transcript_dir
        self.overwrite = overwrite

        self.tokenizer = PyTokenizer(load_nontone_data=True)

    def process_sample(self, sample: dict) -> dict:
        """
        Transcribe for a sample.
        :param sample:          Audio sample.
        :return:                Sample with path to transcript.
        """
        denoised_path = os.path.join(
            self.denoised_dir, sample["channel"], sample["id"] + ".wav"
        )
        transcript_path = os.path.join(
            self.transcript_dir, sample["channel"], sample["id"] + ".txt"
        )

        if self.overwrite or not os.path.exists(transcript_path):
            audio_array, sampling_rate = torchaudio.load(denoised_path)

            transcript = self.transcribe(
                audio_array=audio_array,
                sampling_rate=sampling_rate,
            )

            if self.check_output(transcript=transcript):
                with open(transcript_path, "w", encoding="utf-8") as f:
                    print(transcript.strip(), file=f)
            else:
                sample["id"] = None
        return sample

    def check_output(self, transcript: str) -> str:
        """
        Check output.
        :param transcript:      Transcript.
        :return:                Whether output is valid.
        """
        if len(transcript) == 0:
            return False
        for token in self.tokenizer.word_tokenize(transcript, tokenize_option=0):
            if len(self.tokenizer.word_tokenize(token, tokenize_option=2)) > 1:
                return False
        return True

    def transcribe(
        self, audio_array: torch.Tensor,
        sampling_rate: int = 16000,
        beam_width: int = 500,
    ) -> str:
        """
        Transcribe audio with time offset from audio array.
        :param audio_array:     audio array.
        :param sampling_rate:   sampling rate.
        :param beam_width:      beam width.
        :return:                transcript(s).
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

        input_data = self.processor.feature_extractor(
            audio_array[0],
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        for k, v in input_data.items():
            input_data[k] = v.to(self.device)
        logits = self.model(**input_data).logits

        return self.processor.decode(
            logits[0].cpu().detach().numpy(),
            beam_width=beam_width
        ).text
