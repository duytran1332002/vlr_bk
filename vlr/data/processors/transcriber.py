import os
import torch
import torchaudio
from CocCocTokenizer import PyTokenizer
from huggingface_hub import hf_hub_download
from importlib.machinery import SourceFileLoader
from transformers import Wav2Vec2ProcessorWithLM
from .processor import Processor
from .language_classifier import LanguageClassifier


class Transcriber(Processor):
    """
    This class is used to transcribe audio into text.
    """
    def __init__(self) -> None:
        # Load the model and the processor.
        repo_id = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Prepare tokenizer.
        self.tokenizer = PyTokenizer(load_nontone_data=True)

        # Prepare language classifier.
        self.language_classifier = LanguageClassifier()

    def process(
        self, sample: dict,
        transcript_output_dir: str,
        language_threshold: float = 0.99,
        beam_width: int = 500,
    ) -> dict:
        """
        Transcribe for a sample.
        :param sample:                  Audio sample.
        :param transcript_output_dir:   Path to directory containing transcript.
        :param language_threshold:      Threshold to determine whether audio is Vietnamese.
        :param beam_width:              Beam width.
        :return:                        Sample with path to transcript.
        """
        transcript_output_path = os.path.join(transcript_output_dir, sample["id"][0] + ".txt")
        if not os.path.exists(transcript_output_path):
            try:
                audio_array, sampling_rate = torchaudio.load(sample["audio"][0])
                if self.language_classifier.is_vietnamese(
                    audio_array=audio_array,
                    sampling_rate=sampling_rate,
                    threshold=language_threshold,
                ):
                    raise Exception("Language is not Vietnamese.")

                transcript = self.transcribe(
                    audio_array=audio_array,
                    sampling_rate=sampling_rate,
                    beam_width=beam_width,
                )

                if not self.check_output(transcript=transcript):
                    raise Exception("Transcript is invalid.")

                with open(transcript_output_path, "w", encoding="utf-8") as f:
                    print(transcript.strip(), file=f)
            except Exception:
                sample["id"][0] = None
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
        :return:                transcript.
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
