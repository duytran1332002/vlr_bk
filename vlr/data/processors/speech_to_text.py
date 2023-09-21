import os
import zipfile
import torch
import kenlm
import logging
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from dataclasses import dataclass
from vlr.data.processors.base import Processor


logger = logging.getLogger("__name__")


@dataclass
class Point:
    """
    Point in the trellis.
    """
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    """
    Segment of a word.
    """
    label: str
    start: int
    end: int
    score: float

    @property
    def length(self):
        """
        Get length of the segment.
        :return:    length of the segment.
        """
        return self.end - self.start


class SpeechToText(Processor):
    """
    Transcribe audio into text with time offset.
    """
    def __init__(self,
                 model_path: str,
                 lm_gram_name: str = "vi_lm_4grams.bin.zip",
                 device: str = "cuda",
                 mode: str = "word",
                 segment_duration: float = 10.0,
                 segment_overlap: float = 1.0,
                 keep_last_segment: bool = True):
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

        self.mode = mode
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.keep_last_segment = keep_last_segment

    def process(self, sample: dict):
        """
        Transcribe and include time offset for each word or sentence.
        :param sample:  audio sample.
        :return:        processed sample.
        """
        fn_dict = {
            "word": self.process_word,
            "sentence": self.process_sentence,
        }

        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]

        # Normalize waveform in order to make it comparable with the pretrained model.
        if len(audio_array.shape) == 1:
            audio_array = audio_array[:, np.newaxis]
        audio_array = np.mean(audio_array, axis=1)

        # Segment audio array.
        audio_segments = {}
        start = 0
        num_points_per_segment = int(self.segment_duration * sampling_rate)
        overlap = int(self.segment_overlap * sampling_rate)
        while len(audio_array) > num_points_per_segment:
            end = start + self.segment_duration
            audio_segments[(start, end)] = audio_array[:num_points_per_segment]
            audio_array = audio_array[num_points_per_segment - overlap:]
            start += self.segment_duration - self.segment_overlap
        if self.keep_last_segment:
            duration = round(len(audio_array) / sampling_rate, 1)
            audio_segments[(start, start + duration)] = audio_array

        sample["transcript"] = fn_dict[self.mode](audio_segments, sampling_rate)
        return sample

    def process_sentence(self, audio_segments: dict, sampling_rate: int = 16000):
        """
        Transcribe and include time offset for each sentence.
        :param audio_segments:  audio segments.
        :param sampling_rate:   sampling rate.
        :return:                transcripts of sentences.
        """
        # Get aligned transcript.
        transcript = []
        for (start, end), audio_segment in audio_segments.items():
            text = self.transcribe(
                audio_array=audio_segment, sampling_rate=sampling_rate, align=False
            )
            transcript.append({
                "text": text,
                "start": start,
                "end": end,
            })
        return transcript

    def process_word(self, audio_segments: dict, sampling_rate: int = 16000):
        """
        Transcribe and include time offset for each word.
        :param audio_segments:  audio segments.
        :param sampling_rate:   sampling rate.
        :return:        transcripts of words.
        """
        # Get aligned transcript.
        transcript = []
        for audio_segment in audio_segments.values():
            transcript.extend(
                self.transcribe(audio_array=audio_segment, sampling_rate=sampling_rate)
            )
        return transcript

    def transcribe(self, audio_array: np.ndarray, sampling_rate: int = 16000,
                   beam_width: int = 500, align: bool = True):
        """
        Transcribe audio with time offset from audio array.
        :param audio_array:     audio array.
        :param sampling_rate:   sampling rate.
        :param beam_width:      beam width.
        :return:                aligned transcript.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

        input_values = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values.to(self.device)

        logits = self.model(input_values).logits[0]
        transcript = self.lm.decode(
            logits.cpu().detach().numpy(), beam_width=beam_width)

        return self.align(audio_array, logits, transcript) if align else transcript

    def align(self, audio_array, logits, transcript: str):
        """
        Align transcript with audio.
        :param audio_array:     audio array.
        :param logits:          logits from the model.
        :param transcript:      transcript.
        :return:                aligned transcript.
        """
        vocabs = self.processor.tokenizer.get_vocab()
        transcript = transcript.upper().replace(" ", "|")
        dictionary = {c.upper(): i for i, c in enumerate(vocabs)}
        tokens = [dictionary[c] for c in transcript]

        trellis = self.get_trellis(logits, tokens)
        path = self.backtrack(trellis, logits, tokens)
        segments = self.merge_repeats(path, transcript)
        word_segments = self.merge_words(segments)

        ratio = audio_array.shape[1] / (trellis.shape[0] - 1)
        return [
            {
                "text": word_segment.label,
                "start": int(ratio * word_segment.start),
                "end": int(ratio * word_segment.end),
            }
            for word_segment in word_segments
        ]

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

    def get_trellis(self, logits, tokens, blank_id=0):
        """
        Get trellis for Viterbi decoding.
        :param logits:      logits from the model.
        :param tokens:      tokens.
        :param blank_id:    blank id.
        :return:            trellis.
        """
        num_frame = logits.shape[0]
        num_token = len(tokens)

        trellis = torch.empty((num_frame + 1, num_token + 1))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(logits[:, 0], 0)
        trellis[0, -num_token:] = -float("inf")
        trellis[-num_token:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + logits[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + logits[t, tokens],
            )
        return trellis

    def backtrack(self, trellis, logits, tokens, blank_id=0):
        """
        Get path from trellis.
        :param trellis:     trellis.
        :param logits:      logits from the model.
        :param tokens:      tokens.
        :param blank_id:    blank id.
        :return:            path.
        """
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + logits[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + logits[t - 1, tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            prob = logits[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]

    def merge_repeats(self, path, transcript):
        """
        Merge repeated characters.
        :param path:        path.
        :param transcript:  transcript.
        :return:            segments.
        """
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments

    def merge_words(self, segments, separator="|"):
        """
        Merge words.
        :param segments:    segments.
        :param separator:   separator.
        :return:            words.
        """
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words


if __name__ == "__main__":
    stt = SpeechToText()
    stt.process(r"I:\My Drive\north.wav")
