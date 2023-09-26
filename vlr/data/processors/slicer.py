import numpy as np
from vlr.data.processors import Processor


class Slicer(Processor):
    """
    This class is used to slice audio array into segments.
    """
    def __init__(
        self, segment_duration: float = 10.0,
        segment_overlap: float = 1.0,
        keep_last_segment: bool = True,
    ):
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.keep_last_segment = keep_last_segment

    def process(self, batch: dict):
        """
        Slice audio array into segments.
        :param batch:   Batch with audio array.
        :return:        Sample with sliced audio array.
        """
        processed_batch = {
            "file": [],
            "audio": [],
            "sampling_rate": [],
        }
        for i, audio_array in enumerate(batch["audio"]):
            sampling_rate = batch["sampling_rate"][i]

            # Segment audio array.
            segments = []
            num_points_per_segment = int(self.segment_duration * sampling_rate)
            while len(audio_array) > num_points_per_segment:
                segments.extend(
                    self.slice(
                        audio_array, sampling_rate,
                        end=self.segment_duration,
                    )
                )
                audio_array = self.slice(
                    audio_array, sampling_rate,
                    start=self.segment_duration - self.segment_overlap,
                )
            if self.keep_last_segment:
                segments.append(audio_array)

            # Add segments to batch.
            processed_batch["file"].extend([batch["file"][i]] * len(segments))
            processed_batch["audio"].extend(segments)
            processed_batch["sampling_rate"].extend([sampling_rate] * len(segments))

        return processed_batch

    def slice(
        self, audio_array: np.ndarray,
        sampling_rate: int,
        start: float = 0,
        end: float = None,
    ):
        """
        Slice audio array.
        :param audio_array:     Audio array.
        :param sampling_rate:   Sampling rate.
        :param start:           Start point in seconds.
        :param end:             End point in seconds.
        :return:                Sliced audio array.
        """
        start_point = int(start * sampling_rate)
        end_point = int(end * sampling_rate) if end else len(audio_array)
        return audio_array[start_point:end_point]
