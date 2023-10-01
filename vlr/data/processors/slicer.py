import io
import os
import contextlib
import moviepy.editor as mp
from vlr.data.processors.base import Processor


class Slicer(Processor):
    def __init__(
        self, raw_dir: str,
        visual_dir: str,
        audio_dir: str,
        fps: int = 25,
        duration_threshold: float = 1.0,
        segment_duration: float = 5.0,
        segment_overlap: float = 1.0,
        keep_last_segment: bool = True,
        overwrite: bool = False,
    ):
        """
        :param raw_dir:             Path to directory with raw video files.
        :param visual_dir:          Path to directory with muted video files.
        :param audio_dir:           Path to directory with sound files.
        :param fps:                 Frame rate.
        :param duration_threshold:  Minimum duration of video segment.
        :param segment_duration:    Duration of video segment.
        :param segment_overlap:     Overlap between video segments.
        :param keep_last_segment:   Keep last video segment.
        :param overwrite:           Overwrite existing files.
        """
        self.raw_dir = raw_dir
        self.visual_dir = visual_dir
        self.audio_dir = audio_dir
        self.fps = fps
        self.duration_threshold = duration_threshold
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.keep_last_segment = keep_last_segment
        self.overwrite = overwrite
        self.output_buffer = io.StringIO()

    def process(self, batch: dict):
        """
        Split video into audio and visual.
        :param batch:       Batch with video file name.
        :return:            Sample with audio and visual.
        """
        processed_batch = {
            "file": [],
            "visual": [],
            "fps": [],
            "audio": [],
        }
        for file in batch["file"]:
            file_id = file.split('.')[0]
            raw_video_path = os.path.join(self.raw_dir, file)

            try:
                # Omit what is printed to stdout.
                with contextlib.redirect_stdout(self.output_buffer):
                    video = mp.VideoFileClip(raw_video_path)
                    duration = video.duration

                    if duration < self.duration_threshold:
                        video.close()
                        raise Exception

                    video = video.set_fps(self.fps)
                    # Split video into segments.
                    start = 0
                    end = self.segment_duration
                    segment_id = f"{file_id}" + "-{start}-{end}"
                    audio_path = os.path.join(self.audio_dir, segment_id + ".wav")
                    visual_path = os.path.join(self.visual_dir, segment_id + ".mp4")
                    while end <= duration:
                        segment_visual_path = visual_path.format(start=int(start), end=int(end))
                        keep_visual = os.path.exists(segment_visual_path) and not self.overwrite
                        segment_audio_path = audio_path.format(start=int(start), end=int(end))
                        keep_audio = os.path.exists(segment_audio_path) and not self.overwrite
                        self.separate(
                            segment=video.subclip(start, end),
                            visual_path=segment_visual_path,
                            keep_visual=keep_visual,
                            audio_path=segment_audio_path,
                            keep_audio=keep_audio,
                        )
                        processed_batch["file"].append(file)
                        processed_batch["visual"].append(segment_visual_path)
                        processed_batch["fps"].append(self.fps)
                        processed_batch["audio"].append(segment_audio_path)
                        start += self.segment_duration - self.segment_overlap
                        end = start + self.segment_duration
                    end = duration
                    if end - start >= self.duration_threshold and self.keep_last_segment:
                        segment_visual_path = visual_path.format(start=int(start), end=int(end))
                        keep_visual = os.path.exists(segment_visual_path) and not self.overwrite
                        segment_audio_path = audio_path.format(start=int(start), end=int(end))
                        keep_audio = os.path.exists(segment_audio_path) and not self.overwrite
                        self.separate(
                            segment=video.subclip(start, end),
                            visual_path=segment_visual_path,
                            keep_visual=keep_visual,
                            audio_path=segment_audio_path,
                            keep_audio=keep_audio,
                        )
                        processed_batch["file"].append(file)
                        processed_batch["visual"].append(segment_visual_path)
                        processed_batch["fps"].append(self.fps)
                        processed_batch["audio"].append(segment_audio_path)
                    video.close()
            except Exception as e:
                print(e)
                continue

        return processed_batch

    def separate(
        self, segment: mp.VideoFileClip,
        visual_path: str,
        keep_visual: bool,
        audio_path: str,
        keep_audio: bool,
    ):
        """
        Separate video into audio and visual.
        :param segment:     Video segment.
        :param visual_path:  Path to visual file.
        :param audio_path:  Path to audio file.
        """
        if not keep_visual:
            segment.without_audio().write_videofile(visual_path, codec="libx264")
        if not keep_audio:
            segment.audio.write_audiofile(audio_path, codec="pcm_s16le")
