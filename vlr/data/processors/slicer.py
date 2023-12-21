import os
import moviepy.editor as mp
from .processor import Processor


class Slicer(Processor):
    def __init__(
        self, visual_dir: str,
        audio_dir: str,
        fps: int = 25,
        duration_threshold: float = 1.0,
        segment_duration: float = 5.0,
        segment_overlap: float = 1.0,
        keep_last_segment: bool = True,
        overwrite: bool = False,
    ) -> None:
        """
        :param visual_dir:          Path to directory with muted video files.
        :param audio_dir:           Path to directory with sound files.
        :param fps:                 Frame rate.
        :param duration_threshold:  Minimum duration of video segment.
        :param segment_duration:    Duration of video segment.
        :param segment_overlap:     Overlap between video segments.
        :param keep_last_segment:   Keep last video segment.
        :param overwrite:           Overwrite existing files.
        """
        self.visual_dir = visual_dir
        self.audio_dir = audio_dir
        self.fps = fps
        self.duration_threshold = duration_threshold
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.keep_last_segment = keep_last_segment
        self.overwrite = overwrite

    def process_batch(self, batch: dict) -> dict:
        """
        Split video into audio and visual.
        :param batch:           Batch with path to video file.
        :return:                Samples with paths to audio and visual.
        """
        ids = []
        durations = []
        channel_names = []
        for raw_video_path, channel_name in zip(batch["file"], batch["channel"]):
            file_id = os.path.basename(raw_video_path).split('.')[0]

            try:
                video = mp.VideoFileClip(raw_video_path)
                duration = video.duration

                if duration < self.duration_threshold:
                    video.close()
                    raise Exception("Video is too short")

                video = video.set_fps(self.fps)

                segment_id = f"{file_id}" + "-{start}-{end}"
                visual_path = os.path.join(
                    self.visual_dir, channel_name, segment_id + ".mp4"
                )
                audio_path = os.path.join(
                    self.audio_dir, channel_name, segment_id + ".wav"
                )

                # Split video into segments.
                start = 0
                end = self.segment_duration
                while end <= duration:
                    segment_visual_path = visual_path.format(start=int(start), end=int(end))
                    segment_audio_path = audio_path.format(start=int(start), end=int(end))

                    self.separate(
                        segment=video.subclip(start, end),
                        visual_path=segment_visual_path,
                        audio_path=segment_audio_path,
                    )

                    ids.append(segment_id.format(start=int(start), end=int(end)))
                    channel_names.append(channel_name)
                    durations.append(int(end - start))

                    start += self.segment_duration - self.segment_overlap
                    end = start + self.segment_duration

                if self.keep_last_segment and int(duration - end) >= self.segment_overlap:
                    end = duration
                    start = end - self.segment_duration
                    segment_visual_path = visual_path.format(start=int(start), end=int(end))
                    segment_audio_path = audio_path.format(start=int(start), end=int(end))

                    self.separate(
                        segment=video.subclip(start, end),
                        visual_path=segment_visual_path,
                        audio_path=segment_audio_path,
                    )

                    ids.append(segment_id.format(start=int(start), end=int(end)))
                    channel_names.append(channel_name)
                    durations.append(int(end - start))

                video.close()
            except Exception:
                pass

        batch["id"] = ids
        batch["channel"] = channel_names
        batch["fps"] = [self.fps] * len(ids)
        batch["duration"] = durations
        return batch

    def separate(
        self, segment: mp.VideoFileClip,
        visual_path: str,
        audio_path: str,
    ) -> None:
        """
        Separate video into audio and visual.
        :param segment:         Video segment.
        :param visual_path:     Path to visual file.
        :param audio_path:      Path to audio file.
        """
        if self.overwrite or not os.path.exists(visual_path):
            segment.without_audio().write_videofile(
                visual_path,
                codec="libx264",
                logger=None,
            )
        if self.overwrite or not os.path.exists(audio_path):
            segment.audio.write_audiofile(
                audio_path,
                codec="pcm_s16le",
                logger=None,
            )
