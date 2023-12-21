import os
import moviepy.editor as mp
from .processor import Processor


class Slicer(Processor):
    def process_batch(
        self, batch: dict,
        visual_output_dir: str,
        audio_output_dir: str,
        fps: int = 25,
        clip_duration: float = 3.0,
        clip_overlap: float = 1.0,
        keep_last: bool = True,
    ) -> dict:
        """
        Split video into audio and visual.
        :param batch:           Batch with path to video file.
        :return:                Samples with paths to audio and visual.
        """
        new_batch = {
            "id": [],
            "channel": [],
            "duration": [],
            "fps": [],
            "sampling_rate": [],
        }

        for id, video_path in zip(batch["id"], batch["video"]):
            segment_ids = []
            with mp.VideoFileClip(video_path) as video:
                duration = video.duration

                if duration < clip_duration:
                    continue

                video = video.set_fps(fps)
                sampling_rate = int(video.audio.fps)

                # Split video into segments.
                start = 0
                end = clip_duration
                while end <= duration:
                    segment_id = self.save_visual_and_audio_clip(
                        id=id,
                        video=video,
                        start=start,
                        end=end,
                        visual_dir=visual_output_dir,
                        audio_dir=audio_output_dir,
                    )
                    segment_ids.append(segment_id)

                    start += clip_duration - clip_overlap
                    end = start + clip_duration

                if keep_last and int(duration - start) >= 0.5 * clip_duration:
                    segment_id = self.save_visual_and_audio_clip(
                        id=id,
                        video=video,
                        start=duration - clip_duration,
                        end=duration,
                        visual_dir=visual_output_dir,
                        audio_dir=audio_output_dir,
                    )
                    segment_ids.append(segment_id)

                new_batch["id"].extend(segment_ids)
                new_batch["channel"].extend([batch["channel"]] * len(segment_ids))
                new_batch["duration"].extend([clip_duration] * len(segment_ids))
                new_batch["fps"].extend([fps] * len(segment_ids))
                new_batch["sampling_rate"].extend([sampling_rate] * len(segment_ids))
        return new_batch

    def save_visual_and_audio_clip(
        self, id: str,
        video: mp.VideoFileClip,
        start: float,
        end: float,
        visual_dir: str,
        audio_dir: str,
    ) -> str:
        """
        Separate video into audio and visual.
        :param segment:         Video segment.
        :param visual_path:     Path to visual file.
        :param audio_path:      Path to audio file.
        """
        segment_id = f"{id}-{int(start)}-{int(end)}"
        segment = video.subclip(start, end)
        self.save_visual_clip(
            segment=segment,
            segment_id=segment_id,
            visual_dir=visual_dir,
        )
        self.save_audio_clip(
            segment=segment,
            segment_id=segment_id,
            audio_dir=audio_dir,
        )
        return segment_id

    def save_visual_clip(
        self, segment: mp.VideoFileClip,
        segment_id: str,
        visual_dir: str,
    ) -> None:
        """
        Separate video into audio and visual.
        :param segment:         Video segment.
        :param visual_path:     Path to visual file.
        :param audio_path:      Path to audio file.
        """
        visual_path = os.path.join(visual_dir, f"{segment_id}.mp4")
        if not os.path.exists(visual_path):
            segment.without_audio().write_videofile(
                visual_path,
                codec="libx264",
                logger=None,
            )

    def save_audio_clip(
        self, segment: mp.VideoFileClip,
        segment_id: str,
        audio_dir: str,
    ) -> None:
        """
        Separate video into audio and visual.
        :param segment:         Video segment.
        :param visual_path:     Path to visual file.
        :param audio_path:      Path to audio file.
        """
        audio_path = os.path.join(audio_dir, f"{segment_id}.wav")
        if not os.path.exists(audio_path):
            segment.audio.write_audiofile(
                audio_path,
                codec="pcm_s16le",
                logger=None,
            )
