import os
import cv2
import numpy as np
import mediapipe as mp
from vlr.data.processors.processor import Processor


class Cropper(Processor):
    """
    This class is used to crop mouth region.
    """
    def __init__(
        self, visual_dir: str,
        mouth_dir: str,
        padding: int = 10,
        overwrite: bool = False,
    ) -> None:
        """
        :param visual_dir:                  Path to directory with video files.
        :param mouth_dir:                   Path to directory with mouth region.
        :param padding:                     Padding.
        :param overwrite:                   Overwrite existing files.
        """
        self.visual_dir = visual_dir
        self.mouth_dir = mouth_dir
        self.landmark_detector = mp.solutions.face_mesh.FaceMesh()
        self.padding = padding
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.overwrite = overwrite

        self.mouth_landmark_idxes = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        ]

    def process_sample(self, sample: dict) -> dict:
        """
        Crop mouth region in video.
        :param batch:           Sample.
        :return:                Sample with path to video of cropped mouth region.
        """
        visual_path = os.path.join(
            self.visual_dir, sample["channel"], sample["id"] + ".mp4"
        )
        mouth_path = os.path.join(
            self.mouth_dir, sample["channel"], sample["id"] + ".mp4"
        )

        if self.overwrite or not os.path.exists(mouth_path):
            cap = cv2.VideoCapture(visual_path)

            mouths = []
            max_width, max_height = 0, 0
            for frame in self.get_frames(cap):
                mouth = self.crop_mouth(frame, self.padding)
                if mouth is None or mouth.shape[0] == 0 or mouth.shape[1] == 0:
                    continue
                max_width = max(max_width, mouth.shape[1])
                max_height = max(max_height, mouth.shape[0])
                mouths.append(mouth)

            if self.check_output(
                num_cropped=len(mouths),
                sample_fps=sample["fps"],
                sample_duration=sample["duration"],
            ):
                self.write_video(
                    video_path=mouth_path,
                    frames=mouths,
                    frame_width=max_width,
                    frame_height=max_height,
                    fps=sample["fps"],
                )
            else:
                sample["id"] = None
            cap.release()

        return sample

    def check_output(
        self, num_cropped: int,
        sample_fps: int,
        sample_duration: int
    ) -> int:
        """
        Check output.
        :param num_cropped:         Number of cropped frames.
        :param sample_fps:          Sample FPS.
        :param sample_duration:     Sample duration.
        :return:                    Whether output is valid.
        """
        if abs(num_cropped / sample_fps - sample_duration) > 0.1:
            return False
        return True

    def get_frames(self, cap: cv2.VideoCapture) -> np.ndarray:
        """
        Get frames from sample.
        :param cap:     Video capture.
        """
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    def crop_mouth(self, frame: np.ndarray, padding: int) -> np.ndarray:
        """
        Crop mouth region in frame.
        :param frame:   Frame.
        :param padding: Padding.
        :return:        Mouth region.
        """
        face_landmarks = self.landmark_detector.process(frame).multi_face_landmarks

        if face_landmarks:
            mouth_landmarks = np.array(face_landmarks[0].landmark)[self.mouth_landmark_idxes]
            max_x, max_y = 0, 0
            min_x, min_y = frame.shape[1], frame.shape[0]
            for landmark in mouth_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                max_x, max_y = max(max_x, x), max(max_y, y)
                min_x, min_y = min(min_x, x), min(min_y, y)
            max_x += padding
            max_y += padding
            min_x -= padding
            min_y -= padding
            return frame[min_y:max_y, min_x:max_x]
        return None

    def write_video(
        self, video_path: str,
        frames: list,
        frame_width: int,
        frame_height: int,
        fps: int,
    ) -> None:
        """
        Write video.
        :param video_path:      Path to video.
        :param frames:          Frames.
        :param frame_width:     Frame width.
        :param frame_height:    Frame height.
        :param fps:             FPS.
        """
        video_writer = cv2.VideoWriter(
            video_path, self.fourcc, fps, (frame_width, frame_height)
        )
        for mouth in frames:
            mouth = cv2.resize(mouth, (frame_width, frame_height))
            video_writer.write(mouth)
        video_writer.release()
