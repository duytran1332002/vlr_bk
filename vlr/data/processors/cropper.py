import os
import cv2
import numpy as np
import mediapipe as mp
from vlr.data.processors.base import Processor


class Cropper(Processor):
    """
    This class is used to crop mouth region.
    """
    def __init__(
        self, visual_dir: str,
        mouth_dir: str,
        min_detection_confidence: float = 0.9,
        overwrite: bool = False,
    ) -> None:
        """
        :param visual_dir:                  Path to directory with video files.
        :param mouth_dir:                   Path to directory with mouth region.
        :param min_detection_confidence:    Minimum confidence value for face detection.
        :param overwrite:                   Overwrite existing files.
        """
        self.visual_dir = visual_dir
        self.mouth_dir = mouth_dir
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.overwrite = overwrite

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

        try:
            if self.overwrite or not os.path.exists(mouth_path):
                cap = cv2.VideoCapture(visual_path)
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))

                mouths = []
                for frame in self.get_frames(cap):
                    mouths.append(self.crop_mouth(frame))

                video_writer = cv2.VideoWriter(
                    mouth_path, self.fourcc, sample["fps"], (frame_width, frame_height)
                )
                for mouth in mouths:
                    mouth = cv2.resize(mouth, (frame_width, frame_height))
                    video_writer.write(mouth)
                video_writer.release()
                cap.release()
        except Exception:
            sample["id"] = None

        return sample

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

    def crop_mouth(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop mouth region in frame.
        :param frame:   Frame.
        :return:        Mouth region.
        """
        detections = self.face_detector.process(frame).detections

        mouth = None
        if detections and len(detections) == 1:
            for detection in detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                mouth = frame[y + h//2:y + h, x:x + w]
        if mouth is None or mouth.shape[0] == 0 or mouth.shape[1] == 0:
            raise Exception("No mouth detected.")
        return mouth
