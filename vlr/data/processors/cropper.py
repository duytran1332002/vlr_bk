import os
import cv2
import mediapipe as mp
from vlr.data.processors.base import Processor


class Cropper(Processor):
    """
    This class is used to crop mouth region.
    """
    def __init__(
        self, mouth_dir: str,
        min_detection_confidence: float = 0.9,
        overwrite: bool = False,
    ):
        """
        :param mouth_dir:                   Path to directory with mouth region.
        :param min_detection_confidence:    Minimum confidence value for face detection.
        :param overwrite:                   Overwrite existing files.
        """
        self.mouth_dir = mouth_dir
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.overwrite = overwrite

    def process_sample(self, sample: dict, channel_name: str):
        """
        Crop mouth region.
        :param batch:           Sample.
        :param channel_name:    Channel name.
        :return:                Sample with path to video of cropped mouth region.
        """
        id = sample["id"]
        visual_path = sample["visual"]["path"]
        fps = sample["visual"]["fps"]
        mouth_path = os.path.join(self.mouth_dir, channel_name, f"{id}-mouth.mp4")

        try:
            if self.overwrite or not os.path.exists(mouth_path):
                cap = cv2.VideoCapture(visual_path)
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))

                mouths = []
                for frame in self.get_frames(cap):
                    mouths.append(self.crop_mouth(frame))

                video_writer = cv2.VideoWriter(
                    mouth_path, self.fourcc, fps, (frame_width, frame_height)
                )
                for mouth in mouths:
                    mouth = cv2.resize(mouth, (frame_width, frame_height))
                    video_writer.write(mouth)
                video_writer.release()
                cap.release()
            sample["visual"]["path"] = mouth_path
        except Exception:
            sample["visual"]["path"] = None

        return sample

    def get_frames(self, cap: cv2.VideoCapture):
        """
        Get frames from sample.
        :param cap:     Video capture.
        """
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    def crop_mouth(self, frame: np.ndarray):
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
