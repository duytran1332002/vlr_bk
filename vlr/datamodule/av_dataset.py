import os

import torch
import torchaudio
import torchvision
import av
import io
import os
import numpy as np



from .transforms import TextTransform, AudioTransform, VideoTransform


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, padding), "constant")
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path, normalize=True)
    return waveform.transpose(1, 0)

def extract_frames(video_bytes):
    # Create a memory-mapped file from the bytes
    container = av.open(io.BytesIO(video_bytes))

    # Find the video stream
    visual_stream = next(iter(container.streams.video), None)
    if not audio_stream:
        return None, None

    # Extract video properties
    video_fps = visual_stream.average_rate

    # Initialize arrays to store frames
    frames_array = []

    # Extract frames
    for packet in container.demux([visual_stream]):
        for frame in packet.decode():
            img_array = np.array(frame.to_image())
            frames_array.append(img_array)

    return np.array(frames_array), video_fps


def extract_audio_array(audio_bytes):
    # Create a memory-mapped file from the bytes
    container = av.open(io.BytesIO(audio_bytes))

    # Find the audio stream
    audio_stream = next(iter(container.streams.audio), None)
    if not audio_stream:
        return None, None

    # Extract audio properties
    audio_fps = audio_stream.rate

    # Initialize arrays to store audio
    audio_array = []

    # Iterate over packets and extract audio
    for packet in container.demux([audio_stream]):
        for frame in packet.decode():
            audio_array.extend(frame.to_ndarray())

    return np.array(audio_array), audio_fps
class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        root_dir,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
        from_="local",
    ):
        self.root_dir = root_dir
        self.modality = modality
        self.rate_ratio = rate_ratio

        self.dataset = dataset
        self.text_transform = TextTransform()

        self.audio_transform = audio_transform
        self.video_transform = video_transform
        if from_ not in ["local", "hf"]:
            raise ValueError("from_ must be either local or hf")
        self.from_ = from_

    def get_from_local(self, idx):
        item = self.dataset[int(idx)]
        channel = item["channel"]
            
        # load text and transform it into token ids
        transcript_path =  os.path.join(self.root_dir, "transcripts" + "/" + channel + "/" + item["id"] + ".txt")
        text = open(transcript_path , encoding="utf8").read().strip()
        token_id = self.text_transform.tokenize(text)

        video_path = os.path.join(self.root_dir, "mouths" + "/" + channel + "/" + item["id"] + ".mp4")
        audio_path = os.path.join(self.root_dir, "denoised" + "/" + channel + "/" + item["id"] + ".wav")

        if self.modality == "video":
            video = load_video(video_path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            audio = load_audio(audio_path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}
        elif self.modality == "audiovisual":
            video = load_video(video_path)
            audio = load_audio(audio_path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            return {"video": video, "audio": audio, "target": token_id}
    def get_from_hf(self, idx):
        pass


    def __getitem__(self, idx):
        # item format:
        # {'id': '', 'visual': {'fps': 25, 'path': 'video_path'}, \
        # 'audio': {'path': 'audio_path', 'sampling_rate': 16000}, \
        # 'duration': 3, 'transcript': 'transcript_path'}
        return self.get_from_local(idx) if self.from_ == "local" else self.get_from_hf(idx)
        

    def __len__(self):
        return len(self.dataset)

class AVDatasetIterable(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        root_dir,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
        from_="local",
    ):
        self.root_dir = root_dir
        self.modality = modality
        self.rate_ratio = rate_ratio

        self.dataset = dataset
        self.text_transform = TextTransform()

        self.audio_transform = audio_transform
        self.video_transform = video_transform

        if from_ not in ["local", "hf"]:
            raise ValueError("from_ must be either local or hf")
        self.from_ = from_

    def get_from_local(self, idx):
        item = self.dataset[int(idx)]
        channel = item["channel"]
            
        # load text and transform it into token ids
        transcript_path =  os.path.join(self.root_dir, "transcripts" + "/" + channel + "/" + item["id"] + ".txt")
        text = open(transcript_path , encoding="utf8").read().strip()
        token_id = self.text_transform.tokenize(text)

        video_path = os.path.join(self.root_dir, "mouths" + "/" + channel + "/" + item["id"] + ".mp4")
        audio_path = os.path.join(self.root_dir, "denoised" + "/" + channel + "/" + item["id"] + ".wav")

        if self.modality == "video":
            video = load_video(video_path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            audio = load_audio(audio_path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}
        elif self.modality == "audiovisual":
            video = load_video(video_path)
            audio = load_audio(audio_path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            return {"video": video, "audio": audio, "target": token_id}
    def get_from_hf(self, idx):
        pass

    def __iter__(self):
        # item format:
        # {'channel': 'mcnguyenkhang_output', 'sampling_rate': 16000, 'id': '720183465804312089700020-0-3', 'duration': 3, 'fps': 25}
        for idx in range(len(self.dataset)):
            if self.from_ == "local":
                yield self.get_from_local(idx)
            elif self.from_ == "hf":
                yield self.get_from_hf(idx)
    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    from datasets import load_dataset
    dataset = AVDataset(
        root_dir= "/home/duytran/Downloads/vlr/",
        dataset=load_dataset("/home/duytran/Downloads/vlr/cropping")["train"],
        modality="audio",
        audio_transform=AudioTransform("train"),
        video_transform=VideoTransform("train"),
    )
    for i in range(10):
        print(dataset[i]["input"].shape)
        print(dataset[i]["target"])


   



