import os

import torch
import torchaudio
import torchvision


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


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        root_dir,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
    ):
        self.root_dir = root_dir
        self.modality = modality
        self.rate_ratio = rate_ratio

        self.dataset = dataset
        self.text_transform = TextTransform()

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    

    def __getitem__(self, idx):
        # item format:
        # {'id': '', 'visual': {'fps': 25, 'path': 'video_path'}, \
        # 'audio': {'path': 'audio_path', 'sampling_rate': 16000}, \
        # 'duration': 3, 'transcript': 'transcript_path'}
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
    ):
        self.root_dir = root_dir
        self.modality = modality
        self.rate_ratio = rate_ratio

        self.dataset = dataset
        self.text_transform = TextTransform()

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    

    def __iter__(self):
        # item format:
        # {'channel': 'mcnguyenkhang_output', 'sampling_rate': 16000, 'id': '720183465804312089700020-0-3', 'duration': 3, 'fps': 25}
        for idx in range(len(self.dataset)):
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
                yield {"input": video, "target": token_id}
            elif self.modality == "audio":
                audio = load_audio(audio_path)
                audio = self.audio_transform(audio)
                yield {"input": audio, "target": token_id}
            elif self.modality == "audiovisual":
                video = load_video(video_path)
                audio = load_audio(audio_path)
                audio = cut_or_pad(audio, len(video) * self.rate_ratio)
                video = self.video_transform(video)
                audio = self.audio_transform(audio)
                yield {"video": video, "audio": audio, "target": token_id}
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


   



