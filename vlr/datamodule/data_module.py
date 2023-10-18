import os

import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset, AVDatasetIterable
from .samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from .transforms import AudioTransform, VideoTransform


# https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517
def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


class DataModule(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.cfg.devices = torch.cuda.device_count()
        self.total_gpus = self.cfg.devices * self.cfg.trainer.num_nodes
        self.dataset = load_dataset(os.path.join(self.cfg.data.dataset.root_dir, self.cfg.data.dataset.train_dir))
        if self.cfg.data.select != -1:
            self.dataset["train"] = self.dataset["train"].select(range(self.cfg.data.select))
        # split dataset
        self.dataset = self.dataset["train"].train_test_split(test_size=self.cfg.data.dataset.test_size)

    def _dataloader(self, ds, collate_fn, sampler=None):
        if self.cfg.streaming:
            return torch.utils.data.DataLoader(
                ds,
                batch_size=self.cfg.data.batch_size,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else: 
            return torch.utils.data.DataLoader(
                ds,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                batch_sampler=sampler,
                collate_fn=collate_fn,
            )

    def train_dataloader(self):
        if self.cfg.streaming:
            train_ds = AVDatasetIterable(root_dir=self.cfg.data.dataset.root_dir,
                dataset=self.dataset["train"],
                modality=self.cfg.data.modality,
                audio_transform=AudioTransform("train"),
                video_transform=VideoTransform("train"),
            )
            return self._dataloader(train_ds, collate_pad)
        else:
            train_ds = AVDataset(
                root_dir=self.cfg.data.dataset.root_dir,
                dataset=self.dataset["train"],
                modality=self.cfg.data.modality,
                audio_transform=AudioTransform("train"),
                video_transform=VideoTransform("train"),
            )
            sampler = ByFrameCountSampler(train_ds, self.cfg.data.max_frames, self.cfg.data.shuffle)
            if self.total_gpus > 1:
                sampler = DistributedSamplerWrapper(sampler)
            else:
                sampler = RandomSamplerWrapper(sampler)
            return self._dataloader(train_ds, collate_pad, sampler)

    def val_dataloader(self):
        if self.cfg.streaming:
            val_ds = AVDatasetIterable(
                root_dir=self.cfg.data.dataset.root_dir,
                dataset=self.dataset["test"],
                modality=self.cfg.data.modality,
                audio_transform=AudioTransform("val"),
                video_transform=VideoTransform("val"),
            )
            return self._dataloader(val_ds, collate_pad)
        else:
            val_ds = AVDataset(
                root_dir=self.cfg.data.dataset.root_dir,
                dataset=self.dataset["test"],
                modality=self.cfg.data.modality,
                audio_transform=AudioTransform("val"),
                video_transform=VideoTransform("val"),
            )
            sampler = ByFrameCountSampler(
                val_ds, self.cfg.data.max_frames_val, shuffle=False
            )
            if self.total_gpus > 1:
                sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
            return self._dataloader(val_ds, collate_pad, sampler)
        
    def test_dataloader(self):
        dataset = AVDataset(
            root_dir=self.cfg.data.dataset.root_dir,
            dataset=self.dataset["test"],
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform(
                "test", snr_target=self.cfg.decode.snr_target
            ),
            video_transform=VideoTransform("test"),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader
