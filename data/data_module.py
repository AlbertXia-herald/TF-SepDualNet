import os
import random
import librosa
import numpy as np
import torch
import lightning as L
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from util import unique_labels


class AudioDataset(Dataset):
    """
    Dataset containing pairs of audio waveform and filename.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        subset (str): Name of required meta file. e.g. ``train``, ``valid``, ``test``...
        sampling_rate (int): Sampling rate of waveforms.
    """

    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.subset = subset
        self.sr = sampling_rate
        self.meta_subset = pd.read_csv(f"{self.meta_dir}/{self.subset}.csv", sep='\t')

        # 检查文件是否存在，过滤无效条目
        self._filter_missing_files()

    def _filter_missing_files(self):
        """过滤元数据中不存在的音频文件"""
        valid_indices = []
        missing_files = []
        for idx, row in self.meta_subset.iterrows():
            filename = row["filename"]
            # 拼接完整路径（使用 os.path.join 兼容不同系统）
            file_path = os.path.join(self.audio_dir, filename)
            if os.path.exists(file_path):
                valid_indices.append(idx)
            else:
                missing_files.append(file_path)
        # 保留有效数据
        self.meta_subset = self.meta_subset.loc[valid_indices].reset_index(drop=True)
        # 打印缺失文件信息
        if missing_files:
            print(f"警告：{self.subset} 集缺失 {len(missing_files)} 个文件，已过滤，示例：")
            for path in missing_files[:3]:
                print(f"  - {path}")

    def __len__(self):
        return len(self.meta_subset)

    def __getitem__(self, i):
        # Get the ith row from the meta csv file
        row_i = self.meta_subset.iloc[i]
        # Get the filename of audio
        filename = row_i["filename"]
        # Load audio waveform with a resample rate
        wav, _ = librosa.load(f"{self.audio_dir}/{filename}", sr=self.sr)
        wav = torch.from_numpy(wav)
        return wav, filename


class AudioLabelsDataset(AudioDataset):
    """
    Dataset containing tuples of audio waveform, scene label, device label and city label.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        subset (str): Name of required meta file. e.g. ``train``, ``valid``, ``test``...
        sampling_rate (int): Sampling rate of waveforms.
    """
    def __init__(self, meta_dir: str, audio_dir: str, subset: str, sampling_rate: int = 16000):
        super().__init__(meta_dir, audio_dir, subset, sampling_rate)

    def __getitem__(self, i):
        # Get the filename
        wav, filename = super().__getitem__(i)
        scene_label = filename.split('/')[-1].split('-')[0]
        device_label = filename.split('-')[-1].split('.')[0]
        city_label = filename.split('-')[1]
        # Encode the scene labels from string to integers
        scene_label = unique_labels['scene'].index(scene_label)
        scene_label = torch.from_numpy(np.array(scene_label, dtype=np.int64))
        # Encode the device labels from string to integers
        device_label = unique_labels['device'].index(device_label)
        device_label = torch.from_numpy(np.array(device_label, dtype=np.int64))
        # Encode the city labels from string to integers
        city_label = unique_labels['city'].index(city_label)
        city_label = torch.from_numpy(np.array(city_label, dtype=np.int64))
        return wav, scene_label, device_label, city_label


class AudioLabelsDatasetWithLogits(AudioLabelsDataset):
    """
    AudioLabelsDataset with additional logits of teacher ensemble for knowledge distillation.

    Args:
        logits_files (list): List of directories of teacher logits. e.g. ["path/to/logit/predictions.pt", ...]
    """
    def __init__(self, logits_files: list, **kwargs):
        super().__init__(**kwargs)
        logits_all = []
        for file in logits_files:
            # Load teacher logit and append to a list
            logit = torch.load(file).float()
            logits_all.append(logit)
        # Average the logits from multiple teachers
        logit_all = sum(logits_all)
        self.teacher_logit = logit_all / len(logits_files)

    def __getitem__(self, i):
        wav, scene_label, device_label, city_label = super().__getitem__(i)
        return wav, scene_label, device_label, city_label, self.teacher_logit[i]


class DCASEDataModule(L.LightningDataModule):
    """
    DCASE DataModule wrapping train, validation, test and predict DataLoaders.

    Args:
        meta_dir (str): Directory of meta files, which should include meta files in csv formate.
        audio_dir (str): Directory of audios.
        batch_size (int): Batch size.
        num_workers (int): Number of workers to use for DataLoaders. Will save time for loading data to GPU but increase CPU usage.
        pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them. Will save time for data loading.
        logits_files (list): List of directories of teacher logits, e.g. ["path/to/logit/predictions.pt", ...]. If not ``None``, knowledge distillation will be applied.
        train_subset (str): Name of train meta file. e.g. train, split5, split10...
        test_subset (str): Name of test meta file.
        predict_subset (str): Name of predict meta file.
    """
    def __init__(self, meta_dir: str, audio_dir: str, batch_size: int = 16, num_workers: int = 0, pin_memory: bool=False,
                 logits_files=None, train_subset="train", test_subset="test", predict_subset="test", **kwargs):
        super().__init__()
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.predict_subset = predict_subset
        self.logits_files = logits_files
        self.kwargs = kwargs

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Add teacher logits to the dataset if using knowledge distillation
            if self.logits_files is not None:
                self.train_set = AudioLabelsDatasetWithLogits(logits_files=self.logits_files, meta_dir=self.meta_dir, audio_dir=self.audio_dir, subset=self.train_subset, **self.kwargs)
            else:
                self.train_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset=self.train_subset, **self.kwargs)
            self.valid_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset="valid", **self.kwargs)
        if stage == "validate":
            self.valid_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset="valid", **self.kwargs)
        if stage == "test":
            self.test_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset=self.test_subset, **self.kwargs)
        if stage == "predict":
            self.predict_set = AudioDataset(self.meta_dir, self.audio_dir, subset=self.predict_subset, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

class AudioLabelsDatasetWithTarget(AudioLabelsDataset):
    def __init__(self, target_set="test", **kwargs):
        super().__init__(**kwargs)
        self.target_set = target_set

        self.meta_test = pd.read_csv(f"{self.meta_dir}/{self.target_set}.csv", sep='\t')

    def __getitem__(self, i):
        wav_s, scene_label_s, device_label_s, city_label_s = super().__getitem__(i)
        # Get the ith row from the meta csv file，随机选取一行，使其成为目标域
        test_i = random.randint(0, len(self.meta_test) - 1)
        row_i = self.meta_test.iloc[test_i]
        # Get the filename of audio
        filename = row_i["filename"]
        # Load audio waveform with a resample rate
        wav_t, _ = librosa.load(f"{self.audio_dir}/{filename}", sr=self.sr)
        wav_t = torch.from_numpy(wav_t)
        scene_label_t = filename.split('/')[-1].split('-')[0]
        device_label_t = filename.split('-')[-1].split('.')[0]
        city_label_t = filename.split('-')[1]
        # Encode the scene labels from string to integers，得到数字索引
        scene_label_t = unique_labels['scene'].index(scene_label_t)
        scene_label_t = torch.from_numpy(np.array(scene_label_t, dtype=np.int64))
        # Encode the device labels from string to integers
        device_label_t = unique_labels['device'].index(device_label_t)
        device_label_t = torch.from_numpy(np.array(device_label_t, dtype=np.int64))
        # Encode the city labels from string to integers
        city_label_t = unique_labels['city'].index(city_label_t)
        city_label_t = torch.from_numpy(np.array(city_label_t, dtype=np.int64))
        return wav_s, scene_label_s, device_label_s, city_label_s, wav_t, scene_label_t, device_label_t, city_label_t

class DataModuleWithTarget(DCASEDataModule):
    def __init__(self, target_set, valid_subset: str = "valid", **kwargs):
        super().__init__(**kwargs)
        self.target_set = target_set
        self.valid_subset = valid_subset

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Add teacher logits to the dataset if using knowledge distillation
            self.train_set = AudioLabelsDatasetWithTarget(target_set=self.target_set, meta_dir=self.meta_dir, audio_dir=self.audio_dir, subset=self.train_subset, **self.kwargs)
            self.valid_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset=self.valid_subset, **self.kwargs)
        if stage == "validate":
            self.valid_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset=self.valid_subset, **self.kwargs)
        if stage == "test":
            self.test_set = AudioLabelsDataset(self.meta_dir, self.audio_dir, subset=self.test_subset, **self.kwargs)
        if stage == "predict":
            self.predict_set = AudioDataset(self.meta_dir, self.audio_dir, subset=self.predict_subset, **self.kwargs)