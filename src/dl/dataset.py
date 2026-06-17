import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from sklearn.model_selection import train_test_split

from src.dl.constants import EMOTIONS, LABEL_MAP, DATA_DIR


class FERDataset(Dataset):
    """FER-2013 dataset filtered to 3 emotions (happy, neutral, sad).

    Split into train/val/test using filename prefixes and a validation
    carve-out from the training set.
    """

    def __init__(self, split: str = "train", val_ratio: float = 0.15,
                 transform=None, seed: int = 42, data_dir: Path = DATA_DIR):
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        all_train = []
        all_test = []

        for emotion in EMOTIONS:
            emotion_dir = data_dir / emotion
            label = LABEL_MAP[emotion]
            for fname in os.listdir(emotion_dir):
                if not fname.endswith(".jpg"):
                    continue
                path = str(emotion_dir / fname)
                if fname.startswith("train_"):
                    all_train.append((path, label))
                elif fname.startswith("test_"):
                    all_test.append((path, label))

        if split == "test":
            self.samples = all_test
        else:
            train_paths, train_labels = zip(*all_train)
            tr_paths, val_paths, tr_labels, val_labels = train_test_split(
                train_paths, train_labels,
                test_size=val_ratio, random_state=seed, stratify=train_labels,
            )
            if split == "train":
                self.samples = list(zip(tr_paths, tr_labels))
            else:
                self.samples = list(zip(val_paths, val_labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = read_image(path, mode=ImageReadMode.GRAY).float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict[str, int]:
        counts = {e: 0 for e in EMOTIONS}
        for _, label in self.samples:
            counts[EMOTIONS[label]] += 1
        return counts
