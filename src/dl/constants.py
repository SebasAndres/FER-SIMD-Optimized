from pathlib import Path

import torch

EMOTIONS = ["happy", "neutral", "sad"]
LABEL_MAP = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "dataset" / "fer2013"

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
NUM_CLASSES = len(EMOTIONS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
