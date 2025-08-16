import io
import os
from tqdm import tqdm
from PIL import Image
import polars as pl
import deeplake


def download_fer2013_dataset():
    """
    Downloads the FER2013 dataset from Deep Lake and converts it to PNG format.
    """
    ds = deeplake.load('hub://activeloop/fer2013-train', read_only=True)
    output_dir = 'dataset/fer2013'
    os.makedirs(output_dir, exist_ok=True)
    emotion_labels = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
    }
    with tqdm(total=len(ds), desc='Converting to png format') as pbar:
        for i, sample in enumerate(ds):

            # Extract image and label
            img_array = sample.images.numpy()  # shape: (height, width), grayscale
            label_num = int(sample.labels.numpy().item()) # numeric label (0-6)

            # Convert to PIL Image
            img = Image.fromarray(img_array.astype('uint8'), mode='L')  # 'L' for grayscale

            # Save image with label in filename or separately
            label_dir = os.path.join(output_dir, emotion_labels[label_num])

            # Create label directory if it doesn't exist
            os.makedirs(label_dir, exist_ok=True)
            img.save(os.path.join(label_dir, f'{i}.png'))

            pbar.update(1)


if __name__ == "__main__":
    print("> Downloading dataset...")
    download_fer2013_dataset()
    print("> Download complete.")