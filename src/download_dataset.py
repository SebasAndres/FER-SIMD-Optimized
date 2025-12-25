import kaggle
import os
import zipfile
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd


def download_fer2013_dataset():
    """
    Downloads the FER2013 dataset from Kaggle and converts it to PNG format.

    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Set up Kaggle API credentials:
       - Go to https://www.kaggle.com/settings -> API -> Create New Token
       - Save kaggle.json to ~/.kaggle/kaggle.json
       - chmod 600 ~/.kaggle/kaggle.json
    """

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

    # Create emotion directories
    for label in emotion_labels.values():
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    # Download dataset from Kaggle
    zip_path = os.path.join(output_dir, 'fer2013.zip')
    csv_path = os.path.join(output_dir, 'fer2013.csv')

    if not os.path.exists(csv_path):
        print("> Downloading from Kaggle...")
        kaggle.api.dataset_download_files(
            'msambare/fer2013',
            path=output_dir,
            unzip=False
        )

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)

    # The msambare/fer2013 dataset contains train/test folders with images
    # Let's check what we got
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    if os.path.exists(train_dir):
        print("> Dataset already in image format, reorganizing...")
        # Dataset is already in image format, just need to move files
        for split_dir in [train_dir, test_dir]:
            if not os.path.exists(split_dir):
                continue
            for emotion_folder in os.listdir(split_dir):
                src_folder = os.path.join(split_dir, emotion_folder)
                if not os.path.isdir(src_folder):
                    continue
                dst_folder = os.path.join(output_dir, emotion_folder.lower())
                os.makedirs(dst_folder, exist_ok=True)

                for img_file in os.listdir(src_folder):
                    src_path = os.path.join(src_folder, img_file)
                    # Add prefix to avoid name conflicts between train/test
                    prefix = 'train_' if 'train' in split_dir else 'test_'
                    dst_path = os.path.join(dst_folder, prefix + img_file)
                    if os.path.isfile(src_path):
                        os.rename(src_path, dst_path)

        # Clean up train/test directories
        import shutil
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        print("> Reorganization complete.")
    else:
        # Fallback: CSV format (original fer2013 format)
        print("> Processing CSV format...")
        df = pd.read_csv(csv_path)

        with tqdm(total=len(df), desc='Converting to png format') as pbar:
            for i, row in df.iterrows():
                pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                img_array = pixels.reshape(48, 48)
                label_num = int(row['emotion'])

                img = Image.fromarray(img_array, mode='L')
                label_dir = os.path.join(output_dir, emotion_labels[label_num])
                img.save(os.path.join(label_dir, f'{i}.png'))

                pbar.update(1)


if __name__ == "__main__":
    print("> Downloading dataset...")
    download_fer2013_dataset()
    print("> Download complete.")