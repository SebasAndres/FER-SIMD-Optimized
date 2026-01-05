import kaggle
import os
import zipfile
import shutil


def download_fer2013_dataset():
    output_dir = '../data/dataset/fer2013'
    os.makedirs(output_dir, exist_ok=True)

    emotion_labels = {
        "happy",
        "sad",
        "neutral"
        # "angry",
        # "disgust",
        # "fear",
        # "surprise",
    }

    for label in emotion_labels:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    zip_path = os.path.join(output_dir, 'fer2013.zip')
    csv_path = os.path.join(output_dir, 'fer2013.csv')

    if not os.path.exists(csv_path):
        kaggle.api.dataset_download_files(
            'msambare/fer2013',
            path=output_dir,
            unzip=False
        )

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)

    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    for split_dir in [train_dir, test_dir]:
        for emotion_folder in os.listdir(split_dir):
            src_folder = os.path.join(split_dir, emotion_folder)
            if not os.path.isdir(src_folder):
                continue
            dst_folder = os.path.join(output_dir, emotion_folder.lower())
            os.makedirs(dst_folder, exist_ok=True)
            for img_file in os.listdir(src_folder):
                src_path = os.path.join(src_folder, img_file)
                prefix = 'train_' if 'train' in split_dir else 'test_'
                dst_path = os.path.join(dst_folder, prefix + img_file)
                if os.path.isfile(src_path):
                    os.rename(src_path, dst_path)

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("Descargando dataset de Kaggle...")
    try:
        download_fer2013_dataset()
        print("Descarga finalizada.")
    except Exception as e:
        print(f"Error durante la descarga: {e}")
