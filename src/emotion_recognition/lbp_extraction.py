from skimage.feature import local_binary_pattern
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def extract_lbp(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist

if __name__ == "__main__":
    os.makedirs("data/lbp_faces", exist_ok=True)
    for _dir in os.listdir("dataset/fer2013"):
        dir_path = f"dataset/fer2013/{_dir}"
        images = [img for img in os.listdir(dir_path) if img.endswith(".png")]
        rows = []
        for _img in images:
            img_path = os.path.join(dir_path, _img)
            img = plt.imread(img_path)
            lbp_hist = extract_lbp(img)
            row = list(lbp_hist) + [_dir]
            rows.append(row)
        lbp_hist_df = pd.DataFrame(rows, columns=[*range(256), 'label'])
        lbp_hist_df.to_csv(f"data/lbp_faces/{_dir}_faces.csv", mode='a', header=False, index=False)