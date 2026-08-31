import cv2
import torch
import numpy as np
from pathlib import Path

from src.dl.constants import EMOTIONS, DEVICE
from src.dl.nn import SimpleCNN

CASCADE_PATH = str(Path(__file__).resolve().parent.parent / "face_detection" / "haarcascade_frontalface_default.xml")
MODEL_PATH = "best_model.pt"
IMG_SIZE = 48


def load_model(path: str = MODEL_PATH) -> SimpleCNN:
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def classify_face(model: SimpleCNN, face_gray: np.ndarray) -> tuple[str, float]:
    face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(face_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)
    probs = torch.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return EMOTIONS[idx], probs[idx].item()


def main():
    model = load_model()
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.equalizeHist(face_roi)
            emotion, confidence = classify_face(model, face_roi)

            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{emotion} ({confidence:.0%})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("FER - Neural Network", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
