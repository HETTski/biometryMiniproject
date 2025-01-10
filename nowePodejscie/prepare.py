import pandas as pd
import numpy as np
import os
import cv2

# Ścieżki
FER_CSV_PATH = "fer2013.csv"
OUTPUT_DIR = "fer_images/"

# Mapowanie emocji
EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "suprise",
    6: "neutral",
}

# Funkcja do wczytywania danych i zapisywania obrazów
def save_images_from_csv(csv_path, output_dir):
    data = pd.read_csv(csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for emotion_id, emotion_name in EMOTION_MAP.items():
        emotion_dir = os.path.join(output_dir, emotion_name)
        os.makedirs(emotion_dir, exist_ok=True)

    for idx, row in data.iterrows():
        emotion = EMOTION_MAP[row["emotion"]]
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        usage = row["Usage"]
        output_path = os.path.join(output_dir, emotion, f"{usage}_{idx}.png")
        cv2.imwrite(output_path, pixels)

    print(f"Obrazy zapisane w folderze: {output_dir}")

# Wywołanie funkcji
if __name__ == "__main__":
    save_images_from_csv(FER_CSV_PATH, OUTPUT_DIR)
