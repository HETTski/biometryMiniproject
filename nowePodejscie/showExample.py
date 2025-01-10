import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ścieżka do CSV
FER_CSV_PATH = "fer2013.csv"

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

# Funkcja do wyświetlania przykładów obrazów
def display_sample_images(data, num_samples=3):
    plt.figure(figsize=(12, 10))
    for emotion_id, emotion_name in EMOTION_MAP.items():
        emotion_data = data[data["emotion"] == emotion_id]
        if len(emotion_data) > 0:
            samples = emotion_data.sample(n=min(num_samples, len(emotion_data)), random_state=42)
            for i, pixels in enumerate(samples["pixels"]):
                image = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
                plt.subplot(len(EMOTION_MAP), num_samples, emotion_id * num_samples + i + 1)
                plt.imshow(image, cmap="gray")
                plt.title(emotion_name)
                plt.axis("off")
    plt.tight_layout()
    plt.show()

# Funkcja do wizualizacji rozkładu emocji
def visualize_emotion_distribution(data):
    counts = data["emotion"].value_counts()
    counts.index = counts.index.map(EMOTION_MAP)
    counts.plot(kind='bar')
    plt.title("Rozkład emocji w bazie FER+")
    plt.xlabel("Emocje")
    plt.ylabel("Liczba zdjęć")
    plt.show()

# Główna funkcja
if __name__ == "__main__":
    data = pd.read_csv(FER_CSV_PATH)
    display_sample_images(data, num_samples=3)
    visualize_emotion_distribution(data)
