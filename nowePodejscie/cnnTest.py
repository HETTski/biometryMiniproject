import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Ścieżka do danych
FER_CSV_PATH = "fer2013.csv"
RESULTS_PATH = "emotion_influence_results.csv"
MODEL_PATH = "models/"

# Poprawne mapowanie emocji
EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "suprise",
    6: "neutral",
}

# Funkcja do wczytania danych i przygotowania obrazów
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    pixels = data["pixels"].apply(lambda x: np.array(x.split(), dtype=np.float32))
    X = np.stack(pixels)
    y = data["emotion"]
    return X, y

# Funkcja do budowy modelu CNN
def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(7, activation='softmax')  # 7 klas emocji
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Funkcja do testowania wpływu emocji
def test_emotion_influence(X_train, X_test, y_train, y_test):
    results = []

    # Trening na poszczególnych emocjach
    for emotion_id, emotion_name in EMOTION_MAP.items():
        print(f"Testowanie emocji: {emotion_name}")
        
        # Filtrujemy dane tylko dla jednej emocji
        emotion_train = X_train[y_train == emotion_id]
        emotion_test = X_test[y_test == emotion_id]
        emotion_train_labels = y_train[y_train == emotion_id]
        emotion_test_labels = y_test[y_test == emotion_id]
        
        # Budowanie modelu CNN
        model = build_cnn_model((48, 48, 1))
        
        # Trenowanie modelu na GPU
        with tf.device('/GPU:0'):  # Wymuszenie używania GPU
            model.fit(emotion_train, emotion_train_labels, epochs=10, batch_size=64, validation_data=(emotion_test, emotion_test_labels), verbose=1)
        
        # Ocena modelu
        y_pred = np.argmax(model.predict(emotion_test), axis=1)
        accuracy = accuracy_score(emotion_test_labels, y_pred)
        report = classification_report(emotion_test_labels, y_pred, target_names=[emotion_name])

        results.append({
            "Emotion": emotion_name,
            "Accuracy": accuracy,
            "Classification Report": report
        })
        
    # Zapis wyników do pliku
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Wyniki zapisane do: {RESULTS_PATH}")
    
    return results_df

# Główna funkcja
if __name__ == "__main__":
    # Wczytanie danych
    print("Wczytuję dane...")
    X, y = load_data(FER_CSV_PATH)
    
    # Normalizacja danych
    X = X / 255.0

    # Podział na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Przekształcenie danych do odpowiednich kształtów (48x48 pikseli, 1 kanał - grayscale)
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)

    # Testowanie wpływu emocji na dokładność
    results = test_emotion_influence(X_train, X_test, y_train, y_test)

    # Wyświetlanie wyników
    print("\nWyniki testów:")
    print(results)

    # Wizualizacja wyników
    plt.figure(figsize=(12, 6))
    plt.bar(results["Emotion"], results["Accuracy"], color="skyblue")
    plt.title("Wpływ emocji na dokładność rozpoznawania twarzy")
    plt.ylabel("Dokładność")
    plt.xlabel("Emocje")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
