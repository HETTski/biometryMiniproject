import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ścieżki
FER_CSV_PATH = "fer2013.csv"
RESULTS_PATH = "results_detailed.csv"
MODEL_PATH = "models/"

# Poprawne mapowanie emocji
EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

# Funkcja do wczytywania danych
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    X = data['pixels'].apply(lambda x: np.array(x.split(), dtype=np.float32))
    X = np.stack(X.values)
    y = data['emotion'].values
    return X, y

# Funkcja do trenowania i testowania modeli
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear SVM": SVC(kernel="linear"),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    results = []
    os.makedirs(MODEL_PATH, exist_ok=True)  # Ensure the directory exists

    for name, model in models.items():
        print(f"Trenuję model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Szczegółowe wyniki
        report = classification_report(y_test, y_pred, target_names=EMOTION_MAP.values(), output_dict=True)
        
        detailed_results = [
            {
                "Model": name,
                "Emotion": emotion,
                "Precision": report[emotion]["precision"],
                "Recall": report[emotion]["recall"],
                "F1-Score": report[emotion]["f1-score"],
            }
            for emotion in EMOTION_MAP.values() if emotion in report
        ]
        results.extend(detailed_results)

        # Zapis modelu
        joblib.dump(model, f"{MODEL_PATH}{name.lower().replace(' ', '_')}.pkl")
        print(f"Model {name} zapisany.")
    
    return results

# Główna funkcja
if __name__ == "__main__":
    print("Wczytywanie danych...")
    X, y = load_data(FER_CSV_PATH)
    
    # Ograniczenie liczby danych (opcja przyspieszenia procesu)
    X, y = X[:1000], y[:1000]
    
    # Normalizacja danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Redukcja wymiarowości za pomocą PCA
    print("Redukcja wymiarowości za pomocą PCA...")
    pca = PCA(n_components=550)  # Redukcja do 50 wymiarów
    X = pca.fit_transform(X)

    # Podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Trenowanie modeli
    print("Rozpoczynam trenowanie modeli...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Zapis wyników do pliku
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Wyniki zapisane do: {RESULTS_PATH}")
