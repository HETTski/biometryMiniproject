import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Ścieżki
FER_CSV_PATH = "fer2013.csv"
RESULTS_PATH = "results.csv"
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

# Funkcja do wczytywania danych
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    pixels = data["pixels"].apply(lambda x: np.array(x.split(), dtype=np.float32))
    X = np.stack(pixels)
    y = data["emotion"]
    return X, y

# Funkcja do trenowania i testowania modeli
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = []
    models = {
        "SVM": SVC(kernel="linear", probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        print(f"Trenuję model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Zapis wyników
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=EMOTION_MAP.values())
        results.append({"Model": name, "Accuracy": accuracy, "Report": report})
        
        # Zapis modelu
        joblib.dump(model, f"{MODEL_PATH}{name.lower().replace(' ', '_')}.pkl")
        print(f"Model {name} zapisany.")
    
    return results

# Główna funkcja
if __name__ == "__main__":
    # Wczytanie danych
    print("Wczytywanie danych...")
    X, y = load_data(FER_CSV_PATH)
    
    # Normalizacja danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Trenowanie modeli
    print("Rozpoczynam trenowanie modeli...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Zapis wyników do pliku
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Wyniki zapisane do: {RESULTS_PATH}")
