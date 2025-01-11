import pandas as pd
import matplotlib.pyplot as plt

# Ścieżki
RESULTS_PATH = "results_detailed.csv"
PLOT_PATH = "emotion_accuracy_plot.png"

# Wczytanie wyników
def load_results(results_path):
    return pd.read_csv(results_path)

# Analiza i wizualizacja wyników
def analyze_results(results):
    print("Analiza wyników...")
    grouped = results.groupby("Emotion")[["Precision", "Recall", "F1-Score"]].mean()
    print(grouped)
    
    # Wizualizacja
    grouped.plot(kind="bar", figsize=(10, 6))
    plt.title("Średnie wyniki dla każdej emocji")
    plt.ylabel("Wartość")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Wykres zapisany do: {PLOT_PATH}")

if __name__ == "__main__":
    # Wczytaj dane
    results = load_results(RESULTS_PATH)
    
    # Analizuj wyniki
    analyze_results(results)
