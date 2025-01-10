import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ścieżki
RESULTS_PATH = "results.csv"

# Funkcja do analizy wyników
def analyze_results(results_path):
    # Wczytanie wyników
    results = pd.read_csv(results_path)
    
    # Wyświetlenie dokładności modeli
    print("Dokładność modeli:")
    print(results[["Model", "Accuracy"]])
    
    # Wizualizacja dokładności
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Model", y="Accuracy", data=results, palette="viridis")
    plt.title("Dokładność modeli na zbiorze testowym")
    plt.ylabel("Dokładność")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.show()
    plt.savefig("accuracy_plot.png", dpi=300)


    # Wyświetlenie raportów klasyfikacyjnych
    for index, row in results.iterrows():
        print(f"\nRaport dla modelu {row['Model']}:\n")
        print(row["Report"])

# Główna funkcja
if __name__ == "__main__":
    print("Analiza wyników...")
    analyze_results(RESULTS_PATH)
    
