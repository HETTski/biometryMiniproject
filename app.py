import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import cv2

# Step 1: Load and preprocess the dataset
def load_images_from_folder(folder, target_size=(100, 100)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    images.append(img.flatten())
                    labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset (FERET or any other dataset path)
dataset_path = "./dataset"  # Replace with the actual path
X, y = load_images_from_folder(dataset_path)

# Normalize pixel values to range [0, 1]
X = X / 255.0

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 2: Feature extraction using PCA
def extract_features_pca(X_train, X_test, n_components=100):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

X_train_pca, X_test_pca = extract_features_pca(X_train, X_test)

# Step 3: Model training and ensemble methods
# Base classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(probability=True, kernel='linear', random_state=42)
dt = DecisionTreeClassifier(random_state=42)

# Random Subspace Method
class RandomSubspaceEnsemble:
    def __init__(self, base_model, n_estimators=10, subspace_size=0.5):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.subspace_size = subspace_size
        self.models = []
        self.features_idx = []

    def fit(self, X, y):
        n_features = X.shape[1]
        subspace_size = int(self.subspace_size * n_features)
        for _ in range(self.n_estimators):
            features = np.random.choice(range(n_features), subspace_size, replace=False)
            model = self.base_model()
            model.fit(X[:, features], y)
            self.models.append(model)
            self.features_idx.append(features)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, (model, features) in enumerate(zip(self.models, self.features_idx)):
            predictions[:, i] = model.predict(X[:, features])
        return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)

random_subspace = RandomSubspaceEnsemble(RandomForestClassifier, n_estimators=10, subspace_size=0.5)
random_subspace.fit(X_train_pca, y_train)
random_subspace_preds = random_subspace.predict(X_test_pca)

# Voting Ensemble
voting = VotingClassifier(estimators=[('rf', rf), ('knn', knn), ('svm', svm), ('dt', dt)], voting='soft')
voting.fit(X_train_pca, y_train)
voting_preds = voting.predict(X_test_pca)

# Step 4: Evaluation
def evaluate_model(name, y_test, predictions):
    print(f"\n{name} Performance:")
    print(classification_report(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.show()
    return accuracy

random_subspace_accuracy = evaluate_model("Random Subspace", y_test, random_subspace_preds)
voting_accuracy = evaluate_model("Voting Ensemble", y_test, voting_preds)

# Step 5: Visualization
methods = ["Random Subspace", "Voting Ensemble"]
accuracies = [random_subspace_accuracy, voting_accuracy]

plt.figure(figsize=(8, 5))
sns.barplot(x=methods, y=accuracies, palette="viridis")
plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")
plt.show()

print("Mini Project Completed Successfully!")
