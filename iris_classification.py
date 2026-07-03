"""
Iris Flower Classification — CodeAlpha Data Science Internship (Task 1)
Author: Imran Ali Memon
Description: Classify Iris flowers into Setosa, Versicolor, and Virginica
             using multiple ML models with full EDA and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")
RANDOM_STATE = 42
FIGURE_DIR = "figures"

import os
os.makedirs(FIGURE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 60)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n🔍 First 5 Rows:")
print(df.head())
print(f"\n📈 Statistical Summary:")
print(df.describe())
print(f"\n🏷️ Class Distribution:")
print(df["species"].value_counts())
print(f"\n❌ Missing Values: {df.isnull().sum().sum()}")

# ═══════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 2a. Pairplot
fig = sns.pairplot(df, hue="species", diag_kind="kde", corner=True,
                   plot_kws={"alpha": 0.7, "s": 40})
fig.savefig(f"{FIGURE_DIR}/01_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 01_pairplot.png")

# 2b. Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df.drop("species", axis=1).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()
print("✅ Saved: 02_correlation_heatmap.png")

# 2c. Box Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(iris.feature_names):
    ax = axes[i // 2, i % 2]
    sns.boxplot(x="species", y=col, data=df, ax=ax, palette="Set2")
    ax.set_title(col.title(), fontsize=12, fontweight="bold")
plt.suptitle("Feature Distribution by Species", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/03_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 03_boxplots.png")

# 2d. Violin Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(iris.feature_names):
    ax = axes[i // 2, i % 2]
    sns.violinplot(x="species", y=col, data=df, ax=ax, palette="muted", inner="quart")
    ax.set_title(col.title(), fontsize=12, fontweight="bold")
plt.suptitle("Violin Plots — Feature Distribution", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/04_violinplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 04_violinplots.png")

# ═══════════════════════════════════════════════════════════
# 3. DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

X = df.drop("species", axis=1)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════
# 4. MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL TRAINING & COMPARISON")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=RANDOM_STATE),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    # Predict
    y_pred = model.predict(X_test_scaled)
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    results[name] = {
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "y_pred": y_pred,
    }
    print(f"\n{'─' * 50}")
    print(f"📌 {name}")
    print(f"   Test Accuracy:  {acc:.4f}")
    print(f"   CV Accuracy:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ═══════════════════════════════════════════════════════════
# 5. RESULTS VISUALIZATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# 5a. Accuracy Comparison Bar Chart
model_names = list(results.keys())
accuracies = [results[m]["accuracy"] for m in model_names]
cv_means = [results[m]["cv_mean"] for m in model_names]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x - width / 2, accuracies, width, label="Test Accuracy", color="#2196F3")
bars2 = ax.bar(x + width / 2, cv_means, width, label="CV Mean Accuracy", color="#FF9800")
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Model Comparison — Accuracy Scores", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.legend()
ax.set_ylim(0.8, 1.02)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/05_model_comparison.png", dpi=150)
plt.close()
print("✅ Saved: 05_model_comparison.png")

# 5b. Confusion Matrices
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_pred = results[best_model_name]["y_pred"]

fig, axes = plt.subplots(1, len(models), figsize=(20, 4))
for idx, (name, data) in enumerate(results.items()):
    cm = confusion_matrix(y_test, data["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    axes[idx].set_title(name, fontsize=10, fontweight="bold")
    axes[idx].set_ylabel("Actual" if idx == 0 else "")
    axes[idx].set_xlabel("Predicted")
plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/06_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 06_confusion_matrices.png")

# 5c. Best Model Report
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"\n📋 Classification Report ({best_model_name}):")
print(classification_report(y_test, best_pred, target_names=iris.target_names))

# Summary Table
summary_df = pd.DataFrame({
    "Model": model_names,
    "Test Accuracy": [f"{results[m]['accuracy']:.4f}" for m in model_names],
    "CV Mean ± Std": [f"{results[m]['cv_mean']:.4f} ± {results[m]['cv_std']:.4f}" for m in model_names],
})
print("\n📊 Final Comparison Table:")
print(summary_df.to_string(index=False))

print("\n✅ Iris Classification Complete! Check 'figures/' for all visualizations.")
