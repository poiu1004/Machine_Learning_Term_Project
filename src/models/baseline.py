import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "features.csv"))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "figures"))

def train_and_evaluate():
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    # 1. Load Data
    print("Loading extracted features...")
    if not os.path.exists(DATA_PATH):
        print("Error: features.csv not found. Please run DAY 2 extraction first.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # Remove ambiguous labels (-1)
    df = df[df['label'] != -1].copy()
    print(f"Total valid samples (Label 0 or 1): {len(df)}")
    
    features = ['rg', 'mean_dist', 'std_dist', 'clash_count', 'density', 'n_residues']
    X = df[features]
    y = df['label']
    groups = df['target'] # Use target for GroupShuffleSplit to prevent data leakage!

    # 2. Target-level Split (Prevent Data Leakage)
    print("\nSplitting data at the target level to prevent leakage...")
    # n_splits=1 gives us a single train/test split.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Train targets: {df.iloc[train_idx]['target'].unique()}")
    print(f"Test targets: {df.iloc[test_idx]['target'].unique()}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Standardization (crucial for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Models
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    print("--- Training SVM ---")
    svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    svm.fit(X_train_scaled, y_train)

    # 5. Evaluate
    def evaluate_model(name, model, X_t, y_t):
        preds = model.predict(X_t)
        probs = model.predict_proba(X_t)[:, 1]
        
        acc = accuracy_score(y_t, preds)
        f1 = f1_score(y_t, preds)
        auc = roc_auc_score(y_t, probs)
        
        print(f"\n[{name} Results]")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_t, preds))
        return probs

    rf_probs = evaluate_model("Random Forest", rf, X_test, y_test)
    svm_probs = evaluate_model("SVM", svm, X_test_scaled, y_test)

    # 6. Feature Importance (Random Forest)
    print("\nExtracting Feature Importance from Random Forest...")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Feature Importances (Geometric Handcrafted Features)")
    plt.bar(range(X.shape[1]), importances[indices], align="center", color='#4C72B0')
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.ylabel("Relative Importance")
    plt.tight_layout()
    importance_path = os.path.join(FIGURES_DIR, "feature_importance_rf.png")
    plt.savefig(importance_path)
    print(f"Feature importance plot saved to {importance_path}")

if __name__ == "__main__":
    train_and_evaluate()
