# 본 코드는 Test Set의 오염을 방지하기 위해 3단계 데이터 분할 및 Validation 기반 하이퍼파라미터 튜닝을 수행했음

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pointnet import ProteinPointCloudDataset, PointNetCls
from tqdm import tqdm

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "features.csv"))

def get_subset_labels(subset):
    """Fast label extraction from Dataset memory to avoid slow sequential PDB parsing."""
    if hasattr(subset, 'dataset') and hasattr(subset, 'indices'):
        return [subset.dataset.data[i]['label'] for i in subset.indices]
    elif hasattr(subset, 'data'):
        return [item['label'] for item in subset.data]
    else:
        return [item[1].item() for item in subset]

def train_and_tune_pointnet(train_dataset, val_dataset, test_dataset, scenario_name, epochs=25):
    """Trains and tunes PointNet using Train and Val sets, then evaluates on Test set once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- PointNet Hyperparameter Tuning for {scenario_name} (Using {device}) ---")
    
    # 1. Setup DataLoaders (Use drop_last=True for Train, and batch_size=len(...) for Val/Test to prevent BatchNorm size=1 errors)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)
    
    # 2. Fast Class Weights calculation
    train_labels = get_subset_labels(train_dataset)
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (2.0 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[{scenario_name}] Class Counts: {class_counts}")
    print(f"[{scenario_name}] Class Weights (Stable vs Defective): {class_weights}")
    
    # Hyperparameters to tune
    configs = [
        {"lr": 0.001, "name": "Adam, LR=0.001, ClassWeight=Applied"},
        {"lr": 0.0005, "name": "Adam, LR=0.0005, ClassWeight=Applied"}
    ]
    
    best_overall_auc = -1.0
    best_overall_state = None
    best_overall_config = None
    best_overall_epoch_info = {}
    
    for config in configs:
        lr = config["lr"]
        config_name = config["name"]
        print(f"\n>> Tuning Config: {config_name}")
        
        model = PointNetCls(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_config_auc = -1.0
        best_config_state = None
        best_config_epoch_info = {}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for points, batch_labels in train_loader:
                points, batch_labels = points.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                logits, _, _ = model(points)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * points.size(0)
            epoch_train_loss = running_loss / len(train_dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds, all_probs, all_labels = [], [], []
            with torch.no_grad():
                for points, batch_labels in val_loader:
                    points, batch_labels = points.to(device), batch_labels.to(device)
                    logits, _, _ = model(points)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item() * points.size(0)
                    
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds = torch.argmax(logits, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
            
            epoch_val_loss = val_loss / len(val_dataset)
            try:
                val_auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                val_auc = 0.5
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            # Save the best epoch for this config based on Val AUC
            if val_auc > best_config_auc:
                best_config_auc = val_auc
                best_config_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_config_epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "val_auc": val_auc,
                    "val_f1": val_f1
                }
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
        
        print(f"-> Best for Config [{config_name}] at Epoch {best_config_epoch_info['epoch']} | Val AUC: {best_config_auc:.4f}")
        
        # Compare to find the overall best configuration
        if best_config_auc > best_overall_auc:
            best_overall_auc = best_config_auc
            best_overall_state = best_config_state
            best_overall_config = config
            best_overall_epoch_info = best_config_epoch_info
            
    print(f"\n>> Final Selected PointNet Config: {best_overall_config['name']} (Best Epoch: {best_overall_epoch_info['epoch']} with Val AUC: {best_overall_auc:.4f})")
    
    # 3. Load the best overall checkpoint and evaluate ONCE on Test set
    best_model = PointNetCls(num_classes=2).to(device)
    best_model.load_state_dict({k: v.to(device) for k, v in best_overall_state.items()})
    best_model.eval()
    
    test_probs, test_preds, test_labels = [], [], []
    with torch.no_grad():
        for points, batch_labels in test_loader:
            points, batch_labels = points.to(device), batch_labels.to(device)
            logits, _, _ = best_model(points)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch_labels.cpu().numpy())
            
    try:
        test_auc = roc_auc_score(test_labels, test_probs)
    except ValueError:
        test_auc = 0.5
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_cm = confusion_matrix(test_labels, test_preds)
    
    return test_auc, test_f1, test_cm, best_overall_epoch_info["train_loss"], best_overall_epoch_info["val_loss"], best_overall_epoch_info["epoch"]

def main():
    print("Loading features data...")
    df = pd.read_csv(DATA_PATH)
    df = df[df['label'] != -1].reset_index(drop=True)
    
    features = ['rg', 'mean_dist', 'std_dist', 'clash_count', 'density', 'n_residues']
    X = df[features]
    y = df['label']
    
    full_dataset = ProteinPointCloudDataset(DATA_PATH, num_points=128)
    
    # ---------------------------------------------------------
    # Scenario A: Random Split (6:2:2)
    # ---------------------------------------------------------
    print("\n=========================================================")
    print("[Scenario A: Random 3-Stage Split (60% / 20% / 20%)]")
    print("=========================================================")
    indices = np.arange(len(df))
    
    # 60% Train, 40% Temp (which splits 50/50 into 20% Val, 20% Test)
    train_idx_A, temp_idx_A = train_test_split(indices, test_size=0.4, random_state=42)
    val_idx_A, test_idx_A = train_test_split(temp_idx_A, test_size=0.5, random_state=42)
    
    X_train_A, X_val_A, X_test_A = X.iloc[train_idx_A], X.iloc[val_idx_A], X.iloc[test_idx_A]
    y_train_A, y_val_A, y_test_A = y.iloc[train_idx_A], y.iloc[val_idx_A], y.iloc[test_idx_A]
    
    print(f"Split sizes -> Train: {len(X_train_A)}, Val: {len(X_val_A)}, Test: {len(X_test_A)}")
    
    # --- Random Forest A Tuning ---
    print("\n>> Tuning Random Forest A...")
    best_rf_auc = -1.0
    best_rf = None
    for n_est in [50, 100, 150]:
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42, class_weight='balanced')
        rf.fit(X_train_A, y_train_A)
        val_probs = rf.predict_proba(X_val_A)[:, 1]
        val_auc = roc_auc_score(y_val_A, val_probs)
        print(f"  RF (n_estimators={n_est}) -> Val AUC: {val_auc:.4f}")
        if val_auc > best_rf_auc:
            best_rf_auc = val_auc
            best_rf = rf
            
    rf_A_probs = best_rf.predict_proba(X_test_A)[:, 1]
    rf_A_preds = best_rf.predict(X_test_A)
    rf_A_auc = roc_auc_score(y_test_A, rf_A_probs)
    rf_A_f1 = f1_score(y_test_A, rf_A_preds, zero_division=0)
    
    # --- SVM A Tuning ---
    print("\n>> Tuning SVM A...")
    scaler_A = StandardScaler()
    X_train_A_scaled = scaler_A.fit_transform(X_train_A)
    X_val_A_scaled = scaler_A.transform(X_val_A)
    X_test_A_scaled = scaler_A.transform(X_test_A)
    
    best_svm_auc = -1.0
    best_svm = None
    for C_val in [0.1, 1.0, 10.0]:
        svm = SVC(C=C_val, kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        svm.fit(X_train_A_scaled, y_train_A)
        val_probs = svm.predict_proba(X_val_A_scaled)[:, 1]
        val_auc = roc_auc_score(y_val_A, val_probs)
        print(f"  SVM (C={C_val}) -> Val AUC: {val_auc:.4f}")
        if val_auc > best_svm_auc:
            best_svm_auc = val_auc
            best_svm = svm
            
    svm_A_probs = best_svm.predict_proba(X_test_A_scaled)[:, 1]
    svm_A_preds = best_svm.predict(X_test_A_scaled)
    svm_A_auc = roc_auc_score(y_test_A, svm_A_probs)
    svm_A_f1 = f1_score(y_test_A, svm_A_preds, zero_division=0)
    
    # --- PointNet A Tuning & Eval ---
    train_ds_A = Subset(full_dataset, train_idx_A)
    val_ds_A = Subset(full_dataset, val_idx_A)
    test_ds_A = Subset(full_dataset, test_idx_A)
    
    pn_A_auc, pn_A_f1, pn_A_cm, pn_A_loss, pn_A_val_loss, pn_A_epoch = train_and_tune_pointnet(
        train_ds_A, val_ds_A, test_ds_A, "Scenario A", epochs=25
    )
    
    # ---------------------------------------------------------
    # Scenario B: Target-level Group Split (6:2:2 by groups)
    # ---------------------------------------------------------
    print("\n=========================================================")
    print("[Scenario B: Target-level 3-Stage Split (Non-overlapping Groups)]")
    print("=========================================================")
    # Train: 2cro, 4icb (~61.7%)
    # Val: 1fc2 (~21.0%)
    # Test: 1hdd-C (~17.3%)
    train_targets = ['2cro', '4icb']
    val_targets = ['1fc2']
    test_targets = ['1hdd-C']
    
    train_idx_B = df[df['target'].isin(train_targets)].index
    val_idx_B = df[df['target'].isin(val_targets)].index
    test_idx_B = df[df['target'].isin(test_targets)].index
    
    X_train_B, X_val_B, X_test_B = X.iloc[train_idx_B], X.iloc[val_idx_B], X.iloc[test_idx_B]
    y_train_B, y_val_B, y_test_B = y.iloc[train_idx_B], y.iloc[val_idx_B], y.iloc[test_idx_B]
    
    print(f"Group Split targets -> Train: {train_targets}, Val: {val_targets}, Test: {test_targets}")
    print(f"Split sizes -> Train: {len(X_train_B)}, Val: {len(X_val_B)}, Test: {len(X_test_B)}")
    
    # --- Random Forest B Tuning ---
    print("\n>> Tuning Random Forest B...")
    best_rf_auc_B = -1.0
    best_rf_B = None
    for n_est in [50, 100, 150]:
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42, class_weight='balanced')
        rf.fit(X_train_B, y_train_B)
        val_probs = rf.predict_proba(X_val_B)[:, 1]
        val_auc = roc_auc_score(y_val_B, val_probs)
        print(f"  RF (n_estimators={n_est}) -> Val AUC: {val_auc:.4f}")
        if val_auc > best_rf_auc_B:
            best_rf_auc_B = val_auc
            best_rf_B = rf
            
    rf_B_probs = best_rf_B.predict_proba(X_test_B)[:, 1]
    rf_B_preds = best_rf_B.predict(X_test_B)
    rf_B_auc = roc_auc_score(y_test_B, rf_B_probs)
    rf_B_f1 = f1_score(y_test_B, rf_B_preds, zero_division=0)
    rf_B_cm = confusion_matrix(y_test_B, rf_B_preds)
    
    # --- SVM B Tuning ---
    print("\n>> Tuning SVM B...")
    scaler_B = StandardScaler()
    X_train_B_scaled = scaler_B.fit_transform(X_train_B)
    X_val_B_scaled = scaler_B.transform(X_val_B)
    X_test_B_scaled = scaler_B.transform(X_test_B)
    
    best_svm_auc_B = -1.0
    best_svm_B = None
    for C_val in [0.1, 1.0, 10.0]:
        svm = SVC(C=C_val, kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        svm.fit(X_train_B_scaled, y_train_B)
        val_probs = svm.predict_proba(X_val_B_scaled)[:, 1]
        val_auc = roc_auc_score(y_val_B, val_probs)
        print(f"  SVM (C={C_val}) -> Val AUC: {val_auc:.4f}")
        if val_auc > best_svm_auc_B:
            best_svm_auc_B = val_auc
            best_svm_B = svm
            
    svm_B_probs = best_svm_B.predict_proba(X_test_B_scaled)[:, 1]
    svm_B_preds = best_svm_B.predict(X_test_B_scaled)
    svm_B_auc = roc_auc_score(y_test_B, svm_B_probs)
    svm_B_f1 = f1_score(y_test_B, svm_B_preds, zero_division=0)
    svm_B_cm = confusion_matrix(y_test_B, svm_B_preds)
    
    # --- PointNet B Tuning & Eval ---
    train_ds_B = Subset(full_dataset, train_idx_B)
    val_ds_B = Subset(full_dataset, val_idx_B)
    test_ds_B = Subset(full_dataset, test_idx_B)
    
    pn_B_auc, pn_B_f1, pn_B_cm, pn_B_loss, pn_B_val_loss, pn_B_epoch = train_and_tune_pointnet(
        train_ds_B, val_ds_B, test_ds_B, "Scenario B", epochs=25
    )
    
    # ---------------------------------------------------------
    # Output Final Redesigned Results Table
    # ---------------------------------------------------------
    print("\n==================================================================")
    print("      FINAL REDESIGNED RESULTS (3-STAGE VALIDATION PIPELINE)")
    print("==================================================================")
    print(f"Scenario A (Random 3-Stage Split):")
    print(f"  - Random Forest | Test ROC-AUC: {rf_A_auc:.4f} | Test F1-Score: {rf_A_f1:.4f}")
    print(f"  - SVM (RBF)     | Test ROC-AUC: {svm_A_auc:.4f} | Test F1-Score: {svm_A_f1:.4f}")
    print(f"  - PointNet (Tuned)| Test ROC-AUC: {pn_A_auc:.4f} | Test F1-Score: {pn_A_f1:.4f} | Best Epoch: {pn_A_epoch} (Train Loss: {pn_A_loss:.4f}, Val Loss: {pn_A_val_loss:.4f})")
    
    print(f"\nScenario B (Target-level 3-Stage Split):")
    print(f"  - Random Forest | Test ROC-AUC: {rf_B_auc:.4f} | Test F1-Score: {rf_B_f1:.4f}")
    print(f"  - SVM (RBF)     | Test ROC-AUC: {svm_B_auc:.4f} | Test F1-Score: {svm_B_f1:.4f}")
    print(f"  - PointNet (Tuned)| Test ROC-AUC: {pn_B_auc:.4f} | Test F1-Score: {pn_B_f1:.4f} | Best Epoch: {pn_B_epoch} (Train Loss: {pn_B_loss:.4f}, Val Loss: {pn_B_val_loss:.4f})")
    
    print("\n[Confusion Matrices for Scenario B (Target Split Test Set)]")
    print(f"Random Forest CM:\n{rf_B_cm}")
    print(f"SVM (RBF) CM:\n{svm_B_cm}")
    print(f"PointNet CM:\n{pn_B_cm}")
    print("==================================================================")

if __name__ == '__main__':
    main()
