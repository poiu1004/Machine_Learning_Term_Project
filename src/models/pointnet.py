import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "dd", "multiple", "fisa"))

class ProteinPointCloudDataset(Dataset):
    """
    PyTorch Dataset for converting PDB files into 3D Point Clouds.
    """
    def __init__(self, features_csv, target_list=None, num_points=128):
        """
        Args:
            features_csv (str): Path to features.csv containing labels.
            target_list (list): List of targets to include (for train/test split).
            num_points (int): Fixed number of points for PointNet input.
        """
        self.df = pd.read_csv(features_csv)
        self.df = self.df[self.df['label'] != -1] # Remove ambiguous labels
        
        if target_list is not None:
            self.df = self.df[self.df['target'].isin(target_list)]
            
        self.num_points = num_points
        self.parser = PDBParser()
        self.data = self.df.to_dict('records')
        self.coords_cache = {}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Check cache first for the fully processed (points, label) tuple
        if idx in self.coords_cache:
            return self.coords_cache[idx]
            
        row = self.data[idx]
        target = row['target']
        filename = row['filename']
        label = row['label']
        
        pdb_path = os.path.join(DATA_DIR, target, filename)
        
        # 1. Parse PDB and extract C-alpha coordinates
        try:
            structure = self.parser.get_structure("struct", pdb_path)
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords.append(residue['CA'].get_coord())
            coords = np.array(coords)
            
            if len(coords) == 0:
                coords = np.zeros((self.num_points, 3)) # Fallback
        except Exception:
            coords = np.zeros((self.num_points, 3)) # Fallback
            
        # 2. Resample to fixed num_points
        n_res = coords.shape[0]
        if n_res >= self.num_points:
            # Randomly downsample
            indices = np.random.choice(n_res, self.num_points, replace=False)
        else:
            # Randomly oversample (repeat points)
            indices = np.random.choice(n_res, self.num_points, replace=True)
            
        points = coords[indices, :]
        
        # 3. Normalize Point Cloud (Crucial for PointNet)
        # Center to origin
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if m > 0:
            points = points / m
            
        # PointNet expects input shape: (Channels, Num_Points) -> (3, N)
        points = torch.tensor(points, dtype=torch.float32).transpose(0, 1)
        label = torch.tensor(label, dtype=torch.long)
        
        result = (points, label)
        self.coords_cache[idx] = result
        return result

# ----------------- PointNet Architecture -----------------

class TNet(nn.Module):
    """ T-Net (Spatial Transformer Network) to align point clouds """
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(128, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(1024, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(512, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(256, track_running_stats=False)

    def forward(self, x):
        # x: (B, k, N)
        B = x.size(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Add identity matrix
        iden = torch.eye(self.k, requires_grad=True).repeat(B, 1, 1).to(x.device)
        x = x.view(-1, self.k, self.k) + iden
        return x

class PointNetCls(nn.Module):
    """ PointNet Classifier for Structural Defect Detection """
    def __init__(self, num_classes=2):
        super(PointNetCls, self).__init__()
        self.tnet = TNet(k=3)
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64, track_running_stats=False)
        
        self.feat_tnet = TNet(k=64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128, track_running_stats=False)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024, track_running_stats=False)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512, track_running_stats=False)
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256, track_running_stats=False)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, 3, N)
        B, _, N = x.size()
        
        # Input transform
        trans = self.tnet(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # MLP
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Feature transform (optional in basic PointNet, but good for invariance)
        trans_feat = self.feat_tnet(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        
        # MLP
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Symmetric Function (Global Max Pooling)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # MLP for Classification
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x, trans, trans_feat

if __name__ == "__main__":
    # Simple unit test
    print("Testing PointNet Architecture with dummy data...")
    dummy_points = torch.rand(16, 3, 128) # B=16, Channels=3, N=128
    model = PointNetCls(num_classes=2)
    out, t1, t2 = model(dummy_points)
    print(f"Input shape: {dummy_points.shape}")
    print(f"Output shape (Logits): {out.shape}")
    print("Architecture is ready!")
