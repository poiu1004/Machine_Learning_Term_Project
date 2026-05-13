import os
import glob
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import warnings
from Bio import BiopythonWarning
from scipy.spatial.distance import pdist
from tqdm import tqdm

warnings.simplefilter('ignore', BiopythonWarning)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "dd", "multiple", "fisa"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))

def parse_rmsds(rmsd_file):
    """Parses the rmsds file and returns a dictionary mapping decoy name to RMSD."""
    rmsd_dict = {}
    with open(rmsd_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6 and parts[0] == "cRMSD":
                # cRMSD between 1fc2.pdb and axproa00-min.pdb is   3.946
                decoy_name = parts[4]
                try:
                    rmsd_val = float(parts[6])
                    rmsd_dict[decoy_name] = rmsd_val
                except ValueError:
                    pass
    return rmsd_dict

def extract_features_from_pdb(pdb_path):
    """Extracts handcrafted geometric features from a PDB file."""
    parser = PDBParser()
    try:
        structure = parser.get_structure("struct", pdb_path)
    except Exception:
        return None
        
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Use C-alpha for geometry if available
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
                else:
                    atoms = list(residue.get_atoms())
                    if atoms:
                        coords.append(atoms[0].get_coord())
                        
    if len(coords) < 2:
        return None
        
    coords = np.array(coords)
    n_residues = len(coords)
    
    # 1. Radius of Gyration
    center = np.mean(coords, axis=0)
    rg = np.sqrt(np.sum((coords - center)**2) / n_residues)
    
    # 2. Pairwise distances
    dists = pdist(coords)
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    
    # 3. Clash count (C-alpha distances < 3.0 Å are unphysical)
    clash_count = np.sum(dists < 3.0)
    
    # 4. Residue density (residues per cubic Angstrom of bounding box)
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    volume = np.prod(max_coords - min_coords)
    density = n_residues / volume if volume > 0 else 0
    
    return {
        'rg': rg,
        'mean_dist': mean_dist,
        'std_dist': std_dist,
        'clash_count': clash_count,
        'density': density,
        'n_residues': n_residues
    }

def process_dataset():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    targets = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d not in ['doc']]
    
    all_data = []
    
    for target in targets:
        print(f"Processing target: {target}")
        target_dir = os.path.join(DATA_DIR, target)
        rmsd_file = os.path.join(target_dir, "rmsds")
        
        if not os.path.exists(rmsd_file):
            print(f"  Missing rmsds file for {target}, skipping.")
            continue
            
        rmsd_dict = parse_rmsds(rmsd_file)
        
        # Process each PDB
        pdb_files = glob.glob(os.path.join(target_dir, "*.pdb"))
        for pdb_path in tqdm(pdb_files, desc=f"Extracting {target}"):
            filename = os.path.basename(pdb_path)
            if filename not in rmsd_dict:
                continue
                
            rmsd = rmsd_dict[filename]
            
            # Simple label definition based on user feedback:
            # Stable: RMSD < 4.0
            # Defective: RMSD > 8.0
            # Note: We can add complex labeling later if we parse Rosetta energy. 
            # For now, we implement the basic RMSD classification logic.
            if rmsd < 4.0:
                label = 0 # Stable
            elif rmsd > 8.0:
                label = 1 # Defective
            else:
                label = -1 # Ambiguous (will be filtered out)
                
            features = extract_features_from_pdb(pdb_path)
            if features:
                row = {
                    'target': target,
                    'filename': filename,
                    'rmsd': rmsd,
                    'label': label
                }
                row.update(features)
                all_data.append(row)
                
    df = pd.DataFrame(all_data)
    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    df.to_csv(out_path, index=False)
    print(f"Extraction complete! Saved {len(df)} records to {out_path}")

if __name__ == "__main__":
    process_dataset()
