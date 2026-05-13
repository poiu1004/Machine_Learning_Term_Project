# Detection of Structural Defects in Designed Protein Structures

This repository contains the source code for detecting structural defects in generated protein structures using geometric machine learning techniques.

## Project Structure
- `data/`: Contains raw PDB files and processed features (`data/raw/` is ignored by Git to prevent large uploads).
- `src/preprocessing/`: Scripts to parse PDB files and extract handcrafted geometric features.
- `src/models/`: Scripts to train baseline (RF, SVM) and geometric deep learning models (PointNet).
- `figures/`: Output plots (e.g., feature importance, ROC curves).
- `requirements.txt`: Python package dependencies.

## Setup
Install the necessary packages:
```bash
pip install -r requirements.txt
```

## How to Run
1. Place your target PDB decoy sets inside `data/raw/`.
2. Run the preprocessing script to extract geometric features:
   ```bash
   python src/preprocessing/extract_features.py
   ```
3. Run the baseline models to establish performance:
   ```bash
   python src/models/baseline.py
   ```
