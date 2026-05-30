Molecular Property Prediction using Graph Transformer

Predicting HOMO-LUMO gaps of drug-like molecules from the QM9 dataset using a Graph Transformer neural network built with PyTorch Geometric.


Results at a Glance
MetricValueTest MAE0.2170 eVTest RMSE0.3048 eVTest R²0.9361Naive baseline (predict mean)1.2916 eVImprovement over baseline83.2%
Trained on CPU only in ~90 minutes. Competitive with published GNN baselines on QM9.

Why This Problem Matters
The HOMO-LUMO gap — the energy difference between a molecule's highest occupied and lowest unoccupied molecular orbital — is one of the most important quantum-chemical properties in drug discovery and materials science. It governs:

Molecular reactivity — small gaps indicate high reactivity
Charge transfer — critical for organic semiconductors and photovoltaics
Drug-target binding — influences electronic complementarity between ligand and receptor

Traditionally, computing this property requires Density Functional Theory (DFT) calculations that take hours per molecule. A well-trained GNN can approximate it in milliseconds, enabling high-throughput virtual screening across millions of candidates.

Architecture
Raw molecule (3D geometry)
        │
        ▼
 Node features (11-dim)          Edge features (5-dim)
 ┌────────────────────┐          ┌──────────────────┐
 │ Element (one-hot)  │          │ Bond type (×4)   │
 │ Hybridization (sp  │          │ Bond length (Å)  │
 │   sp2, sp3)        │          └──────────────────┘
 │ Aromaticity        │
 │ # Hydrogens        │
 └────────────────────┘
        │                               │
        ▼                               ▼
   Linear projection (→ 128-dim)   Linear projection (→ 128-dim)
        │
        ▼
 ┌─────────────────────────────────────┐
 │  TransformerConv Layer 1            │
 │  (4 attention heads, edge-conditioned│
 │   attention + residual connection)  │
 └─────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────┐
 │  TransformerConv Layer 2            │
 └─────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────┐
 │  TransformerConv Layer 3            │
 └─────────────────────────────────────┘
        │
        ▼
 Global Mean Pooling  (atoms → molecule)
        │
        ▼
 MLP Readout:  128 → 64 → 32 → 1
        │
        ▼
 Predicted HOMO-LUMO gap (eV)
Key design choices:

TransformerConv (Shi et al., 2020): edge features are injected directly into the attention score computation, so bond type and bond length influence which atoms the model attends to — not just node features.
Residual connections at every layer prevent gradient vanishing on deeper graphs.
Bond length as edge feature: computed from QM9's 3D atomic coordinates, gives the model geometric information without using a full 3D architecture.
Target normalization: HOMO-LUMO gap is z-score normalized during training and denormalized for reporting.

Total parameters: 311,425

Dataset
QM9 (Ramakrishnan et al., 2014) contains ~134,000 small organic molecules with up to 9 heavy atoms (C, H, O, N, F) and 19 quantum-chemical properties computed via DFT.
This project uses the first 20,000 molecules for computational efficiency. QM9 is downloaded automatically by PyTorch Geometric on first run (~575 MB on disk after processing).
SplitMoleculesTrain16,000Validation2,000Test2,000

Explainability — Attention Heatmaps
The model exposes per-atom attention weights from each TransformerConv layer. These are aggregated across layers and heads to produce a normalized importance score per atom, visualized as a colour heatmap (yellow = low attention, red = high attention).
Show Image
Five molecules spanning the low-to-high gap range (5.8 eV → 8.9 eV) are shown. Prediction errors range from 0.007 eV to 0.211 eV.
Honest note on the attention visualization: after aggregating across 3 layers and 4 heads, scores saturate toward high values for most atoms. This is a known limitation of layer-aggregated attention in deep GNNs — not a bug in the implementation. Single-layer attention maps would show sharper contrast. Future work could use GNNExplainer or IntegratedGradients (Captum) for sharper attribution.

Training Dynamics
Show Image

Loss converges smoothly; the LR scheduler (ReduceLROnPlateau, patience=5) reduces the learning rate 4 times across 100 epochs.
Best validation MAE of 0.1714 eV achieved at epoch 77. Early stopping (patience=15) did not trigger as the model kept improving throughout.
Mild train/val gap indicates slight overfitting, acceptable at this scale.


Test Set Evaluation
Show Image

Parity plot shows tight clustering around the diagonal across the full 4–11 eV range.
Residuals are approximately Gaussian centered at 0 with a slight right tail — the model mildly underestimates high-gap molecules (>9 eV), which are underrepresented in the first 20k QM9 molecules.


Installation
bash# Create environment
conda create -n molprop python=3.9 -y
conda activate molprop

# PyTorch (CPU)
conda install pytorch==2.0.0 cpuonly -c pytorch -c conda-forge -y

# PyTorch Geometric
conda install pyg -c pyg -c conda-forge -y

# Supporting libraries
conda install jupyterlab matplotlib scikit-learn tqdm rdkit -c conda-forge -y

# Pin NumPy (required — PyG C++ extensions built against NumPy 1.x)
pip install numpy==1.26.4

Usage
bashgit clone https://github.com/Amir-Shokrzadeh/molecular-graph-transformer.git
cd molecular-graph-transformer
jupyter lab
# Open: molecular_property_prediction.ipynb
# Run all cells top to bottom
QM9 downloads automatically on first run (~575 MB on disk). Training takes ~90 minutes on CPU.

Project Structure
molecular-graph-transformer/
│
├── molecular_property_prediction.ipynb   # Main notebook (all code)
├── best_model.pt                         # Saved model weights
│
├── figures/
│   ├── eda_plots.png                     # Dataset distributions
│   ├── training_curves.png               # Loss + MAE + LR curves
│   ├── test_evaluation.png               # Parity plot + residuals
│   └── attention_heatmaps.png            # Atom-level attention
│
├── data/                                 # QM9 data (auto-downloaded, git-ignored)
├── .gitignore
└── README.md

Limitations & Future Work
LimitationPotential fix20k molecule subsetTrain on full 134k QM9CPU only, ~90 min trainingGPU training would allow larger hidden dim / more layersSaturated attention heatmapsReplace with GNNExplainer or Captum IntegratedGradientsSingle target predictionMulti-task learning across all 19 QM9 propertiesNo 3D geometry beyond bond lengthFull 3D architecture (DimeNet++, SphereNet)

References

Shi, Y. et al. (2020). Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification. arXiv:2009.03509 — TransformerConv
Ramakrishnan, R. et al. (2014). Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data. — QM9 dataset
Fey, M. & Lenssen, J. (2019). Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop. — PyG


Environment
PackageVersionPython3.9.25PyTorch2.0.0PyTorch Geometric2.5.2NumPy1.26.4RDKit2025.03.5Scikit-learn1.6.1Matplotlib3.9.4

Built as part of a computational drug discovery portfolio. All training done on CPU — no GPU required.
