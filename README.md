# Replication Package: Predicting Urban Economic Resilience via Graph Convolutional Networks and Origin-Destination Flows

This package reproduces all experiments from the manuscript submitted to *Computers, Environment and Urban Systems* (CEUS).

## Contents

```
replication_package/
├── run_pooled_classification.py   # Main: pooled 93-city classification (Table 3)
├── run_classification.py          # Single-agglomeration benchmarks (Table 2)
├── run_topk_sweep.py              # Top-k sparsification sweep (Figure 5)
├── run_ablation_quick.py          # GCN ablation: OD vs Random vs FC graphs
├── generate_figures.py            # Publication figures (Figures 1-5)
├── data_loader.py                 # Data loading utilities
├── models.py                      # Model implementations (GCN, baseline LOOCV)
├── show_results.py                # Results display
├── requirements.txt
├── data/
│   ├── assembled/
│   │   ├── JJJ/            # Jing-Jin-Ji (N=13)
│   │   ├── YRD/            # Yangtze River Delta (N=41)
│   │   ├── PRD/            # Pearl River Delta (N=21)
│   │   └── CD_CQ/          # Chengdu-Chongqing (N=18)
│   └── gdp/
│       └── prefecture_gdp.xlsx          # GDP statistics (2017-2023)
└── results/                       # Output directory (created on first run)
```

**Data files per agglomeration:**
- `A_city.npy` — N x N OD flow adjacency matrix
- `X_city.npy` — N x D feature matrix (13-14 structural features)
- `y_city.npy` — N x 1 GDP-growth Resilience Index (RI)
- `city_metadata.csv` — City names, coordinates
- `feature_names.txt` — Column names for X_city.npy

## Environment Setup

### 1. Create Conda environment (recommended)

```bash
conda create -n gcn_urban python=3.10
conda activate gcn_urban
```

### 2. Install PyTorch (with CUDA if GPU available)

```bash
# CPU-only
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install PyTorch Geometric

```bash
pip install torch-geometric
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

All commands should be run from the `replication_package/` directory.

### Main result: Pooled 93-city classification (Table 3)

```bash
python run_pooled_classification.py
```

This runs:
- GCN (OD Block-Diagonal) LOOCV on N=93 pooled sample
- GCN (Geographic), GCN (Random graph), GCN (Fully-connected) ablations
- Random Forest, XGBoost, Logistic Regression baselines
- McNemar test and permutation test for statistical significance

Expected runtime: ~20-30 minutes (CPU), ~5-10 minutes (GPU).

### Single-agglomeration benchmarks (Table 2)

```bash
# All 4 agglomerations
python run_classification.py --all

# Single agglomeration
python run_classification.py --data data/assembled/JJJ
```

### Top-k sparsification sweep (Figure 5, Weak Ties discovery)

```bash
# Full sweep (10 thresholds, 3 runs, 200 epochs)
python run_topk_sweep.py

# Quick test (1 run, 150 epochs)
python run_topk_sweep.py --quick
```

### GCN ablation: OD vs Random vs Fully-connected

```bash
python run_ablation_quick.py
```

### Generate publication figures

```bash
# All figures (PDF + PNG, 300 dpi)
python generate_figures.py

# Specific figure only
python generate_figures.py --fig 1
```

Figures are saved to `replication_package/figures/`.

## Feature Description (13-14 dimensions)

| # | Feature | Description |
|---|---------|-------------|
| 1 | in_strength | Weighted in-degree in OD network |
| 2 | out_strength | Weighted out-degree in OD network |
| 3 | betweenness_centrality | Node betweenness (YRD only; dropped in aligned set) |
| 4 | gdp_log_2019 | Log of 2019 GDP (10k RMB) |
| 5 | gdp_pc_log_2019 | Log of 2019 GDP per capita |
| 6 | trend_growth_17_19 | Average GDP growth rate 2017-2019 |
| 7 | primary_pct_2019 | Primary industry share (2019) |
| 8 | secondary_pct_2019 | Secondary industry share (2019) |
| 9 | tertiary_pct_2019 | Tertiary industry share (2019) |
| 10 | industry_entropy | Shannon entropy of 3-sector composition |
| 11 | poi_density | OSM POI density per km² |
| 12 | poi_shannon | OSM POI category Shannon entropy |
| 13 | ntl_log_2019 | Log(1 + nighttime light DN value, 2019) |
| 14 | admin_level | Administrative hierarchy (直辖市=4, 副省级/省会=3, 地级市=2) |

## Target Variable

**GDP-growth Resilience Index (RI_growth):**

```
FR = max(0, (g_2019 - g_2020) / g_2019)     # Fall: growth slowdown
RR = max(0, (g_2023 - g_2020) / (g_2019 - g_2020))  # Recovery
RI = sqrt(FR * RR)
```

where g_t is the GDP growth rate in year t.

For classification, cities are split into Low/Medium/High tertiles **within each agglomeration** (controlling for regional fixed effects).

## Random Seeds

All experiments use fixed random seeds for reproducibility:
- NumPy/Torch: seed = 42
- GCN LOOCV: 10 runs (seeds 42-51), majority voting
- Random graph generation: numpy.random.seed(42)
- Permutation test: 10,000 permutations, seed=42

## Key Results (Quick Reference)

| Model | Accuracy | F1-Macro | Notes |
|-------|----------|----------|-------|
| GCN (OD, topk=1.0) | 0.4194 | 0.4192 | Full OD graph, best |
| GCN (OD, topk=0.3) | 0.3441 | 0.3442 | Economic skeleton |
| GCN (Random graph) | 0.3763 | 0.3748 | ER null model, matched density |
| Random Forest | 0.3333 | 0.3810 | Within-group baseline = random |
| XGBoost | 0.3548 | 0.3548 | |
| Random baseline | 0.3333 | — | 3-class chance |

**Key finding:** OD topology is most informative at high density (topk >= 0.4), supporting the "weak ties" hypothesis — medium and weak OD connections collectively encode resilience-relevant information beyond what strong economic corridors alone provide.

## Citation

If you use this code or data, please cite:

> [Authors]. (2026). Predicting Urban Economic Resilience via Graph Convolutional Networks and Origin-Destination Flows. *Computers, Environment and Urban Systems*.

## License

Data: CC-BY 4.0
Code: MIT
