"""
Pooled 93-city classification benchmark with Block-Diagonal super-adjacency.

Constructs a block-diagonal adjacency matrix from 4 agglomeration OD matrices,
runs LOOCV GCN + traditional classifiers on the full N=93 sample, and compares
against per-agglomeration baselines.

This directly addresses the JGS editor's instruction to "pool several Chinese
urban agglomerations" and eliminates small-N concerns.

Usage:
    python run_pooled_classification.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy import stats
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_adjacency, load_features_from_npy, load_target_from_npy

# Preferred feature order for aligned feature set
FEATURE_ORDER = {
    "in_strength": 0, "out_strength": 1, "betweenness_centrality": 2,
    "gdp_log_2019": 3, "gdp_pc_log_2019": 4, "trend_growth_17_19": 5,
    "primary_pct_2019": 6, "secondary_pct_2019": 7, "tertiary_pct_2019": 8,
    "industry_entropy": 9, "poi_density": 10, "poi_shannon": 11,
    "ntl_log_2019": 12, "admin_level": 13,
}

# ═══════════════════════════════════════════════════════════════════
# Block-Diagonal Adjacency Builder
# ═══════════════════════════════════════════════════════════════════

def build_block_diagonal_adjacency(data_dirs, norm="minmax", topk=0.3):
    """
    Build block-diagonal super-adjacency from multiple agglomerations.

    Each agglomeration's OD matrix is normalized, sparsified to effective
    economic skeleton (top-k edges retained), and placed as a diagonal block.
    Inter-agglomeration edges are set to zero (pure friction narrative).

    Returns:
        A_super: (N_total, N_total) ndarray
        block_boundaries: list of (start_idx, end_idx, agg_name) tuples
    """
    blocks = []
    boundaries = []
    offset = 0

    for data_dir in data_dirs:
        A_path = data_dir / "A_city.npy"
        if not A_path.exists():
            print(f"  WARNING: {A_path} not found, skipping")
            continue

        A_block = load_adjacency(A_path, normalize=norm, topk=topk)
        N_block = A_block.shape[0]
        blocks.append(A_block)
        agg_name = data_dir.name
        boundaries.append((offset, offset + N_block, agg_name))
        offset += N_block

    N_total = offset
    A_super = np.zeros((N_total, N_total))

    for (start, end, _), block in zip(boundaries, blocks):
        A_super[start:end, start:end] = block

    return A_super, boundaries


def build_block_diagonal_geo(data_dirs, k=3):
    """
    Build geographic adjacency for pooled sample.
    Uses city coordinates from metadata to construct k-NN graph across all cities,
    then applies block-diagonal mask (no inter-agglomeration edges).

    Returns: A_geo_super (N_total, N_total)
    """
    from sklearn.neighbors import NearestNeighbors

    all_coords = []
    blocks = []
    boundaries = []
    offset = 0

    for data_dir in data_dirs:
        meta_path = data_dir / "city_metadata.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            if "lon" in meta.columns and "lat" in meta.columns:
                coords = meta[["lon", "lat"]].values
            else:
                coords = np.zeros((1, 2))
        else:
            # Fallback
            n_cities = np.load(data_dir / "X_city.npy").shape[0]
            coords = np.zeros((n_cities, 2))

        all_coords.append(coords)
        N_block = coords.shape[0]
        blocks.append(N_block)
        boundaries.append((offset, offset + N_block, data_dir.name))
        offset += N_block

    N_total = offset
    all_coords = np.vstack(all_coords)

    # k-NN across all cities
    A_geo = np.zeros((N_total, N_total))
    knn = NearestNeighbors(n_neighbors=min(k + 1, N_total), metric="euclidean")
    knn.fit(all_coords)
    dist, idx = knn.kneighbors(all_coords)

    for i in range(N_total):
        for jj in range(1, min(k + 1, N_total)):
            j = idx[i, jj]
            w = 1.0 / (1.0 + dist[i, jj])
            A_geo[i, j] = w
            A_geo[j, i] = w

    # Apply block-diagonal mask: zero out inter-agglomeration edges
    mask = np.zeros((N_total, N_total))
    for start, end, _ in boundaries:
        mask[start:end, start:end] = 1.0

    A_geo_masked = A_geo * mask

    # Add self-loops
    A_geo_masked = A_geo_masked + np.eye(N_total)

    return A_geo_masked, boundaries


def build_random_graph_adjacency(data_dirs):
    """
    Build block-diagonal random-graph adjacency as GCN null model.

    Within each block, generates a symmetric Erdős–Rényi random graph
    matching the edge density of the real OD graph. This tests whether
    GCN's performance relies on the specific OD topology, or whether
    any sparse graph with the same density would yield similar results.

    Returns:
        A_rand: (N_total, N_total) ndarray with self-loops
    """
    offset = 0
    blocks = []
    boundaries = []

    for data_dir in data_dirs:
        A_path = data_dir / "A_city.npy"
        if not A_path.exists():
            continue
        # Must match the sparse "economic skeleton" density (topk=0.3)
        A_real = load_adjacency(A_path, normalize="minmax", topk=0.3)
        N = A_real.shape[0]

        # Compute edge density from sparse economic skeleton (excluding self-loops)
        n_edges_real = int((A_real > 0).sum()) - N  # exclude diagonal
        n_possible = N * (N - 1)
        if n_possible == 0:
            A_rand = np.eye(N)
        else:
            p_edge = n_edges_real / n_possible
            # Symmetric random graph
            upper_mask = np.triu(np.random.rand(N, N) < p_edge, k=1)
            lower = upper_mask.T
            A_rand = (upper_mask | lower).astype(np.float64)
            np.fill_diagonal(A_rand, 0.0)
            # Use uniform edge weights
            A_rand[A_rand > 0] = 1.0

        A_rand = A_rand + np.eye(N)
        blocks.append(A_rand)
        boundaries.append((offset, offset + N, data_dir.name))
        offset += N

    N_total = offset
    A_super = np.zeros((N_total, N_total))
    for (start, end, _), block in zip(boundaries, blocks):
        A_super[start:end, start:end] = block

    return A_super, boundaries


def build_fully_connected_adjacency(data_dirs):
    """
    Build block-diagonal fully-connected adjacency as GCN null model.

    Within each block, every node is connected to every other node with
    uniform edge weight = 1. This erases all topological information
    while preserving the block-diagonal structure (no inter-block edges).
    GCN message passing then reduces to simple intra-block averaging.

    If GCN (fully-connected) performs as well as GCN (OD), then OD
    topology contributes nothing beyond knowing which agglomeration
    a city belongs to.

    Returns:
        A_fc: (N_total, N_total) ndarray with self-loops
    """
    offset = 0
    blocks = []
    boundaries = []

    for data_dir in data_dirs:
        A_path = data_dir / "A_city.npy"
        if not A_path.exists():
            continue
        N = np.load(A_path).shape[0]
        A_fc = np.ones((N, N), dtype=np.float64)
        np.fill_diagonal(A_fc, 0.0)
        A_fc = A_fc + np.eye(N)
        blocks.append(A_fc)
        boundaries.append((offset, offset + N, data_dir.name))
        offset += N

    N_total = offset
    A_super = np.zeros((N_total, N_total))
    for (start, end, _), block in zip(boundaries, blocks):
        A_super[start:end, start:end] = block

    return A_super, boundaries


# ═══════════════════════════════════════════════════════════════════
# GCN Classifier (same as run_classification.py)
# ═══════════════════════════════════════════════════════════════════

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, n_classes=3, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.classifier(x)
        return x


def adjacency_to_edge_index(A, threshold=0.0):
    N = A.shape[0]
    sources, targets, weights = [], [], []
    for i in range(N):
        for j in range(N):
            if i != j and A[i, j] > threshold:
                sources.append(i)
                targets.append(j)
                weights.append(A[i, j])
    if len(sources) == 0:
        for i in range(N):
            for j in range(N):
                if i != j:
                    sources.append(i)
                    targets.append(j)
                    weights.append(1.0)
    return (torch.tensor([sources, targets], dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32))


def _rebuild_edge_index_drop_node(A, drop_idx):
    N = A.shape[0]
    keep = [i for i in range(N) if i != drop_idx]
    A_sub = A[np.ix_(keep, keep)]
    return adjacency_to_edge_index(A_sub)[0], adjacency_to_edge_index(A_sub)[1], keep


def loocv_gcn_classifier_pooled(A, X, y_labels, n_classes=3, hidden_dim=32,
                                epochs=300, lr=0.01, n_runs=10, seed=42):
    """LOOCV GCN classification for pooled N=93 sample."""
    N = A.shape[0]
    D = X.shape[1]
    all_preds = np.zeros((n_runs, N), dtype=np.int64)

    for run in range(n_runs):
        np.random.seed(seed + run)
        torch.manual_seed(seed + run)

        for i in range(N):
            edge_index, edge_weight, keep = _rebuild_edge_index_drop_node(A, i)
            X_train = torch.tensor(X[keep], dtype=torch.float32)
            y_train = torch.tensor(y_labels[keep], dtype=torch.long)

            # Standardize within training fold
            train_mean = X_train.mean(dim=0, keepdim=True)
            train_std = X_train.std(dim=0, keepdim=True)
            train_std[train_std == 0] = 1.0
            X_train = (X_train - train_mean) / train_std

            model = GCNClassifier(D, hidden_dim=hidden_dim, n_classes=n_classes)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            for _ in range(epochs):
                optimizer.zero_grad()
                logits = model(X_train, edge_index, edge_weight)
                loss = criterion(logits, y_train)
                loss.backward()
                optimizer.step()

            # Predict held-out city
            X_full = torch.tensor(X, dtype=torch.float32)
            X_full = (X_full - train_mean) / train_std
            edge_idx_full, edge_w_full = adjacency_to_edge_index(A)

            model.eval()
            with torch.no_grad():
                logits = model(X_full, edge_idx_full, edge_w_full)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            all_preds[run, i] = np.argmax(probs[i])

        if (run + 1) % 5 == 0:
            print(f"    Run {run+1}/{n_runs} complete")

    final_preds = stats.mode(all_preds, axis=0)[0].flatten()
    return final_preds


# ═══════════════════════════════════════════════════════════════════
# Classical classifiers
# ═══════════════════════════════════════════════════════════════════

def loocv_sklearn_classifier_pooled(model_class, X, y_labels, **kwargs):
    N = X.shape[0]
    preds = np.zeros(N, dtype=np.int64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for i in range(N):
        X_train = np.delete(X_scaled, i, axis=0)
        y_train = np.delete(y_labels, i)
        X_test = X_scaled[i:i+1]

        model = model_class(**kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        preds[i] = model.predict(X_test)[0]

    return preds


# ═══════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════

def mcnemar_test(y_true, preds_a, preds_b):
    """
    McNemar's test for paired nominal data.

    Compares whether two classifiers (A and B) have the same error rate
    under LOOCV. The null hypothesis is that the two models have equal
    accuracy — any observed difference is due to chance.

    Builds a 2×2 contingency table:
                 B correct   B wrong
    A correct       a          b
    A wrong         c          d

    McNemar's χ² = (|b - c| - 1)² / (b + c)  (with continuity correction)

    Returns:
        stat: chi-squared statistic
        p_value: two-tailed p-value
        table: dict with a, b, c, d counts
        delta_acc: accuracy difference (acc_a - acc_b)
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    a = int(np.sum(correct_a & correct_b))
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    d = int(np.sum(~correct_a & ~correct_b))

    # Continuity-corrected McNemar
    if b + c == 0:
        stat = 0.0
        p_value = 1.0
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2
        p_value = 1.0 - chi2.cdf(stat, df=1)

    acc_a = np.mean(correct_a)
    acc_b = np.mean(correct_b)

    return {
        'statistic': stat,
        'p_value': p_value,
        'table': {'a': a, 'b': b, 'c': c, 'd': d},
        'delta_acc': acc_a - acc_b,
        'acc_a': acc_a,
        'acc_b': acc_b,
    }


def permutation_test_graph(y_true, preds_od, preds_null, n_perm=10000, seed=42):
    """
    Permutation test: is ΔAcc = Acc(OD) − Acc(null graph) significantly > 0?

    Randomly flips the winner for each sample (with probability 0.5),
    building a null distribution of accuracy differences. Tests whether
    the observed OD advantage could arise from random chance.

    Returns:
        observed: observed accuracy difference
        p_value: one-tailed p-value (H₁: OD > null)
    """
    rng = np.random.RandomState(seed)
    correct_od = (preds_od == y_true).astype(np.int32)
    correct_null = (preds_null == y_true).astype(np.int32)
    observed = np.mean(correct_od) - np.mean(correct_null)

    diffs = correct_od - correct_null
    null_dist = np.zeros(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        null_dist[i] = np.mean(signs * diffs)

    p_value = np.mean(null_dist >= observed)

    return {
        'observed_delta': observed,
        'p_value': p_value,
        'null_mean': np.mean(null_dist),
        'null_std': np.std(null_dist),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    base_dir = Path(__file__).parent
    assembled = base_dir / "data" / "assembled"
    agg_dirs = sorted([d for d in assembled.iterdir() if d.is_dir()])

    print("=" * 70)
    print("POOLED 93-CITY CLASSIFICATION BENCHMARK")
    print("Block-Diagonal Super-Adjacency Matrix")
    print("=" * 70)

    # ── Load all data ──────────────────────────────────────────
    X_blocks, y_blocks, city_blocks, agg_labels = [], [], [], []
    boundaries = []
    offset = 0

    for data_dir in agg_dirs:
        name = data_dir.name
        X, feat_names = load_features_from_npy(data_dir, cities=None,
                                                drop_constant=True, add_admin_level=True)
        y = load_target_from_npy(data_dir).flatten()
        N = X.shape[0]

        # Load city names if available
        meta_path = data_dir / "city_metadata.csv"
        cities_in_agg = [f"{name}_{i}" for i in range(N)]
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            if "city" in meta.columns:
                cities_in_agg = list(meta["city"][:N])

        X_blocks.append(X)
        y_blocks.append(y)
        city_blocks.append(cities_in_agg)
        agg_labels.extend([name] * N)
        boundaries.append((offset, offset + N, name))
        offset += N

        print(f"  {name}: N={N}, D={X.shape[1]}, features={feat_names}")

    # Align feature dimensions: find common features across all agglomerations
    # YRD has 14D (with betweenness_centrality), others have 13D
    common_features = None
    all_feature_sets = []
    for data_dir in agg_dirs:
        _, feat_names = load_features_from_npy(data_dir, cities=None,
                                                drop_constant=True, add_admin_level=True)
        all_feature_sets.append(set(feat_names))
    common_features = all_feature_sets[0]
    for fs in all_feature_sets[1:]:
        common_features = common_features & fs
    common_features = sorted(common_features, key=lambda f: FEATURE_ORDER.get(f, 99))

    # Slice each block to common features
    for i, data_dir in enumerate(agg_dirs):
        _, feat_names = load_features_from_npy(data_dir, cities=None,
                                                drop_constant=True, add_admin_level=True)
        col_idx = [feat_names.index(f) for f in common_features if f in feat_names]
        X_blocks[i] = X_blocks[i][:, col_idx]

    print(f"\n  Aligned to {len(common_features)} common features: {common_features}")

    X_pooled = np.vstack(X_blocks)
    y_pooled = np.concatenate(y_blocks)
    cities_pooled = [c for block in city_blocks for c in block]
    N_total = X_pooled.shape[0]
    D = len(common_features)

    # Global standardization
    X_scaled = StandardScaler().fit_transform(X_pooled)

    # ── Within-group tertile labels (control for regional fixed effects) ──
    # Each agglomeration is split independently into Low/Medium/High
    # based on its OWN RI distribution, preventing the model from
    # cheating by inferring class from agglomeration membership.
    y_labels = np.full(N_total, -1, dtype=np.int64)

    for start, end, name in boundaries:
        y_agg = y_pooled[start:end]
        agg_tertiles = np.percentile(y_agg, [33.33, 66.67])
        y_agg_labels = np.zeros(end - start, dtype=np.int64)
        y_agg_labels[y_agg >= agg_tertiles[1]] = 2      # High
        y_agg_labels[(y_agg >= agg_tertiles[0]) & (y_agg < agg_tertiles[1])] = 1  # Medium
        # Low stays 0
        y_labels[start:end] = y_agg_labels

    # Also compute global tertiles for diagnostic comparison
    global_tertiles = np.percentile(y_pooled, [33.33, 66.67])
    y_labels_global = np.zeros(N_total, dtype=np.int64)
    y_labels_global[y_pooled >= global_tertiles[1]] = 2
    y_labels_global[(y_pooled >= global_tertiles[0]) & (y_pooled < global_tertiles[1])] = 1

    class_names = ["Low", "Medium", "High"]
    n_per_class = {c: int(np.sum(y_labels == i)) for i, c in enumerate(class_names)}

    print(f"\n  Total: N={N_total}, D={D}")
    print(f"  Global RI range: [{y_pooled.min():.3f}, {y_pooled.max():.3f}]")
    print(f"  Labeling: WITHIN-GROUP tertiles (controls agglomeration FE)")
    print(f"  Class distribution: {n_per_class}")
    print(f"  Random baseline (3-class): 0.3333")

    # Diagnostic: compare global vs within-group labeling
    n_mismatch = int(np.sum(y_labels != y_labels_global))
    print(f"  Within-group vs Global labeling mismatch: {n_mismatch}/{N_total} "
          f"({n_mismatch/N_total*100:.1f}%)")
    if n_mismatch / N_total > 0.15:
        print(f"  *** WARNING: >15% mismatch — regional FE are substantial ***")
        print(f"  *** Within-group labeling is the correct specification.   ***")

    # Per-agglomeration class distribution (within-group)
    print(f"\n  Class distribution per agglomeration (within-group tertiles):")
    for start, end, name in boundaries:
        y_agg = y_labels[start:end]
        counts = {c: int(np.sum(y_agg == i)) for i, c in enumerate(class_names)}
        print(f"    {name}: {counts}")

    # ── Build adjacency matrices ───────────────────────────────
    A_od, _ = build_block_diagonal_adjacency(agg_dirs)
    A_geo, _ = build_block_diagonal_geo(agg_dirs, k=3)

    # Ablation: null-model graphs for GCN baselines
    np.random.seed(42)  # fixed seed for reproducibility
    A_rand, _ = build_random_graph_adjacency(agg_dirs)
    A_fc, _ = build_fully_connected_adjacency(agg_dirs)

    n_od_edges = int((A_od > 0).sum() - N_total)
    n_geo_edges = int((A_geo > 0).sum() - N_total)
    n_rand_edges = int((A_rand > 0).sum() - N_total)
    n_fc_edges = int((A_fc > 0).sum() - N_total)
    print(f"\n  OD graph:           {n_od_edges} edges (block-diagonal)")
    print(f"  Geo graph:          {n_geo_edges} edges (k=3, within-block)")
    print(f"  Random graph:       {n_rand_edges} edges (same density, ER model)")
    print(f"  Fully-connected:    {n_fc_edges} edges (uniform intra-block)")

    # ── Run models ─────────────────────────────────────────────
    results = []

    # GCN (OD graph = block-diagonal)
    print(f"\n  Running GCN (OD Block-Diagonal) LOOCV on N={N_total}...")
    preds_gcn_od = loocv_gcn_classifier_pooled(
        A_od, X_scaled, y_labels, n_runs=10, seed=42
    )
    acc = accuracy_score(y_labels, preds_gcn_od)
    f1m = f1_score(y_labels, preds_gcn_od, average='macro', zero_division=0)
    results.append({"Model": "GCN (OD Block-Diagonal)", "Accuracy": acc, "F1_Macro": f1m,
                    "preds": preds_gcn_od})
    print(f"    Acc={acc:.4f}, F1_macro={f1m:.4f}")

    # GCN (Geographic block-diagonal)
    print(f"\n  Running GCN (Geo Block-Diagonal) LOOCV on N={N_total}...")
    preds_gcn_geo = loocv_gcn_classifier_pooled(
        A_geo, X_scaled, y_labels, n_runs=10, seed=42
    )
    acc = accuracy_score(y_labels, preds_gcn_geo)
    f1m = f1_score(y_labels, preds_gcn_geo, average='macro', zero_division=0)
    results.append({"Model": "GCN (Geo Block-Diagonal)", "Accuracy": acc, "F1_Macro": f1m,
                    "preds": preds_gcn_geo})
    print(f"    Acc={acc:.4f}, F1_macro={f1m:.4f}")

    # GCN (Random graph = ER null model) — ablation baseline
    print(f"\n  Running GCN (Random graph) LOOCV on N={N_total}...")
    preds_gcn_rand = loocv_gcn_classifier_pooled(
        A_rand, X_scaled, y_labels, n_runs=10, seed=42
    )
    acc = accuracy_score(y_labels, preds_gcn_rand)
    f1m = f1_score(y_labels, preds_gcn_rand, average='macro', zero_division=0)
    results.append({"Model": "GCN (Random graph)", "Accuracy": acc, "F1_Macro": f1m,
                    "preds": preds_gcn_rand})
    print(f"    Acc={acc:.4f}, F1_macro={f1m:.4f}")

    # GCN (Fully-connected) — ablation: erases all topology within each block
    print(f"\n  Running GCN (Fully-connected) LOOCV on N={N_total}...")
    preds_gcn_fc = loocv_gcn_classifier_pooled(
        A_fc, X_scaled, y_labels, n_runs=10, seed=42
    )
    acc = accuracy_score(y_labels, preds_gcn_fc)
    f1m = f1_score(y_labels, preds_gcn_fc, average='macro', zero_division=0)
    results.append({"Model": "GCN (Fully-connected)", "Accuracy": acc, "F1_Macro": f1m,
                    "preds": preds_gcn_fc})
    print(f"    Acc={acc:.4f}, F1_macro={f1m:.4f}")

    # Random Forest
    print(f"\n  Running Random Forest LOOCV on N={N_total}...")
    preds_rf = loocv_sklearn_classifier_pooled(
        RandomForestClassifier, X_scaled, y_labels,
        n_estimators=200, max_depth=5, random_state=42
    )
    acc = accuracy_score(y_labels, preds_rf)
    f1m = f1_score(y_labels, preds_rf, average='macro', zero_division=0)
    results.append({"Model": "Random Forest", "Accuracy": acc, "F1_Macro": f1m,
                    "preds": preds_rf})
    print(f"    Acc={acc:.4f}, F1_macro={f1m:.4f}")

    # XGBoost
    print(f"\n  Running XGBoost LOOCV on N={N_total}...")
    preds_xgb = loocv_sklearn_classifier_pooled(
        XGBClassifier, X_scaled, y_labels,
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    acc = accuracy_score(y_labels, preds_xgb)
    f1m = f1_score(y_labels, preds_xgb, average='macro', zero_division=0)
    results.append({"Model": "XGBoost", "Accuracy": acc, "F1_Macro": f1m,
                    "preds": preds_xgb})
    print(f"    Acc={acc:.4f}, F1_macro={f1m:.4f}")

    # Logistic Regression
    print(f"\n  Running Logistic Regression LOOCV on N={N_total}...")
    preds_lr = loocv_sklearn_classifier_pooled(
        LogisticRegression, X_scaled, y_labels,
        penalty='l2', C=1.0, max_iter=2000, multi_class='multinomial',
        random_state=42
    )
    acc = accuracy_score(y_labels, preds_lr)
    f1m = f1_score(y_labels, preds_lr, average='macro', zero_division=0)
    results.append({"Model": "Logistic Regression", "Accuracy": acc, "F1_Macro": f1m,
                    "preds": preds_lr})
    print(f"    Acc={acc:.4f}, F1_macro={f1m:.4f}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"POOLED N={N_total} RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':<30s} {'Acc':>8s} {'F1_macro':>10s} {'vs_Random':>10s}")
    print(f"  {'-'*60}")
    best_acc, best_model = 0, ""
    for r in results:
        gain = (r['Accuracy'] - 0.3333) / 0.3333 * 100
        marker = "  <<< BEST" if r['Accuracy'] > best_acc else ""
        if r['Accuracy'] > best_acc:
            best_acc, best_model = r['Accuracy'], r['Model']
        print(f"  {r['Model']:<30s} {r['Accuracy']:>8.4f} {r['F1_Macro']:>10.4f} {gain:>+9.0f}%{marker}")

    # ── Best model confusion matrix ────────────────────────────
    best_idx = max(range(len(results)), key=lambda i: results[i]['Accuracy'])
    best_r = results[best_idx]
    cm = confusion_matrix(y_labels, best_r['preds'])

    print(f"\n  Confusion Matrix ({best_r['Model']}):")
    print(f"  {'':>10s} {'Pred_Low':>10s} {'Pred_Med':>10s} {'Pred_High':>10s}")
    for i, cls in enumerate(class_names):
        print(f"  {'True_'+cls:<10s} {cm[i,0]:>10d} {cm[i,1]:>10d} {cm[i,2]:>10d}")

    print(f"\n  Classification Report:")
    print(classification_report(y_labels, best_r['preds'], target_names=class_names,
                                zero_division=0))

    # ── Per-agglomeration breakdown ────────────────────────────
    print(f"\n  Per-Agglomeration Accuracy ({best_r['Model']}):")
    print(f"  {'Agglomeration':<15s} {'N':>5s} {'Acc':>8s} {'F1_macro':>10s}")
    print(f"  {'-'*40}")
    for start, end, name in boundaries:
        y_true_agg = y_labels[start:end]
        y_pred_agg = best_r['preds'][start:end]
        acc_agg = accuracy_score(y_true_agg, y_pred_agg)
        f1_agg = f1_score(y_true_agg, y_pred_agg, average='macro', zero_division=0)
        print(f"  {name:<15s} {end-start:>5d} {acc_agg:>8.4f} {f1_agg:>10.4f}")

    # ── GCN Ablation Analysis ──────────────────────────────────
    gcn_models = [r for r in results if r['Model'].startswith('GCN')]
    print(f"\n{'='*70}")
    print(f"GCN ABLATION ANALYSIS: Does OD topology matter?")
    print(f"{'='*70}")
    print(f"  {'Graph Type':<30s} {'Acc':>8s} {'F1_macro':>10s} {'Δ vs OD':>10s} {'Interpretation':<30s}")
    print(f"  {'-'*90}")

    od_acc = next(r['Accuracy'] for r in gcn_models if 'OD' in r['Model'])
    for r in gcn_models:
        delta = r['Accuracy'] - od_acc
        if 'OD' in r['Model']:
            interp = "OD topology (real flows)"
        elif 'Random' in r['Model']:
            interp = "ER null (same density)" + (" — OD adds nothing" if delta >= -0.01 else " — OD topology is informative")
        elif 'Fully-connected' in r['Model']:
            interp = "Uniform intra-block (no topology)" + (" — OD adds nothing" if delta >= -0.01 else " — OD topology is informative")
        elif 'Geo' in r['Model']:
            interp = "Geographic proximity (k=3)"
        else:
            interp = ""
        print(f"  {r['Model']:<30s} {r['Accuracy']:>8.4f} {r['F1_Macro']:>10.4f} {delta:>+10.4f}  {interp}")

    print(f"\n  Key question: Does the specific OD network structure convey")
    print(f"  information beyond (a) what any random graph of equal density")
    print(f"  provides, and (b) what simple intra-block averaging provides?")
    delta_rand = od_acc - next(r['Accuracy'] for r in gcn_models if 'Random' in r['Model'])
    delta_fc = od_acc - next(r['Accuracy'] for r in gcn_models if 'Fully-connected' in r['Model'])
    print(f"  OD − Random graph = {delta_rand:+.4f}  {'*** OD topology is informative ***' if delta_rand > 0.02 else '— negligible contribution'}")
    print(f"  OD − Fully-connected = {delta_fc:+.4f}  {'*** OD topology is informative ***' if delta_fc > 0.02 else '— negligible contribution'}")

    # ── Statistical significance tests ─────────────────────────
    print(f"\n{'='*70}")
    print(f"STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*70}")

    # McNemar: GCN(OD) vs Random Forest
    mcnemar_rf = mcnemar_test(y_labels, preds_gcn_od, preds_rf)
    tbl = mcnemar_rf['table']
    print(f"\n  McNemar Test: GCN (OD) vs Random Forest")
    print(f"  Contingency table (correct/wrong under LOOCV):")
    print(f"                 RF correct   RF wrong   Total")
    print(f"  GCN correct      {tbl['a']:>5d}       {tbl['b']:>5d}      {tbl['a']+tbl['b']:>5d}")
    print(f"  GCN wrong        {tbl['c']:>5d}       {tbl['d']:>5d}      {tbl['c']+tbl['d']:>5d}")
    print(f"  Total            {tbl['a']+tbl['c']:>5d}       {tbl['b']+tbl['d']:>5d}      {N_total:>5d}")
    print(f"  χ² = {mcnemar_rf['statistic']:.3f}, p = {mcnemar_rf['p_value']:.4f}")
    print(f"  ΔAcc (OD − RF) = {mcnemar_rf['delta_acc']:+.4f}")
    if mcnemar_rf['p_value'] < 0.05:
        print(f"  *** GCN (OD) significantly outperforms RF at α=0.05 ***")
    elif mcnemar_rf['p_value'] < 0.10:
        print(f"  * GCN (OD) marginally outperforms RF at α=0.10")
    else:
        print(f"  — Difference is not statistically significant (p > 0.10)")

    # McNemar: GCN(OD) vs GCN(Random)
    mcnemar_rand = mcnemar_test(y_labels, preds_gcn_od, preds_gcn_rand)
    print(f"\n  McNemar Test: GCN (OD) vs GCN (Random graph)")
    print(f"  χ² = {mcnemar_rand['statistic']:.3f}, p = {mcnemar_rand['p_value']:.4f}")
    print(f"  ΔAcc (OD − Random) = {mcnemar_rand['delta_acc']:+.4f}")
    if mcnemar_rand['p_value'] < 0.05:
        print(f"  *** OD topology is significantly informative (p < 0.05) ***")
    elif mcnemar_rand['p_value'] < 0.10:
        print(f"  * OD topology is marginally informative (p < 0.10)")
    else:
        print(f"  — Cannot reject H₀: OD = random wiring (p > 0.10)")

    # Permutation test: OD vs Random graph
    perm_test = permutation_test_graph(y_labels, preds_gcn_od, preds_gcn_rand, n_perm=10000)
    print(f"\n  Permutation Test: OD vs Random graph (10,000 permutations)")
    print(f"  Observed ΔAcc = {perm_test['observed_delta']:+.4f}")
    print(f"  Null distribution: mean={perm_test['null_mean']:+.6f}, std={perm_test['null_std']:.4f}")
    print(f"  p = {perm_test['p_value']:.4f} (one-tailed, H₁: OD > Random)")
    if perm_test['p_value'] < 0.05:
        print(f"  *** OD topology significantly outperforms random wiring ***")
    elif perm_test['p_value'] < 0.10:
        print(f"  * OD topology marginally outperforms random wiring")

    # ── Comparison with per-agglomeration baselines ─────────────
    print(f"\n{'='*70}")
    print(f"POOLED vs SINGLE-AGGLOMERATION COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Setting':<25s} {'N':>5s} {'Best Model':<28s} {'Acc':>8s} {'F1_macro':>10s} {'vs_Random':>10s}")
    print(f"  {'-'*85}")

    # Single-agg baselines (from previous run)
    single_results = {
        "JJJ (single)": (13, "Logistic Regression", 0.5385, 0.5162),
        "YRD (single)": (41, "GCN (OD graph)", 0.4634, 0.4516),
        "PRD (single)": (21, "GCN (Geographic)", 0.6190, 0.6190),
        "CD_CQ (single)": (18, "XGBoost", 0.3333, 0.3349),
    }

    for label, (n, model, acc, f1) in single_results.items():
        gain = (acc - 0.3333) / 0.3333 * 100
        print(f"  {label:<25s} {n:>5d} {model:<28s} {acc:>8.4f} {f1:>10.4f} {gain:>+9.0f}%")

    # Pooled result
    pooled_gain = (best_acc - 0.3333) / 0.3333 * 100
    print(f"  {'POOLED (93-city)':<25s} {N_total:>5d} {best_model:<28s} {best_acc:>8.4f} {best_r['F1_Macro']:>10.4f} {pooled_gain:>+9.0f}%")

    # ── Feature importance (RF) ────────────────────────────────
    print(f"\n  Feature Importance (Random Forest, N={N_total}):")
    rf_full = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf_full.fit(X_scaled, y_labels)
    importances = rf_full.feature_importances_

    # Use aligned feature names
    if len(common_features) == len(importances):
        feat_names = common_features
        ranked = sorted(zip(feat_names, importances), key=lambda x: -x[1])
        for name, imp in ranked:
            bar = '█' * int(imp * 50)
            print(f"    {name:<28s} {imp:.4f} {bar}")

    # ── Save ──────────────────────────────────────────────────
    out_dir = base_dir / "results" / "pooled"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'preds'}
                                for r in results])
    df_results.round(4).to_csv(out_dir / "pooled_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'pooled_results.csv'}")

    # Also save per-city predictions for spatial figure generation
    preds_dict = {}
    for r in results:
        safe_name = r['model'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
        preds_dict[safe_name] = r['preds']
    preds_dict['y_true'] = y_labels
    preds_dict['city_idx'] = np.arange(N_total)
    df_preds = pd.DataFrame(preds_dict)
    df_preds.to_csv(out_dir / "pooled_predictions.csv", index=False)
    print(f"Saved: {out_dir / 'pooled_predictions.csv'} ({N_total} cities)")

    print("\nDone. Pooled 93-city benchmark complete.")


if __name__ == "__main__":
    main()
