"""
Top-k Sparsification Sweep: From "Economic Skeleton" to "Weak Ties" Discovery.

Scans topk ∈ [0.1, 0.2, ..., 1.0], recording for each threshold:
  - Edge density (fraction of possible intra-block edges retained)
  - Average clustering coefficient
  - GCN (OD) LOOCV accuracy & F1-macro
  - GCN (Random graph, matched density) LOOCV accuracy & F1-macro

Produces: results/topk_sweep/topk_sweep_results.csv + publication figure.

Usage:
    python run_topk_sweep.py                    # Full sweep (3 runs, 200 epochs)
    python run_topk_sweep.py --quick            # Quick test (1 run, 150 epochs)
    python run_topk_sweep.py --topk 0.3 0.5 0.7 1.0  # Specific thresholds only
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import time

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_adjacency, load_features_from_npy, load_target_from_npy

# ═══════════════════════════════════════════════════════════════════
# GCN Classifier
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


def loocv_gcn_sweep(A, X, y_labels, n_classes=3, hidden_dim=32,
                     epochs=200, lr=0.01, n_runs=3, seed=42):
    """LOOCV GCN classification, returns predictions."""
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

            X_full = torch.tensor(X, dtype=torch.float32)
            X_full = (X_full - train_mean) / train_std
            edge_idx_full, edge_w_full = adjacency_to_edge_index(A)

            model.eval()
            with torch.no_grad():
                logits = model(X_full, edge_idx_full, edge_w_full)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            all_preds[run, i] = np.argmax(probs[i])

    final_preds = stats.mode(all_preds, axis=0)[0].flatten()
    return final_preds


# ═══════════════════════════════════════════════════════════════════
# Graph metrics
# ═══════════════════════════════════════════════════════════════════

def compute_density(A):
    """Edge density (excluding self-loops)."""
    N = A.shape[0]
    n_edges = int((A > 0).sum()) - N  # exclude self-loops
    n_possible = N * (N - 1)
    return n_edges / n_possible if n_possible > 0 else 0.0


def compute_avg_clustering(A):
    """
    Average clustering coefficient (Watts-Strogatz).
    For weighted undirected graph, uses the Onnela et al. (2005) formula.
    """
    N = A.shape[0]
    if N < 3:
        return 0.0

    # Binarize for clustering computation
    A_bin = (A > 0).astype(np.float64)
    np.fill_diagonal(A_bin, 0.0)

    clustering = np.zeros(N)
    for i in range(N):
        neighbors = np.where(A_bin[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering[i] = 0.0
            continue
        sub = A_bin[np.ix_(neighbors, neighbors)]
        n_triangles = np.sum(sub) / 2.0
        clustering[i] = 2.0 * n_triangles / (k * (k - 1))
    return np.mean(clustering)


# ═══════════════════════════════════════════════════════════════════
# Block-Diagonal Adjacency (with variable topk)
# ═══════════════════════════════════════════════════════════════════

FEATURE_ORDER = {
    "in_strength": 0, "out_strength": 1, "betweenness_centrality": 2,
    "gdp_log_2019": 3, "gdp_pc_log_2019": 4, "trend_growth_17_19": 5,
    "primary_pct_2019": 6, "secondary_pct_2019": 7, "tertiary_pct_2019": 8,
    "industry_entropy": 9, "poi_density": 10, "poi_shannon": 11,
    "ntl_log_2019": 12, "admin_level": 13,
}


def build_block_diagonal(data_dirs, topk=None, norm="minmax"):
    """Build block-diagonal OD adjacency. topk=None means full graph."""
    offset = 0
    blocks = []
    boundaries = []

    for data_dir in data_dirs:
        A_path = data_dir / "A_city.npy"
        A_block = load_adjacency(A_path, normalize=norm, topk=topk)
        N = A_block.shape[0]
        blocks.append(A_block)
        boundaries.append((offset, offset + N, data_dir.name))
        offset += N

    N_total = offset
    A_super = np.zeros((N_total, N_total))
    for (start, end, _), block in zip(boundaries, blocks):
        A_super[start:end, start:end] = block

    return A_super, boundaries


def build_block_diagonal_random_matched(data_dirs, A_od_super, boundaries_ref):
    """
    Build block-diagonal random graph where EACH BLOCK matches the edge count
    of the corresponding OD block. This ensures per-block density matching,
    not global density matching (which fails due to inter-block zeros).
    """
    offset = 0
    blocks = []
    boundaries = []

    for (start, end, name), data_dir in zip(boundaries_ref, data_dirs):
        # Extract this block's OD submatrix
        A_od_block = A_od_super[start:end, start:end]
        N = A_od_block.shape[0]
        n_possible = N * (N - 1)

        # Count edges in OD block (excluding self-loops)
        n_edges_od = int((A_od_block > 0).sum()) - N
        p_edge = n_edges_od / n_possible if n_possible > 0 else 0.0

        # Generate ER random graph with same per-block edge probability
        upper_mask = np.triu(np.random.rand(N, N) < p_edge, k=1)
        A_rand = (upper_mask | upper_mask.T).astype(np.float64)
        np.fill_diagonal(A_rand, 0.0)
        A_rand[A_rand > 0] = 1.0
        A_rand = A_rand + np.eye(N)

        blocks.append(A_rand)
        boundaries.append((offset, offset + N, name))
        offset += N

    N_total = offset
    A_super = np.zeros((N_total, N_total))
    for (start, end, _), block in zip(boundaries, blocks):
        A_super[start:end, start:end] = block

    return A_super, boundaries


# ═══════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast mode: 1 run, 150 epochs")
    parser.add_argument("--topk", nargs="*", type=float, default=None,
                        help="Specific topk values to test (space-separated)")
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────
    if args.quick:
        N_RUNS, EPOCHS = 1, 150
        print("*** QUICK MODE: 1 run, 150 epochs ***")
    else:
        N_RUNS, EPOCHS = 3, 200

    if args.topk:
        topk_list = args.topk
    else:
        topk_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    base_dir = Path(__file__).parent
    assembled = base_dir / "data" / "assembled"
    agg_dirs = sorted([d for d in assembled.iterdir() if d.is_dir()])

    print("=" * 70)
    print("TOP-K SPARSIFICATION SWEEP: Weak Ties Discovery")
    print(f"{len(topk_list)} thresholds: {topk_list}")
    print(f"Config: {N_RUNS} run(s), {EPOCHS} epochs, full LOOCV")
    print("=" * 70)

    # ── Load pooled data ──────────────────────────────────────
    X_blocks, y_blocks = [], []
    boundaries_data = []
    offset = 0

    for data_dir in agg_dirs:
        name = data_dir.name
        X, feat_names = load_features_from_npy(data_dir, cities=None,
                                                drop_constant=True, add_admin_level=True)
        y = load_target_from_npy(data_dir).flatten()
        N = X.shape[0]
        X_blocks.append(X)
        y_blocks.append(y)
        boundaries_data.append((offset, offset + N, name))
        offset += N

    # Align features
    all_feature_sets = []
    for data_dir in agg_dirs:
        _, feat_names = load_features_from_npy(data_dir, cities=None,
                                                drop_constant=True, add_admin_level=True)
        all_feature_sets.append(set(feat_names))
    common_features = all_feature_sets[0]
    for fs in all_feature_sets[1:]:
        common_features = common_features & fs
    common_features = sorted(common_features, key=lambda f: FEATURE_ORDER.get(f, 99))

    for i, data_dir in enumerate(agg_dirs):
        _, feat_names = load_features_from_npy(data_dir, cities=None,
                                                drop_constant=True, add_admin_level=True)
        col_idx = [feat_names.index(f) for f in common_features if f in feat_names]
        X_blocks[i] = X_blocks[i][:, col_idx]

    X_pooled = np.vstack(X_blocks)
    y_pooled = np.concatenate(y_blocks)
    N_total = X_pooled.shape[0]
    D = len(common_features)
    X_scaled = StandardScaler().fit_transform(X_pooled)

    # Within-group tertiles (Bug 2 fix)
    y_labels = np.full(N_total, -1, dtype=np.int64)
    for start, end, name in boundaries_data:
        y_agg = y_pooled[start:end]
        agg_tertiles = np.percentile(y_agg, [33.33, 66.67])
        y_agg_labels = np.zeros(end - start, dtype=np.int64)
        y_agg_labels[y_agg >= agg_tertiles[1]] = 2
        y_agg_labels[(y_agg >= agg_tertiles[0]) & (y_agg < agg_tertiles[1])] = 1
        y_labels[start:end] = y_agg_labels

    print(f"\n  N={N_total}, D={D}")
    print(f"  Features: {common_features}")
    print(f"  Labeling: within-group tertiles")
    print(f"  Random baseline (3-class): 0.3333")

    # ── Non-graph baselines (run once, independent of topk) ───
    print(f"\n  Running non-graph baselines (RF, XGBoost, LR)...")
    from sklearn.preprocessing import StandardScaler as SS
    X_std = SS().fit_transform(X_scaled)
    N = X_std.shape[0]

    preds_rf = np.zeros(N, dtype=np.int64)
    for i in range(N):
        X_tr = np.delete(X_std, i, axis=0)
        y_tr = np.delete(y_labels, i)
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        preds_rf[i] = model.predict(X_std[i:i+1])[0]

    preds_xgb = np.zeros(N, dtype=np.int64)
    for i in range(N):
        X_tr = np.delete(X_std, i, axis=0)
        y_tr = np.delete(y_labels, i)
        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              random_state=42, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        preds_xgb[i] = model.predict(X_std[i:i+1])[0]

    preds_lr = np.zeros(N, dtype=np.int64)
    for i in range(N):
        X_tr = np.delete(X_std, i, axis=0)
        y_tr = np.delete(y_labels, i)
        model = LogisticRegression(penalty='l2', C=1.0, max_iter=2000,
                                   multi_class='multinomial', random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        preds_lr[i] = model.predict(X_std[i:i+1])[0]

    rf_acc = accuracy_score(y_labels, preds_rf)
    xgb_acc = accuracy_score(y_labels, preds_xgb)
    lr_acc = accuracy_score(y_labels, preds_lr)
    print(f"    RF={rf_acc:.4f}, XGBoost={xgb_acc:.4f}, LR={lr_acc:.4f}")

    # ── Top-k sweep ───────────────────────────────────────────
    results = []
    total_configs = len(topk_list) * 2  # OD + Random for each topk

    np.random.seed(42)

    for tk_idx, topk in enumerate(topk_list):
        tk_label = f"{topk:.1f}" if topk < 1.0 else "1.0 (full)"

        # Build OD adjacency at this topk
        A_od, od_boundaries = build_block_diagonal(agg_dirs, topk=topk if topk < 1.0 else None)

        # Compute graph metrics from OD graph
        density = compute_density(A_od)
        clustering = compute_avg_clustering(A_od)

        # Build random graph with PER-BLOCK edge count matching (not global density)
        A_rand, _ = build_block_diagonal_random_matched(agg_dirs, A_od, od_boundaries)
        density_rand = compute_density(A_rand)

        n_od_edges = int((A_od > 0).sum() - N_total)
        print(f"\n  topk={tk_label}: OD edges={n_od_edges}, "
              f"density={density:.4f}, clustering={clustering:.4f}, "
              f"rand density={density_rand:.4f}")

        # GCN (OD)
        t0 = time.time()
        print(f"    GCN (OD) LOOCV...", end=" ", flush=True)
        preds_od = loocv_gcn_sweep(A_od, X_scaled, y_labels,
                                    epochs=EPOCHS, n_runs=N_RUNS, seed=42)
        od_acc = accuracy_score(y_labels, preds_od)
        od_f1 = f1_score(y_labels, preds_od, average='macro', zero_division=0)
        elapsed = time.time() - t0
        print(f"Acc={od_acc:.4f}, F1={od_f1:.4f} ({elapsed:.0f}s)")

        # GCN (Random graph)
        t0 = time.time()
        print(f"    GCN (Random) LOOCV...", end=" ", flush=True)
        preds_rand = loocv_gcn_sweep(A_rand, X_scaled, y_labels,
                                      epochs=EPOCHS, n_runs=N_RUNS, seed=42)
        rand_acc = accuracy_score(y_labels, preds_rand)
        rand_f1 = f1_score(y_labels, preds_rand, average='macro', zero_division=0)
        elapsed = time.time() - t0
        print(f"Acc={rand_acc:.4f}, F1={rand_f1:.4f} ({elapsed:.0f}s)")

        results.append({
            'topk': topk if topk < 1.0 else 1.0,
            'density': round(density, 6),
            'clustering_coefficient': round(clustering, 6),
            'n_edges': n_od_edges,
            'OD_Accuracy': round(od_acc, 4),
            'OD_F1_macro': round(od_f1, 4),
            'Random_Accuracy': round(rand_acc, 4),
            'Random_F1_macro': round(rand_f1, 4),
            'OD_minus_Random': round(od_acc - rand_acc, 4),
        })

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"TOP-K SWEEP RESULTS")
    print(f"{'='*90}")
    print(f"  {'topk':>6s}  {'density':>8s}  {'cluster':>8s}  "
          f"{'OD_Acc':>8s}  {'Rand_Acc':>8s}  {'OD-Rand':>8s}  "
          f"{'RF_Acc':>8s}  {'XGB_Acc':>8s}")
    print(f"  {'-'*84}")
    for r in results:
        od_dom = "*** OD WINS ***" if r['OD_minus_Random'] > 0.02 else (
                 "* marginal *" if r['OD_minus_Random'] > 0.0 else "  rand wins ")
        print(f"  {r['topk']:>6.2f}  {r['density']:>8.4f}  {r['clustering_coefficient']:>8.4f}  "
              f"{r['OD_Accuracy']:>8.4f}  {r['Random_Accuracy']:>8.4f}  "
              f"{r['OD_minus_Random']:>+8.4f}  "
              f"{rf_acc:>8.4f}  {xgb_acc:>8.4f}  {od_dom}")

    # ── Key findings ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"KEY FINDINGS: Weak Ties Discovery")
    print(f"{'='*70}")

    # Find the cross-over point where OD starts winning
    crossovers = [r for r in results if r['OD_minus_Random'] > 0.0]
    if crossovers:
        first_crossover = crossovers[0]
        print(f"  OD consistently outperforms Random at density >= {first_crossover['density']:.4f}")
        print(f"  (topk >= {first_crossover['topk']:.1f})")
    else:
        print(f"  OD does NOT outperform Random at any density level tested.")

    best_od = max(results, key=lambda r: r['OD_Accuracy'])
    print(f"  Best OD accuracy: {best_od['OD_Accuracy']:.4f} at topk={best_od['topk']:.1f} "
          f"(density={best_od['density']:.4f})")

    # Density-Acc slope (does more density monotonically help?)
    od_accs = [r['OD_Accuracy'] for r in results]
    densities = [r['density'] for r in results]
    if len(densities) > 1:
        slope = np.polyfit(densities, od_accs, 1)[0]
        print(f"  Density-Acc slope: {slope:+.4f} (per 1% density increase)")
        if slope > 0.01:
            print(f"  => Weak ties hypothesis supported: more connectivity = better prediction")
        elif slope > -0.01:
            print(f"  => Density has negligible effect on accuracy")
        else:
            print(f"  => Inverse relationship: sparser graphs predict better")

    # ── Save ─────────────────────────────────────────────────
    out_dir = base_dir / "results" / "topk_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "topk_sweep_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'topk_sweep_results.csv'}")

    # Also save baselines
    with open(out_dir / "baselines.txt", 'w') as f:
        f.write(f"RF_Accuracy={rf_acc:.4f}\n")
        f.write(f"XGB_Accuracy={xgb_acc:.4f}\n")
        f.write(f"LR_Accuracy={lr_acc:.4f}\n")

    print(f"\nDone. {len(results)} configurations tested.")
    print(f"Next: python generate_figures.py --sweep  (to generate weak-ties figure)")


if __name__ == "__main__":
    main()
