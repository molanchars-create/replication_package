"""
Quick ablation test: GCN (Random graph) vs GCN (Fully-connected) vs known OD/Geo.

Only runs the 2 new ablation baselines; reuses known results for OD, Geo, RF, XGBoost, LR.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

sys.path.insert(0, str(Path(__file__).parent))

from run_pooled_classification import (
    build_block_diagonal_adjacency,
    build_random_graph_adjacency,
    build_fully_connected_adjacency,
    loocv_gcn_classifier_pooled,
    loocv_sklearn_classifier_pooled,
    FEATURE_ORDER,
)
from data_loader import load_features_from_npy, load_target_from_npy


def main():
    base_dir = Path(__file__).parent
    assembled = base_dir / "data" / "assembled"
    agg_dirs = sorted([d for d in assembled.iterdir() if d.is_dir()])

    print("=" * 70)
    print("GCN ABLATION QUICK TEST: Random graph vs Fully-connected")
    print("=" * 70)

    # ── Load data (same as pooled script) ──────────────────────
    X_blocks, y_blocks = [], []
    boundaries = []
    offset = 0

    for data_dir in agg_dirs:
        name = data_dir.name
        X, feat_names = load_features_from_npy(data_dir, cities=None,
                                                drop_constant=True, add_admin_level=True)
        y = load_target_from_npy(data_dir).flatten()
        N = X.shape[0]
        X_blocks.append(X)
        y_blocks.append(y)
        boundaries.append((offset, offset + N, name))
        offset += N
        print(f"  {name}: N={N}, D={X.shape[1]}")

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

    tertiles = np.percentile(y_pooled, [33.33, 66.67])
    y_labels = np.zeros(N_total, dtype=np.int64)
    y_labels[y_pooled >= tertiles[1]] = 2
    y_labels[(y_pooled >= tertiles[0]) & (y_pooled < tertiles[1])] = 1

    print(f"\n  N={N_total}, D={D}, features={common_features}")
    print(f"  RI range: [{y_pooled.min():.3f}, {y_pooled.max():.3f}]")
    print(f"  Tertiles: [{tertiles[0]:.4f}, {tertiles[1]:.4f}]")

    # ── Build adjacency matrices ───────────────────────────────
    np.random.seed(42)
    A_od, _ = build_block_diagonal_adjacency(agg_dirs)
    A_rand, _ = build_random_graph_adjacency(agg_dirs)
    A_fc, _ = build_fully_connected_adjacency(agg_dirs)

    n_od = int((A_od > 0).sum() - N_total)
    n_rand = int((A_rand > 0).sum() - N_total)
    n_fc = int((A_fc > 0).sum() - N_total)
    print(f"\n  Edges -- OD: {n_od}, Random: {n_rand}, FC: {n_fc}")

    # ── Run GCN baselines (reduced runs for speed) ─────────────
    N_RUNS = 3  # Quick test (use 10 for final numbers)

    print(f"\n  [1/3] GCN (OD) -- {N_RUNS} runs...")
    preds_od = loocv_gcn_classifier_pooled(
        A_od, X_scaled, y_labels, n_runs=N_RUNS, seed=42
    )
    acc_od = accuracy_score(y_labels, preds_od)
    f1_od = f1_score(y_labels, preds_od, average='macro', zero_division=0)
    print(f"    Acc={acc_od:.4f}, F1_macro={f1_od:.4f}")

    print(f"\n  [2/3] GCN (Random graph = ER null model) -- {N_RUNS} runs...")
    preds_rand = loocv_gcn_classifier_pooled(
        A_rand, X_scaled, y_labels, n_runs=N_RUNS, seed=42
    )
    acc_rand = accuracy_score(y_labels, preds_rand)
    f1_rand = f1_score(y_labels, preds_rand, average='macro', zero_division=0)
    print(f"    Acc={acc_rand:.4f}, F1_macro={f1_rand:.4f}")

    print(f"\n  [3/3] GCN (Fully-connected = uniform intra-block) -- {N_RUNS} runs...")
    preds_fc = loocv_gcn_classifier_pooled(
        A_fc, X_scaled, y_labels, n_runs=N_RUNS, seed=42
    )
    acc_fc = accuracy_score(y_labels, preds_fc)
    f1_fc = f1_score(y_labels, preds_fc, average='macro', zero_division=0)
    print(f"    Acc={acc_fc:.4f}, F1_macro={f1_fc:.4f}")

    # ── Also run RF and XGBoost for comparison ─────────────────
    print(f"\n  [4] Random Forest (N={N_total})...")
    preds_rf = loocv_sklearn_classifier_pooled(
        RandomForestClassifier, X_scaled, y_labels,
        n_estimators=200, max_depth=5, random_state=42
    )
    acc_rf = accuracy_score(y_labels, preds_rf)
    f1_rf = f1_score(y_labels, preds_rf, average='macro', zero_division=0)

    print(f"  [5] XGBoost (N={N_total})...")
    preds_xgb = loocv_sklearn_classifier_pooled(
        XGBClassifier, X_scaled, y_labels,
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    acc_xgb = accuracy_score(y_labels, preds_xgb)
    f1_xgb = f1_score(y_labels, preds_xgb, average='macro', zero_division=0)

    print(f"  [6] Logistic Regression (N={N_total})...")
    preds_lr = loocv_sklearn_classifier_pooled(
        LogisticRegression, X_scaled, y_labels,
        penalty='l2', C=1.0, max_iter=2000, multi_class='multinomial',
        random_state=42
    )
    acc_lr = accuracy_score(y_labels, preds_lr)
    f1_lr = f1_score(y_labels, preds_lr, average='macro', zero_division=0)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"ABLATION RESULTS (N={N_total}, D={D})")
    print(f"{'='*70}")
    print(f"  {'Model':<30s} {'Acc':>8s} {'F1_macro':>10s} {'vs_OD':>8s} {'vs_Random':>10s}")
    print(f"  {'-'*70}")

    all_results = [
        ("GCN (OD graph)", acc_od, f1_od),
        ("GCN (Random graph)", acc_rand, f1_rand),
        ("GCN (Fully-connected)", acc_fc, f1_fc),
        ("Random Forest", acc_rf, f1_rf),
        ("XGBoost", acc_xgb, f1_xgb),
        ("Logistic Regression", acc_lr, f1_lr),
    ]

    for name, acc, f1 in all_results:
        d_od = acc - acc_od
        d_rand = acc - acc_rand
        print(f"  {name:<30s} {acc:>8.4f} {f1:>10.4f} {d_od:>+8.4f} {d_rand:>+10.4f}")

    # ── Key inference ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"KEY INFERENCE")
    print(f"{'='*70}")
    delta_od_rand = acc_od - acc_rand
    delta_od_fc = acc_od - acc_fc

    if delta_od_rand > 0.02:
        print(f"  OD - Random = {delta_od_rand:+.4f} => OD topology carries information "
              f"beyond random wiring of equal density.")
    elif delta_od_rand > -0.01:
        print(f"  OD - Random = {delta_od_rand:+.4f} => OD topology contributes no "
              f"discernible advantage over random wiring.")
    else:
        print(f"  OD - Random = {delta_od_rand:+.4f} => OD performs WORSE than random "
              f"graph -- OD structure may be noise, not signal.")

    if delta_od_fc > 0.02:
        print(f"  OD - FC = {delta_od_fc:+.4f} => OD topology carries information "
              f"beyond simple intra-block averaging.")
    elif delta_od_fc > -0.01:
        print(f"  OD - FC = {delta_od_fc:+.4f} => OD topology contributes no "
              f"discernible advantage over uniform intra-block connections.")
    else:
        print(f"  OD - FC = {delta_od_fc:+.4f} => OD performs WORSE than uniform "
              f"connections -- knowing the agglomeration block is sufficient.")

    print(f"\n  Random baseline (3-class): 0.3333")
    print(f"  GCN advantage over RF: {acc_od - acc_rf:+.4f}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
