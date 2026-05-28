"""
Classification benchmark: 3-class resilience prediction (High/Medium/Low).

Compares GCN, Random Forest, XGBoost, Ridge Classifier under LOOCV across
all 4 urban agglomerations. Target: tertile-based RI categories.

Usage:
    python run_classification.py --all
    python run_classification.py --data data/assembled/JJJ
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    load_adjacency, build_geographic_adjacency,
    load_features_from_npy, load_target_from_npy,
)
from models import adjacency_to_edge_index, _rebuild_edge_index_drop_node


# ═══════════════════════════════════════════════════════════════════
# GCN Classifier
# ═══════════════════════════════════════════════════════════════════

class GCNClassifier(nn.Module):
    """Two-layer GCN for node classification."""
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


def train_gcn_classifier(model, x, edge_index, edge_weight, y_labels,
                         epochs=300, lr=0.01, weight_decay=5e-4):
    """Train GCN classifier."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_weight)
        loss = criterion(logits, y_labels)
        loss.backward()
        optimizer.step()

    return model


def loocv_gcn_classifier(A, X, y_labels, hidden_dim=32, epochs=300, lr=0.01,
                         n_runs=10, seed=42):
    """LOOCV for GCN classification. Returns predictions and probabilities."""
    N = A.shape[0]
    D = X.shape[1]
    n_classes = len(np.unique(y_labels))
    all_preds = np.zeros((n_runs, N), dtype=np.int64)
    all_probs = np.zeros((n_runs, N, n_classes), dtype=np.float32)

    for run in range(n_runs):
        np.random.seed(seed + run)
        torch.manual_seed(seed + run)

        for i in range(N):
            edge_index, edge_weight, keep = _rebuild_edge_index_drop_node(A, i)
            X_train = torch.tensor(X[keep], dtype=torch.float32)
            y_train = torch.tensor(y_labels[keep], dtype=torch.long)

            # Standardize features within training fold
            train_mean = X_train.mean(dim=0, keepdim=True)
            train_std = X_train.std(dim=0, keepdim=True)
            train_std[train_std == 0] = 1.0
            X_train = (X_train - train_mean) / train_std

            model = GCNClassifier(D, hidden_dim=hidden_dim, n_classes=n_classes)
            train_gcn_classifier(model, X_train, edge_index, edge_weight, y_train,
                                 epochs=epochs, lr=lr)

            # Predict on full graph
            X_full = torch.tensor(X, dtype=torch.float32)
            X_full = (X_full - train_mean) / train_std
            edge_idx_full, edge_w_full = adjacency_to_edge_index(A)

            model.eval()
            with torch.no_grad():
                logits = model(X_full, edge_idx_full, edge_w_full)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            all_preds[run, i] = np.argmax(probs[i])
            all_probs[run, i] = probs[i]

    # Majority vote across runs
    from scipy import stats
    final_preds = stats.mode(all_preds, axis=0)[0].flatten()
    return final_preds, all_probs.mean(axis=0)


# ═══════════════════════════════════════════════════════════════════
# Classical classifiers with LOOCV
# ═══════════════════════════════════════════════════════════════════

def loocv_sklearn_classifier(model_class, X, y_labels, **kwargs):
    """LOOCV for sklearn-compatible classifiers."""
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
# Evaluation
# ═══════════════════════════════════════════════════════════════════

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def evaluate(y_true, y_pred, class_names):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'Accuracy': round(acc, 4),
        'F1_Macro': round(f1_macro, 4),
        'F1_Weighted': round(f1_weighted, 4),
        'ConfusionMatrix': cm,
        'ClassReport': classification_report(y_true, y_pred, target_names=class_names,
                                             zero_division=0)
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gcn-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    if args.all:
        data_dirs = sorted([d for d in (base_dir / "data" / "assembled").iterdir() if d.is_dir()])
    elif args.data:
        data_dirs = [Path(args.data)]
    else:
        data_dirs = [base_dir / "data" / "assembled" / "JJJ"]

    all_results = []

    for data_dir in data_dirs:
        name = data_dir.name
        print(f"\n{'='*70}")
        print(f"Classification Benchmark: {name}")
        print(f"{'='*70}")

        # Load data
        meta_path = data_dir / "city_metadata.csv"
        coords = None
        cities = None
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            if "lon" in meta.columns and "lat" in meta.columns:
                coords = meta[["lon", "lat"]].values
            if "city" in meta.columns:
                cities = list(meta["city"])

        X, feature_names = load_features_from_npy(
            data_dir, cities=cities, drop_constant=True, add_admin_level=True
        )
        y = load_target_from_npy(data_dir)
        N, D = X.shape

        # Create tertile-based labels
        y_flat = y.flatten()
        tertiles = np.percentile(y_flat, [33.33, 66.67])
        y_labels = np.zeros(N, dtype=np.int64)
        y_labels[y_flat >= tertiles[1]] = 2     # High
        y_labels[(y_flat >= tertiles[0]) & (y_flat < tertiles[1])] = 1  # Medium
        # Low stays 0

        class_names = ["Low", "Medium", "High"]
        n_per_class = {c: int(np.sum(y_labels == i)) for i, c in enumerate(class_names)}

        print(f"N={N}, D={D}")
        print(f"RI range: [{y_flat.min():.3f}, {y_flat.max():.3f}]")
        print(f"Tertile boundaries: [{tertiles[0]:.4f}, {tertiles[1]:.4f}]")
        print(f"Class distribution: {n_per_class}")
        print(f"Random baseline (3-class): 0.3333")

        # Load adjacency
        A_od = load_adjacency(data_dir / "A_city.npy", normalize="minmax", topk=0.3)
        A_geo = build_geographic_adjacency(N, coords, k=3)

        # Ablation: random graph (ER, same edge density as OD)
        n_od_edges = int((A_od > 0).sum() - N)  # excl. self-loops
        n_possible = N * (N - 1)
        p_edge = n_od_edges / n_possible if n_possible > 0 else 0.0
        np.random.seed(args.seed)
        upper_mask = np.triu(np.random.rand(N, N) < p_edge, k=1)
        A_rand = (upper_mask | upper_mask.T).astype(np.float64)
        np.fill_diagonal(A_rand, 0.0)
        A_rand[A_rand > 0] = 1.0
        A_rand = A_rand + np.eye(N)

        # Ablation: fully-connected (uniform intra-group)
        A_fc = np.ones((N, N), dtype=np.float64)
        np.fill_diagonal(A_fc, 0.0)
        A_fc = A_fc + np.eye(N)

        # ── Run models ──────────────────────────────────────────
        results = []

        # GCN (OD graph)
        print("\n  Running GCN (OD graph) LOOCV...")
        preds_gcn_od, probs_gcn_od = loocv_gcn_classifier(
            A_od, X, y_labels, n_runs=args.gcn_runs, seed=args.seed
        )
        metrics = evaluate(y_labels, preds_gcn_od, class_names)
        metrics["Model"] = "GCN (OD graph)"
        results.append(metrics)
        print(f"    Acc={metrics['Accuracy']:.4f}, F1_macro={metrics['F1_Macro']:.4f}")

        # GCN (Geographic)
        print("  Running GCN (Geographic) LOOCV...")
        preds_gcn_geo, probs_gcn_geo = loocv_gcn_classifier(
            A_geo, X, y_labels, n_runs=args.gcn_runs, seed=args.seed
        )
        metrics = evaluate(y_labels, preds_gcn_geo, class_names)
        metrics["Model"] = "GCN (Geographic)"
        results.append(metrics)
        print(f"    Acc={metrics['Accuracy']:.4f}, F1_macro={metrics['F1_Macro']:.4f}")

        # GCN (Random graph) — ablation baseline
        print("  Running GCN (Random graph) LOOCV...")
        preds_gcn_rand, _ = loocv_gcn_classifier(
            A_rand, X, y_labels, n_runs=args.gcn_runs, seed=args.seed
        )
        metrics = evaluate(y_labels, preds_gcn_rand, class_names)
        metrics["Model"] = "GCN (Random graph)"
        results.append(metrics)
        print(f"    Acc={metrics['Accuracy']:.4f}, F1_macro={metrics['F1_Macro']:.4f}")

        # GCN (Fully-connected) — ablation baseline
        print("  Running GCN (Fully-connected) LOOCV...")
        preds_gcn_fc, _ = loocv_gcn_classifier(
            A_fc, X, y_labels, n_runs=args.gcn_runs, seed=args.seed
        )
        metrics = evaluate(y_labels, preds_gcn_fc, class_names)
        metrics["Model"] = "GCN (Fully-connected)"
        results.append(metrics)
        print(f"    Acc={metrics['Accuracy']:.4f}, F1_macro={metrics['F1_Macro']:.4f}")

        # Random Forest
        print("  Running Random Forest LOOCV...")
        preds_rf = loocv_sklearn_classifier(
            RandomForestClassifier, X, y_labels,
            n_estimators=200, max_depth=5, random_state=args.seed
        )
        metrics = evaluate(y_labels, preds_rf, class_names)
        metrics["Model"] = "Random Forest"
        results.append(metrics)
        print(f"    Acc={metrics['Accuracy']:.4f}, F1_macro={metrics['F1_Macro']:.4f}")

        # XGBoost
        print("  Running XGBoost LOOCV...")
        preds_xgb = loocv_sklearn_classifier(
            XGBClassifier, X, y_labels,
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=args.seed, verbosity=0
        )
        metrics = evaluate(y_labels, preds_xgb, class_names)
        metrics["Model"] = "XGBoost"
        results.append(metrics)
        print(f"    Acc={metrics['Accuracy']:.4f}, F1_macro={metrics['F1_Macro']:.4f}")

        # Logistic Regression (Ridge penalty)
        print("  Running Logistic Regression LOOCV...")
        preds_lr = loocv_sklearn_classifier(
            LogisticRegression, X, y_labels,
            penalty='l2', C=1.0, max_iter=2000, multi_class='multinomial',
            random_state=args.seed
        )
        metrics = evaluate(y_labels, preds_lr, class_names)
        metrics["Model"] = "Logistic Regression"
        results.append(metrics)
        print(f"    Acc={metrics['Accuracy']:.4f}, F1_macro={metrics['F1_Macro']:.4f}")

        # MLP
        print("  Running MLP LOOCV...")
        preds_mlp = loocv_sklearn_classifier(
            MLPClassifier, X, y_labels,
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=args.seed,
            early_stopping=True, validation_fraction=0.2
        )
        metrics = evaluate(y_labels, preds_mlp, class_names)
        metrics["Model"] = "MLP"
        results.append(metrics)
        print(f"    Acc={metrics['Accuracy']:.4f}, F1_macro={metrics['F1_Macro']:.4f}")

        # ── Summary table ───────────────────────────────────────
        print(f"\n  {'Model':<25s} {'Acc':>8s} {'F1_macro':>10s} {'F1_weight':>10s}")
        print(f"  {'-'*55}")
        best_acc, best_model = 0, ""
        for r in results:
            marker = "  <<< BEST" if r['Accuracy'] > best_acc else ""
            if r['Accuracy'] > best_acc:
                best_acc, best_model = r['Accuracy'], r['Model']
            print(f"  {r['Model']:<25s} {r['Accuracy']:>8.4f} {r['F1_Macro']:>10.4f} {r['F1_Weighted']:>10.4f}{marker}")

        print(f"\n  Random baseline (3-class): 0.3333")
        gain = (best_acc - 0.3333) / 0.3333 * 100
        print(f"  Best model: {best_model} ({best_acc:.4f}), +{gain:.0f}% above random")

        # ── Confusion matrix for best model ─────────────────────
        best_idx = max(range(len(results)), key=lambda i: results[i]['Accuracy'])
        best_r = results[best_idx]
        print(f"\n  Confusion Matrix ({best_r['Model']}):")
        print(f"  {'':>8s} {'Pred_Low':>10s} {'Pred_Med':>10s} {'Pred_High':>10s}")
        cm = best_r['ConfusionMatrix']
        for i, cls in enumerate(class_names):
            print(f"  {'True_'+cls:<8s} {cm[i,0]:>10d} {cm[i,1]:>10d} {cm[i,2]:>10d}")

        print(f"\n  Classification Report ({best_r['Model']}):")
        print(f"  {best_r['ClassReport']}")

        # ── Store for aggregate ─────────────────────────────────
        for r in results:
            r['Agglomeration'] = name
            r['N'] = N
            r['D'] = D
            del r['ConfusionMatrix']
            del r['ClassReport']
        all_results.extend(results)

    # ── Aggregate summary ──────────────────────────────────────
    if args.all and len(data_dirs) > 1:
        print(f"\n{'='*70}")
        print(f"AGGREGATE SUMMARY: Classification (3-class)")
        print(f"{'='*70}")
        df = pd.DataFrame(all_results)
        summary = df.pivot_table(
            index=['Agglomeration', 'N'],
            columns='Model', values='Accuracy', aggfunc='first'
        )
        print(summary.round(4).to_string())

        # Best per agglomeration
        print(f"\n{'Agg':<10s} {'Best Model':<25s} {'Acc':>8s} {'F1_macro':>10s} {'vs_Random':>10s}")
        print(f"{'-'*65}")
        for agg in df['Agglomeration'].unique():
            subset = df[df['Agglomeration'] == agg]
            best = subset.loc[subset['Accuracy'].idxmax()]
            gain = (best['Accuracy'] - 0.3333) / 0.3333 * 100
            print(f"{agg:<10s} {best['Model']:<25s} {best['Accuracy']:>8.4f} {best['F1_Macro']:>10.4f} {gain:>+9.0f}%")

        # Save
        out_dir = base_dir / "results" / "classification"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.round(4).to_csv(out_dir / "classification_results.csv", index=False)
        print(f"\nSaved: {out_dir / 'classification_results.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
