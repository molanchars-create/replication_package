"""
Standard data loader for urban agglomeration resilience prediction.
Loads preprocessed .npy files directly (not CSV reconstruction).

X_city.npy column layout (confirmed by rank-correlation matching):
    0:  baseline_mean        [LEAKAGE — used to compute target]
    1:  shock_min            [LEAKAGE]
    2:  recovery_mean        [LEAKAGE]
    3:  FR                   [LEAKAGE — component of RI]
    4:  RR                   [LEAKAGE — component of RI]
    5:  recovery_time_days   [LEAKAGE — process-derived]
    ---------------------------------------------------------
    6:  in_strength          [OK — OD network property]
    7:  out_strength         [OK]
    8:  total_strength       [OK]
    9:  degree               [OK but CONSTANT for JJJ — zero variance]
    10: betweenness_centrality [OK]
    11: poi_count_2017       [OK]
    12: poi_count_2019       [OK — has 1 NaN]
    13: poi_count_2023       [OK]
    14: shannon_2017         [OK]
    15: shannon_2019         [OK — has 1 NaN]
    16: shannon_2023         [OK]
    17: poi_total_all_years  [OK]

Clean feature set: columns 6–17 (12 features), dropping constant col 9 → 11 usable.
For expanded samples: add admin_level, gdp_per_capita from external data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------

def load_adjacency(path, normalize="minmax", topk=None):
    """
    Load OD adjacency from .npy, apply normalization and optional sparsification.

    Parameters
    ----------
    path : str or Path
        Path to A_city.npy (N×N, NaN on diagonal).
    normalize : str
        "minmax", "row", "binary", or "none".
    topk : float or None
        If set, retain only top-k fraction of outgoing edges per node.

    Returns
    -------
    A : (N, N) ndarray with self-loops added.
    """
    A = np.load(path).copy()
    N = A.shape[0]
    A = np.nan_to_num(A, nan=0.0)

    # Node-level sparsification
    if topk is not None and 0 < topk < 1:
        for i in range(N):
            row = A[i].copy()
            row[i] = 0.0
            nonzero = row[row > 0]
            if len(nonzero) == 0:
                continue
            k = max(1, int(np.ceil(len(nonzero) * topk)))
            threshold = np.sort(nonzero)[-k]
            A[i, A[i] < threshold] = 0.0

    # Normalization
    if normalize == "minmax":
        vmin, vmax = A.min(), A.max()
        if vmax > vmin:
            A = (A - vmin) / (vmax - vmin)
    elif normalize == "row":
        rowsum = A.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        A = A / rowsum
    elif normalize == "binary":
        A = (A > 0).astype(np.float64)

    # Self-loops
    return A + np.eye(N)


def build_geographic_adjacency(N, coords=None, k=3):
    """
    Build k-NN geographic adjacency.

    Parameters
    ----------
    N : int
        Number of cities.
    coords : (N, 2) ndarray or None
        lon/lat coordinates. Falls back to uniform if None.
    k : int
        Number of neighbours.

    Returns
    -------
    A_geo : (N, N) ndarray with self-loops.
    """
    A = np.zeros((N, N))
    if coords is not None:
        knn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="euclidean")
        knn.fit(coords)
        dist, idx = knn.kneighbors(coords)
        for i in range(N):
            for jj in range(1, min(k + 1, N)):
                j = idx[i, jj]
                w = 1.0 / (1.0 + dist[i, jj])
                A[i, j] = w
                A[j, i] = w
    else:
        A = np.ones((N, N)) - np.eye(N)
    return A + np.eye(N)


# ---------------------------------------------------------------------------
# Feature loading (from .npy, not CSV)
# ---------------------------------------------------------------------------

# Column indices in X_city.npy (18 columns total)
LEAKAGE_COLS = list(range(0, 6))   # 0–5: baseline, shock, recovery, FR, RR, recov_time
STRUCT_COLS  = list(range(6, 18))  # 6–17: network + POI features
CONSTANT_COL = 9                    # degree (same for all JJJ cities)

STRUCT_FEATURE_NAMES = [
    "in_strength", "out_strength", "total_strength",
    "degree", "betweenness_centrality",
    "poi_count_2017", "poi_count_2019", "poi_count_2023",
    "shannon_2017", "shannon_2019", "shannon_2023",
    "poi_total_all_years",
]

# Admin level mapping (complete, 93-city coverage)
# Level 4: Direct-controlled municipalities (直辖市)
# Level 3: Sub-provincial cities (副省级) + Separately-planned cities (计划单列市) + Provincial capitals (省会)
# Level 2: Prefecture-level cities (地级市)
# Note: No county-level cities in our 4-agglomeration sample
_DIRECT_MUNICIPALITIES = {"北京市", "天津市", "上海市", "重庆市"}
_SUB_PROVINCIAL_AND_CAPITALS = {
    # 副省级城市 (subset within our 4 agglomerations)
    "南京市", "杭州市", "宁波市", "广州市", "深圳市", "成都市",
    # 省会 (not already sub-provincial within our sample)
    "石家庄市", "合肥市",
}

def get_admin_level(city_name):
    """Return admin level for any Chinese prefecture-level or above city."""
    if city_name in _DIRECT_MUNICIPALITIES:
        return 4
    if city_name in _SUB_PROVINCIAL_AND_CAPITALS:
        return 3
    # All other prefecture-level cities: default level 2
    return 2

# Complete 93-city mapping (for direct dict lookup)
ADMIN_LEVEL = {
    # JJJ (13)
    "北京市": 4, "天津市": 4,
    "石家庄市": 3, "唐山市": 2, "保定市": 2, "邯郸市": 2,
    "张家口市": 2, "承德市": 2, "廊坊市": 2, "沧州市": 2,
    "衡水市": 2, "邢台市": 2, "秦皇岛市": 2,
    # YRD (41)
    "上海市": 4,
    "南京市": 3, "杭州市": 3, "合肥市": 3, "宁波市": 3,
    "无锡市": 2, "徐州市": 2, "常州市": 2, "苏州市": 2,
    "南通市": 2, "连云港市": 2, "淮安市": 2, "盐城市": 2,
    "扬州市": 2, "镇江市": 2, "泰州市": 2, "宿迁市": 2,
    "温州市": 2, "嘉兴市": 2, "湖州市": 2, "绍兴市": 2,
    "金华市": 2, "衢州市": 2, "舟山市": 2, "台州市": 2,
    "丽水市": 2, "芜湖市": 2, "蚌埠市": 2, "淮南市": 2,
    "马鞍山市": 2, "淮北市": 2, "铜陵市": 2, "安庆市": 2,
    "黄山市": 2, "滁州市": 2, "阜阳市": 2, "宿州市": 2,
    "六安市": 2, "亳州市": 2, "池州市": 2, "宣城市": 2,
    # PRD (21)
    "广州市": 3, "深圳市": 3,
    "韶关市": 2, "珠海市": 2, "汕头市": 2, "佛山市": 2,
    "江门市": 2, "湛江市": 2, "茂名市": 2, "肇庆市": 2,
    "惠州市": 2, "梅州市": 2, "汕尾市": 2, "河源市": 2,
    "阳江市": 2, "清远市": 2, "东莞市": 2, "中山市": 2,
    "潮州市": 2, "揭阳市": 2, "云浮市": 2,
    # CD_CQ (18)
    "重庆市": 4,
    "成都市": 3,
    "自贡市": 2, "攀枝花市": 2, "泸州市": 2, "德阳市": 2,
    "绵阳市": 2, "广元市": 2, "遂宁市": 2, "内江市": 2,
    "乐山市": 2, "南充市": 2, "眉山市": 2, "宜宾市": 2,
    "广安市": 2, "达州市": 2, "雅安市": 2, "资阳市": 2,
}


def load_features_from_npy(base_dir, cities=None, drop_constant=True,
                           add_admin_level=True):
    """
    Load clean features directly from X_city.npy.

    Supports both:
    - Old format (18 columns): first 6 are leakage (FR, RR, etc.), cols 6-17 structural
    - New format (N columns): all columns are structural features

    Returns
    -------
    X : (N, D) standardized feature array
    feature_names : list of str
    """
    X_full = np.load(Path(base_dir) / "X_city.npy")
    N_cols = X_full.shape[1]

    # Detect format: old format (18 cols with 6 leakage cols) vs new format (≤20 cols, all structural)
    if N_cols == 18:
        # Old format — slice structural columns (6–17)
        X = X_full[:, STRUCT_COLS].copy()
        feature_names = list(STRUCT_FEATURE_NAMES)
    else:
        # New format (9-14+ cols) — all columns are structural
        # Try to read feature_names.txt if available
        names_path = Path(base_dir) / "feature_names.txt"
        if names_path.exists():
            with open(names_path, 'r', encoding='utf-8') as f:
                feature_names = [line.strip() for line in f if line.strip()]
            feature_names = feature_names[:N_cols]
        else:
            # Default names based on column count
            if N_cols == 14:
                feature_names = [
                    "in_strength", "out_strength", "betweenness_centrality",
                    "gdp_log_2019", "gdp_pc_log_2019", "trend_growth_17_19",
                    "primary_pct_2019", "secondary_pct_2019", "tertiary_pct_2019", "industry_entropy",
                    "poi_density", "poi_shannon",
                    "ntl_log_2019", "admin_level",
                ]
            else:
                feature_names = [f"feat_{j}" for j in range(N_cols)]
        X = X_full.copy()
    # (old format handled above)

    # Fill NaN with column median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        if nan_mask[:, j].any():
            X[nan_mask[:, j], j] = col_medians[j]

    # Drop constant columns
    if drop_constant:
        keep = []
        for j in range(X.shape[1]):
            if np.std(X[:, j]) > 1e-10:
                keep.append(j)
        X = X[:, keep]
        feature_names = [feature_names[j] for j in keep]

    # Add admin_level if not already present
    if add_admin_level and cities is not None and "admin_level" not in feature_names:
        admin = np.array([ADMIN_LEVEL.get(c, 2) for c in cities], dtype=np.float64)
        X = np.column_stack([X, admin])
        feature_names.append("admin_level")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, feature_names


def load_target_from_npy(base_dir):
    """Load y_city.npy (RI values)."""
    y = np.load(Path(base_dir) / "y_city.npy")
    return y


def load_city_names(base_dir):
    """Extract city names from features CSV (only the 'city' column)."""
    csv_path = Path(base_dir) / "city_metadata.csv"
    df = pd.read_csv(csv_path)
    return list(df["city"])


def load_agglomeration(data_dir, edge_norm="minmax", topk=None,
                       geo_k=3, drop_constant=True):
    """
    Load all data for one urban agglomeration.

    Returns
    -------
    A_od : (N, N) ndarray
    A_geo : (N, N) ndarray
    X : (N, D) ndarray — clean structural features (standardized)
    y : (N, 1) ndarray — resilience index
    cities : list of str
    feature_names : list of str
    """
    data_dir = Path(data_dir)

    cities = load_city_names(data_dir)
    N = len(cities)

    # Adjacency
    A_od = load_adjacency(data_dir / "A_city.npy", normalize=edge_norm, topk=topk)

    # Geographic adjacency (try loading metadata for coordinates)
    meta_path = data_dir / "city_metadata.csv"
    coords = None
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "lon" in meta.columns and "lat" in meta.columns:
            coords = meta[["lon", "lat"]].values
    A_geo = build_geographic_adjacency(N, coords, k=geo_k)

    # Features (clean, from .npy)
    X, feature_names = load_features_from_npy(
        data_dir, cities=cities, drop_constant=drop_constant,
        add_admin_level=True
    )

    # Target
    y = load_target_from_npy(data_dir)

    return A_od, A_geo, X, y, cities, feature_names
