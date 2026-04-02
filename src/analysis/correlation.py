from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple


def encode_labels(labels: List[str]) -> np.ndarray:
    """Encode string labels to integer codes."""
    le = LabelEncoder()
    return le.fit_transform(labels).astype(float)


def latent_linguistic_correlation(
    z: np.ndarray,
    labels: List[str],
    method: str = "spearman",
) -> np.ndarray:
    """
    Compute correlation between each latent dimension and a linguistic label.

    Args:
        z:       (N, z_dim) latent matrix.
        labels:  (N,) string labels.
        method:  "spearman" or "pearson".

    Returns:
        correlations: (z_dim,) correlation coefficient per dimension.
    """
    y = encode_labels(labels)
    z_dim = z.shape[1]
    correlations = np.zeros(z_dim)

    for i in range(z_dim):
        if method == "spearman":
            corr, _ = spearmanr(z[:, i], y)
        elif method == "pearson":
            corr, _ = pearsonr(z[:, i], y)
        else:
            raise ValueError(f"Unknown method: {method}")
        correlations[i] = corr if not np.isnan(corr) else 0.0

    return correlations


def correlation_matrix(
    z: np.ndarray,
    labels_dict: Dict[str, List[str]],
    method: str = "spearman",
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a full correlation matrix: latent dims x linguistic categories.

    Args:
        z:           (N, z_dim) latent matrix.
        labels_dict: Dict of category_name -> string label list.
        method:      Correlation method.

    Returns:
        matrix:     (z_dim, n_categories) correlation matrix.
        categories: List of category names (column labels).
    """
    categories = list(labels_dict.keys())
    z_dim = z.shape[1]
    matrix = np.zeros((z_dim, len(categories)))

    for j, cat in enumerate(categories):
        matrix[:, j] = latent_linguistic_correlation(
            z, labels_dict[cat], method=method
        )

    return matrix, categories


def top_correlated_dimensions(
    z: np.ndarray,
    labels: List[str],
    top_k: int = 10,
    method: str = "spearman",
) -> List[Tuple[int, float]]:
    """
    Return the top-k most correlated latent dimensions for a given label.

    Args:
        z:       (N, z_dim) latent matrix.
        labels:  (N,) string labels.
        top_k:   Number of top dimensions to return.
        method:  Correlation method.

    Returns:
        List of (dimension_index, correlation_value) sorted by |correlation|.
    """
    corrs = latent_linguistic_correlation(z, labels, method)
    ranked = sorted(enumerate(corrs), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:top_k]


def subspace_correlation_summary(
    z: np.ndarray,
    morpho_dim: int,
    labels_dict: Dict[str, List[str]],
    method: str = "spearman",
) -> Dict[str, Dict[str, float]]:
    """
    Summarise mean absolute correlation per subspace per linguistic category.

    Args:
        z:           (N, z_dim) full latent matrix.
        morpho_dim:  Boundary between morpho and syntax subspaces.
        labels_dict: Linguistic category labels.
        method:      Correlation method.

    Returns:
        Nested dict: subspace -> category -> mean_abs_correlation.
    """
    z_morpho = z[:, :morpho_dim]
    z_syntax = z[:, morpho_dim:]

    results: Dict[str, Dict[str, float]] = {
        "morpho": {},
        "syntax": {},
    }

    for cat, labels in labels_dict.items():
        corr_m = latent_linguistic_correlation(z_morpho, labels, method)
        corr_s = latent_linguistic_correlation(z_syntax, labels, method)
        results["morpho"][cat] = float(np.mean(np.abs(corr_m)))
        results["syntax"][cat] = float(np.mean(np.abs(corr_s)))

    return results