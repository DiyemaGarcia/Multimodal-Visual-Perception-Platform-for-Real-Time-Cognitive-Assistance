from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from typing import Dict, List


def beta_vae_score(
    z_fixed: np.ndarray,
    z_varied: np.ndarray,
    factor_idx: int,
    classifier,
) -> float:
    """
    Beta-VAE disentanglement metric (Higgins et al., 2017).
    Measures whether a single latent dimension captures a single factor.

    Args:
        z_fixed:    (N, z_dim) latent vectors with one factor fixed.
        z_varied:   (N, z_dim) latent vectors with same factor varied.
        factor_idx: Index of the generative factor being tested.
        classifier: A trained classifier to predict factor_idx from |z_fixed - z_varied|.

    Returns:
        Accuracy of predicting the fixed factor from the difference vector.
    """
    diff = np.abs(z_fixed - z_varied)
    predictions = classifier.predict(diff)
    accuracy = (predictions == factor_idx).mean()
    return float(accuracy)


def mutual_information_gap(
    z: np.ndarray,
    factors: np.ndarray,
) -> float:
    """
    Mutual Information Gap (MIG) metric (Chen et al., 2018).

    For each generative factor, compute MI with all latent dimensions.
    MIG = mean over factors of (top1_MI - top2_MI) / H(factor).

    Args:
        z:       (N, z_dim) latent matrix.
        factors: (N, n_factors) ground truth factor matrix (integer-encoded).

    Returns:
        MIG score in [0, 1].
    """
    n_factors = factors.shape[1]
    z_dim = z.shape[1]
    mig_scores = []

    # Discretise z for MI estimation
    n_bins = 20
    z_discretised = np.zeros_like(z, dtype=int)
    for i in range(z_dim):
        z_discretised[:, i] = np.digitize(
            z[:, i],
            bins=np.linspace(z[:, i].min(), z[:, i].max(), n_bins),
        )

    for k in range(n_factors):
        factor_k = factors[:, k]
        mi_per_dim = np.array([
            mutual_info_score(factor_k, z_discretised[:, i])
            for i in range(z_dim)
        ])
        sorted_mi = np.sort(mi_per_dim)[::-1]
        h_factor = _entropy(factor_k)
        if h_factor > 0:
            gap = (sorted_mi[0] - sorted_mi[1]) / h_factor
            mig_scores.append(gap)

    return float(np.mean(mig_scores)) if mig_scores else 0.0


def _entropy(labels: np.ndarray) -> float:
    """Compute the empirical entropy of a discrete label array."""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-8)))


def dci_score(
    z_train: np.ndarray,
    z_test: np.ndarray,
    factors_train: np.ndarray,
    factors_test: np.ndarray,
) -> Dict[str, float]:
    """
    DCI (Disentanglement, Completeness, Informativeness) metric
    (Eastwood & Williams, 2018).

    Uses linear regression importance scores per factor.

    Args:
        z_train, z_test:           (N, z_dim) latent matrices.
        factors_train, factors_test: (N, n_factors) factor matrices.

    Returns:
        Dict with keys: disentanglement, completeness, informativeness.
    """
    n_factors = factors_train.shape[1]
    z_dim = z_train.shape[1]

    # Importance matrix R[i, j] = importance of z_i for factor j
    R = np.zeros((z_dim, n_factors))

    for j in range(n_factors):
        reg = LinearRegression()
        reg.fit(z_train, factors_train[:, j])
        R[:, j] = np.abs(reg.coef_)

    # Normalise columns
    col_sums = R.sum(axis=0, keepdims=True) + 1e-8
    R_norm = R / col_sums

    # Disentanglement: each z dimension should predict exactly one factor
    rho = R_norm / (R_norm.sum(axis=1, keepdims=True) + 1e-8)
    disentanglement = 1.0 - _entropy_matrix(rho)

    # Completeness: each factor should be predicted by exactly one z dimension
    phi = R_norm / (R_norm.sum(axis=0, keepdims=True) + 1e-8)
    completeness = 1.0 - _entropy_matrix(phi.T)

    # Informativeness: held-out prediction error
    preds = z_test @ R
    errors = np.abs(preds - factors_test).mean()
    informativeness = 1.0 / (1.0 + errors)

    return {
        "disentanglement": float(disentanglement),
        "completeness": float(completeness),
        "informativeness": float(informativeness),
    }


def _entropy_matrix(matrix: np.ndarray) -> float:
    """Mean entropy over rows of a probability matrix."""
    entropies = -np.sum(matrix * np.log(matrix + 1e-8), axis=1)
    return float(np.mean(entropies))