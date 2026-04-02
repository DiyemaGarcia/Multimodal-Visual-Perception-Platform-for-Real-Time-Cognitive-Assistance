from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_completeness_v_measure,
)
from typing import Dict, List, Optional, Tuple


def kmeans_clustering(
    z: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """
    Fit K-Means on the latent space.

    Args:
        z:           (N, z_dim) latent vectors.
        n_clusters:  Number of clusters.
        n_init:      Number of initialisations.
        random_state: Random seed.

    Returns:
        labels:  (N,) cluster assignments.
        model:   Fitted KMeans object.
    """
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = model.fit_predict(z)
    return labels, model


def gmm_clustering(
    z: np.ndarray,
    n_components: int,
    covariance_type: str = "full",
    random_state: int = 42,
) -> Tuple[np.ndarray, GaussianMixture]:
    """
    Fit a Gaussian Mixture Model on the latent space.

    Args:
        z:               (N, z_dim) latent vectors.
        n_components:    Number of mixture components.
        covariance_type: Type of covariance matrix.
        random_state:    Random seed.

    Returns:
        labels:  (N,) component assignments.
        model:   Fitted GaussianMixture object.
    """
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    model.fit(z)
    labels = model.predict(z)
    return labels, model


def evaluate_clustering(
    predicted_labels: np.ndarray,
    true_labels: List[str],
    z: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate clustering quality against gold linguistic labels.

    Args:
        predicted_labels: (N,) integer cluster assignments.
        true_labels:      (N,) string gold labels (e.g., UPOS tags).
        z:                Optional latent matrix for silhouette score.

    Returns:
        Dictionary of clustering evaluation metrics.
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)

    ari = adjusted_rand_score(true_encoded, predicted_labels)
    nmi = normalized_mutual_info_score(true_encoded, predicted_labels)
    hom, comp, v_meas = homogeneity_completeness_v_measure(
        true_encoded, predicted_labels
    )

    metrics = {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
        "homogeneity": float(hom),
        "completeness": float(comp),
        "v_measure": float(v_meas),
    }

    if z is not None and len(np.unique(predicted_labels)) > 1:
        sil = silhouette_score(z, predicted_labels, sample_size=min(5000, len(z)))
        metrics["silhouette_score"] = float(sil)

    return metrics


def sweep_cluster_counts(
    z: np.ndarray,
    true_labels: List[str],
    n_clusters_list: List[int],
    method: str = "kmeans",
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate clustering across multiple values of k.

    Args:
        z:               (N, z_dim) latent vectors.
        true_labels:     (N,) gold string labels.
        n_clusters_list: List of k values to try.
        method:          "kmeans" or "gmm".

    Returns:
        Dict mapping k -> evaluation metrics.
    """
    results = {}
    for k in n_clusters_list:
        if method == "kmeans":
            labels, _ = kmeans_clustering(z, n_clusters=k)
        elif method == "gmm":
            labels, _ = gmm_clustering(z, n_components=k)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        results[k] = evaluate_clustering(labels, true_labels, z)
    return results