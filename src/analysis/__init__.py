"""
src.analysis
------------
Post-training analysis of latent representations.

Modules:
    latent_probing    — Linear probing classifiers on latent subspaces
    clustering        — K-Means and GMM clustering with linguistic evaluation
    correlation       — Spearman/Pearson correlation between z and linguistic features
    disentanglement   — Beta-VAE score, MIG, and DCI metrics
    visualization     — t-SNE, UMAP, PCA scatter plots and heatmaps
"""

from src.analysis.latent_probing import extract_latent_representations, probe_subspaces
from src.analysis.clustering import kmeans_clustering, gmm_clustering, sweep_cluster_counts
from src.analysis.correlation import correlation_matrix, subspace_correlation_summary
from src.analysis.disentanglement import mutual_information_gap, dci_score
from src.analysis.visualization import (
    reduce_tsne,
    reduce_umap,
    reduce_pca,
    plot_latent_scatter,
    plot_correlation_heatmap,
    plot_training_curves,
    plot_latent_traversal,
)

__all__ = [
    "extract_latent_representations",
    "probe_subspaces",
    "kmeans_clustering",
    "gmm_clustering",
    "sweep_cluster_counts",
    "correlation_matrix",
    "subspace_correlation_summary",
    "mutual_information_gap",
    "dci_score",
    "reduce_tsne",
    "reduce_umap",
    "reduce_pca",
    "plot_latent_scatter",
    "plot_correlation_heatmap",
    "plot_training_curves",
    "plot_latent_traversal",
]