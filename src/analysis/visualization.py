from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def reduce_pca(z: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Reduce dimensionality with PCA."""
    pca = PCA(n_components=min(n_components, z.shape[1]))
    return pca.fit_transform(z)


def reduce_tsne(
    z: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce dimensionality with t-SNE (optionally after PCA pre-reduction)."""
    if z.shape[1] > 50:
        z = reduce_pca(z, n_components=50)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
    )
    return tsne.fit_transform(z)


def reduce_umap(
    z: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce dimensionality with UMAP."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(z)


def plot_latent_scatter(
    z_2d: np.ndarray,
    labels: List[str],
    title: str = "Latent Space",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
) -> plt.Figure:
    """
    Scatter plot of 2D latent representations coloured by label.

    Args:
        z_2d:      (N, 2) reduced latent matrix.
        labels:    (N,) string labels for colouring.
        title:     Plot title.
        save_path: If provided, save the figure here.
        figsize:   Figure dimensions.
        dpi:       Resolution for saving.

    Returns:
        Matplotlib Figure object.
    """
    unique_labels = sorted(set(labels))
    palette = cm.get_cmap("tab20", len(unique_labels))
    label2color = {lbl: palette(i) for i, lbl in enumerate(unique_labels)}
    colors = [label2color[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, alpha=0.5, s=8)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=label2color[lbl], markersize=8, label=lbl
        )
        for lbl in unique_labels
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    categories: List[str],
    top_dims: int = 30,
    title: str = "Latent-Linguistic Correlation",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Heatmap of correlation between latent dimensions and linguistic categories.

    Args:
        corr_matrix: (z_dim, n_categories) correlation matrix.
        categories:  Column labels.
        top_dims:    Show only the top-k most informative dimensions.
        title:       Plot title.
        save_path:   If provided, save the figure here.
        dpi:         Resolution.

    Returns:
        Matplotlib Figure object.
    """
    # Select top dimensions by max absolute correlation across categories
    max_corr = np.abs(corr_matrix).max(axis=1)
    top_idx = np.argsort(max_corr)[::-1][:top_dims]
    sub_matrix = corr_matrix[top_idx, :]

    fig, ax = plt.subplots(figsize=(len(categories) * 1.2, top_dims * 0.4 + 2))
    sns.heatmap(
        sub_matrix,
        xticklabels=categories,
        yticklabels=[f"z_{i}" for i in top_idx],
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot training and validation loss curves over epochs.

    Args:
        history:   Dict of metric_name -> list of values per epoch.
        save_path: If provided, save the figure here.
        dpi:       Resolution.

    Returns:
        Matplotlib Figure object.
    """
    keys = list(history.keys())
    n_plots = len(keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        ax.plot(history[key], label=key)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_latent_traversal(
    values: np.ndarray,
    dim_idx: int,
    label: str = "",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Visualise how a single latent dimension varies across its range.

    Args:
        values:    (N,) values of the latent dimension across the dataset.
        dim_idx:   Index of the latent dimension.
        label:     Linguistic label associated with this dimension.
        save_path: If provided, save the figure.
        dpi:       Resolution.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(values, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_title(f"z_{dim_idx} distribution — correlated with: {label}")
    ax.set_xlabel(f"z_{dim_idx}")
    ax.set_ylabel("Count")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig