from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, List, Tuple

from src.models.vae import MorphoSyntaxVAE
from src.utils.cuda_utils import move_to_device


def extract_latent_representations(
    model: MorphoSyntaxVAE,
    loader: DataLoader,
    device: torch.device,
    use_mean: bool = True,
) -> Tuple[np.ndarray, List[List[str]], List[List[Dict]]]:
    """
    Run the encoder on a dataset and collect latent vectors
    along with their linguistic annotations.

    Args:
        model:    Trained VAE.
        loader:   DataLoader over annotated data.
        device:   Computation device.
        use_mean: If True, use mu directly; otherwise sample z.

    Returns:
        z_all:       (N, z_dim) array of latent vectors.
        upos_all:    List of UPOS tag lists per sentence.
        morph_all:   List of morphological feature dicts per sentence.
    """
    model.eval()
    z_list = []
    upos_list = []
    morph_list = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            token_ids = batch["token_ids"]
            lengths = batch["lengths"]

            mu, logvar = model.encode(token_ids, lengths)
            z = mu if use_mean else model.latent_space.reparameterise(mu, logvar)

            z_list.append(z.cpu().numpy())
            upos_list.extend(batch.get("upos_tags", [[] for _ in range(z.size(0))]))
            morph_list.extend(batch.get("morph_feats", [[] for _ in range(z.size(0))]))

    z_all = np.concatenate(z_list, axis=0)
    return z_all, upos_list, morph_list


def probe_latent_space(
    z: np.ndarray,
    labels: List[str],
    n_folds: int = 5,
    max_iter: int = 1000,
) -> Dict[str, float]:
    """
    Train a logistic regression probe on latent representations
    to predict a linguistic label (e.g., UPOS tag).

    Args:
        z:        (N, z_dim) latent vectors.
        labels:   (N,) string labels to predict.
        n_folds:  Number of cross-validation folds.
        max_iter: Maximum iterations for logistic regression.

    Returns:
        Dictionary with mean accuracy and macro F1.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []

    for train_idx, test_idx in skf.split(z, y):
        clf = LogisticRegression(
            max_iter=max_iter,
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
        )
        clf.fit(z[train_idx], y[train_idx])
        y_pred = clf.predict(z[test_idx])
        accuracies.append(accuracy_score(y[test_idx], y_pred))
        f1_scores.append(f1_score(y[test_idx], y_pred, average="macro"))

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "f1_macro_mean": float(np.mean(f1_scores)),
        "f1_macro_std": float(np.std(f1_scores)),
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
    }


def probe_subspaces(
    z: np.ndarray,
    morpho_dim: int,
    labels_dict: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """
    Probe both the morphological and syntactic subspaces
    against multiple linguistic targets.

    Args:
        z:          Full latent matrix (N, z_dim).
        morpho_dim: Dimension boundary between morpho and syntax subspaces.
        labels_dict: Dict of target_name -> list of string labels.

    Returns:
        Nested dict: subspace -> target -> metrics.
    """
    z_morpho = z[:, :morpho_dim]
    z_syntax = z[:, morpho_dim:]

    results = {"morpho": {}, "syntax": {}, "full": {}}

    for target_name, labels in labels_dict.items():
        if len(set(labels)) < 2:
            continue
        results["morpho"][target_name] = probe_latent_space(z_morpho, labels)
        results["syntax"][target_name] = probe_latent_space(z_syntax, labels)
        results["full"][target_name] = probe_latent_space(z, labels)

    return results