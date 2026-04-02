from __future__ import annotations
import re
from typing import Dict, List, Set, Tuple


def _get_morpheme_boundaries(segmentation: List[str]) -> Set[int]:
    """
    Convert a morpheme list to a set of boundary positions
    (character offsets within the concatenated word).

    Example: ["run", "ning"] → {3}  (boundary after position 3)
    """
    boundaries = set()
    pos = 0
    for morpheme in segmentation[:-1]:
        pos += len(morpheme)
        boundaries.add(pos)
    return boundaries


def boundary_f1(
    predicted: List[str],
    gold: List[str],
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 on morpheme boundary positions.

    Args:
        predicted: Predicted segmentation (list of morpheme strings).
        gold:      Gold segmentation (list of morpheme strings).

    Returns:
        Dict with precision, recall, f1.
    """
    pred_boundaries = _get_morpheme_boundaries(predicted)
    gold_boundaries = _get_morpheme_boundaries(gold)

    tp = len(pred_boundaries & gold_boundaries)
    fp = len(pred_boundaries - gold_boundaries)
    fn = len(gold_boundaries - pred_boundaries)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_morpho_segmentation(
    predictions: Dict[str, List[str]],
    gold: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Evaluate morphological segmentation over a full lexicon.

    Args:
        predictions: Dict of word -> predicted morpheme list.
        gold:        Dict of word -> gold morpheme list.

    Returns:
        Aggregated precision, recall, F1 over all shared words.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_exact = 0
    n_total = 0

    shared_words = set(predictions.keys()) & set(gold.keys())

    for word in shared_words:
        pred_seg = predictions[word]
        gold_seg = gold[word]

        pred_b = _get_morpheme_boundaries(pred_seg)
        gold_b = _get_morpheme_boundaries(gold_seg)

        total_tp += len(pred_b & gold_b)
        total_fp += len(pred_b - gold_b)
        total_fn += len(gold_b - pred_b)

        if pred_seg == gold_seg:
            n_exact += 1
        n_total += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    exact_match = n_exact / n_total if n_total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "n_evaluated": n_total,
        "coverage": n_total / max(len(gold), 1),
    }