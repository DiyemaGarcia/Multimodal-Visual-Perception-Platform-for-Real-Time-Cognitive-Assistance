from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DependencyArc:
    """Represents a predicted or gold dependency arc."""
    head: int
    dependent: int
    label: str


def unlabeled_attachment_score(
    predicted: List[DependencyArc],
    gold: List[DependencyArc],
) -> float:
    """
    Compute Unlabeled Attachment Score (UAS).
    Fraction of tokens whose head is correctly predicted.

    Args:
        predicted: List of predicted arcs.
        gold:      List of gold arcs.

    Returns:
        UAS as a float in [0, 1].
    """
    assert len(predicted) == len(gold), "Predicted and gold arc lists must have equal length."
    correct = sum(
        1 for p, g in zip(predicted, gold)
        if p.head == g.head
    )
    return correct / len(gold) if gold else 0.0


def labeled_attachment_score(
    predicted: List[DependencyArc],
    gold: List[DependencyArc],
) -> float:
    """
    Compute Labeled Attachment Score (LAS).
    Fraction of tokens whose head AND dependency label are both correct.

    Args:
        predicted: List of predicted arcs.
        gold:      List of gold arcs.

    Returns:
        LAS as a float in [0, 1].
    """
    assert len(predicted) == len(gold)
    correct = sum(
        1 for p, g in zip(predicted, gold)
        if p.head == g.head and p.label == g.label
    )
    return correct / len(gold) if gold else 0.0


def evaluate_syntax(
    predicted_sentences: List[List[DependencyArc]],
    gold_sentences: List[List[DependencyArc]],
) -> Dict[str, float]:
    """
    Evaluate syntactic parsing quality over a full corpus.

    Args:
        predicted_sentences: List of per-sentence arc lists (predicted).
        gold_sentences:      List of per-sentence arc lists (gold).

    Returns:
        Dict with corpus-level UAS and LAS.
    """
    total_tokens = 0
    uas_correct = 0
    las_correct = 0

    for pred_arcs, gold_arcs in zip(predicted_sentences, gold_sentences):
        for p, g in zip(pred_arcs, gold_arcs):
            total_tokens += 1
            if p.head == g.head:
                uas_correct += 1
                if p.label == g.label:
                    las_correct += 1

    uas = uas_correct / total_tokens if total_tokens > 0 else 0.0
    las = las_correct / total_tokens if total_tokens > 0 else 0.0

    return {
        "uas": uas,
        "las": las,
        "n_tokens": total_tokens,
    }


def pos_accuracy(
    predicted_tags: List[List[str]],
    gold_tags: List[List[str]],
) -> Dict[str, float]:
    """
    Token-level POS tagging accuracy.

    Args:
        predicted_tags: List of per-sentence tag lists.
        gold_tags:      List of per-sentence gold tag lists.

    Returns:
        Dict with overall accuracy and per-tag accuracy.
    """
    total = 0
    correct = 0
    per_tag_correct: Dict[str, int] = {}
    per_tag_total: Dict[str, int] = {}

    for pred_sent, gold_sent in zip(predicted_tags, gold_tags):
        for pred, gold in zip(pred_sent, gold_sent):
            total += 1
            per_tag_total[gold] = per_tag_total.get(gold, 0) + 1
            if pred == gold:
                correct += 1
                per_tag_correct[gold] = per_tag_correct.get(gold, 0) + 1

    overall = correct / total if total > 0 else 0.0
    per_tag = {
        tag: per_tag_correct.get(tag, 0) / count
        for tag, count in per_tag_total.items()
    }

    return {"overall_accuracy": overall, "per_tag": per_tag, "n_tokens": total}