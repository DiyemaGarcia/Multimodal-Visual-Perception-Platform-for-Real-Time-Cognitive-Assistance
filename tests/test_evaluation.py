import pytest
import torch
from src.evaluation.morpho_eval import evaluate_morpho_segmentation, boundary_f1
from src.evaluation.syntax_eval import (
    unlabeled_attachment_score,
    labeled_attachment_score,
    DependencyArc,
)
from src.evaluation.reconstruction_eval import reconstruction_accuracy, perplexity


def test_boundary_f1_perfect():
    pred = ["run", "ning"]
    gold = ["run", "ning"]
    result = boundary_f1(pred, gold)
    assert result["f1"] == 1.0


def test_boundary_f1_wrong():
    pred = ["runn", "ing"]
    gold = ["run", "ning"]
    result = boundary_f1(pred, gold)
    assert result["f1"] < 1.0


def test_evaluate_morpho_segmentation():
    preds = {"running": ["run", "ning"], "cats": ["cat", "s"]}
    gold  = {"running": ["run", "ning"], "cats": ["cat", "s"]}
    result = evaluate_morpho_segmentation(preds, gold)
    assert result["f1"] == 1.0
    assert result["exact_match"] == 1.0


def test_uas_perfect():
    arcs = [DependencyArc(2, 1, "det"), DependencyArc(0, 2, "root")]
    score = unlabeled_attachment_score(arcs, arcs)
    assert score == 1.0


def test_las_imperfect():
    gold = [DependencyArc(2, 1, "det"), DependencyArc(0, 2, "root")]
    pred = [DependencyArc(2, 1, "nsubj"), DependencyArc(0, 2, "root")]
    score = labeled_attachment_score(pred, gold)
    assert score == 0.5


def test_reconstruction_accuracy():
    vocab_size = 50
    logits = torch.zeros(2, 5, vocab_size)
    targets = torch.zeros(2, 5, dtype=torch.long)
    # Set logits so argmax = targets
    logits[:, :, 0] = 10.0
    acc = reconstruction_accuracy(logits, targets, pad_idx=1)
    assert acc == 1.0


def test_perplexity_finite():
    logits = torch.randn(4, 9, 200)
    targets = torch.randint(1, 200, (4, 9))
    ppl = perplexity(logits, targets, pad_idx=0)
    assert ppl > 0.0
    assert not (ppl != ppl)  # not NaN