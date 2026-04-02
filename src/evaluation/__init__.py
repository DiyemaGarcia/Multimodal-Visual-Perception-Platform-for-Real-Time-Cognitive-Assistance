"""
src.evaluation
--------------
Evaluation metrics for the morphosyntactic VAE.

Modules:
    morpho_eval          — Morpheme boundary F1 and segmentation accuracy
    syntax_eval          — UAS, LAS, and POS tagging accuracy
    reconstruction_eval  — Reconstruction accuracy, perplexity, BLEU
"""

from src.evaluation.morpho_eval import evaluate_morpho_segmentation, boundary_f1
from src.evaluation.syntax_eval import (
    unlabeled_attachment_score,
    labeled_attachment_score,
    evaluate_syntax,
    pos_accuracy,
    DependencyArc,
)
from src.evaluation.reconstruction_eval import (
    reconstruction_accuracy,
    perplexity,
    corpus_bleu_score,
    evaluate_reconstruction,
)

__all__ = [
    "evaluate_morpho_segmentation",
    "boundary_f1",
    "unlabeled_attachment_score",
    "labeled_attachment_score",
    "evaluate_syntax",
    "pos_accuracy",
    "DependencyArc",
    "reconstruction_accuracy",
    "perplexity",
    "corpus_bleu_score",
    "evaluate_reconstruction",
]