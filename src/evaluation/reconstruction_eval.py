from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def reconstruction_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0,
) -> float:
    """
    Token-level reconstruction accuracy (ignoring padding).

    Args:
        logits:  (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        pad_idx: Padding token index to ignore.

    Returns:
        Accuracy as a float in [0, 1].
    """
    predictions = logits.argmax(dim=-1)
    mask = targets != pad_idx
    correct = (predictions == targets) & mask
    return (correct.sum() / mask.sum()).item()


def perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0,
) -> float:
    """
    Compute token-level perplexity from logits.

    Args:
        logits:  (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        pad_idx: Padding token index.

    Returns:
        Perplexity value.
    """
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    nll = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=pad_idx,
        reduction="mean",
    )
    return math.exp(nll.item())


def corpus_bleu_score(
    hypotheses: List[List[str]],
    references: List[List[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU score.

    Args:
        hypotheses:  List of predicted token lists.
        references:  List of reference token lists.
        max_n:       Maximum n-gram order.

    Returns:
        Dict with bleu_1 through bleu_4.
    """
    smoothing = SmoothingFunction().method1
    refs_wrapped = [[ref] for ref in references]

    results = {}
    for n in range(1, max_n + 1):
        weights = tuple([1.0 / n] * n + [0.0] * (max_n - n))
        score = corpus_bleu(
            refs_wrapped,
            hypotheses,
            weights=weights,
            smoothing_function=smoothing,
        )
        results[f"bleu_{n}"] = float(score)

    return results


def evaluate_reconstruction(
    model,
    loader,
    device: torch.device,
    vocab,
    pad_idx: int = 0,
) -> Dict[str, float]:
    """
    Full reconstruction evaluation over a DataLoader.
    Computes metrics incrementally to avoid OOM.
    """
    from src.utils.cuda_utils import move_to_device

    model.eval()

    total_correct = 0
    total_tokens = 0
    total_nll = 0.0
    total_batches = 0
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            token_ids = batch["token_ids"]
            lengths = batch["lengths"]

            output = model(token_ids, lengths)

            # Work on CPU immediately to free GPU memory
            logits = output["logits"].float().cpu()
            targets = token_ids[:, 1:].cpu()

            # Accuracy incrementally
            predictions = logits.argmax(dim=-1)
            mask = targets != pad_idx
            total_correct += ((predictions == targets) & mask).sum().item()
            total_tokens += mask.sum().item()

            # NLL incrementally
            batch_size, seq_len, vocab_size = logits.shape
            nll = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=pad_idx,
                reduction="sum",
            )
            total_nll += nll.item()
            total_batches += mask.sum().item()

            # BLEU — only keep decoded strings, not tensors
            for i in range(predictions.size(0)):
                hyp = vocab.decode(predictions[i].tolist(), remove_special=True)
                ref = vocab.decode(targets[i].tolist(), remove_special=True)
                all_hypotheses.append(hyp)
                all_references.append(ref)

            # Free memory explicitly
            del logits, targets, output
            torch.cuda.empty_cache()

    acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(total_nll / total_batches) if total_batches > 0 else float("inf")
    bleu = corpus_bleu_score(all_hypotheses, all_references)

    return {"accuracy": acc, "perplexity": ppl, **bleu}

'''def evaluate_reconstruction(
    model,
    loader,
    device: torch.device,
    vocab,
    pad_idx: int = 0,
) -> Dict[str, float]:
    """
    Full reconstruction evaluation over a DataLoader.

    Args:
        model:   Trained MorphoSyntaxVAE.
        loader:  DataLoader for the evaluation split.
        device:  Computation device.
        vocab:   Vocabulary for decoding.
        pad_idx: Padding index.

    Returns:
        Dict with accuracy, perplexity, and BLEU scores.
    """
    import numpy as np
    from src.utils.cuda_utils import move_to_device

    model.eval()
    all_logits = []
    all_targets = []
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            token_ids = batch["token_ids"]
            lengths = batch["lengths"]

            output = model(token_ids, lengths)
            logits = output["logits"]
            targets = token_ids[:, 1:]

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

            preds = logits.argmax(dim=-1)
            for i in range(preds.size(0)):
                hyp = vocab.decode(preds[i].tolist(), remove_special=True)
                ref = vocab.decode(targets[i].tolist(), remove_special=True)
                all_hypotheses.append(hyp)
                all_references.append(ref)

    all_logits_cat = torch.cat(all_logits, dim=0)
    all_targets_cat = torch.cat(all_targets, dim=0)

    acc = reconstruction_accuracy(all_logits_cat, all_targets_cat, pad_idx)
    ppl = perplexity(all_logits_cat, all_targets_cat, pad_idx)
    bleu = corpus_bleu_score(all_hypotheses, all_references)

    return {"accuracy": acc, "perplexity": ppl, **bleu}'''