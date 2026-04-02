from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class MorphoEntry:
    """
    Represents a single morphological entry from Morpho Challenge
    or UniMorph format.
    """
    word: str
    segmentation: List[str]       # List of morpheme strings
    features: Dict[str, str] = field(default_factory=dict)  # UniMorph features
    source: str = "morpho_challenge"

    @property
    def morphemes(self) -> List[str]:
        return self.segmentation

    @property
    def n_morphemes(self) -> int:
        return len(self.segmentation)

    def __repr__(self) -> str:
        seg = " | ".join(self.segmentation)
        return f"MorphoEntry(word='{self.word}', seg=[{seg}])"


def parse_morpho_challenge_train(path: str | Path) -> Dict[str, int]:
    """
    Parse the Morpho Challenge training file (word frequency list).

    Real format: frequency word (frequency first, word second)
    Example:
        5395134 a
        878036  running

    Args:
        path: Path to the training file.

    Returns:
        Dictionary mapping word -> frequency.
    """
    path = Path(path)
    word_freqs: Dict[str, int] = {}

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                # Only one token on the line — skip
                continue
            # First column is frequency, second is word
            try:
                freq = int(parts[0])
                word = parts[1]
            except ValueError:
                # First column is not an integer — treat whole line as word
                word = parts[0]
                freq = 1
            word_freqs[word] = freq

    return word_freqs


def parse_morpho_challenge_gold(path: str | Path) -> Dict[str, MorphoEntry]:
    """
    Parse the Morpho Challenge gold standard segmentation file.

    Real format (tab-separated):
        word\tmorpheme:lemma_POS tag:+FEAT, morpheme:lemma_POS tag:+FEAT
    Example:
        ablatives\tablative:ablative_A s:+PL
        abusing\tab:ab_p us:use_V ing:+PCP1

    Morphemes are space-separated. Each morpheme token has the form
    'surface:lemma_POS' or 'tag:+FEAT'. We extract only the surface
    form (part before the colon) as the morpheme string.
    Multiple analyses separated by ', ' — we take the first one.

    Args:
        path: Path to the gold standard file.

    Returns:
        Dictionary mapping word -> MorphoEntry.
    """
    path = Path(path)
    entries: Dict[str, MorphoEntry] = {}

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split word from analysis on tab
            if "\t" not in line:
                continue
            word, analysis_str = line.split("\t", 1)

            # Take only the first analysis if multiple separated by ", "
            first_analysis = analysis_str.split(", ")[0]

            # Each token is "surface:lemma_POS" or "tag:+FEAT"
            morphemes = []
            for token in first_analysis.split():
                # Extract surface form before the colon
                surface = token.split(":")[0]
                # Skip empty surfaces and pure feature tags (starting with +)
                if surface and not surface.startswith("+") and not surface.startswith("~"):
                    morphemes.append(surface)

            if not morphemes:
                morphemes = [word]

            entries[word] = MorphoEntry(
                word=word,
                segmentation=morphemes,
                source="morpho_challenge_gold",
            )

    return entries


def parse_unimorph(path: str | Path) -> Dict[str, List[MorphoEntry]]:
    """
    Parse a UniMorph segmentation file (eng.segmentations format).

    Format (tab-separated, 4 columns):
        lemma    form    features    segmentation
    Example:
        eat    eating    V|V.PTCP;PRS    eat|ing

    Segmentation uses '|' as morpheme boundary marker.
    Features use '|' as separator between POS and feature bundle,
    and ';' as separator within the feature bundle.

    Args:
        path: Path to the en_wiktionary_morph.tsv file.

    Returns:
        Dictionary mapping lemma -> list of MorphoEntry (one per inflected form).
    """
    path = Path(path)
    paradigms: Dict[str, List[MorphoEntry]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            # Require at least 3 columns (lemma, form, features)
            if len(parts) < 3:
                continue

            lemma    = parts[0]
            form     = parts[1]
            feat_str = parts[2]

            # Parse segmentation if 4th column exists and is not "-"
            if len(parts) >= 4 and parts[3] != "-":
                # "eat|ing" → ["eat", "ing"]
                segmentation = parts[3].split("|")
                segmentation = [m for m in segmentation if m]
            else:
                # No segmentation available → use full form as single morpheme
                segmentation = [form]

            # Features: "V|V.PTCP;PRS" → take everything after first "|"
            if "|" in feat_str:
                feat_str = feat_str.split("|", 1)[1]

            features = _parse_unimorph_features(feat_str)

            entry = MorphoEntry(
                word=form,
                segmentation=segmentation,
                features=features,
                source="unimorph",
            )

            if lemma not in paradigms:
                paradigms[lemma] = []
            paradigms[lemma].append(entry)

    return paradigms


def _parse_unimorph_features(feat_str: str) -> Dict[str, str]:
    """
    Parse a UniMorph feature string into a key-value dict.
    UniMorph uses semicolon-separated bundle like: V;V.PTCP;PRS
    We map them to simplified categories.
    """
    feature_map = {
        "PRS": ("tense", "present"),
        "PST": ("tense", "past"),
        "FUT": ("tense", "future"),
        "SG": ("number", "singular"),
        "PL": ("number", "plural"),
        "1": ("person", "1"),
        "2": ("person", "2"),
        "3": ("person", "3"),
        "MASC": ("gender", "masculine"),
        "FEM": ("gender", "feminine"),
        "NEUT": ("gender", "neuter"),
        "NOM": ("case", "nominative"),
        "ACC": ("case", "accusative"),
        "GEN": ("case", "genitive"),
        "DAT": ("case", "dative"),
        "IND": ("mood", "indicative"),
        "SBJV": ("mood", "subjunctive"),
        "IMP": ("mood", "imperative"),
    }
    features: Dict[str, str] = {}
    for tag in feat_str.split(";"):
        tag = tag.strip()
        if tag in feature_map:
            key, val = feature_map[tag]
            features[key] = val
    return features


def get_morpheme_set(entries: Dict[str, MorphoEntry]) -> Set[str]:
    """Extract the set of all unique morphemes from a gold standard dict."""
    morphemes: Set[str] = set()
    for entry in entries.values():
        morphemes.update(entry.segmentation)
    return morphemes