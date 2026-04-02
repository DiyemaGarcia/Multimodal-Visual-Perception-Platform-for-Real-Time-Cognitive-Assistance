from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional


@dataclass
class Token:
    """Represents a single token in a CoNLL-U sentence."""
    id: str                         # Token ID (can be range "1-2" for MWTs)
    form: str                       # Surface form
    lemma: str                      # Lemma
    upos: str                       # Universal POS tag
    xpos: str                       # Language-specific POS tag
    feats: Dict[str, str]           # Morphological features
    head: Optional[int]             # Dependency head index
    deprel: str                     # Dependency relation label
    deps: str                       # Enhanced dependencies
    misc: str                       # Miscellaneous

    @property
    def is_multiword(self) -> bool:
        return "-" in str(self.id)

    @property
    def is_empty(self) -> bool:
        return "." in str(self.id)


@dataclass
class Sentence:
    """Represents a full CoNLL-U annotated sentence."""
    sent_id: Optional[str]
    text: Optional[str]
    tokens: List[Token] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def forms(self) -> List[str]:
        return [t.form for t in self.tokens if not t.is_multiword and not t.is_empty]

    @property
    def upos_tags(self) -> List[str]:
        return [t.upos for t in self.tokens if not t.is_multiword and not t.is_empty]

    @property
    def deprels(self) -> List[str]:
        return [t.deprel for t in self.tokens if not t.is_multiword and not t.is_empty]

    @property
    def morph_feats(self) -> List[Dict[str, str]]:
        return [t.feats for t in self.tokens if not t.is_multiword and not t.is_empty]


def _parse_feats(feats_str: str) -> Dict[str, str]:
    """Parse the FEATS field into a key-value dictionary."""
    if feats_str == "_":
        return {}
    feats = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            key, val = pair.split("=", 1)
            feats[key] = val
    return feats


def _parse_token(fields: List[str]) -> Token:
    """Parse a list of 10 CoNLL-U fields into a Token object."""
    assert len(fields) == 10, f"Expected 10 fields, got {len(fields)}"
    return Token(
        id=fields[0],
        form=fields[1],
        lemma=fields[2],
        upos=fields[3],
        xpos=fields[4],
        feats=_parse_feats(fields[5]),
        head=int(fields[6]) if fields[6] not in ("_", "") else None,
        deprel=fields[7],
        deps=fields[8],
        misc=fields[9],
    )


def parse_conllu_file(path: str | Path) -> Generator[Sentence, None, None]:
    """
    Parse a CoNLL-U file and yield Sentence objects one at a time.

    Args:
        path: Path to the .conllu file.

    Yields:
        Sentence objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CoNLL-U file not found: {path}")

    current_tokens: List[Token] = []
    current_meta: Dict[str, str] = {}
    sent_id: Optional[str] = None
    text: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#"):
                # Metadata comment
                match = re.match(r"#\s*(\w[\w-]*)\s*=\s*(.*)", line)
                if match:
                    key, val = match.group(1), match.group(2).strip()
                    current_meta[key] = val
                    if key == "sent_id":
                        sent_id = val
                    elif key == "text":
                        text = val
                continue

            if line.strip() == "":
                # End of sentence
                if current_tokens:
                    yield Sentence(
                        sent_id=sent_id,
                        text=text,
                        tokens=current_tokens,
                        metadata=current_meta,
                    )
                current_tokens = []
                current_meta = {}
                sent_id = None
                text = None
                continue

            fields = line.split("\t")
            if len(fields) == 10:
                token = _parse_token(fields)
                current_tokens.append(token)

    # Yield last sentence if file doesn't end with blank line
    if current_tokens:
        yield Sentence(
            sent_id=sent_id,
            text=text,
            tokens=current_tokens,
            metadata=current_meta,
        )


def load_conllu_sentences(path: str | Path) -> List[Sentence]:
    """Load all sentences from a CoNLL-U file into memory."""
    return list(parse_conllu_file(path))


def get_conllu_stats(path: str | Path) -> Dict[str, int]:
    """Compute basic statistics of a CoNLL-U file."""
    n_sents = 0
    n_tokens = 0
    upos_counts: Dict[str, int] = {}

    for sent in parse_conllu_file(path):
        n_sents += 1
        for tag in sent.upos_tags:
            n_tokens += 1
            upos_counts[tag] = upos_counts.get(tag, 0) + 1

    return {
        "sentences": n_sents,
        "tokens": n_tokens,
        **{f"upos_{k}": v for k, v in sorted(upos_counts.items())},
    }