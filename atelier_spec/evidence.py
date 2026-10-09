"""Evidence labels (atelier-evidence/1.0): where a number came from, weakest first."""

from __future__ import annotations

STANDARD = "atelier-evidence/1.0"

# weakest to strongest; a number built from several records carries the weakest label
LABELS = ("estimate", "datasheet", "rtl-sim", "synthesis", "post-layout", "transistor-sim", "silicon")
_RANK = {label: i for i, label in enumerate(LABELS)}


def weakest(labels) -> str:
    labels = list(labels)
    for label in labels:
        if label not in _RANK:
            raise ValueError(f"evidence {label!r} is not a label of {STANDARD}")
    if not labels:
        raise ValueError("no evidence")
    return min(labels, key=_RANK.__getitem__)


def leaves(section: dict, prefix: str = ""):
    """(key, entry) for every entry of a timing or cost section that carries a value or an expression."""
    for k, v in section.items():
        if isinstance(v, dict) and ("value" in v or "expr" in v):
            yield prefix + k, v
        elif isinstance(v, dict):
            yield from leaves(v, prefix + k + ".")
