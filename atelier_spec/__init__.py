"""The Atelier standards: block contracts, machine descriptions, the op set and evidence labels."""

from atelier_spec.block import STANDARD as BLOCK_STANDARD
from atelier_spec.evidence import STANDARD as EVIDENCE_STANDARD
from atelier_spec.machine import STANDARD as MACHINE_STANDARD

OPS_STANDARD = "atelier-ops/0.1"
STANDARDS = (BLOCK_STANDARD, MACHINE_STANDARD, OPS_STANDARD, EVIDENCE_STANDARD)

__version__ = "0.1.0"
