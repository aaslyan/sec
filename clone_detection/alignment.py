"""
Smith-Waterman local sequence alignment (pure numpy) for opcode sequences.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from clone_detection.lsh import CandidatePair

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    raw_score: float
    normalized_score: float   # in [0, 1]
    func_id1: str
    func_id2: str
    aligned_len: int


def smith_waterman(
    seq1: List[str],
    seq2: List[str],
    match: int = 2,
    mismatch: int = -1,
    gap: int = -1,
) -> float:
    """
    Smith-Waterman local alignment on opcode sequences.
    Returns the maximum cell value in the DP matrix.
    """
    m, n = len(seq1), len(seq2)
    # Use a single previous row to save memory
    prev = np.zeros(n + 1, dtype=np.float32)
    max_score = 0.0

    for i in range(1, m + 1):
        curr = np.zeros(n + 1, dtype=np.float32)
        for j in range(1, n + 1):
            diag = prev[j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)
            up   = prev[j] + gap
            left = curr[j - 1] + gap
            val  = max(0.0, float(diag), float(up), float(left))
            curr[j] = val
            if val > max_score:
                max_score = val
        prev = curr

    return max_score


def compute_alignment(
    func_id1: str,
    func_id2: str,
    seq1: List[str],
    seq2: List[str],
    min_length: int = 5,
) -> Optional[AlignmentResult]:
    """
    Compute Smith-Waterman alignment and normalize.
    Returns None if either sequence is too short.
    """
    if len(seq1) < min_length or len(seq2) < min_length:
        return None

    raw = smith_waterman(seq1, seq2)
    # Perfect match score for the shorter sequence (match=2 per token)
    max_possible = 2.0 * min(len(seq1), len(seq2))
    normalized = float(raw) / max_possible if max_possible > 0 else 0.0
    normalized = min(1.0, normalized)

    return AlignmentResult(
        raw_score=float(raw),
        normalized_score=normalized,
        func_id1=func_id1,
        func_id2=func_id2,
        aligned_len=min(len(seq1), len(seq2)),
    )


class CandidateAligner:
    """Align all candidate pairs using Smith-Waterman."""

    def __init__(self, binaries):
        # Build seq_map from binaries
        self._seq_map: Dict[str, List[str]] = {}
        for b in binaries:
            for f in b.functions:
                fid = f"{b.name}|{f.name}"
                self._seq_map[fid] = f.opcode_sequence

    def align_candidates(
        self,
        candidates: List[CandidatePair],
        seq_map: Optional[Dict[str, List[str]]] = None,
        show_progress: bool = True,
    ) -> List[AlignmentResult]:
        """Align all candidate pairs; skip missing / too-short functions."""
        sm = seq_map if seq_map is not None else self._seq_map
        results: List[AlignmentResult] = []

        total = len(candidates)
        log_every = max(1, total // 10)

        for idx, cp in enumerate(candidates):
            if show_progress and idx % log_every == 0:
                logger.info(f"Aligning candidates: {idx}/{total}")

            seq1 = sm.get(cp.func_id1)
            seq2 = sm.get(cp.func_id2)
            if seq1 is None or seq2 is None:
                continue

            result = compute_alignment(cp.func_id1, cp.func_id2, seq1, seq2)
            if result is not None:
                results.append(result)

        logger.info(f"Alignment complete: {len(results)}/{total} pairs aligned")
        return results
