"""
LSH candidate pairing using datasketch MinHashLSH.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


@dataclass
class CandidatePair:
    func_id1: str
    func_id2: str
    jaccard_estimate: float
    k: int


class LSHCandidateFinder:
    """Find candidate clone pairs via Locality-Sensitive Hashing."""

    def __init__(self, thresholds: List[float] = None, num_perm: int = 128):
        self.thresholds = thresholds if thresholds is not None else [0.5, 0.7, 0.9]
        self.num_perm = num_perm

    def find_candidates(
        self,
        signatures: Dict[str, Dict[int, MinHash]],
        k: int,
    ) -> List[CandidatePair]:
        """
        For a given shingle size k, query all LSH thresholds and collect
        candidate pairs with their Jaccard estimates.
        """
        # Only keep functions that have a signature for this k
        func_ids = [fid for fid, sigs in signatures.items() if k in sigs]
        if len(func_ids) < 2:
            return []

        seen: set = set()
        candidates: List[CandidatePair] = []

        for threshold in self.thresholds:
            lsh = MinHashLSH(threshold=threshold, num_perm=self.num_perm)
            # Insert all signatures
            for fid in func_ids:
                try:
                    lsh.insert(fid, signatures[fid][k])
                except Exception:
                    pass  # duplicate key, skip

            # Query each function
            for fid in func_ids:
                try:
                    neighbors = lsh.query(signatures[fid][k])
                except Exception:
                    continue
                for nbr in neighbors:
                    if nbr == fid:
                        continue
                    pair_key = (min(fid, nbr), max(fid, nbr))
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    jaccard = signatures[fid][k].jaccard(signatures[nbr][k])
                    candidates.append(
                        CandidatePair(
                            func_id1=pair_key[0],
                            func_id2=pair_key[1],
                            jaccard_estimate=float(jaccard),
                            k=k,
                        )
                    )

        return candidates

    def filter_self_pairs(self, candidates: List[CandidatePair]) -> List[CandidatePair]:
        """Remove pairs from the same function name in the same binary."""
        return [
            cp for cp in candidates
            if cp.func_id1.split('|')[0] != cp.func_id2.split('|')[0]
            or cp.func_id1.split('|', 1)[1] != cp.func_id2.split('|', 1)[1]
        ]
