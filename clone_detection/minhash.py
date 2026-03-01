"""
MinHash signatures and shingle extraction for function-level clone detection.
"""

import logging
from typing import Dict, List, Set, TYPE_CHECKING

from datasketch import MinHash

if TYPE_CHECKING:
    from utils.helpers import Binary

logger = logging.getLogger(__name__)


def extract_shingles(opcode_seq: List[str], k: int) -> Set[str]:
    """Extract k-shingles (k-gram strings) from an opcode sequence."""
    if len(opcode_seq) < k:
        return set()
    return {' '.join(opcode_seq[i:i + k]) for i in range(len(opcode_seq) - k + 1)}


def build_minhash(shingles: Set[str], num_perm: int = 128) -> MinHash:
    """Build a MinHash signature from a set of shingles."""
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode('utf8'))
    return m


class FunctionMinHasher:
    """Build MinHash signatures for every function in a binary corpus."""

    def __init__(
        self,
        k_values: List[int] = None,
        num_perm: int = 128,
        min_func_size: int = 10,
    ):
        self.k_values = k_values if k_values is not None else [3, 4, 5]
        self.num_perm = num_perm
        self.min_func_size = min_func_size

    def build_signatures(
        self, binaries
    ) -> tuple:
        """
        Build MinHash signatures and opcode sequence map.

        Returns
        -------
        signatures : Dict[func_id, Dict[k, MinHash]]
        seq_map    : Dict[func_id, List[str]]
        """
        signatures: Dict[str, Dict[int, MinHash]] = {}
        seq_map: Dict[str, List[str]] = {}

        total_funcs = 0
        skipped = 0
        for binary in binaries:
            for func in binary.functions:
                func_id = f"{binary.name}|{func.name}"
                seq = func.opcode_sequence
                seq_map[func_id] = seq
                if len(seq) < self.min_func_size:
                    skipped += 1
                    total_funcs += 1
                    continue
                sigs: Dict[int, MinHash] = {}
                for k in self.k_values:
                    shingles = extract_shingles(seq, k)
                    if shingles:
                        sigs[k] = build_minhash(shingles, self.num_perm)
                if sigs:
                    signatures[func_id] = sigs
                total_funcs += 1

        logger.info(
            f"Built MinHash signatures for {len(signatures)}/{total_funcs} functions "
            f"(skipped {skipped} tiny functions < {self.min_func_size} instructions, "
            f"k={self.k_values}, num_perm={self.num_perm})"
        )
        return signatures, seq_map
