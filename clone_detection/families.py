"""
Clone family extraction, LCS conserved core, and clone density computation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set

from clone_detection.taxonomy import ClonePair

logger = logging.getLogger(__name__)


@dataclass
class CloneFamily:
    family_id: int
    members: List[str]         # func_ids
    clone_type: int            # most common type among pairs in this family
    binary_names: List[str]    # deduplicated binary names
    is_cross_binary: bool
    size: int                  # number of members
    conserved_core: List[str]  # LCS of all member sequences
    divergence_score: float    # 1 - len(core)/mean_seq_len


# ---------------------------------------------------------------------------
# LCS helpers
# ---------------------------------------------------------------------------

def _lcs_two(seq1: List[str], seq2: List[str]) -> List[str]:
    """O(m*n) DP to compute the longest common subsequence of two lists."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    result: List[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            result.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    result.reverse()
    return result


def compute_longest_common_subsequence(sequences: List[List[str]]) -> List[str]:
    """Iterative fold: LCS of N sequences via pairwise reduction."""
    if not sequences:
        return []
    current = sequences[0]
    for seq in sequences[1:]:
        current = _lcs_two(current, seq)
        if not current:
            break
    return current


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_clone_families(
    components: List[List[str]],
    clone_pairs: List[ClonePair],
    seq_map: Dict[str, List[str]],
    min_size: int = 2,
) -> List['CloneFamily']:
    """
    Turn connected components into CloneFamily objects with conserved cores.
    """
    # Build pair lookup for type resolution
    type_map: Dict[tuple, int] = {}
    for cp in clone_pairs:
        key = (min(cp.func_id1, cp.func_id2), max(cp.func_id1, cp.func_id2))
        type_map[key] = cp.clone_type

    families: List[CloneFamily] = []
    family_id = 0

    for comp in components:
        if len(comp) < min_size:
            continue

        # Determine dominant clone type for pairs within this component
        comp_set = set(comp)
        type_counts: Dict[int, int] = {}
        for cp in clone_pairs:
            if cp.func_id1 in comp_set and cp.func_id2 in comp_set:
                type_counts[cp.clone_type] = type_counts.get(cp.clone_type, 0) + 1
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else 3

        # Binary names
        binary_names: List[str] = list({fid.split('|')[0] for fid in comp})
        is_cross_binary = len(binary_names) > 1

        # Compute conserved core (LCS of all member sequences)
        sequences = [seq_map[fid] for fid in comp if fid in seq_map]
        if sequences:
            # Cap individual sequence lengths to avoid O(n^2) explosion
            trimmed = [s[:200] for s in sequences]
            core = compute_longest_common_subsequence(trimmed)
        else:
            core = []

        # Divergence score
        mean_len = sum(len(s) for s in sequences) / len(sequences) if sequences else 1
        divergence = 1.0 - len(core) / mean_len if mean_len > 0 else 1.0
        divergence = max(0.0, min(1.0, divergence))

        families.append(
            CloneFamily(
                family_id=family_id,
                members=list(comp),
                clone_type=dominant_type,
                binary_names=binary_names,
                is_cross_binary=is_cross_binary,
                size=len(comp),
                conserved_core=core,
                divergence_score=divergence,
            )
        )
        family_id += 1

    logger.info(f"Extracted {len(families)} clone families (min_size={min_size})")
    return families


def compute_clone_density(
    families: List[CloneFamily],
    binaries,
) -> Dict[str, float]:
    """
    Compute clone density per binary:
    fraction of its functions that appear in at least one clone family.
    """
    # Count functions per binary
    func_count: Dict[str, int] = {b.name: b.function_count for b in binaries}

    # Count cloned functions per binary
    cloned: Dict[str, Set[str]] = {b.name: set() for b in binaries}
    for fam in families:
        for fid in fam.members:
            binary_name = fid.split('|')[0]
            if binary_name in cloned:
                cloned[binary_name].add(fid)

    density: Dict[str, float] = {}
    for b in binaries:
        total = func_count.get(b.name, 0)
        n_cloned = len(cloned.get(b.name, set()))
        density[b.name] = n_cloned / total if total > 0 else 0.0

    return density
