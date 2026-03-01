"""
Clone graph construction, BFS connected components, and cross-binary matrix.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from clone_detection.taxonomy import ClonePair

logger = logging.getLogger(__name__)


@dataclass
class CloneGraph:
    nodes: Dict[str, Dict]                  # func_id -> {binary_name, func_name, size}
    edges: List[Tuple[str, str, Dict]]      # (func_id1, func_id2, attrs)
    adjacency: Dict[str, List[str]]         # func_id -> [neighbor func_ids]


def build_clone_graph(clone_pairs: List[ClonePair], binaries) -> CloneGraph:
    """Build a clone graph from clone pairs."""
    # Build func size lookup
    size_map: Dict[str, int] = {}
    for b in binaries:
        for f in b.functions:
            fid = f"{b.name}|{f.name}"
            size_map[fid] = f.size

    nodes: Dict[str, Dict] = {}
    edges: List[Tuple[str, str, Dict]] = []
    adjacency: Dict[str, List[str]] = {}

    for cp in clone_pairs:
        for fid in (cp.func_id1, cp.func_id2):
            if fid not in nodes:
                binary_name, func_name = fid.split('|', 1)
                nodes[fid] = {
                    'binary_name': binary_name,
                    'func_name': func_name,
                    'size': size_map.get(fid, 0),
                }
                adjacency[fid] = []

        edges.append((
            cp.func_id1,
            cp.func_id2,
            {
                'jaccard': cp.jaccard,
                'alignment_score': cp.alignment_score,
                'clone_type': cp.clone_type,
                'cross_binary': cp.cross_binary,
            },
        ))

        if cp.func_id2 not in adjacency[cp.func_id1]:
            adjacency[cp.func_id1].append(cp.func_id2)
        if cp.func_id1 not in adjacency[cp.func_id2]:
            adjacency[cp.func_id2].append(cp.func_id1)

    logger.info(
        f"Clone graph: {len(nodes)} nodes, {len(edges)} edges"
    )
    return CloneGraph(nodes=nodes, edges=edges, adjacency=adjacency)


def find_connected_components(graph: CloneGraph) -> List[List[str]]:
    """BFS-based connected component finder (no networkx dependency)."""
    visited: set = set()
    components: List[List[str]] = []

    for start in graph.nodes:
        if start in visited:
            continue
        component: List[str] = []
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for nbr in graph.adjacency.get(node, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        components.append(component)

    # Sort: largest first
    components.sort(key=len, reverse=True)
    logger.info(f"Found {len(components)} connected components")
    return components


def compute_cross_binary_matrix(
    clone_pairs: List[ClonePair],
    binary_names: List[str],
) -> np.ndarray:
    """
    Compute a symmetric matrix where entry [i, j] = number of clone pairs
    between binary i and binary j.  Diagonal = within-binary clones.
    """
    idx = {name: i for i, name in enumerate(binary_names)}
    n = len(binary_names)
    matrix = np.zeros((n, n), dtype=int)

    for cp in clone_pairs:
        i = idx.get(cp.binary1, -1)
        j = idx.get(cp.binary2, -1)
        if i < 0 or j < 0:
            continue
        matrix[i, j] += 1
        if i != j:
            matrix[j, i] += 1

    return matrix
