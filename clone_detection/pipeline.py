"""
Clone detection pipeline orchestrator.

run_clone_detection() — main entry point for programmatic use
run_clones_command()  — CLI handler (loads corpus.pkl, calls run_clone_detection)
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.helpers import load_pickle, ensure_output_dir, save_json

logger = logging.getLogger(__name__)


def run_clone_detection(
    binaries,
    output_dir: Path,
    k_values: List[int] = None,
    num_perm: int = 128,
    thresholds: List[float] = None,
    run_embeddings: bool = True,
) -> Dict:
    """
    Full clone detection pipeline.

    Returns a results dict with keys:
        corpus_info, clone_pairs, family_stats, clone_density_per_binary,
        cross_binary_matrix, families (list of dicts)
    """
    if k_values is None:
        k_values = [3, 4, 5]
    if thresholds is None:
        thresholds = [0.7, 0.9]  # start high to avoid O(N^2) blowup on large corpora

    output_dir = ensure_output_dir(output_dir)
    plots_dir = ensure_output_dir(output_dir / 'plots')

    total_functions = sum(b.function_count for b in binaries)
    binary_names = [b.name for b in binaries]
    logger.info(
        f"Clone detection: {len(binaries)} binaries, {total_functions} functions"
    )

    # ------------------------------------------------------------------ #
    # Step 1: MinHash signatures                                           #
    # ------------------------------------------------------------------ #
    from clone_detection.minhash import FunctionMinHasher
    hasher = FunctionMinHasher(k_values=k_values, num_perm=num_perm)
    signatures, seq_map = hasher.build_signatures(binaries)

    # ------------------------------------------------------------------ #
    # Step 2: LSH candidate finding                                        #
    # ------------------------------------------------------------------ #
    from clone_detection.lsh import LSHCandidateFinder, CandidatePair
    finder = LSHCandidateFinder(thresholds=thresholds, num_perm=num_perm)
    best_candidate: dict = {}  # (fid1, fid2) -> CandidatePair with highest jaccard
    for k in k_values:
        candidates_k = finder.find_candidates(signatures, k)
        candidates_k = finder.filter_self_pairs(candidates_k)
        for cp in candidates_k:
            key = (min(cp.func_id1, cp.func_id2), max(cp.func_id1, cp.func_id2))
            if key not in best_candidate or cp.jaccard_estimate > best_candidate[key].jaccard_estimate:
                best_candidate[key] = cp
    all_candidates = list(best_candidate.values())
    logger.info(f"Total unique candidates after LSH (all k): {len(all_candidates)}")

    if not all_candidates:
        logger.warning("No candidate pairs found — corpus may be too small or too diverse")
        return _empty_results(binaries, total_functions)

    # ------------------------------------------------------------------ #
    # Step 3: Smith-Waterman alignment                                     #
    # ------------------------------------------------------------------ #
    from clone_detection.alignment import CandidateAligner
    aligner = CandidateAligner(binaries)
    alignments = aligner.align_candidates(all_candidates, seq_map=seq_map)

    # ------------------------------------------------------------------ #
    # Step 4: Clone pair taxonomy                                          #
    # ------------------------------------------------------------------ #
    from clone_detection.taxonomy import build_clone_pairs
    clone_pairs = build_clone_pairs(all_candidates, alignments)

    # ------------------------------------------------------------------ #
    # Step 5: Clone graph + connected components                           #
    # ------------------------------------------------------------------ #
    from clone_detection.graph import (
        build_clone_graph,
        find_connected_components,
        compute_cross_binary_matrix,
    )
    graph = build_clone_graph(clone_pairs, binaries)
    components = find_connected_components(graph)
    cross_matrix = compute_cross_binary_matrix(clone_pairs, binary_names)

    # ------------------------------------------------------------------ #
    # Step 6: Clone families                                               #
    # ------------------------------------------------------------------ #
    from clone_detection.families import extract_clone_families, compute_clone_density
    families = extract_clone_families(components, clone_pairs, seq_map)
    density = compute_clone_density(families, binaries)

    # ------------------------------------------------------------------ #
    # Step 7: Embeddings (bonus)                                           #
    # ------------------------------------------------------------------ #
    embedding_results: Dict = {}
    if run_embeddings and len(seq_map) >= 5:
        try:
            from clone_detection.embeddings import FunctionEmbedder
            embedder = FunctionEmbedder(k=3, max_features=5000)
            func_ids = list(seq_map.keys())
            tfidf = embedder.build_tfidf_matrix(func_ids, seq_map)

            # UMAP / PCA
            umap_embedding, umap_info = embedder.reduce_umap(tfidf)
            pca_embedding, pca_info = embedder.reduce_pca(tfidf)
            embedding_results = {
                'func_ids': func_ids,
                'umap_info': umap_info,
                'pca_info': pca_info,
            }

            # Build labels for plots
            binary_labels = [fid.split('|')[0] for fid in func_ids]

            # Clone-family labels (family_id or -1)
            fid_to_family: Dict[str, int] = {}
            for fam in families:
                for fid in fam.members:
                    fid_to_family[fid] = fam.family_id
            family_labels = [fid_to_family.get(fid, -1) for fid in func_ids]

            from clone_detection.plots import plot_function_umap
            plot_function_umap(
                umap_embedding, func_ids, family_labels,
                'clone_family',
                plots_dir / 'function_umap_clone_family.png',
            )
            plot_function_umap(
                umap_embedding, func_ids, binary_labels,
                'binary',
                plots_dir / 'function_umap_binary.png',
            )
        except Exception as e:
            logger.warning(f"Embeddings step failed (non-fatal): {e}")

    # ------------------------------------------------------------------ #
    # Step 8: Visualizations                                               #
    # ------------------------------------------------------------------ #
    from clone_detection.plots import (
        plot_clone_family_size_distribution,
        plot_cross_binary_heatmap,
        plot_clone_type_distribution,
    )
    try:
        plot_clone_family_size_distribution(
            families, plots_dir / 'clone_family_size_dist.png'
        )
    except Exception as e:
        logger.warning(f"Size distribution plot failed: {e}")

    try:
        plot_cross_binary_heatmap(
            cross_matrix, binary_names, plots_dir / 'clone_cross_binary_heatmap.png'
        )
    except Exception as e:
        logger.warning(f"Cross-binary heatmap failed: {e}")

    try:
        plot_clone_type_distribution(
            families, plots_dir / 'clone_type_distribution.png'
        )
    except Exception as e:
        logger.warning(f"Clone type distribution plot failed: {e}")

    # ------------------------------------------------------------------ #
    # Step 9: Compute summary statistics                                   #
    # ------------------------------------------------------------------ #
    functions_in_clone: set = set()
    for fam in families:
        functions_in_clone.update(fam.members)

    type_dist_pairs: Dict[str, int] = {}
    for cp in clone_pairs:
        key = str(cp.clone_type)
        type_dist_pairs[key] = type_dist_pairs.get(key, 0) + 1

    type_dist_families: Dict[str, int] = {}
    for fam in families:
        key = str(fam.clone_type)
        type_dist_families[key] = type_dist_families.get(key, 0) + 1

    cross_binary_families = [f for f in families if f.is_cross_binary]
    family_sizes = [f.size for f in families]

    results = {
        'corpus_info': {
            'num_binaries': len(binaries),
            'total_functions': total_functions,
            'functions_in_any_clone': len(functions_in_clone),
            'clone_fraction': len(functions_in_clone) / total_functions if total_functions else 0,
        },
        'clone_pairs': {
            'total_candidates': len(all_candidates),
            'total_clone_pairs': len(clone_pairs),
            'type_distribution': type_dist_pairs,
        },
        'family_stats': {
            'total_families': len(families),
            'cross_binary_families': len(cross_binary_families),
            'type_distribution': type_dist_families,
            'mean_family_size': float(np.mean(family_sizes)) if family_sizes else 0,
            'largest_family_size': max(family_sizes) if family_sizes else 0,
        },
        'clone_density_per_binary': density,
        'cross_binary_matrix': {
            'binary_names': binary_names,
            'matrix': cross_matrix.tolist(),
        },
    }

    # ------------------------------------------------------------------ #
    # Step 10: Write JSON outputs                                          #
    # ------------------------------------------------------------------ #

    # clone_stats.json
    save_json(results, output_dir / 'clone_stats.json')

    # clone_families.json
    families_out = []
    for fam in families:
        members_list = []
        for fid in fam.members:
            binary_name, func_name = fid.split('|', 1)
            size_val = 0
            for b in binaries:
                if b.name == binary_name:
                    for f in b.functions:
                        if f.name == func_name:
                            size_val = f.size
                            break
                    break
            members_list.append({
                'func_id': fid,
                'binary': binary_name,
                'size': size_val,
            })
        families_out.append({
            'family_id': fam.family_id,
            'size': fam.size,
            'clone_type': fam.clone_type,
            'is_cross_binary': fam.is_cross_binary,
            'binary_names': fam.binary_names,
            'members': members_list,
            'conserved_core': fam.conserved_core,
            'conserved_core_length': len(fam.conserved_core),
            'divergence_score': fam.divergence_score,
        })
    save_json(
        {'families': families_out, 'total_families': len(families)},
        output_dir / 'clone_families.json',
    )

    # clone_graph.json
    graph_out = {
        'nodes': graph.nodes,
        'edges': [
            {'source': e[0], 'target': e[1], **e[2]}
            for e in graph.edges
        ],
        'adjacency': graph.adjacency,
    }
    save_json(graph_out, output_dir / 'clone_graph.json')

    logger.info(
        f"Clone detection complete: {len(clone_pairs)} clone pairs, "
        f"{len(families)} families, "
        f"clone fraction = {results['corpus_info']['clone_fraction']:.3f}"
    )
    return results


def _empty_results(binaries, total_functions: int) -> Dict:
    return {
        'corpus_info': {
            'num_binaries': len(binaries),
            'total_functions': total_functions,
            'functions_in_any_clone': 0,
            'clone_fraction': 0.0,
        },
        'clone_pairs': {'total_candidates': 0, 'total_clone_pairs': 0, 'type_distribution': {}},
        'family_stats': {
            'total_families': 0, 'cross_binary_families': 0,
            'type_distribution': {}, 'mean_family_size': 0, 'largest_family_size': 0,
        },
        'clone_density_per_binary': {b.name: 0.0 for b in binaries},
        'cross_binary_matrix': {
            'binary_names': [b.name for b in binaries],
            'matrix': np.zeros((len(binaries), len(binaries)), dtype=int).tolist(),
        },
    }


def run_clones_command(args) -> int:
    """CLI entry point for `binary_dna.py clones`."""
    import logging
    logger = logging.getLogger(__name__)

    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)

    corpus_file = corpus_dir / 'corpus.pkl'
    if not corpus_file.exists():
        logger.error(f"Corpus file not found: {corpus_file}")
        return 1

    try:
        binaries = load_pickle(corpus_file)
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        return 1

    if not binaries:
        logger.error("Corpus is empty")
        return 1

    logger.info(f"Loaded {len(binaries)} binaries from {corpus_file}")

    try:
        run_clone_detection(binaries, output_dir)
        return 0
    except Exception as e:
        logger.error(f"Clone detection failed: {e}", exc_info=True)
        return 1
