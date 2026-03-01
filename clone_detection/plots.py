"""
All Phase 7 visualizations for clone detection results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from clone_detection.families import CloneFamily

logger = logging.getLogger(__name__)


def plot_clone_family_size_distribution(
    families: List[CloneFamily],
    output_path: Path,
) -> None:
    """Histogram of clone family sizes."""
    if not families:
        return

    sizes = [f.size for f in families]
    fig, ax = plt.subplots(figsize=(8, 5))
    max_size = max(sizes)
    bins = range(2, max_size + 2)
    ax.hist(sizes, bins=list(bins), color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Clone Family Size (number of functions)')
    ax.set_ylabel('Number of Families')
    ax.set_title('Clone Family Size Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate with stats
    ax.axvline(np.mean(sizes), color='coral', linestyle='--',
               linewidth=1.5, label=f'Mean = {np.mean(sizes):.1f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved clone family size distribution to {output_path}")


def plot_cross_binary_heatmap(
    matrix: np.ndarray,
    binary_names: List[str],
    output_path: Path,
) -> None:
    """Heatmap of cross-binary clone pair counts."""
    if matrix.size == 0:
        return

    n = len(binary_names)
    fig_size = max(6, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Clone pair count')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [name[:12] for name in binary_names]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=max(6, 10 - n // 5))
    ax.set_yticklabels(labels, fontsize=max(6, 10 - n // 5))
    ax.set_title('Cross-Binary Clone Pair Heatmap')

    # Annotate cells
    if n <= 20:
        for i in range(n):
            for j in range(n):
                if matrix[i, j] > 0:
                    ax.text(j, i, str(int(matrix[i, j])),
                            ha='center', va='center',
                            fontsize=max(6, 8 - n // 8),
                            color='white' if matrix[i, j] > matrix.max() * 0.6 else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cross-binary heatmap to {output_path}")


def plot_function_umap(
    embedding: np.ndarray,
    func_ids: List[str],
    labels: List,
    label_name: str,
    output_path: Path,
) -> None:
    """Scatter plot of 2D function embeddings coloured by a label."""
    if embedding is None or len(embedding) == 0:
        return

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap('tab20', len(unique_labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        pts = embedding[[i for i, m in enumerate(mask) if m]]
        if len(pts) == 0:
            continue
        color = cmap(label_to_idx[lbl])
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.7,
                   color=color, label=str(lbl), linewidths=0)

    ax.set_title(f'Function Embedding (colored by {label_name})')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    if len(unique_labels) <= 20:
        ax.legend(markerscale=2, fontsize=8, loc='best',
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved function embedding plot ({label_name}) to {output_path}")


def plot_clone_type_distribution(
    families: List[CloneFamily],
    output_path: Path,
) -> None:
    """Bar chart of clone families broken down by Type 1/2/3."""
    if not families:
        return

    counts = {1: 0, 2: 0, 3: 0}
    for f in families:
        counts[f.clone_type] = counts.get(f.clone_type, 0) + 1

    labels = [f'Type {t}' for t in sorted(counts)]
    values = [counts[t] for t in sorted(counts)]
    colors = ['#2ecc71', '#3498db', '#e67e22']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor='white', alpha=0.9)
    ax.set_ylabel('Number of Clone Families')
    ax.set_title('Clone Family Distribution by Type')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontweight='bold')

    type_desc = {
        'Type 1': 'Near-identical',
        'Type 2': 'Structurally similar',
        'Type 3': 'Semantically related',
    }
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [f'{lbl}\n({type_desc.get(lbl, "")})' for lbl in labels],
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved clone type distribution to {output_path}")
