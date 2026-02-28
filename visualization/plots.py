"""
Plotting and visualization functions for Binary DNA analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def plot_zipf_distribution(frequency_data: List[tuple], zipf_params: Dict, output_path: Path) -> None:
    """Plot rank-frequency distribution with Zipf's law fit."""
    ranks = [item[2] for item in frequency_data]
    frequencies = [item[1] for item in frequency_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear plot
    ax1.plot(ranks, frequencies, 'bo', alpha=0.6, markersize=4)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Opcode Frequency Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Log-log plot with Zipf fit
    log_ranks = np.array(ranks)
    log_frequencies = np.array(frequencies)
    
    ax2.loglog(ranks, frequencies, 'bo', alpha=0.6, markersize=4, label='Observed')
    
    # Plot Zipf fit line
    if zipf_params.get('alpha', 0) > 0:
        alpha = zipf_params['alpha']
        constant = zipf_params['constant']
        fit_frequencies = constant / (log_ranks ** alpha)
        ax2.loglog(ranks, fit_frequencies, 'r-', linewidth=2, 
                  label=f'Zipf fit: α={alpha:.3f}, R²={zipf_params.get("r_squared", 0):.3f}')
    
    ax2.set_xlabel('Rank (log scale)')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Zipf\'s Law Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Zipf distribution plot to {output_path}")

def plot_entropy_rates(entropy_data: List[Dict], output_path: Path,
                       baseline_data: List[Dict] = None) -> None:
    """Plot entropy rate vs n-gram length, with optional shuffled baseline overlay."""
    n_values = [item['n'] for item in entropy_data]
    entropy_rates = [item['entropy_rate'] for item in entropy_data]
    entropies = [item['entropy'] for item in entropy_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Entropy rate plot
    ax1.plot(n_values[1:], entropy_rates[1:], 'bo-', linewidth=2, markersize=8, label='Real')
    if baseline_data:
        b_n = [item['n'] for item in baseline_data]
        b_rates = [item['entropy_rate'] for item in baseline_data]
        ax1.plot(b_n[1:], b_rates[1:], 'r--s', linewidth=2, markersize=8, label='Shuffled baseline')
        ax1.legend()
    ax1.set_xlabel('N-gram Length (n)')
    ax1.set_ylabel('Entropy Rate (bits)')
    ax1.set_title('Entropy Rate vs N-gram Length')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_values)

    # Cumulative entropy plot
    ax2.plot(n_values, entropies, 'bo-', linewidth=2, markersize=8, label='Real')
    if baseline_data:
        b_entropies = [item['entropy'] for item in baseline_data]
        ax2.plot(b_n, b_entropies, 'r--s', linewidth=2, markersize=8, label='Shuffled baseline')
        ax2.legend()
    ax2.set_xlabel('N-gram Length (n)')
    ax2.set_ylabel('Cumulative Entropy (bits)')
    ax2.set_title('Cumulative Entropy vs N-gram Length')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(n_values)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved entropy rate plot to {output_path}")

def plot_compression_ratios(compression_data: List[Dict], output_path: Path) -> None:
    """Plot compression ratios for different algorithms."""
    binary_names = [item['binary_name'] for item in compression_data]
    zlib_ratios = [item['zlib_ratio'] for item in compression_data]
    lzma_ratios = [item['lzma_ratio'] for item in compression_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Compression ratio comparison
    x = np.arange(len(binary_names))
    width = 0.35
    
    ax1.bar(x - width/2, zlib_ratios, width, label='zlib', alpha=0.8)
    ax1.bar(x + width/2, lzma_ratios, width, label='LZMA', alpha=0.8)
    
    ax1.set_xlabel('Binary')
    ax1.set_ylabel('Compression Ratio')
    ax1.set_title('Compression Ratios by Algorithm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(binary_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Scatter plot: size vs compression ratio
    sizes = [item['instruction_count'] for item in compression_data]
    ax2.scatter(sizes, zlib_ratios, alpha=0.6, s=50)
    
    # Add binary names as annotations for interesting points
    for i, (size, ratio, name) in enumerate(zip(sizes, zlib_ratios, binary_names)):
        if ratio < 0.3 or ratio > 0.9 or size > np.percentile(sizes, 90):
            ax2.annotate(name, (size, ratio), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Instruction Count')
    ax2.set_ylabel('Zlib Compression Ratio')
    ax2.set_title('Compression Ratio vs Binary Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved compression ratio plot to {output_path}")

def plot_frequency_heatmap(frequency_data: List[tuple], top_n: int, output_path: Path) -> None:
    """Plot heatmap of top opcode frequencies."""
    # Get top N opcodes
    top_opcodes = frequency_data[:top_n]
    opcodes = [item[0] for item in top_opcodes]
    counts = [item[1] for item in top_opcodes]
    
    # Create a single-row heatmap
    data = np.array(counts).reshape(1, -1)
    
    plt.figure(figsize=(15, 3))
    sns.heatmap(data, xticklabels=opcodes, yticklabels=['Frequency'], 
                annot=True, fmt='d', cmap='YlOrRd')
    
    plt.title(f'Top {top_n} Opcode Frequencies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved frequency heatmap to {output_path}")

def plot_motif_heatmap(motif_results: Dict, output_path: Path) -> None:
    """Plot heatmap showing top motifs across different lengths."""
    # Collect top motifs from each length
    all_motifs = []
    lengths = []
    
    for k_mer, motifs in motif_results.items():
        k = int(k_mer.replace('mer', ''))
        for i, motif in enumerate(motifs[:5]):  # Top 5 per length
            all_motifs.append({
                'length': k,
                'rank': i + 1,
                'motif': motif['motif'][:30] + '...' if len(motif['motif']) > 30 else motif['motif'],
                'score': motif['frequency'] * motif['function_coverage'],
                'frequency': motif['frequency'],
                'coverage': motif['function_coverage']
            })
        lengths.append(k)
    
    if not all_motifs:
        logger.warning("No motifs to plot")
        return
    
    # Create matrix for heatmap
    lengths = sorted(set(lengths))
    max_rank = 5
    
    # Create score matrix
    score_matrix = np.zeros((max_rank, len(lengths)))
    labels = []
    
    for i, length in enumerate(lengths):
        length_motifs = [m for m in all_motifs if m['length'] == length][:max_rank]
        for j, motif in enumerate(length_motifs):
            score_matrix[j, i] = motif['score']
        
        # Labels for first column
        if i == 0:
            labels = [f"Rank {j+1}" for j in range(max_rank)]
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(score_matrix, 
                xticklabels=[f'{k}-mer' for k in lengths],
                yticklabels=labels,
                annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Motif Score (frequency × coverage)'})
    
    plt.title('Top Motifs by Length - Score Heatmap')
    plt.xlabel('Motif Length')
    plt.ylabel('Motif Rank')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved motif heatmap to {output_path}")

def plot_positional_entropy(positional_results: Dict, output_path: Path) -> None:
    """Plot entropy vs position for function boundaries."""
    start_patterns = positional_results.get('start_patterns', {})
    end_patterns = positional_results.get('end_patterns', {})
    
    start_entropies = start_patterns.get('entropies', [])
    end_entropies = end_patterns.get('entropies', [])
    
    if not start_entropies and not end_entropies:
        logger.warning("No positional entropy data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Start position entropies
    if start_entropies:
        positions = range(len(start_entropies))
        ax1.plot(positions, start_entropies, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Position from Function Start')
        ax1.set_ylabel('Shannon Entropy (bits)')
        ax1.set_title('Entropy vs Position - Function Start')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, len(start_entropies), 2))
    
    # End position entropies
    if end_entropies:
        positions = range(len(end_entropies))
        ax2.plot(positions, end_entropies, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Position from Function End')
        ax2.set_ylabel('Shannon Entropy (bits)')
        ax2.set_title('Entropy vs Position - Function End')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, len(end_entropies), 2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved positional entropy plot to {output_path}")

def plot_positional_heatmap(positional_results: Dict, output_path: Path,
                             top_n_opcodes: int = 15) -> None:
    """
    Heatmap of opcode probability at each position from function start and end.

    Rows = top-N most common opcodes across all positions.
    Columns = position index (0 = boundary, 1 = one step in, ...).
    Two panels: function-start boundary (left) and function-end boundary (right).
    """
    start_dist = positional_results.get('start_patterns', {}).get('distributions', {})
    end_dist   = positional_results.get('end_patterns',   {}).get('distributions', {})

    if not start_dist and not end_dist:
        logger.warning("No positional distribution data to plot heatmap")
        return

    def _build_matrix(distributions):
        """distributions: {pos_int: {opcode: prob}} → (matrix, opcodes, positions)"""
        if not distributions:
            return None, [], []
        positions = sorted(int(p) for p in distributions)
        # Collect union of opcodes and their total probability mass
        opcode_total: dict = {}
        for pos_data in distributions.values():
            for op, prob in pos_data.items():
                opcode_total[op] = opcode_total.get(op, 0) + prob
        top_ops = [op for op, _ in sorted(opcode_total.items(),
                                          key=lambda x: -x[1])][:top_n_opcodes]
        matrix = np.zeros((len(top_ops), len(positions)))
        for col, pos in enumerate(positions):
            pos_data = distributions.get(pos, distributions.get(str(pos), {}))
            for row, op in enumerate(top_ops):
                matrix[row, col] = pos_data.get(op, 0.0)
        return matrix, top_ops, positions

    s_mat, s_ops, s_pos = _build_matrix(start_dist)
    e_mat, e_ops, e_pos = _build_matrix(end_dist)

    n_panels = (1 if s_mat is None else 1) + (1 if e_mat is None else 1)
    if s_mat is not None and e_mat is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    elif s_mat is not None:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
        ax2 = None
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 7))
        ax1 = None

    def _draw(ax, matrix, ops, positions, title):
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([str(p) for p in positions], fontsize=8)
        ax.set_yticks(range(len(ops)))
        ax.set_yticklabels(ops, fontsize=9)
        ax.set_xlabel('Position from boundary')
        ax.set_ylabel('Opcode')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Probability')

    if ax1 is not None:
        _draw(ax1, s_mat, s_ops, s_pos, 'Function Start — opcode probability by position')
    if ax2 is not None:
        _draw(ax2, e_mat, e_ops, e_pos, 'Function End — opcode probability by position')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved positional heatmap to {output_path}")


def plot_performance_comparison(benchmark_results: Dict, output_path: Path) -> None:
    """Plot performance comparison results."""
    perf_data = benchmark_results.get('performance_comparison', {})
    
    if not perf_data:
        logger.warning("No performance data to plot")
        return
    
    # Create bar chart comparing sequential vs parallel
    methods = ['Sequential', 'Parallel']
    times = [perf_data.get('sequential_time', 0), perf_data.get('parallel_time', 0)]
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(methods, times, color=['lightblue', 'lightcoral'])
    plt.ylabel('Execution Time (seconds)')
    plt.title(f"Performance Comparison - {perf_data.get('speedup', 1):.1f}x Speedup")
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # Add speedup annotation
    speedup = perf_data.get('speedup', 1.0)
    plt.text(0.5, max(times) * 0.8, f'Speedup: {speedup:.2f}x', 
             ha='center', fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved performance comparison plot to {output_path}")

def plot_distance_matrix_heatmap(distance_matrix: List[List[float]], binary_names: List[str], 
                                title: str, output_path: Path) -> None:
    """Plot distance matrix as heatmap."""
    import numpy as np
    
    matrix = np.array(distance_matrix)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with binary names as labels
    sns.heatmap(matrix, 
                xticklabels=binary_names,
                yticklabels=binary_names,
                annot=True, fmt='.3f', cmap='viridis',
                square=True, cbar_kws={'label': 'Distance'})
    
    plt.title(title)
    plt.xlabel('Binary')
    plt.ylabel('Binary')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved distance matrix heatmap to {output_path}")

def plot_entropy_profile(entropy_data: Dict, binary_name: str, output_path: Path) -> None:
    """Plot sliding window entropy profile for a binary."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(len(entropy_data), 1, figsize=(12, 4 * len(entropy_data)))
    
    if len(entropy_data) == 1:
        axes = [axes]
    
    for i, (window_name, data) in enumerate(entropy_data.items()):
        if 'entropies' in data and 'positions' in data:
            positions = data['positions']
            entropies = data['entropies']
            
            axes[i].plot(positions, entropies, 'b-', linewidth=2, alpha=0.7)
            axes[i].set_xlabel('Position in Sequence')
            axes[i].set_ylabel('Shannon Entropy (bits)')
            axes[i].set_title(f'{binary_name} - Entropy Profile ({window_name})')
            axes[i].grid(True, alpha=0.3)
            
            # Add mean line
            mean_entropy = data.get('mean_entropy', np.mean(entropies))
            axes[i].axhline(y=mean_entropy, color='red', linestyle='--', 
                          label=f'Mean: {mean_entropy:.2f}', alpha=0.7)
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved entropy profile plot to {output_path}")

def plot_mutual_information_decay(mi_data: Dict, binary_name: str, output_path: Path) -> None:
    """Plot mutual information decay curve."""
    import matplotlib.pyplot as plt
    
    if 'mi_by_lag' not in mi_data:
        logger.warning(f"No MI data available for {binary_name}")
        return

    mi_by_lag = mi_data['mi_by_lag']
    # JSON keys are strings; cast to int before sorting
    lags = sorted(int(k) for k in mi_by_lag.keys())
    mi_values = [mi_by_lag[str(lag)] for lag in lags]
    shuffled_by_lag = mi_data.get('mi_shuffled_by_lag', {})
    
    plt.figure(figsize=(10, 6))

    plt.plot(lags, mi_values, 'bo-', linewidth=2, markersize=6, alpha=0.7, label='Real')

    # Overlay shuffled baseline if present
    if shuffled_by_lag:
        shuf_lags = sorted(int(k) for k in shuffled_by_lag.keys())
        shuf_vals = [shuffled_by_lag[str(lag)] for lag in shuf_lags]
        plt.plot(shuf_lags, shuf_vals, 'r--s', linewidth=1.5, markersize=4,
                 alpha=0.6, label='Shuffled baseline')
        plt.legend()

    plt.xlabel('Lag (positions)')
    plt.ylabel('Mutual Information (bits)')
    plt.title(f'{binary_name} - Mutual Information Decay')
    plt.grid(True, alpha=0.3)

    # Add half-life line if available
    if 'decay_analysis' in mi_data and 'half_life' in mi_data['decay_analysis']:
        half_life = mi_data['decay_analysis']['half_life']
        if half_life is not None:
            plt.axvline(x=half_life, color='green', linestyle='--',
                        label=f'Half-life: {half_life}', alpha=0.7)
            plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved MI decay plot to {output_path}")

def plot_corpus_mi_decay(corpus_analysis: Dict, output_path: Path) -> None:
    """
    Plot corpus-average MI decay: real sequences vs shuffled baseline.

    Shows how quickly mutual information drops with lag, averaged across all
    binaries.  The gap between real and shuffled curves is evidence of genuine
    higher-order sequential structure.
    """
    mean_real = corpus_analysis.get('mean_mi_decay', {})
    mean_shuf = corpus_analysis.get('mean_shuffled_mi_decay', {})

    if not mean_real:
        logger.warning("No corpus-level MI decay data to plot")
        return

    lags_real = sorted(int(k) for k in mean_real.keys())
    vals_real = [mean_real[str(lag)] for lag in lags_real]

    plt.figure(figsize=(10, 6))
    plt.plot(lags_real, vals_real, 'bo-', linewidth=2, markersize=6, label='Real (corpus mean)')

    if mean_shuf:
        lags_shuf = sorted(int(k) for k in mean_shuf.keys())
        vals_shuf = [mean_shuf[str(lag)] for lag in lags_shuf]
        plt.plot(lags_shuf, vals_shuf, 'r--s', linewidth=1.5, markersize=4,
                 alpha=0.7, label='Shuffled baseline (corpus mean)')
        plt.fill_between(lags_real,
                         [mean_shuf.get(str(lag), 0) for lag in lags_real],
                         vals_real,
                         alpha=0.15, color='blue', label='Structure (real − shuffled)')

    plt.xlabel('Lag (positions)')
    plt.ylabel('Mean Mutual Information (bits)')
    plt.title('Corpus-Average Mutual Information Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved corpus MI decay plot to {output_path}")


def plot_space_coverage(coverage_data: Dict, output_path: Path) -> None:
    """Plot program space coverage analysis."""
    import matplotlib.pyplot as plt
    
    # Extract data for different n-gram lengths
    n_values = []
    coverage_ratios = []
    unique_counts = []
    theoretical_maxs = []
    
    for key, data in coverage_data.items():
        if key.endswith('gram'):
            n = int(key.replace('gram', ''))
            n_values.append(n)
            coverage_ratios.append(data.get('coverage_ratio', 0))
            unique_counts.append(data.get('unique_ngrams', 0))
            theoretical_maxs.append(data.get('theoretical_maximum', 1))
    
    if not n_values:
        logger.warning("No coverage data available for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Coverage ratio plot
    ax1.semilogy(n_values, coverage_ratios, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('N-gram Length')
    ax1.set_ylabel('Coverage Ratio (log scale)')
    ax1.set_title('Program Space Coverage by N-gram Length')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_values)
    
    # Unique vs theoretical comparison
    width = 0.35
    x = np.arange(len(n_values))
    
    ax2.bar(x - width/2, unique_counts, width, label='Observed Unique N-grams', alpha=0.7)
    ax2.bar(x + width/2, theoretical_maxs, width, label='Theoretical Maximum', alpha=0.7)
    ax2.set_xlabel('N-gram Length')
    ax2.set_ylabel('Count')
    ax2.set_title('Observed vs Theoretical N-gram Counts')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{n}-gram' for n in n_values])
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved space coverage plot to {output_path}")

def plot_dimensionality_analysis(dimensionality_results: Dict, output_path: Path) -> None:
    """Plot dimensionality analysis results."""
    import matplotlib.pyplot as plt
    
    # Extract PCA results
    pca_results = []
    mle_results = []
    
    for key, data in dimensionality_results.items():
        if key.startswith('dimensionality_') and isinstance(data, dict):
            n_gram = key.replace('dimensionality_', '')
            
            if 'pca' in data and 'error' not in data['pca']:
                pca_data = data['pca']
                pca_results.append({
                    'n_gram': n_gram,
                    'effective_dim': pca_data.get('effective_dimensionality', 0),
                    'components_95': pca_data.get('components_for_95_variance', 0),
                    'total_dims': pca_data.get('total_dimensions', 0)
                })
            
            if 'mle' in data and 'error' not in data['mle']:
                mle_data = data['mle']
                mle_results.append({
                    'n_gram': n_gram,
                    'mean_dim': mle_data.get('mean_dimensionality', 0),
                    'std_dim': mle_data.get('std_dimensionality', 0)
                })
    
    if not pca_results and not mle_results:
        logger.warning("No dimensionality data available for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA dimensionality plot
    if pca_results:
        n_grams = [r['n_gram'] for r in pca_results]
        effective_dims = [r['effective_dim'] for r in pca_results]
        components_95 = [r['components_95'] for r in pca_results]
        total_dims = [r['total_dims'] for r in pca_results]
        
        x = np.arange(len(n_grams))
        width = 0.25
        
        axes[0].bar(x - width, effective_dims, width, label='Effective Dimensionality', alpha=0.7)
        axes[0].bar(x, components_95, width, label='95% Variance Components', alpha=0.7)
        axes[0].bar(x + width, total_dims, width, label='Total Dimensions', alpha=0.7)
        
        axes[0].set_xlabel('N-gram Type')
        axes[0].set_ylabel('Dimensionality')
        axes[0].set_title('PCA Dimensionality Analysis')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(n_grams)
        axes[0].legend()
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # MLE dimensionality plot with error bars
    if mle_results:
        n_grams = [r['n_gram'] for r in mle_results]
        mean_dims = [r['mean_dim'] for r in mle_results]
        std_dims = [r['std_dim'] for r in mle_results]
        
        axes[1].errorbar(range(len(n_grams)), mean_dims, yerr=std_dims, 
                        fmt='bo-', linewidth=2, markersize=8, capsize=5)
        axes[1].set_xlabel('N-gram Type')
        axes[1].set_ylabel('MLE Dimensionality Estimate')
        axes[1].set_title('Maximum Likelihood Dimensionality Estimation')
        axes[1].set_xticks(range(len(n_grams)))
        axes[1].set_xticklabels(n_grams)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved dimensionality analysis plot to {output_path}")

def plot_compiler_distribution(compiler_results: Dict, output_path: Path) -> None:
    """Plot compiler and optimization distribution from fingerprinting analysis."""
    import matplotlib.pyplot as plt
    
    corpus_summary = compiler_results.get('corpus_summary', {})
    
    if not corpus_summary:
        logger.warning("No compiler fingerprinting data available for plotting")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Compiler distribution
    compiler_dist = corpus_summary.get('compiler_distribution', {})
    if compiler_dist:
        compilers = list(compiler_dist.keys())
        counts = list(compiler_dist.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(compilers)))
        bars = axes[0].bar(compilers, counts, color=colors, alpha=0.8)
        axes[0].set_xlabel('Predicted Compiler')
        axes[0].set_ylabel('Number of Binaries')
        axes[0].set_title('Compiler Distribution')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
    
    # Optimization level distribution
    opt_dist = corpus_summary.get('optimization_distribution', {})
    if opt_dist:
        opt_levels = list(opt_dist.keys())
        opt_counts = list(opt_dist.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(opt_levels)))
        bars = axes[1].bar(opt_levels, opt_counts, color=colors, alpha=0.8)
        axes[1].set_xlabel('Predicted Optimization Level')
        axes[1].set_ylabel('Number of Binaries')
        axes[1].set_title('Optimization Level Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, opt_counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
    
    # Language distribution
    lang_dist = corpus_summary.get('language_distribution', {})
    if lang_dist:
        languages = list(lang_dist.keys())
        lang_counts = list(lang_dist.values())
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(languages)))
        bars = axes[2].bar(languages, lang_counts, color=colors, alpha=0.8)
        axes[2].set_xlabel('Source Language')
        axes[2].set_ylabel('Number of Binaries')
        axes[2].set_title('Source Language Distribution')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, lang_counts):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved compiler distribution plot to {output_path}")

def plot_compiler_feature_comparison(compiler_results: Dict, output_path: Path) -> None:
    """Plot comparison of compiler features across different compilers."""
    import matplotlib.pyplot as plt
    
    aggregate_features = compiler_results.get('aggregate_features', {})
    
    if not aggregate_features:
        logger.warning("No aggregate compiler features available for plotting")
        return
    
    # Select top discriminative features
    feature_names = set()
    for compiler_data in aggregate_features.values():
        feature_names.update(compiler_data.keys())
    
    # Filter to most interesting features
    interesting_features = [f for f in feature_names if any(keyword in f for keyword in 
                           ['ratio_', 'vectorization', 'lea_', 'jump_', 'call_', 'cmov_'])][:10]
    
    if not interesting_features:
        logger.warning("No interesting compiler features found")
        return
    
    compilers = list(aggregate_features.keys())
    n_features = len(interesting_features)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create matrix of feature means
    feature_matrix = []
    for compiler in compilers:
        compiler_row = []
        for feature in interesting_features:
            mean_val = aggregate_features[compiler].get(feature, {}).get('mean', 0)
            compiler_row.append(mean_val)
        feature_matrix.append(compiler_row)
    
    # Create heatmap
    im = ax.imshow(feature_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(len(compilers)))
    ax.set_xticklabels([f.replace('ratio_', '').replace('_', ' ') for f in interesting_features])
    ax.set_yticklabels(compilers)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Feature Mean Value')
    
    # Add text annotations
    for i in range(len(compilers)):
        for j in range(n_features):
            text = ax.text(j, i, f'{feature_matrix[i][j]:.3f}', 
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Compiler Feature Comparison Heatmap')
    ax.set_xlabel('Features')
    ax.set_ylabel('Predicted Compiler')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved compiler feature comparison plot to {output_path}")

def generate_all_plots(results_dir: Path, plot_dir: Path) -> List[Path]:
    """Generate all plots from analysis results."""
    from utils.helpers import load_json
    
    plot_files = []
    
    try:
        # Load analysis results
        freq_results = load_json(results_dir / "frequency_analysis.json")
        ngram_results = load_json(results_dir / "ngram_analysis.json")
        compression_results = load_json(results_dir / "compression_analysis.json")
        
        # Load additional results if available
        motif_results = None
        benchmark_results = None
        clustering_results = None
        
        try:
            motif_results = load_json(results_dir / "motif_analysis.json")
        except FileNotFoundError:
            logger.debug("No motif analysis results found")
        
        try:
            benchmark_results = load_json(results_dir / "performance_benchmark.json")
        except FileNotFoundError:
            logger.debug("No benchmark results found")
            
        try:
            clustering_results = load_json(results_dir / "clustering_analysis.json")
        except FileNotFoundError:
            logger.debug("No clustering analysis results found")
            
        try:
            information_results = load_json(results_dir / "information_analysis.json")
        except FileNotFoundError:
            logger.debug("No information analysis results found")
            information_results = None
            
        try:
            compiler_results = load_json(results_dir / "compiler_fingerprinting.json")
        except FileNotFoundError:
            logger.debug("No compiler fingerprinting results found")
            compiler_results = None
        
        # Generate Zipf plot
        if "frequency_distribution" in freq_results:
            zipf_plot = plot_dir / "zipf_distribution.png"
            plot_zipf_distribution(
                freq_results["frequency_distribution"]["top_50_opcodes"],
                freq_results["zipf_analysis"]["global_zipf"],
                zipf_plot
            )
            plot_files.append(zipf_plot)
        
        # Generate entropy rate plot
        if "entropy_analysis" in ngram_results:
            entropy_plot = plot_dir / "entropy_rates.png"
            baseline = ngram_results["entropy_analysis"].get("shuffled_baseline_rates")
            plot_entropy_rates(
                ngram_results["entropy_analysis"]["entropy_rates"],
                entropy_plot,
                baseline_data=baseline,
            )
            plot_files.append(entropy_plot)
        
        # Generate compression plots
        if "per_binary_results" in compression_results:
            compression_plot = plot_dir / "compression_ratios.png"
            plot_compression_ratios(
                compression_results["per_binary_results"],
                compression_plot
            )
            plot_files.append(compression_plot)
        
        # Generate frequency heatmap
        if "frequency_distribution" in freq_results:
            heatmap_plot = plot_dir / "frequency_heatmap.png"
            plot_frequency_heatmap(
                freq_results["frequency_distribution"]["top_50_opcodes"],
                20,
                heatmap_plot
            )
            plot_files.append(heatmap_plot)
        
        # Generate motif plots if available
        if motif_results:
            motif_discovery = motif_results.get("motif_discovery", {})
            if motif_discovery:
                motif_heatmap = plot_dir / "motif_heatmap.png"
                plot_motif_heatmap(motif_discovery, motif_heatmap)
                plot_files.append(motif_heatmap)
            
            positional_patterns = motif_results.get("positional_patterns", {})
            if positional_patterns:
                positional_plot = plot_dir / "positional_entropy.png"
                plot_positional_entropy(positional_patterns, positional_plot)
                plot_files.append(positional_plot)
                heatmap_plot = plot_dir / "positional_heatmap.png"
                plot_positional_heatmap(positional_patterns, heatmap_plot)
                plot_files.append(heatmap_plot)
        
        # Generate performance plots if available
        if benchmark_results:
            performance_plot = plot_dir / "performance_comparison.png"
            plot_performance_comparison(benchmark_results, performance_plot)
            plot_files.append(performance_plot)
        
        # Generate clustering plots if available
        if clustering_results:
            # NCD heatmaps
            ncd_analysis = clustering_results.get('ncd_analysis', {})
            for compressor in ['zlib', 'lzma']:
                if compressor in ncd_analysis and 'error' not in ncd_analysis[compressor]:
                    matrix_data = ncd_analysis[compressor]['matrix']
                    binary_names = ncd_analysis[compressor]['binary_names']
                    
                    ncd_heatmap = plot_dir / f"ncd_heatmap_{compressor}.png"
                    plot_distance_matrix_heatmap(matrix_data, binary_names, 
                                                f"NCD Distance Matrix ({compressor})", ncd_heatmap)
                    plot_files.append(ncd_heatmap)
        
        # Generate information-theoretic plots if available
        if information_results:
            # Entropy profiles for each binary
            per_binary = information_results.get('per_binary_analysis', {})
            for binary_name, binary_data in per_binary.items():
                if 'entropy_profiles' in binary_data:
                    entropy_plot = plot_dir / f"entropy_profile_{binary_name}.png"
                    plot_entropy_profile(binary_data['entropy_profiles'], binary_name, entropy_plot)
                    plot_files.append(entropy_plot)
                
                if 'mutual_information' in binary_data:
                    mi_plot = plot_dir / f"mi_decay_{binary_name}.png"
                    plot_mutual_information_decay(binary_data['mutual_information'], binary_name, mi_plot)
                    plot_files.append(mi_plot)
            
            # Corpus-level plots
            corpus_analysis = information_results.get('corpus_analysis', {})
            if 'space_coverage' in corpus_analysis:
                coverage_plot = plot_dir / "space_coverage.png"
                plot_space_coverage(corpus_analysis['space_coverage'], coverage_plot)
                plot_files.append(coverage_plot)

            if 'mean_mi_decay' in corpus_analysis:
                corpus_mi_plot = plot_dir / "corpus_mi_decay.png"
                plot_corpus_mi_decay(corpus_analysis, corpus_mi_plot)
                plot_files.append(corpus_mi_plot)
            
            # Dimensionality plots (from any binary with dimensionality data)
            for binary_name, binary_data in per_binary.items():
                dim_keys = [k for k in binary_data.keys() if k.startswith('dimensionality_')]
                if dim_keys:
                    dim_plot = plot_dir / f"dimensionality_{binary_name}.png"
                    plot_dimensionality_analysis(binary_data, dim_plot)
                    plot_files.append(dim_plot)
                    break  # Only plot once for the corpus
        
        # Generate compiler fingerprinting plots if available
        if compiler_results:
            # Compiler distribution plot
            compiler_dist_plot = plot_dir / "compiler_distribution.png"
            plot_compiler_distribution(compiler_results, compiler_dist_plot)
            plot_files.append(compiler_dist_plot)
            
            # Compiler feature comparison plot
            if 'aggregate_features' in compiler_results and compiler_results['aggregate_features']:
                compiler_features_plot = plot_dir / "compiler_feature_comparison.png"
                plot_compiler_feature_comparison(compiler_results, compiler_features_plot)
                plot_files.append(compiler_features_plot)
        
        # Note: Dimensionality reduction plots are already generated by the clustering pipeline
        
        logger.info(f"Generated {len(plot_files)} plots")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    return plot_files