"""
Information-theoretic analysis for binary instruction sequences.

Implements sliding window entropy, mutual information analysis, and intrinsic
dimensionality estimation to understand the information content and structure
of compiled programs.
"""

import numpy as np
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json
from analysis.ngrams import extract_ngrams

logger = logging.getLogger(__name__)

# Performance caps: keep analysis tractable on large binaries
_MAX_SEQ_FOR_MI      = 50_000   # max instructions used for MI and dim-reduction
_MAX_ENTROPY_TRACE   = 500      # max stored data-points in sliding-window entropy trace

class InformationAnalyzer:
    """Analyzer for information-theoretic properties of instruction sequences."""
    
    def __init__(self, vocab_size: int):
        """
        Initialize analyzer with vocabulary size for theoretical calculations.
        
        Args:
            vocab_size: Size of the opcode vocabulary
        """
        self.vocab_size = vocab_size
        self.max_entropy = np.log2(vocab_size) if vocab_size > 0 else 0.0
    
    def compute_sliding_window_entropy(self, sequence: List[str], window_size: int, 
                                     step_size: int = 1) -> Tuple[List[float], List[int]]:
        """
        Compute Shannon entropy over sliding windows of the sequence.
        
        Args:
            sequence: Opcode sequence
            window_size: Size of sliding window
            step_size: Step size for window advancement
            
        Returns:
            entropies: List of entropy values
            positions: List of window center positions
        """
        if len(sequence) < window_size:
            logger.warning(f"Sequence too short for window size {window_size}")
            return [], []
        
        entropies = []
        positions = []
        
        for start in range(0, len(sequence) - window_size + 1, step_size):
            window = sequence[start:start + window_size]
            
            # Compute entropy for this window
            counter = Counter(window)
            total = len(window)
            
            entropy = 0.0
            for count in counter.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            
            entropies.append(entropy)
            positions.append(start + window_size // 2)  # Window center
        
        return entropies, positions
    
    def compute_mutual_information(self, sequence: List[str], max_lag: int = 50) -> Dict[int, float]:
        """
        Compute mutual information between positions separated by various lags.
        
        Args:
            sequence: Opcode sequence
            max_lag: Maximum lag to analyze
            
        Returns:
            Dictionary mapping lag -> mutual information
        """
        if len(sequence) < max_lag + 1:
            logger.warning(f"Sequence too short for max lag {max_lag}")
            max_lag = len(sequence) - 1
        
        mutual_info = {}
        
        for lag in range(1, max_lag + 1):
            # Extract pairs (X_i, X_{i+lag})
            if len(sequence) <= lag:
                break
                
            x_values = sequence[:-lag]
            y_values = sequence[lag:]
            
            if not x_values or not y_values:
                continue
            
            # Compute joint distribution P(X_i, X_{i+lag})
            joint_counter = Counter(zip(x_values, y_values))
            
            # Compute marginal distributions
            x_counter = Counter(x_values)
            y_counter = Counter(y_values)
            
            total_pairs = len(x_values)
            
            # Compute mutual information
            mi = 0.0
            for (x, y), joint_count in joint_counter.items():
                p_joint = joint_count / total_pairs
                p_x = x_counter[x] / total_pairs
                p_y = y_counter[y] / total_pairs
                
                if p_joint > 0 and p_x > 0 and p_y > 0:
                    mi += p_joint * np.log2(p_joint / (p_x * p_y))
            
            mutual_info[lag] = mi
        
        return mutual_info
    
    def estimate_intrinsic_dimensionality_pca(self, ngram_matrix: np.ndarray, 
                                            variance_threshold: float = 0.95) -> Dict:
        """
        Estimate intrinsic dimensionality using PCA on n-gram vectors.
        
        Args:
            ngram_matrix: Matrix where rows are positions, columns are n-grams
            variance_threshold: Cumulative variance threshold
            
        Returns:
            Dictionary with dimensionality estimates
        """
        if ngram_matrix.shape[0] < 2 or ngram_matrix.shape[1] < 2:
            return {"error": "Insufficient data for PCA dimensionality estimation"}
        
        try:
            # Fit PCA
            pca = PCA()
            pca.fit(ngram_matrix)
            
            # Find number of components for variance threshold
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components_95 = np.argmax(cumulative_variance >= variance_threshold) + 1
            
            # Effective dimensionality (participation ratio)
            eigenvalues = pca.explained_variance_
            participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
            
            return {
                "method": "PCA",
                "total_dimensions": ngram_matrix.shape[1],
                "components_for_95_variance": int(n_components_95),
                "effective_dimensionality": float(participation_ratio),
                "explained_variance_ratio": pca.explained_variance_ratio_[:10].tolist(),  # Top 10
                "cumulative_variance": cumulative_variance[:10].tolist()
            }
            
        except Exception as e:
            logger.error(f"PCA dimensionality estimation failed: {e}")
            return {"error": str(e)}
    
    def estimate_intrinsic_dimensionality_mle(self, ngram_matrix: np.ndarray, 
                                            k: int = 5) -> Dict:
        """
        Estimate intrinsic dimensionality using Maximum Likelihood Estimation.
        
        Based on Levina & Bickel (2005) method using k-nearest neighbors.
        
        Args:
            ngram_matrix: Data matrix
            k: Number of nearest neighbors to use
            
        Returns:
            Dictionary with MLE dimensionality estimate
        """
        if ngram_matrix.shape[0] < k + 1:
            return {"error": f"Need at least {k+1} samples for k={k} MLE estimation"}
        
        try:
            # Fit k-NN model
            nbrs = NearestNeighbors(n_neighbors=k+1)  # +1 because first neighbor is the point itself
            nbrs.fit(ngram_matrix)
            
            distances, _ = nbrs.kneighbors(ngram_matrix)
            
            # Remove first column (distance to self, which is 0)
            distances = distances[:, 1:]
            
            # MLE estimate
            n_samples = distances.shape[0]
            dimensionalities = []
            
            for i in range(n_samples):
                # Get distances for this point
                dists = distances[i]
                
                # Remove zero distances to avoid log(0)
                dists = dists[dists > 0]
                
                if len(dists) < 2:
                    continue
                
                # MLE formula: d = (k-1) / sum(log(r_k / r_i)) for i = 1..k-1
                r_k = dists[-1]  # Farthest neighbor
                
                if r_k > 0:
                    log_ratios = []
                    for j in range(len(dists) - 1):
                        r_i = dists[j]
                        if r_i > 0:
                            log_ratios.append(np.log(r_k / r_i))
                    
                    if log_ratios:
                        sum_log_ratios = np.sum(log_ratios)
                        if sum_log_ratios > 0:
                            dim_estimate = (len(dists) - 1) / sum_log_ratios
                            dimensionalities.append(dim_estimate)
            
            if dimensionalities:
                mean_dim = np.mean(dimensionalities)
                std_dim = np.std(dimensionalities)
                
                return {
                    "method": "MLE",
                    "k_neighbors": k,
                    "mean_dimensionality": float(mean_dim),
                    "std_dimensionality": float(std_dim),
                    "valid_estimates": len(dimensionalities),
                    "total_points": n_samples
                }
            else:
                return {"error": "No valid dimensionality estimates computed"}
                
        except Exception as e:
            logger.error(f"MLE dimensionality estimation failed: {e}")
            return {"error": str(e)}
    
    def analyze_program_space_coverage(self, binaries: List[Binary]) -> Dict:
        """
        Analyze how much of the theoretical program space is covered.
        
        Args:
            binaries: List of binary objects
            
        Returns:
            Dictionary with space coverage analysis
        """
        coverage_analysis = {}
        
        # Collect all sequences
        all_sequences = []
        for binary in binaries:
            all_sequences.append(binary.full_opcode_sequence)
        
        # Analyze for different n-gram lengths
        for n in [1, 2, 3, 4]:
            all_ngrams = set()
            total_ngrams = 0
            
            for sequence in all_sequences:
                if len(sequence) >= n:
                    ngrams = extract_ngrams(sequence, n)
                    all_ngrams.update(ngrams)
                    total_ngrams += len(ngrams)
            
            # Theoretical maximum
            theoretical_max = self.vocab_size ** n
            
            # Coverage metrics
            unique_ngrams = len(all_ngrams)
            coverage_ratio = unique_ngrams / theoretical_max if theoretical_max > 0 else 0.0
            
            # Redundancy (how often n-grams repeat)
            redundancy = (total_ngrams - unique_ngrams) / max(total_ngrams, 1)
            
            coverage_analysis[f"{n}gram"] = {
                "unique_ngrams": unique_ngrams,
                "total_ngrams": total_ngrams,
                "theoretical_maximum": theoretical_max,
                "coverage_ratio": coverage_ratio,
                "redundancy": redundancy,
                "interpretation": interpret_coverage_ratio(coverage_ratio)
            }
        
        return coverage_analysis

def interpret_coverage_ratio(ratio: float) -> str:
    """Interpret the coverage ratio of program space."""
    if ratio > 0.5:
        return "High coverage - programs explore much of the theoretical space"
    elif ratio > 0.1:
        return "Moderate coverage - programs use a significant portion of possible patterns"
    elif ratio > 0.01:
        return "Low coverage - programs constrained to small subset of possibilities"
    else:
        return "Very low coverage - programs highly constrained to tiny manifold"

def create_ngram_position_matrix(sequence: List[str], n: int, window_size: int) -> np.ndarray:
    """
    Create matrix where rows are sequence positions and columns are n-gram features.
    
    This is used for dimensionality analysis of local sequence structure.
    """
    if len(sequence) < window_size + n - 1:
        logger.warning("Sequence too short for n-gram position matrix")
        return np.array([])
    
    # Get all possible n-grams in the sequence
    all_ngrams = extract_ngrams(sequence, n)
    unique_ngrams = sorted(set(all_ngrams))
    ngram_to_idx = {ngram: i for i, ngram in enumerate(unique_ngrams)}
    
    # Create matrix
    num_positions = len(sequence) - window_size + 1
    num_features = len(unique_ngrams)
    
    if num_positions < 1 or num_features < 1:
        return np.array([])
    
    matrix = np.zeros((num_positions, num_features))
    
    for pos in range(num_positions):
        window = sequence[pos:pos + window_size]
        
        if len(window) >= n:
            # Count n-grams in this window
            window_ngrams = extract_ngrams(window, n)
            ngram_counts = Counter(window_ngrams)
            
            for ngram, count in ngram_counts.items():
                if ngram in ngram_to_idx:
                    matrix[pos, ngram_to_idx[ngram]] = count
    
    return matrix

def run_information_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """Run complete information-theoretic analysis."""
    logger.info("Running information-theoretic analysis...")
    
    if not binaries:
        logger.error("No binaries provided for information analysis")
        return {}
    
    # Build vocabulary for theoretical calculations
    from utils.helpers import build_vocabulary
    vocab = build_vocabulary(binaries)
    vocab_size = len(vocab)
    
    analyzer = InformationAnalyzer(vocab_size)
    
    results = {
        "vocabulary_info": {
            "vocab_size": vocab_size,
            "max_theoretical_entropy": analyzer.max_entropy
        },
        "per_binary_analysis": {},
        "corpus_analysis": {}
    }
    
    # Per-binary analysis
    for binary in binaries:
        sequence = binary.full_opcode_sequence
        
        if len(sequence) < 10:  # Skip very short sequences
            logger.warning(f"Skipping {binary.name} - sequence too short for analysis")
            continue
        
        logger.info(f"Analyzing information content of {binary.name}")
        
        binary_results = {}
        
        # Sliding window entropy
        try:
            window_sizes = [16, 32, 64]
            entropy_profiles = {}
            
            for window_size in window_sizes:
                if len(sequence) >= window_size:
                    entropies, positions = analyzer.compute_sliding_window_entropy(
                        sequence, window_size, step_size=max(1, window_size // 4)
                    )

                    if entropies:
                        # Downsample trace before storing so JSON stays manageable
                        if len(entropies) > _MAX_ENTROPY_TRACE:
                            step = max(1, len(entropies) // _MAX_ENTROPY_TRACE)
                            entropies_ds = entropies[::step]
                            positions_ds = positions[::step]
                        else:
                            entropies_ds = entropies
                            positions_ds = positions

                        entropy_profiles[f"window_{window_size}"] = {
                            "entropies": entropies_ds,
                            "positions": positions_ds,
                            "mean_entropy": float(np.mean(entropies)),
                            "std_entropy": float(np.std(entropies)),
                            "min_entropy": float(np.min(entropies)),
                            "max_entropy": float(np.max(entropies)),
                            "entropy_range": float(np.max(entropies) - np.min(entropies)),
                            "num_windows": len(entropies)
                        }
            
            binary_results["entropy_profiles"] = entropy_profiles
            
        except Exception as e:
            logger.error(f"Entropy profiling failed for {binary.name}: {e}")
            binary_results["entropy_profiles"] = {"error": str(e)}
        
        # Mutual information decay (capped sequence for performance)
        try:
            seq_for_mi = sequence[:_MAX_SEQ_FOR_MI]
            max_lag = min(50, len(seq_for_mi) // 2)
            mutual_info = analyzer.compute_mutual_information(seq_for_mi, max_lag)
            
            if mutual_info:
                # Shuffled baseline: destroy higher-order dependencies while
                # preserving the unigram distribution
                rng = np.random.default_rng(42)
                shuffled_seq = list(seq_for_mi)
                rng.shuffle(shuffled_seq)
                shuffled_mi = analyzer.compute_mutual_information(shuffled_seq, max_lag)

                # Analyze decay pattern
                lags = sorted(mutual_info.keys())
                mi_values = [mutual_info[lag] for lag in lags]
                decay_analysis = analyze_mi_decay(lags, mi_values)

                # MI gain over shuffled (how much structure beyond unigrams)
                mi_gain = {
                    lag: float(mutual_info.get(lag, 0) - shuffled_mi.get(lag, 0))
                    for lag in lags
                }

                binary_results["mutual_information"] = {
                    "mi_by_lag": {str(k): v for k, v in mutual_info.items()},
                    "mi_shuffled_by_lag": {str(k): v for k, v in shuffled_mi.items()},
                    "mi_gain_over_shuffled": {str(k): v for k, v in mi_gain.items()},
                    "decay_analysis": decay_analysis,
                    "max_lag_analyzed": max_lag
                }
            
        except Exception as e:
            logger.error(f"Mutual information analysis failed for {binary.name}: {e}")
            binary_results["mutual_information"] = {"error": str(e)}
        
        # Local dimensionality analysis (capped sequence for performance)
        try:
            seq_for_dim = sequence[:_MAX_SEQ_FOR_MI]
            for n in [2, 3]:
                window_size = min(32, len(seq_for_dim) // 4)
                if window_size >= n:
                    ngram_matrix = create_ngram_position_matrix(seq_for_dim, n, window_size)
                    
                    if ngram_matrix.size > 0:
                        # PCA-based dimensionality
                        pca_dim = analyzer.estimate_intrinsic_dimensionality_pca(ngram_matrix)
                        
                        # MLE-based dimensionality (if we have enough data)
                        if ngram_matrix.shape[0] >= 10:
                            mle_dim = analyzer.estimate_intrinsic_dimensionality_mle(ngram_matrix, k=5)
                        else:
                            mle_dim = {"error": "Insufficient data for MLE estimation"}
                        
                        binary_results[f"dimensionality_{n}gram"] = {
                            "pca": pca_dim,
                            "mle": mle_dim,
                            "matrix_shape": list(ngram_matrix.shape)
                        }
            
        except Exception as e:
            logger.error(f"Dimensionality analysis failed for {binary.name}: {e}")
            binary_results["dimensionality_analysis"] = {"error": str(e)}
        
        results["per_binary_analysis"][binary.name] = binary_results
    
    # Corpus-level analysis
    try:
        # Program space coverage
        coverage_analysis = analyzer.analyze_program_space_coverage(binaries)
        results["corpus_analysis"]["space_coverage"] = coverage_analysis

        # Corpus manifold dimensionality for n=2 and n=3
        for n in [2, 3]:
            manifold_dim = estimate_corpus_manifold_dimensionality(binaries, n=n)
            results["corpus_analysis"][f"manifold_dimensionality_{n}gram"] = manifold_dim
        
        # Global entropy statistics
        all_entropies = []
        for binary_name, binary_analysis in results["per_binary_analysis"].items():
            entropy_profiles = binary_analysis.get("entropy_profiles", {})
            for profile_name, profile_data in entropy_profiles.items():
                if "entropies" in profile_data:
                    all_entropies.extend(profile_data["entropies"])
        
        if all_entropies:
            results["corpus_analysis"]["global_entropy_stats"] = {
                "mean_entropy": float(np.mean(all_entropies)),
                "std_entropy": float(np.std(all_entropies)),
                "min_entropy": float(np.min(all_entropies)),
                "max_entropy": float(np.max(all_entropies)),
                "entropy_vs_max": float(np.mean(all_entropies) / analyzer.max_entropy) if analyzer.max_entropy > 0 else 0
            }

        # Corpus-level MI decay: average real and shuffled MI across all binaries
        all_real_mi: dict = {}   # lag (int) -> list of per-binary MI values
        all_shuf_mi: dict = {}
        for binary_data in results["per_binary_analysis"].values():
            mi_info = binary_data.get("mutual_information", {})
            for lag_str, val in mi_info.get("mi_by_lag", {}).items():
                lag = int(lag_str)
                all_real_mi.setdefault(lag, []).append(val)
            for lag_str, val in mi_info.get("mi_shuffled_by_lag", {}).items():
                lag = int(lag_str)
                all_shuf_mi.setdefault(lag, []).append(val)

        if all_real_mi:
            lags_sorted = sorted(all_real_mi)
            results["corpus_analysis"]["mean_mi_decay"] = {
                str(lag): float(np.mean(all_real_mi[lag])) for lag in lags_sorted
            }
        if all_shuf_mi:
            lags_sorted = sorted(all_shuf_mi)
            results["corpus_analysis"]["mean_shuffled_mi_decay"] = {
                str(lag): float(np.mean(all_shuf_mi[lag])) for lag in lags_sorted
            }

    except Exception as e:
        logger.error(f"Corpus analysis failed: {e}")
        results["corpus_analysis"] = {"error": str(e)}
    
    # Save results
    save_json(results, output_dir / "information_analysis.json")
    
    logger.info("Information-theoretic analysis completed")
    return results

def estimate_corpus_manifold_dimensionality(binaries: List[Binary], n: int = 2,
                                            top_k: int = 200) -> Dict:
    """
    Estimate the intrinsic dimensionality of the corpus manifold.

    Each binary becomes one data point: a normalized n-gram frequency vector
    restricted to the top-K most common n-grams across the corpus.
    PCA on this N×K matrix reveals how many directions capture 95% of variance —
    a low number supports the "tiny structured manifold" hypothesis.
    """
    # Collect corpus-wide n-gram counts
    corpus_counts: Counter = Counter()
    binary_ngram_counts = []

    for binary in binaries:
        seq = binary.full_opcode_sequence[:_MAX_SEQ_FOR_MI]
        if len(seq) < n:
            binary_ngram_counts.append(Counter())
            continue
        counts = Counter(extract_ngrams(seq, n))
        corpus_counts.update(counts)
        binary_ngram_counts.append(counts)

    if not corpus_counts:
        return {"error": "No n-gram data available"}

    # Restrict to top-K most common n-grams
    top_ngrams = [ng for ng, _ in corpus_counts.most_common(top_k)]
    K = len(top_ngrams)
    if K < 2:
        return {"error": "Insufficient unique n-grams"}

    ng_idx = {ng: i for i, ng in enumerate(top_ngrams)}

    # Build N × K frequency matrix (rows = binaries, normalized)
    rows = []
    for counts in binary_ngram_counts:
        vec = np.zeros(K)
        total = sum(counts.get(ng, 0) for ng in top_ngrams)
        if total > 0:
            for ng, i in ng_idx.items():
                vec[i] = counts.get(ng, 0) / total
        rows.append(vec)

    matrix = np.array(rows)
    N = matrix.shape[0]

    if N < 2:
        return {"error": "Need at least 2 binaries"}

    try:
        n_components = min(N - 1, K)
        pca = PCA(n_components=n_components)
        pca.fit(matrix)

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_for_95 = int(np.argmax(cum_var >= 0.95) + 1)
        eigenvalues = pca.explained_variance_
        participation_ratio = float((np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2))

        return {
            "n_gram": n,
            "n_binaries": N,
            "n_features": K,
            "components_for_95_variance": n_for_95,
            "participation_ratio": participation_ratio,
            "explained_variance_ratio_top10": pca.explained_variance_ratio_[:10].tolist(),
            "cumulative_variance_top10": cum_var[:10].tolist(),
            "interpretation": (
                f"Corpus manifold: {n_for_95}/{K} {n}-gram dimensions capture 95% of variance "
                f"({100 * n_for_95 / K:.1f}% of theoretical space used)"
            )
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_mi_decay(lags: List[int], mi_values: List[float]) -> Dict:
    """Analyze the decay pattern of mutual information."""
    if len(lags) < 3:
        return {"error": "Insufficient data points for decay analysis"}
    
    try:
        # Simple decay metrics
        initial_mi = mi_values[0] if mi_values else 0
        final_mi = mi_values[-1] if mi_values else 0
        
        # Decay rate (simple linear fit)
        if initial_mi > 0:
            decay_rate = (initial_mi - final_mi) / initial_mi
        else:
            decay_rate = 0.0
        
        # Find half-life (lag where MI drops to 50% of initial)
        half_life = None
        target = initial_mi / 2
        
        for i, mi in enumerate(mi_values):
            if mi <= target:
                half_life = lags[i]
                break
        
        return {
            "initial_mi": initial_mi,
            "final_mi": final_mi,
            "decay_rate": decay_rate,
            "half_life": half_life,
            "interpretation": interpret_mi_decay(decay_rate, half_life)
        }
        
    except Exception as e:
        return {"error": str(e)}

def interpret_mi_decay(decay_rate: float, half_life: Optional[int]) -> str:
    """Interpret mutual information decay patterns."""
    if decay_rate > 0.8:
        return "Rapid MI decay - strong local structure, weak long-range dependencies"
    elif decay_rate > 0.5:
        return "Moderate MI decay - balanced local and long-range structure"
    elif decay_rate > 0.2:
        return "Slow MI decay - significant long-range dependencies"
    else:
        return "Minimal MI decay - strong long-range correlations"