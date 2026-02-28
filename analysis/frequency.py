"""
Frequency distribution analysis including Zipf's law testing.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json

logger = logging.getLogger(__name__)

def compute_frequency_distribution(binaries: List[Binary]) -> Dict[str, int]:
    """Compute opcode frequency distribution across all binaries."""
    frequencies = {}
    total_instructions = 0
    
    for binary in binaries:
        for opcode in binary.full_opcode_sequence:
            frequencies[opcode] = frequencies.get(opcode, 0) + 1
            total_instructions += 1
    
    logger.info(f"Computed frequencies for {len(frequencies)} unique opcodes "
               f"across {total_instructions} total instructions")
    
    return frequencies

def compute_rank_frequency(frequencies: Dict[str, int]) -> List[Tuple[str, int, int, float]]:
    """Convert frequency dict to rank-frequency list."""
    # Sort by frequency (descending)
    sorted_items = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    
    total_count = sum(frequencies.values())
    rank_frequency = []
    
    for rank, (opcode, count) in enumerate(sorted_items, 1):
        relative_freq = count / total_count
        rank_frequency.append((opcode, count, rank, relative_freq))
    
    return rank_frequency

def fit_zipf_law(rank_frequency: List[Tuple[str, int, int, float]]) -> Dict[str, float]:
    """Fit Zipf's law to the rank-frequency distribution."""
    if len(rank_frequency) < 2:
        logger.warning("Not enough data points to fit Zipf's law")
        return {"alpha": 0.0, "constant": 0.0, "r_squared": 0.0}
    
    # Extract ranks and frequencies
    ranks = np.array([item[2] for item in rank_frequency])
    frequencies = np.array([item[1] for item in rank_frequency])
    
    # Fit log-log linear relationship: log(f) = log(C) - α * log(r)
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)
    
    # Linear regression on log-log data
    coeffs = np.polyfit(log_ranks, log_frequencies, 1)
    alpha = -coeffs[0]  # Slope (negative of coefficient)
    log_constant = coeffs[1]  # Intercept
    constant = np.exp(log_constant)
    
    # Compute R-squared
    predicted_log_freq = log_constant - alpha * log_ranks
    ss_res = np.sum((log_frequencies - predicted_log_freq) ** 2)
    ss_tot = np.sum((log_frequencies - np.mean(log_frequencies)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    result = {
        "alpha": float(alpha),
        "constant": float(constant),
        "r_squared": float(r_squared)
    }
    
    logger.info(f"Zipf's law fit: α={alpha:.3f}, C={constant:.1f}, R²={r_squared:.3f}")
    
    return result

def fit_zipf_with_ci(rank_frequency: List[Tuple[str, int, int, float]],
                     n_bootstrap: int = 200) -> Dict[str, float]:
    """Fit Zipf's law and estimate 95% CI on α via bootstrap resampling."""
    if len(rank_frequency) < 5:
        return {"alpha": 0.0, "alpha_ci_low": 0.0, "alpha_ci_high": 0.0,
                "constant": 0.0, "r_squared": 0.0}

    point = fit_zipf_law(rank_frequency)
    alpha_point = point["alpha"]

    # Bootstrap: resample (rank, freq) pairs with replacement, refit α each time
    log_ranks = np.array([np.log(item[2]) for item in rank_frequency])
    log_freqs  = np.array([np.log(item[1]) for item in rank_frequency])
    n = len(log_ranks)

    boot_alphas = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        coeffs = np.polyfit(log_ranks[idx], log_freqs[idx], 1)
        boot_alphas.append(-coeffs[0])

    ci_low  = float(np.percentile(boot_alphas, 2.5))
    ci_high = float(np.percentile(boot_alphas, 97.5))

    point["alpha_ci_low"]  = ci_low
    point["alpha_ci_high"] = ci_high
    logger.info(f"Zipf α = {alpha_point:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")
    return point


def analyze_per_binary_frequencies(binaries: List[Binary]) -> Dict[str, Dict]:
    """Analyze frequency distributions for each binary individually."""
    per_binary_stats = {}
    
    for binary in binaries:
        frequencies = {}
        for opcode in binary.full_opcode_sequence:
            frequencies[opcode] = frequencies.get(opcode, 0) + 1
        
        if frequencies:
            rank_frequency = compute_rank_frequency(frequencies)
            zipf_params = fit_zipf_law(rank_frequency)
            
            per_binary_stats[binary.name] = {
                "instruction_count": binary.instruction_count,
                "unique_opcodes": len(frequencies),
                "zipf_alpha": zipf_params["alpha"],
                "zipf_r_squared": zipf_params["r_squared"],
                "top_10_opcodes": rank_frequency[:10]
            }
    
    return per_binary_stats

def run_frequency_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """Run complete frequency analysis and save results."""
    logger.info("Running frequency distribution analysis...")
    
    # Global frequency analysis
    frequencies = compute_frequency_distribution(binaries)
    rank_frequency = compute_rank_frequency(frequencies)
    zipf_params = fit_zipf_with_ci(rank_frequency)
    
    # Per-binary analysis
    per_binary_stats = analyze_per_binary_frequencies(binaries)
    
    # Compile results
    results = {
        "corpus_stats": {
            "total_binaries": len(binaries),
            "total_instructions": sum(b.instruction_count for b in binaries),
            "unique_opcodes": len(frequencies),
            "vocabulary_size": len(frequencies)
        },
        "zipf_analysis": {
            "global_zipf": zipf_params,
            "interpretation": interpret_zipf_alpha(zipf_params["alpha"]),
            "reference_values": {
                "natural_language_alpha": 1.0,
                "note": "English text α≈1.0; values >1 indicate heavier tail than natural language"
            },
        },
        "frequency_distribution": {
            "top_50_opcodes": rank_frequency[:50],
            "bottom_10_opcodes": rank_frequency[-10:] if len(rank_frequency) >= 10 else rank_frequency
        },
        "per_binary_analysis": per_binary_stats
    }
    
    # Save results
    save_json(results, output_dir / "frequency_analysis.json")
    
    # Save frequency table as CSV-like structure
    frequency_table = []
    for opcode, count, rank, rel_freq in rank_frequency:
        frequency_table.append({
            "opcode": opcode,
            "count": count,
            "rank": rank,
            "relative_frequency": rel_freq
        })
    save_json(frequency_table, output_dir / "frequency_table.json")
    
    logger.info("Frequency analysis completed")
    return results

def interpret_zipf_alpha(alpha: float) -> str:
    """Provide interpretation of the Zipf alpha parameter."""
    if alpha < 0.5:
        return "Very flat distribution - less concentrated than typical natural language"
    elif alpha < 0.8:
        return "Moderately flat distribution - somewhat less concentrated than natural language"
    elif alpha < 1.2:
        return "Similar to natural language (α ≈ 1.0) - classic Zipf distribution"
    elif alpha < 1.5:
        return "More concentrated than natural language - highly skewed distribution"
    else:
        return "Extremely concentrated distribution - dominated by very few opcodes"