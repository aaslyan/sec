"""
Compression and complexity analysis module.
"""

import zlib
import lzma
import numpy as np
import logging
from typing import List, Dict
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json, build_vocabulary, encode_sequence

logger = logging.getLogger(__name__)

def compute_lz_complexity(sequence: List[int]) -> int:
    """
    LZ78 complexity: count of unique phrases produced by LZ78 greedy parsing.

    This is a standard proxy for Kolmogorov complexity.  Runs in O(n) average
    time via set lookups on tuple prefixes — no string-encoding ambiguity.
    """
    if not sequence:
        return 0
    dictionary: set = {()}
    current: tuple = ()
    complexity = 0
    for token in sequence:
        extended = current + (token,)
        if extended in dictionary:
            current = extended
        else:
            dictionary.add(extended)
            current = ()
            complexity += 1
    if current:           # partial phrase at end
        complexity += 1
    return complexity

def compute_compression_ratios(binaries: List[Binary]) -> List[Dict]:
    """Compute compression ratios for each binary using different algorithms."""
    vocab = build_vocabulary(binaries)
    results = []
    
    for binary in binaries:
        # Encode sequence as bytes
        opcodes = binary.full_opcode_sequence
        encoded_sequence = encode_sequence(opcodes, vocab)
        
        # Convert to bytes for compression (handle values > 255)
        # Use struct or numpy to pack integers properly
        import struct
        try:
            # Pack as 16-bit integers if vocabulary is large
            if max(encoded_sequence) > 255:
                original_bytes = struct.pack(f'<{len(encoded_sequence)}H', *encoded_sequence)
            else:
                original_bytes = bytes(encoded_sequence)
        except (ValueError, struct.error):
            # Fallback: mod by 256 to fit in byte range
            encoded_sequence_safe = [x % 256 for x in encoded_sequence]
            original_bytes = bytes(encoded_sequence_safe)
        
        original_size = len(original_bytes)
        
        if original_size == 0:
            logger.warning(f"Empty sequence for binary {binary.name}")
            continue
        
        # Zlib compression
        try:
            zlib_compressed = zlib.compress(original_bytes)
            zlib_ratio = len(zlib_compressed) / original_size
        except Exception as e:
            logger.warning(f"Zlib compression failed for {binary.name}: {e}")
            zlib_ratio = 1.0
        
        # LZMA compression
        try:
            lzma_compressed = lzma.compress(original_bytes)
            lzma_ratio = len(lzma_compressed) / original_size
        except Exception as e:
            logger.warning(f"LZMA compression failed for {binary.name}: {e}")
            lzma_ratio = 1.0
        
        # LZ complexity — O(n) average with LZ78; cap at 100k tokens for safety
        try:
            cap = 100_000
            sample = encoded_sequence[:cap]
            lz_complexity = compute_lz_complexity(sample)
            if len(encoded_sequence) > cap:
                lz_complexity = int(lz_complexity * len(encoded_sequence) / cap)
            
            # Normalize by theoretical minimum (log2 of length)
            theoretical_min = np.log2(max(1, len(encoded_sequence)))
            normalized_complexity = lz_complexity / theoretical_min if theoretical_min > 0 else 1.0
        except Exception as e:
            logger.warning(f"LZ complexity computation failed for {binary.name}: {e}")
            lz_complexity = len(encoded_sequence)
            normalized_complexity = 1.0
        
        result = {
            "binary_name": binary.name,
            "original_size": original_size,
            "instruction_count": len(opcodes),
            "zlib_compressed_size": len(zlib_compressed) if 'zlib_compressed' in locals() else original_size,
            "zlib_ratio": zlib_ratio,
            "lzma_compressed_size": len(lzma_compressed) if 'lzma_compressed' in locals() else original_size,
            "lzma_ratio": lzma_ratio,
            "lz_complexity": lz_complexity,
            "normalized_lz_complexity": normalized_complexity
        }
        
        results.append(result)
        
        logger.debug(f"{binary.name}: zlib={zlib_ratio:.3f}, lzma={lzma_ratio:.3f}, "
                    f"lz_complexity={lz_complexity}")
    
    return results

def generate_random_baseline(vocab_size: int, sequence_length: int, num_samples: int = 10) -> Dict:
    """Generate random sequences as baseline for compression comparison."""
    random_results = []
    
    for i in range(num_samples):
        # Generate random sequence
        random_sequence = np.random.randint(0, vocab_size, sequence_length)
        import struct
        if vocab_size > 255:
            random_bytes = struct.pack(f'<{sequence_length}H', *random_sequence.tolist())
        else:
            random_bytes = bytes(random_sequence.tolist())
        
        # Compress random sequence
        zlib_compressed = zlib.compress(random_bytes)
        lzma_compressed = lzma.compress(random_bytes)
        lz_complexity = compute_lz_complexity(random_sequence.tolist())
        
        random_results.append({
            "zlib_ratio": len(zlib_compressed) / len(random_bytes),
            "lzma_ratio": len(lzma_compressed) / len(random_bytes),
            "lz_complexity": lz_complexity
        })
    
    # Compute averages
    avg_result = {
        "zlib_ratio": np.mean([r["zlib_ratio"] for r in random_results]),
        "lzma_ratio": np.mean([r["lzma_ratio"] for r in random_results]),
        "lz_complexity": np.mean([r["lz_complexity"] for r in random_results]),
        "num_samples": num_samples
    }
    
    return avg_result

def generate_unigram_shuffled_baseline(binaries: List[Binary],
                                       num_shuffles: int = 5,
                                       rng_seed: int = 42) -> Dict:
    """
    Compression baseline using unigram-matched shuffled sequences.

    For each binary, randomly permute its opcode-ID sequence and compress
    the permuted bytes.  Averaging over *num_shuffles* permutations per
    binary gives a stable estimate.

    This baseline preserves the marginal (Zipfian) frequency distribution
    while destroying all sequential structure, so the gap between the
    corpus mean and this baseline isolates *sequential* redundancy from
    the contribution of mere unigram skew.
    """
    vocab = build_vocabulary(binaries)
    import struct

    rng = np.random.default_rng(rng_seed)
    shuffle_results: list = []

    for binary in binaries:
        encoded = np.array(encode_sequence(binary.full_opcode_sequence, vocab),
                           dtype=np.int32)
        if len(encoded) == 0:
            continue

        for _ in range(num_shuffles):
            perm = rng.permutation(encoded)

            # Pack identically to compute_compression_ratios
            try:
                if perm.max() > 255:
                    perm_bytes = struct.pack(f'<{len(perm)}H', *perm.tolist())
                else:
                    perm_bytes = bytes(perm.tolist())
            except (ValueError, struct.error):
                perm_bytes = bytes((perm % 256).tolist())

            n = len(perm_bytes)
            try:
                zlib_r = len(zlib.compress(perm_bytes)) / n
            except Exception:
                zlib_r = 1.0
            try:
                lzma_r = len(lzma.compress(perm_bytes)) / n
            except Exception:
                lzma_r = 1.0
            try:
                lz = compute_lz_complexity(perm.tolist())
            except Exception:
                lz = len(perm)

            shuffle_results.append({
                "zlib_ratio": zlib_r,
                "lzma_ratio": lzma_r,
                "lz_complexity": lz,
            })

    if not shuffle_results:
        return {"zlib_ratio": 1.0, "lzma_ratio": 1.0, "lz_complexity": 0,
                "num_samples": 0}

    return {
        "zlib_ratio": float(np.mean([r["zlib_ratio"] for r in shuffle_results])),
        "lzma_ratio": float(np.mean([r["lzma_ratio"] for r in shuffle_results])),
        "lz_complexity": float(np.mean([r["lz_complexity"] for r in shuffle_results])),
        "num_samples": len(shuffle_results),
        "description": (
            "Sequences with identical unigram distribution but random order; "
            "gap to corpus mean = sequential redundancy only."
        ),
    }


def analyze_compression_vs_size(compression_results: List[Dict]) -> Dict:
    """Analyze correlation between binary size and compression ratio."""
    if len(compression_results) < 2:
        return {"interpretation": "Insufficient data for size correlation analysis"}
    
    sizes = [r["instruction_count"] for r in compression_results]
    zlib_ratios = [r["zlib_ratio"] for r in compression_results]
    
    # Compute correlation
    correlation = np.corrcoef(sizes, zlib_ratios)[0, 1] if len(sizes) > 1 else 0.0
    
    interpretation = ""
    if abs(correlation) < 0.2:
        interpretation = "No significant correlation between size and compression ratio"
    elif correlation > 0.5:
        interpretation = "Larger binaries tend to be less compressible (more diverse patterns)"
    elif correlation < -0.5:
        interpretation = "Larger binaries tend to be more compressible (more repetitive patterns)"
    else:
        interpretation = f"Weak correlation between size and compression ratio (r={correlation:.3f})"
    
    return {
        "correlation": correlation,
        "interpretation": interpretation
    }

def run_compression_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """Run complete compression and complexity analysis."""
    logger.info("Running compression analysis...")
    
    # Compute compression ratios for all binaries
    compression_results = compute_compression_ratios(binaries)
    
    if not compression_results:
        logger.error("No compression results generated")
        return {}
    
    # Uniform-random baseline (legacy: uniform over vocab, ignores unigram skew)
    vocab = build_vocabulary(binaries)
    avg_length = int(np.mean([binary.instruction_count for binary in binaries]))
    random_baseline = generate_random_baseline(len(vocab), avg_length)

    # Unigram-shuffled baseline (preserves marginal distribution; gap = sequential only)
    unigram_shuffled_baseline = generate_unigram_shuffled_baseline(binaries)
    
    # Compute statistics
    zlib_ratios = [r["zlib_ratio"] for r in compression_results]
    lzma_ratios = [r["lzma_ratio"] for r in compression_results]
    lz_complexities = [r["normalized_lz_complexity"] for r in compression_results]
    
    statistics = {
        "zlib": {
            "mean": np.mean(zlib_ratios),
            "std": np.std(zlib_ratios),
            "min": np.min(zlib_ratios),
            "max": np.max(zlib_ratios),
            "median": np.median(zlib_ratios)
        },
        "lzma": {
            "mean": np.mean(lzma_ratios),
            "std": np.std(lzma_ratios),
            "min": np.min(lzma_ratios),
            "max": np.max(lzma_ratios),
            "median": np.median(lzma_ratios)
        },
        "lz_complexity": {
            "mean": np.mean(lz_complexities),
            "std": np.std(lz_complexities),
            "min": np.min(lz_complexities),
            "max": np.max(lz_complexities),
            "median": np.median(lz_complexities)
        }
    }
    
    # Size correlation analysis
    size_correlation = analyze_compression_vs_size(compression_results)
    
    # Compile results
    results = {
        "compression_statistics": statistics,
        "random_baseline": random_baseline,
        "unigram_shuffled_baseline": unigram_shuffled_baseline,
        "size_correlation": size_correlation,
        "per_binary_results": compression_results,
        "interpretation": interpret_compression_results(statistics, random_baseline)
    }
    
    # Save results
    save_json(results, output_dir / "compression_analysis.json")
    
    # Save detailed per-binary results
    save_json(compression_results, output_dir / "compression_ratios.json")
    
    logger.info("Compression analysis completed")
    return results

def interpret_compression_results(statistics: Dict, random_baseline: Dict) -> Dict:
    """Interpret the compression analysis results."""
    zlib_mean = statistics["zlib"]["mean"]
    random_zlib = random_baseline["zlib_ratio"]
    
    # Compare to random baseline
    improvement_over_random = (random_zlib - zlib_mean) / random_zlib if random_zlib > 0 else 0.0
    
    interpretation = {
        "compressibility": "",
        "vs_random": "",
        "structure_level": ""
    }
    
    # Compressibility interpretation
    if zlib_mean > 0.8:
        interpretation["compressibility"] = "Low compressibility - sequences are relatively diverse"
    elif zlib_mean > 0.6:
        interpretation["compressibility"] = "Moderate compressibility - some repetitive patterns"
    elif zlib_mean > 0.4:
        interpretation["compressibility"] = "High compressibility - significant repetitive patterns"
    else:
        interpretation["compressibility"] = "Very high compressibility - highly repetitive sequences"
    
    # Comparison to random
    if improvement_over_random > 0.5:
        interpretation["vs_random"] = "Much more compressible than random - strong structural patterns"
    elif improvement_over_random > 0.2:
        interpretation["vs_random"] = "More compressible than random - clear structural patterns"
    elif improvement_over_random > 0.05:
        interpretation["vs_random"] = "Slightly more compressible than random - weak structural patterns"
    else:
        interpretation["vs_random"] = "Similar to random sequences - limited structural patterns"
    
    # Structure level
    lz_mean = statistics["lz_complexity"]["mean"]
    if lz_mean < 0.3:
        interpretation["structure_level"] = "Highly structured - very low algorithmic complexity"
    elif lz_mean < 0.6:
        interpretation["structure_level"] = "Moderately structured - constrained complexity"
    elif lz_mean < 0.8:
        interpretation["structure_level"] = "Somewhat structured - moderate complexity"
    else:
        interpretation["structure_level"] = "Weakly structured - approaching random complexity"
    
    return interpretation