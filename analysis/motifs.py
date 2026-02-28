"""
Motif discovery and positional pattern analysis module.

Finds recurring instruction patterns across functions and analyzes
positional distributions relative to function boundaries.
"""

import numpy as np
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, Function, save_json
from analysis.ngrams import extract_ngrams

logger = logging.getLogger(__name__)

def find_exact_motifs(binaries: List[Binary], k: int, min_frequency: int = 3, 
                     min_function_coverage: float = 0.1) -> List[Dict]:
    """
    Find exact k-mer motifs that appear frequently across functions.
    
    Args:
        binaries: List of binary objects
        k: Length of motifs to find
        min_frequency: Minimum number of occurrences
        min_function_coverage: Minimum fraction of functions containing the motif
    """
    logger.info(f"Finding exact motifs of length {k}")
    
    # Collect all functions
    all_functions = []
    for binary in binaries:
        all_functions.extend(binary.functions)
    
    if not all_functions:
        return []
    
    # Count k-mer occurrences across functions
    motif_counts = defaultdict(int)
    motif_functions = defaultdict(set)
    
    for func_idx, function in enumerate(all_functions):
        if len(function.instructions) < k:
            continue
            
        sequence = function.opcode_sequence
        kmers = extract_ngrams(sequence, k)
        
        # Track unique k-mers per function to avoid double-counting
        unique_kmers_in_func = set(kmers)
        
        for kmer in unique_kmers_in_func:
            motif_counts[kmer] += 1
            motif_functions[kmer].add(func_idx)
    
    # Second pass: count total raw occurrences (not just per-function)
    motif_total_occurrences: dict = defaultdict(int)
    for function in all_functions:
        if len(function.instructions) < k:
            continue
        for kmer in extract_ngrams(function.opcode_sequence, k):
            motif_total_occurrences[kmer] += 1

    # Filter motifs by frequency and coverage
    total_functions = len(all_functions)
    min_func_count = max(1, int(min_function_coverage * total_functions))
    
    filtered_motifs = []
    for motif, count in motif_counts.items():
        func_count = len(motif_functions[motif])
        
        if count >= min_frequency and func_count >= min_func_count:
            total_occurrences = motif_total_occurrences[motif]
            motif_data = {
                "motif": " ".join(motif),
                "frequency": count,
                "total_occurrences": total_occurrences,
                "function_count": func_count,
                "function_coverage": func_count / total_functions,
                "annotation": annotate_motif(motif)
            }
            filtered_motifs.append(motif_data)
    
    # Sort by frequency * coverage score
    filtered_motifs.sort(key=lambda x: x["frequency"] * x["function_coverage"], reverse=True)
    
    logger.info(f"Found {len(filtered_motifs)} motifs of length {k}")
    return filtered_motifs

def annotate_motif(motif: Tuple[str, ...]) -> str:
    """Provide semantic annotation for common mnemonic-only patterns."""
    m = list(motif)

    # Function prologue: push then mov at start
    if len(m) >= 2 and m[0] == "push" and m[1] == "mov":
        return "Function prologue — stack frame setup"
    if m[0] == "push" and len(m) >= 1:
        return "Function prologue — save register"

    # Function epilogue
    if "leave" in m and "ret" in m:
        return "Function epilogue — leave/ret"
    if "pop" in m and "ret" in m:
        return "Function epilogue — pop/ret"
    if m[-1] == "ret":
        return "Function epilogue — return"

    # Loop / branch
    branch_ops = {"je", "jne", "jl", "jle", "jg", "jge", "jb", "jbe", "ja", "jae",
                  "js", "jns", "jo", "jno", "jp", "jnp", "jmp"}
    if "cmp" in m and any(op in branch_ops for op in m):
        return "Loop/branch — compare and jump"
    if "test" in m and any(op in branch_ops for op in m):
        return "Loop/branch — test and jump"

    # Register zeroing (xor reg, reg encodes as a single xor mnemonic)
    if m.count("xor") >= 1 and len(m) <= 3:
        return "Register zeroing — xor idiom"

    # Call sequences
    if "call" in m:
        return "Function call — subroutine invocation"

    # Memory/data movement
    if m.count("mov") >= 3:
        return "Data movement — bulk register/memory transfers"
    if "lea" in m and "mov" in m:
        return "Memory access — address calculation and load"

    # Arithmetic
    if any(op in m for op in ("add", "sub", "imul", "idiv")):
        return "Arithmetic — integer operations"

    # Stack
    if m.count("push") >= 2:
        return "Stack — multiple pushes"
    if m.count("pop") >= 2:
        return "Stack — multiple pops"

    return "Generic pattern"

def analyze_positional_patterns(binaries: List[Binary], window_size: int = 20) -> Dict:
    """
    Analyze instruction distributions relative to function boundaries.
    
    Args:
        binaries: List of binary objects
        window_size: Number of positions to analyze from start/end of functions
    """
    logger.info(f"Analyzing positional patterns with window size {window_size}")
    
    # Collect position-opcode data
    start_positions = defaultdict(list)  # position -> list of opcodes
    end_positions = defaultdict(list)    # position -> list of opcodes
    
    valid_functions = 0
    
    for binary in binaries:
        for function in binary.functions:
            if len(function.instructions) < window_size * 2:
                continue  # Skip functions too small for analysis
                
            valid_functions += 1
            sequence = function.opcode_sequence
            
            # Analyze start positions
            for pos in range(min(window_size, len(sequence))):
                start_positions[pos].append(sequence[pos])
            
            # Analyze end positions (negative indexing from end)
            for pos in range(window_size):
                if pos < len(sequence):
                    end_positions[pos].append(sequence[-(pos + 1)])
    
    if valid_functions == 0:
        logger.warning("No functions large enough for positional analysis")
        return {}
    
    # Compute frequency distributions for each position
    start_distributions = {}
    end_distributions = {}
    
    for pos in range(window_size):
        if pos in start_positions:
            start_dist = Counter(start_positions[pos])
            total = sum(start_dist.values())
            start_distributions[pos] = {
                opcode: count / total 
                for opcode, count in start_dist.most_common(10)
            }
        
        if pos in end_positions:
            end_dist = Counter(end_positions[pos])
            total = sum(end_dist.values())
            end_distributions[pos] = {
                opcode: count / total 
                for opcode, count in end_dist.most_common(10)
            }
    
    # Compute positional entropy
    start_entropies = compute_positional_entropies(start_positions, window_size)
    end_entropies = compute_positional_entropies(end_positions, window_size)
    
    # Generate consensus sequences
    start_consensus = generate_consensus_sequence(start_distributions, window_size)
    end_consensus = generate_consensus_sequence(end_distributions, window_size)
    
    results = {
        "analysis_params": {
            "window_size": window_size,
            "valid_functions": valid_functions
        },
        "start_patterns": {
            "distributions": start_distributions,
            "entropies": start_entropies,
            "consensus": start_consensus
        },
        "end_patterns": {
            "distributions": end_distributions,
            "entropies": end_entropies,
            "consensus": end_consensus
        },
        "insights": generate_positional_insights(start_entropies, end_entropies, 
                                                start_consensus, end_consensus)
    }
    
    logger.info(f"Completed positional analysis on {valid_functions} functions")
    return results

def compute_positional_entropies(position_data: Dict[int, List[str]], 
                                window_size: int) -> List[float]:
    """Compute Shannon entropy for each position."""
    entropies = []
    
    for pos in range(window_size):
        if pos in position_data and position_data[pos]:
            opcodes = position_data[pos]
            counter = Counter(opcodes)
            total = len(opcodes)
            
            entropy = 0.0
            for count in counter.values():
                p = count / total
                entropy -= p * np.log2(p)
            
            entropies.append(entropy)
        else:
            entropies.append(0.0)
    
    return entropies

def generate_consensus_sequence(distributions: Dict[int, Dict[str, float]], 
                               window_size: int) -> List[Dict]:
    """Generate consensus sequence showing most likely opcodes at each position."""
    consensus = []
    
    for pos in range(window_size):
        if pos in distributions and distributions[pos]:
            # Get most frequent opcode at this position
            top_opcode = max(distributions[pos].items(), key=lambda x: x[1])
            consensus.append({
                "position": pos,
                "opcode": top_opcode[0],
                "probability": top_opcode[1],
                "alternatives": list(distributions[pos].items())[:3]  # Top 3
            })
        else:
            consensus.append({
                "position": pos,
                "opcode": "N/A",
                "probability": 0.0,
                "alternatives": []
            })
    
    return consensus

def generate_positional_insights(start_entropies: List[float], end_entropies: List[float],
                                start_consensus: List[Dict], end_consensus: List[Dict]) -> Dict:
    """Generate insights from positional analysis."""
    insights = {}
    
    # Entropy analysis
    if start_entropies:
        avg_start_entropy = np.mean(start_entropies[:5])  # First 5 positions
        insights["start_predictability"] = (
            "High" if avg_start_entropy < 1.0 else
            "Medium" if avg_start_entropy < 2.0 else "Low"
        )
    
    if end_entropies:
        avg_end_entropy = np.mean(end_entropies[:5])  # Last 5 positions
        insights["end_predictability"] = (
            "High" if avg_end_entropy < 1.0 else
            "Medium" if avg_end_entropy < 2.0 else "Low"
        )
    
    # Consensus analysis
    start_strong_positions = sum(1 for c in start_consensus[:10] if c["probability"] > 0.5)
    end_strong_positions = sum(1 for c in end_consensus[:10] if c["probability"] > 0.5)
    
    insights["function_structure"] = {
        "start_conserved_positions": start_strong_positions,
        "end_conserved_positions": end_strong_positions,
        "interpretation": (
            "Highly structured" if start_strong_positions + end_strong_positions > 8 else
            "Moderately structured" if start_strong_positions + end_strong_positions > 4 else
            "Loosely structured"
        )
    }
    
    return insights

def run_motif_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """Run complete motif discovery and positional pattern analysis."""
    logger.info("Running motif discovery and positional analysis...")

    # Scale thresholds with corpus size so they work for both small and large corpora.
    # min_frequency  : appear in at least 0.5% of functions, floor 5
    # min_function_coverage : appear in at least 1% of functions
    total_functions = sum(b.function_count for b in binaries)
    min_freq = max(5, int(0.005 * total_functions))
    min_cov  = 0.01
    top_per_k = 100          # cap saved motifs per k to keep JSON manageable
    logger.info(f"Motif thresholds: min_frequency={min_freq}, min_coverage={min_cov:.2f}, "
                f"top_per_k={top_per_k}  (total_functions={total_functions})")

    # Find exact motifs for different lengths
    motif_results = {}
    for k in range(4, 13):
        motifs = find_exact_motifs(binaries, k, min_frequency=min_freq,
                                   min_function_coverage=min_cov)
        top_motifs = motifs[:top_per_k]
        motif_results[f"{k}mer"] = top_motifs

        # Save individual motif files (full list)
        save_json(motifs[:top_per_k], output_dir / f"motifs_k{k}.json")
    
    # Positional pattern analysis
    positional_results = analyze_positional_patterns(binaries, window_size=20)
    
    # Compile comprehensive results
    results = {
        "motif_discovery": motif_results,
        "positional_patterns": positional_results,
        "summary": {
            "total_motifs_saved": sum(len(m) for m in motif_results.values()),
            "thresholds": {"min_frequency": min_freq, "min_coverage": min_cov,
                           "top_per_k": top_per_k},
            "most_common_motifs": get_top_motifs_summary(motif_results),
            "function_structure_summary": positional_results.get("insights", {})
        }
    }
    
    # Save comprehensive results
    save_json(results, output_dir / "motif_analysis.json")
    
    logger.info("Motif discovery and positional analysis completed")
    return results

def get_top_motifs_summary(motif_results: Dict) -> List[Dict]:
    """Get summary of top motifs across all lengths."""
    all_motifs = []
    
    for k, motifs in motif_results.items():
        for motif in motifs[:5]:  # Top 5 from each length
            all_motifs.append({
                "length": int(k.replace("mer", "")),
                "motif": motif["motif"],
                "score": motif["frequency"] * motif["function_coverage"],
                "annotation": motif["annotation"]
            })
    
    # Sort by score and return top 10
    all_motifs.sort(key=lambda x: x["score"], reverse=True)
    return all_motifs[:10]