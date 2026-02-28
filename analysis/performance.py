"""
Performance optimization utilities for large-scale analysis.

Provides parallel processing, memory-efficient algorithms, and streaming
computation for handling large corpora of binaries.
"""

import numpy as np
import logging
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Iterator, Tuple, Any, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary
from analysis.ngrams import extract_ngrams
import random
import time
from functools import wraps

logger = logging.getLogger(__name__)

def timing_decorator(func):
    """Decorator to measure and log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class MemoryEfficientAnalyzer:
    """Memory-efficient analyzer for large binary sequences."""
    
    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.max_sequence_length = max_memory_mb * 1000  # Rough estimation
    
    def sample_sequence(self, sequence: List[str], max_size: Optional[int] = None) -> List[str]:
        """Sample sequence if it's too large for memory."""
        if max_size is None:
            max_size = self.max_sequence_length
        
        if len(sequence) <= max_size:
            return sequence
        
        logger.info(f"Sampling {max_size} elements from sequence of length {len(sequence)}")
        
        # Use systematic sampling to maintain temporal structure
        step = len(sequence) // max_size
        if step <= 1:
            return random.sample(sequence, max_size)
        
        sampled = []
        for i in range(0, len(sequence), step):
            if len(sampled) >= max_size:
                break
            sampled.append(sequence[i])
        
        return sampled
    
    def streaming_ngram_count(self, sequence: List[str], n: int, 
                            batch_size: int = 10000) -> Dict[Tuple[str, ...], int]:
        """Compute n-gram counts in batches to manage memory."""
        ngram_counts = {}
        
        for i in range(0, len(sequence) - n + 1, batch_size):
            batch_end = min(i + batch_size + n - 1, len(sequence))
            batch = sequence[i:batch_end]
            
            batch_ngrams = extract_ngrams(batch, n)
            
            for ngram in batch_ngrams:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            
            if i % (batch_size * 10) == 0:
                logger.debug(f"Processed {i} positions for {n}-grams")
        
        return ngram_counts
    
    def chunk_sequence(self, sequence: List[str], chunk_size: int) -> Iterator[List[str]]:
        """Yield chunks of sequence for batch processing."""
        for i in range(0, len(sequence), chunk_size):
            yield sequence[i:i + chunk_size]

def analyze_binary_parallel(args: Tuple[Binary, Dict]) -> Dict:
    """Analyze a single binary - designed for parallel execution."""
    binary, config = args
    
    try:
        analyzer = MemoryEfficientAnalyzer(config.get('max_memory_mb', 500))
        
        # Sample sequence if too large
        sequence = binary.full_opcode_sequence
        sampled_sequence = analyzer.sample_sequence(sequence, config.get('max_seq_length'))
        
        results = {
            'binary_name': binary.name,
            'original_length': len(sequence),
            'analyzed_length': len(sampled_sequence),
            'sampled': len(sampled_sequence) < len(sequence)
        }
        
        # Efficient frequency analysis
        if 'frequency' in config.get('analyses', []):
            from collections import Counter
            freq_counter = Counter(sampled_sequence)
            results['frequencies'] = dict(freq_counter.most_common(50))
        
        # Memory-efficient n-gram analysis
        if 'ngrams' in config.get('analyses', []):
            ngram_results = {}
            for n in config.get('ngram_lengths', [2, 3, 4]):
                if len(sampled_sequence) >= n:
                    ngrams = analyzer.streaming_ngram_count(sampled_sequence, n)
                    ngram_results[f'{n}gram_count'] = len(ngrams)
                    # Convert tuple keys to strings for JSON serialization
                    top_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)[:20]
                    ngram_results[f'top_{n}grams'] = {
                        ' '.join(ngram): count for ngram, count in top_ngrams
                    }
            results['ngrams'] = ngram_results
        
        # Fast compression estimate
        if 'compression' in config.get('analyses', []):
            results['compression'] = estimate_compression_fast(sampled_sequence)
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing binary {binary.name}: {e}")
        return {'binary_name': binary.name, 'error': str(e)}

def estimate_compression_fast(sequence: List[str]) -> Dict:
    """Fast compression estimation without actual compression."""
    from collections import Counter
    
    # Shannon entropy as compression lower bound
    counter = Counter(sequence)
    total = len(sequence)
    
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * np.log2(p)
    
    # Theoretical compression ratio
    max_entropy = np.log2(len(counter))
    compression_estimate = entropy / max_entropy if max_entropy > 0 else 1.0
    
    # Simple repetition measure
    unique_ratio = len(counter) / total
    
    return {
        'entropy': entropy,
        'max_entropy': max_entropy,
        'compression_estimate': compression_estimate,
        'unique_token_ratio': unique_ratio,
        'vocabulary_size': len(counter)
    }

@timing_decorator
def analyze_corpus_parallel(binaries: List[Binary], 
                          analyses: List[str] = None,
                          max_workers: Optional[int] = None,
                          max_memory_per_worker: int = 500) -> List[Dict]:
    """
    Analyze corpus of binaries in parallel.
    
    Args:
        binaries: List of binary objects to analyze
        analyses: List of analysis types ['frequency', 'ngrams', 'compression']
        max_workers: Number of parallel workers (default: CPU count)
        max_memory_per_worker: Memory limit per worker in MB
    """
    if analyses is None:
        analyses = ['frequency', 'ngrams', 'compression']
    
    if max_workers is None:
        max_workers = min(cpu_count(), len(binaries))
    
    logger.info(f"Starting parallel analysis of {len(binaries)} binaries "
               f"with {max_workers} workers")
    
    # Configuration for each worker
    config = {
        'analyses': analyses,
        'max_memory_mb': max_memory_per_worker,
        'max_seq_length': max_memory_per_worker * 1000,
        'ngram_lengths': [2, 3, 4]
    }
    
    # Prepare arguments for parallel processing
    args_list = [(binary, config) for binary in binaries]
    
    try:
        with Pool(processes=max_workers) as pool:
            results = pool.map(analyze_binary_parallel, args_list)
        
        # Filter out errors
        successful_results = [r for r in results if 'error' not in r]
        failed_count = len(results) - len(successful_results)
        
        if failed_count > 0:
            logger.warning(f"{failed_count} binaries failed analysis")
        
        logger.info(f"Parallel analysis completed: {len(successful_results)} successful")
        return successful_results
        
    except Exception as e:
        logger.error(f"Parallel analysis failed: {e}")
        # Fallback to sequential processing
        logger.info("Falling back to sequential processing...")
        return [analyze_binary_parallel((binary, config)) for binary in binaries]

def adaptive_sampling_strategy(binary: Binary, target_size: int) -> List[str]:
    """
    Adaptive sampling that preserves important structural information.
    
    Samples more densely from:
    - Function boundaries (start/end of functions)  
    - Transition regions between different instruction types
    - Areas with high local diversity
    """
    sequence = binary.full_opcode_sequence
    
    if len(sequence) <= target_size:
        return sequence
    
    logger.info(f"Adaptive sampling: {len(sequence)} -> {target_size} instructions")
    
    # Calculate importance scores for each position
    importance_scores = np.ones(len(sequence))
    
    # Function boundary importance
    func_boundaries = set()
    current_pos = 0
    for func in binary.functions:
        func_boundaries.add(current_pos)  # Function start
        current_pos += len(func.instructions)
        func_boundaries.add(current_pos - 1)  # Function end
    
    for pos in func_boundaries:
        if 0 <= pos < len(importance_scores):
            # High importance for boundaries and nearby positions
            for offset in range(-5, 6):
                boundary_pos = pos + offset
                if 0 <= boundary_pos < len(importance_scores):
                    importance_scores[boundary_pos] *= 3.0
    
    # Transition importance (instruction type changes)
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            importance_scores[i] *= 1.5
    
    # Local diversity importance
    window_size = 20
    for i in range(len(sequence) - window_size):
        window = sequence[i:i + window_size]
        diversity = len(set(window)) / window_size
        if diversity > 0.7:  # High diversity
            for j in range(i, i + window_size):
                importance_scores[j] *= 1.2
    
    # Sample based on importance scores
    probabilities = importance_scores / np.sum(importance_scores)
    sampled_indices = np.random.choice(
        len(sequence), size=target_size, replace=False, p=probabilities
    )
    sampled_indices.sort()
    
    return [sequence[i] for i in sampled_indices]

class ProgressiveAnalyzer:
    """
    Progressive analyzer that provides intermediate results for long-running analyses.
    """
    
    def __init__(self, callback_interval: int = 1000):
        self.callback_interval = callback_interval
        self.progress_callbacks = []
    
    def add_progress_callback(self, callback):
        """Add a callback function to receive progress updates."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, current: int, total: int, message: str = ""):
        """Notify all callbacks of current progress."""
        for callback in self.progress_callbacks:
            try:
                callback(current, total, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def progressive_ngram_analysis(self, binaries: List[Binary], n: int) -> Dict:
        """N-gram analysis with progress reporting."""
        total_operations = len(binaries)
        ngram_counts = {}
        
        for i, binary in enumerate(binaries):
            if i % self.callback_interval == 0:
                self._notify_progress(i, total_operations, f"Processing {binary.name}")
            
            sequence = binary.full_opcode_sequence
            if len(sequence) >= n:
                binary_ngrams = extract_ngrams(sequence, n)
                for ngram in binary_ngrams:
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        self._notify_progress(total_operations, total_operations, "Complete")
        
        return {
            'ngram_counts': ngram_counts,
            'unique_ngrams': len(ngram_counts),
            'total_ngrams': sum(ngram_counts.values())
        }

def benchmark_analysis_methods(binaries: List[Binary], sample_size: int = 3) -> Dict:
    """Benchmark different analysis approaches for performance comparison."""
    if len(binaries) > sample_size:
        test_binaries = random.sample(binaries, sample_size)
    else:
        test_binaries = binaries
    
    results = {}
    
    # Benchmark sequential vs parallel
    logger.info("Benchmarking sequential analysis...")
    start_time = time.time()
    sequential_results = []
    for binary in test_binaries:
        config = {'analyses': ['frequency'], 'max_memory_mb': 500}
        result = analyze_binary_parallel((binary, config))
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    logger.info("Benchmarking parallel analysis...")
    start_time = time.time()
    parallel_results = analyze_corpus_parallel(
        test_binaries, 
        analyses=['frequency'], 
        max_workers=min(2, len(test_binaries))
    )
    parallel_time = time.time() - start_time
    
    results['performance_comparison'] = {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': sequential_time / parallel_time if parallel_time > 0 else 1.0,
        'test_binaries': len(test_binaries)
    }
    
    # Benchmark memory usage patterns
    memory_stats = []
    for binary in test_binaries:
        stats = {
            'binary': binary.name,
            'instruction_count': binary.instruction_count,
            'memory_estimate_mb': binary.instruction_count * 50 / (1024 * 1024)  # Rough estimate
        }
        memory_stats.append(stats)
    
    results['memory_analysis'] = {
        'per_binary_stats': memory_stats,
        'total_estimated_mb': sum(s['memory_estimate_mb'] for s in memory_stats)
    }
    
    return results

def run_fast_analysis(args) -> int:
    """Run performance-optimized analysis pipeline."""
    try:
        from analysis.pipeline import load_corpus
        from utils.helpers import ensure_output_dir
        
        corpus_dir = Path(args.corpus_dir)
        output_dir = ensure_output_dir(Path(args.output_dir))
        
        logger.info("Starting fast analysis pipeline...")
        
        # Load corpus
        binaries = load_corpus(corpus_dir)
        
        if not binaries:
            logger.error("No binaries loaded from corpus")
            return 1
        
        # Performance benchmark first
        benchmark_results = benchmark_analysis_methods(binaries[:3])  # Sample for benchmark
        logger.info(f"Performance benchmark: {benchmark_results['performance_comparison']['speedup']:.2f}x speedup with parallelization")
        
        # Configure analysis parameters
        max_workers = args.workers
        max_memory = args.max_memory
        sample_size = args.sample_size
        
        # Run parallel analysis
        parallel_results = analyze_corpus_parallel(
            binaries,
            analyses=['frequency', 'ngrams', 'compression'],
            max_workers=max_workers,
            max_memory_per_worker=max_memory
        )
        
        # Aggregate results
        aggregated_results = aggregate_parallel_results(parallel_results)
        
        # Save results
        from utils.helpers import save_json
        save_json(parallel_results, output_dir / "fast_analysis_raw.json")
        save_json(aggregated_results, output_dir / "fast_analysis_summary.json")
        save_json(benchmark_results, output_dir / "performance_benchmark.json")
        
        logger.info(f"Fast analysis completed: {len(parallel_results)} binaries analyzed")
        return 0
        
    except Exception as e:
        logger.error(f"Fast analysis failed: {e}")
        return 1

def aggregate_parallel_results(parallel_results: List[Dict]) -> Dict:
    """Aggregate results from parallel binary analysis."""
    if not parallel_results:
        return {}
    
    # Aggregate frequency data
    global_frequencies = {}
    total_instructions = 0
    
    for result in parallel_results:
        if 'frequencies' in result:
            for opcode, count in result['frequencies'].items():
                global_frequencies[opcode] = global_frequencies.get(opcode, 0) + count
        total_instructions += result.get('analyzed_length', 0)
    
    # Top opcodes globally
    sorted_frequencies = sorted(global_frequencies.items(), key=lambda x: x[1], reverse=True)
    
    # Aggregate n-gram statistics
    ngram_stats = {}
    for n in [2, 3, 4]:
        total_unique = sum(result.get('ngrams', {}).get(f'{n}gram_count', 0) for result in parallel_results)
        ngram_stats[f'{n}gram_unique_total'] = total_unique
    
    # Compression statistics
    compression_estimates = [result.get('compression', {}).get('compression_estimate', 1.0) 
                           for result in parallel_results if 'compression' in result]
    
    aggregated = {
        'corpus_summary': {
            'total_binaries': len(parallel_results),
            'total_instructions_analyzed': total_instructions,
            'unique_opcodes': len(global_frequencies)
        },
        'frequency_analysis': {
            'top_20_opcodes': sorted_frequencies[:20],
            'vocabulary_size': len(global_frequencies)
        },
        'ngram_analysis': ngram_stats,
        'compression_analysis': {
            'mean_compression_estimate': np.mean(compression_estimates) if compression_estimates else 0.0,
            'std_compression_estimate': np.std(compression_estimates) if compression_estimates else 0.0
        },
        'performance_stats': {
            'sampled_binaries': sum(1 for r in parallel_results if r.get('sampled', False)),
            'average_sequence_length': np.mean([r.get('analyzed_length', 0) for r in parallel_results])
        }
    }
    
    return aggregated