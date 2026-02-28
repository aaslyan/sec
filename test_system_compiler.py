#!/usr/bin/env python3
"""
Test compiler fingerprinting on real system binaries.
"""

import logging
from pathlib import Path
from utils.helpers import load_pickle
from analysis.compiler_fingerprinting_simple import run_compiler_fingerprinting

def main():
    """Test compiler fingerprinting on system binaries."""
    logging.basicConfig(level=logging.INFO)
    
    print("=== System Binary Compiler Fingerprinting ===")
    
    # Load system corpus
    corpus_file = Path("./system_analysis/corpus/corpus.pkl")
    if not corpus_file.exists():
        print("Error: System corpus not found")
        return 1
    
    binaries = load_pickle(corpus_file)
    print(f"Loaded {len(binaries)} system binaries")
    
    # Show basic info
    for binary in binaries[:5]:  # First 5
        print(f"  {binary.name}: {binary.instruction_count:,} instructions, {binary.function_count} functions")
    
    # Run compiler fingerprinting
    output_dir = Path("./system_compiler_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nRunning compiler fingerprinting analysis...")
    results = run_compiler_fingerprinting(binaries, output_dir)
    
    # Print detailed results
    print("\n=== Compiler Fingerprinting Results ===")
    
    corpus_summary = results.get('corpus_summary', {})
    print(f"Total binaries analyzed: {corpus_summary.get('total_binaries', 0)}")
    print(f"Compiler distribution: {corpus_summary.get('compiler_distribution', {})}")
    print(f"Optimization distribution: {corpus_summary.get('optimization_distribution', {})}")
    
    confidence_stats = corpus_summary.get('confidence_stats', {})
    print(f"Mean compiler confidence: {confidence_stats.get('compiler_confidence_mean', 0):.3f}")
    print(f"Mean optimization confidence: {confidence_stats.get('optimization_confidence_mean', 0):.3f}")
    
    # Show most interesting binary analyses
    print(f"\n=== Individual Binary Analysis ===")
    per_binary = results.get('per_binary_analysis', {})
    for binary_name, analysis in list(per_binary.items())[:5]:  # First 5
        print(f"\n{binary_name}:")
        print(f"  Compiler: {analysis.get('predicted_compiler', 'unknown')} (confidence: {analysis.get('compiler_confidence', 0):.3f})")
        print(f"  Optimization: {analysis.get('predicted_optimization', 'unknown')} (confidence: {analysis.get('optimization_confidence', 0):.3f})")
        
        details = analysis.get('analysis_details', {})
        print(f"  Instructions: {details.get('total_instructions', 0):,} total, {details.get('unique_instructions', 0)} unique")
        print(f"  Functions: {details.get('function_count', 0)}")
        
        # Show key features
        features = analysis.get('features', {})
        key_features = ['vectorization_ratio', 'lea_ratio', 'cmov_ratio', 'call_ratio', 'jump_ratio']
        feature_str = ', '.join([f"{f.replace('_ratio', '')}: {features.get(f, 0):.3f}" for f in key_features])
        print(f"  Key features: {feature_str}")
    
    # Pattern analysis
    pattern_analysis = results.get('pattern_analysis', {})
    instr_stats = pattern_analysis.get('instruction_stats', {})
    if instr_stats:
        print(f"\n=== Corpus-Wide Patterns ===")
        print(f"Total instructions: {instr_stats.get('total_instructions', 0):,}")
        print(f"Unique instructions: {instr_stats.get('unique_instructions', 0)}")
        print(f"Diversity ratio: {instr_stats.get('diversity_ratio', 0):.4f}")
        
        print(f"Top 10 instructions:")
        for instr, count in instr_stats.get('top_10_instructions', []):
            percentage = (count / instr_stats.get('total_instructions', 1)) * 100
            print(f"  {instr}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nDetailed results saved to: {output_dir}")
    return 0

if __name__ == "__main__":
    exit(main())