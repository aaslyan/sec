#!/usr/bin/env python3
"""
Test script for compiler fingerprinting functionality.
"""

import logging
from pathlib import Path
from utils.helpers import load_pickle
from analysis.compiler_fingerprinting_simple import SimpleCompilerFingerprinter

def test_compiler_fingerprinting():
    """Test compiler fingerprinting with synthetic data."""
    print("Testing compiler fingerprinting...")
    
    # Load synthetic corpus
    corpus_file = Path("./test_clustering_synthetic/corpus.pkl")
    if not corpus_file.exists():
        print("Error: Synthetic corpus not found. Run clustering test first.")
        return False
    
    try:
        binaries = load_pickle(corpus_file)
        print(f"Loaded {len(binaries)} synthetic binaries")
        
        # Create fingerprinter
        fingerprinter = SimpleCompilerFingerprinter()
        
        # Test feature extraction for each binary
        for binary in binaries:
            print(f"\nAnalyzing {binary.name}:")
            print(f"  Instruction sequence: {binary.full_opcode_sequence}")
            
            # Extract features
            features = fingerprinter.extract_compiler_features(binary)
            print(f"  Extracted {len(features)} features:")
            
            for feature_name, feature_value in features.items():
                if feature_value > 0:  # Only show non-zero features
                    print(f"    {feature_name}: {feature_value:.4f}")
            
            # Run heuristic identification
            analysis = fingerprinter.identify_compiler_heuristic(binary)
            if 'error' not in analysis:
                print(f"  Predicted compiler: {analysis['predicted_compiler']} (confidence: {analysis['compiler_confidence']:.3f})")
                print(f"  Predicted optimization: {analysis['predicted_optimization']}")
        
        return True
        
    except Exception as e:
        print(f"Compiler fingerprinting test failed: {e}")
        return False

def main():
    """Run compiler fingerprinting test."""
    logging.basicConfig(level=logging.INFO)
    
    print("=== Compiler Fingerprinting Test ===\n")
    
    success = test_compiler_fingerprinting()
    print(f"\nCompiler fingerprinting test: {'✓ PASS' if success else '✗ FAIL'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())