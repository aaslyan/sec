"""
Compiler fingerprinting for Binary DNA analysis.

Identifies compiler types, versions, and optimization levels from 
instruction sequence patterns and statistical signatures.
"""

import numpy as np
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json
from analysis.ngrams import extract_ngrams
from analysis.frequency import compute_zipf_analysis

logger = logging.getLogger(__name__)

class CompilerFingerprinter:
    """Classifier for identifying compiler signatures in binaries."""
    
    def __init__(self):
        """Initialize fingerprinter with signature patterns."""
        self.models = {}
        self.feature_extractors = {}
        self.signature_patterns = self._load_signature_patterns()
        
    def _load_signature_patterns(self) -> Dict:
        """Load known compiler signature patterns."""
        # Known instruction patterns for different compilers
        return {
            'gcc_signatures': {
                # GCC tends to use specific instruction sequences
                'function_prologue': ['pushq %rbp', 'movq %rsp, %rbp'],
                'function_epilogue': ['leaveq', 'retq'],
                'optimization_patterns': {
                    'O0': ['movl', 'addl', 'subl'],  # Unoptimized, literal translations
                    'O2': ['leaq', 'testq', 'jmp'],   # Optimized addressing, control flow
                    'O3': ['vmov', 'vpadd', 'unroll']  # Vectorization, loop unrolling
                }
            },
            'clang_signatures': {
                'function_prologue': ['pushq %rbp', 'movq %rsp, %rbp'],
                'function_epilogue': ['popq %rbp', 'retq'],
                'optimization_patterns': {
                    'O0': ['movq', 'callq'],
                    'O2': ['leaq', 'testq', 'cmov'],
                    'O3': ['vzeroupper', 'vpxor', 'vmul']
                }
            },
            'rustc_signatures': {
                # Rust compiler specific patterns
                'panic_handling': ['ud2', 'callq panic'],
                'bounds_checking': ['cmpq', 'jae', 'ud2'],
                'optimization_patterns': {
                    'debug': ['callq', 'panic', 'bounds_check'],
                    'release': ['assume', 'unchecked', 'inline']
                }
            }
        }
    
    def extract_compiler_features(self, binary: Binary) -> Dict[str, float]:
        """
        Extract features that indicate compiler type and optimization.
        
        Args:
            binary: Binary object to analyze
            
        Returns:
            Dictionary of compiler fingerprint features
        """
        sequence = binary.full_opcode_sequence
        if not sequence:
            return {}
        
        features = {}
        
        # 1. Instruction frequency patterns
        instruction_counts = Counter(sequence)
        total_instructions = len(sequence)
        
        # Key instruction ratios that differ between compilers
        key_instructions = [
            'movq', 'movl', 'pushq', 'popq', 'leaq', 'addq', 'subq',
            'testq', 'cmpq', 'jmp', 'je', 'jne', 'callq', 'retq',
            'vmov', 'vadd', 'vmul', 'vxor', 'ud2'
        ]
        
        for instr in key_instructions:
            features[f'ratio_{instr}'] = instruction_counts.get(instr, 0) / total_instructions
        
        # 2. N-gram patterns (compiler-specific instruction sequences)
        bigrams = extract_ngrams(sequence, 2)
        trigrams = extract_ngrams(sequence, 3)
        
        # Count specific patterns
        prologue_patterns = [
            ('pushq', 'movq'),  # GCC/Clang standard prologue
            ('endbr64', 'pushq'),  # Intel CET enabled
            ('pushq', 'leaq'),   # Alternative prologue
        ]
        
        for pattern in prologue_patterns:
            pattern_str = ' '.join(pattern)
            features[f'prologue_{pattern_str}'] = bigrams.count(pattern) / max(len(bigrams), 1)
        
        # 3. Control flow patterns
        features['jump_ratio'] = sum(instruction_counts.get(j, 0) for j in ['jmp', 'je', 'jne', 'jl', 'jg']) / total_instructions
        features['call_ratio'] = instruction_counts.get('callq', 0) / total_instructions
        features['ret_ratio'] = instruction_counts.get('retq', 0) / total_instructions
        
        # 4. Optimization indicators
        # Vectorization (indicates O2/O3)
        vector_instrs = ['vmov', 'vadd', 'vmul', 'vpadd', 'vpxor', 'vzeroupper']
        features['vectorization_ratio'] = sum(instruction_counts.get(v, 0) for v in vector_instrs) / total_instructions
        
        # Complex addressing (indicates optimization)
        features['lea_ratio'] = instruction_counts.get('leaq', 0) / total_instructions
        
        # Conditional moves (compiler optimization)
        cmov_instrs = ['cmovl', 'cmovg', 'cmove', 'cmovne']
        features['cmov_ratio'] = sum(instruction_counts.get(c, 0) for c in cmov_instrs) / total_instructions
        
        # 5. Stack operations pattern
        features['push_pop_ratio'] = (instruction_counts.get('pushq', 0) + instruction_counts.get('popq', 0)) / total_instructions
        
        # 6. Function structure indicators
        if binary.functions:
            func_lengths = [len(func.opcode_sequence) for func in binary.functions if func.opcode_sequence]
            if func_lengths:
                features['avg_function_length'] = np.mean(func_lengths)
                features['function_length_std'] = np.std(func_lengths)
                features['max_function_length'] = max(func_lengths)
            else:
                features['avg_function_length'] = 0
                features['function_length_std'] = 0
                features['max_function_length'] = 0
        
        # 7. Entropy and complexity measures
        # Use Zipf analysis to measure instruction distribution regularity
        from analysis.frequency import analyze_opcode_frequencies
        freq_data = analyze_opcode_frequencies([binary])
        if freq_data and 'zipf_analysis' in freq_data:
            global_zipf = freq_data['zipf_analysis'].get('global_zipf', {})
            features['zipf_alpha'] = global_zipf.get('alpha', 0)
            features['zipf_r_squared'] = global_zipf.get('r_squared', 0)
        
        # 8. Error handling patterns (Rust-specific)
        error_patterns = ['ud2', 'panic', 'abort']
        features['error_handling_ratio'] = sum(instruction_counts.get(e, 0) for e in error_patterns) / total_instructions
        
        return features
    
    def identify_compiler_heuristic(self, binary: Binary) -> Dict[str, any]:
        """
        Use heuristic rules to identify compiler characteristics.
        
        Args:
            binary: Binary to analyze
            
        Returns:
            Dictionary with compiler identification results
        """
        features = self.extract_compiler_features(binary)
        
        if not features:
            return {'error': 'No features extracted'}
        
        result = {
            'binary_name': binary.name,
            'confidence_scores': {},
            'optimization_indicators': {},
            'compiler_signatures': {}
        }
        
        # Heuristic rules for compiler identification
        
        # 1. GCC vs Clang distinction
        gcc_score = 0.0
        clang_score = 0.0
        rustc_score = 0.0
        
        # GCC tends to have higher lea usage in optimized code
        if features.get('lea_ratio', 0) > 0.05:
            gcc_score += 0.3
        
        # Clang tends to use more vectorization instructions
        if features.get('vectorization_ratio', 0) > 0.02:
            clang_score += 0.3
        
        # Rust specific patterns
        if features.get('error_handling_ratio', 0) > 0.001:
            rustc_score += 0.4
        
        # Pattern-based scoring
        for pattern_key, pattern_value in features.items():
            if 'prologue' in pattern_key and pattern_value > 0:
                gcc_score += 0.2
                clang_score += 0.2
            elif 'vectorization' in pattern_key and pattern_value > 0.01:
                clang_score += 0.3
            elif 'cmov' in pattern_key and pattern_value > 0.005:
                gcc_score += 0.2
                clang_score += 0.3  # Clang uses conditional moves more aggressively
        
        result['confidence_scores'] = {
            'gcc': min(gcc_score, 1.0),
            'clang': min(clang_score, 1.0),
            'rustc': min(rustc_score, 1.0)
        }
        
        # 2. Optimization level detection
        opt_indicators = {
            'O0_indicators': {
                'high_mov_ratio': features.get('ratio_movq', 0) > 0.2,
                'low_lea_ratio': features.get('lea_ratio', 0) < 0.01,
                'high_call_ratio': features.get('call_ratio', 0) > 0.05
            },
            'O2_indicators': {
                'moderate_lea_ratio': 0.01 <= features.get('lea_ratio', 0) <= 0.1,
                'cmov_usage': features.get('cmov_ratio', 0) > 0.002,
                'balanced_jumps': 0.02 <= features.get('jump_ratio', 0) <= 0.1
            },
            'O3_indicators': {
                'high_vectorization': features.get('vectorization_ratio', 0) > 0.01,
                'high_lea_ratio': features.get('lea_ratio', 0) > 0.05,
                'complex_patterns': features.get('zipf_r_squared', 0) > 0.8
            }
        }
        
        # Score optimization levels
        opt_scores = {}
        for level, indicators in opt_indicators.items():
            score = sum(1 for indicator in indicators.values() if indicator) / len(indicators)
            opt_level = level.replace('_indicators', '')
            opt_scores[opt_level] = score
        
        result['optimization_indicators'] = opt_indicators
        result['optimization_scores'] = opt_scores
        result['predicted_optimization'] = max(opt_scores.items(), key=lambda x: x[1])[0]
        
        # 3. Overall prediction
        best_compiler = max(result['confidence_scores'].items(), key=lambda x: x[1])
        result['predicted_compiler'] = best_compiler[0]
        result['compiler_confidence'] = best_compiler[1]
        
        # 4. Architecture and build info from metadata
        if hasattr(binary, 'metadata') and binary.metadata:
            result['metadata_info'] = {
                'source_language': binary.metadata.get('language', 'unknown'),
                'source_repo': binary.metadata.get('source_repo', 'unknown')
            }
        
        return result
    
    def analyze_corpus_compilers(self, binaries: List[Binary]) -> Dict:
        """
        Analyze compiler patterns across an entire corpus.
        
        Args:
            binaries: List of binaries to analyze
            
        Returns:
            Corpus-wide compiler analysis
        """
        logger.info(f"Analyzing compiler patterns for {len(binaries)} binaries")
        
        results = {
            'corpus_summary': {
                'total_binaries': len(binaries),
                'compiler_distribution': defaultdict(int),
                'optimization_distribution': defaultdict(int),
                'language_distribution': defaultdict(int)
            },
            'per_binary_analysis': {},
            'aggregate_features': {},
            'pattern_analysis': {}
        }
        
        all_features = []
        compiler_predictions = []
        
        # Analyze each binary
        for binary in binaries:
            try:
                analysis = self.identify_compiler_heuristic(binary)
                
                if 'error' not in analysis:
                    results['per_binary_analysis'][binary.name] = analysis
                    
                    # Collect for aggregate analysis
                    predicted_compiler = analysis.get('predicted_compiler', 'unknown')
                    predicted_opt = analysis.get('predicted_optimization', 'unknown')
                    
                    results['corpus_summary']['compiler_distribution'][predicted_compiler] += 1
                    results['corpus_summary']['optimization_distribution'][predicted_opt] += 1
                    
                    # Language from metadata
                    if hasattr(binary, 'metadata') and binary.metadata:
                        language = binary.metadata.get('language', 'unknown')
                        results['corpus_summary']['language_distribution'][language] += 1
                    
                    # Collect features for aggregate analysis
                    features = self.extract_compiler_features(binary)
                    if features:
                        all_features.append(features)
                        compiler_predictions.append(predicted_compiler)
                
            except Exception as e:
                logger.error(f"Error analyzing {binary.name}: {e}")
                continue
        
        # Aggregate feature analysis
        if all_features:
            # Calculate mean feature values by compiler
            compiler_feature_means = defaultdict(lambda: defaultdict(list))
            
            for features, compiler in zip(all_features, compiler_predictions):
                for feature_name, feature_value in features.items():
                    compiler_feature_means[compiler][feature_name].append(feature_value)
            
            # Compute means and identify discriminative features
            discriminative_features = {}
            for compiler, feature_dict in compiler_feature_means.items():
                discriminative_features[compiler] = {}
                for feature_name, values in feature_dict.items():
                    discriminative_features[compiler][feature_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
            
            results['aggregate_features'] = discriminative_features
        
        # Pattern analysis
        results['pattern_analysis'] = self._analyze_corpus_patterns(binaries)
        
        # Summary statistics
        total = results['corpus_summary']['total_binaries']
        results['corpus_summary']['compiler_distribution'] = dict(results['corpus_summary']['compiler_distribution'])
        results['corpus_summary']['optimization_distribution'] = dict(results['corpus_summary']['optimization_distribution'])
        results['corpus_summary']['language_distribution'] = dict(results['corpus_summary']['language_distribution'])
        
        logger.info(f"Compiler fingerprinting completed for {total} binaries")
        
        return results
    
    def _analyze_corpus_patterns(self, binaries: List[Binary]) -> Dict:
        """Analyze patterns across the corpus."""
        patterns = {
            'instruction_diversity': {},
            'common_sequences': {},
            'optimization_patterns': {}
        }
        
        # Collect all instruction sequences
        all_instructions = []
        all_bigrams = []
        
        for binary in binaries:
            if binary.full_opcode_sequence:
                all_instructions.extend(binary.full_opcode_sequence)
                bigrams = extract_ngrams(binary.full_opcode_sequence, 2)
                all_bigrams.extend([' '.join(bg) for bg in bigrams])
        
        if all_instructions:
            # Most common instructions
            instr_counter = Counter(all_instructions)
            patterns['instruction_diversity'] = {
                'total_instructions': len(all_instructions),
                'unique_instructions': len(instr_counter),
                'top_20_instructions': instr_counter.most_common(20)
            }
            
            # Most common instruction pairs
            bigram_counter = Counter(all_bigrams)
            patterns['common_sequences'] = {
                'total_bigrams': len(all_bigrams),
                'unique_bigrams': len(bigram_counter),
                'top_20_bigrams': bigram_counter.most_common(20)
            }
        
        return patterns

def run_compiler_fingerprinting(binaries: List[Binary], output_dir: Path) -> Dict:
    """Run complete compiler fingerprinting analysis."""
    logger.info("Running compiler fingerprinting analysis...")
    
    fingerprinter = CompilerFingerprinter()
    results = fingerprinter.analyze_corpus_compilers(binaries)
    
    # Save results
    save_json(results, output_dir / "compiler_fingerprinting.json")
    
    logger.info("Compiler fingerprinting analysis completed")
    return results