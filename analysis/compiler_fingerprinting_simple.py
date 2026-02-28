"""
Simple compiler fingerprinting for Binary DNA analysis.

Identifies compiler types and optimization levels using heuristic rules
without requiring machine learning dependencies.
"""

import numpy as np
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json
from analysis.ngrams import extract_ngrams

logger = logging.getLogger(__name__)

class SimpleCompilerFingerprinter:
    """Heuristic-based classifier for identifying compiler signatures in binaries."""
    
    def __init__(self):
        """Initialize fingerprinter with signature patterns."""
        self.signature_patterns = self._load_signature_patterns()
        
    def _load_signature_patterns(self) -> Dict:
        """Load known compiler signature patterns."""
        return {
            'gcc_signatures': {
                'function_prologue': ['pushq', 'movq'],
                'function_epilogue': ['leaveq', 'retq'],
                'preferred_instructions': ['leaq', 'addq', 'subq'],
                'optimization_indicators': ['cmovl', 'cmovg', 'cmove']
            },
            'clang_signatures': {
                'function_prologue': ['pushq', 'movq'],
                'function_epilogue': ['popq', 'retq'],
                'preferred_instructions': ['testq', 'cmpq'],
                'vectorization': ['vmov', 'vadd', 'vmul', 'vpadd']
            },
            'rustc_signatures': {
                'error_handling': ['ud2', 'panic'],
                'bounds_checking': ['cmpq', 'jae', 'ud2'],
                'preferred_instructions': ['movq', 'callq']
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
        instruction_counts = Counter(sequence)
        total_instructions = len(sequence)
        
        # 1. Basic instruction ratios
        key_instructions = [
            'movq', 'movl', 'pushq', 'popq', 'leaq', 'addq', 'subq',
            'testq', 'cmpq', 'jmp', 'je', 'jne', 'callq', 'retq',
            'vmov', 'vadd', 'vmul', 'vxor', 'ud2'
        ]
        
        for instr in key_instructions:
            features[f'ratio_{instr}'] = instruction_counts.get(instr, 0) / total_instructions
        
        # 2. Pattern-based features
        bigrams = extract_ngrams(sequence, 2)
        bigram_counter = Counter([' '.join(bg) for bg in bigrams])
        
        # Common prologue patterns
        prologue_patterns = ['pushq movq', 'endbr64 pushq', 'pushq leaq']
        for pattern in prologue_patterns:
            features[f'prologue_{pattern.replace(" ", "_")}'] = bigram_counter.get(pattern, 0) / max(len(bigrams), 1)
        
        # 3. Control flow ratios
        jump_instrs = ['jmp', 'je', 'jne', 'jl', 'jg', 'jle', 'jge']
        features['jump_ratio'] = sum(instruction_counts.get(j, 0) for j in jump_instrs) / total_instructions
        features['call_ratio'] = instruction_counts.get('callq', 0) / total_instructions
        features['ret_ratio'] = instruction_counts.get('retq', 0) / total_instructions
        
        # 4. Optimization indicators
        # Vectorization (indicates O2/O3)
        vector_instrs = ['vmov', 'vadd', 'vmul', 'vpadd', 'vpxor', 'vzeroupper']
        features['vectorization_ratio'] = sum(instruction_counts.get(v, 0) for v in vector_instrs) / total_instructions
        
        # Complex addressing (LEA usage)
        features['lea_ratio'] = instruction_counts.get('leaq', 0) / total_instructions
        
        # Conditional moves (compiler optimization)
        cmov_instrs = ['cmovl', 'cmovg', 'cmove', 'cmovne', 'cmovle', 'cmovge']
        features['cmov_ratio'] = sum(instruction_counts.get(c, 0) for c in cmov_instrs) / total_instructions
        
        # 5. Stack operations
        features['push_pop_ratio'] = (instruction_counts.get('pushq', 0) + instruction_counts.get('popq', 0)) / total_instructions
        
        # 6. Error handling (Rust-specific)
        error_patterns = ['ud2', 'panic', 'abort']
        features['error_handling_ratio'] = sum(instruction_counts.get(e, 0) for e in error_patterns) / total_instructions
        
        # 7. Function structure
        if binary.functions:
            func_lengths = [len(func.opcode_sequence) for func in binary.functions if func.opcode_sequence]
            if func_lengths:
                features['avg_function_length'] = np.mean(func_lengths)
                features['function_length_std'] = np.std(func_lengths)
                features['max_function_length'] = max(func_lengths)
                features['num_functions'] = len(func_lengths)
            else:
                features['avg_function_length'] = 0
                features['function_length_std'] = 0
                features['max_function_length'] = 0
                features['num_functions'] = 0
        
        # 8. Instruction diversity
        features['unique_instruction_ratio'] = len(instruction_counts) / total_instructions
        
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
            'features': features,
            'confidence_scores': {},
            'optimization_scores': {},
            'analysis_details': {}
        }
        
        # Heuristic scoring for compiler identification
        gcc_score = 0.0
        clang_score = 0.0
        rustc_score = 0.0
        
        # GCC heuristics
        if features.get('lea_ratio', 0) > 0.05:  # GCC uses LEA frequently
            gcc_score += 0.3
        if features.get('ratio_addq', 0) > 0.1:  # GCC prefers explicit arithmetic
            gcc_score += 0.2
        if features.get('prologue_pushq_movq', 0) > 0:  # Standard prologue
            gcc_score += 0.2
        
        # Clang heuristics  
        if features.get('vectorization_ratio', 0) > 0.02:  # Clang vectorizes more aggressively
            clang_score += 0.4
        if features.get('cmov_ratio', 0) > 0.005:  # Clang uses conditional moves more
            clang_score += 0.3
        if features.get('ratio_testq', 0) > features.get('ratio_cmpq', 0):  # Clang prefers TEST
            clang_score += 0.2
        
        # Rust heuristics
        if features.get('error_handling_ratio', 0) > 0.001:  # Rust has explicit error handling
            rustc_score += 0.5
        if features.get('ratio_ud2', 0) > 0:  # Rust uses ud2 for panic
            rustc_score += 0.3
        if features.get('call_ratio', 0) > 0.1:  # Rust has many function calls
            rustc_score += 0.2
        
        # Normalize scores
        result['confidence_scores'] = {
            'gcc': min(max(gcc_score, 0.0), 1.0),
            'clang': min(max(clang_score, 0.0), 1.0),
            'rustc': min(max(rustc_score, 0.0), 1.0)
        }
        
        # Optimization level detection
        o0_score = 0.0  # Unoptimized
        o2_score = 0.0  # Optimized
        o3_score = 0.0  # Highly optimized
        
        # O0 indicators (debug/unoptimized)
        if features.get('ratio_movq', 0) > 0.3:  # Many simple moves
            o0_score += 0.3
        if features.get('call_ratio', 0) > 0.1:  # Many function calls not inlined
            o0_score += 0.2
        if features.get('lea_ratio', 0) < 0.01:  # No complex addressing
            o0_score += 0.2
        if features.get('cmov_ratio', 0) == 0:  # No conditional moves
            o0_score += 0.3
        
        # O2 indicators (optimized)
        if 0.01 <= features.get('lea_ratio', 0) <= 0.1:  # Moderate LEA usage
            o2_score += 0.3
        if features.get('cmov_ratio', 0) > 0.002:  # Some conditional moves
            o2_score += 0.3
        if 0.02 <= features.get('jump_ratio', 0) <= 0.1:  # Optimized control flow
            o2_score += 0.2
        if features.get('push_pop_ratio', 0) < 0.2:  # Efficient stack usage
            o2_score += 0.2
        
        # O3 indicators (highly optimized)
        if features.get('vectorization_ratio', 0) > 0.01:  # Vectorization
            o3_score += 0.4
        if features.get('lea_ratio', 0) > 0.05:  # Heavy LEA usage
            o3_score += 0.3
        if features.get('unique_instruction_ratio', 0) > 0.5:  # Diverse instructions
            o3_score += 0.2
        if features.get('cmov_ratio', 0) > 0.01:  # Heavy conditional move usage
            o3_score += 0.1
        
        result['optimization_scores'] = {
            'O0': min(max(o0_score, 0.0), 1.0),
            'O2': min(max(o2_score, 0.0), 1.0),
            'O3': min(max(o3_score, 0.0), 1.0)
        }
        
        # Make predictions
        best_compiler = max(result['confidence_scores'].items(), key=lambda x: x[1])
        best_optimization = max(result['optimization_scores'].items(), key=lambda x: x[1])
        
        result['predicted_compiler'] = best_compiler[0]
        result['compiler_confidence'] = best_compiler[1]
        result['predicted_optimization'] = best_optimization[0]
        result['optimization_confidence'] = best_optimization[1]
        
        # Analysis details
        result['analysis_details'] = {
            'total_instructions': len(binary.full_opcode_sequence) if binary.full_opcode_sequence else 0,
            'unique_instructions': len(set(binary.full_opcode_sequence)) if binary.full_opcode_sequence else 0,
            'has_functions': binary.function_count > 0,
            'function_count': binary.function_count
        }
        
        # Add metadata if available
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
                'language_distribution': defaultdict(int),
                'confidence_stats': {
                    'compiler_confidence_mean': 0.0,
                    'optimization_confidence_mean': 0.0
                }
            },
            'per_binary_analysis': {},
            'aggregate_features': {},
            'pattern_analysis': {}
        }
        
        all_compiler_confidences = []
        all_opt_confidences = []
        feature_aggregates = defaultdict(lambda: defaultdict(list))
        
        # Analyze each binary
        successful_analyses = 0
        for binary in binaries:
            try:
                analysis = self.identify_compiler_heuristic(binary)
                
                if 'error' not in analysis:
                    results['per_binary_analysis'][binary.name] = analysis
                    successful_analyses += 1
                    
                    # Collect distribution data
                    predicted_compiler = analysis.get('predicted_compiler', 'unknown')
                    predicted_opt = analysis.get('predicted_optimization', 'unknown')
                    
                    results['corpus_summary']['compiler_distribution'][predicted_compiler] += 1
                    results['corpus_summary']['optimization_distribution'][predicted_opt] += 1
                    
                    # Confidence tracking
                    all_compiler_confidences.append(analysis.get('compiler_confidence', 0.0))
                    all_opt_confidences.append(analysis.get('optimization_confidence', 0.0))
                    
                    # Language from metadata
                    if hasattr(binary, 'metadata') and binary.metadata:
                        language = binary.metadata.get('language', 'unknown')
                        results['corpus_summary']['language_distribution'][language] += 1
                    
                    # Aggregate features by compiler
                    features = analysis.get('features', {})
                    for feature_name, feature_value in features.items():
                        feature_aggregates[predicted_compiler][feature_name].append(feature_value)
                
            except Exception as e:
                logger.error(f"Error analyzing {binary.name}: {e}")
                continue
        
        # Calculate confidence statistics
        if all_compiler_confidences:
            results['corpus_summary']['confidence_stats']['compiler_confidence_mean'] = np.mean(all_compiler_confidences)
            results['corpus_summary']['confidence_stats']['compiler_confidence_std'] = np.std(all_compiler_confidences)
            results['corpus_summary']['confidence_stats']['optimization_confidence_mean'] = np.mean(all_opt_confidences)
            results['corpus_summary']['confidence_stats']['optimization_confidence_std'] = np.std(all_opt_confidences)
        
        # Compute aggregate features
        for compiler, feature_dict in feature_aggregates.items():
            results['aggregate_features'][compiler] = {}
            for feature_name, values in feature_dict.items():
                if values:  # Only if we have data
                    results['aggregate_features'][compiler][feature_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
        
        # Convert defaultdicts to regular dicts
        results['corpus_summary']['compiler_distribution'] = dict(results['corpus_summary']['compiler_distribution'])
        results['corpus_summary']['optimization_distribution'] = dict(results['corpus_summary']['optimization_distribution'])
        results['corpus_summary']['language_distribution'] = dict(results['corpus_summary']['language_distribution'])
        
        # Pattern analysis
        results['pattern_analysis'] = self._analyze_corpus_patterns(binaries)
        
        logger.info(f"Compiler fingerprinting completed: {successful_analyses}/{len(binaries)} binaries analyzed successfully")
        
        return results
    
    def _analyze_corpus_patterns(self, binaries: List[Binary]) -> Dict:
        """Analyze instruction patterns across the corpus."""
        patterns = {
            'instruction_stats': {},
            'common_sequences': {},
            'compiler_signatures': {}
        }
        
        # Collect all instructions and sequences
        all_instructions = []
        all_bigrams = []
        
        for binary in binaries:
            if binary.full_opcode_sequence:
                all_instructions.extend(binary.full_opcode_sequence)
                bigrams = extract_ngrams(binary.full_opcode_sequence, 2)
                all_bigrams.extend([' '.join(bg) for bg in bigrams])
        
        if all_instructions:
            # Instruction statistics
            instr_counter = Counter(all_instructions)
            patterns['instruction_stats'] = {
                'total_instructions': len(all_instructions),
                'unique_instructions': len(instr_counter),
                'diversity_ratio': len(instr_counter) / len(all_instructions),
                'top_10_instructions': instr_counter.most_common(10)
            }
            
            # Common instruction sequences
            if all_bigrams:
                bigram_counter = Counter(all_bigrams)
                patterns['common_sequences'] = {
                    'total_bigrams': len(all_bigrams),
                    'unique_bigrams': len(bigram_counter),
                    'top_10_bigrams': bigram_counter.most_common(10)
                }
        
        return patterns

def run_compiler_fingerprinting(binaries: List[Binary], output_dir: Path) -> Dict:
    """Run complete compiler fingerprinting analysis."""
    logger.info("Running compiler fingerprinting analysis...")
    
    fingerprinter = SimpleCompilerFingerprinter()
    results = fingerprinter.analyze_corpus_compilers(binaries)
    
    # Save results
    save_json(results, output_dir / "compiler_fingerprinting.json")
    
    logger.info("Compiler fingerprinting analysis completed")
    return results