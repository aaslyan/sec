"""
Main clustering analysis pipeline that orchestrates all clustering methods.

Integrates NCD, n-gram similarity, hierarchical clustering, and dimensionality
reduction into a comprehensive binary clustering analysis.
"""

import numpy as np
import logging
from typing import List, Dict
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json, ensure_output_dir
from clustering.ncd import run_ncd_analysis
from clustering.similarity import run_ngram_similarity_analysis
from clustering.hierarchical import run_hierarchical_clustering
from clustering.dimreduce import run_dimensionality_reduction

logger = logging.getLogger(__name__)

def run_clustering_analysis(binaries: List[Binary], output_dir: Path) -> Dict:
    """
    Run complete clustering analysis pipeline.
    
    Args:
        binaries: List of Binary objects to cluster
        output_dir: Directory for output files and plots
        
    Returns:
        Dictionary containing all clustering results
    """
    logger.info("Starting comprehensive clustering analysis...")
    
    if len(binaries) < 2:
        logger.error("Need at least 2 binaries for clustering analysis")
        return {}
    
    # Ensure output directories exist
    plots_dir = ensure_output_dir(output_dir / "plots")
    
    # Build category mapping from Binary.category metadata (populated in Phase 1)
    binary_categories = {
        b.name: b.category if b.category else 'unknown'
        for b in binaries
    }

    results = {
        'corpus_info': {
            'num_binaries': len(binaries),
            'binary_names': [b.name for b in binaries],
            'binary_categories': binary_categories,
            'total_instructions': sum(b.instruction_count for b in binaries),
            'total_functions': sum(b.function_count for b in binaries)
        }
    }
    
    # Phase 1: Compute similarity/distance matrices
    logger.info("Phase 1: Computing similarity matrices...")
    
    # NCD Analysis (compression-based)
    try:
        ncd_results = run_ncd_analysis(binaries, output_dir)
        results['ncd_analysis'] = ncd_results
    except Exception as e:
        logger.error(f"NCD analysis failed: {e}")
        results['ncd_analysis'] = {'error': str(e)}
    
    # N-gram similarity analysis
    try:
        ngram_results = run_ngram_similarity_analysis(binaries, output_dir)
        results['ngram_similarity'] = ngram_results
    except Exception as e:
        logger.error(f"N-gram similarity analysis failed: {e}")
        results['ngram_similarity'] = {'error': str(e)}
    
    # Phase 2: Hierarchical clustering
    logger.info("Phase 2: Hierarchical clustering...")
    
    # Collect distance matrices for clustering
    distance_matrices = {}
    
    # Add NCD distance matrices
    if 'ncd_analysis' in results and 'error' not in results['ncd_analysis']:
        for compressor in ['zlib', 'lzma']:
            if compressor in results['ncd_analysis'] and 'error' not in results['ncd_analysis'][compressor]:
                matrix_data = results['ncd_analysis'][compressor]['matrix']
                distance_matrices[f'ncd_{compressor}'] = matrix_data
    
    # Add n-gram distance matrices
    if 'ngram_similarity' in results and 'error' not in results['ngram_similarity']:
        for ngram_key in ['2gram', '3gram', '4gram']:
            if (ngram_key in results['ngram_similarity'] and 
                'error' not in results['ngram_similarity'][ngram_key]):
                matrix_data = results['ngram_similarity'][ngram_key]['distance_matrix']
                distance_matrices[f'ngram_{ngram_key}'] = matrix_data
    
    # Run hierarchical clustering
    if distance_matrices:
        try:
            binary_names = [b.name for b in binaries]
            clustering_results = run_hierarchical_clustering(
                distance_matrices, binary_names, output_dir,
                categories=binary_categories)
            results['hierarchical_clustering'] = clustering_results
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            results['hierarchical_clustering'] = {'error': str(e)}
    else:
        logger.warning("No distance matrices available for hierarchical clustering")
        results['hierarchical_clustering'] = {'error': 'No distance matrices available'}
    
    # Phase 3: Dimensionality reduction and visualization
    logger.info("Phase 3: Dimensionality reduction...")
    
    # Collect similarity matrices (for PCA/UMAP)
    similarity_matrices = {}
    
    # Add n-gram similarity matrices
    if 'ngram_similarity' in results and 'error' not in results['ngram_similarity']:
        for ngram_key in ['2gram', '3gram', '4gram']:
            if (ngram_key in results['ngram_similarity'] and 
                'error' not in results['ngram_similarity'][ngram_key]):
                matrix_data = results['ngram_similarity'][ngram_key]['similarity_matrix']
                similarity_matrices[f'ngram_{ngram_key}'] = matrix_data
    
    # Convert NCD distances to similarities (1 - distance)
    if 'ncd_analysis' in results and 'error' not in results['ncd_analysis']:
        for compressor in ['zlib', 'lzma']:
            if compressor in results['ncd_analysis'] and 'error' not in results['ncd_analysis'][compressor]:
                distance_matrix = np.array(results['ncd_analysis'][compressor]['matrix'])
                similarity_matrix = 1.0 - distance_matrix
                similarity_matrices[f'ncd_{compressor}'] = similarity_matrix.tolist()
    
    # Run dimensionality reduction
    if similarity_matrices:
        try:
            binary_names = [b.name for b in binaries]
            dimreduce_results = run_dimensionality_reduction(
                similarity_matrices, binary_names, plots_dir,
                categories=binary_categories)
            results['dimensionality_reduction'] = dimreduce_results
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {e}")
            results['dimensionality_reduction'] = {'error': str(e)}
    else:
        logger.warning("No similarity matrices available for dimensionality reduction")
        results['dimensionality_reduction'] = {'error': 'No similarity matrices available'}
    
    # Phase 4: Comprehensive analysis
    logger.info("Phase 4: Generating comprehensive analysis...")
    
    try:
        comprehensive_analysis = generate_comprehensive_analysis(results)
        results['comprehensive_analysis'] = comprehensive_analysis
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        results['comprehensive_analysis'] = {'error': str(e)}
    
    # Save complete results
    save_json(results, output_dir / "clustering_analysis.json")
    
    logger.info("Clustering analysis pipeline completed")
    return results

def generate_comprehensive_analysis(results: Dict) -> Dict:
    """Generate comprehensive analysis of all clustering results."""
    analysis = {
        'method_summary': {},
        'consistency_analysis': {},
        'key_findings': [],
        'recommendations': []
    }
    
    # Analyze each method
    if 'ncd_analysis' in results and 'error' not in results['ncd_analysis']:
        ncd_analysis = analyze_ncd_results(results['ncd_analysis'])
        analysis['method_summary']['ncd'] = ncd_analysis
    
    if 'ngram_similarity' in results and 'error' not in results['ngram_similarity']:
        ngram_analysis = analyze_ngram_results(results['ngram_similarity'])
        analysis['method_summary']['ngram_similarity'] = ngram_analysis
    
    if 'hierarchical_clustering' in results and 'error' not in results['hierarchical_clustering']:
        clustering_analysis = analyze_clustering_results(results['hierarchical_clustering'])
        analysis['method_summary']['hierarchical_clustering'] = clustering_analysis
    
    # Cross-method consistency analysis
    analysis['consistency_analysis'] = analyze_method_consistency(results)
    
    # Generate key findings
    analysis['key_findings'] = generate_key_findings(results, analysis)
    
    # Generate recommendations
    analysis['recommendations'] = generate_recommendations(results, analysis)
    
    return analysis

def analyze_ncd_results(ncd_results: Dict) -> Dict:
    """Analyze NCD-specific results."""
    analysis = {'compressors': {}}
    
    for compressor in ['zlib', 'lzma']:
        if compressor in ncd_results and 'error' not in ncd_results[compressor]:
            comp_data = ncd_results[compressor]
            stats = comp_data.get('statistics', {})
            
            analysis['compressors'][compressor] = {
                'mean_distance': stats.get('mean_distance', 0),
                'distance_range': stats.get('max_distance', 0) - stats.get('min_distance', 0),
                'interpretation': interpret_ncd_distances(stats.get('mean_distance', 0))
            }
    
    # Compare compressors
    if 'compressor_comparison' in ncd_results:
        analysis['compressor_agreement'] = ncd_results['compressor_comparison']
    
    return analysis

def analyze_ngram_results(ngram_results: Dict) -> Dict:
    """Analyze n-gram similarity results."""
    analysis = {'ngram_lengths': {}}
    
    for ngram_key in ['2gram', '3gram', '4gram']:
        if ngram_key in ngram_results and 'error' not in ngram_results[ngram_key]:
            ngram_data = ngram_results[ngram_key]
            stats = ngram_data.get('statistics', {})
            
            analysis['ngram_lengths'][ngram_key] = {
                'mean_similarity': stats.get('mean_similarity', 0),
                'similarity_range': stats.get('max_distance', 0),  # Using distance for range
                'feature_count': ngram_data.get('feature_count', 0),
                'interpretation': interpret_ngram_similarity(stats.get('mean_similarity', 0))
            }
    
    # Best n-gram analysis
    if 'best_ngram_analysis' in ngram_results:
        analysis['best_ngram'] = ngram_results['best_ngram_analysis']
    
    return analysis

def analyze_clustering_results(clustering_results: Dict) -> Dict:
    """Analyze hierarchical clustering results."""
    analysis = {'matrices': {}}
    
    # Analyze each distance matrix clustering
    for matrix_name, matrix_result in clustering_results.items():
        if matrix_name != 'summary' and 'error' not in matrix_result:
            if 'best_result' in matrix_result:
                best_result = matrix_result['best_result']
                cluster_analysis = best_result.get('cluster_analysis', {})
                
                analysis['matrices'][matrix_name] = {
                    'num_clusters': cluster_analysis.get('num_clusters', 0),
                    'largest_cluster_size': cluster_analysis.get('largest_cluster_size', 0),
                    'cophenetic_correlation': best_result.get('cophenetic_correlation', 0),
                    'best_linkage_method': matrix_result.get('best_method', 'unknown')
                }
    
    # Overall consistency
    if 'summary' in clustering_results:
        analysis['consistency'] = clustering_results['summary']
    
    return analysis

def analyze_method_consistency(results: Dict) -> Dict:
    """Analyze consistency between different clustering methods."""
    consistency = {
        'distance_matrix_correlations': {},
        'clustering_agreement': {},
        'overall_assessment': ''
    }
    
    # Extract distance matrices for correlation analysis
    distance_matrices = {}
    
    # NCD matrices
    if 'ncd_analysis' in results and 'error' not in results['ncd_analysis']:
        for compressor in ['zlib', 'lzma']:
            if compressor in results['ncd_analysis'] and 'error' not in results['ncd_analysis'][compressor]:
                matrix = results['ncd_analysis'][compressor]['matrix']
                distance_matrices[f'ncd_{compressor}'] = np.array(matrix)
    
    # N-gram distance matrices
    if 'ngram_similarity' in results and 'error' not in results['ngram_similarity']:
        for ngram_key in ['2gram', '3gram', '4gram']:
            if (ngram_key in results['ngram_similarity'] and 
                'error' not in results['ngram_similarity'][ngram_key]):
                matrix = results['ngram_similarity'][ngram_key]['distance_matrix']
                distance_matrices[f'ngram_{ngram_key}'] = np.array(matrix)
    
    # Compute pairwise correlations
    matrix_names = list(distance_matrices.keys())
    for i, name1 in enumerate(matrix_names):
        for name2 in matrix_names[i+1:]:
            try:
                # Get upper triangular parts (exclude diagonal)
                mat1 = distance_matrices[name1]
                mat2 = distance_matrices[name2]
                
                upper_indices = np.triu_indices(mat1.shape[0], k=1)
                corr = np.corrcoef(mat1[upper_indices], mat2[upper_indices])[0, 1]
                
                consistency['distance_matrix_correlations'][f'{name1}_vs_{name2}'] = float(corr)
                
            except Exception as e:
                logger.warning(f"Failed to compute correlation between {name1} and {name2}: {e}")
    
    # Overall assessment
    if consistency['distance_matrix_correlations']:
        mean_correlation = np.mean(list(consistency['distance_matrix_correlations'].values()))
        consistency['overall_assessment'] = interpret_method_consistency(mean_correlation)
    else:
        consistency['overall_assessment'] = "Unable to assess method consistency"
    
    return consistency

def generate_key_findings(results: Dict, analysis: Dict) -> List[str]:
    """Generate key findings from clustering analysis."""
    findings = []
    
    # NCD findings
    if 'ncd' in analysis.get('method_summary', {}):
        ncd_summary = analysis['method_summary']['ncd']
        if 'compressor_agreement' in ncd_summary:
            agreement = ncd_summary['compressor_agreement']
            findings.append(f"Compression-based similarity: {agreement.get('interpretation', 'Unknown')}")
    
    # N-gram findings
    if 'ngram_similarity' in analysis.get('method_summary', {}):
        ngram_summary = analysis['method_summary']['ngram_similarity']
        if 'best_ngram' in ngram_summary:
            best_n = ngram_summary['best_ngram'].get('best_n')
            findings.append(f"Optimal n-gram length for binary similarity: {best_n}")
    
    # Clustering findings
    if 'hierarchical_clustering' in analysis.get('method_summary', {}):
        clustering_summary = analysis['method_summary']['hierarchical_clustering']
        if clustering_summary.get('matrices'):
            num_clusters = []
            for matrix_data in clustering_summary['matrices'].values():
                num_clusters.append(matrix_data.get('num_clusters', 0))
            
            if num_clusters:
                avg_clusters = np.mean(num_clusters)
                findings.append(f"Average number of clusters across methods: {avg_clusters:.1f}")
    
    # Consistency findings
    consistency = analysis.get('consistency_analysis', {})
    if 'overall_assessment' in consistency:
        findings.append(f"Method consistency: {consistency['overall_assessment']}")
    
    return findings

def generate_recommendations(results: Dict, analysis: Dict) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []
    
    # Method selection recommendations
    consistency = analysis.get('consistency_analysis', {})
    correlations = consistency.get('distance_matrix_correlations', {})
    
    if correlations:
        # Find most consistent methods
        best_correlation = max(correlations.values()) if correlations else 0
        if best_correlation > 0.7:
            recommendations.append("High consistency between methods - results are reliable")
        else:
            recommendations.append("Consider validating results with domain knowledge due to method disagreement")
    
    # Sample size recommendations
    num_binaries = results.get('corpus_info', {}).get('num_binaries', 0)
    if num_binaries < 10:
        recommendations.append("Consider analyzing more binaries for more robust clustering results")
    
    # Clustering method recommendations
    if 'hierarchical_clustering' in analysis.get('method_summary', {}):
        clustering_summary = analysis['method_summary']['hierarchical_clustering']
        if 'consistency' in clustering_summary:
            mean_consistency = clustering_summary['consistency'].get('mean_consistency', 0)
            if mean_consistency < 0.5:
                recommendations.append("Different distance metrics produce different clusterings - consider ensemble methods")
    
    return recommendations

def interpret_ncd_distances(mean_distance: float) -> str:
    """Interpret mean NCD distance."""
    if mean_distance < 0.2:
        return "Very similar binaries - high structural similarity"
    elif mean_distance < 0.5:
        return "Moderately similar binaries - some shared patterns"
    elif mean_distance < 0.8:
        return "Dissimilar binaries - limited shared structure"
    else:
        return "Very dissimilar binaries - minimal shared structure"

def interpret_ngram_similarity(mean_similarity: float) -> str:
    """Interpret mean n-gram similarity."""
    if mean_similarity > 0.8:
        return "Very high n-gram similarity - binaries share many instruction patterns"
    elif mean_similarity > 0.6:
        return "High n-gram similarity - significant shared instruction patterns"
    elif mean_similarity > 0.4:
        return "Moderate n-gram similarity - some shared instruction patterns"
    else:
        return "Low n-gram similarity - limited shared instruction patterns"

def interpret_method_consistency(mean_correlation: float) -> str:
    """Interpret consistency between clustering methods."""
    if mean_correlation > 0.8:
        return "Very high consistency - all methods agree strongly"
    elif mean_correlation > 0.6:
        return "High consistency - most methods agree"
    elif mean_correlation > 0.4:
        return "Moderate consistency - some agreement between methods"
    else:
        return "Low consistency - significant disagreement between methods"