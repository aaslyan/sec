"""
Dimensionality reduction and visualization for binary similarity analysis.

Provides t-SNE, UMAP, and PCA for visualizing binary relationships in 2D space.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import Binary, save_json

logger = logging.getLogger(__name__)

class DimensionalityReducer:
    """Dimensionality reduction for binary similarity visualization."""
    
    def __init__(self, random_state: int = 42):
        """Initialize with fixed random state for reproducibility."""
        self.random_state = random_state
        self.embeddings = {}
        self.binary_names = []
    
    def fit_pca(self, data_matrix: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, Dict]:
        """
        Fit PCA and return 2D embedding.
        
        Args:
            data_matrix: Input data matrix (n_samples, n_features)
            n_components: Number of PCA components
            
        Returns:
            embedding: 2D coordinates
            info: PCA information (explained variance, etc.)
        """
        from sklearn.decomposition import PCA
        
        logger.info(f"Fitting PCA with {n_components} components")
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        embedding = pca.fit_transform(data_matrix)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        info = {
            'method': 'PCA',
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'total_variance_explained': float(cumulative_variance[-1]),
            'n_components': n_components
        }
        
        self.embeddings['pca'] = embedding
        
        logger.info(f"PCA completed: {info['total_variance_explained']:.2%} variance explained")
        
        return embedding, info
    
    def fit_tsne(self, data_matrix: np.ndarray, perplexity: float = 30.0, 
                 n_iter: int = 1000) -> Tuple[np.ndarray, Dict]:
        """
        Fit t-SNE and return 2D embedding.
        
        Args:
            data_matrix: Input data matrix or distance matrix
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
            
        Returns:
            embedding: 2D coordinates
            info: t-SNE information
        """
        from sklearn.manifold import TSNE
        
        # Adjust perplexity for small datasets
        n_samples = data_matrix.shape[0]
        adjusted_perplexity = min(perplexity, (n_samples - 1) / 3.0)
        
        logger.info(f"Fitting t-SNE with perplexity={adjusted_perplexity:.1f}")
        
        tsne = TSNE(
            n_components=2,
            perplexity=adjusted_perplexity,
            n_iter=n_iter,
            random_state=self.random_state,
            init='pca'
        )
        
        embedding = tsne.fit_transform(data_matrix)
        
        info = {
            'method': 't-SNE',
            'perplexity': adjusted_perplexity,
            'n_iter': n_iter,
            'kl_divergence': float(tsne.kl_divergence_) if hasattr(tsne, 'kl_divergence_') else None
        }
        
        self.embeddings['tsne'] = embedding
        
        logger.info(f"t-SNE completed: KL divergence = {info['kl_divergence']}")
        
        return embedding, info
    
    def fit_umap(self, data_matrix: np.ndarray, n_neighbors: int = 15, 
                 min_dist: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        Fit UMAP and return 2D embedding.
        
        Args:
            data_matrix: Input data matrix
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            
        Returns:
            embedding: 2D coordinates  
            info: UMAP information
        """
        try:
            import umap
        except ImportError:
            logger.warning("UMAP not available - skipping UMAP embedding")
            return np.zeros((data_matrix.shape[0], 2)), {'method': 'UMAP', 'error': 'UMAP not installed'}
        
        # Adjust n_neighbors for small datasets
        n_samples = data_matrix.shape[0]
        adjusted_neighbors = min(n_neighbors, n_samples - 1)
        
        logger.info(f"Fitting UMAP with n_neighbors={adjusted_neighbors}")
        
        reducer = umap.UMAP(
            n_neighbors=adjusted_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=self.random_state
        )
        
        embedding = reducer.fit_transform(data_matrix)
        
        info = {
            'method': 'UMAP',
            'n_neighbors': adjusted_neighbors,
            'min_dist': min_dist
        }
        
        self.embeddings['umap'] = embedding
        
        logger.info("UMAP completed")
        
        return embedding, info
    
    def plot_embedding(self, embedding: np.ndarray, binary_names: List[str], 
                      categories: Optional[Dict[str, str]] = None,
                      method_name: str = "Embedding", output_path: Path = None,
                      figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot 2D embedding with optional category coloring.
        
        Args:
            embedding: 2D coordinates
            binary_names: Names of binaries
            categories: Optional mapping from binary name to category
            method_name: Name of embedding method for title
            output_path: Path to save plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        if categories:
            # Color by category
            unique_categories = sorted(set(categories.values()))
            colors = sns.color_palette("husl", len(unique_categories))
            category_colors = dict(zip(unique_categories, colors))
            
            for category in unique_categories:
                # Find binaries in this category
                category_indices = [i for i, name in enumerate(binary_names) 
                                  if categories.get(name) == category]
                
                if category_indices:
                    plt.scatter(embedding[category_indices, 0], 
                              embedding[category_indices, 1],
                              c=[category_colors[category]], 
                              label=category, s=60, alpha=0.7)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Single color
            plt.scatter(embedding[:, 0], embedding[:, 1], s=60, alpha=0.7)
        
        # Add binary name annotations
        for i, name in enumerate(binary_names):
            plt.annotate(name, (embedding[i, 0], embedding[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
        
        plt.title(f'{method_name} - Binary Similarity Visualization')
        plt.xlabel(f'{method_name} Component 1')
        plt.ylabel(f'{method_name} Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved {method_name} plot to {output_path}")
        else:
            plt.show()

def infer_binary_categories(binary_names: List[str]) -> Dict[str, str]:
    """
    Infer binary categories from names using common patterns.
    
    This is a heuristic approach for demonstration purposes.
    """
    categories = {}
    
    # Common binary categories and patterns
    category_patterns = {
        'text_processing': ['grep', 'sed', 'awk', 'cut', 'sort', 'uniq', 'tr', 'wc'],
        'file_operations': ['ls', 'cp', 'mv', 'rm', 'mkdir', 'rmdir', 'find', 'locate'],
        'system': ['ps', 'kill', 'top', 'df', 'du', 'mount', 'umount'],
        'compression': ['gzip', 'gunzip', 'zip', 'unzip', 'tar', 'bzip2'],
        'networking': ['ping', 'wget', 'curl', 'ssh', 'scp'],
        'development': ['gcc', 'make', 'git', 'python', 'node'],
        'utilities': ['true', 'false', 'yes', 'echo', 'cat', 'head', 'tail']
    }
    
    # Assign categories based on patterns
    for binary_name in binary_names:
        assigned = False
        for category, patterns in category_patterns.items():
            if any(pattern in binary_name.lower() for pattern in patterns):
                categories[binary_name] = category
                assigned = True
                break
        
        if not assigned:
            categories[binary_name] = 'other'
    
    return categories

def run_dimensionality_reduction(similarity_matrices: Dict, binary_names: List[str],
                                output_dir: Path,
                                categories: Optional[Dict[str, str]] = None) -> Dict:
    """
    Run dimensionality reduction on similarity matrices.

    Args:
        similarity_matrices: Dict with similarity matrices (NOT distance matrices)
        binary_names: Names of binaries
        output_dir: Output directory for plots and results
        categories: Optional mapping from binary name to category label.
                    If None, falls back to heuristic name-based inference.
    """
    logger.info("Running dimensionality reduction analysis...")

    results = {}

    # Use provided categories or fall back to heuristic inference
    if categories is None:
        categories = infer_binary_categories(binary_names)
        logger.info("Using heuristic category inference (Binary.category not available)")
    
    for matrix_name, similarity_matrix in similarity_matrices.items():
        if isinstance(similarity_matrix, list):
            similarity_matrix = np.array(similarity_matrix)
        
        logger.info(f"Processing {matrix_name} similarity matrix...")
        
        try:
            reducer = DimensionalityReducer()
            matrix_results = {}
            
            # PCA
            try:
                pca_embedding, pca_info = reducer.fit_pca(similarity_matrix)
                matrix_results['pca'] = {
                    'embedding': pca_embedding.tolist(),
                    'info': pca_info
                }
                
                # Plot PCA
                pca_plot_path = output_dir / f"pca_{matrix_name}.png"
                reducer.plot_embedding(pca_embedding, binary_names, categories, 
                                     "PCA", pca_plot_path)
                
            except Exception as e:
                logger.warning(f"PCA failed for {matrix_name}: {e}")
                matrix_results['pca'] = {'error': str(e)}
            
            # t-SNE (use distance matrix)
            try:
                distance_matrix = 1.0 - similarity_matrix
                tsne_embedding, tsne_info = reducer.fit_tsne(distance_matrix)
                matrix_results['tsne'] = {
                    'embedding': tsne_embedding.tolist(),
                    'info': tsne_info
                }
                
                # Plot t-SNE
                tsne_plot_path = output_dir / f"tsne_{matrix_name}.png"
                reducer.plot_embedding(tsne_embedding, binary_names, categories, 
                                     "t-SNE", tsne_plot_path)
                
            except Exception as e:
                logger.warning(f"t-SNE failed for {matrix_name}: {e}")
                matrix_results['tsne'] = {'error': str(e)}
            
            # UMAP
            try:
                umap_embedding, umap_info = reducer.fit_umap(similarity_matrix)
                if 'error' not in umap_info:
                    matrix_results['umap'] = {
                        'embedding': umap_embedding.tolist(),
                        'info': umap_info
                    }
                    
                    # Plot UMAP
                    umap_plot_path = output_dir / f"umap_{matrix_name}.png"
                    reducer.plot_embedding(umap_embedding, binary_names, categories,
                                         "UMAP", umap_plot_path)
                else:
                    matrix_results['umap'] = umap_info
                    
            except Exception as e:
                logger.warning(f"UMAP failed for {matrix_name}: {e}")
                matrix_results['umap'] = {'error': str(e)}
            
            results[matrix_name] = matrix_results
            
        except Exception as e:
            logger.error(f"Dimensionality reduction failed for {matrix_name}: {e}")
            results[matrix_name] = {'error': str(e)}
    
    # Add category information
    results['categories'] = categories
    results['category_summary'] = analyze_categories(categories)
    
    # Save results
    save_json(results, output_dir / "dimensionality_reduction.json")
    
    logger.info("Dimensionality reduction analysis completed")
    return results

def analyze_categories(categories: Dict[str, str]) -> Dict:
    """Analyze the distribution of binary categories."""
    from collections import Counter
    
    category_counts = Counter(categories.values())
    
    return {
        'category_distribution': dict(category_counts),
        'num_categories': len(category_counts),
        'largest_category': category_counts.most_common(1)[0] if category_counts else None,
        'single_binary_categories': sum(1 for count in category_counts.values() if count == 1)
    }