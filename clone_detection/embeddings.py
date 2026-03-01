"""
TF-IDF function embeddings with UMAP / PCA dimensionality reduction (bonus).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class FunctionEmbedder:
    """Build TF-IDF embeddings of opcode k-shingles per function."""

    def __init__(self, k: int = 3, max_features: int = 5000):
        self.k = k
        self.max_features = max_features
        self._vectorizer: Optional[TfidfVectorizer] = None

    def _to_document(self, seq: List[str]) -> str:
        """Convert an opcode sequence to a shingle document."""
        if len(seq) < self.k:
            return ''
        shingles = [' '.join(seq[i:i + self.k]) for i in range(len(seq) - self.k + 1)]
        return ' | '.join(shingles)

    def build_tfidf_matrix(
        self,
        func_ids: List[str],
        seq_map: Dict[str, List[str]],
    ) -> np.ndarray:
        """Return a (n_funcs, max_features) dense TF-IDF matrix."""
        docs = [self._to_document(seq_map.get(fid, [])) for fid in func_ids]

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            token_pattern=r'[^\|]+',
            sublinear_tf=True,
        )
        try:
            mat = self._vectorizer.fit_transform(docs)
            return mat.toarray()
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}")
            return np.zeros((len(func_ids), 1))

    def reduce_pca(
        self,
        tfidf_matrix: np.ndarray,
        n_components: int = 2,
    ) -> Tuple[np.ndarray, Dict]:
        """PCA reduction to 2D. Returns (embedding, info_dict)."""
        n_components = min(n_components, tfidf_matrix.shape[0], tfidf_matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        embedding = pca.fit_transform(tfidf_matrix)
        info = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(pca.explained_variance_ratio_.sum()),
        }
        return embedding, info

    def reduce_umap(
        self,
        tfidf_matrix: np.ndarray,
        n_components: int = 2,
    ) -> Tuple[np.ndarray, Dict]:
        """UMAP reduction to 2D. Falls back to PCA if umap-learn unavailable."""
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=min(15, tfidf_matrix.shape[0] - 1),
            )
            embedding = reducer.fit_transform(tfidf_matrix)
            info = {'method': 'umap', 'n_components': n_components}
            return embedding, info
        except Exception as e:
            logger.warning(f"UMAP failed ({e}), falling back to PCA")
            return self.reduce_pca(tfidf_matrix, n_components)
