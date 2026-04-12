"""
clustering.py
-------------
Semantic clustering of reasoning traces for DN-BoN.

Provides the ClusterManager class which:
  1. Embeds each trace with a sentence-transformers model.
  2. Groups traces into semantic clusters via agglomerative clustering
     (or a simpler cosine-threshold approach when sklearn is available).
  3. Returns cluster assignments and cluster sizes.

If sentence-transformers is unavailable, falls back to a lightweight
TF-IDF + cosine-similarity clustering so the code still runs.
"""

import logging
import re
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding back-ends
# ──────────────────────────────────────────────────────────────────────────────

def _embed_sentence_transformers(texts: List[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _embed_tfidf(texts: List[str]) -> np.ndarray:
    """
    Fallback: sparse TF-IDF embeddings normalised to unit L2 norm.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(max_features=4096, sublinear_tf=True)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _embed_hf_transformer(texts: List[str], model_name: str) -> np.ndarray:
    """
    Encode texts with a HF AutoModel + mean-pooling (last hidden state).
    Used when sentence-transformers is unavailable but transformers is.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    all_embs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        # mean pool over non-padding tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


# ──────────────────────────────────────────────────────────────────────────────
# Clustering back-ends
# ──────────────────────────────────────────────────────────────────────────────

def _cluster_agglomerative(
    embeddings: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    """
    Agglomerative clustering with a cosine distance threshold.
    Returns integer cluster-id array of shape (N,).
    """
    from sklearn.cluster import AgglomerativeClustering  # type: ignore

    n = len(embeddings)
    if n == 1:
        return np.array([0])

    # cosine distance = 1 - cosine_similarity (embeddings are L2-normalised)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    return labels.astype(int)


def _cluster_threshold_greedy(
    embeddings: np.ndarray,
    similarity_threshold: float,
) -> np.ndarray:
    """
    Simple greedy cosine-threshold clustering (O(N²)).
    Assigns each trace to the first existing cluster whose centroid has
    cosine similarity ≥ similarity_threshold, or starts a new cluster.
    """
    n = len(embeddings)
    labels = -np.ones(n, dtype=int)
    centroids: List[np.ndarray] = []
    cluster_sizes: List[int] = []

    for i in range(n):
        if not centroids:
            labels[i] = 0
            centroids.append(embeddings[i].copy())
            cluster_sizes.append(1)
            continue
        sims = np.array([embeddings[i] @ c for c in centroids])
        best = int(np.argmax(sims))
        if sims[best] >= similarity_threshold:
            labels[i] = best
            # update centroid (running mean)
            sz = cluster_sizes[best]
            centroids[best] = (centroids[best] * sz + embeddings[i]) / (sz + 1)
            centroids[best] /= np.linalg.norm(centroids[best]) + 1e-12
            cluster_sizes[best] += 1
        else:
            new_id = len(centroids)
            labels[i] = new_id
            centroids.append(embeddings[i].copy())
            cluster_sizes.append(1)

    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Main ClusterManager class
# ──────────────────────────────────────────────────────────────────────────────

class ClusterManager:
    """
    Semantic clustering of reasoning traces.

    Parameters
    ----------
    embedding_model : sentence-transformers model id
                      (e.g. 'all-MiniLM-L6-v2') or 'tfidf' for fallback
    similarity_threshold : cosine-similarity threshold in [0, 1].
                           Traces with similarity ≥ threshold are grouped.
                           Distance threshold = 1 - similarity_threshold.
    clustering_method : 'agglomerative' | 'greedy'
    use_answer_only   : if True, cluster only the extracted \boxed{} answer
                        instead of the full trace (faster, less semantic)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        clustering_method: str = "agglomerative",
        use_answer_only: bool = False,
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.distance_threshold = 1.0 - similarity_threshold
        self.clustering_method = clustering_method
        self.use_answer_only = use_answer_only
        self._embedder = self._select_embedder()

    # ── private ───────────────────────────────────────────────────────────────

    def _select_embedder(self):
        if self.embedding_model == "tfidf":
            return "tfidf"
        try:
            import sentence_transformers  # noqa: F401
            return "sentence_transformers"
        except ImportError:
            pass
        try:
            import transformers  # noqa: F401
            return "hf_transformer"
        except ImportError:
            pass
        logger.warning(
            "Neither sentence-transformers nor transformers found. "
            "Falling back to TF-IDF embeddings."
        )
        return "tfidf"

    def _get_texts_to_embed(self, traces: List[str]) -> List[str]:
        if not self.use_answer_only:
            return traces
        # Extract boxed answer; fall back to full trace if missing
        from verifier import extract_boxed_answer  # type: ignore
        return [extract_boxed_answer(t) or t for t in traces]

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self._embedder == "sentence_transformers":
            return _embed_sentence_transformers(texts, self.embedding_model)
        if self._embedder == "hf_transformer":
            return _embed_hf_transformer(texts, self.embedding_model)
        return _embed_tfidf(texts)

    # ── public ────────────────────────────────────────────────────────────────

    def cluster(self, traces: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster the traces and return (labels, cluster_sizes).

        Parameters
        ----------
        traces : list of model-generated reasoning strings

        Returns
        -------
        labels       : np.ndarray shape (N,), integer cluster id per trace
        cluster_sizes: np.ndarray shape (N,), size of each trace's cluster
        """
        n = len(traces)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int), np.array([1], dtype=int)

        texts = self._get_texts_to_embed(traces)
        embeddings = self._embed(texts)

        if self.clustering_method == "agglomerative":
            try:
                labels = _cluster_agglomerative(embeddings, self.distance_threshold)
            except Exception as e:
                logger.warning(f"Agglomerative clustering failed ({e}); using greedy.")
                labels = _cluster_threshold_greedy(embeddings, self.similarity_threshold)
        else:
            labels = _cluster_threshold_greedy(embeddings, self.similarity_threshold)

        # Compute cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        size_map = dict(zip(unique.tolist(), counts.tolist()))
        cluster_sizes = np.array([size_map[int(l)] for l in labels], dtype=int)

        return labels, cluster_sizes
