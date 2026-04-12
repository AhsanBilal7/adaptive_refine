"""
clustering.py
-------------
Semantic clustering of reasoning traces for DN-BoN.

The SentenceTransformer (or HF AutoModel) is loaded ONCE in ClusterManager.__init__
and stored as self._st_model / self._hf_model.  The cluster() method is called once
per question but never reloads the embedding model — it just runs a forward pass on
the already-loaded model.

If sentence-transformers is unavailable, falls back to a HF AutoModel mean-pool
encoder, then to lightweight TF-IDF if neither is installed.
"""

import logging
import re
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Clustering back-ends  (pure numpy / sklearn, no model loading here)
# ──────────────────────────────────────────────────────────────────────────────

def _cluster_agglomerative(
    embeddings: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering  # type: ignore
    n = len(embeddings)
    if n == 1:
        return np.array([0])
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(embeddings).astype(int)


def _cluster_threshold_greedy(
    embeddings: np.ndarray,
    similarity_threshold: float,
) -> np.ndarray:
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
# TF-IDF fallback  (stateless, no persistent model needed)
# ──────────────────────────────────────────────────────────────────────────────

def _embed_tfidf(texts: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(max_features=4096, sublinear_tf=True)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ──────────────────────────────────────────────────────────────────────────────
# ClusterManager
# ──────────────────────────────────────────────────────────────────────────────

class ClusterManager:
    """
    Semantic clustering of reasoning traces.

    The embedding model is loaded ONCE at __init__ time and stored as an
    instance attribute.  Every call to cluster() reuses the same in-memory
    model object — no reloading, no re-downloading.

    Parameters
    ----------
    embedding_model    : sentence-transformers model id (e.g. 'all-MiniLM-L6-v2')
                         or 'tfidf' for the lightweight fallback.
    similarity_threshold: cosine-similarity cutoff in [0, 1].
    clustering_method  : 'agglomerative' | 'greedy'
    use_answer_only    : cluster only the \\boxed{} answer instead of full trace.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        clustering_method: str = "agglomerative",
        use_answer_only: bool = False,
    ):
        self.embedding_model     = embedding_model
        self.similarity_threshold = similarity_threshold
        self.distance_threshold  = 1.0 - similarity_threshold
        self.clustering_method   = clustering_method
        self.use_answer_only     = use_answer_only

        # These are set once below and reused on every cluster() call
        self._st_model   = None   # SentenceTransformer instance
        self._hf_model   = None   # HF AutoModel instance
        self._hf_tok     = None   # HF AutoTokenizer instance
        self._hf_device  = None
        self._embedder   = self._load_embedder()  # sets the fields above

    # ── model loading  (called ONCE in __init__) ──────────────────────────────

    def _load_embedder(self) -> str:
        """
        Load the embedding model once and cache it.
        Returns a string tag: 'sentence_transformers' | 'hf_transformer' | 'tfidf'.
        """
        if self.embedding_model == "tfidf":
            logger.info("ClusterManager: using TF-IDF embedder (no model load).")
            return "tfidf"

        # Try sentence-transformers first (best quality / easiest API)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info(
                f"ClusterManager: loading SentenceTransformer "
                f"'{self.embedding_model}' — this happens ONCE."
            )
            self._st_model = SentenceTransformer(self.embedding_model)
            logger.info("ClusterManager: SentenceTransformer ready.")
            return "sentence_transformers"
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; trying HF AutoModel fallback."
            )
        except Exception as e:
            logger.warning(f"SentenceTransformer load failed ({e}); trying HF fallback.")

        # Try HF AutoModel fallback
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            logger.info(
                f"ClusterManager: loading HF AutoModel "
                f"'{self.embedding_model}' — this happens ONCE."
            )
            self._hf_tok    = AutoTokenizer.from_pretrained(self.embedding_model)
            self._hf_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._hf_model  = AutoModel.from_pretrained(
                self.embedding_model
            ).to(self._hf_device).eval()
            logger.info("ClusterManager: HF AutoModel ready.")
            return "hf_transformer"
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"HF AutoModel load failed ({e}); using TF-IDF fallback.")

        logger.warning(
            "ClusterManager: no neural embedder available — using TF-IDF fallback."
        )
        return "tfidf"

    # ── embedding  (uses the already-loaded model) ────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the pre-loaded model. Never reloads anything."""
        if self._embedder == "sentence_transformers":
            # _st_model is already in memory from __init__
            return self._st_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )

        if self._embedder == "hf_transformer":
            import torch
            all_embs = []
            bs = 32
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                enc = self._hf_tok(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(self._hf_device)
                with torch.no_grad():
                    out = self._hf_model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                emb  = torch.nn.functional.normalize(emb, dim=-1)
                all_embs.append(emb.cpu().numpy())
            return np.vstack(all_embs)

        # TF-IDF — stateless, no model to cache
        return _embed_tfidf(texts)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_texts_to_embed(self, traces: List[str]) -> List[str]:
        if not self.use_answer_only:
            return traces
        from verifier import extract_boxed_answer  # type: ignore
        return [extract_boxed_answer(t) or t for t in traces]

    # ── public API ────────────────────────────────────────────────────────────

    def cluster(self, traces: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster traces into semantic reasoning directions.

        Called once per question.  The embedding model is NOT reloaded here —
        it was loaded once in __init__ and is reused every call.

        Returns
        -------
        labels        : np.ndarray shape (N,)  — integer cluster id per trace
        cluster_sizes : np.ndarray shape (N,)  — size of each trace's cluster
        """
        n = len(traces)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int), np.array([1], dtype=int)

        texts      = self._get_texts_to_embed(traces)
        embeddings = self._embed(texts)

        if self.clustering_method == "agglomerative":
            try:
                labels = _cluster_agglomerative(embeddings, self.distance_threshold)
            except Exception as e:
                logger.warning(f"Agglomerative failed ({e}); using greedy.")
                labels = _cluster_threshold_greedy(embeddings, self.similarity_threshold)
        else:
            labels = _cluster_threshold_greedy(embeddings, self.similarity_threshold)

        unique, counts = np.unique(labels, return_counts=True)
        size_map       = dict(zip(unique.tolist(), counts.tolist()))
        cluster_sizes  = np.array([size_map[int(l)] for l in labels], dtype=int)

        return labels, cluster_sizes