#!/usr/bin/env python
"""
NLI-based Direction Similarity for DORA.

Implements the NLI-based direction similarity proposed in:
  "AdaptSci: Training-Free Adaptive Test-Time Compute for Agentic Scientific Equation Discovery"

Instead of relying on cosine similarity (which captures semantic resemblance),
this module uses Natural Language Inference (NLI) to measure whether two
reasoning trajectories are logically equivalent, mutually supportive, or
contradictory — a better proxy for "same reasoning direction" in structured
reasoning tasks.

Key formula:
  s(i,j) = 0.5*(Pe(ti,tj) + Pe(tj,ti)) - λ * 0.5*(Pc(ti,tj) + Pc(tj,ti))

where Pe = entailment probability, Pc = contradiction probability, and λ
controls contradiction penalty strength.
"""

import logging
import numpy as np
import torch
from typing import List, Optional, Tuple
from copy import deepcopy

logger = logging.getLogger(__name__)


class NLISimilarityModel:
    """
    Wraps a pretrained NLI classifier to compute pairwise direction similarity
    between reasoning trajectories.

    Supported models:
      - FacebookAI/roberta-large-mnli
      - microsoft/deberta-large-mnli
      - MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingual)
    """

    def __init__(
        self,
        model_path: str = "FacebookAI/roberta-large-mnli",
        lambda_penalty: float = 1.0,
        max_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.lambda_penalty = lambda_penalty
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading NLI model from {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Detect label ordering from model config
        self._detect_label_order()

    def _detect_label_order(self):
        """Detect which index corresponds to entailment/contradiction/neutral."""
        id2label = self.model.config.id2label
        self.entailment_idx = None
        self.contradiction_idx = None
        self.neutral_idx = None

        for idx, label in id2label.items():
            label_lower = label.lower()
            if "entail" in label_lower:
                self.entailment_idx = int(idx)
            elif "contra" in label_lower:
                self.contradiction_idx = int(idx)
            elif "neutral" in label_lower:
                self.neutral_idx = int(idx)

        if self.entailment_idx is None or self.contradiction_idx is None:
            # Fallback: assume standard MNLI ordering [contradiction, neutral, entailment]
            logger.warning(
                "Could not detect label ordering from model config. "
                "Falling back to [contradiction=0, neutral=1, entailment=2]."
            )
            self.contradiction_idx = 0
            self.neutral_idx = 1
            self.entailment_idx = 2

        logger.info(
            f"NLI label indices: entailment={self.entailment_idx}, "
            f"contradiction={self.contradiction_idx}, neutral={self.neutral_idx}"
        )

    @torch.no_grad()
    def _predict_batch(
        self, premises: List[str], hypotheses: List[str]
    ) -> np.ndarray:
        """
        Run NLI inference on a batch of (premise, hypothesis) pairs.

        Returns:
            np.ndarray of shape (batch_size, 3) with probabilities for each class.
        """
        all_probs = []

        for start in range(0, len(premises), self.batch_size):
            end = min(start + self.batch_size, len(premises))
            batch_premises = premises[start:end]
            batch_hypotheses = hypotheses[start:end]

            encodings = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def compute_pairwise_nli_score(
        self, text_i: str, text_j: str
    ) -> float:
        """
        Compute the symmetric NLI direction similarity score between two texts.

        s(i,j) = 0.5*(Pe(ti,tj) + Pe(tj,ti)) - λ * 0.5*(Pc(ti,tj) + Pc(tj,ti))
        """
        # Forward: ti as premise, tj as hypothesis
        probs_ij = self._predict_batch([text_i], [text_j])[0]
        # Backward: tj as premise, ti as hypothesis
        probs_ji = self._predict_batch([text_j], [text_i])[0]

        pe = 0.5 * (probs_ij[self.entailment_idx] + probs_ji[self.entailment_idx])
        pc = 0.5 * (probs_ij[self.contradiction_idx] + probs_ji[self.contradiction_idx])

        return float(pe - self.lambda_penalty * pc)

    def _normalize_for_dora(self, matrix: np.ndarray) -> np.ndarray:
        """
        Map NLI scores from [-lambda, 1.0] to [0, 1] so they are compatible
        with DORA's softmax_with_temperature_1 uniqueness estimator (γ_i).

        Without this, the negative off-diagonal values caused by contradiction
        penalty make the row-softmax diagonal entries all collapse to ~1.0
        at the low balance_alpha temperature (0.01), destroying the uniqueness
        discrimination that DORA relies on to separate clustered from unique beams.

        The shift is: s_norm = (s + lambda) / (1 + lambda)
        which maps: -lambda → 0.0, 0.0 → lambda/(1+lambda), 1.0 → 1.0
        This preserves relative ordering and keeps diagonal = 1.0.
        """
        out = (matrix + self.lambda_penalty) / (1.0 + self.lambda_penalty)
        # Clamp to [0,1] to absorb any floating-point overshoot
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def compute_nli_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Compute the full pairwise NLI similarity matrix for a list of texts.

        Returns:
            np.ndarray of shape (n, n) with NLI direction similarity scores
            normalized to [0, 1] for compatibility with DORA's uniqueness
            estimator (γ_i = diagonal of row-softmax at low temperature).
        """
        n = len(texts)
        sim_matrix = np.zeros((n, n), dtype=np.float32)

        if n <= 1:
            sim_matrix[0, 0] = 1.0
            return sim_matrix

        # Build all ordered pairs (i, j) where i < j, interleaved as
        # (forward, backward) so the stride-2 assembly loop stays correct.
        premises = []
        hypotheses = []
        pair_indices = []

        for i in range(n):
            for j in range(i + 1, n):
                # Forward direction
                premises.append(texts[i])
                hypotheses.append(texts[j])
                pair_indices.append((i, j, "forward"))
                # Backward direction
                premises.append(texts[j])
                hypotheses.append(texts[i])
                pair_indices.append((i, j, "backward"))

        # Run batch inference
        all_probs = self._predict_batch(premises, hypotheses)

        # Assemble the raw similarity matrix
        for k in range(0, len(pair_indices), 2):
            i, j, _ = pair_indices[k]
            probs_ij = all_probs[k]      # forward
            probs_ji = all_probs[k + 1]  # backward

            pe = 0.5 * (probs_ij[self.entailment_idx] + probs_ji[self.entailment_idx])
            pc = 0.5 * (probs_ij[self.contradiction_idx] + probs_ji[self.contradiction_idx])
            score = pe - self.lambda_penalty * pc

            sim_matrix[i, j] = score
            sim_matrix[j, i] = score

        # Diagonal: self-similarity = 1.0
        np.fill_diagonal(sim_matrix, 1.0)

        # Normalize to [0, 1] so DORA's balance_alpha temperature is meaningful
        return self._normalize_for_dora(sim_matrix)

    def compute_nli_matrix_hybrid(
        self,
        texts: List[str],
        cosine_sim_matrix: np.ndarray,
        top_k: int = 10,
    ) -> np.ndarray:
        """
        Two-stage hybrid: use cosine similarity to select top-k neighbors,
        then refine only those pairs with NLI scoring.

        NLI scores for refined pairs are normalized to [0, 1] via
        _normalize_for_dora() so they are on the same scale as the cosine
        values retained for non-refined pairs.

        Args:
            texts: List of trajectory texts.
            cosine_sim_matrix: Pre-computed cosine similarity matrix (n x n).
            top_k: Number of nearest neighbors per trajectory to refine with NLI.

        Returns:
            np.ndarray of shape (n, n) with hybrid similarity scores in [0, 1].
        """
        n = len(texts)
        # Start with cosine similarity as the base
        hybrid_matrix = cosine_sim_matrix.copy().astype(np.float32)

        if n <= 1:
            return hybrid_matrix

        # For each trajectory, find top-k most similar neighbors by cosine
        premises = []
        hypotheses = []
        pair_indices = []
        pairs_seen = set()

        for i in range(n):
            # Get top-k neighbors (excluding self)
            cos_scores = cosine_sim_matrix[i].copy()
            cos_scores[i] = -np.inf  # exclude self
            top_neighbors = np.argsort(cos_scores)[-top_k:]

            for j in top_neighbors:
                pair_key = (min(i, j), max(i, j))
                if pair_key not in pairs_seen:
                    pairs_seen.add(pair_key)
                    # Forward
                    premises.append(texts[i])
                    hypotheses.append(texts[j])
                    pair_indices.append((i, j, "forward"))
                    # Backward
                    premises.append(texts[j])
                    hypotheses.append(texts[i])
                    pair_indices.append((i, j, "backward"))

        if len(premises) == 0:
            return hybrid_matrix

        # Batch NLI inference on shortlisted pairs
        all_probs = self._predict_batch(premises, hypotheses)

        # Update the matrix with NLI scores for refined pairs
        for k in range(0, len(pair_indices), 2):
            i, j, _ = pair_indices[k]
            probs_ij = all_probs[k]
            probs_ji = all_probs[k + 1]

            pe = 0.5 * (probs_ij[self.entailment_idx] + probs_ji[self.entailment_idx])
            pc = 0.5 * (probs_ij[self.contradiction_idx] + probs_ji[self.contradiction_idx])
            raw_score = pe - self.lambda_penalty * pc
            # Normalize to [0,1] to stay on the same scale as cosine values
            norm_score = float(np.clip(
                (raw_score + self.lambda_penalty) / (1.0 + self.lambda_penalty), 0.0, 1.0
            ))
            hybrid_matrix[i, j] = norm_score
            hybrid_matrix[j, i] = norm_score

        # Defensive: ensure diagonal is exactly 1.0 (cosine copy should already
        # have this, but guard against edge cases)
        np.fill_diagonal(hybrid_matrix, 1.0)

        return hybrid_matrix


def summarize_trajectory(text: str, max_steps: int = 5) -> str:
    """
    Compress a reasoning trajectory into a step-wise summary to reduce noise
    and NLI inference cost.

    Extracts step headers (## Step N: ...) and their first sentence,
    or falls back to the first max_steps paragraphs.
    """
    import re

    # Try to extract structured steps
    step_pattern = re.compile(
        r"##\s*Step\s*\d+[:\s]*(.*?)(?=##\s*Step|\Z)", re.DOTALL | re.IGNORECASE
    )
    steps = step_pattern.findall(text)

    if steps and len(steps) >= 1:
        # Use first sentence of each step
        summaries = []
        for step in steps[:max_steps]:
            step = step.strip()
            # Get first sentence
            sentence_end = re.search(r'[.!?\n]', step)
            if sentence_end:
                summaries.append(step[: sentence_end.end()].strip())
            else:
                summaries.append(step[:200].strip())
        return " | ".join(summaries)

    # Fallback: split by double newlines and take first max_steps paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        selected = paragraphs[:max_steps]
        return " | ".join(
            p[:200] if len(p) > 200 else p for p in selected
        )

    # Last fallback: truncate
    return text[:1000]


def load_nli_model(config) -> Optional[NLISimilarityModel]:
    """
    Load the NLI model based on config settings.
    Returns None if NLI is not needed.
    """
    if config.similarity_mode not in ("nli", "hybrid"):
        return None

    if config.nli_model_path is None:
        logger.warning(
            f"similarity_mode='{config.similarity_mode}' but nli_model_path is None. "
            "Falling back to cosine similarity."
        )
        return None

    return NLISimilarityModel(
        model_path=config.nli_model_path,
        lambda_penalty=config.nli_lambda_penalty,
        max_length=config.nli_max_length,
        batch_size=config.nli_batch_size,
    )