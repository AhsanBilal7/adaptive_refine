"""
DORA: Direction-Oriented Resource Allocation with NLI-based Direction Similarity.
HuggingFace Transformers backend (no vLLM dependency).

Supports three similarity modes:
  1. "cosine" (default): Original DORA using embedding cosine similarity.
  2. "nli": Full pairwise NLI-based direction similarity.
  3. "hybrid": Cosine for coarse retrieval → NLI for top-k refinement.
"""

import copy
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from sal.config import Config
from sal.models.reward_models import PRM
from sal.models.nli_similarity import NLISimilarityModel, summarize_trajectory
from .utils import Beam, GenResult, build_conv, generate_step, last

logger = logging.getLogger(__name__)

from sal.utils.score import aggregate_scores
from copy import deepcopy
import torch


# =============================================================================
# Embedding-based Cosine Similarity (Original DORA)
# =============================================================================

def get_embedding_local(all_completions, embedding_model, config):
    # Encode the entire list in one call instead of one-at-a-time.
    # BGE-M3 accepts a list directly; batch_size controls the internal
    # mini-batch size used by the encoder (tune to VRAM).
    with torch.no_grad():
        result = embedding_model.encode(
            all_completions,
            batch_size=getattr(config, "embedding_batch_size", 32),
            max_length=config.max_tokens,
        )
    # result['dense_vecs'] is shape (n, dim) — convert to a plain list of arrays
    return list(result['dense_vecs'])


def compute_cosine_similarity_numpy(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    normalized = embeddings / norms
    cosine_sim = np.dot(normalized, normalized.T)
    return cosine_sim.astype(np.float32)


# =============================================================================
# Similarity Matrix Construction (Cosine / NLI / Hybrid / Ablations)
# =============================================================================

def get_similarity_matrix(all_beams, config, embedding_model=None, nli_model=None):
    mode = config.similarity_mode
    if config.ablation_similarity is not None:
        mode = config.ablation_similarity

    all_texts = [beam.current_text for beam in all_beams]

    if mode == "cosine":
        return _get_cosine_matrix(all_texts, config, embedding_model)
    elif mode == "nli":
        if nli_model is None:
            logger.warning("NLI model not loaded, falling back to cosine.")
            return _get_cosine_matrix(all_texts, config, embedding_model)
        return _get_nli_matrix(all_texts, config, nli_model)
    elif mode == "hybrid":
        if nli_model is None:
            logger.warning("NLI model not loaded for hybrid, falling back to cosine.")
            return _get_cosine_matrix(all_texts, config, embedding_model)
        return _get_hybrid_matrix(all_texts, config, embedding_model, nli_model)
    elif mode == "kmeans":
        return _get_kmeans_matrix(all_texts, config, embedding_model)
    elif mode == "hierarchical":
        return _get_hierarchical_matrix(all_texts, config, embedding_model)
    else:
        raise ValueError(f"Unknown similarity mode: {mode}")


def _get_cosine_matrix(all_texts, config, embedding_model):
    if embedding_model is None:
        raise ValueError("Embedding model required for cosine similarity mode.")
    embeddings = get_embedding_local(all_texts, embedding_model, config)
    return compute_cosine_similarity_numpy(embeddings)


def _get_nli_matrix(all_texts, config, nli_model):
    if config.nli_use_summary:
        processed = [summarize_trajectory(t, config.nli_summary_max_steps) for t in all_texts]
    else:
        processed = all_texts
    return nli_model.compute_nli_matrix(processed)


def _get_hybrid_matrix(all_texts, config, embedding_model, nli_model):
    cosine_matrix = _get_cosine_matrix(all_texts, config, embedding_model)
    if config.nli_use_summary:
        processed = [summarize_trajectory(t, config.nli_summary_max_steps) for t in all_texts]
    else:
        processed = all_texts
    return nli_model.compute_nli_matrix_hybrid(processed, cosine_matrix, config.nli_top_k)


def _get_kmeans_matrix(all_texts, config, embedding_model):
    from sklearn.cluster import KMeans
    if embedding_model is None:
        raise ValueError("Embedding model required for kmeans.")
    embeddings = get_embedding_local(all_texts, embedding_model, config)
    arr = np.array(embeddings).squeeze()
    k = min(config.ablation_kmeans_k, len(all_texts))
    labels = KMeans(n_clusters=k, random_state=config.seed, n_init=10).fit_predict(arr)
    n = len(all_texts)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sim[i, j] = 1.0 if labels[i] == labels[j] else 0.0
    return sim


def _get_hierarchical_matrix(all_texts, config, embedding_model):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if embedding_model is None:
        raise ValueError("Embedding model required for hierarchical.")
    embeddings = get_embedding_local(all_texts, embedding_model, config)
    cosine_sim = compute_cosine_similarity_numpy(embeddings)
    dist = np.maximum(1.0 - cosine_sim, 0.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist), method='average')
    labels = fcluster(Z, t=1.0 - config.ablation_hierarchical_threshold, criterion='distance')
    n = len(all_texts)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sim[i, j] = 1.0 if labels[i] == labels[j] else 0.0
    return sim


# =============================================================================
# Score Processing Utilities
# =============================================================================

def softmax_with_temperature_1(matrix, temperature=1.0):
    matrix = matrix.astype(np.float64)
    scaled = matrix / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return (exp_values / exp_values.sum(axis=1, keepdims=True)).astype(np.float32)


def softmax_with_temperature(agg_scores, temperature=0.1):
    scores = np.array([score[0] for score in agg_scores], dtype=np.float64)
    scaled_scores = scores / temperature
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    softmax_scores = exp_scores / np.sum(exp_scores)
    return [float(s) for s in softmax_scores]


def get_diversity_score(M, temperature=1.0):
    N = softmax_with_temperature_1(M, temperature)
    return [N[i][i] for i in range(len(N))]


def trace_of_softmax_with_temperature(M, temperature=1.0):
    return np.trace(softmax_with_temperature_1(M, temperature))


def extract_submatrix_np(matrix, indices):
    return matrix[np.ix_(indices, indices)]


def get_diversity_quality(cos_matrix, quality_score_list, choose_candidate):
    quality_score_list = [item[0] for item in quality_score_list]
    all_candidates = list(range(len(quality_score_list)))
    all_quality = sum(quality_score_list) / len(quality_score_list)
    if len(all_candidates) > 1:
        all_diversity = (trace_of_softmax_with_temperature(
            extract_submatrix_np(cos_matrix, all_candidates), 0.1
        ) / len(all_candidates) - 1 / len(all_candidates)) / (1 - 1 / len(all_candidates))
    else:
        all_diversity = 0
    final_quality = sum(quality_score_list[idx] for idx in choose_candidate) / len(choose_candidate)
    if len(choose_candidate) > 1:
        final_diversity = (trace_of_softmax_with_temperature(
            extract_submatrix_np(cos_matrix, choose_candidate), 0.1
        ) / len(choose_candidate) - 1 / len(choose_candidate)) / (1 - 1 / len(choose_candidate))
    else:
        final_diversity = 0
    return final_quality, final_diversity, all_quality, all_diversity


def get_diversity_quality_all(cos_matrix, quality_score_list):
    quality_score_list = [item[0] for item in quality_score_list]
    all_candidates = list(range(len(quality_score_list)))
    all_quality = sum(quality_score_list) / len(quality_score_list)
    if len(all_candidates) > 1:
        all_diversity = (trace_of_softmax_with_temperature(
            extract_submatrix_np(cos_matrix, all_candidates), 0.1
        ) / len(all_candidates) - 1 / len(all_candidates)) / (1 - 1 / len(all_candidates))
    else:
        all_diversity = 0
    return all_quality, all_diversity


# =============================================================================
# Core DORA Search (Transformers backend)
# =============================================================================

def _dora(batch_of_prompts, config, model, tokenizer, prm, embedding_model=None, nli_model=None):
    """
    Core DORA search loop using HuggingFace Transformers.

    Args:
        batch_of_prompts: List of problem prompt strings.
        config: Config object.
        model: HuggingFace PreTrainedModel (policy LM).
        tokenizer: HuggingFace tokenizer.
        prm: Process Reward Model.
        embedding_model: BGE-M3 embedding model (for cosine/hybrid).
        nli_model: NLISimilarityModel (for nli/hybrid).
    """
    beams = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(Beam(prompt=prompt, index=i))

    completed_beams = []
    final_quality_list, final_diversity_list = [], []
    all_quality_list, all_diversity_list = [], []

    for iteration in tqdm(range(config.num_iterations), desc="DORA iterations"):
        if iteration == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            rebase_beams = []
            for k in range(len(allocate_num)):
                if allocate_num[k] > 0:
                    rebase_beams.extend(
                        [copy.deepcopy(active_beams[k]) for _ in range(allocate_num[k])]
                    )
            active_beams = rebase_beams

        # Pad/trim to config.n beams
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            active_beams = [copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]]

        # Build conversations and apply chat template
        is_last = iteration == config.num_iterations - 1
        convs = [build_conv(b.prompt, b.current_text, config.system_prompt) for b in active_beams]
        continue_final = iteration > 0
        add_gen_prompt = iteration == 0

        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_gen_prompt,
            continue_final_message=continue_final,
            tokenize=False,
        )

        # Determine max_new_tokens and stop strings for this step
        max_new = config.max_tokens if is_last else config.max_tokens_per_step
        stop_strings = None if is_last else ["\n\n"]
        left_tokens = [config.max_tokens - b.completion_tokens for b in active_beams]

        # Generate one step for all active beams (batched)
        gen_results = generate_step(
            templated_convs, model, tokenizer,
            temperature=config.temperature, top_p=config.top_p,
            max_new_tokens=max_new, stop_strings=stop_strings,
            left_completion_tokens=left_tokens,
            generation_batch_size=getattr(config, "generation_batch_size", 1),
        )

        # Update beams with generation results
        prompts_for_prm, completions_for_prm = [], []
        for beam, gen_result in zip(active_beams, gen_results):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
                or beam.completion_tokens >= config.max_tokens
            ):
                if is_last:
                    beam.completed = True
                    completed_beams.append(beam)
                else:
                    if gen_result.completion_tokens == config.max_tokens_per_step:
                        beam.pruned = True
                        beam.completed = True
                    else:
                        beam.completed = True
                        completed_beams.append(beam)

            prompts_for_prm.append(beam.prompt)
            completions_for_prm.append([beam.current_text])

        # Score with PRM — now a single batched forward pass (see reward_models.py)
        scores = prm.score(prompts_for_prm, completions_for_prm)
        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in sc]
            for sc in scores
        ]
        for beam, sc in zip(active_beams, scores):
            beam.all_scores = sc[0]

        # Filter out completed beams
        agg_scores = [agg_scores[i] for i, b in enumerate(active_beams) if not b.completed]
        active_beams = [b for b in active_beams if not b.completed]

        if len(active_beams) == 0:
            break

        # =================================================================
        # DORA Resource Allocation with Configurable Similarity
        # =================================================================
        all_agg_scores = deepcopy(agg_scores)

        sim_matrix = get_similarity_matrix(
            active_beams, config,
            embedding_model=embedding_model, nli_model=nli_model,
        )
        all_quality, all_diversity = get_diversity_quality_all(sim_matrix, all_agg_scores)

        # Softmax quality weights from PRM scores
        softmax_agg_scores = softmax_with_temperature(agg_scores, config.rebase_temperature)

        # Initial REBASE-style allocation
        allocate_num = [int(x) for x in np.round(np.array(softmax_agg_scores) * config.n)]

        # Filter zero-allocation
        choose_indices = [j for j in range(len(allocate_num)) if allocate_num[j] > 0]
        active_beams = [active_beams[j] for j in choose_indices]
        agg_scores = [agg_scores[j] for j in choose_indices]

        # Recompute softmax on survivors
        softmax_agg_scores = softmax_with_temperature(agg_scores, config.rebase_temperature)

        # Extract sub-matrix, compute uniqueness γ_i
        sub_sim = extract_submatrix_np(sim_matrix, choose_indices)
        diversity_scores = get_diversity_score(sub_sim, config.balance_alpha)

        # DORA reweighting: w'_i = (w_i * γ_i) / Σ(w_j * γ_j)
        balance = [diversity_scores[j] * softmax_agg_scores[j] for j in range(len(softmax_agg_scores))]
        total = sum(balance)
        final_score = [x / total for x in balance] if total > 0 else [1.0 / len(balance)] * len(balance)

        allocate_num = [int(x) for x in np.round(np.array(final_score) * config.n)]

        # Track metrics
        choose_final = [j for j in range(len(allocate_num)) if allocate_num[j] > 0]
        if choose_final:
            fq, fd, _, _ = get_diversity_quality(sub_sim, agg_scores, choose_final)
            final_quality_list.append(fq)
            final_diversity_list.append(fd)
        all_quality_list.append(all_quality)
        all_diversity_list.append(all_diversity)

        for idx, beam in enumerate(active_beams):
            if allocate_num[idx] == 0:
                beam.pruned = True

        if len(completed_beams) >= config.n:
            break

    # Select final beams
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        repeats = (config.n // len(completed_beams)) + 1
        completed_beams = [copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]]

    return completed_beams, final_quality_list, final_diversity_list, all_quality_list, all_diversity_list


# =============================================================================
# Public Entry Point
# =============================================================================

def dora(examples, config, model, tokenizer, prm, embedding_model=None, nli_model=None):
    """DORA search entry point (Transformers backend)."""
    problems = examples["problem"]
    print(problems)

    beam_results, fq, fd, aq, ad = _dora(
        problems, config, model, tokenizer, prm, embedding_model, nli_model
    )

    grouped = defaultdict(list)
    for r in beam_results:
        grouped[r.prompt].append(r)

    results = {
        "completions": [], "pred": [], "completion_tokens": [],
        "scores": [], "quality_results": [], "diversity_results": [],
        "all_quality_results": [], "all_diversity_results": [], "similarity_mode": [],
    }

    for p in problems:
        beams = grouped[p]
        completions = [b.current_text for b in beams]
        agg = [aggregate_scores(b.all_scores, config.agg_strategy) for b in beams]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(completions[int(np.argmax(agg))] if agg else "")
        results["completion_tokens"].append([b.completion_tokens for b in beams])
        results["quality_results"].append(fq)
        results["diversity_results"].append(fd)
        results["all_quality_results"].append(aq)
        results["all_diversity_results"].append(ad)
        results["similarity_mode"].append(config.similarity_mode)

    return results