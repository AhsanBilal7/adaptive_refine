#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Stripped-down runner for DORA (cosine / NLI / hybrid similarity modes).
# Only supports approach="dora". Uses the HuggingFace Transformers backend.

import logging
import os
import random
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.models.nli_similarity import load_nli_model
from sal.search import dora
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_policy_model(config):
    """Load the policy LM and tokenizer via HuggingFace Transformers."""
    logger.info(f"Loading policy model: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Policy model loaded.")
    return model, tokenizer


def load_embedding_model(config):
    """Load BGE-M3 embedding model (needed for cosine / hybrid modes)."""
    if not getattr(config, "embedding_path", None):
        return None
    logger.info(f"Loading embedding model: {config.embedding_path}")
    emb_model = BGEM3FlagModel(config.embedding_path, use_fp16=False)
    target = emb_model.model if hasattr(emb_model, "model") else emb_model
    target.eval()
    if hasattr(emb_model, "tokenizer"):
        emb_model.tokenizer.parallelism = False
    return emb_model


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _make_output_filename(config):
    sim   = getattr(config, "similarity_mode", "cosine")
    alpha = config.balance_alpha
    n     = config.n
    seed  = config.seed
    temp  = config.temperature
    ts    = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"samples_dora_{sim}_n_{n}_alpha_{alpha}_seed_{seed}_temp_{temp}_{ts}.jsonl"


def _make_log_filename(config):
    sim   = getattr(config, "similarity_mode", "cosine")
    alpha = config.balance_alpha
    n     = config.n
    seed  = config.seed
    temp  = config.temperature
    return f"processed_dora_{sim}_n_{n}_alpha_{alpha}_seed_{seed}_temp_{temp}.log"


def _load_processed(log_path):
    if not os.path.exists(log_path):
        return set()
    with open(log_path) as f:
        return {int(l.strip()) for l in f if l.strip().isdigit()}


def _save_random_states():
    return (
        random.getstate(),
        torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_random_states(states, seed):
    torch.set_rng_state(states[1])
    if states[2] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states[2])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    # Hard-enforce DORA only
    if config.approach != "dora":
        raise ValueError(
            f"This script only supports approach='dora', got '{config.approach}'."
        )

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Validate similarity_mode ──────────────────────────────────────────────
    valid_modes = ("cosine", "nli", "hybrid")
    sim_mode = getattr(config, "similarity_mode", "cosine")
    if sim_mode not in valid_modes:
        raise ValueError(
            f"similarity_mode='{sim_mode}' is not supported. "
            f"Choose one of: {valid_modes}"
        )
    logger.info(f"Similarity mode: {sim_mode}")

    # ── Load models ───────────────────────────────────────────────────────────
    model, tokenizer   = load_policy_model(config)
    prm                = load_prm(config)
    embedding_model    = load_embedding_model(config)   # None if no embedding_path
    nli_model          = load_nli_model(config)         # None unless mode in (nli, hybrid)

    # Sanity checks
    if sim_mode == "cosine" and embedding_model is None:
        raise ValueError(
            "similarity_mode='cosine' requires embedding_path to be set in config."
        )
    if sim_mode in ("nli", "hybrid") and nli_model is None:
        raise ValueError(
            f"similarity_mode='{sim_mode}' requires nli_model_path to be set in config."
        )
    if sim_mode == "hybrid" and embedding_model is None:
        raise ValueError(
            "similarity_mode='hybrid' requires embedding_path (for cosine stage)."
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset  = get_dataset(config)
    all_data = []
    for idx, sample in enumerate(dataset):
        sample["idx"] = idx
        all_data.append(deepcopy(sample))

    # Shuffle for load-balanced checkpointing
    original_states = _save_random_states()
    rng_seed_time   = int(time.time() * 1e7)
    random.seed(rng_seed_time)
    random_indices  = list(range(len(all_data)))
    random.shuffle(random_indices)
    random.seed(config.seed)

    # ── Output / checkpoint files ─────────────────────────────────────────────
    output_file       = os.path.join(config.output_dir, _make_output_filename(config))
    processed_log     = os.path.join(config.output_dir, _make_log_filename(config))
    config.output_file = output_file

    logger.info(f"Output  : {output_file}")
    logger.info(f"Log file: {processed_log}")

    # ── Main processing loop ──────────────────────────────────────────────────
    queries        = []
    ids_to_process = []
    all_result     = []
    all_times      = []

    for idx in tqdm(random_indices, desc="Processing problems"):

        # Skip already-processed samples (crash-resume support)
        if idx in _load_processed(processed_log):
            continue

        queries.append(deepcopy(all_data[idx]))
        ids_to_process.append(idx)

        batch_ready = (
            len(queries) >= config.search_batch_size
            or idx == random_indices[-1]
        )
        if not batch_ready:
            continue

        # Build input batch dict
        input_batch = {k: [] for k in queries[0].keys()}
        for q in queries:
            for k, v in q.items():
                input_batch[k].append(deepcopy(v))

        # Restore deterministic RNG state for reproducibility
        _restore_random_states(original_states, config.seed)

        # ── Run DORA ─────────────────────────────────────────────────────────
        t0 = time.time()
        processed_sample = dora(
            input_batch,
            config         = config,
            model          = model,
            tokenizer      = tokenizer,
            prm            = prm,
            embedding_model= embedding_model,
            nli_model      = nli_model,
        )
        all_times.append(time.time() - t0)

        # ── Merge results back into queries and write checkpoint ──────────────
        for i, (query, qid) in enumerate(zip(queries, ids_to_process)):
            cur_result = {k: processed_sample[k][i] for k in processed_sample}
            query |= cur_result
            all_result.append(deepcopy(query))
            with open(processed_log, "a") as f:
                f.write(f"{qid}\n")

        queries.clear()
        ids_to_process.clear()

        # Incremental save
        save_dataset(Dataset.from_list(all_result), config)

    if all_times:
        logger.info(f"Avg time per batch: {sum(all_times)/len(all_times):.2f}s")
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()