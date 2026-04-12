#!/usr/bin/env python
"""
Main entry point for test-time compute scaling experiments.
Uses HuggingFace Transformers (no vLLM dependency).
"""

import logging
import os
import sys
import random
import time
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path so sal is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.models.nli_similarity import load_nli_model
from sal.search import dora
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None
    logger.warning("FlagEmbedding not installed. Cosine/hybrid similarity won't work.")


APPROACHES = {
    "dora": dora,
}


def save_random_states():
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return (py_state, torch_state, cuda_states)


def load_policy_model(config):
    """Load the policy LM using HuggingFace Transformers."""
    logger.info(f"Loading policy model: {config.model_path}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    # Quantization support
    if config.load_in_4bit or config.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            model_kwargs["quantization_config"] = quant_config
        except ImportError:
            logger.warning("bitsandbytes not installed, ignoring quantization config.")

    # Device map
    if config.device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = config.device

    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Policy model loaded on {model.device}")
    return model, tokenizer


def load_embedding_model(config):
    """Load BGE-M3 embedding model."""
    if config.embedding_path is None or BGEM3FlagModel is None:
        return None
    model = BGEM3FlagModel(config.embedding_path, use_fp16=False)
    if hasattr(model, "model"):
        model.model.eval()
    else:
        model.eval()
    if hasattr(model, "tokenizer"):
        model.tokenizer.parallelism = False
    return model


def load_nli_model_from_config(config):
    """Load NLI model if needed."""
    effective_mode = config.similarity_mode
    if config.ablation_similarity is not None:
        effective_mode = config.ablation_similarity
    if effective_mode not in ("nli", "hybrid"):
        return None
    if config.nli_model_path is None:
        logger.warning(f"similarity_mode='{effective_mode}' but nli_model_path not set.")
        return None
    return load_nli_model(config)


def generate_random_indices(dataset):
    current_time = time.time() * 10000000
    random.seed(int(current_time))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices


def build_output_filename(config):
    sim_mode = config.similarity_mode
    if config.ablation_similarity is not None:
        sim_mode = f"ablation_{config.ablation_similarity}"

    if config.approach == "dora":
        tag = sim_mode
        if sim_mode == "nli" and config.nli_model_path:
            nli_name = config.nli_model_path.split("/")[-1]
            tag = f"nli_{nli_name}_lambda{config.nli_lambda_penalty}"
        elif sim_mode == "hybrid" and config.nli_model_path:
            nli_name = config.nli_model_path.split("/")[-1]
            tag = f"hybrid_{nli_name}_topk{config.nli_top_k}_lambda{config.nli_lambda_penalty}"

        base = f"samples_{tag}_n_{config.n}_alpha_{config.balance_alpha}_seed_{config.seed}_temp_{config.temperature}"
    else:
        base = f"samples_n_{config.n}_seed_{config.seed}_temp_{config.temperature}"

    return base + f"_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jsonl"


def build_log_filename(config):
    sim_mode = config.similarity_mode
    if config.ablation_similarity is not None:
        sim_mode = f"ablation_{config.ablation_similarity}"
    base = f"processed_{sim_mode}_n_{config.n}_alpha_{config.balance_alpha}_seed_{config.seed}_temp_{config.temperature}"
    return base + ".log"


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    print(config.n)
    os.makedirs(config.output_dir, exist_ok=True)

    approach_fn = APPROACHES[config.approach]

    print(f"Approach: {config.approach}")
    print(f"Similarity mode: {config.similarity_mode}")
    print(f"Model: {config.model_path}")
    print(f"Device: {config.device}, dtype: {config.dtype}")

    # Load policy model (Transformers)
    model, tokenizer = load_policy_model(config)

    # Load PRM, embedding model, NLI model
    prm = load_prm(config)
    embedding_model = load_embedding_model(config)
    nli_model = load_nli_model_from_config(config)

    # Load dataset
    dataset = get_dataset(config)
    all_data = []
    for idx, sample in enumerate(dataset):
        sample["idx"] = idx
        all_data.append(deepcopy(sample))

    original_states = save_random_states()
    random_indices = generate_random_indices(all_data)
    random.seed(config.seed)

    output_file = os.path.join(config.output_dir, build_output_filename(config))
    config.output_file = output_file
    processed_log_file = os.path.join(config.output_dir, build_log_filename(config))

    queries = []
    ids_to_process = []
    all_sample_times = []
    all_result = []

    for idx in tqdm(random_indices, desc="Processing samples"):
        batch = all_data[idx]

        # Check if already processed
        processed_items = set()
        if os.path.exists(processed_log_file):
            with open(processed_log_file, "r") as f:
                processed_items = set(
                    int(line.strip()) for line in f if line.strip().isdigit()
                )
        if idx in processed_items:
            continue

        queries.append(deepcopy(batch))
        ids_to_process.append(idx)

        if len(queries) >= config.search_batch_size or idx == random_indices[-1]:
            # Build input batch
            input_batch = {key: [] for key in queries[0].keys()}
            for q in queries:
                for key in q:
                    input_batch[key].append(deepcopy(q[key]))

            # Restore deterministic state
            torch.set_rng_state(original_states[1])
            if original_states[2] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(original_states[2])
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

            start_time = time.time()

            processed_sample = approach_fn(
                input_batch,
                config=config,
                model=model,
                tokenizer=tokenizer,
                prm=prm,
                embedding_model=embedding_model,
                nli_model=nli_model,
            )

            elapsed = time.time() - start_time
            all_sample_times.append(elapsed)

            for i in range(len(queries)):
                cur_result = {key: processed_sample[key][i] for key in processed_sample}
                queries[i] |= cur_result
                all_result.append(deepcopy(queries[i]))
                with open(processed_log_file, "a") as f:
                    f.write(f"{ids_to_process[i]}\n")

            queries.clear()
            ids_to_process.clear()

            cur_dataset = Dataset.from_list(all_result)
            save_dataset(cur_dataset, config)

    if all_sample_times:
        print(f"avg sample time = {sum(all_sample_times) / len(all_sample_times):.2f}s")
    logger.info("Done!")


if __name__ == "__main__":
    main()