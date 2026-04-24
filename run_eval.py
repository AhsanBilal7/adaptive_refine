"""
run_eval.py
-----------
Main entry point for all five test-time compute strategies:

  greedy        — single argmax decode per question
  bon           — Best-of-N, select by verifier score
  dn_bon        — Direction-Normalised BoN (proposed method)
  greedy_pass_n — N samples, select by max sequence log-probability
  bon_majority  — N samples, select by majority vote (self-consistency)

Batch-size semantics
--------------------
--batch_size controls how many sequences are processed in ONE model.generate()
call.  The value is stored in model.batch_size and used as:

  Greedy        : questions chunked into groups of batch_size; each group
                  is one generate() call.

  BoN / DN-BoN / greedy_pass_n / bon_majority:
                  generate_samples_flat() builds a flat Q×N prompt list,
                  chunks by batch_size, and yields completed questions as
                  each chunk finishes.  With batch_size=128, N=8 → 16
                  questions decoded per generate() call.

Quick-start examples
--------------------
# All five methods, MATH-500, N=8, batch=128
python run_eval.py \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --dataset math500 \\
    --methods greedy bon dn_bon greedy_pass_n bon_majority \\
    --N 8 --batch_size 128 \\
    --max_samples 500

# DN-BoN only
python run_eval.py \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --dataset math500 \\
    --methods dn_bon \\
    --N 8 --batch_size 128 \\
    --max_samples 500

# Greedy Pass@N (no verifier needed)
python run_eval.py \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --dataset math500 \\
    --methods greedy_pass_n \\
    --N 8 --batch_size 128 \\
    --max_samples 500

# BoN Majority / self-consistency (no verifier needed)
python run_eval.py \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --dataset aime24 \\
    --methods bon_majority \\
    --N 16 --batch_size 128 \\
    --max_samples 32

# Alpha ablation for DN-BoN on AIME-24
python run_eval.py \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --dataset aime24 \\
    --alpha_ablation 0 0.25 0.5 0.75 1.0 \\
    --N 8 --batch_size 128 \\
    --max_samples 32

# Budget scaling: vary N
for N in 4 8 16 32; do
  python run_eval.py \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --dataset math500 \\
    --methods bon dn_bon greedy_pass_n bon_majority \\
    --N $N --batch_size 128 \\
    --max_samples 200 \\
    --output_dir ./results/scaling
done

# Low-VRAM with 4-bit quantisation
python run_eval.py \\
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
    --load_in_4bit \\
    --dataset gsm8k \\
    --methods greedy bon dn_bon greedy_pass_n bon_majority \\
    --N 8 --batch_size 32
"""

import argparse
import json
import logging
import math
import os
import sys
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Greedy / BoN / DN-BoN evaluation harness (batched)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--model", required=True,
                   help="HuggingFace model id or local path")
    g.add_argument(
        "--batch_size", type=int, default=1,
        help=(
            "Number of sequences in one model.generate() call.  "
            "For BoN/DN-BoN set this equal to --N so all N samples "
            "for a question are decoded in a single batched forward pass."
        ),
    )
    g.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    g.add_argument("--device", default="auto",
                   help=(
                       "'auto'   — use all visible GPUs via device_map=auto  "
                       "(recommended; works for 1 or N GPUs)  |  "
                       "'cuda:0' — pin to GPU 0  |  "
                       "'cuda:1' — pin to GPU 1  |  "
                       "'cpu'    — CPU only (debug)"
                   ))
    g.add_argument(
        "--gpus", default=None,
        help=(
            "Comma-separated GPU indices to use, e.g. '0' or '0,1,2,3'.  "
            "Sets CUDA_VISIBLE_DEVICES before loading the model.  "
            "Overrides any existing CUDA_VISIBLE_DEVICES in the environment.  "
            "Example: --gpus 0,1  uses GPU 0 and GPU 1 only."
        ),
    )
    g.add_argument("--load_in_4bit", action="store_true")
    g.add_argument("--load_in_8bit", action="store_true")
    g.add_argument("--max_new_tokens", type=int, default=1024)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top_p", type=float, default=0.95)
    g.add_argument("--trust_remote_code", action="store_true")
    g.add_argument("--hf_cache_dir", default=None,
                   help="HuggingFace cache dir for models and datasets")

    # ── Dataset ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Dataset")
    g.add_argument(
        "--dataset", required=True,
        choices=["math500", "aime24", "aime25", "amc23", "gsm8k",
                 "olympiadbench", "college_math", "minerva_math",
                 "mmlu", "bbh", "arc_challenge", "truthfulqa", "strategyqa",
                 "custom"],
    )
    g.add_argument("--split", default=None)
    g.add_argument("--max_samples", type=int, default=None)
    g.add_argument("--custom_path", default=None)

    # ── Methods ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Methods")
    g.add_argument("--methods", nargs="+",
                   default=["greedy", "bon", "dn_bon"],
                   choices=["greedy", "bon", "dn_bon", "dn_bon_updated",
                            "greedy_pass_n", "bon_majority"],
                   help=(
                       "greedy          — single argmax decode\n"
                       "bon             — Best-of-N with verifier score\n"
                       "dn_bon          — Direction-Normalised BoN (hard cluster penalty)\n"
                       "dn_bon_updated  — DORA: soft quality x uniqueness scoring\n"
                       "greedy_pass_n   — N samples, pick by max log-probability\n"
                       "bon_majority    — N samples, pick by majority vote"
                   ))
    g.add_argument(
        "--N", type=int, default=8,
        help="Number of samples for BoN / DN-BoN / greedy_pass_n / bon_majority",
    )
    g.add_argument("--alpha", type=float, default=0.5,
                   help="DN-BoN normalisation exponent (0 = plain BoN)")
    g.add_argument(
        "--alpha_ablation", nargs="+", type=float, default=None,
        help="Run DN-BoN with multiple alpha values, e.g. --alpha_ablation 0 0.5 1",
    )
    g.add_argument("--T_b", type=float, default=1.0,
                   help="DORA quality softmax temperature (dn_bon_updated only, default 1.0)")
    g.add_argument("--T_s", type=float, default=0.1,
                   help="DORA similarity softmax temperature (dn_bon_updated only, default 0.1)")

    # ── Verifier ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("Verifier")
    g.add_argument("--verifier", default="outcome",
                   choices=["outcome", "length", "prm"])
    g.add_argument("--prm_model", default="Qwen/Qwen2.5-Math-PRM-7B")
    g.add_argument("--prm_aggregation", default="min",
                   choices=["min", "mean", "last"])

    # ── Clustering ─────────────────────────────────────────────────────────────
    g = p.add_argument_group("Clustering (DN-BoN)")
    g.add_argument("--embedding_model", default="all-MiniLM-L6-v2",
                   help="sentence-transformers model id, or 'tfidf'")
    g.add_argument("--similarity_threshold", type=float, default=0.85)
    g.add_argument("--clustering_method", default="agglomerative",
                   choices=["agglomerative", "greedy"])
    g.add_argument("--cluster_on_answer", action="store_true",
                   help="Cluster only the \\boxed{} answer (faster)")

    # ── Output ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="./results")
    g.add_argument("--no_resume", action="store_true")
    g.add_argument("--log_every", type=int, default=10)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 0. GPU setup ───────────────────────────────────────────────────────────
    # Must happen before any torch/CUDA call so the device list is fixed.
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"CUDA_VISIBLE_DEVICES set to: {args.gpus}")

    import torch
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        logger.info(f"GPUs available: {n_gpus}  →  {gpu_names}")
        if n_gpus > 1:
            logger.info(
                "Multi-GPU detected — model will be sharded across all "
                f"{n_gpus} GPUs via device_map='auto' (unless --device is pinned)."
            )
    else:
        logger.info("No CUDA GPUs available — running on CPU")

    # Warn when batch_size < N for any sampling method
    sampling_methods = {"bon", "dn_bon", "greedy_pass_n", "bon_majority"}
    needs_sampling = (
        bool(set(args.methods) & sampling_methods)
        or args.alpha_ablation is not None
    )
    if needs_sampling and args.batch_size < args.N:
        logger.warning(
            f"batch_size={args.batch_size} < N={args.N}.  "
            f"N samples per question will be split across "
            f"{math.ceil(args.N / args.batch_size)} generate() calls.  "
            f"Set --batch_size {args.N} to decode all N samples in one call."
        )

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}")
    from dataset_loader import load_dataset_split
    examples = load_dataset_split(
        data_name=args.dataset, split=args.split,
        max_samples=args.max_samples, custom_path=args.custom_path,
        cache_dir=args.hf_cache_dir,
    )
    data_name = examples[0]["_data_name"] if examples else args.dataset
    logger.info(f"Loaded {len(examples)} examples.")

    # ── 2. Load model ──────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model}  batch_size={args.batch_size}")
    from model_wrapper import HFModelWrapper
    model = HFModelWrapper(
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.hf_cache_dir,
    )

    # ── 3. Build verifier ──────────────────────────────────────────────────────
    logger.info(f"Building verifier: {args.verifier}")
    from verifier import build_verifier
    verifier = build_verifier(
        verifier_type=args.verifier, data_name=data_name,
        prm_model=args.prm_model, prm_aggregation=args.prm_aggregation,
        device=args.device, cache_dir=args.hf_cache_dir,
        batch_size=args.batch_size,
    )

    # ── 4. Build cluster manager (only if DN-BoN or DN-BoN Updated is needed) ──
    cluster_manager = None
    needs_clustering = (
        "dn_bon" in args.methods
        or "dn_bon_updated" in args.methods
        or args.alpha_ablation is not None
    )
    if needs_clustering:
        logger.info(f"Building ClusterManager  embedding={args.embedding_model}")
        from clustering import ClusterManager
        cluster_manager = ClusterManager(
            embedding_model=args.embedding_model,
            similarity_threshold=args.similarity_threshold,
            clustering_method=args.clustering_method,
            use_answer_only=args.cluster_on_answer,
            batch_size=args.batch_size,
        )
        print("🧩Cluster on Answer: ", args.cluster_on_answer)
        print("🧩Alpha Value: ", args.alpha)
    # ── 5. Build run configs ───────────────────────────────────────────────────
    run_configs = []   # list of (label, method_object)

    if args.alpha_ablation is not None:
        from inference_methods import DNBestOfN
        for a in args.alpha_ablation:
            m = DNBestOfN(
                model=model, verifier=verifier,
                cluster_manager=cluster_manager,
                N=args.N, alpha=a, data_name=data_name,
                temperature=args.temperature, top_p=args.top_p,
            )
            run_configs.append((f"dn_bon_alpha{a}", m))
    else:
        from inference_methods import build_inference_method
        for method_name in args.methods:
            m = build_inference_method(
                method_name=method_name,
                model=model, verifier=verifier,
                cluster_manager=cluster_manager,
                N=args.N, alpha=args.alpha,
                T_b=args.T_b, T_s=args.T_s,
                data_name=data_name,
                temperature=args.temperature, top_p=args.top_p,
            )
            run_configs.append((method_name, m))

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    from evaluator import evaluate_batch, save_summary, compare_methods, redundancy_stats
    all_summaries = {}
    # tag = f"{args.dataset}_N{args.N}_bs{args.batch_size}"
    tag = f"{args.dataset}_N{args.N}"

    for run_label, method in run_configs:
        logger.info(f"\n{'='*64}")
        logger.info(f" Running : {run_label}")
        logger.info(
            f" Dataset : {args.dataset}  N={args.N}  batch_size={args.batch_size}"
        )
        logger.info(f"{'='*64}")

        jsonl_path   = os.path.join(args.output_dir, f"{run_label}_{tag}.jsonl")
        summary_path = os.path.join(args.output_dir, f"{run_label}_{tag}_summary.json")

        # summary_path is passed into evaluate_batch so the summary JSON
        # is atomically rewritten after EVERY question — not only at the end.
        # If the run crashes, restart and it will resume from the JSONL.
        summary = evaluate_batch(
            method=method,
            examples=examples,
            max_new_tokens=args.max_new_tokens,
            save_path=jsonl_path,
            summary_path=summary_path,   # written on-the-go, every question
            resume=not args.no_resume,
            log_every=args.log_every,
            summary_every=1,             # rewrite after every single question
        )

        # Log cluster/vote stats for methods that group traces
        if "dn_bon" in run_label or "bon_majority" in run_label:
            r_stats = redundancy_stats(summary["results"])
            summary["redundancy_stats"] = r_stats
            logger.info(f"Redundancy/vote stats: {r_stats}")
            save_summary(summary, summary_path)

        all_summaries[run_label] = summary

    # ── 7. Print comparison table ──────────────────────────────────────────────
    comparison = compare_methods(all_summaries)
    logger.info(comparison)

    cmp_path = os.path.join(args.output_dir, f"comparison_{tag}.txt")
    with open(cmp_path, "w") as f:
        f.write(comparison)
    logger.info(f"Comparison table saved to {cmp_path}")


if __name__ == "__main__":
    main()