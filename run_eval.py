"""
run_eval.py
-----------
Main entry point for Greedy / BoN / DN-BoN evaluation.

Batch-size semantics
--------------------
--batch_size  controls how many sequences are processed in ONE model.generate()
              call.  The value is stored in model.batch_size and used as:

  Greedy : questions are chunked into groups of --batch_size; each group
           is one generate() call with a left-padded batch.

  BoN / DN-BoN : set --batch_size = N (recommended).  Each question's N
           sample prompts are stacked into one batch of size N and decoded
           in a single generate() call.  If --batch_size < N the N prompts
           are split into ceil(N/batch_size) calls; if > N only N are used.

In practice you almost always want --batch_size == --N for BoN/DN-BoN.

Quick-start examples
--------------------
# All three methods, MATH-500, N=8, batch=8
python run_eval.py \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --dataset math500 \\
    --methods greedy bon dn_bon \\
    --N 8 --batch_size 8 \\
    --max_samples 500
python run_eval.py  --model Qwen/Qwen2.5-Math-7B-Instruct  --dataset math500  --methods greedy bon dn_bon  --N 8 --batch_size 8  --max_samples 500
python run_eval.py  --model Qwen/Qwen2.5-Math-7B-Instruct  --dataset aime24  --methods greedy bon dn_bon  --N 8 --batch_size 8  --max_samples 32

# Alpha ablation for DN-BoN on AIME-24
python run_eval.py \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --dataset aime24 \\
    --alpha_ablation 0 0.5 1 \\
    --N 16 --batch_size 16 \\
    --max_samples 30

# Budget scaling: vary N
for N in 4 8 16 32; do
  python run_eval.py \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --dataset math500 \\
    --methods bon dn_bon \\
    --N $N --batch_size $N \\
    --max_samples 100 \\
    --output_dir ./results/scaling
done

# Low-VRAM with 4-bit quantisation (use smaller batch)
python run_eval.py \\
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
    --load_in_4bit \\
    --dataset gsm8k \\
    --methods greedy bon dn_bon \\
    --N 8 --batch_size 8
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
                   help="'auto', 'cuda', 'cpu', or 'cuda:0'")
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
                 "gaokao2023en", "mmlu_stem", "custom"],
    )
    g.add_argument("--split", default=None)
    g.add_argument("--max_samples", type=int, default=None)
    g.add_argument("--custom_path", default=None)

    # ── Methods ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Methods")
    g.add_argument("--methods", nargs="+",
                   default=["greedy", "bon", "dn_bon"],
                   choices=["greedy", "bon", "dn_bon"])
    g.add_argument(
        "--N", type=int, default=8,
        help="Number of samples for BoN / DN-BoN (recommend --batch_size == --N)",
    )
    g.add_argument("--alpha", type=float, default=0.5,
                   help="DN-BoN normalisation exponent (0 = plain BoN)")
    g.add_argument(
        "--alpha_ablation", nargs="+", type=float, default=None,
        help="Run DN-BoN with multiple alpha values, e.g. --alpha_ablation 0 0.5 1",
    )

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

    # Warn only when batch_size is SMALLER than N for BoN/DN-BoN,
    # which means N samples will be split across multiple generate() calls.
    # batch_size > N is fine (unused capacity); batch_size == N is ideal.
    needs_bon = ("bon" in args.methods or "dn_bon" in args.methods
                 or args.alpha_ablation is not None)
    if needs_bon and args.batch_size < args.N:
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
    )

    # ── 4. Build cluster manager (only if DN-BoN is needed) ───────────────────
    cluster_manager = None
    needs_clustering = (
        "dn_bon" in args.methods or args.alpha_ablation is not None
    )
    if needs_clustering:
        logger.info(f"Building ClusterManager  embedding={args.embedding_model}")
        from clustering import ClusterManager
        cluster_manager = ClusterManager(
            embedding_model=args.embedding_model,
            similarity_threshold=args.similarity_threshold,
            clustering_method=args.clustering_method,
            use_answer_only=args.cluster_on_answer,
        )

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
                data_name=data_name,
                temperature=args.temperature, top_p=args.top_p,
            )
            run_configs.append((method_name, m))

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    from evaluator import evaluate_batch, save_summary, compare_methods, redundancy_stats
    all_summaries = {}
    tag = f"{args.dataset}_N{args.N}_bs{args.batch_size}"

    for run_label, method in run_configs:
        logger.info(f"\n{'='*64}")
        logger.info(f" Running : {run_label}")
        logger.info(
            f" Dataset : {args.dataset}  N={args.N}  batch_size={args.batch_size}"
        )
        logger.info(f"{'='*64}")

        jsonl_path   = os.path.join(args.output_dir, f"{run_label}_{tag}.jsonl")
        summary_path = os.path.join(args.output_dir, f"{run_label}_{tag}_summary.json")

        summary = evaluate_batch(
            method=method,
            examples=examples,
            max_new_tokens=args.max_new_tokens,
            save_path=jsonl_path,
            resume=not args.no_resume,
            log_every=args.log_every,
        )

        if "dn_bon" in run_label:
            r_stats = redundancy_stats(summary["results"])
            summary["redundancy_stats"] = r_stats
            logger.info(f"Redundancy stats: {r_stats}")

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