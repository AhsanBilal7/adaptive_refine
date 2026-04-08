"""
main.py – AdaptSci entry point.

AdaptSci additions vs. original SR-Scientist main.py
=====================================================
• --adaptive-compute / --no-adaptive-compute   master switch
• --tau1, --tau2                               κ thresholds for restart count
• --epsilon-target, --beta-star                early-stopping hyperparameters
• --coupling-threshold, --n-eff-ratio-threshold  degeneracy detection
• --exploit-frac                               70/30 buffer split
• --frontier-alpha                             α in s_fr
• --frontier-window                            momentum window w
• Ablation flags (all from Section 6 "Key ablations"):
    --ablation-disable-early-stopping
    --ablation-fix-n-restarts
    --ablation-alpha-override
    --ablation-disable-c-diversity
    --ablation-random-frontier
    --ablation-hessian-at-first-run
• --no-hessian-in-prompt                       suppress κ/n_eff guidance in prompt
"""

import argparse
import asyncio
import json
import os

from infer.inference import run_inference_batch, AblationConfig
from infer.analysis import analyze_results


def build_ablation_config(args) -> dict:
    """Convert argparse namespace to ablation_config dict."""
    config = {}
    if args.ablation_disable_early_stopping:
        config["disable_early_stopping"] = True
    if args.ablation_fix_n_restarts:
        config["fix_n_restarts"] = True
    if args.ablation_alpha_override is not None:
        config["alpha_override"] = args.ablation_alpha_override
    if args.ablation_disable_c_diversity:
        config["disable_c_diversity"] = True
    if args.ablation_random_frontier:
        config["random_frontier"] = True
    if args.ablation_hessian_at_first_run:
        config["hessian_at_first_run"] = True
    # Hyperparameters
    config["tau1"] = args.tau1
    config["tau2"] = args.tau2
    config["epsilon_target"] = args.epsilon_target
    config["beta_star"] = args.beta_star
    config["coupling_threshold"] = args.coupling_threshold
    config["n_eff_ratio_threshold"] = args.n_eff_ratio_threshold
    config["exploit_frac"] = args.exploit_frac
    config["frontier_alpha"] = args.frontier_alpha
    config["frontier_window"] = args.frontier_window
    config["include_hessian_in_prompt"] = not args.no_hessian_in_prompt
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AdaptSci: Adaptive Test-Time Compute for Agentic Equation Discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -----------------------------------------------------------------------
    # SR-Scientist original arguments (unchanged)
    # -----------------------------------------------------------------------
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name identifier for the vLLM server.")
    parser.add_argument("--model-url", type=str, required=True,
                        help="URL of the vLLM OpenAI-compatible server.")
    parser.add_argument("--sandbox-urls", nargs="+", required=True,
                        help="One or more sandbox execution environment URLs.")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key for the model endpoint.")
    parser.add_argument("--parquet-file-path", type=str, required=True,
                        help="Path to input Parquet data file.")
    parser.add_argument("--mape-threshold", type=float, required=True,
                        help="MAPE threshold for success (e.g. 0.001 = 0.1%).")
    parser.add_argument("--output-json-path", type=str, required=True,
                        help="Path for output JSON results file.")
    parser.add_argument("--max-assistant-turns", type=int, default=100,
                        help="Max tool-calling turns per main turn.")
    parser.add_argument("--source", nargs="+", default=None,
                        help="Dataset source filter (e.g. lsr_synth/phys_osc).")
    parser.add_argument("--num-turns", type=int, required=True,
                        help="Number of outer turns (iterations).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Seed set size K for the experience buffer.")

    # -----------------------------------------------------------------------
    # AdaptSci master switch
    # -----------------------------------------------------------------------
    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument("--adaptive-compute", dest="adaptive_compute",
                                action="store_true", default=True,
                                help="Enable AdaptSci adaptive compute (default: ON).")
    adaptive_group.add_argument("--no-adaptive-compute", dest="adaptive_compute",
                                action="store_false",
                                help="Disable AdaptSci; run in SR-Scientist mode.")

    # -----------------------------------------------------------------------
    # AdaptSci hyperparameters (calibrated on validation split)
    # -----------------------------------------------------------------------
    parser.add_argument("--tau1", type=float, default=100.0,
                        help="κ threshold below which 1 BFGS restart suffices.")
    parser.add_argument("--tau2", type=float, default=1000.0,
                        help="κ threshold above which 5 BFGS restarts are used.")
    parser.add_argument("--epsilon-target", type=float, default=1e-4,
                        help="NMSE convergence target ε_target for early stopping.")
    parser.add_argument("--beta-star", type=float, default=5000.0,
                        help="Efficiency ratio threshold β* for early stopping.")
    parser.add_argument("--coupling-threshold", type=float, default=0.95,
                        help="|C_ij| threshold for degeneracy detection (γ in paper).")
    parser.add_argument("--n-eff-ratio-threshold", type=float, default=0.5,
                        help="n_eff/p ratio below which skeleton is degenerate.")
    parser.add_argument("--exploit-frac", type=float, default=0.70,
                        help="Fraction of K allocated to exploitation pool (default 0.70 = 70/30).")
    parser.add_argument("--frontier-alpha", type=float, default=0.5,
                        help="α weighting improvement-rate vs C-diversity in s_fr.")
    parser.add_argument("--frontier-window", type=int, default=3,
                        help="Window w for NMSE momentum computation.")

    # -----------------------------------------------------------------------
    # Ablation study flags  (Section 6, Table 3 in AdaptSci paper)
    # -----------------------------------------------------------------------
    ablation_group = parser.add_argument_group(
        "Ablation Study Flags",
        "Disable individual AdaptSci mechanisms to reproduce ablation rows."
    )
    ablation_group.add_argument("--ablation-disable-early-stopping", action="store_true",
                                default=False, help="Ablation (i): remove early stopping.")
    ablation_group.add_argument("--ablation-fix-n-restarts", action="store_true",
                                default=False, help="Ablation (ii): fix n_restarts=1.")
    ablation_group.add_argument("--ablation-alpha-override", type=float, default=None,
                                metavar="ALPHA",
                                help="Ablation (iii): fix α ∈ {0,0.25,0.5,0.75,1}.")
    ablation_group.add_argument("--ablation-disable-c-diversity", action="store_true",
                                default=False,
                                help="Ablation (iv): C-diversity off (equiv. α=1).")
    ablation_group.add_argument("--ablation-random-frontier", action="store_true",
                                default=False,
                                help="Ablation (iv): random frontier sampling.")
    ablation_group.add_argument("--ablation-hessian-at-first-run", action="store_true",
                                default=False,
                                help="Ablation (v): compute Hessian at first BFGS run, not best-of-n.")
    ablation_group.add_argument("--no-hessian-in-prompt", action="store_true",
                                default=False,
                                help="Suppress κ/n_eff guidance from the LLM prompt.")

    args = parser.parse_args()

    print("AdaptSci – Starting inference with configuration:")
    print(json.dumps(vars(args), indent=2))

    ablation_config = build_ablation_config(args)

    asyncio.run(run_inference_batch(
        model_name=args.model_name,
        model_url=args.model_url,
        sandbox_urls=args.sandbox_urls,
        parquet_file_path=args.parquet_file_path,
        mape_threshold=args.mape_threshold,
        num_turns=args.num_turns,
        top_k=args.top_k,
        output_json_path=args.output_json_path,
        max_assistant_turns=args.max_assistant_turns,
        api_key=args.api_key,
        source=args.source,
        adaptive_compute=args.adaptive_compute,
        ablation_config=ablation_config,
    ))

    analyze_results(args.output_json_path)
