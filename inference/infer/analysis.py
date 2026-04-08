"""
analysis.py – Results analysis for AdaptSci / SR-Scientist.

AdaptSci additions vs. original SR-Scientist analysis.py
=========================================================
• Hessian statistics section: κ distribution, n_eff distribution, degeneracy rate.
• Buffer statistics section: average buffer size, exploitation vs. frontier usage.
• Early-stopping statistics: how many examples triggered early stop and at which call.
• n_restarts distribution across problems.
• All original sections are preserved unchanged.
"""

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def analyze_results(json_file_path: str):
    """
    Analyzes model response results from a JSON file.
    Provides all original SR-Scientist statistics plus AdaptSci-specific sections.
    """
    # --- Data Loading ---
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{json_file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'.")
        return

    # --- Initialization ---
    total_responses = 0
    total_tool_calls = 0
    finish_reasons = []
    present_indices = set()
    dataset_stats = defaultdict(lambda: {"tool_calls": 0, "responses": 0})
    output_distribution = {
        "Has stdout, No stderr": 0,
        "Has stdout, Has stderr": 0,
        "No stdout,  Has stderr": 0,
        "No stdout,  No stderr": 0,
    }
    analysis_code_count, evaluation_code_count = 0, 0
    analysis_with_stderr, evaluation_with_stderr = 0, 0
    execution_status_distribution = Counter()

    # Turn-based
    tool_calls_per_turn: Counter = Counter()
    responses_per_turn: Counter = Counter()
    errors_per_turn: Counter = Counter()
    turns_per_problem: dict = {}

    # AdaptSci-specific
    n_restarts_distribution: Counter = Counter()
    kappa_values: list = []
    n_eff_values: list = []
    degeneracy_flags: list = []
    buffer_sizes: list = []
    early_stop_calls_list: list = []

    print(f"Analyzing {len(results_data)} examples from '{json_file_path}'...")

    for example in results_data:
        if "index" in example:
            present_indices.add(example["index"])

        dataset_name = example.get("source", "Unknown")
        max_turn_for_example = 0

        # AdaptSci buffer summary
        buf_summary = example.get("adaptsci_buffer_summary")
        if buf_summary:
            buffer_sizes.append(buf_summary.get("buffer_size", 0))
            kappa = buf_summary.get("best_kappa")
            neff  = buf_summary.get("best_n_eff")
            if kappa is not None and np.isfinite(kappa):
                kappa_values.append(kappa)
            if neff is not None:
                n_eff_values.append(neff)

        # Early stopping
        es_calls = example.get("early_stop_calls")
        if es_calls is not None:
            early_stop_calls_list.append(es_calls)

        # Submitted results Hessian
        sub = example.get("submitted_results", {}) or {}
        hdiag = sub.get("hessian_diag")
        if isinstance(hdiag, dict):
            if hdiag.get("valid"):
                degeneracy_flags.append(hdiag.get("kappa", float("inf")) > 1e3)

        if "responses" in example and isinstance(example["responses"], list):
            dataset_stats[dataset_name]["responses"] += len(example["responses"])

            for response_branch in example["responses"]:
                total_responses += 1
                turn_num = response_branch.get("turn")
                if turn_num:
                    responses_per_turn[turn_num] += 1
                    max_turn_for_example = max(max_turn_for_example, turn_num)

                if "finish_reason" in response_branch:
                    finish_reasons.append(response_branch["finish_reason"])

                # AdaptSci: n_restarts_used
                nr = response_branch.get("n_restarts_used", 1)
                n_restarts_distribution[nr] += 1

                if "tool_executions" in response_branch and isinstance(response_branch["tool_executions"], list):
                    num_calls = len(response_branch["tool_executions"])
                    total_tool_calls += num_calls
                    dataset_stats[dataset_name]["tool_calls"] += num_calls
                    if turn_num:
                        tool_calls_per_turn[turn_num] += num_calls

                    for tc_meta in response_branch["tool_executions"]:
                        has_stderr = bool(tc_meta.get("stderr"))
                        status = tc_meta.get("status", "Status Not Found")
                        execution_status_distribution[status] += 1

                        if has_stderr and turn_num:
                            errors_per_turn[turn_num] += 1

                        if tc_meta.get("tool_name") == "equation_evaluator":
                            evaluation_code_count += 1
                            if has_stderr:
                                evaluation_with_stderr += 1
                        else:
                            analysis_code_count += 1
                            if has_stderr:
                                analysis_with_stderr += 1

                        has_stdout = bool(tc_meta.get("stdout"))
                        if has_stdout and not has_stderr:
                            output_distribution["Has stdout, No stderr"] += 1
                        elif has_stdout and has_stderr:
                            output_distribution["Has stdout, Has stderr"] += 1
                        elif not has_stdout and has_stderr:
                            output_distribution["No stdout,  Has stderr"] += 1
                        else:
                            output_distribution["No stdout,  No stderr"] += 1

        if max_turn_for_example > 0:
            turns_per_problem[example.get("index")] = max_turn_for_example

    # ======================================================================
    # Display
    # ======================================================================
    print("=" * 70)
    print("              Analysis of AdaptSci Model Responses")
    print("=" * 70)

    # --- Overall tool call statistics ---
    print("\n## 🛠️  Overall Tool Call Statistics\n")
    avg_calls = total_tool_calls / total_responses if total_responses > 0 else 0
    print(f"Total responses analyzed: {total_responses}")
    print(f"Total tool calls made:    {total_tool_calls}")
    print(f"Average tool calls/response: {avg_calls:.2f}")
    print("-" * 70)

    # --- Per-dataset ---
    print("\n## 📊 Average Tool Calls per Dataset\n")
    dataset_order = [
        "lsr_transform", "lsr_synth/matsci", "lsr_synth/chem_react",
        "lsr_synth/bio_pop_growth", "lsr_synth/phys_osc",
    ]
    for ds in dataset_order:
        stats = dataset_stats.get(ds)
        if stats and stats["responses"] > 0:
            avg = stats["tool_calls"] / stats["responses"]
            print(f"  - {ds}: {avg:.2f} (Total: {stats['tool_calls']})")
        else:
            print(f"  - {ds}: No data.")
    print("-" * 70)

    # --- Finish reason distribution ---
    print("\n## 🛑 Finish Reason Distribution\n")
    frd = Counter(finish_reasons)
    for reason, count in frd.most_common():
        pct = (count / total_responses * 100) if total_responses else 0
        print(f"  - {reason}: {count} ({pct:.2f}%)")
    print("-" * 70)

    # --- Output distribution ---
    print("\n## 📊 Tool Call Output & Error Distribution\n")
    if total_tool_calls > 0:
        dist_data = {
            "Case": list(output_distribution.keys()),
            "Count": list(output_distribution.values()),
            "Percentage": [f"{v / total_tool_calls * 100:.2f}%" for v in output_distribution.values()],
        }
        print(pd.DataFrame(dist_data).to_string(index=False))
    print("-" * 70)

    # --- Turn-based analysis ---
    print("\n## 🔄 Turn-Based Analysis\n")
    if responses_per_turn:
        max_turn = max(responses_per_turn.keys())
        turn_data = {
            "Turn":             list(range(1, max_turn + 1)),
            "Total Responses":  [responses_per_turn.get(t, 0) for t in range(1, max_turn + 1)],
            "Total Tool Calls": [tool_calls_per_turn.get(t, 0) for t in range(1, max_turn + 1)],
            "Total Errors":     [errors_per_turn.get(t, 0) for t in range(1, max_turn + 1)],
        }
        tdf = pd.DataFrame(turn_data)
        tdf["Avg. Calls/Response"] = (
            tdf["Total Tool Calls"] / tdf["Total Responses"]
        ).fillna(0).map("{:.2f}".format)
        print("--- Per-Turn Statistics ---")
        print(tdf.to_string(index=False))

        # Success by turns
        success_by_turns = defaultdict(lambda: {"success": 0, "total": 0})
        for res in results_data:
            nt = turns_per_problem.get(res.get("index"))
            if nt:
                success_by_turns[nt]["total"] += 1
                if res.get("submitted_results", {}).get("success") == 1.0:
                    success_by_turns[nt]["success"] += 1

        if success_by_turns:
            sdf = pd.DataFrame({
                "Turns Taken":    sorted(success_by_turns.keys()),
                "Num Problems":   [success_by_turns[t]["total"]   for t in sorted(success_by_turns.keys())],
                "Num Successful": [success_by_turns[t]["success"] for t in sorted(success_by_turns.keys())],
            })
            sdf["Success Rate"] = (
                sdf["Num Successful"] / sdf["Num Problems"] * 100
            ).map("{:.2f}%".format)
            print("\n--- Success Rate by Number of Turns ---")
            print(sdf.to_string(index=False))
    print("-" * 70)

    # --- Code execution status ---
    print("\n## 📈 Code Execution Status\n")
    if total_tool_calls > 0:
        rows = [
            ("Analysis - Correct",   analysis_code_count - analysis_with_stderr),
            ("Analysis - Error",     analysis_with_stderr),
            ("Evaluation - Correct", evaluation_code_count - evaluation_with_stderr),
            ("Evaluation - Error",   evaluation_with_stderr),
        ]
        sdf2 = pd.DataFrame({
            "Category": [r[0] for r in rows],
            "Count":    [r[1] for r in rows],
            "% of Total": [f"{r[1] / total_tool_calls * 100:.2f}%" for r in rows],
        })
        print(sdf2.to_string(index=False))
    print("-" * 70)

    # ======================================================================
    # ★  AdaptSci-Specific Sections
    # ======================================================================

    print("\n" + "=" * 70)
    print("              AdaptSci Adaptive Compute Statistics")
    print("=" * 70)

    # --- n_restarts distribution ---
    print("\n## 🔁 BFGS Restart Count Distribution (per response turn)\n")
    if n_restarts_distribution:
        total_turns_with_nr = sum(n_restarts_distribution.values())
        for nr, cnt in sorted(n_restarts_distribution.items()):
            pct = cnt / total_turns_with_nr * 100 if total_turns_with_nr else 0
            label = {1: "κ<τ₁", 3: "τ₁≤κ<τ₂", 5: "κ≥τ₂"}.get(nr, "?")
            print(f"  n_restarts={nr} ({label}): {cnt} turns ({pct:.1f}%)")
    else:
        print("  No restart data found (SR-Scientist mode or first turn).")
    print("-" * 70)

    # --- Hessian / κ / n_eff distribution ---
    print("\n## 🔬 Hessian Signal Distribution (best equation per problem)\n")
    if kappa_values:
        kv = np.array(kappa_values)
        kv_finite = kv[np.isfinite(kv)]
        print(f"  κ (condition number) across {len(kv_finite)} problems:")
        print(f"    Median: {np.median(kv_finite):.2e}")
        print(f"    Mean:   {np.mean(kv_finite):.2e}")
        print(f"    P25:    {np.percentile(kv_finite, 25):.2e}")
        print(f"    P75:    {np.percentile(kv_finite, 75):.2e}")
        # Count restart buckets
        n1 = int(np.sum(kv_finite < 100))
        n3 = int(np.sum((kv_finite >= 100) & (kv_finite < 1000)))
        n5 = int(np.sum(kv_finite >= 1000))
        print(f"    Would use 1 restart (κ<100):          {n1} problems")
        print(f"    Would use 3 restarts (100≤κ<1000):    {n3} problems")
        print(f"    Would use 5 restarts (κ≥1000):        {n5} problems")
    else:
        print("  No κ values recorded.")

    if n_eff_values:
        ne = np.array(n_eff_values)
        print(f"\n  n_eff (effective param count) across {len(ne)} problems:")
        print(f"    Median: {np.median(ne):.2f}")
        print(f"    Mean:   {np.mean(ne):.2f}")
        print(f"    P25:    {np.percentile(ne, 25):.2f}")
        print(f"    P75:    {np.percentile(ne, 75):.2f}")
        over_param = int(np.sum(ne < 5.0))
        print(f"    Over-parameterised (n_eff<5): {over_param} problems "
              f"({over_param / len(ne) * 100:.1f}%)")
    else:
        print("\n  No n_eff values recorded.")
    print("-" * 70)

    # --- Buffer size distribution ---
    print("\n## 📦 AdaptSci Experience Buffer Statistics\n")
    if buffer_sizes:
        bs = np.array(buffer_sizes)
        print(f"  Buffer sizes across {len(bs)} problems:")
        print(f"    Mean:   {np.mean(bs):.1f}")
        print(f"    Median: {np.median(bs):.1f}")
        print(f"    Min:    {int(np.min(bs))}")
        print(f"    Max:    {int(np.max(bs))}")
    else:
        print("  No buffer data recorded.")
    print("-" * 70)

    # --- Early stopping statistics ---
    print("\n## ⏹  Early Stopping Statistics\n")
    if early_stop_calls_list:
        es = np.array(early_stop_calls_list)
        print(f"  Problems with recorded call counts: {len(es)}")
        print(f"    Mean calls at stop:   {np.mean(es):.1f}")
        print(f"    Median calls at stop: {np.median(es):.1f}")
        triggered = int(np.sum(es < 1000))
        print(f"    Triggered early stop (< 1000 calls): {triggered} ({triggered/len(es)*100:.1f}%)")
    else:
        print("  No early-stopping data recorded.")
    print("-" * 70)

    # ======================================================================
    # Final Results by Dataset
    # ======================================================================
    print("\n" + "=" * 70)
    print("        Final Submitted Results Statistics by Dataset")
    print("=" * 70)

    grouped_results = defaultdict(list)
    for res in results_data:
        sr = res.get("submitted_results")
        if sr and isinstance(sr.get("mse"), (int, float)):
            grouped_results[res["source"]].append(sr)

    dataset_order_full = [
        "lsr_transform", "lsr_synth/matsci", "lsr_synth/chem_react",
        "lsr_synth/bio_pop_growth", "lsr_synth/phys_osc",
    ]
    for ds in dataset_order_full:
        vsubs = grouped_results.get(ds)
        print(f"\n--- 📊 {ds} ---")
        if not vsubs:
            print("  No valid results.")
            continue
        print(f"  Total valid: {len(vsubs)}")
        avg_metrics = {}
        all_keys = set().union(*(s.keys() for s in vsubs))
        for key in all_keys:
            vals = [s[key] for s in vsubs if key in s and isinstance(s[key], (int, float))]
            if vals:
                avg_metrics[key] = np.mean(vals)
        for key in ["acc_0.1", "acc_0.01", "acc_0.001", "success"]:
            if key in avg_metrics:
                print(f"    {key}: {avg_metrics[key]:.2%}")
        for key in sorted(avg_metrics):
            if key not in ["acc_0.1", "acc_0.01", "acc_0.001", "success"]:
                v = avg_metrics[key]
                if key in ["nmse", "mse"]:
                    print(f"    {key}: {v:.2e}")
                elif key == "mape":
                    print(f"    {key}: {v:.2%}")

    # --- Overall ---
    print("\n" + "=" * 70)
    print("    🌍 Overall Final Statistics (All Datasets)")
    print("=" * 70 + "\n")
    all_subs = [
        res["submitted_results"]
        for res in results_data
        if res.get("submitted_results") and isinstance(res["submitted_results"].get("mse"), (int, float))
    ]
    if not all_subs:
        print("  No valid results.")
    else:
        print(f"  Total valid: {len(all_subs)}")
        all_keys2 = set().union(*(s.keys() for s in all_subs))
        overall_avg = {}
        for key in all_keys2:
            vals = [s[key] for s in all_subs if key in s and isinstance(s[key], (int, float))]
            if vals:
                overall_avg[key] = np.mean(vals)
        for key in ["acc_0.1", "acc_0.01", "acc_0.001", "success"]:
            if key in overall_avg:
                print(f"    {key}: {overall_avg[key]:.2%}")
        for key in sorted(overall_avg):
            if key not in ["acc_0.1", "acc_0.01", "acc_0.001", "success"]:
                v = overall_avg[key]
                if key in ["nmse", "mse"]:
                    print(f"    {key}: {v:.2e}")
                elif key == "mape":
                    print(f"    {key}: {v:.2%}")

    # --- Execution status breakdown ---
    print("\n" + "=" * 70)
    print("## ⚙️  Tool Execution Status Distribution\n")
    if total_tool_calls > 0 and execution_status_distribution:
        sdf3 = pd.DataFrame({
            "Status": [s for s, _ in execution_status_distribution.most_common()],
            "Count":  [c for _, c in execution_status_distribution.most_common()],
            "% of Total": [
                f"{c / total_tool_calls * 100:.2f}%"
                for _, c in execution_status_distribution.most_common()
            ],
        })
        print(sdf3.to_string(index=False))


if __name__ == "__main__":
    import sys
    output_file = sys.argv[1] if len(sys.argv) > 1 else "output_file_path"
    analyze_results(output_file)
