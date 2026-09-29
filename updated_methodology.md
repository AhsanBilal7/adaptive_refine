# Training-Free Procedure RSI for Breadth–Depth Self-Refinement

**Proposed method: improve refinement using measured action effects and reusable, guarded two-slot repair plans.**

**Status:** Research proposal, not an experimentally established method. Literature checked on 28 September 2026. The contribution below is a specific novelty hypothesis; a literature review cannot guarantee priority or acceptance.

## 1. Main idea

The original paper improves a solution by repeatedly applying **generate → critique → correct**, then voting over the final rollouts. We add an outer loop that improves **the procedure deciding which rollout to refine, how to refine it, and when to stop**.

On development problems, the system saves a reasoning state, tries alternative actions from that exact state, and measures which action helps the final vote under the same remaining compute limit. It converts repeated findings into executable rules used on later problems. All model weights remain fixed.

**The contribution hypothesis is interaction-aware procedure evolution:** identify cases where two coordinated repairs help the final vote more than either repair alone, convert them into short guarded plans, and improve those plans across problems. Fixed-history counterfactual credit already exists in C3, and learned component skipping already exists in LW2S; neither is claimed as new here. [13,14]

This transfers **KernelAscent Task 3 into mathematical reasoning**. The attached integration report largely takes the opposite direction, applying reasoning refinement to kernels. Here, mathematical answers and plurality voting remain central; a kernel grader or profiler is not assumed to exist during reasoning inference.

## 2. What the attachments establish—and what remains unresolved

The attached *Refining Over Resampling* manuscript uses the same model in generator, critic, and corrector roles. Its default is $N=8,D=4$, requiring $N(1+3D)=104$ logical sequence generations per problem before batching. The method addresses some limitations of independent sampling by revising each candidate before aggregation. Its critique ablation supports explicit feedback in several settings.

The paper motivates depth through two limits of resampling: repeated draws can revisit the same likely reasoning modes, and plausible traces can retain local errors. Critique and correction aim to repair those errors before voting. The paper itself acknowledges higher inference cost, dependence on critique/repair quality, and evaluation restricted to math with extractable answers.

The extension should address the following concrete gaps.

| Observation in the manuscript | Implication for this proposal |
|---|---|
| Every rollout receives the same three-stage cycle and depth. | Spend compute conditionally on preservation, repair, full refinement, or a fresh attempt. |
| Correct-to-wrong flips occur; Figure 4 includes negative rollout-level net correction benefit. | Record harmful interventions as well as successful ones. Evaluate their effect on the final vote. |
| Table 3 shows nonmonotonic accuracy as breadth/depth increases. | Learn when another refinement is useful; do not assume deeper is better. |
| Main comparisons match initial sample count, not total compute (§5). | Test against genuinely budget-matched sampling and refinement. Accuracy gain per extra TFLOP does not establish superiority at equal compute. |
| No procedure state persists across problems. | Retain validated control rules and their evidence across development rounds. |
| Evaluation relies on well-defined mathematical answers. | Retain that scope; generalization to open-ended tasks is not established by this proposal. |

Three interpretations also need care. Equation (2) describes single-trace accuracy under a perturbation model; it does **not** prove an asymptotic ceiling for majority voting. Independent samples do not guarantee distinct useful reasoning strategies. Finally, answer disagreement in Appendix A.4 is not proof of reasoning diversity or correctness. The geometry example in Appendix B also contains an incorrect intermediate implication: $AB=BC$ entails $\angle BAC=\angle ACB$, not $\angle ABC=\angle ACB$. Therefore, correct final answers must not be treated as verified rationales.

The supplied KernelAscent webpage retracts its original model-level “no compounding” interpretation, documents insufficient update throughput, and marks historical speed results provisional after a compiler-environment defect. Its strategy-cap experiment also distinguishes **more stored strategies** from **better performance**. We transfer its separation of procedure/weights, lineage records, frozen controls, and measurement checks—not an assumption that compounding has been demonstrated. [K]

## 3. System definition

Let $\pi_\theta$ be the frozen LLM. Maintain an editable procedure

$$
H_t=(P_t,\mathcal R_t,\mathcal M_t),
$$

where $P_t$ contains role prompts, $\mathcal R_t$ is an ordered list of conditional action rules, and $\mathcal M_t$ stores evidence for those rules. The outer-round index is $t$; within-problem decisions use $d$.

| Component | Editable? | Function |
|---|---|---|
| Generator, critic, corrector, procedure-editor weights | No | Use the same frozen model with role prompts. |
| Critique/correction instructions | Yes | Change the error-checking or revision instruction. |
| Action predicates and priority | Yes | Decide which branch gets which operation. |
| Evidence memory | Yes | Store useful, ineffective, and harmful interventions. |
| Answer extractor, gold grader, final voting rule, budget meter | No | Keep measurement and output selection invariant. |

Start $H_0$ with the paper's round-robin refinement schedule. Only critique/correction prompts and control rules may change; initial and rewrite generator prompts remain fixed. The editor cannot redefine correctness, read test labels, alter vote weights, or exempt calls from compute accounting.

“Training-free” means **no gradient updates or newly trained controller/verifier**. Procedure development still uses labeled development questions. It is not supervision-free or cost-free. At final inference, neither gold answers nor an external correctness scorer is available.

## 4. Inner loop: adaptive refinement with one vote per slot

### 4.1 Initialize independent solutions

Draw $N$ initial traces independently from the same problem-only generator prompt:

$$
r_i^{(0)}\sim\pi_\theta^G(\cdot\mid x),\qquad i=1,\ldots,N.
$$

Keep one active trace in each of the $N$ slots. Store earlier versions for analysis, but only the current version votes. A restart replaces a slot; it does not add votes. No candidate is copied into multiple slots.

### 4.2 Construct the observable state

The decision state $S_d$ includes the problem, current traces, normalized answer counts, per-slot depth, previous answer changes, frozen-slot flags, and remaining compute. Include a critique category only if a critique has already been generated; its call is counted. Initially that category is `unknown`.

Rule predicates use a fixed feature schema: depth, remaining-budget fraction, answer-frequency fraction, number of recent answer changes, normalized-answer equality, and the latest critique category. Categories such as `arithmetic`, `assumption`, `case omission`, or `no specific error` are model assertions, not correctness labels. No rule may contain a problem ID, a gold answer, or a lookup of solved evaluation questions. Reject prompt patches containing benchmark answer lists or problem-specific lookup instructions.

### 4.3 Choose and execute an operation

| Action | Exact behavior | Solver calls |
|---|---|---:|
| `FULL_REFINE$i$` | Continue/rewrite $r_i$, critique the intermediate trace, then correct it. | 3 |
| `DIRECT_REPAIR$i$` | Critique the current $r_i$, then correct it without an extra generator rewrite. | 2 |
| `RESTART$i$` | Replace slot $i$ with a new problem-only independent solution. | 1 |
| `KEEP$i$` | Freeze its current trace for this solve and allocate remaining compute elsewhere. | 0 |
| `STOP` | End all refinement and vote. | 0 |
| `PAIR(m,i,j)` | Execute an accepted two-operation plan on distinct slots, checking its continuation guard after the first operation. | Sum of its primitive calls |

For full refinement, retain the paper's equations, now conditioned on the current procedure's prompts:

$$
g_i\sim\pi_\theta^G(\cdot\mid x,r_i;P_t),\quad
c_i\sim\pi_\theta^C(\cdot\mid x,g_i;P_t),\quad
r_i'\sim\pi_\theta^R(\cdot\mid x,g_i,c_i;P_t).
$$

The first matching rule chooses a legal action; ties between slots use a fixed slot order. If no rule matches, use $H_0$'s next affordable action. `KEEP` disables that slot, preventing zero-cost loops. A fixed maximum number of decisions also guarantees termination.

The call counts above describe structure, not equal compute. Meter actual input/output tokens and estimated model FLOPs, including longer contexts. Choose one predeclared cost measure for the hard cap and utility; report the other measures alongside it.

Each multi-call operation reserves enough budget for its capped inputs and outputs. An incomplete operation leaves the old slot unchanged, but its spent compute is still charged. Restarting does not restore the global budget or decision limit.

### 4.4 Return the answer

At termination, use ordinary plurality voting:

$$
\hat y=\arg\max_y\sum_{i=1}^{N}\mathbf1[\Phi$r_i$=y].
$$

Invalid extractions abstain. Break ties by the lowest slot index supporting a tied answer; if every extraction fails, return a designated invalid answer. Fix these conventions for every baseline. Agreement may inform control, but it never supplies a truth label or a preference label.

## 5. Main mechanism: measure alternative actions from the same state

A successful trajectory alone does not show that its critique caused the success. The parent may already have been correct, or a cheaper restart may have worked equally well.

On **development data only**, save complete states reached by $H_t$. At a sampled state, evaluate the incumbent action and alternatives using exact copies of the population, history, procedure, and remaining budget. For each alternative $a$:

1. Execute $a$ on its designated slot, or stop immediately.
2. Continue with the same incumbent procedure $H_t$ until termination.
3. Grade the final population vote against the development answer.
4. Record all additional inference cost, including unsuccessful calls.
5. Repeat with several random seeds.

Define the measured action value

$$
\widehat Q_t(S,a)=\frac1K\sum_{k=1}^{K}
\left[\mathbf1\{\hat y_{S,a,k}=y^*\}
-\lambda\frac{C_{S,a,k}}{B}\right],
$$

where $C_{S,a,k}$ is **future** inference cost from the saved state, $B$ is the full per-problem limit, and all continuations respect the same remaining cap. Past cost is identical and cancels. $\lambda$ expresses a development-set accuracy–cost trade-off; also report accuracy and cost separately.

Relative to the incumbent decision $a_0$, store

$$
\widehat\Delta_t(S,a)=\widehat Q_t(S,a)-\widehat Q_t(S,a_0).
$$

This is an empirical intervention comparison conditional on the current continuation procedure. It is not proof of a universal causal effect or an optimal policy. Fresh end-to-end evaluation is necessary after a procedure changes.

**Why population credit matters.** Suppose the three current answers are $[0,0,1]$, with gold answer $1$. Repairing one `0` to `1` changes the winning answer. In $[0,0,0,0,1]$, the same individual repair does not immediately change the winner. A controller trained only on trace correctness misses this distinction; continuing each experiment to its terminal vote can capture subsequent effects too.

**Keep negative evidence.** Store state features, action, parent/child answers, terminal vote, costs, seed, model/procedure hashes, and recovery/regression indicators. Do not require a newly correct answer before storing a useful result: an action that reliably avoids harming a correct solution or saves compute is also informative.

For efficiency, first study two selected slots per state plus `STOP`, rotating slot coverage across states. Untested actions remain unknown; they are never labeled unsuccessful. Budget all counterfactual calls as procedure-development cost.

### 5.1 Distinct extension: learn when two repairs must be coordinated

Plurality voting can hide useful partial progress. For $[0,0,0,0,1]$ with gold $1$, repairing either of two wrong slots leaves the wrong answer ahead; repairing both produces $[1,1,0,0,1]$. This example concerns the immediate vote; measured values below include the full remaining continuation.

For a small set of candidate pairs on distinct slots, measure four branches from the same state: incumbent continuation, force $a$ then continue, force $b$ then continue, and force $a$ followed by $b$ then continue. Let their values be $Q_\varnothing,Q_a,Q_b,Q_{a\to b}$. Compute

$$
I(S,a,b)=Q_{a\to b}-Q_a-Q_b+Q_\varnothing.
$$

This is a descriptive interaction contrast under the specified continuation, not a universal causal decomposition. Evaluate every branch under the same remaining cap; do not add savings from separately evaluated branches.

A pair becomes a candidate only if it has reproducibly positive interaction and its utility exceeds the incumbent and both single-action alternatives. Also require positive interaction in correctness probability alone, computed by replacing each $Q$ with its empirical success rate; saving controller overhead must not masquerade as voting synergy. Try at most two ordered pairs per sampled state in the pilot, including some exploration outside high-disagreement states. Pair operations are chosen from full refinement, direct repair, and restart. Reverse order is a separate experiment.

Compile repeated findings into a short plan:

~~~text
start guard: observable conditions selecting two distinct eligible slots
first operation: a
continuation guard: observable condition checked after a
second operation: b, only if its guard remains true
~~~

The editor chooses guards from the same fixed feature schema used by ordinary rules. It cannot use the gold answer. Evaluate the actual guarded plan again: evidence for an unconditional pair does not automatically validate a conditional plan.

Reserve budget for both operations. If the continuation guard fails, return to ordinary scheduling. If the second operation fails, retain any completed first operation, charge all work, and resume scheduling. Never assume individual gains add up.

The reusable unit is this **guarded two-slot plan**, not a memorized solution. Keep plans within a declared prompt/rule budget, permit deletion, and admit them only through fresh full-solve confirmation. This specifically targets vote-threshold interactions; it does not resolve arbitrarily large coalitions.

## 6. Outer loop: turn evidence into a reusable procedure

Use three disjoint data roles: **search data**, **fresh confirmation blocks**, and a **sealed final test set**. Deduplicate by problem and source template before splitting. All states and descendants of one problem remain in its split.

At each outer round:

1. Run $H_t$ on the next search batch and collect action comparisons.
2. Ask the frozen editor to propose a small patch: one new/revised/deleted rule, one guarded two-slot plan, or one role-prompt edit. It must name the supporting and contradicting development evidence.
3. Evaluate several patches on search data; select one before opening the next confirmation block.
4. Compare that patch with $H_t$ from the start of each confirmation problem under matched caps and seeds.
5. Accept only if the predeclared utility-improvement criterion passes and accuracy does not violate a predeclared noninferiority margin. Otherwise retain $H_t$.

Use paired question-level uncertainty, grouping repeated seeds within questions. For formal acceptance tests across $R$ rounds, allocate a total error budget in advance, for example $\alpha_t=0.05/R$, and never reuse a confirmation block for another adaptive decision. Size these blocks from pilot discordance rates; a tiny block cannot validate a one-point gain.

Example rule hypothesis—not a measured result:

```text
IF a slot has changed answers twice
AND its latest critique identified no specific error
AND sufficient budget remains:
    RESTART that slot.
```

The rule survives only if its complete procedure transfers to fresh confirmation questions. Store accepted and rejected edits, evidence counts, and versions. Permit rule replacement and deletion rather than imposing an arbitrary “12 strategies” ceiling. Keep a declared execution/context budget and measure sensitivity to it.

```text
H = original_refinement_procedure
memory = empty
for round in 1..R:
    states = solve_and_log(search_batch[round], H, frozen_model)
    evidence = compare_actions_from_copied_states(states, H)
    evidence.add(compare_bounded_two_slot_plans(states, H))
    memory.add(evidence)  # positive, zero, and negative effects
    patches = frozen_editor.propose(H, memory)
    candidate = choose_on_search_data(patches, H)
    result = paired_full_solve(candidate, H, fresh_confirmation[round])
    if acceptance_rule(result):
        H = candidate
return freeze(H)  # final test answers never enter memory
```

This is persistent **procedure adaptation**. It does not by itself prove that the editor has learned to become a better editor, or that gains compound indefinitely.

## 7. Position relative to the closest research

“Refinement + memory + an outer loop” is already established. The table identifies overlap and the narrower distinction to test; it does not claim prior methods are incapable of implementing it.

| Prior work and verified status | Established contribution | Proposed distinction |
|---|---|---|
| Self-Refine, NeurIPS 2023 [1]; Reflexion, NeurIPS 2023 [2] | Iterative feedback/revision; persistent verbal experience. | Store measured alternatives from the same population state, including harmful actions, and evolve executable decision rules. |
| ADAS and AFlow, ICLR 2025 [3,4] | Automatic agent/workflow discovery. | Restrict search to the paper's refinement operations and supply direct intervention evidence for which operation earns its cost. |
| GEPA, ICLR 2026 Oral [5] | Reflective prompt mutation and Pareto selection from execution/evaluation feedback. | The main variable is a state-dependent refinement decision; feedback compares actions with the same starting population and continuation budget. GEPA is a mandatory optimizer baseline. |
| DGM, ICLR 2026 [6]; Hyperagents, March 2026 preprint [7] | Code self-modification; editable task and meta-level improvement mechanisms. | We study a narrower reasoning-control problem with fixed grading/voting and explicit action credit. Self-modification itself is not new. |
| Meta-Harness, COLM 2026 as listed by its authors [8] | Code-space harness search using detailed execution records; includes math retrieval. | Replace observational diagnosis alone with controlled action experiments inside the refinement population. Reuse its optimizer as a strong baseline. |
| CoRefine, February 2026 preprint [9] | Trained confidence controller for halt/rethink/alternative; hybrid tree variant. | Our weights remain frozen; action effects are measured on terminal population utility rather than inferred from confidence/history heuristics. Adaptive control itself is not new. |
| RSA, public paper revised February 2026 [10] | Population self-aggregation and aggregation-aware training. | Improve the reusable control procedure across problems while retaining the original paper's one-slot-one-vote output rule. Population refinement itself is not new. |
| MoMHa, September 25, 2026 preprint [11] | Joint optimization of harness accuracy, safety, and token cost. | A cost-aware harness objective is insufficient novelty; the proposed distinction is intervention-based population credit. |
| C3 / *Exact Is Easier*, revised May 2026 preprint [13] | Restores history, evaluates alternative actions through terminal continuations, and trains from their credit. | This is the foundation, not our invention. Test whether discovering reusable two-slot plans adds value over primitive-action credit. |
| LW2S, September 25, 2026 preprint [14] | Learns calibrated component-skip decisions from controlled interventions, including a debate-refine setting. | Test coordinated repair/restart plans and persistent rule evolution with all weights frozen; controlled skipping alone is not new. |
| CCPO, revised June 2026 preprint [15] | Attributes team reward through agent-removal counterfactuals. | Keep the population intact and compare repair operations and their interactions. |

**Defensible proposed contribution:** “We evolve a frozen-model refinement procedure by identifying nonadditive voting gains from bounded two-slot interventions and compiling them into reusable guarded plans.” The combination is a candidate contribution; the reviewed credit-assignment frameworks can already express broad intervention ideas. Its value must come from the specific mechanism and evidence.

C3's reported validation uses short two-/three-decision protocols and identifies stale policy values and diminishing action diversity as limitations. LW2S chiefly tests reward-preserving omission rather than acquiring new repair capability. Our bounded plans and fresh measurements address those specific design gaps, but broader scalability and distribution-shift robustness remain experimental questions. [13,14]

## 8. Experiments needed to support the method

**Primary outcome:** held-out final-answer accuracy versus measured compute, including all control, generation, critique, and correction work. Use both equal hard caps and comparisons at comparable realized cost. Report latency separately because sequential depth and parallel breadth differ operationally.

Use the original five math benchmarks for continuity, plus a larger disjoint test split for statistical power and a source/template-held-out transfer split. AIME sets contain few questions; report solved counts and seed variability, not only percentages. Freeze checkpoint revisions and prompts for reproducibility.

Required baselines are compute-matched majority sampling, original fixed refinement, fixed heuristic adaptive refinement, GEPA on the role prompts, AFlow/Meta-Harness on the same allowed operations, Reflexion-style memory, and CoRefine/RSA under their disclosed access requirements. Add an LW2S-style skip controller, C3-style primitive-action credit, and a C3 optimizer given the same pair-action vocabulary. The last comparison separates a useful action vocabulary from a genuinely better improvement procedure. Separate frozen-weight optimizer comparisons from trained-controller comparisons; label each access regime. Give all development optimizers comparable search and evaluator budgets. Recent *Sample More, Reflect Less* reinforces this need in its tested small-model settings; it does not settle the result for every model or this proposal. [12]

The decisive ablations are:

- Guarded pair plans versus primitive-only control at equal total intervention/search budgets.
- Interaction-selected plans versus equally many randomly selected or immediate-vote-greedy plans.
- Population-vote action credit versus individual-trace correctness credit.
- Matched-state experiments versus ordinary successful-trajectory summaries.
- Signed evidence versus success-only memory.
- Evolved rules versus evolved prompts alone; `KEEP`/`RESTART` removed separately.
- Persistent procedure versus reset procedure with the **same accumulated evidence** and total search budget.
- At a fixed checkpoint, freeze both procedure and memory in one arm while continuing adaptation in the other; keep task stream and subsequent budgets matched.

Report conditional wrong-to-correct and correct-to-wrong rates, final-vote recoveries/regressions, candidate accuracy, extraction failures, action frequencies, accepted edits, and evidence yield. Cluster longitudinal inference by independent optimization run; rounds are not independent replicates. A positive gain over $H_0$ establishes adaptation. A continued-lineage advantage over strong reset/frozen controls across later rounds is evidence for sustained benefits, not proof of unlimited recursive improvement.

## 9. Minimal implementation and remaining risk

Start with $N=4$, a cap allowing roughly two full refinement cycles per slot after initialization, four continuation seeds per tested action, and five outer rounds. These are pilot settings, not validated optima. Move to the paper's $N=8,D=4$ setting only after measuring whether action preferences have usable signal and development cost is justified.

Implement five modules: a capped role-call wrapper; a slot/history store; a state-copy/action-comparison runner; a restricted rule editor; and a sealed evaluator. Before the study, check extraction on known equivalent/inequivalent answers and test that the evaluator detects deliberately changed predictions. Estimate the number of informative action comparisons expected per round. A channel producing almost no distinguishable interventions is not a meaningful recursion experiment.

The main unresolved risks are noisy action values, missed interactions between simultaneous repairs, rule overfitting, shared model errors, and costly development. `KEEP` can preserve a wrong answer and `RESTART` can discard a useful one. Full-solve validation reduces these risks; it cannot guarantee correctness. The method can improve allocation only where the underlying model has usable repair or resampling capability.

## References and source boundaries

**Attached sources read in full:** *7357_Refining_Over_Resampling_ (2)(2).pdf* (16 pages, including appendices); *Integrating Kernel-Ascent RSI with LLM Self-Refinement.pdf* (21 pages); *Pasted text(20260928-070349).txt*. The live KernelAscent page was also retrieved; its static caveats match the supplied snapshot. The attached text supplies the dynamically populated board values. No unpublished experiments or reviewer reports were available.

- **[K]** [KernelAscent project](https://ahmd-mohsin.github.io/KernelAscent/) and [repository](https://github.com/ahmd-mohsin/KernelAscent). Project evidence, not a verified top-tier publication. The supplied material contains both old leaderboard language and later withdrawals; the caveats take precedence.
- **[1]** [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651), NeurIPS 2023.
- **[2]** [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366), NeurIPS 2023.
- **[3]** [Automated Design of Agentic Systems](https://proceedings.iclr.cc/paper_files/paper/2025/hash/36b7acf6f6010652b3f2a433774a66fe-Abstract-Conference.html), ICLR 2025.
- **[4]** [AFlow: Automating Agentic Workflow Generation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5492ecbce4439401798dcd2c90be94cd-Abstract-Conference.html), ICLR 2025.
- **[5]** [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html), ICLR 2026; [full paper](https://arxiv.org/abs/2507.19457).
- **[6]** [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954), [ICLR 2026 paper](https://openreview.net/pdf/ed3ddb9e9f1b6ad69626362874aa9d16e434528b.pdf).
- **[7]** [Hyperagents](https://arxiv.org/abs/2603.19461), March 2026 preprint.
- **[8]** [Meta-Harness](https://arxiv.org/abs/2603.28052); [author project and COLM 2026 listing](https://yoonholee.com/meta-harness/).
- **[9]** [CoRefine](https://arxiv.org/abs/2602.08948), February 2026 preprint, especially §3.3 on oracle-label generation.
- **[10]** [Recursive Self-Aggregation](https://arxiv.org/abs/2509.26626); [author project](https://rsa-llm.github.io/). Conference acceptance was not independently confirmed in this review.
- **[11]** [MoMHa](https://arxiv.org/abs/2609.30967), September 2026 preprint. A conference-template footer is not evidence of acceptance.
- **[12]** [Sample More, Reflect Less](https://arxiv.org/abs/2607.28576), July 2026 preprint; limited model/task scope and completion-token accounting.
- **[13]** [Exact Is Easier: Credit Assignment for Cooperative LLM Agents (C3)](https://arxiv.org/abs/2603.06859), May 2026 revision of the March manuscript originally titled *Contextual Counterfactual Credit Assignment*.
- **[14]** [Learning What to Skip (LW2S)](https://arxiv.org/abs/2609.30734), September 25, 2026 preprint.
- **[15]** [Counterfactual Credit Policy Optimization (CCPO)](https://arxiv.org/abs/2603.21563), June 2026 revision. The latest version emphasizes Think–Solve; older indexed abstracts also mention voting.

This is a targeted review of the closest mechanisms, not an exhaustive claim to have checked every recent paper.
