# Adaptive Delta Refinement: Repair Only the Reasoning That Must Change

**Revised recommendation — 29 September 2026**

**One problem:** iterative self-refinement repeatedly regenerates reasoning that does not depend on the detected error.

**One change:** replace whole-solution rewriting with a patch to the faulty step and its dependent steps. Reuse everything else.

This directly changes the base reasoning method. It needs no development-set procedure search, learned controller, reward model, model-specific calibration, or cross-question memory. The proposal is training-free and operates on the current question.

## 1. Why this is a better research target

In *Refining Over Resampling*, every active rollout receives a generator continuation, a critique, and a correction at every prescribed depth. This can regenerate a long solution even when the required change is small. It also exposes unrelated reasoning to unnecessary modification.

The concrete research question is:

> Can self-refinement retain its correction benefit while generating only the reasoning affected by each detected error?

This is narrower and more practical than learning an entire refinement procedure. The proposed adaptivity concerns **repair scope**: a local arithmetic error triggers a small patch; an early assumption affecting the whole argument triggers a large repair.

Keep the original paper's independent rollouts and final plurality vote. Change the refinement operator inside each rollout.

| Base method | Revised method |
|---|---|
| Generate a complete initial solution. | Generate a complete solution with numbered steps and dependency references. |
| Rewrite/continue before every critique. | Critique the current solution directly. |
| Produce another complete corrected solution. | Generate only the faulty step and its dependent steps. |
| Use the same depth for every rollout. | Stop a rollout when the critic finds no actionable error; otherwise continue within the shared budget. |
| Aggregate final rollout answers. | Keep the same one-answer-per-rollout vote. |

The main contribution is the change in repair scope. Removing the extra rewrite and stopping early are supporting simplifications, not sufficient novelty claims.

## 2. The closest literature and the remaining gap

| Paper and verified venue | What it establishes | Relevant gap for this proposal |
|---|---|---|
| **Thought-ICS: Structure Enables Effective Self-Localization of Errors in LLMs**, ICML 2026 [1] | Structured thoughts support error localization; correction backtracks and resamples from a retained prefix. It includes an autonomous variant. | Suffix regeneration also discards later steps that are independent of the error. Test noncontiguous repair against its autonomous method. |
| **Graph-Based Chain-of-Thought Pruning**, Findings of ACL 2026 [2] | Uses dependency graphs to identify redundant reflection, followed by SFT, DPO, and GRPO. | Its graph is used for post-training data/behavior optimization. Our graph controls which steps are regenerated immediately on a new question, with fixed weights. |
| **Stop Unnecessary Reflection**, ICLR 2026 [3] | Learns efficient reflection with coordinated RL penalties. | It requires post-training. Our repair scope is computed from the current solution rather than learned through a length/reflection reward. |
| **Thought Calibration**, EMNLP 2025 [4] | Trains and calibrates hidden-state probes to stop reasoning. | It targets when to stop; the proposed method targets how much reasoning to regenerate when correction is still needed. |
| **Wider or Deeper? / AB-MCTS**, NeurIPS 2025 [5] | Adapts search breadth and depth using an external or surrogate scoring function. | Our setting retains the original paper's lack of an external inference-time scorer and avoids a search tree over complete answers. |

The important distinction from Thought-ICS is **dependency locality rather than position locality**. A later step is not necessarily affected just because it appears after the error.

Graph reasoning and selective recomputation themselves are established ideas. Graph of Thoughts already supports dependency-based thought transformations [6]. A recent memory-agent preprint also selectively replays computation downstream of diagnosed faulty memories [7]. Therefore, the defensible contribution is:

> **An inference-only, dependency-closed patch operator for mathematical self-refinement, with measured savings over both full rewriting and prefix-based suffix regeneration.**

This is a specific research hypothesis, not a guarantee of publication priority. The experiments must establish that the finer repair scope is useful enough to justify its small representation overhead.

## 3. Methodology

### 3.1 Generate independent structured solutions

For input question $x$, generate $N$ independent solutions using the same frozen model as the base method. Each solution is a numbered list:

~~~text
1. uses: []       statement and calculation
2. uses: []       statement and calculation
3. uses: [1]      statement using the result of step 1
4. uses: [2]      statement using the result of step 2
5. uses: [3,4]    final calculation and answer
~~~

Each step contains one meaningful calculation or deduction. The model declares the earlier steps whose results it uses. Information taken directly from the question needs no step reference.

Generate the entire list in **one call per rollout**. Do not call a separate model to convert a completed solution into a graph.

Represent the result as $G=(V,E)$, with an edge $u\to v$ when step $v$ uses step $u$. References point backward, so this is a directed acyclic graph. This representation is simply a numbered solution plus dependency IDs; it does not introduce another agent. A rollout with malformed structure uses the same full-rewrite fallback until a valid structured solution is available.

### 3.2 Reuse the existing critic to identify the repair target

The critic reads the question and current solution and returns one of:

~~~text
KEEP
FIX(step_id, concrete_error)
REBUILD(reason)
~~~

- **KEEP:** no actionable error was found; freeze this rollout.
- **FIX:** identify the first faulty step and explain its error briefly.
- **REBUILD:** the approach or dependency structure needs a global change.

Ask the critic to check both the mathematical reasoning and missing dependency references. Do not provide other rollouts' answers or the current majority. Each rollout remains independent.

KEEP is a stopping decision, not a certificate of mathematical correctness.

### 3.3 Compute the repair set deterministically

For a localized faulty step $j$, define

$$
A(j)=\{j\}\cup\operatorname{Desc}_G(j).
$$

Here $\operatorname{Desc}_G(j)$ contains every step that directly or indirectly uses step $j$.

Repair all steps in $A(j)$ and retain every step outside it verbatim. Ordinary graph traversal computes this set in $O(|V|+|E|)$ time. There is no learned score, similarity threshold, top-$k$ selection, or graph-size penalty.

For $m$ topologically ordered steps, $A(j)\subseteq\{j,\ldots,m\}$. Thus the requested repair contains no more steps than suffix regeneration, and strictly fewer when some later steps are independent. This is a basic scope property, not a guarantee about generated-token length or latency.

This rule includes the final answer whenever it depends on the faulty step. It prevents the common mistake of fixing an intermediate value while retaining downstream calculations made from the old value.

### 3.4 Replace the corrector's full rewrite with a patch

Supply the corrector with:

1. The original question.
2. The current numbered solution.
3. The critic's specific error.
4. The exact list of step IDs allowed to change.

The corrector outputs replacements for those IDs only, in dependency order. It may revise their dependency lists, provided every reference points to an existing earlier step. The implementation inserts these replacements and keeps the other steps byte-for-byte unchanged.

Formally,

$$
G^{t+1}=G^t\oplus
\operatorname{Patch}_{\theta}\!\left(x,G^t,c^t,A(j^t)\right),
$$

where $\oplus$ replaces the listed steps rather than appending a second solution.

Keep step IDs and their intended subproblems fixed during a local patch. If a different decomposition is needed, use REBUILD to generate a new complete structured solution. Invalid references or a malformed patch use this same full-rewrite fallback if budget permits; otherwise retain the last complete solution.

This fallback is part of the method and must be counted in its cost.

### 3.5 Repeat only while correction is needed and affordable

Schedule active rollouts in round-robin order. For each:

1. Critique the current solution.
2. Freeze it on KEEP.
3. Apply a local patch on FIX, or a full rewrite on REBUILD.
4. Revisit the resulting solution on its next turn.

A frozen rollout consumes no further calls, so the remaining budget is available to the others. Stop when all rollouts are frozen or the question's compute budget is exhausted. If an update reproduces a previously seen complete state exactly, freeze that rollout to prevent cycles.

Use the last complete solution from each rollout and apply the original final-answer plurality vote. Old versions never receive additional votes. Use a fixed tie-break and extraction policy across all methods.

This is recursive correction of the **current solution state**. It does not improve the model's weights or establish persistent recursive self-improvement across questions. The paper should be positioned as adaptive test-time refinement.

## 4. Worked example

Suppose a rectangle is 12 m by 5 m. Painting costs $3 per square metre, and fencing costs $2 per metre. Compute their combined cost.

An initial solution contains:

| Step | Current statement | Uses | Action after detecting the error in step 1 |
|---|---|---|---|
| 1 | Area: $12\times5=50$ | Question | Replace with $60$. |
| 2 | Perimeter: $2(12+5)=34$ | Question | Keep unchanged. |
| 3 | Painting cost: $3\times50=150$ | 1 | Replace with $180$. |
| 4 | Fencing cost: $2\times34=68$ | 2 | Keep unchanged. |
| 5 | Total: $150+68=218$ | 3, 4 | Replace with $248$. |

The repair set is $\{1,3,5\}$. The perimeter and fencing calculations remain usable even though they occur after the faulty area calculation.

A full rewrite regenerates all five steps. A suffix restart from step 1 also regenerates all five. Adaptive Delta Refinement emits three replacements. This is an illustration of avoided work, not an experimental speedup claim.

If every later step depends on step 1, the repair set becomes the entire suffix. The same rule adapts automatically; no separate policy selects a repair radius.

## 5. Algorithm

~~~text
Input: question x, frozen model, breadth N, total budget B
states = generate N independent structured solutions
active = all rollout IDs

while active is not empty and another update is affordable:
    i = next active rollout in round-robin order
    feedback = critique(x, states[i])

    if feedback is KEEP:
        remove i from active
        continue

    if feedback is FIX(j, error):
        affected = j plus all descendants of j
        candidate = patch_only(x, states[i], affected, error)
    else:
        candidate = full_rewrite(x, states[i], feedback)

    if candidate is structurally invalid:
        candidate = budgeted_full_rewrite_or_keep_previous()

    if candidate repeats an earlier complete state:
        remove i from active

    states[i] = last complete candidate

return plurality_vote(extract_answer(states))
~~~

All calls, failed outputs, and fallbacks count against $B$. Reserve the configured maximum cost of a critique-plus-update before starting a turn. If KEEP uses less than this reservation, return the unused allowance to the question's budget. Incomplete updates never replace a complete solution.

## 6. What makes this adaptive and efficient?

**Adaptive repair scope.** The number of regenerated steps follows the actual dependency closure. A local mistake and a global assumption error naturally receive different amounts of work.

**Adaptive refinement depth.** A rollout stops when the existing critic finds no specific error. Other rollouts continue using the remaining budget.

**No offline deployment cost.** The same prompt contracts and graph traversal apply to every evaluation model. Standard chat templates may differ, but no procedure is fitted separately for each model.

The principal configuration is the existing breadth $N$ and a total compute cap $B$, replacing fixed refinement depth $D$. Decoding settings and per-call output caps are shared with the baselines. There are no additional confidence thresholds or learned scheduling parameters.

For the original method, $N=8,D=4$ gives 104 logical sequence generations. For the revised method,

$$
C_{\mathrm{new}}
=C_{\mathrm{initial}}
+\sum_t C_{\mathrm{critic},t}
+\sum_t C_{\mathrm{patch/rebuild},t}.
$$

The important saving is fewer generated reasoning tokens, not merely fewer calls. The critic still reads the current solution, and the default patch prompt also includes it. Do not claim that all prefill cost disappears or that fewer output tokens imply an equal percentage of wall-clock savings. Ordinary exact-prefix caching can be used equally by every baseline; arbitrary unchanged text does not imply reusable KV states.

## 7. A practical experiment that can establish the contribution

First run a predeclared 50-question feasibility pilot on one model. Check whether dependency references are well formed and whether detected errors produce repair sets substantially smaller than their suffixes. If the solutions are mostly linear or fallbacks dominate, this direction is unlikely to save enough compute; do not begin a large sweep. This pilot fits no rules or thresholds, and its failures must remain visible in the report.

For the main study, start with a single instruction model on MATH500, the paper's existing answer extractor, and $N=4$. Evaluate a small predeclared set of total token/FLOP budgets. This is evaluation of a fixed algorithm, not a search for model-specific rules.

The essential comparison is:

| Variant | Purpose |
|---|---|
| Original three-stage fixed refinement | Establish improvement over the attached method. |
| Direct critique + full rewrite, with the same stopping rule | Separate sparse repair benefits from simply removing one generation stage. |
| Direct critique + suffix regeneration | Test whether dependency locality improves over position locality. |
| Adaptive Delta Refinement | Proposed method. |
| Compute-matched majority sampling | Check whether repair is worth more than additional independent attempts. |

For the three direct-critique variants, start from the same structured solutions and use identical critique outputs at the first repair. Then evaluate their complete end-to-end behavior. Include the published autonomous Thought-ICS method as the closest full-system baseline; never compare its oracle-assisted results with our label-free inference.

Only two ablations are essential:

1. Give every step all earlier steps as dependencies. The repair set then reduces to a suffix, isolating the value of dependency information.
2. Force full rewriting while preserving the structured initial solution, stopping policy, and call budget. This isolates patch scope from formatting and stage-count changes.

Report:

- Final-answer accuracy against total input/output tokens and estimated FLOPs.
- Wall-clock latency under the same batching and caching conditions.
- Fraction of reasoning regenerated per repair.
- Wrong-to-correct and correct-to-wrong answer changes.
- Dependency/format failures and full-rewrite fallback rate.

After the first study, test unchanged prompts on a second model family and the original competition-math benchmarks. Use paired question-level uncertainty. Model-specific rule searches and large controller-training sweeps are unnecessary.

**Success criterion:** at comparable accuracy, the sparse method must cost less than direct full rewriting **and** suffix regeneration. Beating only the original three-call loop would support a simpler implementation change, not the main research claim.

## 8. The remaining technical risk

The method depends on the LLM reporting useful errors and sufficiently complete dependencies. Missing an edge can leave a stale downstream step unchanged. The whole-solution critic and full-rewrite fallback help detect this, but do not guarantee it.

Therefore, audit dependency omissions on a small labeled analysis sample and measure downstream inconsistencies. This is evaluation, not a calibration dataset required for deployment. RefineBench and Thought-ICS both reinforce that repeated self-assessment should not be assumed reliable [1,8].

The expected benefit is reduced unnecessary regeneration and fewer opportunities to disturb unrelated work. Improved accuracy and robust savings must be demonstrated; neither follows automatically from smaller patches.

## 9. Other simple ideas considered

| Idea | Assessment |
|---|---|
| Stop when answers stabilize, or when a critic says KEEP. | Very easy to implement, but overlaps heavily with existing adaptive stopping. Use as a baseline, not the central novelty. |
| Locate the first error and regenerate only its suffix. | Practical and a strong baseline, but already central to Thought-ICS. |
| Repair the error's dependency closure while preserving independent later steps. | Recommended research direction: one concrete change inside refinement, with a clear efficiency hypothesis and direct competing baselines. |

For the main paper, keep only the third idea as the contribution. Treat stopping and budget accounting as its execution details.

## References

1. **Thought-ICS**, *Structure Enables Effective Self-Localization of Errors in LLMs*, ICML 2026. [Paper](https://arxiv.org/abs/2602.02416), [official conference listing](https://icml.cc/Downloads/2026), [author venue confirmation](https://danielrjiang.github.io/). Read especially the suffix-regeneration mechanism and autonomous self-verification analysis.
2. **Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs**, Findings of ACL 2026. [Proceedings and paper](https://aclanthology.org/2026.findings-acl.281/). Findings is identified separately from the ACL main track.
3. **Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty**, ICLR 2026. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/08fe50bf209c57eecf0804f9f9ed639f-Abstract-Conference.html).
4. **Thought Calibration: Efficient and Confident Test-Time Scaling**, EMNLP 2025. [Proceedings and paper](https://aclanthology.org/2025.emnlp-main.722/).
5. **Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search**, NeurIPS 2025. [Proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/32dff2e27b0da1fb5f4209216a948544-Abstract-Conference.html), [paper](https://arxiv.org/abs/2503.04412).
6. **Graph of Thoughts: Solving Elaborate Problems with Large Language Models**, AAAI 2024. [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/29720).
7. **From Faulty Memories to Corrected Actions: Dependency-Guided Rollback Repair for Memory-Augmented Agents**, August 2026 preprint. [Paper](https://arxiv.org/abs/2608.10502). Related selective-replay work; not presented as an accepted top-tier paper.
8. **RefineBench: Evaluating Refinement Capability of Language Models via Checklists**, ICLR 2026. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/835b6c04ca2daaa1a682c2eaad4a70ea-Abstract-Conference.html).
9. **Refining Over Resampling: Test-Time Self-Correction for LLM Reasoning**, attached manuscript; [public paper](https://arxiv.org/abs/2608.05643).

This revision supersedes the earlier development-set procedure search, counterfactual action collection, and two-rollout rule library. Those components are not required by the proposed method.
