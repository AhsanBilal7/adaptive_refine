# Refined Methodology: Training-Free Procedure RSI for Breadth–Depth Self-Refinement

This document describes **exactly what the code in `src/rsi/` does**, step by step, with worked examples. The research proposal is `updated_methodology.md`. The base method is the paper *Refining Over Resampling*.

---

## 0. The idea in five sentences

1. The base paper solves a math problem by sampling **N answers** ("slots"), improving every slot with the same fixed **generate → critique → correct** cycle **D** times, then taking a **majority vote**.
2. That fixed cycle spends the same compute on every slot, even when a slot is already correct, hopeless, or getting worse.
3. We keep the model **frozen** (no training) and instead improve the **procedure**: the small set of rules that decides *which slot to work on, which operation to use, and when to stop*.
4. We learn those rules by **experiment**: on development problems we save the solving state, try different actions from that *exact* state, and measure which action makes the **final vote** more often correct for its cost.
5. Findings that repeat, especially cases where **two repairs together** flip the vote while neither alone does, become short **rules and two-slot plans**. They are kept only if they win a **fresh, statistically tested** comparison.

---

## 1. Base method (what we start from)

For a problem `x`, with the same model playing three roles:

```
for i = 1..N:            r_i ← Generator(x)                       # N independent samples
for d = 1..D:
  for i = 1..N:          g_i ← Generator(x, r_i)                  # continue / rewrite
                         c_i ← Critic(x, g_i)                     # find errors
                         r_i ← Corrector(x, g_i, c_i)             # fix them
answer = plurality vote over the final answers of r_1..r_N
```

Cost: `N(1 + 3D)` model calls. The paper default is N=8, D=4, i.e. 104 calls per problem.

In our code this fixed schedule is called **H0** ("the paper's round-robin"). It is always available as the fallback and as a baseline.

---

## 2. The procedure we evolve: `H = (P, R, M)`

| Part | What it is | Can the editor change it? |
|---|---|---|
| **P**: prompts | Critique instructions and correction instructions (a system prompt plus an extra instruction for each) | **Yes** |
| **R**: rules | An ordered list of "IF observable condition THEN action" rules and two-slot plans | **Yes** (add / revise / delete) |
| **M**: memory | All measured evidence (good, neutral **and** bad effects) | Grows automatically |
| Generator / critic / corrector weights | One frozen LLM | **No** |
| Initial and rewrite prompts | Paper prompts | **No** |
| Answer extraction, gold grader, voting rule, cost meter | Measurement | **No** |

"Training-free" means no gradients and no trained controller. Development still uses labeled problems, so it is **not** supervision-free or cost-free.

---

## 3. Inner loop: solving one problem with procedure `H`

### 3.1 Slots

* There are exactly **N slots**. Each slot holds **one current trace**, and only the current trace votes.
* Old versions are kept in the slot's history for analysis only.
* `RESTART` **replaces** a slot; it never adds an extra vote.

**Step 1: initialize.** Draw N independent solutions from the problem-only prompt (temperature 0.7). That costs N calls.

### 3.2 What the controller can see (the fixed feature list)

Rules may only use these features. No problem text, problem id, or gold answer is ever visible.

| Feature | Level | Meaning |
|---|---|---|
| `depth` | slot | refinement operations applied since the slot's last (re)start |
| `answer_freq` | slot | fraction of the N slots that share this slot's answer |
| `answer_changes` | slot | how many times this slot's normalized answer changed |
| `recent_changes` | slot | answer changes in the slot's last 2 operations |
| `agrees_plurality` | slot | 1 if the slot's answer is the current vote winner |
| `valid` | slot | 1 if an answer could be extracted (`\boxed{}`) |
| `critique` | slot | category of the latest critique: `unknown` until a critique was generated, then one of `no_specific_error, arithmetic, algebra, logic, assumption, case_omission, misread, incomplete, other` |
| `budget_frac` | global | remaining budget ÷ total budget B |
| `top_frac` | global | votes for the current winner ÷ N |
| `margin` | global | (winner votes − runner-up votes) ÷ N |
| `n_distinct_frac` | global | distinct valid answers ÷ N |
| `invalid_frac` | global | slots without an extractable answer ÷ N |

The critic always ends its output with a line `ERROR CATEGORY: <category>`. This is the model's *claim*, never a correctness label.

### 3.3 Actions

| Action | What happens | Model calls |
|---|---|---|
| `FULL_REFINE i` | rewrite slot i → critique the rewrite → correct it (the paper's cycle) | 3 |
| `DIRECT_REPAIR i` | critique slot i as it is → correct it (no rewrite) | 2 |
| `RESTART i` | replace slot i with a fresh problem-only sample (depth resets to 0) | 1 |
| `KEEP i` | freeze slot i; it keeps voting but is never touched again | 0 |
| `STOP` | stop and vote now | 0 |
| `PAIR(plan, i, j)` | op1 on slot i → check a continuation guard → op2 on slot j (i ≠ j) | sum of both |

### 3.4 Budget: a hard cap measured in real compute

* Every call's **real** input and output tokens are counted. FLOPs = `2 × parameters × (input + output tokens)`.
* One **cost measure** is declared up front (`tflops` by default; `tokens` or `calls` are possible). It sets the cap `B` and the cost term in the utility. The other measures are always reported too.
* **Reservation:** before an operation starts, it must be able to afford its **worst case** (every input at its truncation cap, every output at `max_new_tokens`). If it can't, the operation is not allowed.
* **Incomplete operation** (a call errors or returns nothing): the slot stays unchanged, but the compute already spent **is still charged**.
* `RESTART` never gives budget back.
* **How B is set** (`budget_mode=h0_cycles`): `B = N × (worst-case initial sample) + N × cycles × (worst-case FULL_REFINE + prompt-edit allowance)`. So H0 with depth ≤ `cycles` can always finish. The pilot default is `cycles = 2`, i.e. about two full refinement cycles per slot.
* Because the reservation is worst case, H0 usually *spends* less than B (≈ 40% in our 1.5B smoke run). Adaptive procedures may use the rest. That is why we always report utility and real compute, not accuracy alone.

### 3.5 How the next action is chosen

```
for item in H.rules (in order):
    if item is a global rule (only STOP):  if its condition holds → STOP
    if item is a slot rule:  for slot i = 0..N-1 (unfrozen):
                                 if condition(i) holds and the action is affordable → do it on slot i
    if item is a two-slot plan: for slot i: for slot j ≠ i:
                                 if first_guard(i) and second_guard(j) and BOTH ops affordable → run the plan
if nothing matched → H0's next action: FULL_REFINE on the unfrozen slot with the smallest depth
                     (ties → lowest slot index), if depth < h0_depth and affordable; otherwise STOP
```

A decision cap (`4·N·cycles + 8` by default) guarantees termination.

### 3.6 Final answer (the same for every method and baseline)

* Extract the last `\boxed{...}` from each slot. If there is none, the last number is used; this fallback can be turned off.
* Normalize answers the same way as the paper: strip spaces and LaTeX spacing, unwrap `{}`, `3.500 → 3.5`, lowercase, `\dfrac → \frac`.
* **Plurality vote.** Invalid answers abstain. A tie goes to the answer held by the **lowest slot index**. If every slot is invalid, the answer is `[invalid]`.
* Agreement between slots is never used as a truth label.

### Worked example A: H0 vs. one learned rule (N = 4, measure = calls, cycles = 2)

The budget is `B = 4·1 + 4·2·3 = 28` calls.

**H0 (paper schedule, depth 2):**

| Step | Action | Calls used | Slot answers (gold = 12) | Vote |
|---|---|---|---|---|
| init | 4 samples | 4 | 12, 7, 7, 12 | 12 (tie 2–2 → slot 0's answer) |
| 1–4 | FULL_REFINE slot 0,1,2,3 | 16 | 12, 7, 9, 12 | 12 |
| 5–8 | FULL_REFINE slot 0,1,2,3 | 28 | 12, 9, 9, 7 | 9 ✗ (9 has 2 votes, 12 and 7 have 1 each) |
| end | depth 2 reached → STOP | 28 | | **9 ✗** |

Here refinement *hurt*: slot 3 flipped from correct to wrong, a "correct→wrong" regression.

**The same problem with the rule** `IF critique == no_specific_error AND agrees_plurality == 1 AND depth ≥ 1 THEN KEEP`:

| Step | Action | Calls | Answers | Why |
|---|---|---|---|---|
| init | 4 samples | 4 | 12, 7, 7, 12 | |
| 1–4 | H0: FULL_REFINE 0..3 | 16 | 12, 7, 9, 12 | no slot has a critique yet |
| 5 | KEEP slot 0 | 16 | (frozen) | its critique said "no error" and it agrees with the vote |
| 6 | KEEP slot 3 | 16 | (frozen) | same |
| 7–8 | H0: FULL_REFINE 1, 2 | 22 | 12, 9, 12, 12 | only the minority slots are refined |
| end | STOP | 22 | | **12 ✓**, and 6 fewer calls |

Whether such a rule helps **on average** is exactly what the next section measures. One example proves nothing.

---

## 4. Measuring what an action is worth (development data only)

### 4.1 Save states

Run the current procedure `H` on a batch of **search** problems. At one randomly chosen decision point per problem (reservoir sampling), save a **complete copy** of the state: all slots, histories, the budget spent, and the decision `H` was about to make (the **incumbent** action `a0`).

### 4.2 Branch from the exact same state

From each saved state, run these branches. Each branch continues with the **same procedure `H`** until the end, and each is repeated with **K = 4 seeds**:

* **incumbent**: do `a0`, then continue
* **alternatives**: pick **2 slots**. With probability 0.7 these are slots that *disagree with the vote* (or are invalid); otherwise slots chosen by rotation (exploration). For each chosen slot try `FULL_REFINE`, `DIRECT_REPAIR`, `RESTART` and `KEEP`.
* **STOP**: vote immediately
* **pairs**: up to 2 ordered pairs on the two chosen slots, e.g. `DIRECT_REPAIR on slot 1, then RESTART on slot 3`, applied unconditionally and then continued

All branches of a state share seed bases (common random numbers), so differences come from the action, not from luck. Every branch has **the same remaining budget**. An action that isn't affordable is recorded as *infeasible*, never as a failure. Untested actions stay *unknown*.

### 4.3 The value of an action

For each branch and seed `k`:

```
score_k = 1{final vote is correct} − λ · (future cost_k / B)          λ = 0.2
Q(S, a) = average of score_k over the K seeds
Δ(S, a) = Q(S, a) − Q(S, a0)          # better (+) or worse (−) than what H would have done
```

*Future cost* is only the cost spent **after** the saved state; the past is identical in every branch, so it cancels.

**Example.** State S, gold = 12, incumbent `a0 = FULL_REFINE slot 2`:

| Branch | seed 1 | seed 2 | seed 3 | seed 4 | accuracy | mean cost/B | Q | Δ |
|---|---|---|---|---|---|---|---|---|
| a0 = FULL_REFINE 2 | ✓ (0.40) | ✗ (0.45) | ✗ (0.40) | ✓ (0.42) | 0.50 | 0.42 | 0.50 − 0.2·0.42 = **0.416** | 0 |
| RESTART 2 | ✓ (0.30) | ✓ (0.32) | ✗ (0.30) | ✓ (0.33) | 0.75 | 0.31 | 0.75 − 0.062 = **0.688** | **+0.272** |
| KEEP 2 | ✗ | ✗ | ✓ | ✗ | 0.25 | 0.30 | **0.19** | **−0.226** |
| STOP | ✗ | ✗ | ✗ | ✗ | 0.00 | 0.00 | **0.00** | **−0.416** |

**All four rows are stored**, including the harmful ones (KEEP, STOP). A result doesn't need a newly correct answer to be useful: saving compute or avoiding harm is also evidence.

This is a comparison *under the current continuation procedure*. It is not a universal causal effect, which is why every change is re-validated end to end later (§6).

### 4.4 Why the vote, not the single trace

With answers `[7, 7, 12]` and gold 12, fixing one `7` changes the winner. With `[7, 7, 7, 7, 12]`, fixing one `7` does **not** change the winner. A trace-level score ("did slot i become correct?") can't see this difference. The vote-level score above can. (A trace-level credit mode exists only as an ablation.)

---

## 5. Two repairs that only work together (the main contribution)

### 5.1 Interaction

For a pair of actions `a` (on slot i) and `b` (on slot j), measure four branches from the same state:

```
Q_∅   : incumbent continuation
Q_a   : force a, then continue
Q_b   : force b, then continue
Q_a→b : force a then b, then continue

Interaction  I      = Q_a→b − Q_a − Q_b + Q_∅
Same on success rates only:  I_acc  (so that saved compute can't masquerade as synergy)
Utility gain          = Q_a→b − max(Q_∅, Q_a, Q_b)
```

### Worked example B: the vote-threshold effect (N = 5, gold = 12)

Current answers: `[7, 7, 7, 7, 12]`. The vote says **7 ✗**.

| Branch | What happens to the vote | accuracy | cost/B | Q |
|---|---|---|---|---|
| Q_∅ (continue) | usually stays 7 | 0.25 | 0.30 | 0.25 − 0.06 = **0.19** |
| Q_a (repair slot 1 → 12) | `[7,12,7,7,12]`: 7 still wins | 0.25 | 0.45 | 0.25 − 0.09 = **0.16** |
| Q_b (repair slot 2 → 12) | `[7,7,12,7,12]`: 7 still wins | 0.25 | 0.45 | **0.16** |
| Q_a→b (repair both) | `[7,12,12,7,12]`: **12 wins** | 0.75 | 0.60 | 0.75 − 0.12 = **0.63** |

```
I     = 0.63 − 0.16 − 0.16 + 0.19 = +0.50       (strong positive interaction)
I_acc = 0.75 − 0.25 − 0.25 + 0.25 = +0.50       (it is a real accuracy effect, not a cost effect)
gain  = 0.63 − max(0.19, 0.16, 0.16) = +0.44
```

Neither repair alone looks useful (both are slightly *worse* than doing nothing), but together they flip the vote. A single-action learner would reject both repairs. The pair measurement finds them.

### 5.2 From evidence to a guarded plan

A pair pattern becomes a **candidate plan** only if, over at least `min_support = 3` different states:

* the mean interaction is `I > 0`, **and** `I > 0` in at least 60% of those states (reproducible),
* `I_acc > 0` (a real voting effect), and
* `Q_a→b` beats the incumbent and both single actions on average (utility gain > 0).

The plan is written only in terms of the fixed feature list:

```text
plan "fix-two-dissenters":
  first guard  (slot i): valid == 1, agrees_plurality == 0, critique in {arithmetic, logic, ...}, answer_changes == 0
  second guard (slot j): valid == 1, agrees_plurality == 0, critique in {...},                  answer_changes == 0
  first op:   DIRECT_REPAIR on i
  continuation guard (checked on j AFTER op 1): agrees_plurality == 0     # j still needs fixing
  second op:  DIRECT_REPAIR on j
```

**Running a plan:**
1. Reserve budget for **both** operations.
2. Run op 1.
3. If the continuation guard fails (for example, repairing slot i already changed the vote so j now agrees), skip op 2 and return to normal scheduling.
4. If op 2 fails, keep op 1's result and charge everything.

A plan is **re-evaluated as the actual guarded plan**: evidence for an unconditional pair does not automatically validate the guarded version.

---

## 6. Outer loop: turning evidence into a better procedure

### 6.1 Data roles (strictly separate)

| Role | Used for | Default (AIME experiment) |
|---|---|---|
| **Search batches** | collecting evidence and ranking proposed changes | 5 rounds × 20 problems (MATH train, Level 5) |
| **Confirmation blocks** | one fresh block per round, used once, only to accept or reject | 5 × 40 problems |
| **Sealed test** | final evaluation only; never enters memory | AIME 2024 + AIME 2025 (60 problems) |

Problems are de-duplicated by text **and** by template (numbers replaced by `#`). All problems sharing a template go to the same split. Development problems that share a template with a test problem are dropped. The code asserts that no sealed-test id ever reaches the development phases.

### 6.2 One round

```
1. solve_and_log   run H on the search batch; save one state per problem
2. compare         branches from each state (§4, §5) → evidence (+, 0, −, unknown)
3. memory          add this round's evidence to M
4. propose         a few patches: add / revise / delete a rule, add a plan, or edit a prompt
5. select          run H and every valid candidate on the search batch → keep the best by mean utility
6. confirm         only if the best candidate beat H on search: run candidate vs H on this round's
                   FRESH confirmation block, same problems and same seeds (paired)
7. accept?         accept only if BOTH tests pass (below); otherwise keep H
```

### 6.3 Who proposes changes (the restricted editor)

* **Compiler** (deterministic). It groups evidence into coarse bins:
  * slot: valid / agrees with vote / critique ∈ {unknown, no error, error} / answer changes ∈ {0, 1, 2+}
  * budget: low (<0.33), mid, high (≥0.66)
  * vote: split, majority, unanimous

  Then:
  * **rule:** an action whose mean `Δ ≥ 0.02` over ≥3 states, with at least as many positive as negative results, becomes `IF <bin conditions> THEN <action>`
  * **STOP rule:** the same test on STOP evidence, with conditions on vote agreement and budget
  * **plan:** a pair pattern that passes §5.2
  * **delete:** a rule that has fired in ≥3 states where the best tested alternative was better by > 0.02 in at least half of them
* **LLM editor**: the *same frozen model* sees the current procedure, the feature list, and a binned evidence summary. It sees **no** problems, answers, or ids. It must return one JSON patch naming its supporting and contradicting evidence ids.
* **Every patch is validated.** Only known features, operators, and actions are allowed, and the rule, plan, and prompt budgets are respected (12 rules, 4 plans, 1200 prompt characters). Prompt edits are **rejected** if they contain development problem text, development gold answers, a list of numbers, or lookup language ("the answer to problem …").

### 6.4 The acceptance test

Per confirmation question, average the seeds first, then take `d_q = candidate − H`.

| Test | Rule | Default |
|---|---|---|
| Utility improvement | one-sided sign-flip permutation test on mean `d_q(utility)`: p < α_t | α_t = 0.05 / 5 rounds = **0.01** |
| Accuracy non-inferiority | one-sided bootstrap lower bound of mean `d_q(accuracy)` > −margin | margin = **0.02** |

Each block is used for **one** decision only.

**Example.** On 40 confirmation questions the candidate has mean utility difference +0.035, sign-flip p = 0.004 < 0.01 ✓, and accuracy difference +0.025 with a 99% lower bound of −0.010 > −0.02 ✓. The candidate is **accepted** and becomes H for the next round.

With p = 0.03 it would be **rejected** (0.03 > 0.01), even though it looks better. The log then reports the **block size needed** to detect a 1-, 2-, or 5-point gain, given how often the two procedures disagreed.

### 6.5 Output of development

`final_procedure.json` is the frozen procedure, and `lineage.json` records every accepted or rejected edit with its evidence. H is then evaluated **once** on the sealed test against the baselines.

---

## 7. Evaluation and baselines

All methods use the same model, extraction, vote, and the **same cap B**.

| Method | Description |
|---|---|
| `h0` | the paper's fixed refinement schedule |
| `maj_matched` | majority voting with **compute-matched** sampling: keep drawing samples while B allows |
| `heuristic` | fixed hand-written adaptive rules (not learned) |
| `final` | the evolved procedure |

Reported: accuracy and solved counts (with seed spread), utility, calls / tokens / TFLOPs per problem, extraction failures, fraction of correct slots, action frequencies, per-operation wrong→right and right→wrong rates, vote recoveries and regressions, and total **development** compute (`dev_ledger.json`).

**Ablations** (one conf overlay each, in `configs/ablations/`):

| Overlay | Tests |
|---|---|
| `primitive_only.conf` | pairs/plans vs. primitive actions only |
| `random_plans.conf`, `greedy_vote_plans.conf` | interaction-selected plans vs. random or immediate-vote-greedy plans |
| `trace_credit.conf` | vote credit vs. single-trace credit |
| `trajectory_evidence.conf` | same-state experiments vs. plain trajectory summaries |
| `success_only_memory.conf` | signed memory vs. success-only memory |
| `prompts_only.conf` | evolving rules vs. evolving prompts only |
| `no_keep.conf`, `no_restart.conf` | without KEEP / RESTART |
| `reset_procedure.conf` | persistent procedure vs. restart from H0 each round (same evidence) |
| `frozen_after2.conf` | stop adapting after round 2 vs. keep adapting |

---

## 8. Default settings

| Setting | Pilot default | AIME experiment |
|---|---|---|
| Model | Qwen2.5-1.5B-Instruct | Qwen2.5-Math-7B-Instruct |
| N slots | 4 | **8** |
| H0 depth / budget cycles | 2 / 2 | 2 / 2 |
| Cost measure, λ | TFLOPs, 0.2 | TFLOPs, 0.2 |
| Seeds per tested action K | 4 | 4 |
| States per problem / slots per state / pairs per state | 1 / 2 / 2 | 1 / 2 / 2 |
| Rounds, search / confirmation size | 5, 20 / 40 | 5, 20 / 40 |
| α total, non-inferiority margin | 0.05, 0.02 | 0.05, 0.02 |
| Decoding | init/rewrite T=0.7, critique/correct greedy | same; max 1024 new tokens (4k context) |

---

## 9. Running it, continuous saving, and resume

```bash
HF_HUB_OFFLINE=1 nohup bash run_rsi_experiment.sh configs/rsi_qwen_7b.conf,configs/rsi_qwen_7b_aime.conf \
    > logs/aime_N8_qwen7b_experiment.out 2>&1 &
```

* Results are saved continuously under `$SCRATCH/procedure_rsi_results/<model>/`: `final_procedure.json` (the model's frozen procedure), `development/` (everything used to build it) and one folder per test dataset (`aime24/`, `aime25/`, `math500/`, …).
* Finished solves are written immediately. In-flight solves are snapshotted every 60 s. Each phase's output is saved once. The branch plan of each round is fixed on disk.
* When a job is killed, **run the same command again**. It continues where it stopped, without redoing finished work or double-counting evidence.

---

## 10. Where each step lives in the code

| Step | File |
|---|---|
| model calls + real token counts | `src/rsi/backends.py`, `src/model_wrapper.py` |
| budget, reservation | `src/rsi/budget.py`, `src/rsi/solver.py` |
| slots, features | `src/rsi/state.py` |
| prompts, critique category | `src/rsi/prompts.py` |
| rules, plans, H0, controller | `src/rsi/procedure.py` |
| solving one problem | `src/rsi/solver.py`, batched by `src/rsi/scheduler.py` |
| extraction, vote, grader | `src/rsi/voting.py` |
| same-state experiments, Q / Δ / I | `src/rsi/counterfactual.py` |
| evidence memory | `src/rsi/memory.py` |
| editor (compiler + LLM + validation) | `src/rsi/editor.py` |
| splits | `src/rsi/splits.py` |
| acceptance tests | `src/rsi/stats.py` |
| outer loop | `src/rsi/outer_loop.py` |
| continuous saving / resume | `src/rsi/checkpoint.py` |
| baselines, metrics | `src/rsi/runner.py` |
| CLI | `run_procedure_rsi.py`, `run_procedure_rsi.sh`, `run_rsi_experiment.sh` |

---

## 11. What this method does not claim

* Action values are measured **under the current procedure**. They are not universal causal effects, which is why every change is re-validated on fresh data.
* `KEEP` can lock in a wrong answer and `RESTART` can throw away a good one. Only the measured, confirmed average effect justifies using them.
* Critique categories are the model's claims. A correct final answer is not a verified proof.
* Plans cover **two-slot** interactions only, not larger groups of slots.
* The method can only reallocate compute. It helps only where the model can actually repair or resample its way to the right answer.
* Development is expensive: each saved state spawns about 12 branches × 4 seeds. The development compute is reported separately and must be weighed against any gain.
