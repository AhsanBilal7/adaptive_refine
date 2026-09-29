"""
Adaptive Delta Refinement (updated_methodology.md).

N independent structured solutions with dependency ids → per rollout, the
existing critic returns KEEP / FIX(step, error) / REBUILD(reason) → on FIX only
the dependency closure A(j) = {j} ∪ Desc(j) is regenerated and every other step
is kept verbatim → plurality vote.  Training-free, per-question, no dev set.
"""
