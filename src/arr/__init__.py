"""Adaptive Replay Refinement (ARR).

ARR samples N independent solutions ("roots") and revises each one with a
single fixed greedy revision F_x(y) = Greedy(rho(x, y)).  Replay decoding uses
the current solution as a draft: blocks of W draft tokens are checked in one
parallel forward pass, and ordinary greedy decoding takes over after the first
disagreement.  A revision that reproduces its input exactly is a fixed point:
further revision is redundant, so the root keeps its answer and its compute
goes to a new independent root.  The prediction is the majority vote over one
answer per root.  ARR is training-free and needs no critic, threshold, or
learned controller.

Modules:
  config     settings and their environment / command-line overrides
  engine     batched decoder with sample, greedy, replay and PLD modes
  ledger     per-question FLOP accounting under the budget B
  methods    controllers of ARR and the baselines
  conma      the ConMA baseline
  prompts    prompts of every method
  tasks      benchmarks: prompts, answer extraction and scoring
  runner     experiment driver and metrics
  sanity     CPU checks of the decoder and the graders
"""
