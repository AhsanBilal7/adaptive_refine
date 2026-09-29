"""
Training-free Procedure RSI for breadth–depth self-refinement.

Implements ``updated_methodology.md``: an adaptive inner loop (one vote per
slot, FULL_REFINE / DIRECT_REPAIR / RESTART / KEEP / STOP / guarded PAIR
plans under a hard compute cap) and an outer loop that measures alternative
actions from copied states, stores signed evidence, and evolves the
procedure through a restricted editor with paired confirmation.  All model
weights stay frozen.
"""
