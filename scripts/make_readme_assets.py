#!/usr/bin/env python3
"""Draw the two animations of the README (``assets/*.gif``).

  replay_decoding.gif  replay decoding of one revision: the current solution is the
                       draft, blocks of W draft tokens are checked in one parallel pass,
                       and greedy decoding takes over after the first disagreement; a
                       revision that reproduces the draft is a fixed point
  arr_allocation.gif   ARR on N = 4 roots: revise each root until it reaches a fixed
                       point, then spend the released compute on a new root, until the
                       budget B ends; the answer is the majority vote

    python scripts/make_readme_assets.py        # needs matplotlib and pillow
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
INK = "#1f2328"
MUTED = "#6e7781"
GRAY = "#d0d7de"
GREEN = "#2da44e"
RED = "#cf222e"
BLUE = "#0969da"
PURPLE = "#8250df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def box(ax, x, y, w, h, face, edge=None, text="", color=INK, size=11, alpha=1.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=face, ec=edge or face, lw=1.4, alpha=alpha))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size,
                weight=weight, family="DejaVu Sans Mono" if len(text) < 12 else "DejaVu Sans")


# --------------------------------------------------------------------------- replay decoding
DRAFT = ["x", "=", "3", "+", "4", "=", "8", ",", "so", "y", "=", "8", "<eos>"]
GREEDY = ["x", "=", "3", "+", "4", "=", "7", ",", "so", "y", "=", "7", "<eos>"]
W = 4


def replay_frames(draft, target):
    """States after each serial pass: (out tokens with status, draft status, pass count, note)."""
    frames = [dict(out=[], dstat=["wait"] * len(draft), passes=0, block=None,
                   note="The current solution y is the draft. Replay checks W = 4 draft tokens per forward pass.")]
    out, dstat, t, passes, on = [], ["wait"] * len(draft), 0, 0, True
    while (not out or out[-1][0] != "<eos>") and len(out) < len(target):
        passes += 1
        if on:
            blk = list(range(t, min(t + W, len(draft))))
            frames.append(dict(out=list(out), dstat=[("check" if i in blk else s) for i, s in enumerate(dstat)],
                               passes=passes - 1, block=blk, note=f"Pass {passes}: check draft tokens "
                               f"{blk[0] + 1}-{blk[-1] + 1} in parallel."))
            acc = 0
            for i in blk:
                if draft[i] == target[i]:
                    acc += 1
                else:
                    break
            for i in blk[:acc]:
                dstat[i] = "ok"
                out.append((draft[i], "ok"))
            t += acc
            if acc < len(blk):
                i = blk[acc]
                dstat[i] = "bad"
                for j in blk[acc + 1:]:
                    dstat[j] = "drop"
                out.append((target[i], "new"))
                t += 1
                on = False
                note = (f"Pass {passes}: first disagreement at token {i + 1} ('{draft[i]}' vs '{target[i]}'). "
                        "Keep the model's token, drop the rest of the block.")
            else:
                note = (f"Pass {passes}: all {acc} draft tokens match the greedy output and are accepted."
                        if acc > 1 else f"Pass {passes}: the last draft token matches and is accepted.")
        else:
            out.append((target[t], "new"))
            if t < len(dstat) and dstat[t] == "wait":
                dstat[t] = "drop"
            t += 1
            note = f"Pass {passes}: ordinary greedy decoding after the disagreement."
        frames.append(dict(out=list(out), dstat=list(dstat), passes=passes, block=None, note=note))
    return frames


def draw_replay(ax, st, title, ordinary, final_note):
    ax.clear()
    ax.set_xlim(-2.4, 14.2)
    ax.set_ylim(-0.4, 4.6)
    ax.axis("off")
    ax.text(-2.4, 4.3, title, fontsize=13, weight="bold", color=INK, va="center")
    ax.text(-2.3, 2.95, "draft  y", ha="left", va="center", color=MUTED, fontsize=11)
    ax.text(-2.3, 1.55, "revision", ha="left", va="center", color=MUTED, fontsize=11)
    face = dict(wait=("#f6f8fa", GRAY, INK), check=("#fff8c5", "#d4a72c", INK), ok=("#dafbe1", GREEN, INK),
                bad=("#ffebe9", RED, RED), drop=("#f6f8fa", GRAY, "#afb8c1"))
    for i, tok in enumerate(st["draft"]):
        f, e, c = face[st["dstat"][i]]
        box(ax, i * 1.0, 2.6, 0.9, 0.7, f, e, tok, c, size=10)
        if st["dstat"][i] == "drop":
            ax.plot([i * 1.0 + 0.12, i * 1.0 + 0.78], [2.95, 2.95], color="#afb8c1", lw=1.2)
    if st["block"]:
        b = st["block"]
        ax.add_patch(FancyBboxPatch((b[0] - 0.08, 2.5), len(b) - 0.04 + 0.06, 0.9,
                                    boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec="#d4a72c",
                                    lw=2.0, ls="--"))
        ax.text(b[0] + len(b) / 2 - 0.05, 3.62, "one parallel pass", ha="center", color="#9a6700", fontsize=10)
    for i, (tok, s) in enumerate(st["out"]):
        f, e = ("#dafbe1", GREEN) if s == "ok" else ("#ddf4ff", BLUE)
        box(ax, i * 1.0, 1.2, 0.9, 0.7, f, e, tok, INK, size=10)
    ax.text(-2.4, 0.45, st["note"], fontsize=10.5, color=INK, va="center")
    ax.text(14.2, 4.3, f"serial passes: {st['passes']}   (greedy: {ordinary})", ha="right", va="center",
            fontsize=11, color=BLUE, weight="bold")
    if final_note:
        ax.text(14.2, -0.15, final_note, ha="right", va="center", fontsize=11, weight="bold",
                color=GREEN if "fixed point" in final_note else BLUE)


def make_replay_gif(path):
    a = [(f, "Revision of y: the draft has an error") for f in replay_frames(DRAFT, GREEDY)]
    b = [(f, "Revision of the corrected solution") for f in replay_frames(GREEDY, GREEDY)]
    seq = []
    for k, (f, t) in enumerate(a):
        f = dict(f, draft=DRAFT)
        seq += [(f, t, len(GREEDY), "")] * (2 if 0 < k < len(a) - 1 else 3)
    seq += [(dict(a[-1][0], draft=DRAFT), a[-1][1], len(GREEDY), "z differs from y: revise again")] * 5
    for k, (f, t) in enumerate(b):
        f = dict(f, draft=GREEDY)
        seq += [(f, t, len(GREEDY), "")] * (2 if 0 < k < len(b) - 1 else 3)
    seq += [(dict(b[-1][0], draft=GREEDY), b[-1][1], len(GREEDY),
             "z = y: fixed point, further revision is redundant")] * 7
    fig, ax = plt.subplots(figsize=(10.5, 3.6), dpi=100)
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.98)

    def update(k):
        f, t, ordinary, note = seq[k]
        draw_replay(ax, f, t, ordinary, note)

    anim = FuncAnimation(fig, update, frames=len(seq))
    anim.save(path, writer=PillowWriter(fps=2))
    plt.close(fig)


# --------------------------------------------------------------------------- allocation under a budget
# One row per execution slot (N = 4).  Each operation: (kind, answer); kinds:
#   S new root (sample), R revision that changes the solution, F revision that is a fixed point.
LANES = [
    [("S", "5"), ("R", "7"), ("F", "7"), ("S", "7"), ("F", "7"), ("S", "9"), ("F", "9")],
    [("S", "7"), ("F", "7"), ("S", "7"), ("R", "7"), ("F", "7"), ("S", "7"), ("F", "7")],
    [("S", "3"), ("R", "6"), ("R", "7"), ("F", "7"), ("S", "12"), ("F", "12"), ("S", "7")],
    [("S", "7"), ("F", "7"), ("S", "5"), ("R", "7"), ("F", "7"), ("S", "7"), ("R", "7")],
]
COST = {"S": 4.0, "R": 1.5, "F": 1.0}       # replay makes revisions (and fixed points) cheap


def make_allocation_gif(path):
    steps = len(LANES[0])
    budget = sum(COST[k] for lane in LANES for k, _ in lane)
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=100)
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.98)
    colors = dict(S=("#fbefff", PURPLE), R=("#ddf4ff", BLUE), F=("#dafbe1", GREEN))
    frames = list(range(steps + 1)) + [steps + 1] * 8

    def votes_upto(t):
        """Final answer of each root that has one: retired roots and the current root of each slot."""
        v = []
        for lane in LANES:
            cur = None
            for kind, ans in lane[:t]:
                if kind == "S":
                    if cur is not None and cur[0] == "open":
                        v.append(cur[1])
                    cur = ("open", ans)
                elif kind == "R":
                    cur = ("open", ans)
                else:
                    v.append(ans)
                    cur = ("done", ans)
            if cur is not None and cur[0] == "open":
                v.append(cur[1])
        return v

    def update(t):
        ax.clear()
        ax.set_xlim(-1.9, 12.2)
        ax.set_ylim(-1.9, 5.0)
        ax.axis("off")
        shown = min(t, steps)
        ax.text(-1.9, 4.7, "ARR under a compute budget B  (N = 4 roots)", fontsize=13, weight="bold", color=INK)
        for r, lane in enumerate(LANES):
            y = 3.4 - r * 0.95
            ax.text(-1.85, y + 0.32, f"slot {r + 1}", color=MUTED, fontsize=10.5, va="center")
            for k, (kind, ans) in enumerate(lane[:shown]):
                f, e = colors[kind]
                x = k * 1.42
                label = {"S": f"new {ans}", "R": f"rev {ans}", "F": f"{ans}  ✓"}[kind]
                box(ax, x, y, 1.25, 0.64, f, e, label, INK, size=10, weight="bold" if kind == "F" else "normal")
                if kind == "F" and k + 1 < shown:
                    ax.plot([x + 1.335, x + 1.335], [y - 0.06, y + 0.7], color=GREEN, lw=1.6, ls=":")
        spent = sum(COST[k] for lane in LANES for k, _ in lane[:shown])
        ax.add_patch(FancyBboxPatch((0, -0.6), 9.8, 0.32, boxstyle="round,pad=0.01,rounding_size=0.1",
                                    fc="#f6f8fa", ec=GRAY))
        ax.add_patch(FancyBboxPatch((0, -0.6), 9.8 * spent / budget, 0.32,
                                    boxstyle="round,pad=0.01,rounding_size=0.1", fc=BLUE, ec=BLUE, alpha=0.75))
        ax.text(-1.85, -0.44, "budget B", color=MUTED, fontsize=10.5, va="center")
        ax.text(10.0, -0.44, f"{100 * spent / budget:.0f}%", color=INK, fontsize=10.5, va="center")
        if t <= steps:
            notes = ["Sample N = 4 independent roots.",
                     "Revise every root with the same greedy revision F_x (replay decoding).",
                     "A revision that reproduces its root is a fixed point (✓): the root keeps its answer.",
                     "Its compute goes to a new independent root in the same slot.",
                     "Roots that still change keep being revised.",
                     "Depth and breadth adapt per root; nothing is tuned.",
                     "Continue until the budget cannot pay for another step.",
                     "Budget spent."]
            ax.text(-1.9, -1.25, notes[min(t, len(notes) - 1)], fontsize=11, color=INK)
        else:
            v = votes_upto(steps)
            tally = {a: v.count(a) for a in dict.fromkeys(v)}
            win = max(tally, key=tally.get)
            text = "   ".join(f"{a}: {n}" for a, n in sorted(tally.items(), key=lambda kv: -kv[1]))
            ax.text(-1.9, -1.25, f"Majority vote over one answer per root  ({text})  →  answer {win}",
                    fontsize=11.5, color=GREEN, weight="bold")
        handles = [("new root (sample)", PURPLE), ("revision changed the root", BLUE), ("fixed point", GREEN)]
        for i, (lab, c) in enumerate(handles):
            ax.add_patch(FancyBboxPatch((1.4 + i * 3.4, 4.12), 0.28, 0.22, boxstyle="round,pad=0.01", fc=c, ec=c))
            ax.text(1.8 + i * 3.4, 4.23, lab, fontsize=10, color=INK, va="center")

    anim = FuncAnimation(fig, update, frames=frames)
    anim.save(path, writer=PillowWriter(fps=1))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_replay_gif(os.path.join(OUT, "replay_decoding.gif"))
    make_allocation_gif(os.path.join(OUT, "arr_allocation.gif"))
    print("wrote", OUT)
