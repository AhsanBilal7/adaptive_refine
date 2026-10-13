#!/usr/bin/env python3
"""Draw the two animations of the README (``assets/*.gif``).

  replay_decoding.gif  replay decoding of one revision z = F_x(y): the current solution y
                       is the draft, W draft tokens are verified in one forward pass, and
                       greedy decoding takes over after the first disagreement; a revision
                       that reproduces y exactly is a fixed point
  arr_allocation.gif   fixed-point replacement on N = 4 slots under a budget B: a root is
                       revised until it reaches a fixed point, its slot then samples a new
                       root, and the answer is the majority vote over one answer per root

    python scripts/make_readme_assets.py        # needs matplotlib and pillow
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

# Okabe-Ito palette (color-blind safe), muted fills for token boxes.
INK, MUTED, RULE = "#1b1f24", "#5b6470", "#c9ced6"
BLUE, GREEN, VERMIL, ORANGE, PURPLE, SKY = "#0072B2", "#009E73", "#D55E00", "#E69F00", "#8E5BA8", "#56B4E9"
FILL = dict(wait=("#f4f5f7", RULE, INK), check=("#fff4d6", ORANGE, INK), ok=("#e3f4ee", GREEN, INK),
            bad=("#fbe6dc", VERMIL, VERMIL), drop=("#f4f5f7", "#e1e4e8", "#a8b0ba"), new=("#e2eef8", BLUE, INK))
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12})


def rbox(ax, x, y, w, h, face, edge, lw=1.3, r=0.10, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}",
                                fc=face, ec=edge, lw=lw, zorder=z))


def header(ax, x0, x1, y, title, subtitle):
    ax.text(x0, y, title, fontsize=17, color=INK, va="center", ha="left")
    ax.text(x0, y - 0.42, subtitle, fontsize=11.5, color=MUTED, va="center", ha="left", style="italic")
    ax.plot([x0, x1], [y - 0.72, y - 0.72], color=RULE, lw=0.9)


# --------------------------------------------------------------------------- replay decoding
DRAFT = ["x", "=", "3", "+", "4", "=", "8", ",", "so", "y", "=", "8", "EOS"]
GREEDY = ["x", "=", "3", "+", "4", "=", "7", ",", "so", "y", "=", "7", "EOS"]
W = 4


def tok_label(t):
    return "eos" if t == "EOS" else t


def replay_states(draft, target):
    """One state per serial forward pass (plus the start): token status and a caption."""
    st = [dict(out=[], dstat=["wait"] * len(draft), passes=0, block=None, ell=0,
               cap=r"Draft: the current solution $y$.  Each forward pass verifies up to $W=4$ draft tokens.")]
    out, dstat, t, passes, on, ell = [], ["wait"] * len(draft), 0, 0, True, 0
    while (not out or out[-1][0] != "EOS") and len(out) < len(target):
        passes += 1
        if on:
            blk = list(range(t, min(t + W, len(draft))))
            st.append(dict(out=list(out), dstat=[("check" if i in blk else s) for i, s in enumerate(dstat)],
                           passes=passes - 1, block=blk, ell=ell,
                           cap=rf"Pass {passes}: verify $y_{{{blk[0] + 1}:{blk[-1] + 1}}}$ in one parallel "
                               r"forward pass against $\arg\max_v\,p_\theta(v\mid u,\,y_{<j})$."))
            acc = 0
            for i in blk:
                if draft[i] != target[i]:
                    break
                acc += 1
            for i in blk[:acc]:
                dstat[i] = "ok"
                out.append((draft[i], "ok"))
            t += acc
            ell += acc
            if acc < len(blk):
                i = blk[acc]
                dstat[i] = "bad"
                for j in blk[acc + 1:]:
                    dstat[j] = "drop"
                out.append((target[i], "new"))
                t += 1
                on = False
                cap = (rf"Pass {passes}: first disagreement at position {i + 1} "
                       rf"($y_{{{i + 1}}}=${draft[i]}, $\hat z_{{{i + 1}}}=${target[i]}): keep $\hat z$, "
                       r"discard the rest of the block.")
            else:
                cap = rf"Pass {passes}: accepted prefix grows to $\ell={ell}$ tokens."
        else:
            out.append((target[t], "new"))
            if t < len(dstat) and dstat[t] == "wait":
                dstat[t] = "drop"
            t += 1
            cap = rf"Pass {passes}: the output left the draft; ordinary greedy decoding, one token per pass."
        st.append(dict(out=list(out), dstat=list(dstat), passes=passes, block=None, ell=ell, cap=cap))
    return st


def draw_replay(ax, s, draft, scene, verdict):
    ax.clear()
    ax.set_xlim(-2.9, 20.4)
    ax.set_ylim(-1.35, 5.25)
    ax.axis("off")
    header(ax, -2.9, 20.4, 4.85, r"Replay decoding of a revision $z = F_x(y)$",
           scene + r"   ·   block size $W = 4$   ·   output identical to greedy decoding")
    ax.text(-2.75, 3.0, r"draft $y$", fontsize=13, color=MUTED, va="center")
    ax.text(-2.75, 1.45, r"output $z$", fontsize=13, color=MUTED, va="center")
    for i, tk in enumerate(draft):
        f, e, c = FILL[s["dstat"][i]]
        rbox(ax, i * 1.0, 2.62, 0.88, 0.76, f, e)
        ax.text(i + 0.44, 3.0, tok_label(tk), ha="center", va="center", fontsize=13.5 if tk != "EOS" else 11.5,
                color=c, zorder=3, style="italic" if tk == "EOS" else "normal")
        if s["dstat"][i] == "drop":
            ax.plot([i + 0.14, i + 0.74], [3.0, 3.0], color="#a8b0ba", lw=1.1, zorder=4)
        ax.text(i + 0.44, 3.55, str(i + 1), ha="center", va="center", fontsize=9, color="#9aa3ad")
    if s["block"]:
        b = s["block"]
        ax.add_patch(FancyBboxPatch((b[0] - 0.07, 2.53), len(b) - 0.11 + 0.14, 0.94,
                                    boxstyle="round,pad=0.0,rounding_size=0.14", fc="none", ec=ORANGE, lw=2.0,
                                    ls=(0, (4, 2)), zorder=5))
        ax.annotate("", xy=(b[0] + len(b) / 2 - 0.06, 1.9), xytext=(b[0] + len(b) / 2 - 0.06, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6), zorder=5)
        ax.text(b[0] + len(b) / 2 - 0.06, 2.2, "  1 pass", color="#9a6400", fontsize=11, va="center", ha="left")
    for i, (tk, kind) in enumerate(s["out"]):
        f, e, c = FILL["ok" if kind == "ok" else "new"]
        rbox(ax, i * 1.0, 1.07, 0.88, 0.76, f, e)
        ax.text(i + 0.44, 1.45, tok_label(tk), ha="center", va="center", fontsize=13.5 if tk != "EOS" else 11.5,
                color=c, zorder=3, style="italic" if tk == "EOS" else "normal")
    ax.text(-2.9, 0.3, s["cap"], fontsize=12.5, color=INK, va="center")
    # pass counter: replay vs. ordinary greedy decoding
    greedy = len(GREEDY) + 1
    x0, wbar = 16.0, 3.2
    ax.text(x0 - 1.7, 3.55, "serial forward passes", fontsize=11, color=MUTED, va="center")
    for k, (lab, val, col) in enumerate([("replay", s["passes"], BLUE), ("greedy", greedy, "#9aa3ad")]):
        y = 2.95 - k * 0.62
        ax.add_patch(Rectangle((x0, y - 0.17), wbar, 0.34, fc="#f4f5f7", ec="none"))
        ax.add_patch(Rectangle((x0, y - 0.17), wbar * min(val, greedy) / greedy, 0.34, fc=col, ec="none"))
        ax.text(x0 - 0.12, y, lab, fontsize=11, color=INK, va="center", ha="right")
        ax.text(x0 + wbar + 0.1, y, str(val), fontsize=12, color=INK, va="center")
    ax.text(x0 - 1.7, 1.55, r"$S_{\mathrm{replay}} \approx 1+\lceil \ell/W\rceil+(|z|-\ell)$", fontsize=12,
            color=MUTED, va="center")
    ax.text(x0 - 1.7, 1.05, rf"$\ell = {s['ell']}$ accepted draft tokens", fontsize=12, color=MUTED, va="center")
    if verdict:
        good = "fixed point" in verdict
        rbox(ax, 7.4, -1.22, 12.9, 0.68, "#e3f4ee" if good else "#e2eef8", GREEN if good else BLUE, lw=1.4)
        ax.text(13.85, -0.88, verdict, fontsize=13.5, color=GREEN if good else BLUE, ha="center", va="center",
                zorder=3)


def make_replay_gif(path):
    seq = []
    a = replay_states(DRAFT, GREEDY)
    for k, s in enumerate(a):
        seq += [(s, DRAFT, "Round 1: $y$ contains an arithmetic error", "")] * (3 if k in (0, len(a) - 1) else 2)
    seq += [(a[-1], DRAFT, "Round 1: $y$ contains an arithmetic error",
             r"$z \neq y$: the root changed, so ARR revises it again")] * 5
    b = replay_states(GREEDY, GREEDY)
    for k, s in enumerate(b):
        seq += [(s, GREEDY, "Round 2: $y$ is the corrected solution", "")] * (3 if k in (0, len(b) - 1) else 2)
    seq += [(b[-1], GREEDY, "Round 2: $y$ is the corrected solution",
             r"$z = y$  $\Rightarrow$  $F_x^{\,k}(y)=y$ for all $k$: fixed point, stop revising")] * 8
    fig, ax = plt.subplots(figsize=(13.6, 4.4), dpi=96)
    fig.subplots_adjust(0.012, 0.01, 0.988, 0.99)
    anim = FuncAnimation(fig, lambda k: draw_replay(ax, *seq[k]), frames=len(seq))
    anim.save(path, writer=PillowWriter(fps=2))
    plt.close(fig)


# --------------------------------------------------------------------------- allocation under a budget
# One row per slot (N = 4).  Operations: (kind, answer); S new root (sample from sigma(x)),
# R revision that changes the root, F revision that reproduces the root (fixed point).
LANES = [
    [("S", "5"), ("R", "7"), ("F", "7"), ("S", "7"), ("F", "7"), ("S", "9"), ("F", "9")],
    [("S", "7"), ("F", "7"), ("S", "7"), ("R", "7"), ("F", "7"), ("S", "7"), ("F", "7")],
    [("S", "3"), ("R", "6"), ("R", "7"), ("F", "7"), ("S", "12"), ("F", "12"), ("S", "7")],
    [("S", "7"), ("F", "7"), ("S", "5"), ("R", "7"), ("F", "7"), ("S", "7"), ("R", "7")],
]
COST = {"S": 4.0, "R": 1.5, "F": 1.0}       # a fixed point is the cheapest call: replay accepts the whole draft
STYLE = dict(S=("#f2eaf7", PURPLE, r"$y\sim\sigma(x)$"), R=("#e2eef8", BLUE, r"$F_x(y)\neq y$"),
             F=("#e3f4ee", GREEN, r"$F_x(y)=y$"))
CAPTIONS = [
    r"Sample $N=4$ independent roots $y_r \sim \sigma(x)$, one per slot.",
    r"Revise every root with the same stationary greedy operator $F_x$ (replay decoding).",
    r"A revision that reproduces its root is a fixed point: the root keeps its answer as its vote.",
    r"The released compute funds a new independent root in the same slot.",
    r"Roots that still change keep being revised; depth adapts per root.",
    r"No depth limit, threshold, critic, or learned controller is needed.",
    r"Continue while a complete new sample still fits in the budget $B$.",
    r"Budget $B$ exhausted.",
]


def final_votes():
    """One final answer per root: its fixed point, or its last solution if it never converged."""
    votes = []
    for lane in LANES:
        cur = None
        for kind, ans in lane:
            if kind == "S" and cur is not None and not cur[1]:
                votes.append(cur[0])
            if kind == "F":
                votes.append(ans)
                cur = (ans, True)
            else:
                cur = (ans, False)
        if cur is not None and not cur[1]:
            votes.append(cur[0])
    return votes


def make_allocation_gif(path):
    steps = len(LANES[0])
    budget = sum(COST[k] for lane in LANES for k, _ in lane)
    votes = final_votes()
    tally = sorted({a: votes.count(a) for a in votes}.items(), key=lambda kv: (-kv[1], kv[0]))
    fig, ax = plt.subplots(figsize=(12.0, 5.0), dpi=96)
    fig.subplots_adjust(0.012, 0.01, 0.988, 0.99)
    frames = list(range(steps + 1)) + [steps + 1] * 9

    def update(t):
        ax.clear()
        ax.set_xlim(-1.75, 15.2)
        ax.set_ylim(-2.0, 5.85)
        ax.axis("off")
        header(ax, -1.75, 15.2, 5.45, r"Fixed-point replacement under a compute budget $B$",
               r"$N = 4$ slots   ·   each box is one model call   ·   depth and breadth adapt per question")
        for i, kind in enumerate("SRF"):
            f, e, lab = STYLE[kind]
            x = 1.1 + i * 3.35
            rbox(ax, x, 4.22, 0.42, 0.28, f, e, lw=1.2, r=0.06)
            ax.text(x + 0.55, 4.36, {"S": "new root ", "R": "revision changed ", "F": "fixed point "}[kind] + lab,
                    fontsize=11.5, color=INK, va="center")
        shown = min(t, steps)
        for r, lane in enumerate(LANES):
            y = 3.25 - r * 0.92
            ax.text(-1.65, y + 0.3, rf"slot {r + 1}", fontsize=12.5, color=MUTED, va="center")
            for k, (kind, ans) in enumerate(lane[:shown]):
                f, e, _ = STYLE[kind]
                x = k * 1.38
                rbox(ax, x, y, 1.22, 0.6, f, e, lw=1.4 if kind != "F" else 1.8)
                lab = {"S": rf"$\sigma$: {ans}", "R": rf"$F_x$: {ans}", "F": rf"{ans}  $\checkmark$"}[kind]
                ax.text(x + 0.61, y + 0.3, lab, fontsize=12.5, color=INK, ha="center", va="center",
                        weight="bold" if kind == "F" else "normal", zorder=3)
                if kind == "F" and k + 1 < shown:
                    ax.plot([x + 1.30, x + 1.30], [y - 0.05, y + 0.65], color=GREEN, lw=1.4, ls=(0, (1, 2)))
        spent = sum(COST[k] for lane in LANES for k, _ in lane[:shown])
        ax.text(-1.65, -0.42, r"$\mathcal{C}_\pi(x)$", fontsize=13, color=MUTED, va="center")
        ax.add_patch(Rectangle((0, -0.58), 9.5, 0.32, fc="#f4f5f7", ec=RULE, lw=0.8))
        ax.add_patch(Rectangle((0, -0.58), 9.5 * spent / budget, 0.32, fc=SKY, ec="none", alpha=0.9))
        ax.plot([9.5, 9.5], [-0.68, -0.16], color=VERMIL, lw=1.6)
        ax.text(9.5, -0.02, r"$B$", color=VERMIL, fontsize=13, ha="center", va="bottom")
        ax.text(9.7, -0.42, f"{100 * spent / budget:.0f}% of budget", fontsize=12, color=INK, va="center")
        if t <= steps:
            ax.text(-1.75, -1.4, CAPTIONS[min(t, len(CAPTIONS) - 1)], fontsize=13, color=INK, va="center")
        else:
            win = tally[0][0]
            ax.text(-1.75, -1.4, rf"Majority vote over one final answer per root ({len(votes)} roots)  "
                                 rf"$\Rightarrow$  $\hat a = {win}$", fontsize=13.5, color=GREEN, va="center")
            x0, y0, hmax = 11.3, 0.35, 2.9
            ax.text(x0 - 0.1, 3.85, "votes", fontsize=12, color=MUTED, va="center")
            top = tally[0][1]
            for i, (a, n) in enumerate(tally):
                x = x0 + i * 1.15
                h = hmax * n / top
                ax.add_patch(Rectangle((x, y0), 0.8, h, fc=GREEN if a == win else "#c9ced6", ec="none"))
                ax.text(x + 0.4, y0 + h + 0.12, str(n), ha="center", va="bottom", fontsize=12, color=INK)
                ax.text(x + 0.4, y0 - 0.2, a, ha="center", va="top", fontsize=12.5, color=INK)

    anim = FuncAnimation(fig, update, frames=frames)
    anim.save(path, writer=PillowWriter(fps=1))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_replay_gif(os.path.join(OUT, "replay_decoding.gif"))
    make_allocation_gif(os.path.join(OUT, "arr_allocation.gif"))
    print("wrote", OUT)
