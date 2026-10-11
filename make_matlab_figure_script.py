#!/usr/bin/env python3
"""Write the MATLAB scripts of the paper figures.

Inputs: ``results/all_results.csv`` (from ``collect_results.py``) and
``results/analysis.json`` (from ``arr_analysis.py``).  Output: one
self-contained ``fig_*.m`` per figure, with every value hard-coded, plus
``make_all_figures.m``.  Only completed cells (all questions graded) are used.
All figures share one style: LaTeX text, 16 cm width, shared method colors.

  fig_motivation          revision converges after few steps; a fixed schedule wastes compute
  fig_latency_accuracy    accuracy vs. serial forward passes (main result)
  fig_revision_dynamics   net repair by revision depth, revision stage, and budget progress
  fig_regimes             revision quality, stopping rules, breadth vs. latency
  fig_accuracy_budget     accuracy vs. FLOP budget
  fig_compute_use, fig_replay_efficiency, fig_depth_breadth(_accuracy), fig_seed_consistency
  fig_abl_*               ablations: width N, temperature, budget, block size W, prompts, passes
  fig_models_lines        accuracy of every method on the other models

The output directory is ``$ARR_FIGURE_DIR`` (default ``figures/matlab``).

    python collect_results.py && python arr_analysis.py && python make_matlab_figure_script.py
    cd figures/matlab && matlab -batch make_all_figures
"""
import csv
import glob
import os
import statistics
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "results", "all_results.csv")
OUTD = os.environ.get("ARR_FIGURE_DIR") or (os.path.join(ROOT, "overleaf_paper", "matlab_figure")
                                          if os.path.isdir(os.path.join(ROOT, "overleaf_paper"))
                                          else os.path.join(ROOT, "figures", "matlab"))
STAMP = time.strftime("%Y-%m-%d %H:%M")

MODELS = ["Qwen2.5-Math-7B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-8B", "Ministral-8B-Instruct-2410",
          "gemma-4-E4B-it", "Qwen3.5-9B", "Qwen3.5-2B"]
MLBL = {"Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B", "Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen3-8B": "Qwen3-8B",
        "Ministral-8B-Instruct-2410": "Ministral-8B", "gemma-4-E4B-it": "Gemma-4-E4B", "Qwen3.5-9B": "Qwen3.5-9B",
        "Qwen3.5-2B": "Qwen3.5-2B"}
DSETS = ["math500", "olympiadbench", "amc23", "aime24", "aime25", "gsm8k", "livebench_reasoning"]
DLBL = {"math500": "MATH500", "olympiadbench": "OlympiadBench", "amc23": "AMC23", "aime24": "AIME24",
        "aime25": "AIME25", "gsm8k": "GSM8K", "livebench_reasoning": "LiveBench"}
MATH = ["math500", "olympiadbench", "amc23", "aime24", "aime25"]
METHODS = ["arr", "replace_ordinary", "replace_pld", "arr_no_replace", "direct_fixed", "critic_correct",
           "continuation", "maj"]
XMETHODS = METHODS + ["conma"]   # ConMA: single seed, Qwen2.5-Math-7B only; NaN where it was not run
TEXLBL = [r"\textbf{ARR}", r"\ARRar", r"\ARRpld", r"\ARRdepth", r"\FDR", r"\VC", r"\SeqC", r"\SC"]
PRIMARY = "Qwen2.5-Math-7B-Instruct"

# ------------------------------------------------------------------ load completed cells
C = {}
for r in csv.DictReader(open(SRC)):
    if r["variant"] != "main" or r["model"] not in MODELS or r["dataset"] not in DSETS or r["method"] not in XMETHODS:
        continue
    # keep every seed whose questions are all graded; a later seed that is still running does not hide the cell
    size = int(r["dataset_size"])
    done = []
    for tok in r["per_seed"].split():
        k, n = tok.split(":", 1)[1].split("/")
        if int(n) == size:
            done.append(100 * float(k) / float(n))
    if not done:
        continue
    f = lambda k: float(r[k]) if r[k] not in ("", "None") else float("nan")
    C[(r["model"], r["dataset"], float(r["budget"]), r["method"])] = dict(
        acc=sum(done) / len(done), std=statistics.pstdev(done) if len(done) > 1 else 0.0,
        seeds=len(done), tflops=f("mean_tflops"), cp=f("critical_path_passes_p50"),
        roots=f("roots_per_question"), tok=f("output_tokens_per_question"))


def budgets(model, ds):
    return sorted({b for (m, d, b, me) in C if m == model and d == ds and me in METHODS})


def cv(model, ds, b, me, key):
    """Value of one cell, NaN if the method was not run there (ConMA)."""
    c = C.get((model, ds, b, me))
    return c[key] if c else float("nan")


def full_setting(model, ds):
    """True if every method is complete at every budget of this (model, benchmark)."""
    bs = budgets(model, ds)
    return bool(bs) and all((model, ds, b, me) in C for b in bs for me in METHODS)


def mstr(vals, nd=4):
    return "[" + " ".join("NaN" if v != v else f"{v:.{nd}g}" for v in vals) + "]"


def mat(rows, nd=4):
    return "[" + "; ".join(" ".join("NaN" if v != v else f"{v:.{nd}g}" for v in r) for r in rows) + "]"


def cellstr(xs):
    return "{" + ", ".join("'" + x.replace("'", "''") + "'" for x in xs) + "}"


# ------------------------------------------------------------------ shared MATLAB style (local functions)
STYLE = r"""
% =====================================================================================
% Shared style (identical in every figure file of the paper)
% =====================================================================================
function S = figStyle()
S.lbl = {'ARR (ours)', 'ARR$_{\mathrm{AR}}$', 'ARR$_{\mathrm{PLD}}$', 'ARR$_{\mathrm{Depth}}$', ...
         'Self-Refine', 'SETS', 'Sequential Revision', 'Majority Voting', 'ConMA'};
S.col = [0.80 0.11 0.16;   % ARR
         0.95 0.56 0.16;   % ARR_AR
         0.83 0.70 0.12;   % ARR_PLD
         0.52 0.38 0.72;   % ARR_Depth
         0.58 0.58 0.58;   % Self-Refine (fixed depth)
         0.10 0.56 0.47;   % SETS
         0.55 0.40 0.30;   % Sequential Revision
         0.16 0.37 0.75;   % Majority Voting
         0.85 0.33 0.62];  % ConMA
S.mk  = {'o', 's', 'd', '^', 'v', 'p', 'x', 'h', '>'};
S.lw  = [2.0 1.1 1.1 1.1 1.1 1.1 1.1 1.6 1.1];
S.ms  = [4.5 3.6 3.8 3.8 3.8 4.4 4.4 4.6 3.8];
S.width = 16;              % cm, equal to the ACL \textwidth: fonts print at their true size
S.fs = 8;                  % pt
set(groot, 'defaultTextInterpreter', 'latex', 'defaultLegendInterpreter', 'latex', ...
    'defaultAxesTickLabelInterpreter', 'latex', 'defaultAxesFontSize', S.fs, ...
    'defaultTextFontSize', S.fs, 'defaultLegendFontSize', S.fs);
end

function f = newFig(S, height)
f = figure('Units', 'centimeters', 'Position', [2 2 S.width height], 'Color', 'w');
end

function styleAxes(ax, S, ttl)
hold(ax, 'on');
set(ax, 'Box', 'off', 'TickDir', 'out', 'TickLength', [0.018 0.018], 'LineWidth', 0.6, ...
    'XColor', [0.2 0.2 0.2], 'YColor', [0.2 0.2 0.2], 'YGrid', 'on', 'XGrid', 'off', ...
    'GridColor', [0 0 0], 'GridAlpha', 0.10, 'Layer', 'top', 'FontSize', S.fs, ...
    'TickLabelInterpreter', 'latex');
if nargin > 2
    title(ax, ttl, 'FontWeight', 'normal', 'FontSize', S.fs, 'Interpreter', 'latex');
    ax.TitleHorizontalAlignment = 'left';
end
end

function h = drawMethod(ax, S, k, x, y, e)
if nargin > 5 && any(e > 0)
    h = errorbar(ax, x, y, e, ['-' S.mk{k}], 'CapSize', 2.5);
else
    h = plot(ax, x, y, ['-' S.mk{k}]);
end
set(h, 'Color', S.col(k,:), 'LineWidth', S.lw(k), 'MarkerSize', S.ms(k), ...
    'MarkerFaceColor', S.col(k,:), 'MarkerEdgeColor', S.col(k,:));
if strcmp(S.mk{k}, 'x'), h.MarkerFaceColor = 'none'; end
end

function topLegend(t, ax, S, ks)
h = gobjects(numel(ks), 1);
for i = 1:numel(ks)
    k = ks(i);
    h(i) = plot(ax, NaN, NaN, ['-' S.mk{k}], 'Color', S.col(k,:), 'LineWidth', S.lw(k), ...
        'MarkerSize', S.ms(k), 'MarkerFaceColor', S.col(k,:));
end
lg = legend(ax, h, S.lbl(ks), 'Orientation', 'horizontal', 'NumColumns', numel(ks), 'Box', 'off', ...
    'Interpreter', 'latex', 'FontSize', S.fs - 1.5);
lg.ItemTokenSize = [9 6];
lg.Layout.Tile = 'north';
end

function stackLegends(t, lgs)
% Center each legend horizontally and stack them, in order, below the tiled layout.
drawnow;
h = 0;
for i = 1:numel(lgs), lgs(i).Units = 'normalized'; h = h + lgs(i).Position(4); end
t.OuterPosition = [0 h + 0.02 1 1 - h - 0.02];
drawnow;
y = h;
for i = 1:numel(lgs)
    p = lgs(i).Position; y = y - p(4);
    lgs(i).Position = [(1 - p(3)) / 2, y, p(3), p(4)];
end
end

function saveFig(f, name)
outDir = fileparts(mfilename('fullpath')); if isempty(outDir), outDir = pwd; end
exportgraphics(f, fullfile(outDir, [name '.pdf']), 'ContentType', 'vector');
fprintf('wrote %s.pdf\n', fullfile(outDir, name));
end
"""


def header(name, purpose):
    return (f"function {name}()\n%{name.upper()}  {purpose}\n"
            f"%   All values are hard-coded (no file is read). Completed experiments only.\n"
            f"%   Generated {STAMP} by make_matlab_figure_script.py from results/all_results.csv.\n"
            f"%   Writes {name}.pdf next to this file. Requires MATLAB R2020b+.\n"
            f"%   Method order (rows/columns): ARR, ARR_AR, ARR_PLD, ARR_Depth, Self-Refine, SETS, Sequential Revision, Majority Voting, ConMA.\n"
            f"%   ConMA (Wang et al., 2026) has only Qwen2.5-Math-7B results; NaN marks a missing value.\n\n"
            f"S = figStyle();\n")


files = {}
P = PRIMARY
ds6 = [d for d in MATH + ["gsm8k"] if full_setting(P, d)]

# ------------------------------------------------------------------ fig_latency_accuracy (main body)
lines = [header("fig_latency_accuracy",
                "Accuracy vs. median critical-path serial forward passes, Qwen2.5-Math-7B-Instruct."),
         "% D(d).passes / D(d).acc: rows = methods, columns = budgets (one point per budget)",
         "D = struct('name', {}, 'passes', {}, 'acc', {});"]
for d in ds6:
    bs = budgets(P, d)
    lines.append(f"D(end+1) = struct('name', '{DLBL[d]}', "
                 f"'passes', {mat([[cv(P, d, b, me, 'cp') for b in bs] for me in XMETHODS], 6)}, "
                 f"'acc', {mat([[cv(P, d, b, me, 'acc') for b in bs] for me in XMETHODS])});")
lines.append(r"""
show = [1 2 3 4 5 6 8 9];   % SeqCont (7) loses 20-60 points with depth (see fig_accuracy_budget); omitted for scale
f = newFig(S, 8.8);
t = tiledlayout(f, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for d = 1:numel(D)
    ax = nexttile(t);
    styleAxes(ax, S, sprintf('(%s) %s', char('a' + d - 1), D(d).name));
    for k = show(end:-1:1)
        drawMethod(ax, S, k, D(d).passes(k,:), D(d).acc(k,:));
    end
    P = D(d).passes(show,:); xl = [min(P(:), [], 'omitnan') max(P(:), [], 'omitnan')];
    set(ax, 'XScale', 'log', 'XLim', xl .* [0.85 1.18]);
    kTicks(ax, xl);
    if d == 1, ax1 = ax; end
end
xlabel(t, 'Median serial forward passes on the critical path (log scale)', 'Interpreter', 'latex', 'FontSize', S.fs);
ylabel(t, 'Accuracy (\%)', 'Interpreter', 'latex', 'FontSize', S.fs);
topLegend(t, ax1, S, show);
saveFig(f, 'fig_latency_accuracy');
end

function kTicks(ax, xl)
c = [250 500 1000 2000 4000 8000 16000 32000];
c = c(c >= xl(1)*0.85 & c <= xl(2)*1.18);
lab = arrayfun(@(v) sprintf('%gk', v/1000), c, 'UniformOutput', false);
set(ax, 'XTick', c, 'XTickLabel', lab, 'XMinorTick', 'off');
end
""" + STYLE)
files["fig_latency_accuracy.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_accuracy_budget (appendix)
lines = [header("fig_accuracy_budget",
                "Accuracy vs. per-question TFLOPs budget, Qwen2.5-Math-7B-Instruct, mean +- std."),
         "% D(d).budget (TFLOPs); D(d).acc / D(d).std: rows = methods, columns = budgets",
         "D = struct('name', {}, 'budget', {}, 'acc', {}, 'std', {});"]
for d in ds6:
    bs = budgets(P, d)
    lines.append(f"D(end+1) = struct('name', '{DLBL[d]}', "
                 f"'budget', {mstr(bs)}, "
                 f"'acc', {mat([[cv(P, d, b, me, 'acc') for b in bs] for me in XMETHODS])}, "
                 f"'std', {mat([[cv(P, d, b, me, 'std') for b in bs] for me in XMETHODS])});")
lines.append(r"""
f = newFig(S, 9.6);
t = tiledlayout(f, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for d = 1:numel(D)
    ax = nexttile(t);
    styleAxes(ax, S, sprintf('(%s) %s', char('a' + d - 1), D(d).name));
    for k = numel(S.lbl):-1:1
        drawMethod(ax, S, k, D(d).budget, D(d).acc(k,:), D(d).std(k,:));
    end
    set(ax, 'XScale', 'log', 'XTick', D(d).budget, 'XLim', [D(d).budget(1)*0.85 D(d).budget(end)*1.18]);
    if d == 1, ax1 = ax; end
end
xlabel(t, 'Compute budget $B$ (TFLOPs per question, log scale)', 'Interpreter', 'latex', 'FontSize', S.fs);
ylabel(t, 'Accuracy (\%)', 'Interpreter', 'latex', 'FontSize', S.fs);
topLegend(t, ax1, S, 1:numel(S.lbl));
saveFig(f, 'fig_accuracy_budget');
end
""" + STYLE)
files["fig_accuracy_budget.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_compute_use (appendix)
def use_rows(models):
    lbl, A, U, R = [], [], [], []
    for m in models:
        for d in MATH + ["gsm8k"]:
            if not full_setting(m, d):
                continue
            for b in budgets(m, d):
                lbl.append(f"{MLBL[m]} | {DLBL[d]} | B={b:g}")
                A.append([cv(m, d, b, me, "acc") for me in XMETHODS])
                U.append([100 * cv(m, d, b, me, "tflops") / b for me in XMETHODS])
                R.append([cv(m, d, b, me, "cp") for me in XMETHODS])
    return lbl, A, U, R


l1, A1, U1, R1 = use_rows([P])
l2, A2, U2, R2 = use_rows([m for m in MODELS if m != P])
lines = [header("fig_compute_use",
                "Compute use of each method: TFLOPs spent, serial passes, and accuracy difference to ARR."),
         "% rows = (model | benchmark | budget) cells, columns = methods",
         "% A*: accuracy (%), U*: TFLOPs spent as % of the budget B, R*: median critical-path serial passes",
         f"lbl1 = {cellstr(l1)};",
         f"A1 = {mat(A1)};",
         f"U1 = {mat(U1)};",
         f"R1 = {mat(R1, 6)};",
         f"lbl2 = {cellstr(l2)};",
         f"A2 = {mat(A2)};",
         f"U2 = {mat(U2)};",
         f"R2 = {mat(R2, 6)};",
         r"""
G1 = A1 - A1(:,1);  G2 = A2 - A2(:,1);        % accuracy difference to ARR in the same cell
Q1 = R1 ./ R1(:,1); Q2 = R2 ./ R2(:,1);        % serial passes relative to ARR in the same cell
n1 = size(G1,1);     n2 = size(G2,1);
mn = @(X) mean(X, 1, 'omitnan');                 % NaN: method not run in that cell (ConMA)
se = @(X) std(X, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(X), 1));
M = struct('u', {mn(U1), mn(U2)}, ...
           'q', {exp(mn(log(Q1))), exp(mn(log(Q2)))}, ...   % geometric mean of the ratio
           'g', {mn(G1), mn(G2)}, 'e', {se(G1), se(G2)});
y = numel(S.lbl):-1:1; dy = [0.16 -0.16];
ttl  = {'(a) FLOP budget used', '(b) Latency', '(c) Accuracy'};
xlab = {'TFLOPs spent (\% of $B$)', 'Passes / passes of ARR', 'Accuracy $-$ ARR (points)'};
hintL = {'$\leftarrow$ unused budget', '$\leftarrow$ faster', '$\leftarrow$ worse'};
hintR = {'', 'slower $\rightarrow$', 'better $\rightarrow$'};
yTop = numel(y) + 1.15;
f = newFig(S, 7.2);
t = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:3
    ax = nexttile(t); styleAxes(ax, S, ttl{p});
    set(ax, 'YGrid', 'off', 'XGrid', 'on');
    if p == 1, ax1 = ax; xl = [0 105]; end
    if p == 2, set(ax, 'XScale', 'log'); xl = [0.4 3]; end
    if p == 3, xl = [min([M.g] - [M.e]) - 0.8, max([M.g] + [M.e]) + 0.8]; end
    % band that marks the reference method (ARR) in every panel
    patch(ax, xl([1 2 2 1]), numel(y) + [-0.45 -0.45 0.45 0.45], S.col(1,:), 'FaceAlpha', 0.08, 'EdgeColor', 'none');
    if p == 2, plot(ax, [1 1], [0.4 numel(y)+0.6], '-', 'Color', S.col(1,:), 'LineWidth', 0.8); end
    if p == 3, plot(ax, [0 0], [0.4 numel(y)+0.6], '-', 'Color', S.col(1,:), 'LineWidth', 0.8); end
    for k = 1:numel(S.lbl)
        for m = 1:2
            switch p
                case 1, x = M(m).u(k);
                case 2, x = M(m).q(k);
                case 3, x = M(m).g(k);
                    plot(ax, x + [-1 1]*M(m).e(k), y(k)+dy(m)*[1 1], '-', 'Color', S.col(k,:), 'LineWidth', 0.9);
            end
            h = plot(ax, x, y(k)+dy(m), 'o', 'MarkerSize', 5, 'MarkerEdgeColor', S.col(k,:), 'LineWidth', 1.0);
            if m == 1, h.MarkerFaceColor = S.col(k,:); else, h.MarkerFaceColor = 'w'; end
        end
    end
    set(ax, 'XLim', xl, 'YTick', fliplr(y), 'YTickLabel', fliplr(S.lbl), 'YLim', [0.4 yTop]);
    if p > 1, set(ax, 'YTickLabel', {}); end
    if p == 2, set(ax, 'XMinorGrid', 'off', 'XMinorTick', 'off', 'XTick', [0.5 1 2 3], 'XTickLabel', {'0.5$\times$','1$\times$','2$\times$','3$\times$'}); end
    xlabel(ax, xlab{p});
    text(ax, xl(1), yTop - 0.3, [' ' hintL{p}], 'FontSize', S.fs - 1, 'Color', [0.35 0.35 0.35], 'HorizontalAlignment', 'left');
    text(ax, xl(2), yTop - 0.3, [hintR{p} ' '], 'FontSize', S.fs - 1, 'Color', [0.35 0.35 0.35], 'HorizontalAlignment', 'right');
end
% color legend (methods), one row on top
hc = gobjects(numel(S.lbl), 1);
for k = 1:numel(S.lbl)
    hc(k) = plot(ax1, NaN, NaN, 's', 'MarkerSize', 6, 'MarkerFaceColor', S.col(k,:), 'MarkerEdgeColor', S.col(k,:));
end
lg = legend(ax1, hc, S.lbl, 'Orientation', 'horizontal', 'NumColumns', numel(S.lbl), 'Box', 'off', ...
    'Interpreter', 'latex', 'FontSize', S.fs - 2);
lg.ItemTokenSize = [4 4];
lg.Layout.Tile = 'north';
% marker legend (model groups), one row at the bottom
ax3 = ax;
h1 = plot(ax3, NaN, NaN, 'o', 'MarkerSize', 5, 'MarkerFaceColor', [0.3 0.3 0.3], 'MarkerEdgeColor', [0.3 0.3 0.3]);
h2 = plot(ax3, NaN, NaN, 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', [0.3 0.3 0.3]);
lg2 = legend(ax3, [h1 h2], {sprintf('Filled: Qwen2.5-Math-7B (mean of %d cells)', n1), ...
    sprintf('Open: five other models (mean of %d cells)', n2)}, ...
    'Orientation', 'horizontal', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex', 'FontSize', S.fs);
lg2.ItemTokenSize = [8 8];
lg2.Layout.Tile = 'south';
fprintf('passes x ARR (primary):'); fprintf(' %.2f', M(1).q); fprintf('\npasses x ARR (others): '); fprintf(' %.2f', M(2).q); fprintf('\n');
saveFig(f, 'fig_compute_use');
end
""" + STYLE]
files["fig_compute_use.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_depth_breadth (main body)
bs = budgets(P, "math500")
K = ["arr", "arr_no_replace", "direct_fixed"]
roots = [[C[(P, "math500", b, me)]["roots"] for b in bs] for me in K]
used = [[100 * C[(P, "math500", b, me)]["tflops"] / b for b in bs] for me in K]
GSHORT = {"Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math", "Ministral-8B-Instruct-2410": "Ministral",
          "gemma-4-E4B-it": "Gemma-4"}   # short x-tick labels, as in fig_depth_breadth_accuracy
gl, gv, glo, ghi = [], [], [], []
for m in MODELS:
    g = [C[(m, d, budgets(m, d)[-1], "arr")]["acc"] - C[(m, d, budgets(m, d)[-1], "arr_no_replace")]["acc"]
         for d in MATH if full_setting(m, d)]
    if g:
        gl.append(GSHORT.get(m, MLBL[m])); gv.append(sum(g) / len(g)); glo.append(min(g)); ghi.append(max(g))
lines = [header("fig_depth_breadth",
                "Exchange of exhausted depth for new samples: roots, budget use, and gain of replacement."),
         f"budget = {mstr(bs)};   % MATH500, Qwen2.5-Math-7B-Instruct (TFLOPs)",
         f"roots  = {mat(roots)};   % rows: ARR, ARR_Depth, FDR; roots per question",
         f"used   = {mat(used)};   % TFLOPs spent, % of B",
         f"gainLbl = {cellstr(gl)};   % model",
         f"gain    = {mstr(gv)};   % mean ARR - ARR_Depth over benchmarks at the largest budget (points)",
         f"gainLo  = {mstr(glo)};   % minimum over benchmarks",
         f"gainHi  = {mstr(ghi)};   % maximum over benchmarks",
         r"""
idx = [1 4 5];        % style index of ARR, ARR_Depth, FDR
xl  = [budget(1)/1.18, budget(end)*1.18];   % symmetric padding on the log axis
f = newFig(S, 5.6);
t = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(t); styleAxes(ax, S, '(a) Roots per question'); ax1 = ax;
for r = 3:-1:1, drawMethod(ax, S, idx(r), budget, roots(r,:)); end
text(ax, budget(end)/1.07, roots(1,end), sprintf('%.1f', roots(1,end)), 'Color', S.col(1,:), ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'FontSize', S.fs - 0.5);
text(ax, budget(end)/1.07, roots(2,end), sprintf('%.0f', roots(2,end)), 'Color', S.col(4,:), ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'FontSize', S.fs - 0.5);
yt = ceil(max(roots(:))/5)*5;
set(ax, 'XScale', 'log', 'XTick', budget, 'XLim', xl, 'YLim', [0 yt], 'YTick', 0:5:yt);
xlabel(ax, 'Budget $B$ (TFLOPs)'); ylabel(ax, 'Roots per question');

ax = nexttile(t); styleAxes(ax, S, '(b) Budget spent');
for r = 3:-1:1, drawMethod(ax, S, idx(r), budget, used(r,:)); end
set(ax, 'XScale', 'log', 'XTick', budget, 'XLim', xl, 'YLim', [0 105], 'YTick', 0:25:100);
xlabel(ax, 'Budget $B$ (TFLOPs)'); ylabel(ax, 'TFLOPs spent (\% of $B$)');

ax = nexttile(t); styleAxes(ax, S, '(c) Gain of replacement');
n = numel(gain);
plot(ax, [0.4 n+0.6], [0 0], '-', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.6);
for i = 1:n
    plot(ax, [i i], [gainLo(i) gainHi(i)], '-', 'Color', [S.col(1,:) 0.35], 'LineWidth', 3.5);
    plot(ax, i, gain(i), 'o', 'MarkerSize', 5, 'MarkerFaceColor', S.col(1,:), 'MarkerEdgeColor', 'w');
end
set(ax, 'XTick', 1:n, 'XTickLabel', gainLbl, 'XLim', [0.4 n+0.6], 'XTickLabelRotation', 35, ...
    'FontSize', S.fs - 0.5, 'YLim', [min(0, min(gainLo)) - 2, max(gainHi) + 2]);
ylabel(ax, 'ARR $-$ ARR$_{\mathrm{Depth}}$ (points)');
topLegend(t, ax1, S, idx);
saveFig(f, 'fig_depth_breadth');
end
""" + STYLE]
files["fig_depth_breadth.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_motivation (introduction)
RES = os.environ.get("ARR_OUTPUT_ROOT", "experiments_results/arr").rstrip("/")


def jsonl_records(model, ds, b, method):
    """Per-question records of one cell; reads combined and sharded files and keeps one record per (qid, seed)."""
    import json as _json
    import re as _re
    seen = set()
    pat = _re.compile(rf"results_{_re.escape(method)}(\.s\d+of\d+|\.s\d+)?\.jsonl$")
    for f in sorted(glob.glob(f"{RES}/{model}/{ds}/B{b:g}/results_{method}*.jsonl")):
        if not pat.search(os.path.basename(f)):
            continue
        for line in open(f):
            r = _json.loads(line)
            key = (r["qid"], r["seed"])
            if key in seen:
                continue
            seen.add(key)
            yield r


MOT_DS = "math500"
mbs = budgets(P, MOT_DS)
# (a) cumulative share of valid ARR_Depth roots that reached a fixed point after k revisions (largest budget)
fp_k, n_valid, n_fp = [0] * 9, 0, 0
for r in jsonl_records(P, MOT_DS, mbs[-1], "arr_no_replace"):
    for ro in r["roots"]:
        if not ro["valid"]:
            continue
        n_valid += 1
        if ro["status"] == "fixed_point":
            n_fp += 1
            if ro["revisions"] <= 8:
                fp_k[ro["revisions"]] += 1
cum = [100 * sum(fp_k[1:k + 1]) / n_valid for k in range(1, 9)]
fp_ever = 100 * n_fp / n_valid
# (b) share of FDR revision compute spent on roots that were already fixed points
red = []
for b in mbs:
    tot = waste = 0.0
    for r in jsonl_records(P, MOT_DS, b, "direct_fixed"):
        done = set()
        for e in sorted((e for e in r["events"] if e["op"] == "revise"), key=lambda e: e["revisions_before"]):
            tot += e["spent"]
            if e["root"] in done:
                waste += e["spent"]
            if e["identical"]:
                done.add(e["root"])
    red.append(100 * waste / tot)
K4 = ["direct_fixed", "arr_no_replace", "maj", "arr"]
macc = [[C[(P, MOT_DS, b, me)]["acc"] for b in mbs] for me in K4]
mtf = [[C[(P, MOT_DS, b, me)]["tflops"] for b in mbs] for me in K4]
lines = [header("fig_motivation",
                "Why ARR: revision exhausts quickly, and a fixed revision depth wastes the rest of its compute."),
         f"budget = {mstr(mbs)};   % MATH500, Qwen2.5-Math-7B-Instruct (TFLOPs)",
         f"cumFP  = {mstr(cum)};   % (a) % of valid ARR_Depth roots at a fixed point after k = 1..8 revisions, B = {mbs[-1]:g}",
         f"fpEver = {fp_ever:.4g};   % (a) % of valid ARR_Depth roots that reach a fixed point at any depth",
         f"redund = {mstr(red)};   % (b) % of FDR revision TFLOPs spent on roots that were already fixed points",
         f"acc    = {mat(macc)};   % (c) accuracy (%); rows: FDR, ARR_Depth, SC, ARR; columns: budgets",
         f"tf     = {mat(mtf)};   % (c) TFLOPs actually spent per question",
         r"""
f = figure('Units', 'centimeters', 'Position', [2 2 7.8 4.9], 'Color', 'w');   % single column (ACL \columnwidth)
t = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(t); styleAxes(ax, S, '(a) Convergence');
c = S.col(4,:); k = 1:numel(cumFP);
fill(ax, [k fliplr(k)], [cumFP zeros(size(cumFP))], c, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
plot(ax, [0.6 numel(cumFP) + 0.4], [fpEver fpEver], '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.7);
text(ax, numel(cumFP) + 0.35, fpEver + 2, {sprintf('%.0f\\%% converge', fpEver), 'by budget end'}, 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', 'FontSize', S.fs - 2, 'Color', [0.35 0.35 0.35]);
plot(ax, k, cumFP, '-o', 'Color', c, 'LineWidth', 1.6, 'MarkerSize', 3.5, 'MarkerFaceColor', c, 'MarkerEdgeColor', c);
text(ax, 1.3, cumFP(1) - 1, sprintf('%.0f\\%%', cumFP(1)), 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', 'FontSize', S.fs - 1.5, 'Color', c);
set(ax, 'XTick', [1 2 4 6 8], 'XLim', [0.6 numel(cumFP) + 0.4], 'YLim', [0 100], 'YTick', 0:25:100);
xlabel(ax, 'Number of revisions $k$'); ylabel(ax, 'Converged solutions (\%)');

ax = nexttile(t); styleAxes(ax, S, '(b) Wasted compute');
bb = bar(ax, [100 - redund(:), redund(:)], 0.62, 'stacked', 'EdgeColor', 'none');
bb(1).FaceColor = S.col(5,:); bb(2).FaceColor = S.col(2,:);
for i = 1:numel(redund)
    text(ax, i, 100 - redund(i)/2, sprintf('%.0f', redund(i)), 'HorizontalAlignment', 'center', 'Color', 'w', 'FontSize', S.fs - 1);
end
n = numel(budget);
set(ax, 'XTick', 1:n, 'XTickLabel', arrayfun(@(x) sprintf('%g', x), budget, 'UniformOutput', false), ...
    'XLim', [0.4 n + 0.6], 'YLim', [0 100], 'YTick', 0:25:100);
xlabel(ax, 'Budget $B$ (TFLOPs)'); ylabel(ax, 'Revision compute (\%)');
hb = [bar(ax, NaN, 'FaceColor', S.col(2,:), 'EdgeColor', 'none'), bar(ax, NaN, 'FaceColor', S.col(5,:), 'EdgeColor', 'none')];
lg = legend(ax, hb, {'wasted: already converged', 'useful: still changing'}, 'Orientation', 'horizontal', 'NumColumns', 2, 'Box', 'off', ...
    'Interpreter', 'latex', 'FontSize', S.fs - 1);
lg.ItemTokenSize = [8 8]; lg.Layout.Tile = 'north';
saveFig(f, 'fig_motivation');
end
""" + STYLE]
files["fig_motivation.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_depth_breadth_accuracy (main body)
def stages(model, ds, b):
    """Vote accuracy (%) of the N initial samples, of the same roots revised to their end (depth), and of ARR (breadth)."""
    n = d0 = dE = full = 0
    for r in jsonl_records(model, ds, b, "arr"):
        c = r.get("cohort_vote_correct_by_depth") or []
        if not c:
            continue
        n += 1; d0 += c[0]; dE += c[-1]; full += r["correct"]
    return (100 * d0 / n, 100 * dE / n, 100 * full / n) if n else None


# (a) primary model, every benchmark, largest budget
SHORT = {"math500": "MATH", "olympiadbench": "Olymp.", "amc23": "AMC23", "aime24": "AIME24", "aime25": "AIME25",
         "gsm8k": "GSM8K"}
MSHORT = {"Qwen3-8B": "Qwen3-8B", "Ministral-8B-Instruct-2410": "Ministral", "gemma-4-E4B-it": "Gemma-4",
          "Qwen3.5-9B": "Qwen3.5-9B", "Qwen3.5-2B": "Qwen3.5-2B"}
pa_lbl, pa = [], []
for d in ds6:
    v = stages(P, d, budgets(P, d)[-1])
    if v:
        pa_lbl.append(SHORT[d]); pa.append([v[1] - v[0], v[2] - v[1]])
# (b) primary model, mean over its math benchmarks, at every budget
pbud = budgets(P, "math500")
pb = []
for b in pbud:
    g = [stages(P, d, b) for d in MATH if full_setting(P, d) and b in budgets(P, d)]
    g = [v for v in g if v]
    pb.append([sum(v[1] - v[0] for v in g) / len(g), sum(v[2] - v[1] for v in g) / len(g)])
# (c) the other models, mean over their completed math benchmarks at the largest budget
pc_lbl, pc, pc_n = [], [], []
for m in MODELS:
    if m == P:
        continue
    g = [stages(m, d, budgets(m, d)[-1]) for d in MATH if full_setting(m, d)]
    g = [v for v in g if v]
    if g:
        pc_lbl.append(MSHORT[m]); pc_n.append(len(g))
        pc.append([sum(v[1] - v[0] for v in g) / len(g), sum(v[2] - v[1] for v in g) / len(g)])
lines = [header("fig_depth_breadth_accuracy",
                "Accuracy gained by depth (revision of the initial roots) and by breadth (new roots) in ARR."),
         "% every row: [depth gain, breadth gain] in accuracy points over the vote of the N initial samples",
         f"labA = {cellstr(pa_lbl)};",
         f"gainA = {mat(pa)};   % (a) Qwen2.5-Math-7B, each benchmark, largest budget",
         f"budB = {mstr(pbud)};",
         f"gainB = {mat(pb)};   % (b) Qwen2.5-Math-7B, mean over its math benchmarks, each budget",
         f"labC = {cellstr(pc_lbl)};",
         f"nC = {mstr([float(x) for x in pc_n])};   % number of completed math benchmarks per model",
         f"gainC = {mat(pc) if pc else '[]'};   % (c) other models, mean over their math benchmarks, largest budget",
         r"""
cD = S.col(4,:); cB = S.col(1,:);          % depth = ARR_Depth colour, breadth = ARR colour
f = newFig(S, 5.6);
t = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = stackPanel(nexttile(t), S, gainA, labA, '(a) Per benchmark', cD, cB);
ylabel(ax1, 'Gain over $N$ samples (points)');
stackPanel(nexttile(t), S, gainB, arrayfun(@(x) sprintf('%g', x), budB, 'UniformOutput', false), ...
    '(b) Per budget', cD, cB);
xlabel(gca, 'Budget $B$ (TFLOPs)');
stackPanel(nexttile(t), S, gainC, labC, '(c) Other models', cD, cB);
h1 = bar(ax1, NaN, 'FaceColor', cD, 'EdgeColor', 'none'); h2 = bar(ax1, NaN, 'FaceColor', cB, 'EdgeColor', 'none');
lg = legend(ax1, [h1 h2], {'depth: revise the $N$ initial roots', 'breadth: new roots (ARR)'}, ...
    'Orientation', 'horizontal', 'NumColumns', 2, 'Box', 'off', 'Interpreter', 'latex', 'FontSize', S.fs);
lg.ItemTokenSize = [10 8]; lg.Layout.Tile = 'north';
saveFig(f, 'fig_depth_breadth_accuracy');
end

function ax = stackPanel(ax, S, G, labels, ttl, cD, cB)
% stacked depth/breadth bars; the net gain is written above (or below) each bar
styleAxes(ax, S, ttl);
n = size(G, 1);
b = bar(ax, G, 0.62, 'stacked', 'EdgeColor', 'none');
b(1).FaceColor = cD; b(2).FaceColor = cB;
plot(ax, [0.4 n + 0.6], [0 0], '-', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.6);
tot = sum(G, 2); top = sum(max(G, 0), 2); bot = sum(min(G, 0), 2);
span = max(top) - min(0, min(bot));
for i = 1:n
    if tot(i) >= 0, y = top(i) + 0.05*span; va = 'bottom'; else, y = bot(i) - 0.05*span; va = 'top'; end
    text(ax, i, y, sprintf('%+.1f', tot(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', va, ...
        'FontSize', S.fs - 1, 'Color', [0.25 0.25 0.25]);
end
set(ax, 'XTick', 1:n, 'XTickLabel', labels, 'XLim', [0.4 n + 0.6], 'FontSize', S.fs - 0.5, ...
    'YLim', [min(0, min(bot)) - 0.18*span, max(top) + 0.2*span]);
if n > 4, set(ax, 'XTickLabelRotation', 35); end
end
""" + STYLE]
files["fig_depth_breadth_accuracy.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_seed_consistency (appendix)
def per_seed_acc(model, ds, b, method):
    """Accuracy (%) of each fully graded seed of one cell."""
    import csv as _csv
    for r in _csv.DictReader(open(SRC)):
        if r["variant"] == "main" and r["model"] == model and r["dataset"] == ds and float(r["budget"]) == b \
                and r["method"] == method:
            size, out = int(r["dataset_size"]), []
            for tok in r["per_seed"].split():
                k, n = tok.split(":", 1)[1].split("/")
                if int(n) == size:
                    out.append(100 * float(k) / float(n))
            return out
    return None


sc_lbl, s0, sm, d0, dm = [], [], [], [], []
for m in MODELS:
    for d in DSETS:
        if not full_setting(m, d):
            continue
        b = budgets(m, d)[-1]
        a = per_seed_acc(m, d, b, "arr"); c = per_seed_acc(m, d, b, "maj"); z = per_seed_acc(m, d, b, "arr_no_replace")
        if not a or not c or not z or len(a) < 2:
            continue
        sc_lbl.append(f"{MLBL[m]} | {DLBL[d]}")
        s0.append([a[0], c[0]]); sm.append([sum(a) / len(a), sum(c) / len(c)])
        d0.append([a[0] - c[0], a[0] - z[0]])
        dm.append([sum(a) / len(a) - sum(c) / len(c), sum(a) / len(a) - sum(z) / len(z)])
lines = [header("fig_seed_consistency",
                "First seed against the mean over all seeds, for every completed setting with more than one seed."),
         f"settings = {cellstr(sc_lbl)};",
         f"acc0 = {mat(s0) if s0 else '[]'};   % first-seed accuracy (%); cols: ARR, SC",
         f"accM = {mat(sm) if sm else '[]'};   % mean accuracy over all seeds (%); cols: ARR, SC",
         f"dif0 = {mat(d0) if d0 else '[]'};   % first seed: ARR - SC, ARR - ARR_Depth (points)",
         f"difM = {mat(dm) if dm else '[]'};   % all seeds: ARR - SC, ARR - ARR_Depth (points)",
         r"""
f = newFig(S, 6.2);
t = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(t); styleAxes(ax, S, '(a) Accuracy'); ax1 = ax;
lo = min([acc0(:); accM(:)]) - 3; hi = max([acc0(:); accM(:)]) + 3;
plot(ax, [lo hi], [lo hi], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
ks = [1 8];
for j = 1:2
    plot(ax, acc0(:,j), accM(:,j), S.mk{ks(j)}, 'Color', S.col(ks(j),:), 'MarkerFaceColor', S.col(ks(j),:), 'MarkerSize', 5);
end
axis(ax, [lo hi lo hi]); axis(ax, 'square');
xlabel(ax, 'First seed (\%)'); ylabel(ax, 'Mean over all seeds (\%)');

ax = nexttile(t); styleAxes(ax, S, '(b) Differences between methods');
lo = min([dif0(:); difM(:)]) - 1.5; hi = max([dif0(:); difM(:)]) + 1.5;
patch(ax, [0 hi hi 0], [0 0 hi hi], S.col(1,:), 'FaceAlpha', 0.05, 'EdgeColor', 'none');
patch(ax, [lo 0 0 lo], [lo lo 0 0], S.col(1,:), 'FaceAlpha', 0.05, 'EdgeColor', 'none');
plot(ax, [lo hi], [lo hi], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
plot(ax, [lo hi], [0 0], ':', 'Color', [0.5 0.5 0.5]); plot(ax, [0 0], [lo hi], ':', 'Color', [0.5 0.5 0.5]);
mk2 = {'h', '^'}; cc = [8 4];
for j = 1:2
    plot(ax, dif0(:,j), difM(:,j), mk2{j}, 'Color', S.col(cc(j),:), 'MarkerFaceColor', S.col(cc(j),:), 'MarkerSize', 5);
end
axis(ax, [lo hi lo hi]); axis(ax, 'square');
xlabel(ax, 'First seed (points)'); ylabel(ax, 'All seeds (points)');
h = gobjects(4,1);
h(1) = plot(ax1, NaN, NaN, 'o', 'Color', S.col(1,:), 'MarkerFaceColor', S.col(1,:));
h(2) = plot(ax1, NaN, NaN, 'h', 'Color', S.col(8,:), 'MarkerFaceColor', S.col(8,:));
h(3) = plot(ax1, NaN, NaN, 'h', 'Color', S.col(8,:), 'MarkerFaceColor', S.col(8,:));
h(4) = plot(ax1, NaN, NaN, '^', 'Color', S.col(4,:), 'MarkerFaceColor', S.col(4,:));
lg = legend(ax1, h, {'ARR', 'Majority Voting', 'ARR $-$ Majority Voting', 'ARR $-$ ARR$_{\mathrm{Depth}}$'}, ...
    'Orientation', 'horizontal', 'NumColumns', 4, 'Box', 'off', 'Interpreter', 'latex', 'FontSize', S.fs);
lg.ItemTokenSize = [10 8]; lg.Layout.Tile = 'north';
saveFig(f, 'fig_seed_consistency');
end
""" + STYLE]
files["fig_seed_consistency.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_replay_efficiency (appendix)
ds5 = [d for d in MATH if full_setting(P, d)]
bs5 = budgets(P, "math500")
ratio = [[C[(P, d, b, "arr")]["cp"] / C[(P, d, b, "replace_ordinary")]["cp"] for b in bs5] for d in ds5]
tok = [[C[(P, d, bs5[-1], me)]["tok"] / 1000 for me in ["arr", "replace_ordinary", "maj"]] for d in ds5]
lines = [header("fig_replay_efficiency",
                "Serial passes and output tokens of replay decoding vs. autoregressive revision and SC."),
         f"bench  = {cellstr([DLBL[d] for d in ds5])};",
         f"budget = {mstr(bs5)};",
         f"ratio  = {mat(ratio)};   % median serial passes ARR / ARR_AR; rows = benchmarks, cols = budgets",
         f"tokens = {mat(tok)};     % output tokens per question (x1000) at the largest budget; cols = ARR, ARR_AR, SC",
         r"""
f = newFig(S, 5.6);
t = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(t); styleAxes(ax, S, '(a) Serial passes, ARR / ARR$_{\mathrm{AR}}$');
b = bar(ax, ratio, 'grouped', 'EdgeColor', 'none', 'BarWidth', 0.8);
shade = [0.95 0.72 0.72; 0.88 0.42 0.44; S.col(1,:)];
for j = 1:numel(b), b(j).FaceColor = shade(j,:); end
plot(ax, [0.4 numel(bench)+0.6], [1 1], '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.6);
set(ax, 'XTick', 1:numel(bench), 'XTickLabel', bench, 'YLim', [0 1.1]);
lg = legend(b, arrayfun(@(x) sprintf('$B=%g$', x), budget, 'UniformOutput', false), ...
    'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', 'Interpreter', 'latex');
lg.ItemTokenSize = [10 8];
ax = nexttile(t); styleAxes(ax, S, sprintf('(b) Output tokens per question, $B=%g$', budget(end)));
b = bar(ax, tokens, 'grouped', 'EdgeColor', 'none', 'BarWidth', 0.8);
ks = [1 2 8];
for j = 1:3, b(j).FaceColor = S.col(ks(j),:); end
set(ax, 'XTick', 1:numel(bench), 'XTickLabel', bench);
ylabel(ax, 'Tokens ($\times 10^3$)');
lg = legend(b, S.lbl(ks), 'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', 'Interpreter', 'latex');
lg.ItemTokenSize = [10 8];
saveFig(f, 'fig_replay_efficiency');
end
""" + STYLE]
files["fig_replay_efficiency.m"] = "\n".join(lines)

# ------------------------------------------------------------------ analyses of the question-level records
import json as _json
AN = _json.load(open(os.path.join(ROOT, "results", "analysis.json")))
MSHORT2 = {"Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B", "Qwen3-8B": "Qwen3-8B", "Ministral-8B-Instruct-2410": "Ministral-8B",
           "gemma-4-E4B-it": "Gemma-4-E4B", "Qwen3.5-9B": "Qwen3.5-9B", "Qwen3.5-2B": "Qwen3.5-2B",
           "Qwen2.5-7B-Instruct": "Qwen2.5-7B"}
REGMODELS = ["Qwen2.5-Math-7B-Instruct", "Qwen3-8B", "Ministral-8B-Instruct-2410", "gemma-4-E4B-it", "Qwen3.5-9B",
             "Qwen3.5-2B", "Qwen2.5-7B-Instruct"]

# ------------------------------------------------------------------ fig_revision_dynamics (main body)
dd, ds_, da = AN["dyn_depth"], AN["dyn_stage"], AN["dyn_anytime"]
lines = [header("fig_revision_dynamics",
                "Revision depth, revision stage, and budget progress of ARR (Qwen2.5-Math-7B, MATH500, B = 800)."),
         f"k      = {mstr(dd['k'])};   % (a) revisions of a root",
         f"accRev = {mstr(dd['revised'])};   % (a) accuracy (%) of roots after k revisions (roots that reach depth k)",
         f"accIni = {mstr(dd['initial'])};   % (a) accuracy (%) of the same roots before revision (k = 0)",
         f"accNew = {dd['new_sample']:.4g};   % (a) accuracy (%) of new samples (replacement roots)",
         f"share  = {mstr(dd['share'])};   % (a) roots that reach depth k (% of all roots)",
         f"j      = {mstr(ds_['j'])};   % (b) revision index (6 = 6 or later)",
         f"w2c    = {mstr(ds_['w2c'])};   % (b) % of complete revisions that change a wrong answer to the correct one",
         f"c2w    = {mstr(ds_['c2w'])};   % (b) % of complete revisions that change a correct answer to a wrong one",
         f"fp     = {mstr(ds_['fp'])};   % (b) % of complete revisions that are fixed points",
         f"prog   = {mstr(da['p'])};   % (c) budget spent (% of B)",
         f"anyAcc = {mat([da[m] for m in ('arr', 'replace_ordinary', 'arr_no_replace', 'maj')])};   % (c) accuracy (%) of the running vote; rows: ARR, ARR_AR, ARR_Depth, MV",
         r"""
f = newFig(S, 4.4);
t = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
cG = [0.62 0.62 0.62];

% (a) accuracy of a root against its revision depth, and the alternative: a new sample
ax = nexttile(t); styleAxes(ax, S, '(a) Revision depth');
yyaxis(ax, 'right');
bar(ax, k, share, 0.55, 'FaceColor', [0.88 0.88 0.88], 'EdgeColor', 'none');
set(ax, 'YLim', [0 400], 'YTick', [0 50 100]); ax.YAxis(2).Color = cG;
yyaxis(ax, 'left'); ax.YAxis(1).Color = [0.2 0.2 0.2];
h1 = plot(ax, k, accRev, '-o', 'Color', S.col(1,:), 'LineWidth', 1.6, 'MarkerSize', 3.5, 'MarkerFaceColor', S.col(1,:));
h2 = plot(ax, k, accIni, '--s', 'Color', S.col(4,:), 'LineWidth', 1.1, 'MarkerSize', 3.2, 'MarkerFaceColor', S.col(4,:));
h3 = plot(ax, [k(1) k(end)], [accNew accNew], ':', 'Color', S.col(8,:), 'LineWidth', 1.5);
set(ax, 'XLim', [-0.4 k(end) + 0.4], 'XTick', k, 'YLim', [70 95], 'YTick', 70:5:95);
xlabel(ax, 'Revisions $k$ of a root'); ylabel(ax, 'Accuracy of a root (\%)');

% (b) answer changes and fixed points at each revision index
ax = nexttile(t); styleAxes(ax, S, '(b) Revision stage');
yyaxis(ax, 'right');
bar(ax, j, fp, 0.55, 'FaceColor', [0.88 0.88 0.88], 'EdgeColor', 'none');
set(ax, 'YLim', [0 160], 'YTick', [0 25 50]); ax.YAxis(2).Color = cG;
yyaxis(ax, 'left'); ax.YAxis(1).Color = [0.2 0.2 0.2];
h1 = plot(ax, j, w2c, '-^', 'Color', S.col(6,:), 'LineWidth', 1.5, 'MarkerSize', 3.8, 'MarkerFaceColor', S.col(6,:));
h2 = plot(ax, j, c2w, '-v', 'Color', S.col(7,:), 'LineWidth', 1.5, 'MarkerSize', 3.8, 'MarkerFaceColor', S.col(7,:));
set(ax, 'XLim', [0.5 j(end) + 0.5], 'XTick', j, 'XTickLabel', [arrayfun(@(x) sprintf('%d', x), j(1:end-1), 'UniformOutput', false) {'$\geq$6'}], ...
    'YLim', [0 2.0]);
xlabel(ax, 'Revision index of a root'); ylabel(ax, 'Answer changes (\%)');

% (c) accuracy of the running vote as the budget is spent
ax = nexttile(t); styleAxes(ax, S, '(c) Budget progress'); ax1 = ax;
ks = [1 2 4 8];
for i = numel(ks):-1:1
    plot(ax, prog, anyAcc(i,:), '-', 'Color', S.col(ks(i),:), 'LineWidth', S.lw(ks(i)));
    plot(ax, prog(4:4:end), anyAcc(i,4:4:end), S.mk{ks(i)}, 'Color', S.col(ks(i),:), 'MarkerSize', S.ms(ks(i)), ...
        'MarkerFaceColor', S.col(ks(i),:), 'LineStyle', 'none');
end
set(ax, 'XLim', [0 102], 'XTick', 0:25:100, 'YLim', [80 85.5]);
xlabel(ax, 'Budget spent (\% of $B$)'); ylabel(ax, 'Vote accuracy (\%)');
% one legend below the panels: the lines of (a) and (b), then the methods of (c)
axL = axes(f, 'Visible', 'off', 'Position', [0 0 0.01 0.01]); hold(axL, 'on');   % legends need figure-level axes
hl = gobjects(9, 1);
hl(1) = plot(axL, NaN, NaN, '-o', 'Color', S.col(1,:), 'LineWidth', 1.6, 'MarkerSize', 3.5, 'MarkerFaceColor', S.col(1,:));
hl(2) = plot(axL, NaN, NaN, '--s', 'Color', S.col(4,:), 'LineWidth', 1.1, 'MarkerSize', 3.2, 'MarkerFaceColor', S.col(4,:));
hl(3) = plot(axL, NaN, NaN, ':', 'Color', S.col(8,:), 'LineWidth', 1.5);
hl(4) = plot(axL, NaN, NaN, '-^', 'Color', S.col(6,:), 'LineWidth', 1.5, 'MarkerSize', 3.8, 'MarkerFaceColor', S.col(6,:));
hl(5) = plot(axL, NaN, NaN, '-v', 'Color', S.col(7,:), 'LineWidth', 1.5, 'MarkerSize', 3.8, 'MarkerFaceColor', S.col(7,:));
for i = 1:numel(ks)
    hl(5 + i) = plot(axL, NaN, NaN, ['-' S.mk{ks(i)}], 'Color', S.col(ks(i),:), 'LineWidth', S.lw(ks(i)), ...
        'MarkerSize', S.ms(ks(i)), 'MarkerFaceColor', S.col(ks(i),:));
end
lg1 = legend(axL, hl(1:5), {'(a) after $k$ revisions', '(a) same roots at $k{=}0$', '(a) new sample', ...
    '(b) wrong $\rightarrow$ correct', '(b) correct $\rightarrow$ wrong'}, ...
    'Orientation', 'horizontal', 'NumColumns', 5, 'Box', 'off', 'Interpreter', 'latex', 'FontSize', S.fs - 1.5);
lg1.ItemTokenSize = [10 6];
ax2 = axes(f, 'Visible', 'off', 'Position', [0 0 0.01 0.01]); hold(ax2, 'on');   % second legend needs its own axes
h2 = gobjects(numel(ks), 1);
for i = 1:numel(ks)
    h2(i) = plot(ax2, NaN, NaN, ['-' S.mk{ks(i)}], 'Color', S.col(ks(i),:), 'LineWidth', S.lw(ks(i)), ...
        'MarkerSize', S.ms(ks(i)), 'MarkerFaceColor', S.col(ks(i),:));
end
lg2 = legend(ax2, h2, strcat('(c)~', S.lbl(ks)), 'Orientation', 'horizontal', 'NumColumns', numel(ks), ...
    'Box', 'off', 'Interpreter', 'latex', 'FontSize', S.fs - 1.5);
lg2.ItemTokenSize = [10 6];
stackLegends(t, [lg1 lg2]);
saveFig(f, 'fig_revision_dynamics');
end
""" + STYLE]
files["fig_revision_dynamics.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_regimes (main body): when revision should get compute
reg = AN["regimes"]
FORMAT = {("gemma-4-E4B-it", "math500"), ("gemma-4-E4B-it", "olympiadbench")}   # revision repairs a missing \boxed answer
stop = [r for r in AN["stopping"] if r["budget"] == max(x["budget"] for x in AN["stopping"] if x["dataset"] == r["dataset"])]
SLBL = {"math500": "MATH500", "olympiadbench": "OlympiadBench", "amc23": "AMC23", "aime24": "AIME24", "aime25": "AIME25",
        "gsm8k": "GSM8K"}
lines = [header("fig_regimes",
                "When revision should receive compute: revision quality against ARR - MV, and the stopping rule."),
         "% (a) one point per model x benchmark at the largest budget (complete for ARR and MV)",
         f"mdl    = {cellstr([MSHORT2[m] for m in REGMODELS])};",
         f"regM   = {mstr([REGMODELS.index(r['model']) + 1 for r in reg])};   % model index",
         f"regX   = {mstr([r['net_repair'] for r in reg])};   % net repair of F_x: (W->C - C->W) per complete revision (%)",
         f"regY   = {mstr([r['gap'] for r in reg])};   % accuracy of ARR - MV (points)",
         f"regLB  = {mstr([float(r['dataset'] == 'livebench_reasoning') for r in reg])};   % 1 = LiveBench",
         f"regFmt = {mstr([float((r['model'], r['dataset']) in FORMAT) for r in reg])};   % 1 = gain from a repaired answer format",
         "% (b), (c) stopping rule at the largest budget of each benchmark, paired by question with ARR",
         f"bench  = {cellstr([SLBL[r['dataset']] for r in stop])};",
         f"dAns   = {mat([r['delta']['arr_answer'] for r in stop])};   % answer-level stop - ARR: mean, CI low, CI high",
         f"dOnce  = {mat([r['delta']['arr_once'] for r in stop])};   % one revision - ARR",
         f"dMaj   = {mat([r['delta']['maj'] for r in stop])};   % MV - ARR",
         f"rootsR = {mat([[r['roots']['arr_answer'] / r['roots']['arr'], r['roots']['arr_once'] / r['roots']['arr']] for r in stop])};   % roots per question / ARR",
         f"passR  = {mat([[r['passes']['arr_answer'] / r['passes']['arr'], r['passes']['arr_once'] / r['passes']['arr']] for r in stop])};   % median serial passes / ARR",
         r"""
f = newFig(S, 4.7);
t = tiledlayout(f, 1, 10, 'TileSpacing', 'compact', 'Padding', 'compact');   % spans 3 : 4 : 3; (b) needs room for its labels
pal = [0.80 0.11 0.16; 0.16 0.37 0.75; 0.10 0.56 0.47; 0.95 0.56 0.16; 0.52 0.38 0.72; 0.45 0.45 0.45; 0.55 0.40 0.30];
mk = {'o', 's', 'd', '^', 'v', 'p', 'h'};

% (a) net repair of the revision operator against the accuracy difference ARR - MV
ax = nexttile(t, 1, [1 3]); styleAxes(ax, S, '(a) Revision quality'); ax1 = ax;
set(ax, 'XGrid', 'on');
c = polyfit(regX(~regFmt), regY(~regFmt), 1); xx = linspace(min(regX), max(regX), 50);
plot(ax, xx, polyval(c, xx), '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.8);
plot(ax, [min(regX) max(regX)], [0 0], '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.5);
plot(ax, [0 0], [-26 17], '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.5);
hm = gobjects(numel(mdl), 1);
for i = 1:numel(regX)
    m = regM(i);
    h = plot(ax, regX(i), regY(i), mk{m}, 'MarkerSize', 4.5, 'Color', pal(m,:), 'LineWidth', 0.9);
    if regLB(i), h.MarkerFaceColor = 'w'; else, h.MarkerFaceColor = pal(m,:); end
    if regFmt(i), plot(ax, regX(i), regY(i), 'o', 'MarkerSize', 9, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.6); end
    hm(m) = h;
end
r = corrcoef(regX, regY);
text(ax, min(regX) + 0.3, 13.5, sprintf('$r = %.2f$', r(1,2)), 'FontSize', S.fs - 1);
set(ax, 'YLim', [-26 17], 'XLim', [-13.5 2]);
xlabel(ax, 'Net repair of $F_x$ (\% of revisions)'); ylabel(ax, {'ARR $-$ Majority Voting', '(points)'});
axL = axes(f, 'Visible', 'off', 'Position', [0 0 0.01 0.01]); hold(axL, 'on');   % legends need figure-level axes
for m = 1:numel(mdl), hm(m) = plot(axL, NaN, NaN, mk{m}, 'MarkerSize', 4.5, 'Color', pal(m,:), 'MarkerFaceColor', pal(m,:)); end
lgM = legend(axL, hm, strcat('(a)~', mdl), 'Orientation', 'horizontal', 'NumColumns', numel(mdl), 'Box', 'off', ...
    'Interpreter', 'latex', 'FontSize', S.fs - 2);
lgM.ItemTokenSize = [6 6];

% (b) accuracy of the other stopping rules relative to the exact fixed point
ax = nexttile(t, 4, [1 4]); styleAxes(ax, S, '(b) Stopping rule'); axB = ax;
set(ax, 'YGrid', 'off', 'XGrid', 'on');
n = numel(bench); y = n:-1:1; D = {dAns, dOnce, dMaj}; dy = [0.22 0 -0.22];
cc = [S.col(2,:); S.col(4,:); S.col(8,:)]; mm = {'^', 'v', 'h'};
plot(ax, [0 0], [0.4 n + 0.6], '-', 'Color', S.col(1,:), 'LineWidth', 0.8);
hs = gobjects(3, 1);
for s = 1:3
    for i = 1:n
        plot(ax, D{s}(i,2:3), y(i) + dy(s) * [1 1], '-', 'Color', cc(s,:), 'LineWidth', 0.9);
    end
    hs(s) = plot(ax, D{s}(:,1), y + dy(s), mm{s}, 'MarkerSize', 4.5, 'Color', cc(s,:), 'MarkerFaceColor', cc(s,:), 'LineStyle', 'none');
end
set(ax, 'YTick', fliplr(y), 'YTickLabel', fliplr(bench), 'YLim', [0.4 n + 0.6], 'FontSize', S.fs - 0.5);
xlabel(ax, 'Accuracy $-$ ARR (points)');

% (c) what the other stopping rules buy: more roots for more serial passes
ax = nexttile(t, 8, [1 3]); styleAxes(ax, S, '(c) Breadth and latency'); axC = ax;
set(ax, 'XGrid', 'on');
plot(ax, [1 1], [0.8 2.6], '-', 'Color', S.col(1,:), 'LineWidth', 0.8);
plot(ax, [0.9 1.4], [1 1], '-', 'Color', S.col(1,:), 'LineWidth', 0.8);
plot(ax, passR(:,1), rootsR(:,1), '^', 'MarkerSize', 4.5, 'Color', cc(1,:), 'MarkerFaceColor', cc(1,:), 'LineStyle', 'none');
plot(ax, passR(:,2), rootsR(:,2), 'v', 'MarkerSize', 4.5, 'Color', cc(2,:), 'MarkerFaceColor', cc(2,:), 'LineStyle', 'none');
plot(ax, 1, 1, 'o', 'MarkerSize', 5, 'Color', S.col(1,:), 'MarkerFaceColor', S.col(1,:));
text(ax, 1.01, 0.92, 'ARR', 'Color', S.col(1,:), 'FontSize', S.fs - 1.5);
axL2 = axes(f, 'Visible', 'off', 'Position', [0 0 0.01 0.01]); hold(axL2, 'on');
hc = [plot(axL2, NaN, NaN, '^', 'MarkerSize', 4.5, 'Color', cc(1,:), 'MarkerFaceColor', cc(1,:)), ...
      plot(axL2, NaN, NaN, 'v', 'MarkerSize', 4.5, 'Color', cc(2,:), 'MarkerFaceColor', cc(2,:)), ...
      plot(axL2, NaN, NaN, 'h', 'MarkerSize', 4.5, 'Color', cc(3,:), 'MarkerFaceColor', cc(3,:))];
lgS = legend(axL2, hc, {'(b, c) stop when the answer is stable', '(b, c) stop after one revision', '(b) Majority Voting'}, ...
    'Orientation', 'horizontal', 'NumColumns', 3, 'Box', 'off', 'Interpreter', 'latex', 'FontSize', S.fs - 2);
lgS.ItemTokenSize = [6 6];
set(ax, 'XLim', [0.9 1.4], 'YLim', [0.8 2.7], 'YTick', 1:0.5:2.5);
xlabel(ax, 'Serial passes / ARR'); ylabel(ax, 'Roots / ARR');
stackLegends(t, [lgM lgS]);
% center (b) between (a) and (c): equal white space on both sides, labels included
drawnow;
A = [ax1 axB axC]; set(A, 'Units', 'normalized'); to = t.OuterPosition;   % axes positions are relative to the layout box
P = cell(1, 3); L = zeros(1, 3); R = zeros(1, 3);
for k = 1:3
    p = A(k).Position; ti = A(k).TightInset;
    P{k} = [to(1) + p(1) * to(3), to(2) + p(2) * to(4), p(3) * to(3), p(4) * to(4)];
    L(k) = P{k}(1) - ti(1) * to(3); R(k) = P{k}(1) + P{k}(3) + ti(3) * to(3);
end
dx = ((L(3) - R(2)) - (L(2) - R(1))) / 2; P{2}(1) = P{2}(1) + dx;
for k = 1:3, A(k).Parent = f; A(k).Units = 'normalized'; A(k).Position = P{k}; end   % freeze all three, then move (b)
saveFig(f, 'fig_regimes');
end
""" + STYLE]
files["fig_regimes.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_ablations (appendix): N, temperature, budget, W
ab = AN["ablations"]


def abl_acc(var, d, b, m, key="acc"):
    for r in ab.get(var, []):
        if r["dataset"] == d and r["budget"] == b and r["method"] == m:
            return r[key]
    return float("nan")


SEED0 = {}
for _r in csv.DictReader(open(SRC)):
    if _r["variant"] == "main" and _r["model"] == P:
        for _tok in _r["per_seed"].split():
            _s, _kn = _tok.split(":", 1)
            _k, _n = _kn.split("/")
            if _s in ("0", "s0") and int(_n) == int(_r["dataset_size"]):
                SEED0[(_r["dataset"], float(_r["budget"]), _r["method"])] = 100.0 * float(_k) / float(_n)


def base0(d, b, m, key="acc"):
    """Seed 0 of the main run (accuracy from all_results.csv; serial passes from an ablation row that stored it)."""
    if key == "acc" and (d, b, m) in SEED0:
        return SEED0[(d, b, m)]
    for rows in ab.values():
        for r in rows:
            if r["dataset"] == d and r["budget"] == b and r["method"] == m and r["base"] == r["base"]:
                return r["base"] if key == "acc" else r["base_passes"]
    return float("nan")


NB = 400.0   # every N and temperature cell is complete at B = 400
nAcc = [[abl_acc("n2", "math500", NB, m), base0("math500", NB, m), abl_acc("n8", "math500", NB, m)] for m in ("arr", "maj")]
nPas = [[abl_acc("n2", "math500", NB, m, "passes"), base0("math500", NB, m, "passes"), abl_acc("n8", "math500", NB, m, "passes")]
        for m in ("arr", "maj")]
TD = ["math500", "amc23", "aime24"]
tAcc = [[statistics.mean(abl_acc("temp05", d, NB, m) for d in TD), statistics.mean(base0(d, NB, m) for d in TD),
         statistics.mean(abl_acc("temp10", d, NB, m) for d in TD)] for m in ("arr", "maj")]
BD = ["amc23", "aime24", "aime25"]
BM = ["arr", "replace_ordinary", "arr_no_replace", "maj"]
bAcc = [[statistics.mean(base0(d, b, m) for d in BD) for b in (200.0, 400.0, 800.0)]
        + [statistics.mean(abl_acc("budget1600", d, 1600.0, m) for d in BD)] for m in BM]
WS = [8, 16, 32, 64]
wCells = [("amc23", 800.0), ("amc23", 400.0), ("amc23", 200.0), ("math500", 200.0)]
wRat = [[(abl_acc(f"w{w}", d, b, "arr", "passes") if w != 32 else base0(d, b, "arr", "passes")) / base0(d, b, "arr", "passes")
         for w in WS] for d, b in wCells]
lines = [header("fig_ablations",
                "Sensitivity of ARR to the width N, the sampling temperature, the budget B and the replay window W."),
         "% Qwen2.5-Math-7B-Instruct; the default (N = 4, tau = 0.7, W = 32) is the main run",
         f"nN   = [2 4 8];",
         f"nAcc = {mat(nAcc)};   % (a) MATH500, B = 400: accuracy (%); rows: ARR, MV",
         f"nPas = {mat(nPas, 6)};   % (a) median serial passes",
         f"tau  = [0.5 0.7 1.0];",
         f"tAcc = {mat(tAcc)};   % (b) mean accuracy (%) over MATH500, AMC23, AIME24 at B = 400; rows: ARR, MV",
         f"bud  = [200 400 800 1600];",
         f"bAcc = {mat(bAcc)};   % (c) mean accuracy (%) over AMC23, AIME24, AIME25; rows: ARR, ARR_AR, ARR_Depth, MV",
         f"W    = {mstr(WS)};",
         f"wRat = {mat(wRat)};   % (d) median serial passes / W = 32; rows: AMC23 B = 800, 400, 200; MATH500 B = 200",
         r"""
f = newFig(S, 5.2);
t = tiledlayout(f, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
km = [1 8];

ax = nexttile(t); styleAxes(ax, S, '(a) Width $N$'); ax1 = ax;
for i = 2:-1:1
    plot(ax, nPas(i,:), nAcc(i,:), ['-' S.mk{km(i)}], 'Color', S.col(km(i),:), 'LineWidth', S.lw(km(i)), ...
        'MarkerSize', S.ms(km(i)), 'MarkerFaceColor', S.col(km(i),:));
    for q = 1:3
        text(ax, nPas(i,q), nAcc(i,q) + 0.18, sprintf('$N{=}%d$', nN(q)), 'FontSize', S.fs - 2.5, 'Color', S.col(km(i),:), 'HorizontalAlignment', 'center');
    end
end
set(ax, 'YLim', [83 85.5], 'XScale', 'log', 'XLim', [1200 16000], 'XTick', [2000 4000 8000], 'XTickLabel', {'2k', '4k', '8k'}, 'XMinorTick', 'off');
xlabel(ax, 'Serial passes (MATH500)'); ylabel(ax, 'Accuracy (\%)');

ax = nexttile(t); styleAxes(ax, S, '(b) Temperature $\tau$');
for i = 2:-1:1
    plot(ax, tau, tAcc(i,:), ['-' S.mk{km(i)}], 'Color', S.col(km(i),:), 'LineWidth', S.lw(km(i)), ...
        'MarkerSize', S.ms(km(i)), 'MarkerFaceColor', S.col(km(i),:));
end
set(ax, 'XLim', [0.4 1.1], 'XTick', tau);
xlabel(ax, 'Sampling temperature $\tau$');

ax = nexttile(t); styleAxes(ax, S, '(c) Budget $B$');
kb = [1 2 4 8];
for i = numel(kb):-1:1
    drawMethod(ax, S, kb(i), bud, bAcc(i,:));
end
set(ax, 'XScale', 'log', 'XLim', [170 1900], 'XTick', bud, 'XMinorTick', 'off');
xlabel(ax, 'Budget $B$ (TFLOPs)');

ax = nexttile(t); styleAxes(ax, S, '(d) Replay window $W$');
lw = {'-o', '-s', '-d', '-^'}; gs = [0.15 0.35 0.55 0.7];
for i = 1:size(wRat, 1)
    plot(ax, W, wRat(i,:), lw{i}, 'Color', S.col(1,:) * (1 - gs(i)) + gs(i), 'LineWidth', 1.1, 'MarkerSize', 3.5, ...
        'MarkerFaceColor', S.col(1,:) * (1 - gs(i)) + gs(i));
end
plot(ax, [6 80], [1 1], ':', 'Color', [0.4 0.4 0.4]);
set(ax, 'XScale', 'log', 'XLim', [6 80], 'XTick', W, 'XMinorTick', 'off', 'YLim', [0.9 1.2]);
xlabel(ax, 'Window $W$'); ylabel(ax, 'Serial passes / $W{=}32$');
topLegend(t, ax1, S, [1 2 4 8]);
saveFig(f, 'fig_ablations');
end
""" + STYLE]
files["fig_ablations.m"] = "\n".join(lines)

order = ["fig_motivation", "fig_latency_accuracy", "fig_depth_breadth", "fig_depth_breadth_accuracy", "fig_seed_consistency", "fig_accuracy_budget", "fig_compute_use", "fig_replay_efficiency", "fig_revision_dynamics", "fig_regimes", "fig_abl_temperature", "fig_abl_budget", "fig_abl_window", "fig_abl_prompts", "fig_abl_width", "fig_abl_passes", "fig_models", "fig_models_lines"]
files["make_all_figures.m"] = ("%MAKE_ALL_FIGURES  Draw all paper figures (each figure file has its values hard-coded).\n"
                               f"%   Generated {STAMP}.\n" + "\n".join(f"{n}();" for n in order) + "\nclose all;\n")

# ------------------------------------------------------------------ LaTeX table of all completed settings
# rows of the table now in main.tex: a row that is new or whose values changed is printed in purple
_old = open(os.path.join(ROOT, "overleaf_paper", "main.tex")).read()
_old = _old[_old.find("\\label{tab:models}"):]
_old = {" ".join(l.replace("\\textbf{", "").replace("}", "").replace("\\textcolor{purple{", "").split())
        for l in _old[:_old.find("\\bottomrule")].splitlines() if l.count("&") >= 10}
trows = []
for m in MODELS:
    if m == P:          # the primary model is in fig_accuracy_budget; a table must not repeat figure values
        continue
    for d in MATH + ["livebench_reasoning"]:
        if not full_setting(m, d):
            continue
        b = budgets(m, d)[-1]
        v = [C[(m, d, b, me)]["acc"] for me in METHODS]
        best = max(v)
        cells = [(r"\textbf{%.1f}" % x) if abs(x - best) < 1e-9 else "%.1f" % x for x in v]
        row = [MLBL[m], DLBL[d], f"{b:g}", str(C[(m, d, b, 'arr')]['seeds'])] + cells
        plain = " ".join((" & ".join(row) + r" \\").replace("\\textbf{", "").replace("}", "").split())
        if plain not in _old:
            row = [r"\textcolor{purple}{%s}" % c for c in row]
        trows.append(" & ".join(row) + r" \\")
tab = ("\\begin{tabular}{ll rc cccccccc}\n\\toprule\nModel & Benchmark & $B$ & Seeds & "
       + " & ".join(TEXLBL) + " \\\\\n\\midrule\n" + "\n".join(trows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ LaTeX tables of every finished ablation cell (appendix)
def _seed_acc(r):
    size, out = int(r["dataset_size"]), {}
    for tok in r["per_seed"].split():
        s, kn = tok.split(":", 1)
        k, n = kn.split("/")
        if int(n) == size:
            out[s.lstrip("s")] = 100.0 * float(k) / float(n)
    return out


ALLR = list(csv.DictReader(open(SRC)))
VCELL = defaultdict(dict) if "defaultdict" in dir() else None
from collections import defaultdict as _dd
VCELL = _dd(dict)          # (variant, model, dataset, budget) -> method -> {seed: acc}
VPASS = _dd(dict)          # (variant, model, dataset, budget) -> method -> median serial passes
for _r in ALLR:                     # main grid: summary files are final
    sa = _seed_acc(_r)
    if sa and _r["variant"] == "main":
        key = (_r["variant"], _r["model"], _r["dataset"], float(_r["budget"]))
        VCELL[key][_r["method"]] = sa
        VPASS[key][_r["method"]] = float(_r["critical_path_passes_p50"] or "nan")
for _k, _v in AN.get("vcells", {}).items():   # ablations: from the question-level records (sharded summaries can be stale)
    _var, _m, _d, _b, _me = _k.split("|")
    VCELL[(_var, _m, _d, float(_b))][_me] = {s_: x["acc"] for s_, x in _v.items()}

AMETH = ["arr", "replace_ordinary", "replace_pld", "arr_no_replace", "direct_fixed", "critic_correct", "continuation",
         "maj", "arr_answer", "arr_once"]
ALBL = [r"\textbf{ARR}", r"\ARRar", r"\ARRpld", r"\ARRdepth", r"\FDR", r"\VC", r"\SeqC", r"\SC", "Answer", "Once"]
DORD = {d: i for i, d in enumerate(DSETS)}
MORD = {m: i for i, m in enumerate(MODELS)}
pu = lambda x: r"\textcolor{purple}{%s}" % x


def ablation_rows(variants, methods, label, keep=None, use_ref=True):
    """Rows of finished cells with every listed method complete (no missing value).
    Each value is the mean over its complete seeds; a value with fewer seeds than the row carries its own count
    as a superscript. The default ARR uses the same seeds when it has all of them; otherwise it uses all of its
    seeds, and the Seeds column shows (ablation seeds / default seeds)."""
    rows = []
    keys = sorted({k[1:] for k in VCELL if k[0] in variants}, key=lambda k: (MORD.get(k[0], 99), DORD.get(k[1], 99), k[2]))
    for (m, d, b) in keys:
        if keep is not None and not keep(m, d, b):
            continue
        if not all(len({me for me, x in VCELL.get((v, m, d, b), {}).items() if me in methods and x}) == len(methods)
                   for v in variants):
            continue                                        # same comparison: every setting complete at this cell
        for v in variants:
            cell = {me: x for me, x in VCELL.get((v, m, d, b), {}).items() if me in methods and x}
            if len(cell) < len(methods):
                continue                                    # a method of the row is missing or still running
            seeds = sorted(set().union(*[set(x) for x in cell.values()]), key=int)
            base = VCELL.get(("main", m, d, b), {}).get("arr", {})
            if base and all(s_ in base for s_ in seeds):
                ref, nseed = statistics.mean(base[s_] for s_ in seeds), str(len(seeds))
            elif base:
                ref, nseed = statistics.mean(base.values()), f"{len(seeds)} / {len(base)}"
            else:
                ref, nseed = float("nan"), str(len(seeds))
            if use_ref and ref != ref:
                continue
            vals = []
            for me in methods:
                x = cell[me]
                txt = "%.1f" % statistics.mean(x.values())
                vals.append(txt if len(x) == len(seeds) else txt + "$^{(%d)}$" % len(x))
            row = [label(v), (MLBL.get(m, m) + ", " if m != P else "") + DLBL.get(d, d), f"{b:g}", nseed] \
                + (["%.1f" % ref] if use_ref else []) + vals
            rows.append(" & ".join(pu(c) for c in row) + r" \\")
    return rows


VLBL = {"n2": "$N{=}2$", "n8": "$N{=}8$", "abl_stop": "Stop rule", "passes": "Passes"}
TABS = {   # name: (variants, methods, keep, default-ARR column)
    "stop": (["abl_stop"], ["arr_answer", "arr_once"], lambda m, d, b: m != P, True),   # Qwen2.5-Math-7B is in tab:stop
}
abl_tex = []
for name, (vs, ms, keep, use_ref) in TABS.items():
    head = ["Setting", "Benchmark", "$B$", "Seeds"] + (["ARR (default)"] if use_ref else []) + [ALBL[AMETH.index(m)] for m in ms]
    body = ablation_rows(vs, ms, lambda v: VLBL[v], keep=keep, use_ref=use_ref)
    cols = "ll rc " + ("c " if use_ref else "") + "c" * len(ms)
    abl_tex.append(f"% ==== {name} ({len(body)} rows)\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
                   + " & ".join(pu(h) for h in head) + " \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ per-cell ablation figures (appendix), from the question-level records
def vacc(var, d, b, m, seed="0", model=P):
    """Accuracy (%) of one ablation cell on one seed; the default setting is seed `seed` of the main run."""
    if var == "main":
        x = VCELL.get(("main", model, d, float(b)), {}).get(m, {})
    else:
        x = VCELL.get((var, model, d, float(b)), {}).get(m, {})
    return x.get(seed, float("nan"))


def vpass(var, d, b, m="arr", seed="0", model=P):
    if var == "main":
        for _v, _rows in AN["ablations"].items():          # main seed-0 passes are stored next to each ablation row
            for r in _rows:
                if r["dataset"] == d and r["budget"] == float(b) and r["method"] == m and r["base_passes"] == r["base_passes"]:
                    return r["base_passes"]
        return float("nan")
    x = AN.get("vcells", {}).get(f"{var}|{model}|{d}|{float(b):g}|{m}", {})
    return x.get(seed, {}).get("passes", float("nan"))


BUD3 = [200, 400, 800]
# (1) temperature and budget
TBENCH = ["math500", "amc23", "aime24"]
_tok = lambda d, b: all(vacc(v, d, b, m) == vacc(v, d, b, m) for v in ("temp05", "main", "temp10") for m in ("arr", "maj"))
tA = [[[vacc(v, d, b, m) if _tok(d, b) else float("nan") for b in BUD3] for v in ("temp05", "main", "temp10")]
      for d in TBENCH for m in ("arr", "maj")]     # same comparison: every temperature and both methods at the same cell
BBENCH = [d for d in ("math500", "amc23", "aime24", "aime25")
          if all(vacc("budget1600", d, 1600, m) == vacc("budget1600", d, 1600, m)
                 for m in ("arr", "replace_ordinary", "arr_no_replace", "maj"))]
BB = [200, 400, 800, 1600]
bA = [[[vacc("main" if b < 1600 else "budget1600", d, b, m) for b in BB] for m in ("arr", "replace_ordinary", "arr_no_replace", "maj")]
      for d in BBENCH]
# (1a) temperature: change of accuracy against the default tau = 0.7, one row per benchmark and budget
TROWS = [(d, b) for d in TBENCH for b in BUD3 if _tok(d, b)]
tD = {q: [[vacc(v, d, b, m) - vacc("main", d, b, m) for m in ("arr", "maj")] for d, b in TROWS]
      for q, v in (("05", "temp05"), ("10", "temp10"))}
lines = [header("fig_abl_temperature", "Sampling temperature: accuracy change against the default tau = 0.7 (Qwen2.5-Math-7B)."),
         f"cells = {cellstr([f'{DLBL[d]}, {b}' for d, b in TROWS])};   % benchmark, budget B (TFLOPs)",
         f"d05 = {mat(tD['05'])};   % (a) tau = 0.5 minus tau = 0.7 (points); columns: ARR, MV",
         f"d10 = {mat(tD['10'])};   % (b) tau = 1.0 minus tau = 0.7 (points); columns: ARR, MV",
         r"""
f = newFig(S, 6.2);
t = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
D = {d05, d10}; ttl = {'(a) $\tau{=}0.5$', '(b) $\tau{=}1.0$'};
n = numel(cells); y = n:-1:1; dy = [0.17 -0.17]; km = [1 8];
lim = max(abs([d05(:); d10(:)])) + 1.5;
for p = 1:2
    ax = nexttile(t); styleAxes(ax, S, ttl{p}); if p == 1, ax1 = ax; end
    set(ax, 'YGrid', 'off', 'XGrid', 'on');
    plot(ax, [0 0], [0.4 n + 0.6], '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.6);
    for k = 1:2
        c = S.col(km(k),:);
        for i = 1:n
            plot(ax, [0 D{p}(i,k)], y(i) + dy(k) * [1 1], '-', 'Color', c, 'LineWidth', 1.4);
        end
        plot(ax, D{p}(:,k), y + dy(k), S.mk{km(k)}, 'MarkerSize', S.ms(km(k)) + 0.6, 'Color', c, ...
            'MarkerFaceColor', c, 'LineStyle', 'none');
    end
    set(ax, 'XLim', [-lim lim], 'YLim', [0.4 n + 0.6], 'YTick', fliplr(y), 'YTickLabel', fliplr(cells));
    if p > 1, set(ax, 'YTickLabel', {}); end
    xlabel(ax, 'Accuracy $-$ accuracy at $\tau{=}0.7$ (points)');
end
topLegend(t, ax1, S, km);
saveFig(f, 'fig_abl_temperature');
end
""" + STYLE]
files["fig_abl_temperature.m"] = "\n".join(lines)

# (1b) budget extension to B = 1,600
lines = [header("fig_abl_budget", "Accuracy up to B = 1,600 (Qwen2.5-Math-7B; B <= 800 is the main run)."),
         f"bud = {mstr(BB)};",
         f"bName = {cellstr([DLBL[d] for d in BBENCH])};",
         "% bAcc{d}: rows ARR, ARR_AR, ARR_Depth, MV; columns = budgets",
         "bAcc = cell(numel(bName), 1);"]
for i, d in enumerate(BBENCH):
    lines.append(f"bAcc{{{i + 1}}} = {mat(bA[i])};")
lines.append(r"""
f = newFig(S, 5.4);
t = tiledlayout(f, 1, numel(bName), 'TileSpacing', 'compact', 'Padding', 'compact');
ks = [1 2 4 8];
for d = 1:numel(bName)
    ax = nexttile(t); styleAxes(ax, S, sprintf('(%s) %s', char('a' + d - 1), bName{d})); if d == 1, ax1 = ax; end
    for i = numel(ks):-1:1
        drawMethod(ax, S, ks(i), bud, bAcc{d}(i,:));
    end
    set(ax, 'XScale', 'log', 'XTick', bud, 'XLim', [170 1900], 'XMinorTick', 'off');
end
xlabel(t, 'Budget $B$ (TFLOPs per question, log scale)', 'Interpreter', 'latex', 'FontSize', S.fs);
ylabel(t, 'Accuracy (\%)', 'Interpreter', 'latex', 'FontSize', S.fs);
topLegend(t, ax1, S, ks);
saveFig(f, 'fig_abl_budget');
end
""" + STYLE)
files["fig_abl_budget.m"] = "\n".join(lines)

# (2) replay window W
WW = [8, 16, 32, 64]
WCELLS = [(d, b) for d, b in [("math500", 200), ("math500", 400), ("math500", 800), ("amc23", 200), ("amc23", 400), ("amc23", 800)]
          if all(vacc("main" if w == 32 else f"w{w}", d, b, "arr") == vacc("main" if w == 32 else f"w{w}", d, b, "arr")
                 for w in (8, 16, 32, 64))]
wA = [[vacc("main" if w == 32 else f"w{w}", d, b, "arr") for w in WW] for d, b in WCELLS]
wP = [[(vpass("main" if w == 32 else f"w{w}", d, b)) for w in WW] for d, b in WCELLS]
wR = [[p / r[2] if r[2] == r[2] else float("nan") for p in r] for r in wP]
lines = [header("fig_abl_window", "Block size W of replay for every finished cell (Qwen2.5-Math-7B)."),
         f"W = {mstr(WW)};",
         f"cellName = {cellstr([f'{DLBL[d]}, $B{{=}}{b}$' for d, b in WCELLS])};",
         f"wAcc = {mat(wA)};   % accuracy (%); rows = cells, columns = W",
         f"wRat = {mat(wR)};   % median serial passes / W = 32",
         f"wBen = {mstr([1.0 if d == 'math500' else 2.0 for d, b in WCELLS])};   % 1 = MATH500 (red), 2 = AMC23 (blue)",
         f"wBud = {mstr([float({200: 1, 400: 2, 800: 3}[b]) for d, b in WCELLS])};   % 1, 2, 3 = B = 200, 400, 800 (marker, shade)",
         r"""
f = newFig(S, 5.4);
t = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
base = [S.col(1,:); S.col(8,:)]; shade = [0.45 0.2 0]; mks = {'o', 's', 'd'};
cc = base(wBen,:) .* (1 - shade(wBud)') + shade(wBud)';
mk = mks(wBud);
ax = nexttile(t); styleAxes(ax, S, '(a) Accuracy');
for i = 1:size(wAcc, 1)
    plot(ax, W, wAcc(i,:), ['-' mk{i}], 'Color', cc(i,:), 'LineWidth', 1.1, 'MarkerSize', 3.6, 'MarkerFaceColor', cc(i,:));
end
set(ax, 'XScale', 'log', 'XTick', W, 'XLim', [6.5 80], 'XMinorTick', 'off');
xlabel(ax, 'Block size $W$'); ylabel(ax, 'Accuracy (\%)');
ax = nexttile(t); styleAxes(ax, S, '(b) Serial passes');
plot(ax, [6.5 80], [1 1], ':', 'Color', [0.4 0.4 0.4]);
for i = 1:size(wRat, 1)
    plot(ax, W, wRat(i,:), ['-' mk{i}], 'Color', cc(i,:), 'LineWidth', 1.1, 'MarkerSize', 3.6, 'MarkerFaceColor', cc(i,:));
end
set(ax, 'XScale', 'log', 'XTick', W, 'XLim', [6.5 80], 'XMinorTick', 'off');
xlabel(ax, 'Block size $W$'); ylabel(ax, 'Serial passes / $W{=}32$');
axL = axes(f, 'Visible', 'off', 'Position', [0 0 0.01 0.01]); hold(axL, 'on');
h = gobjects(size(wAcc, 1), 1);
for i = 1:size(wAcc, 1)
    h(i) = plot(axL, NaN, NaN, ['-' mk{i}], 'Color', cc(i,:), 'LineWidth', 1.1, 'MarkerSize', 3.6, 'MarkerFaceColor', cc(i,:));
end
lg = legend(axL, h, cellName, 'Orientation', 'horizontal', 'NumColumns', numel(cellName), 'Box', 'off', ...
    'Interpreter', 'latex', 'FontSize', S.fs - 2);
lg.ItemTokenSize = [8 6];
stackLegends(t, lg);
saveFig(f, 'fig_abl_window');
end
""" + STYLE]
files["fig_abl_window.m"] = "\n".join(lines)

# (3) revision-prompt paraphrases
PCELLS = [(P, d) for d in ("math500", "olympiadbench", "amc23", "aime24", "aime25", "gsm8k")] + [("Qwen2.5-7B-Instruct", "livebench_reasoning")]
pB = [[50, 100, 200] if d == "gsm8k" else BUD3 for _, d in PCELLS]
pA = [[[vacc(v, d, b, "arr", model=m) for b in bb] for v in ("main", "revprompt1", "revprompt2")] for (m, d), bb in zip(PCELLS, pB)]
lines = [header("fig_abl_prompts", "Revision prompt rho and two paraphrases for every finished cell."),
         f"pName = {cellstr([DLBL[d] + (' (' + MLBL[m] + ')' if m != P else '') for m, d in PCELLS])};",
         "% pBud{c}: budgets; pAcc{c}: rows rho, paraphrase 1, paraphrase 2; columns = budgets",
         "pBud = cell(numel(pName), 1); pAcc = cell(numel(pName), 1);"]
for i in range(len(PCELLS)):
    lines.append(f"pBud{{{i + 1}}} = {mstr(pB[i])}; pAcc{{{i + 1}}} = {mat(pA[i])};")
lines.append(r"""
f = newFig(S, 8.6);
t = tiledlayout(f, 2, 8, 'TileSpacing', 'compact', 'Padding', 'compact');   % each panel spans two columns
st = {'-o', '--s', ':d'}; cc = [S.col(1,:); S.col(2,:); S.col(4,:)]; lw = [1.6 1.1 1.1];
nTop = 4; nBot = numel(pName) - nTop; off = 8 - 2 * nBot;             % centre the second row
for c = 1:numel(pName)
    if c <= nTop, tile = 2 * c - 1; else, tile = 8 + off / 2 + 2 * (c - nTop) - 1; end
    ax = nexttile(t, tile, [1 2]); styleAxes(ax, S, sprintf('(%s) %s', char('a' + c - 1), pName{c}));
    for q = 3:-1:1
        plot(ax, pBud{c}, pAcc{c}(q,:), st{q}, 'Color', cc(q,:), 'LineWidth', lw(q), 'MarkerSize', 3.6, 'MarkerFaceColor', cc(q,:));
    end
    set(ax, 'XScale', 'log', 'XTick', pBud{c}, 'XLim', pBud{c}([1 end]) .* [0.85 1.18], 'XMinorTick', 'off');
    if c == 1 || c == nTop + 1, ylabel(ax, 'Accuracy (\%)'); end
end
xlabel(t, 'Budget $B$ (TFLOPs per question, log scale)', 'Interpreter', 'latex', 'FontSize', S.fs);
axL = axes(f, 'Visible', 'off', 'Position', [0 0 0.01 0.01]); hold(axL, 'on');
h = gobjects(3, 1);
for q = 1:3
    h(q) = plot(axL, NaN, NaN, st{q}, 'Color', cc(q,:), 'LineWidth', lw(q), 'MarkerSize', 3.6, 'MarkerFaceColor', cc(q,:));
end
lg = legend(axL, h, {'ARR, revision prompt $\rho$', 'ARR, paraphrase 1', 'ARR, paraphrase 2'}, 'Orientation', 'horizontal', ...
    'NumColumns', 3, 'Box', 'off', 'Interpreter', 'latex', 'FontSize', S.fs - 1);
lg.ItemTokenSize = [10 6];
stackLegends(t, lg);
saveFig(f, 'fig_abl_prompts');
end
""" + STYLE)
files["fig_abl_prompts.m"] = "\n".join(lines)

# (4) width N: every benchmark and budget where N = 2, 4, 8 are complete for ARR, ARR_Depth, MV (same comparison)
NM = ["arr", "arr_no_replace", "maj"]
WROWS = [(d, b) for d in ("math500", "olympiadbench", "amc23", "aime24", "aime25", "gsm8k")
         for b in ([50, 100, 200] if d == "gsm8k" else BUD3)
         if all(vacc(v, d, b, m) == vacc(v, d, b, m) for v in ("n2", "main", "n8") for m in NM)]
nAccM = {m: [[vacc(v, d, b, m) for v in ("n2", "main", "n8")] for d, b in WROWS] for m in NM}
# N = 8 with all eight methods (AMC23, AIME24): accuracy minus ARR at the same benchmark and budget
N8ROWS = [(d, b) for d in ("amc23", "aime24") for b in BUD3
          if all(vacc("n8", d, b, m) == vacc("n8", d, b, m) for m in METHODS)]
n8D = [[vacc("n8", d, b, m) - vacc("n8", d, b, "arr") for d, b in N8ROWS] for m in METHODS]
nD = {q: [[vacc(v, d, b, m) - vacc("main", d, b, m) for m in NM] for d, b in WROWS] for q, v in (("2", "n2"), ("8", "n8"))}
lines = [header("fig_abl_width", "Width N: accuracy change against the default N = 4, and N = 8 with all methods (Qwen2.5-Math-7B)."),
         f"cells = {cellstr([f'{DLBL[d]}, {b}' for d, b in WROWS])};   % benchmark, budget B (TFLOPs)",
         f"d2 = {mat(nD['2'])};   % (a) N = 2 minus N = 4 (points); columns: ARR, ARR_Depth, MV",
         f"d8 = {mat(nD['8'])};   % (b) N = 8 minus N = 4 (points)",
         f"n8D = {mat(n8D)};   % (c) N = 8: accuracy minus ARR (points); rows = methods (paper order), columns = cells",
         r"""
f = newFig(S, 7.6);
t = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
D = {d2, d8}; ttl = {'(a) $N{=}2$', '(b) $N{=}8$'};
n = numel(cells); y = n:-1:1; km = [1 4 8]; dy = [0.24 0 -0.24];
lim = max(abs([d2(:); d8(:)])) + 1;
for p = 1:2
    ax = nexttile(t); styleAxes(ax, S, ttl{p}); if p == 1, ax1 = ax; end
    set(ax, 'YGrid', 'off', 'XGrid', 'on');
    plot(ax, [0 0], [0.4 n + 0.6], '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.6);
    for k = 1:3
        c = S.col(km(k),:);
        for i = 1:n
            plot(ax, [0 D{p}(i,k)], y(i) + dy(k) * [1 1], '-', 'Color', c, 'LineWidth', 1.2);
        end
        plot(ax, D{p}(:,k), y + dy(k), S.mk{km(k)}, 'MarkerSize', S.ms(km(k)), 'Color', c, 'MarkerFaceColor', c, 'LineStyle', 'none');
    end
    set(ax, 'XLim', [-lim lim], 'YLim', [0.4 n + 0.6], 'YTick', fliplr(y), 'YTickLabel', fliplr(cells));
    if p > 1, set(ax, 'YTickLabel', {}); end
    xlabel(ax, '$\Delta$ to $N{=}4$ (points)');
end
ax = nexttile(t); styleAxes(ax, S, '(c) $N{=}8$, all methods');
set(ax, 'YGrid', 'off', 'XGrid', 'on');
nm = size(n8D, 1); ym = nm:-1:1;
plot(ax, [0 0], [0.4 nm + 0.6], '-', 'Color', S.col(1,:), 'LineWidth', 0.8);
for k = 1:nm
    c = S.col(k,:);
    plot(ax, n8D(k,:), ym(k) + 0.14 * linspace(-1, 1, size(n8D, 2)), '.', 'Color', c, 'MarkerSize', 6);
    plot(ax, mean(n8D(k,:)), ym(k), S.mk{k}, 'MarkerSize', S.ms(k) + 1, 'Color', c, 'MarkerFaceColor', c);
end
set(ax, 'YLim', [0.4 nm + 0.6], 'YTick', fliplr(ym), 'YTickLabel', fliplr(S.lbl(1:nm)), 'YAxisLocation', 'right');
xlabel(ax, '$\Delta$ to ARR (points)');
topLegend(t, ax1, S, km);
saveFig(f, 'fig_abl_width');
end
""" + STYLE]
files["fig_abl_width.m"] = "\n".join(lines)

# (5) latency-matched budgets: the budget is a number of serial passes per question
PB = [2000, 4000, 8000]
PBENCH = [d for d in ("math500", "amc23", "aime24", "aime25") if any(vacc("passes", d, b, "arr") == vacc("passes", d, b, "arr") for b in PB)]
pAcc = [[[vacc("passes", d, b, m) for b in PB] for m in METHODS] for d in PBENCH]
lines = [header("fig_abl_passes", "Latency-matched budgets: accuracy against a budget of serial passes per question."),
         f"pb = {mstr(PB)};",
         f"pName = {cellstr([DLBL[d] for d in PBENCH])};",
         "% pAcc{d}: rows = methods in paper order (NaN: not run), columns = pass budgets",
         "pAcc = cell(numel(pName), 1);"]
for i in range(len(PBENCH)):
    lines.append(f"pAcc{{{i + 1}}} = {mat(pAcc[i])};")
lines.append(r"""
show = [1 2 3 4 5 6 8];   % Sequential Revision (7) loses up to 30 points; omitted for scale as in fig_latency_accuracy
f = newFig(S, 5.6);
t = tiledlayout(f, 1, numel(pName), 'TileSpacing', 'compact', 'Padding', 'compact');
for d = 1:numel(pName)
    ax = nexttile(t); styleAxes(ax, S, sprintf('(%s) %s', char('a' + d - 1), pName{d})); if d == 1, ax1 = ax; end
    for k = show(end:-1:1)
        if all(isnan(pAcc{d}(k,:))), continue; end
        drawMethod(ax, S, k, pb, pAcc{d}(k,:));
    end
    set(ax, 'XScale', 'log', 'XTick', pb, 'XTickLabel', {'2k', '4k', '8k'}, 'XLim', [1700 9400], 'XMinorTick', 'off');
end
xlabel(t, 'Budget of serial passes per question (log scale)', 'Interpreter', 'latex', 'FontSize', S.fs);
ylabel(t, 'Accuracy (\%)', 'Interpreter', 'latex', 'FontSize', S.fs);
topLegend(t, ax1, S, show);
saveFig(f, 'fig_abl_passes');
end
""" + STYLE)
files["fig_abl_passes.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_models (appendix): every completed setting of the other models
HMOD = [m for m in MODELS if m != P and m != "Qwen2.5-7B-Instruct"]   # Qwen2.5-7B has only LiveBench: no full panel
HDS = MATH + ["livebench_reasoning"]
def _hcell(m, d, me):
    if not full_setting(m, d): return float("nan")
    return C[(m, d, budgets(m, d)[-1], me)]["acc"]
hAcc = [[[_hcell(m, d, me) for me in METHODS] for d in HDS] for m in HMOD]
hB = [max(budgets(m, d)[-1] for d in HDS if full_setting(m, d)) for m in HMOD]
MDATA = [f"mdl = {cellstr([MLBL[m] for m in HMOD])};",
         f"bud = {mstr([float(x) for x in hB])};   % largest budget of each model (TFLOPs)",
         f"bench = {cellstr(['MATH500', 'OlympiadBench', 'AMC23', 'AIME24', 'AIME25', 'LiveBench'])};",
         "acc = cat(3, " + ", ".join(mat(x) for x in hAcc) + ");   % benchmarks x methods x models (accuracy, %); NaN: not complete",
         ]
lines = [header("fig_models", "Accuracy of every method for the other models at the largest budget .")] + MDATA + [r"""
nb = numel(bench); m = size(acc, 2); nm = numel(mdl);
% fixed geometry (cm) so that every cell has the same size; 3 panels on top, 2 centered below
cw = 0.62; ch = 0.34; lab = 2.25; gap = 0.55; top = 1.45; ttl = 0.55; vgap = 0.25; key = 1.35;
pw = nb * cw; ph = m * ch;
rowsOf = {1:3, 4:nm};
H = 2 * (top + ph + ttl) + vgap + key;
f = newFig(S, H);
lim = 10; cm = divergingMap(129);
x0top = (S.width - (lab + 3 * pw + 2 * gap)) / 2 + lab;            % left edge of panel (a)
xc = x0top + (3 * pw + 2 * gap) / 2;                               % center of the panel block
for r = 1:2
    ks = rowsOf{r}; np = numel(ks);
    x0 = xc - (np * pw + (np - 1) * gap) / 2;
    y0 = H - r * (top + ph) - (r - 1) * (ttl + vgap);
    for q = 1:np
        k = ks(q);
        A = acc(:,:,k); keep = find(~isnan(A(:,1)))'; A = A(keep,:)'; D = A - A(1,:);   % methods x completed benchmarks
        nc = numel(keep);
        ax = axes(f, 'Units', 'centimeters', 'Position', [x0 + (q - 1) * (pw + gap) + (nb - nc) * cw / 2, y0, nc * cw, ph]);
        hold(ax, 'on');
        for j = 1:nc
            best = max(A(:,j));
            for i = 1:m
                c = cm(round((max(-lim, min(lim, D(i,j))) + lim) / (2 * lim) * 128) + 1, :);
                rectangle(ax, 'Position', [j - 0.5, i - 0.5, 1, 1], 'FaceColor', c, 'EdgeColor', 'w', 'LineWidth', 1.0);
                s = sprintf('%.1f', A(i,j)); if A(i,j) == best, s = ['\textbf{' s '}']; end
                tc = [0.12 0.12 0.12]; if abs(D(i,j)) > 0.75 * lim, tc = [1 1 1]; end
                text(ax, j, i, s, 'HorizontalAlignment', 'center', 'FontSize', S.fs - 2, 'Color', tc, 'Interpreter', 'latex');
            end
        end
        cl = bench(keep);
        set(ax, 'YDir', 'reverse', 'XLim', [0.5 nc + 0.5], 'YLim', [0.5 m + 0.5], 'TickLength', [0 0], 'Box', 'off', ...
            'XColor', [0.1 0.1 0.1], 'YColor', [0.1 0.1 0.1], 'XAxisLocation', 'top', 'XTick', 1:nc, 'XTickLabel', cl, ...
            'YTick', 1:m, 'YTickLabel', {}, 'FontSize', S.fs - 1.5, 'TickLabelInterpreter', 'latex');
        xtickangle(ax, 40);
        if q == 1, ax.YTickLabel = S.lbl(1:m); end   % method names once per row of panels
        drawnow; ax.XRuler.Axle.Visible = 'off'; ax.YRuler.Axle.Visible = 'off';
        text(ax, (nc + 1) / 2, m + 0.6, sprintf('(%c) %s, $B{=}%d$', 'a' + k - 1, mdl{k}, bud(k)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', S.fs - 0.5, 'Interpreter', 'latex');
    end
end
% one color key, centered below the panels
axK = axes(f, 'Visible', 'off', 'Units', 'centimeters', 'Position', [0 0 0.1 0.1]); colormap(axK, cm); caxis(axK, [-lim lim]);
cb = colorbar(axK, 'Location', 'southoutside', 'Units', 'centimeters', 'Position', [xc - 2.6, 0.98, 5.2, 0.2], 'AxisLocation', 'in', ...
    'Ticks', -lim:5:lim, 'TickLabels', {'$-10$', '$-5$', '$0$', '$+5$', '$+10$'}, 'TickLabelInterpreter', 'latex', 'FontSize', S.fs - 2, 'TickDirection', 'out', 'Box', 'off');
annotation(f, 'textbox', 'Units', 'centimeters', 'Position', [0 0.2 S.width 0.4], 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Interpreter', 'latex', 'FontSize', S.fs - 1.5, ...
    'String', 'Accuracy $-$ ARR (points); \textbf{bold}: best method');
saveFig(f, 'fig_models');
end

function cm = divergingMap(n)
% Muted red (below ARR) - white - blue (above ARR).
lo = [0.79 0.33 0.30]; hi = [0.27 0.50 0.74]; w = [1 1 1];
h = (n - 1) / 2; a = linspace(0, 1, h + 1)';
cm = [lo + (w - lo) .* a(1:end-1); w; w + (hi - w) .* a(2:end)];
end
""" + STYLE]
files["fig_models.m"] = "\n".join(lines)

# ------------------------------------------------------------------ fig_models_lines (appendix variant): one panel per benchmark
LORD = ["Ministral-8B-Instruct-2410", "Qwen3.5-2B", "Qwen3-8B", "gemma-4-E4B-it", "Qwen3.5-9B"]   # x-axis order of the line variant
lines = [header("fig_models_lines", "Accuracy of every method across the other models, one panel per benchmark (largest budget).")] + MDATA + [
         f"ord = {mstr([float(HMOD.index(m) + 1) for m in LORD])};   % x-axis order of the models",
         r"""
acc = acc(:,:,ord); mdl = mdl(ord); bud = bud(ord);
nb = numel(bench); m = size(acc, 2); nm = numel(mdl);
f = newFig(S, 5.0);
t = tiledlayout(f, 1, nb, 'TileSpacing', 'compact', 'Padding', 'compact');
for b = 1:nb
    ax = nexttile(t, b); styleAxes(ax, S, '\phantom{(b)}');   % empty title: the layout keeps room for it
    text(ax, 0.5, 1.03, sprintf('(%c) %s', 'a' + b - 1, bench{b}), 'Units', 'normalized', ...   % same height in every panel
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', S.fs, 'Interpreter', 'latex');
    Y = squeeze(acc(b,:,:));                      % methods x models
    for k = [5 7 4 6 2 3 8 1]                     % ARR and Majority Voting drawn last (on top)
        ok = ~isnan(Y(k,:));                      % only completed settings
        drawMethod(ax, S, k, find(ok), Y(k,ok));
    end
    set(ax, 'XLim', [0.6 nm + 0.4], 'XTick', 1:nm, 'XTickLabel', mdl, 'FontSize', S.fs - 1.5);
    xtickangle(ax, 45);
    if b == 1, ylabel(ax, 'Accuracy (\%)'); end
end
axL = axes(f, 'Visible', 'off', 'Position', [0 0 0.01 0.01]); hold(axL, 'on');   % legend on figure-level axes
h = gobjects(m, 1);
for k = 1:m, h(k) = drawMethod(axL, S, k, NaN, NaN); end
lg = legend(axL, h, S.lbl(1:m), 'Orientation', 'horizontal', 'NumColumns', m, 'Box', 'off', ...
    'Interpreter', 'latex', 'FontSize', S.fs - 1.5);
lg.ItemTokenSize = [9 6];
stackLegends(t, lg);
saveFig(f, 'fig_models_lines');
end
""" + STYLE]
files["fig_models_lines.m"] = "\n".join(lines)


os.makedirs(OUTD, exist_ok=True)
for old in glob.glob(os.path.join(OUTD, "fig*.*")) + glob.glob(os.path.join(OUTD, "plot_arr_main_results.m")):
    if os.path.splitext(os.path.basename(old))[0] not in order:   # remove only figures that are no longer made
        os.remove(old)
for n, txt in files.items():
    open(os.path.join(OUTD, n), "w").write(txt)
open(os.path.join(OUTD, "tab_models.tex"), "w").write(tab)
open(os.path.join(OUTD, "tab_ablations.tex"), "w").write("\n".join(abl_tex))
print(f"wrote {len(files)} .m files and tab_models.tex ({len(trows)} rows) to {OUTD}")
print(f"compute-use cells: primary {len(l1)}, other {len(l2)}; gain models: {gl}")
