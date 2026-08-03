"""The one figure that explains the whole thing, plus the statistics panel.

Both render on a white background only -- these are made to be dropped into
Discord and LinkedIn, where a theme-aware chart is meaningless and a dark one
looks broken on half the clients.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .stats import compare, summarise

WHITE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SYSTEM = "#2a78d6"  # the current system prompt won
STALE = "#e34948"  # the stale history won
NEUTRAL = "#f0efec"
FONT = ["DejaVu Sans", "system-ui", "sans-serif"]


def _card(fig, x, y, w, h, face=WHITE, edge=GRID, lw=1.2, r=0.012, z=1):
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (x, y), w, h, transform=fig.transFigure,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=r),
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
        )
    )


def _wrap(t, n):
    return "\n".join(textwrap.wrap(t, n)) if t else ""


def _wrap_lines(t, n):
    """Wrap without destroying the author's line breaks.

    Matters here: the whole point of the options example is that each numbered
    item sits on its own line. textwrap.wrap() reflows that into a paragraph
    and the reader loses the thing being demonstrated.
    """
    if not t:
        return ""
    return "\n".join(
        "\n".join(textwrap.wrap(line, n)) if line.strip() else ""
        for line in t.splitlines()
    )


def hero_sbs(example: dict, stats: dict, path: Path, model_label: str):
    """Side-by-side hero: the same question, answered under both regimes.

    Optimised for a two-second read on a Discord/LinkedIn timeline: the two
    answers sit at the same height so the eye diffs them directly, and the two
    percentages are the largest ink on the page.
    """
    plt.rcParams["font.sans-serif"] = FONT
    fig = plt.figure(figsize=(13.2, 7.8), dpi=160)
    fig.patch.set_facecolor(WHITE)

    fig.text(0.042, 0.955, "Your app updated. The chat history didn't.",
             fontsize=27, fontweight="700", color=INK, va="top")
    fig.text(0.042, 0.898,
             "An instruction the user gave "
             "before the update is still in the transcript — and it beats the new system prompt every single time.",
             fontsize=12.5, color=INK2, va="top")

    # ---- the setup, one band, three beats ----
    _card(fig, 0.042, 0.700, 0.916, 0.150, face="#fbfbfa")
    cols = [
        (0.062, "SYSTEM PROMPT · AFTER UPDATE", SYSTEM, "#ffffff", example["system"]),
        (0.372, "USER MESSAGE · BEFORE UPDATE", STALE, "#ffffff", example["stale"]),
        (0.682, "USER ASKS NOW", NEUTRAL, INK2, example["query"]),
    ]
    for x, label, face, fg, body in cols:
        fig.patches.append(mpatches.FancyBboxPatch(
            (x, 0.795), 0.006 + 0.0052 * len(label), 0.030,
            transform=fig.transFigure,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.014),
            facecolor=face, edgecolor="none", zorder=3))
        fig.text(x + 0.006, 0.810, label, fontsize=8.8, fontweight="700",
                 color=fg, va="center", zorder=4)
        fig.text(x, 0.772, _wrap(body, 42), fontsize=11.5, color=INK,
                 va="top", linespacing=1.4)

    for x in (0.352, 0.662):
        fig.text(x, 0.775, "vs" if x < 0.4 else "→", fontsize=13,
                 color=MUTED, ha="center", va="center", fontweight="700")

    # ---- the two answers, side by side ----
    panels = [
        (0.042, "WITHOUT THE FIX", STALE, example["before"], stats["rate_a"],
         "of answers followed\nthe current system prompt"),
        (0.512, "WITH V-STEER", SYSTEM, example["after"], stats["rate_b"],
         "of answers followed\nthe current system prompt"),
    ]
    for x, title, color, answer, rate, caption in panels:
        _card(fig, x, 0.215, 0.446, 0.455)
        fig.patches.append(mpatches.FancyBboxPatch(
            (x, 0.640), 0.446, 0.030, transform=fig.transFigure,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.012),
            facecolor=color, edgecolor="none", zorder=2))
        fig.text(x + 0.018, 0.655, title, fontsize=11.5, fontweight="700",
                 color="#ffffff", va="center", zorder=3)

        fig.text(x + 0.018, 0.605, _wrap(answer, 44), fontsize=13.5, color=INK,
                 va="top", linespacing=1.5, family="monospace")

        fig.patches.append(mpatches.Rectangle(
            (x + 0.018, 0.385), 0.410, 0.0012, transform=fig.transFigure,
            facecolor=GRID, edgecolor="none"))
        fig.text(x + 0.018, 0.310, f"{rate*100:.0f}%", fontsize=62,
                 fontweight="700", color=color, va="center")
        fig.text(x + 0.185, 0.312, caption, fontsize=10.5, color=INK2,
                 va="center", linespacing=1.45)

    # ---- footer ----
    _card(fig, 0.042, 0.055, 0.916, 0.128, face="#fbfbfa")
    notes = [
        ("FIXED", f"{stats.get('caused_recall', stats['recall'])*100:.0f}%",
         "of the failures the stale\nhistory caused"),
        ("BROKE", f"{stats['fp']}", "answers that were\nalready correct"),
        ("CEILING", f"{stats.get('ceiling', 0)*100:.0f}%",
         "same model, same task,\nno history at all"),
        ("SIGNIFICANCE", f"p={stats['p_value']:.0e}",
         f"McNemar exact\nn = {stats['n_paired']} paired cases"),
    ]
    for i, (k, big, sub) in enumerate(notes):
        x = 0.068 + i * 0.228
        fig.text(x, 0.155, k, fontsize=8.8, color=MUTED, fontweight="700",
                 va="center")
        fig.text(x, 0.116, big, fontsize=22, color=INK, fontweight="700",
                 va="center")
        fig.text(x, 0.074, sub, fontsize=9.2, color=INK2, va="center",
                 linespacing=1.4)

    fig.text(0.958, 0.022, f"{model_label} · StaleSet · greedy decoding · "
             "V-Steer (arXiv:2607.26228)", fontsize=8.8, color=MUTED, ha="right")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def hero(example: dict, stats: dict, path: Path, model_label: str):
    """example: system/stale/query/before/after strings. stats: paired compare()."""
    plt.rcParams["font.sans-serif"] = FONT
    fig = plt.figure(figsize=(13, 7.4), dpi=160)
    fig.patch.set_facecolor(WHITE)

    fig.text(0.045, 0.945, "Your app updated. The chat history didn't.",
             fontsize=25, fontweight="700", color=INK, va="top")
    fig.text(0.045, 0.885,
             "Instructions the user gave before the update kept overruling the new system prompt. "
             "Editing the\ncached value vectors of the guilty attention heads — at inference time, no retraining — puts the order back.",
             fontsize=12.5, color=INK2, va="top", linespacing=1.5)

    # ---------------- left: the transcript ----------------
    LX, LW = 0.045, 0.40
    _card(fig, LX, 0.175, LW, 0.60)
    fig.text(LX + 0.018, 0.735, "THE CONVERSATION", fontsize=10.5,
             fontweight="700", color=MUTED, va="top")

    def chip(x, y, label, face, fg):
        fig.patches.append(
            mpatches.FancyBboxPatch(
                (x, y), 0.088, 0.030, transform=fig.transFigure,
                boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.014),
                facecolor=face, edgecolor="none", zorder=3))
        fig.text(x + 0.044, y + 0.0145, label, fontsize=8.6, fontweight="700",
                 color=fg, ha="center", va="center", zorder=4)

    y = 0.688
    chip(LX + 0.018, y, "SYSTEM · NOW", SYSTEM, "#ffffff")
    fig.text(LX + 0.018, y - 0.018, _wrap(example["system"], 52), fontsize=11,
             color=INK, va="top", linespacing=1.45)

    y = 0.565
    fig.plot = None
    fig.text(LX + 0.018, y + 0.012, "─── app updated here " + "─" * 22,
             fontsize=9, color=MUTED, va="center", family="monospace")

    y = 0.505
    chip(LX + 0.018, y, "PRE-UPDATE", STALE, "#ffffff")
    fig.text(LX + 0.018, y - 0.018, _wrap(example["stale"], 52), fontsize=11,
             color=INK2, va="top", linespacing=1.45)

    y = 0.345
    chip(LX + 0.018, y, "USER · NOW", NEUTRAL, INK2)
    fig.text(LX + 0.018, y - 0.018, _wrap(example["query"], 52), fontsize=11,
             color=INK, va="top", linespacing=1.45)

    fig.text(LX + 0.018, 0.215,
             "…plus 4 more turns of ordinary pre-update chat.",
             fontsize=9.5, color=MUTED, va="top", style="italic")

    # ---------------- right: the two answers ----------------
    RX, RW = 0.475, 0.48
    for i, (key, title, color, verdict) in enumerate([
        ("before", "WITHOUT THE FIX", STALE, "obeys the old instruction"),
        ("after", "WITH V-STEER", SYSTEM, "obeys the current system prompt"),
    ]):
        top = 0.755 - i * 0.30
        _card(fig, RX, top - 0.245, RW, 0.245)
        fig.patches.append(
            mpatches.FancyBboxPatch(
                (RX, top - 0.245), 0.007, 0.245, transform=fig.transFigure,
                boxstyle=mpatches.BoxStyle("Square", pad=0),
                facecolor=color, edgecolor="none", zorder=2))
        fig.text(RX + 0.022, top - 0.030, title, fontsize=10.5,
                 fontweight="700", color=color, va="center")
        fig.text(RX + 0.022, top - 0.068, _wrap(example[key], 58), fontsize=12.5,
                 color=INK, va="top", linespacing=1.5,
                 family="monospace" if i else None)
        fig.text(RX + 0.022, top - 0.225, verdict, fontsize=10, color=MUTED,
                 va="center", style="italic")

    # ---------------- bottom: the numbers ----------------
    BY = 0.048
    _card(fig, LX, BY, 0.91, 0.108, face="#fbfbfa")
    a, b = stats["rate_a"], stats["rate_b"]
    ceiling = stats.get("ceiling")

    fig.text(LX + 0.022, BY + 0.082, "Answers following the current system prompt",
             fontsize=10.5, color=INK2, va="center")

    bx, bw = LX + 0.022, 0.30
    for val, color, lab, dy in [(a, STALE, "without", 0.048),
                                (b, SYSTEM, "with V-Steer", 0.020)]:
        fig.patches.append(mpatches.FancyBboxPatch(
            (bx, BY + dy - 0.008), bw, 0.016, transform=fig.transFigure,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.008),
            facecolor=NEUTRAL, edgecolor="none", zorder=2))
        if val > 0:
            fig.patches.append(mpatches.FancyBboxPatch(
                (bx, BY + dy - 0.008), max(bw * val, 0.014), 0.016,
                transform=fig.transFigure,
                boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.008),
                facecolor=color, edgecolor="none", zorder=3))
        fig.text(bx + bw + 0.010, BY + dy, f"{val*100:.0f}%  {lab}", fontsize=10.5,
                 fontweight="700", color=color, va="center")

    # the honest ceiling: what this model manages with no history at all
    if ceiling:
        cx = bx + bw * ceiling
        fig.patches.append(mpatches.Rectangle(
            (cx, BY + 0.012), 0.0012, 0.052, transform=fig.transFigure,
            facecolor=MUTED, edgecolor="none", zorder=4))
        fig.text(cx, BY + 0.004,
                 f"{ceiling*100:.0f}% — this model's ceiling with no history at all",
                 fontsize=8.2, color=MUTED, va="center", ha="center")

    notes = [
        ("FIXED", f"{stats.get('caused_recall', stats['recall'])*100:.0f}%",
         "of the failures the\nstale history caused"),
        ("BROKE", f"{stats['fp']}", "answers that were\nalready correct"),
        ("SIGNIFICANCE", f"p={stats['p_value']:.0e}",
         f"McNemar exact\nn = {stats['n_paired']} paired"),
    ]
    for i, (k, big, sub) in enumerate(notes):
        x = LX + 0.505 + i * 0.138
        fig.text(x, BY + 0.086, k, fontsize=8.4, color=MUTED,
                 fontweight="700", va="center")
        fig.text(x, BY + 0.055, big, fontsize=17, color=INK,
                 fontweight="700", va="center")
        fig.text(x, BY + 0.022, sub, fontsize=8.8, color=INK2, va="center",
                 linespacing=1.35)

    fig.text(0.955, 0.018,
             f"{model_label} · StaleSet · greedy decoding · arXiv:2607.26228 (V-Steer)",
             fontsize=8.8, color=MUTED, ha="right")

    fig.savefig(path, facecolor=WHITE, bbox_inches=None)
    plt.close(fig)


def confusion_panel(stats: dict, path: Path, model_label: str):
    """The 2x2 repair matrix plus the three rates, white background."""
    plt.rcParams["font.sans-serif"] = FONT
    fig = plt.figure(figsize=(11, 5.6), dpi=160)
    fig.patch.set_facecolor(WHITE)

    fig.text(0.055, 0.93, "Does it fix the right things, and break nothing?",
             fontsize=19, fontweight="700", color=INK, va="top")
    fig.text(0.055, 0.865,
             "Every case scored twice — same prompt, steering off then on. "
             "Paired, so each cell is one case changing its mind.",
             fontsize=11.5, color=INK2, va="top")

    ax = fig.add_axes([0.175, 0.16, 0.33, 0.55])
    cells = np.array([[stats["tn"], stats["fp"]], [stats["tp"], stats["fn"]]])
    labels = np.array([["preserved", "BROKEN"], ["FIXED", "still broken"]])
    colors = np.array([[NEUTRAL, STALE], [SYSTEM, NEUTRAL]])

    for i in range(2):
        for j in range(2):
            ax.add_patch(mpatches.FancyBboxPatch(
                (j + 0.03, 1 - i + 0.03), 0.94, 0.94,
                boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.04),
                facecolor=colors[i, j], edgecolor="none"))
            strong = colors[i, j] != NEUTRAL
            ax.text(j + 0.5, 1 - i + 0.62, str(cells[i, j]), fontsize=30,
                    fontweight="700", ha="center", va="center",
                    color="#ffffff" if strong else INK)
            ax.text(j + 0.5, 1 - i + 0.26, labels[i, j], fontsize=11,
                    ha="center", va="center",
                    color="#ffffff" if strong else INK2,
                    fontweight="700" if strong else "normal")

    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis("off")
    ax.text(-0.08, 1.5, "after steering:\nright", fontsize=10, color=MUTED,
            ha="right", va="center", linespacing=1.4)
    ax.text(-0.08, 0.5, "after steering:\nwrong", fontsize=10, color=MUTED,
            ha="right", va="center", linespacing=1.4)
    ax.text(0.5, -0.14, "was right before", fontsize=10, color=MUTED, ha="center")
    ax.text(1.5, -0.14, "was wrong before", fontsize=10, color=MUTED, ha="center")

    rows = [
        ("recall", stats["recall"], stats.get("recall_ci"),
         "of the answers that were wrong,\nthis share got fixed"),
        ("precision", stats["precision"], stats.get("precision_ci"),
         "of the answers it changed,\nthis share improved"),
        ("specificity", stats["specificity"], stats.get("specificity_ci"),
         "of the answers already right,\nthis share stayed right"),
    ]
    for i, (name, val, ci, desc) in enumerate(rows):
        y = 0.63 - i * 0.20
        fig.text(0.57, y, name, fontsize=12.5, fontweight="700", color=INK,
                 va="center")
        fig.text(0.945, y + 0.005, "—" if val is None else f"{val*100:.0f}%",
                 fontsize=23, fontweight="700", color=SYSTEM, ha="right",
                 va="center")
        if ci:
            fig.text(0.945, y - 0.043,
                     f"95% CI {ci[0]*100:.0f}–{ci[1]*100:.0f}",
                     fontsize=8.6, color=MUTED, ha="right", va="center")
        fig.text(0.57, y - 0.048, desc, fontsize=9.6, color=INK2, va="center",
                 linespacing=1.4)
        fig.patches.append(mpatches.Rectangle(
            (0.57, y - 0.095), 0.375, 0.0012, transform=fig.transFigure,
            facecolor=GRID, edgecolor="none"))

    fig.text(0.945, 0.035,
             f"{model_label} · n = {stats['n_paired']} paired cases · "
             f"McNemar exact p = {stats['p_value']:.2g}",
             fontsize=9, color=MUTED, ha="right")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def hero_fork(example: dict, stats: dict, path: Path, model_label: str):
    """A timeline that forks: before the update, after it, then with/without.

    Reads left to right in one pass -- the old rule, the update that replaced
    it, and the two futures. The fork is the point: nothing about the model or
    the question changes between the branches, only whether the cached values
    of the old messages were scaled.
    """
    plt.rcParams["font.sans-serif"] = FONT
    fig = plt.figure(figsize=(14.2, 7.6), dpi=160)
    fig.patch.set_facecolor(WHITE)

    fig.text(0.032, 0.960, "Your app updated. The chat history didn't.",
             fontsize=27, fontweight="700", color=INK, va="top")
    fig.text(0.032, 0.902,
             f"The user asks: “{example['query']}”   —   same model, same question, "
             "the only difference is what the old messages are still allowed to do.",
             fontsize=12, color=INK2, va="top")

    # ---------- stage 1: before the update ----------
    ax, aw = 0.032, 0.245
    _card(fig, ax, 0.300, aw, 0.480)
    fig.patches.append(mpatches.FancyBboxPatch(
        (ax, 0.735), aw, 0.045, transform=fig.transFigure,
        boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.010),
        facecolor="#c3c2b7", edgecolor="none", zorder=2))
    fig.text(ax + 0.014, 0.7575, "BEFORE THE UPDATE", fontsize=10,
             fontweight="700", color="#ffffff", va="center", zorder=3)
    fig.text(ax + 0.014, 0.700, "The rule you had:", fontsize=9.5,
             color=MUTED, fontweight="700", va="top")
    fig.text(ax + 0.014, 0.672, _wrap(example["stale"], 32), fontsize=10.5,
             color=INK, va="top", linespacing=1.45, style="italic")
    fig.text(ax + 0.014, 0.540, "and the assistant obeyed it,\nturn after turn.",
             fontsize=10, color=INK2, va="top", linespacing=1.5)
    fig.text(ax + 0.014, 0.430, "↳ still sitting in the\n   transcript today",
             fontsize=10, color=STALE, va="top", linespacing=1.5,
             fontweight="700")

    # ---------- stage 2: the update ----------
    bx, bw = 0.310, 0.215
    fig.text(0.291, 0.540, "→", fontsize=20, color=MUTED, ha="center",
             va="center", fontweight="700")
    _card(fig, bx, 0.300, bw, 0.480)
    fig.patches.append(mpatches.FancyBboxPatch(
        (bx, 0.735), bw, 0.045, transform=fig.transFigure,
        boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.010),
        facecolor=SYSTEM, edgecolor="none", zorder=2))
    fig.text(bx + 0.014, 0.7575, "YOU SHIP AN UPDATE", fontsize=10,
             fontweight="700", color="#ffffff", va="center", zorder=3)
    fig.text(bx + 0.014, 0.700, "The new system prompt:", fontsize=9.5,
             color=MUTED, fontweight="700", va="top")
    fig.text(bx + 0.014, 0.672, _wrap(example["system"], 29), fontsize=10.5,
             color=INK, va="top", linespacing=1.45, style="italic")
    fig.text(bx + 0.014, 0.500,
             "Nothing else changes.\nThe old messages stay\nin the context.",
             fontsize=10, color=INK2, va="top", linespacing=1.55)

    # ---------- the fork ----------
    fork_x = bx + bw + 0.020
    cx, cw = 0.605, 0.363
    tops = (0.545, 0.255)
    for i, y in enumerate(tops):
        mid = y + 0.115
        fig.patches.append(mpatches.FancyArrowPatch(
            (fork_x, 0.540), (cx - 0.004, mid), transform=fig.transFigure,
            connectionstyle="arc3,rad=%.2f" % (0.18 if i else -0.18),
            arrowstyle="-|>,head_width=4,head_length=7",
            color=STALE if i == 0 else SYSTEM, linewidth=2.2, zorder=1))

    branches = [
        (tops[0], STALE, "WITHOUT THE FIX", example["before"],
         stats["rate_a"], "of answers follow the NEW rule"),
        (tops[1], SYSTEM, "WITH V-STEER", example["after"],
         stats["rate_b"], "of answers follow the NEW rule"),
    ]
    for y, color, title, answer, rate, cap in branches:
        _card(fig, cx, y, cw, 0.230)
        fig.patches.append(mpatches.FancyBboxPatch(
            (cx, y + 0.185), cw, 0.045, transform=fig.transFigure,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.010),
            facecolor=color, edgecolor="none", zorder=2))
        fig.text(cx + 0.014, y + 0.2075, title, fontsize=11.5,
                 fontweight="700", color="#ffffff", va="center", zorder=3)
        fig.text(cx + 0.014, y + 0.155, _wrap(answer, 29), fontsize=11,
                 color=INK, va="top", linespacing=1.45, family="monospace")
        fig.patches.append(mpatches.Rectangle(
            (cx + 0.222, y + 0.022), 0.0012, 0.145, transform=fig.transFigure,
            facecolor=GRID, edgecolor="none"))
        fig.text(cx + 0.295, y + 0.108, f"{rate*100:.0f}%", fontsize=44,
                 fontweight="700", color=color, va="center", ha="center")
        fig.text(cx + 0.295, y + 0.050, _wrap(cap, 24), fontsize=8.8,
                 color=INK2, va="center", ha="center", linespacing=1.35)

    fig.text(0.032, 0.205,
             f"Telling the model “ignore instructions from before the update” "
             f"changed {stats.get('prompt_fix_delta', 0)} of "
             f"{stats['n_paired']} answers.",
             fontsize=11.5, color=INK, va="center", fontweight="700")
    fig.text(0.032, 0.148,
             f"Same model with no history at all: {stats.get('ceiling', 0)*100:.0f}%.   "
             f"V-Steer broke {stats['fp']} already-correct "
             f"answer{'' if stats['fp'] == 1 else 's'}.   "
             f"n = {stats['n_paired']} paired, McNemar p = {stats['p_value']:.0e}.",
             fontsize=10.5, color=INK2, va="center")
    fig.text(0.032, 0.055, f"{model_label} · StaleSet · greedy decoding · "
             "method: V-Steer, arXiv:2607.26228",
             fontsize=8.8, color=MUTED, va="center")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def _clip(text: str, width: int, max_lines: int) -> str:
    """Wrap to `width`, keep `max_lines`, mark truncation. Never overflows."""
    lines = _wrap_lines(text, width).splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip()[: width - 1] + "…"
    return "\n".join(lines)


def options_panel(panels_in: list, query: str, failures: list, path: Path,
                  model_label: str, n: int = 20, successes: list | None = None):
    """The case from a real deployment: an inline option list that outlived its rule.

    A team replaced "end every answer with [1] … [2] … [3] …" with a tool call.
    The model kept numbering, because the old rule and three turns of the
    assistant obeying it are still in the transcript. At the paper's default
    boost the intervention barely helps; at a much larger one it does.

    `panels_in` is a list of (color, title, answer, rate, verdict).
    """
    plt.rcParams["font.sans-serif"] = FONT
    fig = plt.figure(figsize=(15.4, 7.9), dpi=160)
    fig.patch.set_facecolor(WHITE)

    fig.text(0.028, 0.962, "The habit that outlived its rule", fontsize=27,
             fontweight="700", color=INK, va="top")
    fig.text(0.028, 0.905,
             "A real deployment: “end every answer with [1] … [2] … [3] …” was replaced by a tool call — and the model "
             "kept numbering.\nThe paper's default barely helps. Turning both knobs up gets most of the way — past a "
             "point the metric keeps rising while the model falls apart.",
             fontsize=12, color=INK2, va="top", linespacing=1.5)

    fig.text(0.028, 0.826, "THE USER ASKS:", fontsize=9.5, color=MUTED,
             fontweight="700", va="center")
    fig.text(0.126, 0.826, f"“{query}”", fontsize=12.5, color=INK,
             va="center", style="italic")

    span = 0.944 / len(panels_in)
    panels = [(0.028 + i * span,) + tuple(p) for i, p in enumerate(panels_in)]
    # without the extra strip the panels get the room back
    pan_y, pan_h = (0.470, 0.345) if successes else (0.400, 0.415)
    fail_y, fail_h = (0.088, 0.235) if successes else (0.105, 0.255)

    for x, color, title, answer, rate, verdict in panels:
        w = span - 0.018
        _card(fig, x, pan_y, w, pan_h)
        fig.patches.append(mpatches.FancyBboxPatch(
            (x, 0.770), w, 0.045, transform=fig.transFigure,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.010),
            facecolor=color, edgecolor="none", zorder=2))
        fig.text(x + 0.012, 0.7925, title, fontsize=10.5, fontweight="700",
                 color="#ffffff", va="center", zorder=3)
        fig.text(x + 0.012, 0.748, _clip(answer, 32, 7 if successes else 9),
                 fontsize=9.8, color=INK, va="top", linespacing=1.5,
                 family="monospace")
        fig.patches.append(mpatches.Rectangle(
            (x + 0.012, pan_y + 0.096), w - 0.024, 0.0012,
            transform=fig.transFigure, facecolor=GRID, edgecolor="none"))
        fig.text(x + 0.012, pan_y + 0.056, f"{rate*100:.0f}%", fontsize=33,
                 fontweight="700", color=color if color != "#c3c2b7" else SYSTEM,
                 va="center")
        fig.text(x + 0.012, pan_y + 0.022, "of answers obey the current rule",
                 fontsize=8.8, color=INK2, va="center")
        fig.text(x + 0.012, pan_y + 0.006, verdict, fontsize=9, color=MUTED,
                 va="center", style="italic")

    # ---- the same intervention on rules it handles well ----
    if successes:
        fig.text(0.028, 0.427,
                 "THE SAME INTERVENTION, SAME DEFAULTS, ON OTHER RULES",
                 fontsize=9.5, color=MUTED, fontweight="700", va="center")
        sw = 0.944 / len(successes)
        for i, (label, before, after) in enumerate(successes):
            sx = 0.028 + i * sw
            fig.text(sx, 0.386, label, fontsize=10.5, fontweight="700",
                     color=INK, va="center")
            fig.text(sx, 0.356, f"{before*100:.0f}%", fontsize=13,
                     fontweight="700", color=STALE, va="center")
            fig.text(sx + 0.030, 0.356, "→", fontsize=12, color=MUTED,
                     va="center")
            fig.text(sx + 0.048, 0.356, f"{after*100:.0f}%", fontsize=18,
                     fontweight="700", color=SYSTEM, va="center")

    # ---- what breaking actually looks like ----
    _card(fig, 0.028, fail_y, 0.944, fail_h, face="#fbfbfa")
    top = fail_y + fail_h
    fig.text(0.044, top - 0.023, "AND WHEN IT DOESN'T WORK, IT FAILS IN THREE WAYS",
             fontsize=9.5, color=MUTED, fontweight="700", va="center")
    fw = 0.944 / len(failures)
    for i, (label, setting, snippet) in enumerate(failures):
        fx = 0.044 + i * fw
        fig.patches.append(mpatches.Rectangle(
            (fx, fail_y + 0.024), 0.0035, fail_h - 0.080,
            transform=fig.transFigure, facecolor=STALE, edgecolor="none"))
        fig.text(fx + 0.014, top - 0.061, label, fontsize=10.5,
                 fontweight="700", color=STALE, va="center")
        fig.text(fx + 0.014, top - 0.083, setting, fontsize=8.8, color=MUTED,
                 va="center", style="italic")
        fig.text(fx + 0.014, top - 0.104, _clip(snippet, 40, 4), fontsize=9.2,
                 color=INK, va="top", linespacing=1.5, family="monospace")

    fig.text(0.028, 0.062,
             "The paper's defaults (γ+ 2.5 / γ− 0.75) barely move this rule. Raising both — γ+ 8, γ− 1.0 — takes it to "
             "75%, with an LLM judge still rating 93% of those answers sound.",
             fontsize=11, color=INK, va="center")
    fig.text(0.972, 0.030, f"{model_label} · n = {n} per condition · "
             "V-Steer, arXiv:2607.26228", fontsize=8.8, color=MUTED, ha="right")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def first_token_panel(rows: list, path: Path, model_label: str):
    """Gain vs whether the first predicted token carries the disputed property."""
    plt.rcParams["font.sans-serif"] = FONT
    rows = sorted(rows, key=lambda r: -r["gain"])
    fig = plt.figure(figsize=(12.2, 7.0), dpi=160)
    fig.patch.set_facecolor(WHITE)
    fig.text(0.045, 0.958, "How much it recovers depends on the constraint",
             fontsize=22, fontweight="700", color=INK, va="top")
    fig.text(0.045, 0.898,
             "The attribution reads only the first token the model is about to write. Recovery loosely tracks how much\n"
             "that token reveals about the disputed rule — loosely, because json breaks the ordering.",
             fontsize=11.5, color=INK2, va="top", linespacing=1.5)

    ax = fig.add_axes([0.335, 0.135, 0.45, 0.68])
    ax.set_facecolor(WHITE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)

    y = np.arange(len(rows))[::-1]
    for yy, r in zip(y, rows):
        color = SYSTEM if r["carries"] == "yes" else (
            "#eda100" if r["carries"] == "absence" else STALE)
        ax.barh(yy, r["gain"], height=0.6, color=color, edgecolor=WHITE,
                linewidth=1.5, zorder=3)
        ax.text(r["gain"] + 0.012, yy, f"+{r['gain']*100:.0f} pts", fontsize=10.5,
                va="center", color=INK, fontweight="700")
    ax.set_yticks(y)
    # family and note on one tick label: drawing the note separately on the
    # left collides with the tick text, which is what the first render did
    ax.set_yticklabels([f"{r['family']}   ·   {r['note']}" for r in rows],
                       fontsize=10.5, color=INK)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "+25", "+50", "+75", "+100 pts"])
    ax.set_xlabel("recovery in answers following the current system prompt",
                  color=INK2, fontsize=10.5, labelpad=9)

    key =[("the first token IS the evidence", SYSTEM),
           ("visible only as an absence", "#eda100"),
           ("first token says nothing about the rule", STALE)]
    for i, (label, color) in enumerate(key):
        x = 0.045 + i * 0.31
        fig.patches.append(mpatches.FancyBboxPatch(
            (x, 0.043), 0.016, 0.016, transform=fig.transFigure,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.004),
            facecolor=color, edgecolor="none"))
        fig.text(x + 0.024, 0.051, label, fontsize=10, color=INK2, va="center")

    fig.text(0.965, 0.958, model_label, fontsize=9, color=MUTED, ha="right")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def hero_rules(example: dict, stats: dict, path: Path, model_label: str):
    """Left: the rule you deleted. Right: the rule you shipped. Which one wins?

    Framing the two panels by *rule* rather than by *treatment* is what makes
    this readable without a caption -- the reader sees two rules and two
    answers and works out the punchline themselves.
    """
    plt.rcParams["font.sans-serif"] = FONT
    fig = plt.figure(figsize=(13.4, 7.9), dpi=160)
    fig.patch.set_facecolor(WHITE)

    fig.text(0.040, 0.960, "Your app updated. The chat history didn't.",
             fontsize=28, fontweight="700", color=INK, va="top")
    fig.text(0.040, 0.900,
             "Same model, same question. The rule that wins is the one still sitting in the transcript.",
             fontsize=13, color=INK2, va="top")

    fig.text(0.040, 0.848, "THE USER ASKS:", fontsize=9.5, color=MUTED,
             fontweight="700", va="center")
    fig.text(0.148, 0.848, f"“{example['query']}”", fontsize=13,
             color=INK, va="center", style="italic")

    panels = [
        (0.040, STALE, "THE OLD RULE",
         "deleted from the app — still in the chat history",
         example["stale"], example["before"], stats["rate_a_stale"],
         "of answers still follow it", "no fix applied"),
        (0.512, SYSTEM, "THE NEW RULE",
         "your current system prompt",
         example["system"], example["after"], stats["rate_b"],
         "of answers follow it", "with V-Steer"),
    ]
    for x, color, title, sub, rule, answer, rate, cap, foot in panels:
        _card(fig, x, 0.115, 0.448, 0.690)
        fig.patches.append(mpatches.FancyBboxPatch(
            (x, 0.735), 0.448, 0.070, transform=fig.transFigure,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.012),
            facecolor=color, edgecolor="none", zorder=2))
        fig.text(x + 0.020, 0.783, title, fontsize=15, fontweight="700",
                 color="#ffffff", va="center", zorder=3)
        fig.text(x + 0.020, 0.755, sub, fontsize=10, color="#ffffff",
                 va="center", zorder=3, alpha=0.92)

        fig.text(x + 0.020, 0.712, _wrap(rule, 52), fontsize=11.5, color=INK2,
                 va="top", linespacing=1.45, style="italic")

        fig.patches.append(mpatches.Rectangle(
            (x + 0.020, 0.575), 0.408, 0.0012, transform=fig.transFigure,
            facecolor=GRID, edgecolor="none"))
        fig.text(x + 0.020, 0.552, "THE MODEL ANSWERS", fontsize=9,
                 color=MUTED, fontweight="700", va="center")
        fig.text(x + 0.020, 0.525, _wrap(answer, 46), fontsize=12.5, color=INK,
                 va="top", linespacing=1.5, family="monospace")

        fig.patches.append(mpatches.Rectangle(
            (x + 0.020, 0.275), 0.408, 0.0012, transform=fig.transFigure,
            facecolor=GRID, edgecolor="none"))
        fig.text(x + 0.020, 0.200, f"{rate*100:.0f}%", fontsize=58,
                 fontweight="700", color=color, va="center")
        fig.text(x + 0.175, 0.213, cap, fontsize=11, color=INK2, va="center")
        fig.text(x + 0.175, 0.184, foot, fontsize=9.5, color=MUTED,
                 va="center", style="italic")

    fig.text(0.040, 0.062,
             f"Without any fix the new rule wins {stats['rate_a']*100:.0f}% of the time.  "
             f"The same model with no history at all: {stats.get('ceiling', 0)*100:.0f}%.  "
             f"V-Steer broke {stats['fp']} already-correct answers  "
             f"(n = {stats['n_paired']} paired, McNemar p = {stats['p_value']:.0e}).",
             fontsize=10.5, color=INK2, va="center")
    fig.text(0.040, 0.028, f"{model_label} · StaleSet · greedy decoding · "
             "V-Steer, arXiv:2607.26228", fontsize=8.8, color=MUTED, va="center")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def gamma_dial(summary: dict, path: Path, model_label: str):
    """One knob, three consequences — the effect-strength figure."""
    plt.rcParams["font.sans-serif"] = FONT
    pts = sorted((v["gamma_minus"], v) for v in summary.values()
                 if "gamma_minus" in v)
    if not pts:
        return
    xs = [p[0] for p in pts]
    series = [
        ("follows the NEW rule", [p[1]["system"] for p in pts], SYSTEM),
        ("follows the OLD rule", [p[1]["stale"] for p in pts], STALE),
    ]
    if all("fact_recall" in p[1] for p in pts):
        series.append(("recalls facts from the old messages",
                       [p[1]["fact_recall"] for p in pts], "#1baf7a"))

    fig = plt.figure(figsize=(11.4, 6.4), dpi=160)
    fig.patch.set_facecolor(WHITE)
    fig.text(0.055, 0.950, "One dial, not a rewrite", fontsize=23,
             fontweight="700", color=INK, va="top")
    fig.text(0.055, 0.888,
             "γ− scales down the cached value vectors of the pre-update messages. "
             "Nothing is retrained;\nnothing is deleted from the context.",
             fontsize=11.5, color=INK2, va="top", linespacing=1.5)

    ax = fig.add_axes([0.075, 0.145, 0.66, 0.62])
    ax.set_facecolor(WHITE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)

    for label, ys, color in series:
        ax.plot(xs, ys, color=color, linewidth=2.5, marker="o", markersize=8,
                markeredgecolor=WHITE, markeredgewidth=1.8, zorder=3)
        # every line is directly labelled: the palette's CVD margin for the
        # third hue is only legal with secondary encoding, and this is it
        ax.text(xs[-1] + 0.015, ys[-1], _wrap(label, 20), fontsize=10,
                fontweight="700", color=color, va="center", linespacing=1.3)

    ax.set_xlim(min(xs) - 0.02, max(xs) + 0.02)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xticks(xs)
    ax.set_xlabel("γ−   suppression strength on the pre-update messages",
                  color=INK2, fontsize=10.5, labelpad=9)
    fig.text(0.945, 0.035, f"{model_label} · n = {pts[0][1]['n']} per point",
             fontsize=9, color=MUTED, ha="right")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def method_diagram(path: Path):
    """What the method actually does, in one pass left to right."""
    plt.rcParams["font.sans-serif"] = FONT
    fig = plt.figure(figsize=(13.4, 6.0), dpi=160)
    fig.patch.set_facecolor(WHITE)

    fig.text(0.038, 0.955, "How it works", fontsize=25, fontweight="700",
             color=INK, va="top")
    fig.text(0.038, 0.895,
             "No retraining, no edits to the prompt, no change to the attention kernel. "
             "One extra pass over the prompt,\nthen decoding runs at its normal speed.",
             fontsize=12, color=INK2, va="top", linespacing=1.5)

    steps = [
        ("1", "Tag the transcript",
         "Mark which tokens came from the current system prompt and which "
         "from messages written before the update."),
        ("2", "Prefill once",
         "Run the prompt through the model, keeping the KV cache and the "
         "attention weights of the last position."),
        ("3", "Ask each head who it listened to",
         "Direct logit attribution splits the first predicted token, per head, "
         "into φ(system) and φ(old messages)."),
        ("4", "Scale the cached values",
         "For heads where φ(old) beats φ(system): multiply V by 3.5 on system "
         "tokens, by 0.25 on the stale ones."),
        ("5", "Decode as usual",
         "The edit lives in the cache, so every later token is free, and fused "
         "attention kernels stay untouched."),
    ]
    w, gap = 0.166, 0.020
    for i, (n, title, body) in enumerate(steps):
        x = 0.038 + i * (w + gap)
        accent = SYSTEM if i in (0, 3) else "#c3c2b7"
        _card(fig, x, 0.235, w, 0.545)
        fig.patches.append(mpatches.FancyBboxPatch(
            (x, 0.735), w, 0.045, transform=fig.transFigure,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.010),
            facecolor=accent, edgecolor="none", zorder=2))
        fig.text(x + 0.012, 0.7575, f"STEP {n}", fontsize=9.5, fontweight="700",
                 color="#ffffff", va="center", zorder=3)
        fig.text(x + 0.012, 0.700, _wrap(title, 20), fontsize=12.5,
                 fontweight="700", color=INK, va="top", linespacing=1.35)
        fig.text(x + 0.012, 0.578, _wrap(body, 26), fontsize=9.6, color=INK2,
                 va="top", linespacing=1.6)
        if i < len(steps) - 1:
            fig.text(x + w + gap / 2, 0.505, "→", fontsize=17, color=MUTED,
                     ha="center", va="center", fontweight="700")

    _card(fig, 0.038, 0.075, 0.924, 0.115, face="#fbfbfa")
    fig.text(0.058, 0.148, "WHAT IT IS NOT", fontsize=9.5, color=MUTED,
             fontweight="700", va="center")
    fig.text(0.058, 0.108,
             "Not fine-tuning · not a prompt rewrite · not deleting history — "
             "the old messages stay fully readable, they just stop giving orders.",
             fontsize=11.5, color=INK, va="center")
    fig.text(0.962, 0.030,
             "V-Steer — Zeng, Lee, Zhao & Hockenmaier, arXiv:2607.26228 (COLM 2026)",
             fontsize=8.8, color=MUTED, ha="right")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def recall_panel(summary: dict, path: Path, model_label: str):
    """Authority falls, memory doesn't — the figure that makes this deployable.

    Two lines against the same suppression dial: how often the model still
    *obeys* an instruction in the demoted span, and how often it can still
    *recall a fact* stated in that same span. They have to come apart, or the
    method is just amnesia.
    """
    plt.rcParams["font.sans-serif"] = FONT
    pts = sorted((v["gamma_minus"], v) for v in summary.values()
                 if "gamma_minus" in v and "fact_recall" in v)
    if not pts:
        return
    xs = [p[0] for p in pts]
    obey = [p[1]["stale"] for p in pts]
    recall = [p[1]["fact_recall"] for p in pts]

    fig = plt.figure(figsize=(11, 6.2), dpi=160)
    fig.patch.set_facecolor(WHITE)
    fig.text(0.055, 0.945, "There is a window where authority goes but memory stays",
             fontsize=21, fontweight="700", color=INK, va="top")
    fig.text(0.055, 0.882,
             "The same demoted messages carry an instruction and a fact. Up to γ− = 0.5 the instruction stops being\n"
             "obeyed while the fact stays retrievable. Past that the fact goes too — at the paper's default of 0.75 "
             "recall is already down to 83%.",
             fontsize=11.5, color=INK2, va="top", linespacing=1.5)

    ax = fig.add_axes([0.075, 0.145, 0.79, 0.63])
    ax.set_facecolor(WHITE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)

    safe = [x for x, r in zip(xs, recall) if r >= 0.999]
    if safe:
        ax.axvspan(min(xs), max(safe), color="#f4f8fd", zorder=0)
        ax.text(max(safe), 1.045, " authority gone, memory intact ", fontsize=9.5,
                color=SYSTEM, ha="right", va="center", fontweight="700")

    ax.plot(xs, recall, color=SYSTEM, linewidth=2.5, marker="o", markersize=8,
            markeredgecolor=WHITE, markeredgewidth=1.8, zorder=3,
            label="recalls a fact stated in those messages")
    ax.plot(xs, obey, color=STALE, linewidth=2.5, marker="o", markersize=8,
            markeredgecolor=WHITE, markeredgewidth=1.8, zorder=3,
            label="still obeys the old instruction")

    ax.text(xs[-1] + 0.02, recall[-1], "memory\nkept", color=SYSTEM,
            fontsize=11, fontweight="700", va="center", linespacing=1.35)
    ax.text(xs[-1] + 0.02, obey[-1] + 0.02, "authority\ngone", color=STALE,
            fontsize=11, fontweight="700", va="center", linespacing=1.35)

    ax.set_xlim(min(xs) - 0.02, max(xs) + 0.16)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xticks(xs)
    ax.set_xlabel("γ−   how hard the pre-update messages are suppressed",
                  color=INK2, fontsize=10.5, labelpad=9)
    ax.legend(frameon=False, fontsize=10.5, labelcolor=INK2, loc="center left",
              bbox_to_anchor=(0.02, 0.42))

    fig.text(0.945, 0.035, f"{model_label} · n = {pts[0][1]['n']} per point",
             fontsize=9, color=MUTED, ha="right")
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)


def main(results="results/tiny_main.json", outdir="results/figures",
         model_label="Qwen2.5-0.5B-Instruct · pilot"):
    data = json.loads(Path(results).read_text())
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stats = compare(data["runs"], "conflict", "vsteer_conflict")

    # Plain recall counts failures the model would make anyway -- constraints a
    # 0.5B cannot follow even with an empty transcript are not this method's to
    # fix. The honest denominator is failures the stale history *caused*.
    if "no_history" in data["runs"]:
        k = lambda r: (r["family"], r["query"], r.get("variant", 0))
        nh = {k(r): r["verdict"] == "system" for r in data["runs"]["no_history"]}
        cf = {k(r): r["verdict"] == "system" for r in data["runs"]["conflict"]}
        vs = {k(r): r["verdict"] == "system" for r in data["runs"]["vsteer_conflict"]}
        keys = [x for x in nh if x in cf and x in vs]
        caused = [x for x in keys if nh[x] and not cf[x]]
        if caused:
            stats["caused_recall"] = sum(vs[x] for x in caused) / len(caused)
            stats["caused_n"] = len(caused)
        stats["ceiling"] = sum(nh[x] for x in keys) / len(keys)

    # how often the OLD rule wins when nothing is done
    conflict = data["runs"]["conflict"]
    stats["rate_a_stale"] = sum(r["verdict"] == "stale" for r in conflict) / len(conflict)

    # how many answers the "ignore earlier instructions" line actually changed
    if "prompt_fix" in data["runs"]:
        k = lambda r: (r["family"], r["query"], r.get("variant", 0))
        cf = {k(r): r["verdict"] for r in conflict}
        pf = {k(r): r["verdict"] for r in data["runs"]["prompt_fix"]}
        stats["prompt_fix_delta"] = sum(
            1 for x in cf if x in pf and cf[x] != pf[x]
        )

    # pick a real case where steering flipped the outcome, preferring one whose
    # content is unchanged so the figure shows format moving, not facts moving
    before = {(r["family"], r["query"]): r for r in data["runs"]["conflict"]}
    after = {(r["family"], r["query"]): r for r in data["runs"]["vsteer_conflict"]}
    pick = None
    # `prefix` first: the constraint is carried by the very first token, which
    # is exactly what the attribution can see, so it is the honest best case.
    # NOT `options` -- that family barely moves (+0.05), and using it here
    # would advertise the one result the method does not deliver.
    for key in ("prefix", "case", "bullet", "lang", "json"):
        for k in before:
            if k[0] == key and before[k]["verdict"] == "stale" \
                    and after[k]["verdict"] == "system":
                pick = k
                break
        if pick:
            break
    if pick is None:
        raise SystemExit("no flipped case found")

    from .staleset import FAMILIES

    fam = next(f for f in FAMILIES if f.key == pick[0])
    example = {
        "system": fam.system,
        "stale": fam.stale,
        "query": pick[1],
        "before": before[pick]["text"].strip(),
        "after": after[pick]["text"].strip(),
    }
    hero_fork(example, stats, out / "hero.png", model_label)
    hero_rules(example, stats, out / "hero_rules.png", model_label)
    hero_sbs(example, stats, out / "hero_sbs.png", model_label)
    confusion_panel(stats, out / "stats.png", model_label)
    print("wrote", out / "hero.png")
    print("wrote", out / "stats.png")
    print("example used:", pick)


if __name__ == "__main__":
    import sys

    main(*(sys.argv[1:] or []))
