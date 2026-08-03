"""Figures for StaleSet results.

The story every chart here has to tell: without the intervention, instructions
left over from before the update decide the answer; with it, the current system
prompt does. Colour is doing polarity work throughout -- blue = the current
system prompt won, red = the stale history won -- so it uses the documented
diverging pair, never a rainbow.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# --- palette ---------------------------------------------------------------
LIGHT = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "system": "#2a78d6",  # diverging pole: current system prompt won
    "stale": "#e34948",  # diverging pole: stale history won
    "neutral": "#f0efec",  # diverging midpoint: neither
    "series2": "#eb6834",
}
DARK = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "system": "#3987e5",
    "stale": "#e66767",
    "neutral": "#383835",
    "series2": "#d95926",
}

FONT = ["DejaVu Sans", "system-ui", "sans-serif"]

CONDITION_LABEL = {
    "no_history": "No history\n(upper bound)",
    "conflict": "Stale history\n(no fix)",
    "prompt_fix": "Stale history\n+ prompt fix",
    "vsteer_conflict": "Stale history\n+ V-Steer",
    "aligned": "Aligned history\n(control)",
    "vsteer_aligned": "Aligned history\n+ V-Steer",
}


def _style(ax, P, grid_axis="x"):
    ax.set_facecolor(P["surface"])
    ax.figure.set_facecolor(P["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(P["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=P["muted"], labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=P["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _title(ax, P, title, subtitle=None):
    ax.set_title(title, color=P["ink"], fontsize=13, fontweight="600",
                 loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=P["ink2"],
                fontsize=9.5, va="bottom")


def _rounded(ax, x, y, w, h, color, P, r=0.012):
    """Bar with 4px-ish rounded data-end, anchored at the baseline."""
    if w <= 0:
        return
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), max(w - r, 1e-6), h,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=r),
            facecolor=color, edgecolor="none", zorder=3,
        )
    )


# --- fig 1: who won --------------------------------------------------------
def fig_outcome(summary: dict, path: Path, mode="light", order=None):
    P = LIGHT if mode == "light" else DARK
    plt.rcParams["font.sans-serif"] = FONT
    order = order or [
        "no_history", "conflict", "prompt_fix", "vsteer_conflict", "aligned",
        "vsteer_aligned",
    ]
    rows = [k for k in order if k in summary]

    fig, ax = plt.subplots(figsize=(9.2, 0.78 * len(rows) + 2.3))
    _style(ax, P, grid_axis="x")

    ypos = np.arange(len(rows))[::-1]
    h = 0.56
    gap = 0.004  # 2px-ish surface gap between stacked fills

    for y, key in zip(ypos, rows):
        s = summary[key]
        segs = [("system", s["system"]), ("neither", s["neither"]),
                ("stale", s["stale"])]
        x = 0.0
        for name, val in segs:
            if val <= 0:
                continue
            color = {"system": P["system"], "stale": P["stale"],
                     "neither": P["neutral"]}[name]
            ax.barh(y, val - gap, left=x, height=h, color=color,
                    edgecolor=P["surface"], linewidth=1.5, zorder=3)
            if val >= 0.10:
                ax.text(x + val / 2, y, f"{val * 100:.0f}%",
                        ha="center", va="center", fontsize=9.5,
                        color=P["ink"] if name == "neither" else "#ffffff",
                        fontweight="600", zorder=4)
            x += val

    ax.set_yticks(ypos)
    ax.set_yticklabels([CONDITION_LABEL.get(k, k) for k in rows],
                       color=P["ink2"], fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylim(-0.7, len(rows) - 0.3)

    _title(ax, P, "Which instruction actually won",
           "share of StaleSet answers obeying each constraint")

    handles = [
        mpatches.Patch(facecolor=P["system"], label="current system prompt"),
        mpatches.Patch(facecolor=P["neutral"], edgecolor=P["axis"], label="neither"),
        mpatches.Patch(facecolor=P["stale"], label="stale history"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, frameon=False, fontsize=9.5, labelcolor=P["ink2"])
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=P["surface"])
    plt.close(fig)


# --- fig 2: per constraint family -----------------------------------------
def fig_family(runs: dict, path: Path, mode="light"):
    P = LIGHT if mode == "light" else DARK
    plt.rcParams["font.sans-serif"] = FONT

    def rate(key):
        out: dict[str, list] = {}
        for r in runs.get(key, []):
            out.setdefault(r["family"], []).append(r["verdict"] == "system")
        return {k: sum(v) / len(v) for k, v in out.items()}

    a, b = rate("conflict"), rate("vsteer_conflict")
    fams = [f for f in a if f in b]
    if not fams:
        return
    fams.sort(key=lambda f: b[f] - a[f], reverse=True)

    x = np.arange(len(fams))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    _style(ax, P, grid_axis="y")

    for i, f in enumerate(fams):
        _rounded(ax, i - w - 0.01, 0, 0, 0, P["system"], P)  # keep import used
    ax.bar(x - w / 2 - 0.01, [a[f] for f in fams], w, color=P["muted"],
           edgecolor=P["surface"], linewidth=1.5, zorder=3, label="no steering")
    ax.bar(x + w / 2 + 0.01, [b[f] for f in fams], w, color=P["system"],
           edgecolor=P["surface"], linewidth=1.5, zorder=3, label="V-Steer")

    for i, f in enumerate(fams):
        if b[f] - a[f] >= 0.15:
            ax.text(i + w / 2 + 0.01, b[f] + 0.03, f"+{(b[f] - a[f]) * 100:.0f}",
                    ha="center", fontsize=9, color=P["ink2"], fontweight="600")

    ax.set_xticks(x)
    ax.set_xticklabels(fams, color=P["ink2"], fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    _title(ax, P, "Where the fix bites, by constraint type",
           "answers following the current system prompt, stale history present")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=P["ink2"], loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=P["surface"])
    plt.close(fig)


# --- fig 3: the bad heads --------------------------------------------------
def fig_heads(delta: np.ndarray, mask: np.ndarray, path: Path, mode="light",
              n_rep: int = 1):
    """delta [L, H_q] inversion scores; mask [L, H_kv] what got edited."""
    P = LIGHT if mode == "light" else DARK
    plt.rcParams["font.sans-serif"] = FONT

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "sys_stale", [P["system"], P["neutral"], P["stale"]]
    )
    lim = float(np.abs(delta).max()) or 1.0

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    im = ax.imshow(delta.T, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim,
                   interpolation="nearest")
    ax.set_facecolor(P["surface"])
    fig.set_facecolor(P["surface"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=P["muted"], labelsize=9, length=0)
    ax.set_xlabel("layer", color=P["ink2"], fontsize=9.5)
    ax.set_ylabel("query head", color=P["ink2"], fontsize=9.5)

    # outline the KV groups that were actually edited
    for l in range(mask.shape[0]):
        for g in range(mask.shape[1]):
            if mask[l, g]:
                ax.add_patch(
                    mpatches.Rectangle(
                        (l - 0.5, g * n_rep - 0.5), 1, n_rep, fill=False,
                        edgecolor=P["ink"], linewidth=1.1, zorder=4,
                    )
                )

    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=P["muted"], labelsize=8.5, length=0)
    cb.set_label("stale advantage   φ(stale) − φ(system)", color=P["ink2"],
                 fontsize=9)

    _title(ax, P, "Where the model hands authority to old messages",
           "direct logit attribution per head; outlined = edited by V-Steer")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=P["surface"])
    plt.close(fig)


# --- fig 4: dial ------------------------------------------------------------
def fig_sweep(summary: dict, path: Path, mode="light"):
    P = LIGHT if mode == "light" else DARK
    plt.rcParams["font.sans-serif"] = FONT
    pts: dict[float, dict[float, float]] = {}
    for k, v in summary.items():
        if "gamma_minus" not in v:
            continue
        pts.setdefault(v["gamma_plus"], {})[v["gamma_minus"]] = v["system"]
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    _style(ax, P, grid_axis="y")
    colors = [P["system"], P["series2"]]
    for (gp, series), color in zip(sorted(pts.items()), colors):
        xs = sorted(series)
        ys = [series[x] for x in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=P["surface"], markeredgewidth=1.5, zorder=3,
                label=f"γ+ = {gp}")
        ax.text(xs[-1] + 0.015, ys[-1], f"γ+ = {gp}", color=color, fontsize=9.5,
                va="center", fontweight="600")

    ax.set_xlabel("γ−   (how hard stale history is suppressed)", color=P["ink2"],
                  fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlim(-0.04, 1.18)
    _title(ax, P, "One dial, not a rewrite",
           "answers following the current system prompt vs suppression strength")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=P["ink2"], loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=P["surface"])
    plt.close(fig)


# --- fig 5: age -------------------------------------------------------------
def fig_age(summary: dict, path: Path, mode="light"):
    P = LIGHT if mode == "light" else DARK
    plt.rcParams["font.sans-serif"] = FONT
    series: dict[str, dict[int, float]] = {}
    for k, v in summary.items():
        if "epochs_back" not in v:
            continue
        series.setdefault(v["policy"], {})[v["epochs_back"]] = v["system"]
    if not series:
        return

    labels = {"none": "no steering", "binary": "V-Steer (binary)",
              "epoch_decay": "age-graded"}
    colors = {"none": P["muted"], "binary": P["system"], "epoch_decay": P["series2"]}

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    _style(ax, P, grid_axis="y")
    for name, pts in series.items():
        xs = sorted(pts)
        ax.plot(xs, [pts[x] for x in xs], color=colors.get(name, P["system"]),
                linewidth=2, marker="o", markersize=6,
                markeredgecolor=P["surface"], markeredgewidth=1.5, zorder=3,
                label=labels.get(name, name))
    ax.set_xlabel("how many app versions ago the history was written",
                  color=P["ink2"], fontsize=9.5)
    ax.set_xticks(sorted({x for p in series.values() for x in p}))
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    _title(ax, P, "Old history should count for less, the older it is",
           "answers following the current system prompt")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=P["ink2"], loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=P["surface"])
    plt.close(fig)


def render_all(results_path: str, outdir: str = "results/figures"):
    data = json.loads(Path(results_path).read_text())
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for mode in ("light", "dark"):
        sfx = f"_{mode}.png"
        if any(k in data["summary"] for k in ("conflict", "vsteer_conflict")):
            fig_outcome(data["summary"], out / f"outcome{sfx}", mode)
            made.append(out / f"outcome{sfx}")
            fig_family(data["runs"], out / f"family{sfx}", mode)
            made.append(out / f"family{sfx}")
        fig_sweep(data["summary"], out / f"sweep{sfx}", mode)
        fig_age(data["summary"], out / f"age{sfx}", mode)
    return [p for p in made if p.exists()]


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "results/tiny_main.json"
    for p in render_all(src):
        print("wrote", p)
