"""Where the V-Steer union rule works, and where it never does.

V-Steer reports aggregate effectiveness per model. This asks the next question
instead: at scale, across models and constraint types, HOW does it fail?

756 cases per model -- 6 constraint families x 6 facts x 21 (gamma+, gamma-)
cells -- run on three models. The bar is how often the SYSTEM instruction won,
i.e. the union rule did its job.

The finding the figure carries: the constraint family predicts the outcome more
sharply than the model does. `options` is recovered on every model (75-93 %);
`bullet` is recovered on none of them (0 %, 0 %, 4 %). Everything else is
model-dependent. A single "does V-Steer work on model X" number averages over a
spread that runs from 0 to 100 within the same model.

    python3 examples/make_atlas_figure.py
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WHITE, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# validated with dataviz/scripts/validate_palette.js -- all six checks pass
COLORS = {"small": "#2a78d6", "mid": "#e34948", "llama": "#8a6fbf"}
LABEL = {"small": "malý", "mid": "střední", "llama": "Llama"}
FAM = {"options": "options", "case": "case", "json": "json",
       "length": "length", "prefix": "prefix", "bullet": "bullet"}


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


data = {}
for name in ("small", "mid", "llama"):
    rs = json.load(open(f"results/atlas_{name}.json"))["records"]
    data[name] = {}
    for fam in FAM:
        c = [r for r in rs if r["family"] == fam]
        k = sum(r["which_rule_won"] == "system" for r in c)
        data[name][fam] = (k, len(c))

# order families by how well they do on average: the spread is the point
order = sorted(FAM, key=lambda f: -sum(data[m][f][0] / data[m][f][1] for m in data))

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "system-ui", "sans-serif"]
fig = plt.figure(figsize=(11, 6.3), dpi=160)
fig.patch.set_facecolor(WHITE)
fig.text(0.055, 0.945, "Co se dá vrátit a co ne", fontsize=22,
         fontweight="700", color=INK, va="top")
fig.text(0.055, 0.878,
         "Jak často zvítězila systémová instrukce (V-Steer, sjednocovací pravidlo, rov. 9).\n"
         "Typ omezení rozhoduje víc než model: „options“ se vrátí všude, „bullet“ nikde.",
         fontsize=12, color=INK2, va="top", linespacing=1.5)

ax = fig.add_axes([0.075, 0.20, 0.90, 0.545])
ax.set_facecolor(WHITE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=11.5, length=0)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, lw=0.8)

w = 0.26
for j, m in enumerate(("small", "mid", "llama")):
    xs = [i + (j - 1) * w for i in range(len(order))]
    ys = [data[m][f][0] / data[m][f][1] for f in order]
    cis = [wilson(*data[m][f]) for f in order]
    # 2px surface gap between adjacent bars: width slightly under the slot
    ax.bar(xs, ys, width=w * 0.88, color=COLORS[m], label=LABEL[m], zorder=2)
    for x, y, (lo, hi) in zip(xs, ys, cis):
        ax.plot([x, x], [lo, hi], color=WHITE, lw=1.6, zorder=3)
        ax.plot([x, x], [lo, hi], color=COLORS[m], lw=1.0, alpha=0.55, zorder=4)

ax.set_xticks(range(len(order)))
ax.set_xticklabels([FAM[f] for f in order], fontsize=12.5, color=INK)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
ax.set_ylim(0, 1.06)
ax.set_xlim(-0.6, len(order) - 0.4)
leg = ax.legend(frameon=False, fontsize=12, loc="upper right", ncol=3,
                handlelength=1.1, columnspacing=1.4)
for t in leg.get_texts():
    t.set_color(INK2)

ax.annotate("na žádném modelu\nnad 4 %", (len(order) - 1, 0.045),
            xytext=(-4, 66), textcoords="offset points", ha="center",
            color=INK, fontsize=11.5, linespacing=1.45,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

fig.text(0.945, 0.075,
         "756 případů na model · 6 rodin omezení × 6 faktů × 21 buněk (γ+, γ−) · "
         "greedy · úsečky 95% Wilson",
         fontsize=9, color=MUTED, ha="right")
fig.text(0.945, 0.028,
         "V-Steer (Zeng et al., COLM 2026, arXiv:2607.26228) · "
         "old-news: examples/failure_atlas.py",
         fontsize=9, color=MUTED, ha="right")

for out in ("results/atlas_families.png",
            os.path.expanduser("~/projekty/personal-goals/career/keynote/assets/"
                               "atlas_families.png")):
    fig.savefig(out, facecolor=WHITE)
    print(out)
