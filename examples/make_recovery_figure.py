"""What a stale instruction costs, and how much of it the union rule buys back.

Three states per model, all measuring the same thing -- a **useful answer**,
meaning the system constraint was obeyed AND the fact was actually in the reply.
Format alone is not enough: under a strong edit one model emits
`{"question": "When does my flight land?"}`, which is perfectly valid JSON and
answers nothing.

  ceiling        same system rule, no conflicting history, no steering.
                 What the model can do when nothing fights it.
  conflict       the stale rule is in the transcript, no steering. What ships.
  best cell      the same conflict with the best (gamma+, gamma-) in the grid.

The gap between the first two is the damage; the gap between the second and
third is what the method recovers.

    python examples/make_recovery_figure.py

Reads the *.rescored.json files -- two checkers were wrong and the numbers here
are the corrected ones (see examples/rescore_atlas.py).
"""
import collections
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WHITE, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CEIL, CONF, BEST = "#8a6fbf", "#e34948", "#2a78d6"
MODELS = ("small", "mid", "llama")
LABEL = {"small": "malý", "mid": "střední", "llama": "Llama"}


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def useful(rs):
    return sum(r["which_rule_won"] == "system" and r["recalled"] for r in rs)


rows = {}
for m in MODELS:
    A = json.load(open(f"results/atlas_{m}.rescored.json"))["records"]
    C = json.load(open(f"results/ceiling_{m}.rescored.json"))["records"]
    base = [r for r in A if r["gamma_minus"] == 0.0]
    cells = collections.defaultdict(list)
    for r in A:
        cells[(r["gamma_plus"], r["gamma_minus"])].append(r)
    gp, gm = max(cells, key=lambda k: useful(cells[k]) / len(cells[k]))
    rows[m] = {"strop": (useful(C), len(C)),
               "konflikt": (useful(base), len(base)),
               "nejlepší": (useful(cells[(gp, gm)]), len(cells[(gp, gm)])),
               "cell": (gp, gm)}

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "system-ui", "sans-serif"]
fig = plt.figure(figsize=(11, 6.3), dpi=160)
fig.patch.set_facecolor(WHITE)
fig.text(0.055, 0.945, "Co zastaralá instrukce stojí a kolik jde vrátit",
         fontsize=22, fontweight="700", color=INK, va="top")
fig.text(0.055, 0.878,
         "Užitečná odpověď = systémové omezení dodrženo A fakt v odpovědi.\n"
         "Jedna věta z minulosti srazí dva ze tří modelů z 93–97 % na 0–3 %.",
         fontsize=12, color=INK2, va="top", linespacing=1.5)

ax = fig.add_axes([0.075, 0.185, 0.90, 0.495])
ax.set_facecolor(WHITE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=12, length=0)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, lw=0.8)

w = 0.25
for j, (key, color) in enumerate((("strop", CEIL), ("konflikt", CONF),
                                  ("nejlepší", BEST))):
    xs = [i + (j - 1) * w for i in range(len(MODELS))]
    ys = [rows[m][key][0] / rows[m][key][1] for m in MODELS]
    ax.bar(xs, ys, width=w * 0.88, color=color, zorder=2)
    for x, m in zip(xs, MODELS):
        lo, hi = wilson(*rows[m][key])
        ax.plot([x, x], [lo, hi], color=WHITE, lw=1.6, zorder=3)
        ax.plot([x, x], [lo, hi], color=color, lw=1.0, alpha=0.55, zorder=4)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.028, f"{100*y:.0f} %", ha="center", fontsize=11,
                color=INK2, fontweight="700")

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=13.5, color=INK)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
ax.set_ylim(0, 1.10)
ax.set_xlim(-0.55, len(MODELS) - 0.45)

# A legend row under the subtitle. Three labels written above one bar group
# overlap each other at this width, so they go outside the axes.
for x, (txt, color) in zip((0.075, 0.30, 0.545),
                           (("bez konfliktu", CEIL),
                            ("konflikt, bez řízení", CONF),
                            ("nejlepší dávka", BEST))):
    fig.patches.append(plt.Rectangle((x, 0.741), 0.017, 0.025, color=color,
                                     transform=fig.transFigure, zorder=5))
    fig.text(x + 0.025, 0.753, txt, fontsize=11.5, color=INK2,
             fontweight="700", va="center")

ax.annotate("malý model se dostane\nnad vlastní strop", (0 + w, 0.86),
            xytext=(46, 40), textcoords="offset points", ha="left",
            color=INK, fontsize=11.5, linespacing=1.45,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1,
                            connectionstyle="arc3,rad=-0.2"))

fig.text(0.945, 0.072,
         "6 rodin × 3 formulace × 6 faktů · strop n=108, bez řízení n=108, "
         "nejlepší buňka n=36 a vybrána post hoc z 21 · úsečky 95% Wilson",
         fontsize=9, color=MUTED, ha="right")
fig.text(0.945, 0.026,
         "V-Steer (Zeng et al., COLM 2026, arXiv:2607.26228) · "
         "old-news: examples/failure_atlas.py, rescore_atlas.py",
         fontsize=9, color=MUTED, ha="right")

for out in ("results/recovery.png",
            os.path.expanduser("~/projekty/personal-goals/career/keynote/assets/"
                               "recovery.png")):
    fig.savefig(out, facecolor=WHITE)
    print(out)
for m in MODELS:
    print(f"  {m:6s} strop {rows[m]['strop']}  konflikt {rows[m]['konflikt']}  "
          f"nejlepsi {rows[m]['nejlepší']} v {rows[m]['cell']}")
