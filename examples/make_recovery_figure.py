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
import itertools
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WHITE, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CEIL, CONF, BEST = "#8a6fbf", "#e34948", "#2a78d6"
# Ordered by ceiling: the whole point of the figure is that where a model ends
# up is predicted by where it could start.
MODELS = ("tiny", "small", "commandr", "phi", "olmo", "mid", "aya", "llama")
LABEL = {"tiny": "Qwen2.5\n0.5B", "small": "Qwen2.5\n1.5B",
         "commandr": "Command-R\n7B", "phi": "Phi-3.5\n3.8B",
         "olmo": "OLMo-2\n7B", "mid": "Qwen3\n4B", "aya": "Aya\n8B",
         "llama": "Llama-3.1\n8B"}


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def useful(rs):
    return sum(r["which_rule_won"] == "system" and r["recalled"] for r in rs)


def load(kind, m):
    """Prefer the rescored file where one exists -- the runs made before the
    two checker fixes have one, the later ones never needed it."""
    for suffix in (".rescored.json", ".json"):
        f = f"results/{kind}_{m}{suffix}"
        if os.path.exists(f):
            return json.load(open(f))["records"]
    raise FileNotFoundError(f"{kind}_{m}")


FAMS = ["case", "prefix", "json", "bullet", "length", "options"]

rows = {}
for m in MODELS:
    A = load("atlas", m)
    C = load("ceiling", m)
    base = [r for r in A if r["gamma_minus"] == 0.0]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in A:
        by[(r["gamma_plus"], r["gamma_minus"])][r["family"]].append(r)
    cells = sorted(by)
    # Held-out dose selection. Picking the best cell on the same data it is
    # reported on inflates it by 7-17 points, so the cell is chosen on three
    # constraint families and scored on the other three, averaged over all 20
    # splits. The bar is the honest number, not the best one.
    tot = [0, 0]
    for tr in itertools.combinations(FAMS, 3):
        te = [f for f in FAMS if f not in tr]
        best = max(cells, key=lambda c: useful([x for f in tr for x in by[c][f]]) /
                   len([x for f in tr for x in by[c][f]]))
        ev = [x for f in te for x in by[best][f]]
        tot[0] += useful(ev)
        tot[1] += len(ev)
    rows[m] = {"strop": (useful(C), len(C)),
               "konflikt": (useful(base), len(base)),
               "nejlepší": tuple(tot)}

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "system-ui", "sans-serif"]
fig = plt.figure(figsize=(13, 6.6), dpi=160)
fig.patch.set_facecolor(WHITE)
fig.text(0.055, 0.945, "Co zastaralá instrukce stojí a kolik jde vrátit",
         fontsize=22, fontweight="700", color=INK, va="top")
fig.text(0.055, 0.878,
         "Užitečná odpověď = systémové omezení dodrženo A fakt v odpovědi. "
         "Modely seřazené podle stropu.\n"
         "Dávka vybraná na jiných rodinách omezení, ne na týchž datech. "
         "Nad strop se dostanou jen dva nejslabší modely.",
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

w = 0.27
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
        ax.text(x, y + 0.025, f"{100*y:.0f}", ha="center", fontsize=9.5,
                color=INK2, fontweight="700")

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=10.5, color=INK)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
ax.set_ylim(0, 1.13)
ax.set_xlim(-0.55, len(MODELS) - 0.45)

# A legend row under the subtitle. Three labels written above one bar group
# overlap each other at this width, so they go outside the axes.
for x, (txt, color) in zip((0.075, 0.30, 0.545),
                           (("bez konfliktu", CEIL),
                            ("konflikt, bez řízení", CONF),
                            ("nejlepší dávka (held-out)", BEST))):
    fig.patches.append(plt.Rectangle((x, 0.741), 0.017, 0.025, color=color,
                                     transform=fig.transFigure, zorder=5))
    fig.text(x + 0.025, 0.753, txt, fontsize=11.5, color=INK2,
             fontweight="700", va="center")

# The split is empirical, not round: every model with a ceiling <= 48 % ends up
# above it, every model from 53 % up does not. Phi (53 %) is the first on the
# far side and it loses slightly, so it belongs on the right.
ax.axvline(1.5, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=1)
ax.text(0.5, 1.02, "edit skončí NAD stropem", fontsize=10.5,
        color=INK, ha="center", fontweight="700")
ax.text(4.8, 1.02, "edit skončí POD stropem — ztráta až 54 bodů",
        fontsize=10.5, color=INK, ha="center", fontweight="700")

fig.text(0.945, 0.072,
         "6 rodin omezení × 6 faktů · strop n=108 (3 formulace), bez řízení n=108, "
         "dávka vybrána held-out na 3 rodinách a skórována na zbylých 3, průměr "
         "přes všech 20 rozdělení · úsečky 95% Wilson",
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
    r = rows[m]
    print(f"  {m:9s} strop {100*r['strop'][0]/r['strop'][1]:5.1f} %  "
          f"konflikt {100*r['konflikt'][0]/r['konflikt'][1]:5.1f} %  "
          f"held-out {100*r['nejlepší'][0]/r['nejlepší'][1]:5.1f} %")
