#!/usr/bin/env python3
"""Conceptual schematic of the iterated donor game: one example round.

Walks through a single round with concrete numbers. The agents pair up, the
donor gives 4 of its 10 units, the recipient receives double, and the figure
then shows how the generation progresses and how the next one is seeded.

Numbers are the defaults in slurm/experiments/donor_game_baseline.py
(NUM_AGENTS=12, INITIAL_ENDOWMENT=10, COOP_GAIN=2, NUM_ROUNDS=12,
NUM_GENERATIONS=10, SELECTION_METHOD='top').

Drawn at final print size (5.5in, the NeurIPS \\textwidth) so it is included
at width=\\textwidth with no rescaling; nothing on it falls below 7pt.

Output: figures/donor_game_setup.pdf and figures/donor_game_setup.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# ── The game, and the worked example ──────────────────────────────────────
N_AGENTS = 12
ENDOWMENT = 10
MULTIPLIER = 2
N_ROUNDS = 12
N_GENERATIONS = 10

GIVEN = 4                                  # the example donation
RECEIVED = MULTIPLIER * GIVEN              # 8
KEPT = ENDOWMENT - GIVEN                   # 6
RECIP_BANKS = ENDOWMENT + RECEIVED         # 18
PAIR_TOTAL = KEPT + RECIP_BANKS            # 24
PAIR_IF_NOTHING = 2 * ENDOWMENT            # 20

# ── Type ──────────────────────────────────────────────────────────────────
# Times matches the NeurIPS body font, so the figure reads as part of the
# page rather than as a pasted-in plot. For a sans-serif paper, swap this
# for "Helvetica Neue" or "Arial" and change FAMILY to "sans-serif".
FONT = "Times New Roman"
FAMILY = "serif"

FS_HEAD = 10.0    # panel headings
FS_BODY = 9.0     # explanatory lines (the floor: nothing smaller than this)
FS_NUM = 10.0     # the banked totals
FS_BIG = 12.0     # the D / R letters in the circles

# ── Palette (validated categorical slots 1-3 + neutral ink) ───────────────
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#7c7b76"
HAIR = "#c9c8c3"

DONOR = "#2563c9"
DONOR_BG = "#dce9f9"
RECIP = "#d1342f"
RECIP_BG = "#fbe3d7"
SURV = "#1baf7a"
SURV_BG = "#e4f5ee"
NEUTRAL_BG = "#eeeeec"

plt.rcParams["font.family"] = FAMILY
plt.rcParams[f"font.{FAMILY}"] = [FONT, "Times", "DejaVu Serif"]
plt.rcParams["pdf.fonttype"] = 42          # keep text editable/selectable
plt.rcParams["ps.fonttype"] = 42

# ── Canvas ────────────────────────────────────────────────────────────────
# W units span FIG_W_IN inches, so 1 unit = 1/13.27 in. Font sizes stay in
# points, which is what keeps them honest at the final printed size.
W = 73.0
FIG_W_IN = 5.5
Y0, Y1 = 0.5, 32.5

fig, ax = plt.subplots(figsize=(FIG_W_IN, (Y1 - Y0) * FIG_W_IN / W))
ax.set_xlim(0, W)
ax.set_ylim(Y0, Y1)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


# ── Helpers ───────────────────────────────────────────────────────────────
def text(x, y, s, size=FS_BODY, color=INK, weight="normal", ha="left",
         va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=6, linespacing=1.45, **kw)


def box(x0, y0, x1, y1, fc="none", ec=HAIR, lw=0.7, r=0.8, ls="solid", z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=z))


def dot(x, y, r, fc, ec, lw=0.8, z=3):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec,
                        linewidth=lw, zorder=z))


def arrow(p0, p1, color=INK2, lw=1.0, head=7, z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                                 linewidth=lw, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0))


def heading(x, y, letter, title):
    text(x, y, f"({letter})", size=FS_HEAD, weight="bold")
    text(x + 3.6, y, title, size=FS_HEAD, weight="bold")


# Two colours only: blue = donor, red = recipient. Everything else is black.
B, R_ = DONOR, RECIP
H1, H2 = 31.0, 14.6          # row headings


def label(x, y, s, **kw):
    text(x, y, s, color=INK, **kw)


# (a) the 12 agents are paired ───────────────────────────────────────────────
heading(2.0, H1, "a", "Agents are paired")
for cx in (8.0, 19.0, 30.0):
    for cy in (26.8, 22.4):
        ax.plot([cx - 2.6, cx + 2.6], [cy, cy], color=INK, lw=0.6, zorder=2)
        dot(cx - 2.6, cy, 1.35, "white", B, lw=1.0)
        dot(cx + 2.6, cy, 1.35, "white", R_, lw=1.0)
label(19.0, 19.2, f"{N_AGENTS} agents, {N_AGENTS // 2} pairs", ha="center")

# (b) one donation ───────────────────────────────────────────────────────────
heading(38.0, H1, "b", "The donor gives")
label(54.5, 28.4, f"Each starts with {ENDOWMENT} units", ha="center")
dx, rx, cy = 43.5, 65.5, 23.2
dot(dx, cy, 2.6, "white", B, lw=1.1)
text(dx, cy, "D", size=FS_BIG, color=B, weight="bold", ha="center")
dot(rx, cy, 2.6, "white", R_, lw=1.1)
text(rx, cy, "R", size=FS_BIG, color=R_, weight="bold", ha="center")
arrow((dx + 2.8, cy), (rx - 2.8, cy), color=B, lw=1.3, head=8, z=3)
box(52.7, cy - 1.3, 56.3, cy + 1.3, fc="white", ec=B, lw=0.9, r=0.5, z=4)
text(54.5, cy, f"×{MULTIPLIER}", color=B, weight="bold", ha="center")
text(54.5, cy + 2.5, f"gives {GIVEN}", color=B, weight="bold", ha="center")
text(54.5, cy - 2.5, f"gets {RECEIVED}", color=R_, weight="bold", ha="center")
text(dx, 19.2, "Donor", color=B, ha="center")
text(rx, 19.2, "Recipient", color=R_, ha="center")

# (c) what each keeps ────────────────────────────────────────────────────────
heading(2.0, H2, "c", "What each keeps")
for y, col, who, expr, val in (
        (10.6, B, "Donor", f"kept {KEPT} of {ENDOWMENT}", KEPT),
        (6.1, R_, "Recipient", f"own {ENDOWMENT} + {RECEIVED} received", RECIP_BANKS)):
    box(2.6, y - 1.6, 6.4, y + 1.6, fc="none", ec=col, lw=1.0, r=0.5)
    text(4.5, y, str(val), size=FS_NUM, color=col, weight="bold", ha="center")
    label(8.0, y + 0.8, who, weight="bold")
    label(8.0, y - 0.9, expr)
label(2.6, 2.0, f"Pair: {PAIR_TOTAL} units ({PAIR_IF_NOTHING} if nothing is given)")

# (d) a generation = two games ────────────────────────────────────────────────
# One agent's actual roles (bipartiteRoundRobin + extendRounds; game 2 reverses
# every pairing), so the agent is donor 6 times and recipient 6 times per game.
GAME1 = "DRDRDRRDRDRD"
GAME2 = "".join("R" if c == "D" else "D" for c in GAME1)
heading(38.0, H2, "d", "Two games per generation")
for y, lab, seq in ((11.0, "Game 1", GAME1), (7.9, "Game 2", GAME2)):
    label(38.0, y, lab)
    for k, c in enumerate(seq):
        x0 = 44.2 + k * 2.25
        col = B if c == "D" else R_
        box(x0, y - 1.05, x0 + 1.7, y + 1.05, fc="none", ec=col, lw=0.9, r=0.35, z=2)
        text(x0 + 0.85, y, c, color=col, weight="bold", ha="center")
label(38.0, 3.9, f"Then the top {N_AGENTS // 2} survive and {N_AGENTS // 2} new agents\n"
                 f"learn from them. Repeat for {N_GENERATIONS} generations.")


# ── Save ──────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "donor_game_setup.pdf")
png_path = os.path.join(out_dir, "donor_game_setup.png")
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02,
            facecolor="white")
print(f"wrote {pdf_path}  ({FIG_W_IN}in wide, {FONT} {FS_BODY}pt floor)")
print(f"wrote {png_path}")
