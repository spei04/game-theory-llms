#!/usr/bin/env python3
"""
Distribution curves of cooperation across setup × model cells.

The experimental design has four axes of variability:
  - MODEL   (claude / gpt / gemini)               — controllable
  - PROMPT/SETUP (baseline, neutral_*, aggressive, ...) — controllable
  - RUN     (independent replicates, ~20 per cell) — stochastic
  - EVOLUTION (10 generations per run)             — stochastic

This script produces distribution curves for each (setup, model) cell so you
can see how much of the outcome variability comes from the controllable dims
vs. the noise dims.

Two metrics are plotted:
  1. RUN-LEVEL:  mean donation % across every donor-round in a run.
                 One value per run → distribution over ~20 replicates.
  2. GEN-LEVEL:  average total_final_score across the 12 agents in a
                 generation. Ten values per run → distribution over
                 ~200 (run, gen) samples per cell.

The RUN-level curves isolate run-stochasticity (one number per full 10-gen
tournament). The GEN-level curves mix in evolution-stochasticity as well.
Difference in spread between the two is the extra noise contributed by
generation-to-generation variability within a single tournament.

Output:  <output_dir>/cooperation_distributions_{run,gen}.png
Usage:
    python figures/plot_cooperation_distributions.py \\
        [--setups neutral_baseline aggressive collaborative ...] \\
        [--models claude gpt gemini] \\
        [--output_dir figures/cooperation_distributions]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ── defaults ────────────────────────────────────────────────────────────────

DEFAULT_SETUPS = [
    "neutral_baseline",
    "aggressive",
    "collaborative",
    "fixed_rounds",
    "neutral_claude",
    "neutral_gpt",
    "neutral_gemini",
    "history_3traces",
    "history_3chain",
]

DEFAULT_MODELS = ["claude", "gpt", "gemini"]

MODEL_COLORS = {
    "claude": "#c85200",
    "gpt":    "#5f9ea0",
    "gemini": "#7a68a6",
}

ENDOWMENT_PER_ROUND = 10
ROUNDS_PER_GAME = 12
GAMES_PER_GEN = 2
BASE_POOL = ENDOWMENT_PER_ROUND * ROUNDS_PER_GAME * GAMES_PER_GEN   # =240


# ── data extraction ─────────────────────────────────────────────────────────

def find_runs(setup, model):
    return sorted(glob.glob(
        os.path.join("simulations", setup, model, "Final_Accum*.json")))


def run_donation_pct(accum_path):
    """Return the mean donation % over EVERY donor-round in this run.
    Range 0-100. Averages evenly across all (agent, generation, round) tuples."""
    with open(accum_path) as f:
        data = json.load(f)
    pcts = []
    for r in data.get("agents_data", []):
        if not r.get("is_donor"):
            continue
        donated   = r.get("donated", 0) or 0
        remaining = r.get("resources", 0) or 0
        denom = donated + remaining          # = endowment for that round
        if denom > 0:
            pcts.append(100.0 * donated / denom)
    return float(np.mean(pcts)) if pcts else None


def gen_avg_final_score(accum_path):
    """Return {gen: avg_total_final_score across 12 agents}.
    total_final_score = 240 - Σ donated + Σ received (the survival-selection metric)."""
    with open(accum_path) as f:
        data = json.load(f)

    per_agent = defaultdict(lambda: defaultdict(lambda: {"donated": 0.0, "received": 0.0}))
    for r in data.get("agents_data", []):
        g = r["current_generation"]
        a = r["agent_name"]
        per_agent[g][a]["donated"]  += r.get("donated", 0) or 0
        per_agent[g][a]["received"] += r.get("received", 0) or 0

    out = {}
    for g, agents in per_agent.items():
        scores = [BASE_POOL - v["donated"] + v["received"] for v in agents.values()]
        if scores:
            out[g] = float(np.mean(scores))
    return out


def collect(setups, models):
    """Returns two nested dicts keyed by (setup, model):
        run_level[(setup, model)] = [mean_donation_pct per run]  (len ≈ 20)
        gen_level[(setup, model)] = [avg_total_final_score per (run, gen)]  (len ≈ 200)"""
    run_level = defaultdict(list)
    gen_level = defaultdict(list)

    for setup in setups:
        for model in models:
            paths = find_runs(setup, model)
            for p in paths:
                mp = run_donation_pct(p)
                if mp is not None:
                    run_level[(setup, model)].append(mp)
                for _, sc in gen_avg_final_score(p).items():
                    gen_level[(setup, model)].append(sc)
    return run_level, gen_level


# ── plotting ────────────────────────────────────────────────────────────────

def kde_lines(ax, values_by_cell, models, xlabel, xlim=None):
    """One horizontal KDE per model within each setup group.
    Uses a Gaussian KDE via numpy; falls back to a rug if <2 samples."""
    from scipy.stats import gaussian_kde
    from itertools import groupby

    setups_in_order = sorted({k[0] for k in values_by_cell})

    # x grid
    all_vals = [v for vs in values_by_cell.values() for v in vs]
    if not all_vals:
        return
    x_min = xlim[0] if xlim else min(all_vals) - 0.05 * (max(all_vals) - min(all_vals))
    x_max = xlim[1] if xlim else max(all_vals) + 0.05 * (max(all_vals) - min(all_vals))
    xs = np.linspace(x_min, x_max, 300)

    n_setups = len(setups_in_order)
    fig, axes = plt.subplots(n_setups, 1, figsize=(9, 1.4 * n_setups + 1),
                             sharex=True)
    if n_setups == 1:
        axes = [axes]

    for i, setup in enumerate(setups_in_order):
        ax = axes[i]
        for model in models:
            vs = values_by_cell.get((setup, model), [])
            if len(vs) < 2:
                if vs:
                    ax.axvline(vs[0], color=MODEL_COLORS[model], lw=1, alpha=0.5)
                continue
            try:
                kde = gaussian_kde(vs)   # bw_method=None → Scott's rule, adapts to N
                ys = kde(xs)
                ax.fill_between(xs, ys, color=MODEL_COLORS[model], alpha=0.25,
                                label=f"{model} (n={len(vs)})")
                ax.plot(xs, ys, color=MODEL_COLORS[model], lw=1.6)
            except Exception:
                ax.hist(vs, bins=15, color=MODEL_COLORS[model], alpha=0.35,
                        density=True, label=f"{model} (n={len(vs)})")

            median = float(np.median(vs))
            ax.axvline(median, color=MODEL_COLORS[model], lw=1.2, ls="--", alpha=0.9)

        ax.set_ylabel(setup, rotation=0, ha="right", va="center",
                      fontsize=9, labelpad=8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    axes[-1].set_xlabel(xlabel, fontsize=10)
    if xlim:
        axes[-1].set_xlim(*xlim)
    return fig


def violin_grid(values_by_cell, models, setups, ylabel, ylim=None):
    """One violin per (setup, model) cell in a single panel.
    setups on x-axis, models jittered within each group."""
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(setups)), 6))
    n_models = len(models)
    group_width = 0.8
    step = group_width / max(n_models, 1)

    for m_idx, model in enumerate(models):
        positions, data = [], []
        for s_idx, setup in enumerate(setups):
            vs = values_by_cell.get((setup, model), [])
            if len(vs) >= 2:
                positions.append(s_idx - group_width / 2 + step * (m_idx + 0.5))
                data.append(vs)
        if not data:
            continue
        parts = ax.violinplot(data, positions=positions, widths=step * 0.85,
                              showmeans=False, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(MODEL_COLORS[model])
            pc.set_edgecolor(MODEL_COLORS[model])
            pc.set_alpha(0.55)
        parts["cmedians"].set_color(MODEL_COLORS[model])
        parts["cmedians"].set_linewidth(2)

        # Scatter individual points for small n
        for pos, vs in zip(positions, data):
            xs = pos + np.random.uniform(-step * 0.15, step * 0.15, len(vs))
            ax.scatter(xs, vs, color=MODEL_COLORS[model], s=6, alpha=0.35,
                       edgecolors="none")

    ax.set_xticks(range(len(setups)))
    ax.set_xticklabels(setups, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=MODEL_COLORS[m], label=m, alpha=0.6) for m in models]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    return fig


def kde_lines_by_model(values_by_cell, models, setups, xlabel, xlim=None):
    """Facet by model (one panel per model). Within each panel, one KDE curve
    per prompt/setup, colored distinctly. Best for asking 'how does prompt
    affect THIS model?'"""
    from scipy.stats import gaussian_kde

    all_vals = [v for vs in values_by_cell.values() for v in vs]
    if not all_vals:
        return None
    x_min = xlim[0] if xlim else min(all_vals) - 0.05 * (max(all_vals) - min(all_vals))
    x_max = xlim[1] if xlim else max(all_vals) + 0.05 * (max(all_vals) - min(all_vals))
    xs = np.linspace(x_min, x_max, 300)

    cmap = plt.get_cmap("tab10")
    setup_colors = {s: cmap(i % 10) for i, s in enumerate(setups)}

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(9, 2.5 * n_models),
                             sharex=True)
    if n_models == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        for setup in setups:
            vs = values_by_cell.get((setup, model), [])
            if len(vs) < 2:
                continue
            try:
                kde = gaussian_kde(vs)   # bw_method=None → Scott's rule, adapts to N
                ys = kde(xs)
                ax.plot(xs, ys, color=setup_colors[setup], lw=1.6,
                        label=f"{setup} (n={len(vs)})")
                ax.fill_between(xs, ys, color=setup_colors[setup], alpha=0.10)
            except Exception:
                continue
        ax.set_title(f"model = {model}", fontsize=10, loc="left")
        ax.set_ylabel("density", fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85, ncol=1)

    axes[-1].set_xlabel(xlabel, fontsize=10)
    if xlim:
        axes[-1].set_xlim(*xlim)
    fig.tight_layout()
    return fig


def mean_heatmap(values_by_cell, models, setups, title):
    """2D heatmap: rows = model, cols = prompt. Cell value = mean of the
    run-level cooperation metric. Includes n-count and std as text overlay."""
    mat = np.full((len(models), len(setups)), np.nan)
    ann = np.empty((len(models), len(setups)), dtype=object)

    for i, m in enumerate(models):
        for j, s in enumerate(setups):
            vs = values_by_cell.get((s, m), [])
            if vs:
                mat[i, j] = float(np.mean(vs))
                std = float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0
                ann[i, j] = f"{mat[i, j]:.1f}\n±{std:.1f}\n(n={len(vs)})"
            else:
                ann[i, j] = "—"

    fig, ax = plt.subplots(figsize=(1.5 * len(setups) + 1, 1.2 * len(models) + 1))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(setups)))
    ax.set_xticklabels(setups, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(setups)):
            v = mat[i, j]
            color = "white" if not np.isnan(v) and (v < 25 or v > 75) else "black"
            ax.text(j, i, ann[i, j], ha="center", va="center",
                    fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("mean donation %", fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    return fig


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", nargs="+", default=DEFAULT_SETUPS)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--output_dir", default="figures/cooperation_distributions")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Collecting data ...")
    run_lv, gen_lv = collect(args.setups, args.models)

    # ── Report cell sizes ──
    print(f"\n{'setup':22s} " + " ".join(f"{m:>10s}" for m in args.models))
    for setup in args.setups:
        row = f"{setup:22s} " + " ".join(
            f"{len(run_lv.get((setup, m), [])):>4d} runs" for m in args.models)
        print(row)

    # ── Figure 1: RUN-LEVEL violin — one number per run ──
    fig = violin_grid(run_lv, args.models, args.setups,
                      ylabel="Mean donation % over the entire run\n(one value per run — ~20 runs per cell)",
                      ylim=(0, 100))
    fig.suptitle("Run-level cooperation distributions across (setup × model) cells",
                 fontsize=12, y=1.02)
    out = os.path.join(args.output_dir, "cooperation_run_level_violin.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # ── Figure 2: GEN-LEVEL violin — one number per (run, gen) ──
    all_gen = [v for vs in gen_lv.values() for v in vs]
    y_min = min(all_gen) - 5 if all_gen else 200
    y_max = max(all_gen) + 5 if all_gen else 400
    fig = violin_grid(gen_lv, args.models, args.setups,
                      ylabel="Per-generation avg total_final_score\n(one value per (run × gen) — ~200 samples per cell)",
                      ylim=(y_min, y_max))
    fig.suptitle("Generation-level cooperation distributions across (setup × model) cells",
                 fontsize=12, y=1.02)
    out = os.path.join(args.output_dir, "cooperation_gen_level_violin.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure 3: RUN-LEVEL KDE ridgeline ──
    fig = kde_lines(None, run_lv, args.models,
                    xlabel="Mean donation % over run",
                    xlim=(0, 100))
    if fig is not None:
        fig.suptitle("Run-level cooperation — density per setup, colored by model",
                     fontsize=11, y=1.005)
        out = os.path.join(args.output_dir, "cooperation_run_level_ridgeline.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # ── Figure 4: GEN-LEVEL KDE ridgeline ──
    fig = kde_lines(None, gen_lv, args.models,
                    xlabel="Per-generation avg total_final_score",
                    xlim=(y_min, y_max))
    if fig is not None:
        fig.suptitle("Generation-level cooperation — density per setup, colored by model",
                     fontsize=11, y=1.005)
        out = os.path.join(args.output_dir, "cooperation_gen_level_ridgeline.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # ── Figure 5: KDE curves faceted by MODEL, prompts as color ──
    fig = kde_lines_by_model(run_lv, args.models, args.setups,
                             xlabel="Mean donation % over run",
                             xlim=(0, 100))
    if fig is not None:
        fig.suptitle("Run-level cooperation — one panel per model, prompts overlaid",
                     fontsize=11, y=1.01)
        out = os.path.join(args.output_dir, "cooperation_run_level_by_model.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # ── Figure 6: heatmap — rows = model, cols = prompt, cell = mean cooperation ──
    fig = mean_heatmap(run_lv, args.models, args.setups,
                       title="Mean run-level donation % — model × prompt")
    out = os.path.join(args.output_dir, "cooperation_heatmap.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Numerical summary ──
    print("\n" + "=" * 78)
    print(f"{'setup':22s} {'model':7s} {'run n':>6s}  {'run donation mean±std':>26s}   {'gen score mean±std':>22s}")
    print("=" * 78)
    for setup in args.setups:
        for model in args.models:
            rv = run_lv.get((setup, model), [])
            gv = gen_lv.get((setup, model), [])
            if not rv:
                continue
            r_m, r_s = np.mean(rv), (np.std(rv, ddof=1) if len(rv) > 1 else 0)
            g_m, g_s = (np.mean(gv), np.std(gv, ddof=1) if len(gv) > 1 else 0) if gv else (0, 0)
            print(f"{setup:22s} {model:7s} {len(rv):>6d}  "
                  f"{r_m:>6.2f} ± {r_s:>5.2f} % donation      "
                  f"{g_m:>7.2f} ± {g_s:>5.2f}")


if __name__ == "__main__":
    main()
