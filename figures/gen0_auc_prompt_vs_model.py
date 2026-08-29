#!/usr/bin/env python3
"""
Generation-0 (initial state, t=0) self-cooperation rate across prompt variants
and executor models.

Self-cooperation rate at t=0 is measured as the AUC of each agent's coded
strategy (strategy_fn over pair_prev_pct in [0, 100], trapezoidal rule,
normalised to [0, 1]) — no games are simulated, only the initial strategies
that agents stated in generation 1 (the first generation, before any
evolution) are evaluated.

Reads figures/reactivity_plots/{variant}/{model}/strategy_analysis_{model}.json
(the coded strategies were produced once by figures/strategy_analysis.py and
cover every run in simulations/{variant}/{model}/).

Outputs (to --output_dir, default figures/gen0_prompt_vs_model/):
  gen0_auc_stats.json                  per-cell stats + two-way ANOVA
  gen0_auc_by_prompt.png               x = prompt variant, bars = model
  gen0_auc_by_model.png                x = model, bars = prompt variant
  gen0_auc_variance_decomposition.png  variance explained: prompt vs model

Usage:
    python figures/gen0_auc_prompt_vs_model.py \
        [--analysis_dir figures/reactivity_plots] \
        [--output_dir figures/gen0_prompt_vs_model]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import eval_strategy, auc_normalised

GEN0 = 1   # generation numbering in the data starts at 1; gen 1 = initial state

# "<variant>_200" = July-era strategies-only reruns (~200 runs/model/variant),
# the default grid. The May-era full-game variants (plain names) remain
# available by editing this list back.
VARIANTS = [
    "neutral_baseline_200", "neutral_claude_200", "neutral_gpt_200",
    "neutral_gemini_200", "neutral_deepseek_200", "collaborative_200",
    "aggressive_200",
]
_BASE_LABELS = {
    "neutral_baseline": "Neutral\n(baseline)",
    "gen0_strategies":  "Neutral\n(baseline)",
    "neutral_claude":   "Neutral\n(Claude-phrased)",
    "neutral_gpt":      "Neutral\n(GPT-phrased)",
    "neutral_gemini":   "Neutral\n(Gemini-phrased)",
    "neutral_deepseek": "Neutral\n(DeepSeek-phrased)",
    "collaborative":    "Collaborative",
    "aggressive":       "Aggressive",
}
# Categorical palette, fixed slot order (validated for CVD separation)
_BASE_COLORS = {
    "neutral_baseline": "#2a78d6",
    "gen0_strategies":  "#2a78d6",   # same slot: it is the baseline condition
    "neutral_claude":   "#1baf7a",
    "neutral_gpt":      "#eda100",
    "neutral_gemini":   "#008300",
    "neutral_deepseek": "#4a3aa7",
    "collaborative":    "#e34948",
    "aggressive":       "#e87ba4",
}
# _200 keys inherit the base variant's label and color
VARIANT_LABELS = dict(_BASE_LABELS)
VARIANT_COLORS = dict(_BASE_COLORS)
for _k in list(_BASE_LABELS):
    VARIANT_LABELS[f"{_k}_200"] = _BASE_LABELS[_k]
    VARIANT_COLORS[f"{_k}_200"] = _BASE_COLORS[_k]

MODELS = ["claude", "gpt", "gemini"]
# Display labels for plots; MODEL_STRINGS records the run-time model ids
# (gemini-1.5-flash is the MODEL_STR in the data files; the API call behind it
# is gemini-2.5-flash-lite, hence the display label).
MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o", "gemini": "Gemini 2.5 Flash"}
MODEL_STRINGS = {"claude": "claude-3-5-sonnet", "gpt": "gpt-4o", "gemini": "gemini-1.5-flash"}
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}

# Chart chrome (light mode)
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def collect_gen0_aucs(analysis_dir):
    """Return {variant: {model: {'agent_aucs': [...], 'run_aucs': {run: [...]},
    coverage counters}}} for generation-1 (t=0) records."""
    cells = {}
    for variant in VARIANTS:
        for model in MODELS:
            path = os.path.join(analysis_dir, variant, model,
                                f"strategy_analysis_{model}.json")
            if not os.path.exists(path):
                print(f"  [{variant}/{model}] not found: {path} — skipping")
                continue
            with open(path) as f:
                records = json.load(f).get("agent_records", [])

            gen0 = [r for r in records if r.get("generation") == GEN0]
            agent_aucs, run_aucs = [], {}
            n_unspecified = n_eval_failed = 0
            for rec in gen0:
                code = rec.get("strategy_code")
                if not rec.get("is_specified", False) or not code:
                    n_unspecified += 1
                    continue
                ys = eval_strategy(code)
                if ys is None:
                    n_eval_failed += 1
                    continue
                auc = auc_normalised(ys)
                agent_aucs.append(auc)
                run_aucs.setdefault(rec["run"], []).append(auc)

            cells.setdefault(variant, {})[model] = {
                "agent_aucs": agent_aucs,
                "run_aucs": run_aucs,
                "n_gen0_agents": len(gen0),
                "n_unspecified": n_unspecified,
                "n_eval_failed": n_eval_failed,
            }
    return cells


def cell_stats(cell):
    agent = np.array(cell["agent_aucs"])
    run_means = np.array([np.mean(v) for v in cell["run_aucs"].values()])
    n_runs = len(run_means)
    return {
        "n_agents_coded": int(agent.size),
        "n_gen0_agents": cell["n_gen0_agents"],
        "n_unspecified": cell["n_unspecified"],
        "n_eval_failed": cell["n_eval_failed"],
        "coded_fraction": float(agent.size / cell["n_gen0_agents"]) if cell["n_gen0_agents"] else None,
        "n_runs": n_runs,
        "mean": float(np.mean(agent)) if agent.size else None,
        "std": float(np.std(agent)) if agent.size else None,
        "run_mean": float(np.mean(run_means)) if n_runs else None,
        "run_std": float(np.std(run_means, ddof=1)) if n_runs > 1 else None,
        "run_sem": float(np.std(run_means, ddof=1) / np.sqrt(n_runs)) if n_runs > 1 else None,
        "run_means": [float(m) for m in run_means],
    }


def two_way_anova(stats):
    """Two-way ANOVA (type II) on run-level mean AUC: prompt, model, interaction.

    Run-level means are the observation unit so unequal per-run agent counts
    don't overweight any run.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # A variant with an empty (variant, model) cell makes the interaction
    # design matrix singular and the type II SS meaningless — restrict the
    # ANOVA to variants observed for every model.
    complete = [v for v in VARIANTS if all(m in stats.get(v, {}) for m in MODELS)]
    dropped = [v for v in VARIANTS if v in stats and v not in complete]
    if dropped:
        print(f"  [anova] dropped variants with missing cells: {', '.join(dropped)}")

    rows = []
    for variant in complete:
        for model in MODELS:
            s = stats[variant][model]
            for m in s["run_means"]:
                rows.append({"variant": variant, "model": model, "auc": m})
    df = pd.DataFrame(rows)

    fit = ols("auc ~ C(variant) * C(model)", data=df).fit()
    table = sm.stats.anova_lm(fit, typ=2)
    ss_total = table["sum_sq"].sum()
    out = {}
    name_map = {"C(variant)": "prompt", "C(model)": "model",
                "C(variant):C(model)": "prompt_x_model", "Residual": "residual"}
    for raw, name in name_map.items():
        row = table.loc[raw]
        out[name] = {
            "sum_sq": float(row["sum_sq"]),
            "df": float(row["df"]),
            "eta_sq": float(row["sum_sq"] / ss_total),
            "F": None if np.isnan(row["F"]) else float(row["F"]),
            "p": None if np.isnan(row["PR(>F)"]) else float(row["PR(>F)"]),
        }
    out["n_obs_runs"] = int(len(df))
    out["variants_included"] = complete
    out["variants_dropped_missing_cells"] = dropped
    return out


def grouped_bars(ax, group_keys, series_keys, stats, get_cell, colors,
                 bar_alpha=0.9, errorbar="sd", show_dots=True):
    """Grouped bar chart. Error bars across runs: errorbar="sd" -> ±1 SD,
    errorbar="ci" -> 95% CI (t-based). show_dots overlays per-run means."""
    n_series = len(series_keys)
    group_w = 0.78
    bar_w = group_w / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * bar_w

    rng = np.random.default_rng(0)
    for s_idx, skey in enumerate(series_keys):
        xs, means, stds = [], [], []
        for g_idx, gkey in enumerate(group_keys):
            s = get_cell(stats, gkey, skey)
            if not s or s["run_mean"] is None:
                continue
            x = g_idx + offsets[s_idx]
            xs.append(x)
            means.append(s["run_mean"])
            if errorbar == "ci" and s["run_sem"] and s["n_runs"] > 1:
                stds.append(sps.t.ppf(0.975, s["n_runs"] - 1) * s["run_sem"])
            else:
                stds.append(s["run_std"] or 0.0)
            if show_dots:
                jitter = rng.uniform(-bar_w * 0.28, bar_w * 0.28,
                                     size=len(s["run_means"]))
                ax.scatter(x + jitter, s["run_means"], s=9, color=INK_2,
                           alpha=0.45, linewidths=0, zorder=3)
            if s["n_runs"] < 10:
                ax.annotate(f"n={s['n_runs']}", (x, 0.012), ha="center",
                            fontsize=6.5, color=INK_2, zorder=4)
        ax.bar(xs, means, width=bar_w * 0.92, color=colors[skey],
               alpha=bar_alpha, edgecolor=SURFACE, linewidth=0.8, zorder=2)
        ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=INK_2,
                    elinewidth=1, capsize=2.5, zorder=4)

    ax.set_xticks(np.arange(len(group_keys)))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Self-cooperation rate at t = 0\n(normalised AUC of coded strategy)",
                  fontsize=9.5, color=INK_2)


def plot_by_prompt(stats, out_path):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    style_axes(ax)
    grouped_bars(ax, VARIANTS, MODELS, stats,
                 lambda st, g, s: st.get(g, {}).get(s), MODEL_COLORS)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=8.5)
    ax.set_title("Initial (generation-0) cooperation by prompt variant",
                 fontsize=12, color=INK, pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in MODELS]
    ax.legend(handles, [MODEL_LABELS[m] for m in MODELS], fontsize=8.5,
              frameon=False, loc="upper left", labelcolor=INK_2)
    fig.text(0.99, 0.01,
             "Bars: mean of per-run means · error bars: ±1 SD across runs · dots: individual runs",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {out_path}")


def plot_by_model(stats, out_path):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    style_axes(ax)
    grouped_bars(ax, MODELS, VARIANTS, stats,
                 lambda st, g, s: st.get(s, {}).get(g), VARIANT_COLORS,
                 errorbar="ci", show_dots=False)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_title("Initial (generation-0) cooperation by model",
                 fontsize=12, color=INK, pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLORS[v]) for v in VARIANTS]
    ax.legend(handles, [VARIANT_LABELS[v].replace("\n", " ") for v in VARIANTS],
              fontsize=8, frameon=False, ncol=2, loc="upper left", labelcolor=INK_2)
    fig.text(0.99, 0.01,
             "Bars: mean of per-run means · error bars: 95% CI across runs (n ≈ 200 runs/cell)",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {out_path}")


def plot_variance_decomposition(anova, out_path):
    comps = ["prompt", "model", "prompt_x_model", "residual"]
    labels = {"prompt": "Prompt variant", "model": "Model",
              "prompt_x_model": "Prompt × model", "residual": "Residual (between runs)"}
    vals = [anova[c]["eta_sq"] for c in comps]

    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ys = np.arange(len(comps))[::-1]
    ax.barh(ys, vals, height=0.55, color="#2a78d6", alpha=0.9, zorder=2)
    for y, v, c in zip(ys, vals, comps):
        p = anova[c]["p"]
        p_txt = "" if p is None else ("  (p < 0.001)" if p < 0.001 else f"  (p = {p:.3f})")
        ax.text(v + 0.012, y, f"{v * 100:.1f}%{p_txt}", va="center",
                fontsize=9, color=INK_2)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[c] for c in comps], fontsize=10, color=INK_2)
    ax.set_xlim(0, max(vals) * 1.35)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("Share of variance in run-level gen-0 AUC (η²)", fontsize=9.5, color=INK_2)
    ax.set_title("How much do prompt and model choice matter at t = 0?",
                 fontsize=12, color=INK, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gen-0 coded-strategy AUC: prompt variant vs model.")
    parser.add_argument("--analysis_dir", default="figures/reactivity_plots")
    parser.add_argument("--output_dir", default="figures/gen0_prompt_vs_model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cells = collect_gen0_aucs(args.analysis_dir)

    stats = {}
    print(f"\n{'variant':<18} {'model':<8} {'runs':>4} {'coded':>5} {'/gen0':>5} "
          f"{'mean':>6} {'run-SEM':>7}")
    for variant in VARIANTS:
        for model in MODELS:
            cell = cells.get(variant, {}).get(model)
            if not cell:
                continue
            s = cell_stats(cell)
            if s["run_mean"] is None:   # cell exists but has no coded strategies
                print(f"{variant:<18} {model:<8}   no coded strategies — skipped")
                continue
            stats.setdefault(variant, {})[model] = s
            sem = f"{s['run_sem']:.4f}" if s["run_sem"] is not None else "  n/a"
            print(f"{variant:<18} {model:<8} {s['n_runs']:>4} {s['n_agents_coded']:>5} "
                  f"{s['n_gen0_agents']:>5} {s['run_mean']:>6.3f} {sem:>7}")

    anova = two_way_anova(stats)
    print("\nTwo-way ANOVA on run-level gen-0 AUC (type II):")
    for comp in ["prompt", "model", "prompt_x_model", "residual"]:
        a = anova[comp]
        p_txt = "" if a["p"] is None else f"  p={a['p']:.2e}"
        print(f"  {comp:<16} eta^2={a['eta_sq']:.3f}{p_txt}")

    out_json = os.path.join(args.output_dir, "gen0_auc_stats.json")
    with open(out_json, "w") as f:
        json.dump({"generation": GEN0, "note": "generation 1 in the data = initial state (t=0)",
                   "model_strings": MODEL_STRINGS,
                   "cells": stats, "anova_run_level": anova}, f, indent=2)
    print(f"\nSaved: {out_json}")

    plot_by_prompt(stats, os.path.join(args.output_dir, "gen0_auc_by_prompt.png"))
    plot_by_model(stats, os.path.join(args.output_dir, "gen0_auc_by_model.png"))
    plot_variance_decomposition(anova, os.path.join(args.output_dir,
                                                    "gen0_auc_variance_decomposition.png"))


if __name__ == "__main__":
    main()
