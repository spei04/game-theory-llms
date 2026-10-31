#!/usr/bin/env python3
"""
Compare generation-0 (initial, t=0) coded strategies between two prompt
variants, per model.

For every gen-1 record (the initial strategies, before any evolution) with a
fully specified coded strategy, evaluates strategy_fn over pair_prev_pct in
[0, 100] plus the no-information first round (strategy_fn(None)).

Outputs (to --output_dir, default figures/gen0_prompt_comparison/):
  gen0_{a}_vs_{b}_reactivity.png   mean ±1 SD curves, one panel per model
  gen0_{a}_vs_{b}_auc.png          AUC bars (mean ±1 SD across runs) + run dots
  gen0_{a}_vs_{b}_stats.json       per-cell stats + significance tests

Usage:
    python figures/compare_gen0_variants.py \
        [--variant_a neutral_baseline] [--variant_b neutral_deepseek] \
        [--analysis_dir figures/reactivity_plots] \
        [--output_dir figures/gen0_prompt_comparison]
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import (eval_strategy, auc_normalised, X_VALUES,
                         _sanitize, _fix_empty_blocks)

GEN0 = 1   # generation numbering in the data starts at 1; gen 1 = initial state

MODELS = ["claude", "gpt", "gemini"]
# Display labels for plots; MODEL_STRINGS records the run-time model ids.
MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o", "gemini": "Gemini 2.5 Flash"}
MODEL_STRINGS = {"claude": "claude-3-5-sonnet", "gpt": "gpt-4o", "gemini": "gemini-2.5-flash"}

VARIANT_LABELS = {
    "neutral_baseline": "Neutral (baseline)",
    "gen0_strategies":  "Neutral (baseline)",   # strategies-only reruns, ~200 runs/model
    "neutral_claude":   "Neutral (Claude-phrased)",
    "neutral_gpt":      "Neutral (GPT-phrased)",
    "neutral_gemini":   "Neutral (Gemini-phrased)",
    "neutral_deepseek": "Neutral (DeepSeek-phrased)",
    "collaborative":    "Collaborative",
    "aggressive":       "Aggressive",
}
VARIANT_COLORS = {
    "neutral_baseline": "#2a78d6",
    "gen0_strategies":  "#2a78d6",   # same slot: it is the baseline condition
    "neutral_claude":   "#1baf7a",
    "neutral_gpt":      "#eda100",
    "neutral_gemini":   "#008300",
    "neutral_deepseek": "#4a3aa7",
    "collaborative":    "#e34948",
    "aggressive":       "#e87ba4",
}

# Variant pairs whose suite colors are too close to overlay (e.g. red vs pink):
# variant_b falls back to neutral gray.
CLASH_PAIRS = {frozenset(("collaborative", "aggressive"))}
FALLBACK_B = "#666666"


def pair_colors(va, vb):
    cb = FALLBACK_B if frozenset((va, vb)) in CLASH_PAIRS else VARIANT_COLORS[vb]
    return VARIANT_COLORS[va], cb

# "<variant>_200" = July-era strategies-only reruns (~200 runs/model/variant)
VARIANT_LABELS.update({
    "neutral_baseline_200": "Neutral (baseline)",
    "neutral_claude_200":   "Neutral (Claude-phrased)",
    "neutral_gpt_200":      "Neutral (GPT-phrased)",
    "neutral_gemini_200":   "Neutral (Gemini-phrased)",
    "neutral_deepseek_200": "Neutral (DeepSeek-phrased)",
    "collaborative_200":    "Collaborative",
    "aggressive_200":       "Aggressive",
})
VARIANT_COLORS.update({
    "neutral_baseline_200": "#2a78d6",
    "neutral_claude_200":   "#1baf7a",
    "neutral_gpt_200":      "#eda100",
    "neutral_gemini_200":   "#008300",
    "neutral_deepseek_200": "#4a3aa7",
    "collaborative_200":    "#e34948",
    "aggressive_200":       "#e87ba4",
})

# Chart chrome (light mode)
SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def eval_first_round(code):
    """Return strategy_fn(None) clipped to [0, 100], or None if not evaluable."""
    ns = {}
    sanitized = _sanitize(code)
    for attempt in [sanitized, _fix_empty_blocks(sanitized)]:
        try:
            exec(attempt, ns)
            break
        except SyntaxError:
            continue
        except Exception:
            return None
    fn = ns.get("strategy_fn")
    if fn is None:
        return None
    try:
        return float(np.clip(fn(None), 0, 100))
    except Exception:
        return None


def load_variant(analysis_dir, variant, model, generation=GEN0):
    """Evaluate a generation's coded strategies: curves, agent AUCs, per-run
    AUCs, first-round donations. Defaults to gen 1 = the initial state (t=0)."""
    path = os.path.join(analysis_dir, variant, model,
                        f"strategy_analysis_{model}.json")
    if not os.path.exists(path):
        print(f"  [{variant}/{model}] not found: {path}")
        return None
    with open(path) as f:
        records = json.load(f).get("agent_records", [])

    curves, aucs, first_round, run_aucs = [], [], [], {}
    for rec in records:
        if rec.get("generation") != generation:
            continue
        code = rec.get("strategy_code")
        if not rec.get("is_specified", False) or not code:
            continue
        ys = eval_strategy(code)
        if ys is None:
            continue
        curves.append(ys)
        auc = auc_normalised(ys)
        aucs.append(auc)
        run_aucs.setdefault(rec["run"], []).append(auc)
        fr = eval_first_round(code)
        if fr is not None:
            first_round.append(fr)

    return {
        "curves": np.array(curves),
        "aucs": np.array(aucs),
        "run_means": np.array([np.mean(v) for v in run_aucs.values()]),
        "first_round": np.array(first_round),
    }


def cell_summary(d):
    rm = d["run_means"]
    return {
        "n_agents": int(d["aucs"].size),
        "n_runs": int(rm.size),
        "auc_mean": float(d["aucs"].mean()),
        "auc_std": float(d["aucs"].std()),
        "auc_run_mean": float(rm.mean()),
        "auc_run_std": float(rm.std(ddof=1)) if rm.size > 1 else None,
        "auc_run_means": [float(m) for m in rm],
        "first_round_mean_pct": float(d["first_round"].mean()) if d["first_round"].size else None,
        "first_round_std_pct": float(d["first_round"].std()) if d["first_round"].size else None,
        "n_first_round": int(d["first_round"].size),
    }


def compare_tests(da, db):
    """Agent-level Mann-Whitney + run-level Welch t on AUC; MW on first round."""
    out = {}
    u, p = sps.mannwhitneyu(da["aucs"], db["aucs"], alternative="two-sided")
    out["auc_mannwhitney_agents"] = {"U": float(u), "p": float(p)}
    if da["run_means"].size > 1 and db["run_means"].size > 1:
        t, p = sps.ttest_ind(da["run_means"], db["run_means"], equal_var=False)
        out["auc_welch_t_runs"] = {"t": float(t), "p": float(p)}
    if da["first_round"].size and db["first_round"].size:
        u, p = sps.mannwhitneyu(da["first_round"], db["first_round"],
                                alternative="two-sided")
        out["first_round_mannwhitney_agents"] = {"U": float(u), "p": float(p)}
    return out


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_reactivity(data, va, vb, out_path):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.8),
                             sharey=True, facecolor=SURFACE)
    ca, cb = pair_colors(va, vb)
    for ax, model in zip(axes, MODELS):
        style_axes(ax, grid_axis="both")
        for variant, color, ls in ((va, ca, "-"), (vb, cb, "--")):
            d = data[model].get(variant)
            if d is None or not d["curves"].size:
                continue
            mean_y = d["curves"].mean(axis=0)
            std_y = d["curves"].std(axis=0)
            ax.plot(X_VALUES, mean_y, color=color, linewidth=2.5, linestyle=ls,
                    label=f"{VARIANT_LABELS[variant]} (n={len(d['curves'])})")
            ax.fill_between(X_VALUES,
                            np.clip(mean_y - std_y, 0, 100),
                            np.clip(mean_y + std_y, 0, 100),
                            alpha=0.15, color=color, linewidth=0)
        ax.set_title(MODEL_LABELS[model], fontsize=11.5, color=INK)
        ax.set_xlabel("Pair's donation % in previous round", fontsize=9.5, color=INK_2)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=8.5, loc="upper left", frameon=False, labelcolor=INK_2)
    axes[0].set_ylabel("My donation %", fontsize=9.5, color=INK_2)
    fig.suptitle(f"Generation-0 coded strategies: {VARIANT_LABELS[va]} vs "
                 f"{VARIANT_LABELS[vb]}", fontsize=13, color=INK)
    fig.text(0.99, 0.01, "Line: mean across agents · band: ±1 SD across agents",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {out_path}")


def plot_auc(data, tests, va, vb, out_path):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    style_axes(ax)
    bar_w = 0.35
    xs = np.arange(len(MODELS))
    rng = np.random.default_rng(0)
    ca, cb = pair_colors(va, vb)

    for i, (variant, color) in enumerate(((va, ca), (vb, cb))):
        offset = (i - 0.5) * bar_w
        for x0, model in zip(xs, MODELS):
            d = data[model].get(variant)
            if d is None or not d["run_means"].size:
                continue
            x = x0 + offset
            mean = d["run_means"].mean()
            std = d["run_means"].std(ddof=1) if d["run_means"].size > 1 else 0.0
            ax.bar(x, mean, bar_w * 0.9, color=color,
                   alpha=0.9, edgecolor=SURFACE, linewidth=0.8, zorder=2,
                   label=VARIANT_LABELS[variant] if x0 == 0 else None)
            ax.errorbar(x, mean, yerr=std, fmt="none", ecolor=INK_2,
                        elinewidth=1, capsize=2.5, zorder=4)
            jitter = rng.uniform(-bar_w * 0.26, bar_w * 0.26,
                                 size=d["run_means"].size)
            ax.scatter(x + jitter, d["run_means"], s=9, color=INK_2,
                       alpha=0.45, linewidths=0, zorder=3)
            if d["run_means"].size < 10:
                ax.annotate(f"n={d['run_means'].size}", (x, 0.012), ha="center",
                            fontsize=6.5, color=INK_2, zorder=4)

    # p-value annotation (run-level Welch t) above each model pair
    for x0, model in zip(xs, MODELS):
        t = tests.get(model, {}).get("auc_welch_t_runs")
        da, db = data[model].get(va), data[model].get(vb)
        if t is None or da is None or db is None:
            continue
        top = max(
            da["run_means"].mean() + (da["run_means"].std(ddof=1) if da["run_means"].size > 1 else 0),
            db["run_means"].mean() + (db["run_means"].std(ddof=1) if db["run_means"].size > 1 else 0))
        y = top + 0.05
        ax.plot([x0 - bar_w / 2, x0 + bar_w / 2], [y, y],
                color=INK_2, linewidth=0.9)
        ax.text(x0, y + 0.012, fmt_p(t["p"]), ha="center",
                fontsize=8.5, color=INK_2)

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Self-cooperation rate at t = 0\n(normalised AUC of coded strategy)",
                  fontsize=9.5, color=INK_2)
    ax.set_title(f"Generation-0 AUC: {VARIANT_LABELS[va]} vs {VARIANT_LABELS[vb]}",
                 fontsize=12, color=INK, pad=12)
    ax.legend(fontsize=9, frameon=False, loc="upper left", labelcolor=INK_2)
    fig.text(0.99, 0.01,
             "Bars: mean of per-run means · error bars: ±1 SD across runs · "
             "dots: individual runs · p: Welch t on run means",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare gen-0 coded strategies between two prompt variants.")
    parser.add_argument("--variant_a", default="neutral_baseline")
    parser.add_argument("--variant_b", default="neutral_deepseek")
    parser.add_argument("--analysis_dir", default="figures/reactivity_plots")
    parser.add_argument("--output_dir", default="figures/gen0_prompt_comparison")
    args = parser.parse_args()

    va, vb = args.variant_a, args.variant_b
    data, summary, tests = {}, {}, {}

    print(f"\n{'model':<8} {'variant':<18} {'runs':>4} {'agents':>6} "
          f"{'AUC mean':>8} {'±SD':>6} {'1st-round %':>11}")
    for model in MODELS:
        data[model], summary[model] = {}, {}
        for variant in (va, vb):
            d = load_variant(args.analysis_dir, variant, model)
            if d is None or not d["aucs"].size:
                continue
            data[model][variant] = d
            s = cell_summary(d)
            summary[model][variant] = s
            fr = (f"{s['first_round_mean_pct']:.1f}"
                  if s["first_round_mean_pct"] is not None else "n/a")
            print(f"{model:<8} {variant:<18} {s['n_runs']:>4} {s['n_agents']:>6} "
                  f"{s['auc_run_mean']:>8.3f} {s['auc_run_std']:>6.3f} {fr:>11}")
        if va in data[model] and vb in data[model]:
            tests[model] = compare_tests(data[model][va], data[model][vb])
            t = tests[model]
            print(f"         -> AUC: runs Welch {fmt_p(t['auc_welch_t_runs']['p'])}, "
                  f"agents MW {fmt_p(t['auc_mannwhitney_agents']['p'])}"
                  + (f"; 1st round MW {fmt_p(t['first_round_mannwhitney_agents']['p'])}"
                     if "first_round_mannwhitney_agents" in t else ""))

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"gen0_{va}_vs_{vb}"

    plot_reactivity(data, va, vb, os.path.join(args.output_dir, f"{tag}_reactivity.png"))
    plot_auc(data, tests, va, vb, os.path.join(args.output_dir, f"{tag}_auc.png"))

    out_json = os.path.join(args.output_dir, f"{tag}_stats.json")
    with open(out_json, "w") as f:
        json.dump({"generation": GEN0,
                   "note": "generation 1 in the data = initial state (t=0)",
                   "model_strings": MODEL_STRINGS,
                   "variants": [va, vb], "summary": summary, "tests": tests},
                  f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
