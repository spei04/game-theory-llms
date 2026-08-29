#!/usr/bin/env python3
"""
Generation-0 self-cooperation rate for the direct-evolution runs.

Gen-0 self-cooperation = AUC of each agent's coded generation-1 strategy
(strategy_fn evaluated over pair_prev_pct in [0, 100], trapezoidal rule,
normalised to [0, 1]) — the strategies agents stated before any gameplay or
evolution. Identical definition to gen0_auc_prompt_vs_model.py, so these
numbers are directly comparable to the prompt-variant grid.

Reads figures/reactivity_plots/direct_evo_new/{model}/strategy_analysis_{model}.json
(produced by figures/strategy_analysis.py on simulations/direct_evo_new/{model}).

Left panel : per-model bars (mean of run means ± 1 SD across runs) with every
             run as a dot.
Right panel: distribution of agent-level gen-0 AUC per model.

Usage:
    python figures/plot_gen0_direct_evo.py \
        [--analysis_dir figures/reactivity_plots/direct_evo_new] \
        [--output figures/gen0_direct_evo.png]
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import eval_strategy, auc_normalised

GEN0 = 1   # generation 1 in the data = initial state, t=0

MODELS = ["claude", "gpt", "gemini"]
MODEL_BASE = {"claude": "Claude", "gpt": "GPT", "gemini": "Gemini"}
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}


def collect(analysis_dir, model):
    """Return (agent_aucs, {run: [aucs]}, coverage dict) for generation-0 records."""
    path = os.path.join(analysis_dir, model, f"strategy_analysis_{model}.json")
    if not os.path.exists(path):
        print(f"  [{model}] not found: {path} — skipping")
        return np.array([]), {}, {}
    with open(path) as f:
        records = json.load(f).get("agent_records", [])

    gen0 = [r for r in records if r.get("generation") == GEN0]
    aucs, run_aucs = [], {}
    n_unspec = n_failed = 0
    for rec in gen0:
        code = rec.get("strategy_code")
        if not rec.get("is_specified", False) or not code:
            n_unspec += 1
            continue
        ys = eval_strategy(code)
        if ys is None:
            n_failed += 1
            continue
        a = auc_normalised(ys)
        aucs.append(a)
        run_aucs.setdefault(rec["run"], []).append(a)
    cov = {"n_gen0_agents": len(gen0), "n_coded": len(aucs),
           "n_unspecified": n_unspec, "n_eval_failed": n_failed,
           "n_runs": len(run_aucs)}
    return np.array(aucs), run_aucs, cov


def main():
    parser = argparse.ArgumentParser(
        description="Plot gen-0 self-cooperation rate for direct-evolution runs.")
    parser.add_argument("--analysis_dir", default="figures/reactivity_plots/direct_evo_new")
    parser.add_argument("--output", default="figures/gen0_direct_evo.png")
    parser.add_argument("--model_suffix", default="(new)",
                        help="Appended to model labels, e.g. '(new)' or '3.5 Sonnet'. "
                             "Pass '' for bare model names.")
    parser.add_argument("--title", default="direct-evolution runs",
                        help="Dataset description used in the figure title")
    args = parser.parse_args()

    labels = {m: f"{MODEL_BASE[m]} {args.model_suffix}".strip() for m in MODELS}
    data, stats = {}, {}
    for model in MODELS:
        aucs, run_aucs, cov = collect(args.analysis_dir, model)
        run_means = np.array([np.mean(v) for v in run_aucs.values()])
        data[model] = (aucs, run_means)
        if aucs.size:
            stats[model] = {
                **cov,
                "agent_mean": float(aucs.mean()), "agent_std": float(aucs.std()),
                "run_mean": float(run_means.mean()),
                "run_std": float(run_means.std(ddof=1)) if run_means.size > 1 else None,
                "run_means": [float(m) for m in run_means],
            }
            print(f"  [{model}] {cov['n_coded']}/{cov['n_gen0_agents']} coded across "
                  f"{cov['n_runs']} runs — gen-0 AUC {aucs.mean():.3f} ± {aucs.std():.3f}")

    if not any(d[0].size for d in data.values()):
        print("\nNo coded gen-0 strategies found. Run figures/strategy_analysis.py first:\n"
              "  python figures/strategy_analysis.py simulations/direct_evo_new/<model> "
              "--model <model> --output_dir figures/reactivity_plots/direct_evo_new")
        return

    fig, (ax_bar, ax_dist) = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(0)

    # ── Left: run-level bars with per-run dots ───────────────────────────────
    xs = np.arange(len(MODELS))
    for x, model in zip(xs, MODELS):
        aucs, run_means = data[model]
        if not aucs.size:
            continue
        sd = run_means.std(ddof=1) if run_means.size > 1 else 0.0
        ax_bar.bar(x, run_means.mean(), 0.6, yerr=sd, capsize=4,
                   color=MODEL_COLORS[model], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
        ax_bar.scatter(x + rng.uniform(-0.16, 0.16, run_means.size), run_means,
                       s=14, color="#333", alpha=0.55, linewidths=0, zorder=3)
        ax_bar.text(x, 0.012, f"n={run_means.size} runs", ha="center",
                    fontsize=8, color="#333")
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([labels[m] for m in MODELS], fontsize=10)
    ax_bar.set_ylabel("Gen-0 self-cooperation rate\n(mean AUC of coded strategy, 0–1)",
                      fontsize=10)
    ax_bar.set_title("Run-level means (dots = runs, bar = mean ± 1 SD)", fontsize=11)
    ax_bar.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax_bar.spines[side].set_visible(False)

    # ── Right: agent-level distributions ─────────────────────────────────────
    present = [m for m in MODELS if data[m][0].size]
    parts = ax_dist.violinplot([data[m][0] for m in present],
                               positions=range(len(present)), widths=0.75,
                               showextrema=False, showmedians=True)
    for body, model in zip(parts["bodies"], present):
        body.set_facecolor(MODEL_COLORS[model])
        body.set_alpha(0.45)
    parts["cmedians"].set_color("#222")
    for i, model in enumerate(present):
        a = data[model][0]
        ax_dist.scatter(i + rng.uniform(-0.09, 0.09, a.size), a, s=5,
                        color=MODEL_COLORS[model], alpha=0.3, linewidths=0, zorder=3)
    ax_dist.set_xticks(range(len(present)))
    ax_dist.set_xticklabels([labels[m] for m in present], fontsize=10)
    ax_dist.set_ylabel("Gen-0 AUC per agent (0–1)", fontsize=10)
    ax_dist.set_title("Agent-level distribution (each dot = one agent)", fontsize=11)
    ax_dist.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax_dist.spines[side].set_visible(False)

    fig.suptitle(f"Generation-0 self-cooperation rate — {args.title} "
                 "(initial strategies, before evolution)", fontsize=12.5)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {args.output}")

    stats_path = os.path.splitext(args.output)[0] + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
