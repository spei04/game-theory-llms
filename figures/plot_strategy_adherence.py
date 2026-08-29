#!/usr/bin/env python3
"""
Compare strategy adherence metrics across models for N simulation batches.

Reads summary data from strategy_analysis_{model}.json files (produced by
figures/strategy_analysis.py) and produces a 4-panel bar chart with one
group of bars per metric.

Metrics:
  1. % Unspecified strategies
  2. Mean cooperation % (all agents)
  3. Adherence gap (predicted − actual, pp; fully specified only)
  4. Mean AUC of coded strategies (fully specified only)

Usage:
    python figures/plot_strategy_adherence.py \\
        --analysis_dirs figures/reactivity_plots/neutral_baseline \\
                        figures/reactivity_plots/neutral_deepseek \\
                        figures/reactivity_plots/neutral_claude \\
                        figures/reactivity_plots/neutral_gpt \\
                        figures/reactivity_plots/neutral_gemini \\
        --output_dir figures/strategy_groups/all_variants
"""

import argparse
import glob
import json
import os
import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


MODEL_LABELS = {
    "claude": "Claude 3.5 Sonnet",
    "gpt":    "GPT-4o",
    "gemini": "Gemini 2.5 Flash",
}

MODEL_COLORS = {
    "claude": "#FF6B35",
    "gpt":    "#10A37F",
    "gemini": "#4285F4",
}

VARIANT_HATCHES = ["", "///", "...", "xxx", "++", "\\\\\\", "ooo"]
VARIANT_ALPHAS  = [0.90, 0.65, 0.50, 0.85, 0.55, 0.75, 0.45]


# ── AUC computation ─────────────────────────────────────────────────────────

_X_GRID = np.linspace(0, 100, 500)   # pair_prev_pct grid for integration
_UNICODE_OPS = [('≥','>='), ('≤','<='), ('≠','!='), ('×','*'), ('÷','/'), ('−','-')]

def _sanitize_code(code: str) -> str:
    for uni, asc in _UNICODE_OPS:
        code = code.replace(uni, asc)
    return code

def _fix_empty_blocks(code: str) -> str:
    lines = code.splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        head = line.split('#')[0].rstrip()
        if not head.endswith(':'):
            continue
        indent = len(line) - len(line.lstrip())
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith('#')):
            j += 1
        if j >= len(lines) or (len(lines[j]) - len(lines[j].lstrip())) <= indent:
            out.append(' ' * (indent + 4) + 'pass\n')
    return ''.join(out)

def _compile_strategy_fn(code: str):
    if not code:
        return None
    sanitized = _sanitize_code(code)
    ns = {}
    for src in (sanitized, _fix_empty_blocks(sanitized)):
        try:
            exec(src, ns)
            fn = ns.get('strategy_fn')
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

def _strategy_auc(code: str):
    """Normalised AUC ∈ [0, 1] of strategy_fn over pair_prev_pct ∈ [0, 100].
    1.0 = always donate 100%, 0.0 = always 0%, 0.5 = always 50%."""
    fn = _compile_strategy_fn(code)
    if fn is None:
        return None
    try:
        ys = np.array([float(np.clip(fn(x), 0, 100)) for x in _X_GRID])
        raw = np.trapezoid(ys, _X_GRID)
        return raw / (100.0 * 100.0)
    except Exception:
        return None


# ── Data loading ────────────────────────────────────────────────────────────

def load_analysis(analysis_dir: str) -> dict:
    """Returns {model_short: {summary_keys..., 'mean_auc_gen1', 'mean_auc_genN',
    and *_std variants}} for each model whose strategy_analysis JSON exists
    in analysis_dir.

    Standard deviations are computed across simulation RUNS (each run gets one
    per-run value of the metric, then we report the population mean ± sample
    std across runs). This is what we plot as error bars."""
    out = {}
    for model in MODEL_LABELS:
        files = sorted(glob.glob(
            os.path.join(analysis_dir, model, "strategy_analysis*.json")))
        if not files:
            continue
        with open(files[0]) as f:
            data = json.load(f)
        summary = dict(data.get("summary", {}))
        records = data.get("agent_records", [])

        # ── Group per (run, generation) for AUC; per run for other metrics ──
        auc_by_run_gen = {}     # (run, gen) -> [auc values for that run/gen]
        coop_by_run    = {}     # run -> [actual cooperation % values]
        gap_by_run     = {}     # run -> [adherence gap values for fully-spec agents]
        unspec_by_run  = {}     # run -> [count, total] for pct_underspecified

        for r in records:
            run = r.get("run", "")
            gen = r.get("generation")

            # cooperation % across ALL agents (specified or not)
            ap = r.get("actual_avg_donation_pct")
            if ap is not None:
                coop_by_run.setdefault(run, []).append(ap)

            # underspec counter (per run)
            unspec_by_run.setdefault(run, [0, 0])
            unspec_by_run[run][1] += 1
            if not r.get("is_specified"):
                unspec_by_run[run][0] += 1

            if not r.get("is_specified"):
                continue

            # adherence gap (fully-specified only)
            adh = r.get("adherence") or {}
            ag = adh.get("adherence_gap")
            if ag is not None:
                gap_by_run.setdefault(run, []).append(ag)

            # AUC
            code = r.get("strategy_code")
            if code:
                v = _strategy_auc(code)
                if v is not None:
                    auc_by_run_gen.setdefault((run, gen), []).append(v)

        # ── Per-run aggregates → mean & std across runs ──
        coop_per_run   = [np.mean(v) for v in coop_by_run.values()   if v]
        gap_per_run    = [np.mean(v) for v in gap_by_run.values()    if v]
        unspec_pct_per_run = [100.0 * u[0] / u[1] for u in unspec_by_run.values() if u[1] > 0]
        unspec_n_per_run   = [u[0] for u in unspec_by_run.values()]

        def stats(values):
            if not values:
                return None, None, 0
            return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, len(values)

        m, s, n = stats(coop_per_run)
        summary["mean_actual_cooperation_pct_all_agents"] = m
        summary["std_actual_cooperation_pct_all_agents"]  = s
        m, s, n = stats(gap_per_run)
        if "fully_specified_adherence" not in summary or not isinstance(summary["fully_specified_adherence"], dict):
            summary["fully_specified_adherence"] = {}
        summary["fully_specified_adherence"]["mean_adherence_gap"] = m
        summary["fully_specified_adherence"]["std_adherence_gap"]  = s
        m, s, n = stats(unspec_pct_per_run)
        summary["pct_underspecified"] = m
        summary["std_pct_underspecified"] = s
        m, s, n = stats(unspec_n_per_run)
        summary["n_underspecified"] = m
        summary["std_n_underspecified"] = s

        # AUC per-run by generation
        gen1_runs   = [np.mean(v) for (run, gen), v in auc_by_run_gen.items() if gen == 1]
        all_gens    = [g for (_, g) in auc_by_run_gen.keys()]
        final_gen   = max(all_gens) if all_gens else None
        finalN_runs = [np.mean(v) for (run, gen), v in auc_by_run_gen.items() if gen == final_gen]

        m, s, n = stats(gen1_runs)
        summary["mean_auc_gen1"] = m
        summary["std_auc_gen1"]  = s
        m, s, n = stats(finalN_runs)
        summary["mean_auc_genN"] = m
        summary["std_auc_genN"]  = s
        summary["_final_gen"] = final_gen

        out[model] = summary
    return out


# ── Plot ────────────────────────────────────────────────────────────────────

def plot_comparison(summaries_list: list, batch_labels: list, output_dir: str):
    n_batches = len(summaries_list)
    models = [m for m in MODEL_LABELS if any(m in s for s in summaries_list)]
    n_models = len(models)
    if n_models == 0 or n_batches == 0:
        raise SystemExit(
            "No data to plot. Verify each --analysis_dirs entry contains "
            "<model>/strategy_analysis_<model>.json files."
        )

    metrics = [
        ("n_underspecified",
         "# Unspecified strategies",
         False),
        ("pct_underspecified",
         "% Unspecified strategies",
         False),
        ("mean_actual_cooperation_pct_all_agents",
         "Mean cooperation %\n(all agents)",
         False),
        ("fully_specified_adherence.mean_adherence_gap",
         "Adherence gap\n(predicted − actual, pp)",
         True),
        ("mean_auc_gen1",
         "Mean AUC of coded strategies\n(generation 1)",
         False),
        ("mean_auc_genN",
         "Mean AUC of coded strategies\n(final generation)",
         False),
    ]

    # 2x3 grid — exactly 6 panels for 6 metrics
    fig, axes = plt.subplots(2, 3, figsize=(max(14, 3 * n_batches), 9))
    axes = axes.flatten()

    total_bars = n_models * n_batches
    bar_width  = 0.85 / total_bars
    offsets    = np.linspace(-(total_bars - 1) / 2, (total_bars - 1) / 2, total_bars) * bar_width

    for ax_idx, (metric_key, ylabel, allow_negative) in enumerate(metrics):
        ax = axes[ax_idx]
        bar_idx = 0
        for b_idx, (summaries, blabel) in enumerate(zip(summaries_list, batch_labels)):
            hatch = VARIANT_HATCHES[b_idx % len(VARIANT_HATCHES)]
            alpha = VARIANT_ALPHAS[b_idx % len(VARIANT_ALPHAS)]
            for m_idx, model in enumerate(models):
                s = summaries.get(model)
                if s is None:
                    bar_idx += 1
                    continue

                # Navigate nested key
                val = s
                try:
                    for part in metric_key.split("."):
                        val = val[part]
                except (KeyError, TypeError):
                    val = None
                if val is None:
                    bar_idx += 1
                    continue

                # Look up the matching std (sibling key with "std_" prefix or
                # "std_" replacing "mean_" inside the nested dict).
                parts = metric_key.split(".")
                std_key = "std_" + parts[-1].replace("mean_", "")
                std_path = parts[:-1] + [std_key]
                std_val = s
                try:
                    for part in std_path:
                        std_val = std_val[part]
                    if std_val is None or std_val == 0:
                        std_val = None
                except (KeyError, TypeError):
                    std_val = None

                x = m_idx + offsets[bar_idx]
                color = MODEL_COLORS[model]
                ax.bar(x, val, width=bar_width * 0.9,
                       color=color, hatch=hatch,
                       edgecolor="white" if not hatch else color,
                       linewidth=0.8, alpha=alpha,
                       yerr=std_val,
                       error_kw=dict(ecolor='#333333', capsize=2.5, capthick=0.8, elinewidth=0.8))

                fmt = f"{val:.3f}" if abs(val) < 1 else f"{val:.1f}"
                va = "bottom" if val >= 0 else "top"
                offset_from_bar = (std_val or 0) + (0.005 if val >= 0 else -0.005)
                ax.text(x, val + (offset_from_bar if val >= 0 else -offset_from_bar),
                        fmt, ha="center", va=va,
                        fontsize=6, color="#333333")

                bar_idx += 1

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=9)
        ax.tick_params(axis='x', length=0)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if allow_negative:
            ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.3 or 1
            ax.set_ylim(-ymax, ymax)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

    # Legend: one entry per variant, ordered left-to-right within each model
    # group. The model colour is reproduced separately by the x-axis labels.
    legend_handles = []
    for b_idx, blabel in enumerate(batch_labels):
        hatch = VARIANT_HATCHES[b_idx % len(VARIANT_HATCHES)]
        alpha = VARIANT_ALPHAS[b_idx % len(VARIANT_ALPHAS)]
        legend_handles.append(mpatches.Patch(
            facecolor="#888888", hatch=hatch,
            edgecolor="#444444", alpha=alpha,
            label=blabel,
        ))

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(batch_labels), 6), fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Strategy Adherence Metrics by Model and Prompt Variant",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "strategy_adherence_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare strategy adherence metrics across models for N batches.")
    parser.add_argument("--analysis_dirs", nargs="+", default=None,
                        help="One or more analysis directories.")
    parser.add_argument("--analysis_dir1", default=None,
                        help="(Legacy) first analysis dir.")
    parser.add_argument("--analysis_dir2", default=None,
                        help="(Legacy) second analysis dir.")
    parser.add_argument("--output_dir", default="figures/strategy_groups")
    args = parser.parse_args()

    if args.analysis_dirs:
        dirs = args.analysis_dirs
    elif args.analysis_dir1 or args.analysis_dir2:
        dirs = [d for d in (args.analysis_dir1, args.analysis_dir2) if d]
    else:
        parser.error("Provide --analysis_dirs (one or more) "
                     "or legacy --analysis_dir1/--analysis_dir2.")

    labels = [os.path.basename(d.rstrip("/")) for d in dirs]
    summaries_list = []
    for d, lbl in zip(dirs, labels):
        print(f"Loading {lbl} from {d} …")
        summaries_list.append(load_analysis(d))

    plot_comparison(summaries_list, labels, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
