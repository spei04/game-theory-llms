#!/usr/bin/env python3
"""
Compare agent-written coded strategies (ASK_CODED runs) against the
text->code pipeline (strategies written in text, classified/compiled by
Claude via figures/strategy_analysis.py) on the neutral baseline.

Two sources:
  ask_coded : simulations/ask_coded/{model}/**/all_strategies_*.json
              (strategy_code written directly by the agent during the run)
  text->code: figures/reactivity_plots/neutral_baseline/{model}/
              strategy_analysis_{model}.json
              (strategy_code produced post-hoc by the classifier LLM)

For generation 1 (= generation 0, the initial strategies) the text->code side
defaults to the strategies-only initial set instead — ~200 runs of gen-0
strategies with no gameplay (figures/reactivity_plots/neutral_baseline_200) —
falling back to neutral_baseline if that set is absent. Pass --baseline_dir
explicitly to override either default.

For each model, evaluates every strategy_fn over pair_prev_pct in [0, 100]
and produces:
  gen{g}_reactivity_askcoded_vs_textcoded.png  mean±std curves, one panel per model
  gen{g}_auc_askcoded_vs_textcoded.png         normalised AUC bars per model x condition
  gen{g}_askcoded_vs_textcoded_stats.json      per-cell stats

Usage:
    python figures/compare_askcoded_strategies.py \
        [--askcoded_dir simulations/ask_coded] \
        [--baseline_dir figures/reactivity_plots/neutral_baseline] \
        [--generation 1] \
        [--output_dir figures/askcoded_comparison]
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import eval_strategy, auc_normalised, X_VALUES
from compare_gen0_variants import eval_first_round

MODELS = ["claude", "gpt", "gemini"]
# Display labels for plots; MODEL_STRINGS records the run-time model ids.
MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o", "gemini": "Gemini 2.5 Flash"}
MODEL_STRINGS = {"claude": "claude-3-5-sonnet", "gpt": "gpt-4o", "gemini": "gemini-2.5-flash"}
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}
TEXTCODE_COLOR = "#666666"

# What the *_new launcher aliases actually resolve to, per the dispatch blocks
# in slurm/experiments/donor_game_baseline.py. Select with --model_set new.
MODEL_LABELS_NEW = {"claude": "Claude Opus 4.8", "gpt": "GPT-5",
                    "gemini": "Gemini 3.6 Flash"}
MODEL_STRINGS_NEW = {"claude": "claude-opus-4-8", "gpt": "gpt-5",
                     "gemini": "gemini-3.6-flash"}

# text->code sources: gen-0 comparisons use the ~200-run strategies-only
# initial set; later generations come from the 20-run full-game baseline.
GEN0_BASELINE_DIR     = "figures/reactivity_plots/neutral_baseline_200"
FULLGAME_BASELINE_DIR = "figures/reactivity_plots/neutral_baseline"

_TS_RE = re.compile(r'_(\d{8}_\d{6})\.json$')


def _run_id(path):
    m = _TS_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def load_askcoded(askcoded_dir, model, generation):
    """Return [(run_id, code), ...] for agent-written strategies of one model."""
    pattern = os.path.join(askcoded_dir, model, "**", "all_strategies_*.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    # Dedupe runs that exist both flat and inside a copied array_* subdir —
    # the timestamp identifies the run regardless of location.
    seen, unique_paths = set(), []
    for p in paths:
        rid = _run_id(p)
        if rid in seen:
            print(f"  [{model}] duplicate run {rid} skipped: {p}")
            continue
        seen.add(rid)
        unique_paths.append(p)
    paths = unique_paths
    out = []
    for p in paths:
        with open(p) as f:
            gens = json.load(f)
        for gen_list in gens:
            for rec in gen_list:
                if rec.get("generation") == generation and rec.get("strategy_code"):
                    out.append((_run_id(p), rec["strategy_code"]))
    if not paths:
        print(f"  [{model}] no all_strategies_*.json under {os.path.join(askcoded_dir, model)}")
    return out


def load_textcoded(baseline_dir, model, generation):
    """Return [(run_id, code), ...] from the classifier-derived analysis JSON."""
    path = os.path.join(baseline_dir, model, f"strategy_analysis_{model}.json")
    if not os.path.exists(path):
        print(f"  [{model}] not found: {path}")
        return []
    with open(path) as f:
        records = json.load(f).get("agent_records", [])
    return [(r.get("run", ""), r["strategy_code"]) for r in records
            if r.get("generation") == generation
            and r.get("is_specified") and r.get("strategy_code")]


def curves_and_aucs(pairs):
    """Evaluate each (run, code): returns curves, agent aucs, first-round
    donations (strategy_fn(None)), {run: [aucs]}, and eval-failure count."""
    curves, aucs, firsts, run_aucs = [], [], [], {}
    n_failed = 0
    for run, code in pairs:
        ys = eval_strategy(code)
        if ys is None:
            n_failed += 1
            continue
        curves.append(ys)
        auc = auc_normalised(ys)
        aucs.append(auc)
        run_aucs.setdefault(run, []).append(auc)
        fr = eval_first_round(code)
        if fr is not None:
            firsts.append(fr)
    return np.array(curves), np.array(aucs), np.array(firsts), run_aucs, n_failed


def cell_stats(aucs, firsts, run_aucs, n_failed):
    run_means = np.array([np.mean(v) for v in run_aucs.values()])
    return {
        "n_agents": int(aucs.size),
        "n_runs": len(run_aucs),
        "n_eval_failed": n_failed,
        "auc_mean": float(np.mean(aucs)) if aucs.size else None,
        "auc_std": float(np.std(aucs)) if aucs.size else None,
        "auc_run_means": [float(m) for m in run_means],
        "auc_run_std": float(np.std(run_means, ddof=1)) if len(run_means) > 1 else None,
        "first_round_mean_pct": float(np.mean(firsts)) if firsts.size else None,
        "first_round_std_pct": float(np.std(firsts)) if firsts.size else None,
        "n_first_round": int(firsts.size),
    }


def condition_tests(ask, txt):
    """Mann-Whitney U (two-sided) on agent-level AUCs and first-round donations."""
    out = {}
    if ask["aucs"].size and txt["aucs"].size:
        u, p = sps.mannwhitneyu(ask["aucs"], txt["aucs"], alternative="two-sided")
        out["auc_mannwhitney_agents"] = {"U": float(u), "p": float(p)}
    if ask["firsts"].size and txt["firsts"].size:
        u, p = sps.mannwhitneyu(ask["firsts"], txt["firsts"], alternative="two-sided")
        out["first_round_mannwhitney_agents"] = {"U": float(u), "p": float(p)}
    return out


def main():
    global MODEL_LABELS, MODEL_STRINGS
    parser = argparse.ArgumentParser(
        description="Compare agent-written coded strategies vs the text->code pipeline.")
    parser.add_argument("--askcoded_dir", default="simulations/ask_coded",
                        help="Directory with {model}/ subdirs of ask_coded run outputs")
    parser.add_argument("--baseline_dir", default=None,
                        help="Directory with {model}/strategy_analysis_{model}.json. "
                             "Default: the gen-0 initial-strategies set "
                             f"({GEN0_BASELINE_DIR}) when --generation 1, "
                             f"else {FULLGAME_BASELINE_DIR}.")
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--output_dir", default="figures/askcoded_comparison")
    parser.add_argument("--n_baseline_runs", type=int, default=0,
                        help="Restrict the text->code side to the first N baseline "
                             "runs (by timestamp) so both sides have matched n "
                             "(e.g. 1 -> one run's 12 initial strategies). 0 = all runs.")
    parser.add_argument("--model_set", default="original",
                        choices=["original", "new"],
                        help="Which model generation these runs used. 'new' "
                             "relabels the panels as Claude Opus 4.8 / GPT-5 / "
                             "Gemini 3.6 Flash (the *_new launcher aliases).")
    parser.add_argument("--suffix", default="",
                        help="Extra tag appended to output filenames, e.g. _new")
    args = parser.parse_args()

    if args.model_set == "new":
        MODEL_LABELS, MODEL_STRINGS = MODEL_LABELS_NEW, MODEL_STRINGS_NEW

    gen = args.generation
    if args.baseline_dir is None:
        if gen == 1 and os.path.isdir(GEN0_BASELINE_DIR):
            args.baseline_dir = GEN0_BASELINE_DIR
        else:
            args.baseline_dir = FULLGAME_BASELINE_DIR
    print(f"[text->code source] {args.baseline_dir}")
    data = {}   # model -> condition -> dict
    stats = {"n_baseline_runs": args.n_baseline_runs or "all",
             "baseline_runs_used": {}}

    for model in MODELS:
        ask_pairs = load_askcoded(args.askcoded_dir, model, gen)
        txt_pairs = load_textcoded(args.baseline_dir, model, gen)
        if args.n_baseline_runs and txt_pairs:
            keep = sorted({r for r, _ in txt_pairs})[:args.n_baseline_runs]
            txt_pairs = [(r, c) for r, c in txt_pairs if r in keep]
            stats["baseline_runs_used"][model] = keep
            print(f"  [{model}] text->code restricted to run(s): {', '.join(keep)}")
        data[model] = {}
        stats[model] = {}
        for cond, pairs in (("ask_coded", ask_pairs), ("text_to_code", txt_pairs)):
            curves, aucs, firsts, run_aucs, n_failed = curves_and_aucs(pairs)
            data[model][cond] = (curves, aucs)
            stats[model][cond] = cell_stats(aucs, firsts, run_aucs, n_failed)
            stats[model][cond]["_firsts"] = firsts
            s = stats[model][cond]
            if s["n_agents"]:
                fr = (f", 1st round {s['first_round_mean_pct']:.1f}%"
                      if s["first_round_mean_pct"] is not None else "")
                print(f"  [{model}/{cond}] n={s['n_agents']} agents, {s['n_runs']} runs, "
                      f"AUC {s['auc_mean']:.3f} ± {s['auc_std']:.3f}{fr}")
            else:
                print(f"  [{model}/{cond}] no evaluable strategies")

        ask_s, txt_s = stats[model]["ask_coded"], stats[model]["text_to_code"]
        tests = condition_tests(
            {"aucs": data[model]["ask_coded"][1], "firsts": ask_s["_firsts"]},
            {"aucs": data[model]["text_to_code"][1], "firsts": txt_s["_firsts"]})
        stats[model]["tests"] = tests
        if "auc_mannwhitney_agents" in tests:
            p1 = tests["auc_mannwhitney_agents"]["p"]
            msg = f"  [{model}] ask_coded vs text->code: AUC MW p={p1:.4f}"
            if "first_round_mannwhitney_agents" in tests:
                msg += f", 1st round MW p={tests['first_round_mannwhitney_agents']['p']:.4f}"
            print(msg)
        for cond in ("ask_coded", "text_to_code"):
            stats[model][cond].pop("_firsts", None)

    if not any(data[m][c][0].size for m in MODELS for c in data[m]):
        print("\nNothing to plot. Copy the ask_coded results down first, e.g.:\n"
              "  scp -r <user>@<cluster-host>:.../simulations/array_gpt_askcoded_<jobid>/ "
              "simulations/ask_coded/gpt")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = (f"_first{args.n_baseline_runs}run" if args.n_baseline_runs else "") \
             + args.suffix

    # ── Figure 1: reactivity curves, one panel per model ─────────────────────
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.8), sharey=True)
    for ax, model in zip(axes, MODELS):
        plotted = False
        for cond, color, ls, label in (
                ("ask_coded", MODEL_COLORS[model], "-", "agent-written code"),
                ("text_to_code", TEXTCODE_COLOR, "--", "text → code (classifier)")):
            curves, _ = data[model][cond]
            if not curves.size:
                continue
            mean_y = curves.mean(axis=0)
            std_y = curves.std(axis=0)
            ax.plot(X_VALUES, mean_y, color=color, linestyle=ls, linewidth=2.5,
                    label=label)
            ax.fill_between(X_VALUES,
                            np.clip(mean_y - std_y, 0, 100),
                            np.clip(mean_y + std_y, 0, 100),
                            alpha=0.15, color=color)
            plotted = True
        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.set_xlabel("Pair's donation % in previous round", fontsize=10)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("My donation %", fontsize=10)
    fig.suptitle(f"Agent-written vs text→code strategies — generation {gen}",
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(args.output_dir, f"gen{gen}_reactivity_askcoded_vs_textcoded{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    # ── Figure 2: AUC bars per model x condition ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_w = 0.35
    xs = np.arange(len(MODELS))
    for i, (cond, label) in enumerate((("ask_coded", "agent-written code"),
                                       ("text_to_code", "text → code (classifier)"))):
        means, stds, colors = [], [], []
        for model in MODELS:
            _, aucs = data[model][cond]
            means.append(aucs.mean() if aucs.size else 0)
            stds.append(aucs.std() if aucs.size else 0)
            colors.append(MODEL_COLORS[model] if cond == "ask_coded" else TEXTCODE_COLOR)
        offset = (i - 0.5) * bar_w
        ax.bar(xs + offset, means, bar_w * 0.92, yerr=stds, capsize=3,
               color=colors, alpha=0.9 if cond == "ask_coded" else 0.55,
               edgecolor="white", linewidth=0.5, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylabel("AUC of strategy_fn (normalised, 0–1)", fontsize=10)
    ax.set_title(f"Strategy cooperativeness — generation {gen}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(args.output_dir, f"gen{gen}_auc_askcoded_vs_textcoded{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    stats_path = os.path.join(args.output_dir, f"gen{gen}_askcoded_vs_textcoded_stats{suffix}.json")
    with open(stats_path, "w") as f:
        json.dump({"model_strings": MODEL_STRINGS, **stats}, f, indent=2)
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
