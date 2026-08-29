#!/usr/bin/env python3
"""
Visualize direct-evolution runs vs an LLM-evolution baseline.

Default comparison is simulations/neutral_baseline (n=20/20/18 runs per model).
Caveat: those baseline runs used the ORIGINAL models while direct_evo_new used
the *_new models, so solid-vs-dashed differences mix evolution mode with model
generation. Pass --baseline_dir simulations/baseline_new for a same-model
contrast (fewer runs available locally).

Three panels:
  A. Cooperation by generation — mean donation % per generation, mean ± SEM
     across runs; direct evolution solid, LLM-evolution baseline dashed.
  B. Strategy diversity by generation — number of distinct strategy texts among
     the 12 agents (direct evolution copies verbatim, so diversity can only
     decay; LLM evolution rewrites strategies, keeping diversity high).
  C. Adoption-rank distribution — how often new agents copied the rank-1..6
     survivor (direct evolution only), vs the uniform 1/6 line. With a large
     beta the softmax should be close to uniform; with a small beta it should
     pile onto rank 1.

Usage:
    python figures/plot_direct_evolution.py \
        [--directevo_dir simulations/direct_evo_new] \
        [--baseline_dir simulations/baseline_new] \
        [--output figures/direct_evolution_summary.png]
"""

import argparse
import glob
import json
import os
import textwrap

import numpy as np
import matplotlib.pyplot as plt

MODELS = ["claude", "gpt", "gemini"]
MODEL_BASE = {"claude": "Claude", "gpt": "GPT", "gemini": "Gemini"}
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}
MODEL_LABELS = dict(MODEL_BASE)   # overwritten from --model_suffix in main()


def per_run_donation_by_gen(folder):
    """[{gen: mean donation %}] — one dict per run, from Final_Accum files.
    Searches recursively so both flat model dirs and ones still wrapping the
    scp'd array_<tag>_<jobid>/ subdirectory work."""
    runs = []
    for p in sorted(glob.glob(os.path.join(folder, "**", "*Final_Accum*.json"),
                              recursive=True)):
        with open(p) as f:
            stats = json.load(f).get("generation_stats", [])
        d = {s["generation"]: s["average_donation_percentage"] for s in stats
             if s.get("average_donation_percentage") is not None}
        if d:
            runs.append(d)
    return runs


def per_run_diversity_by_gen(folder):
    """[{gen: n distinct strategy texts}] — one dict per run, from all_strategies files."""
    runs = []
    for p in sorted(glob.glob(os.path.join(folder, "**", "all_strategies_*.json"),
                              recursive=True)):
        with open(p) as f:
            gens = json.load(f)
        d = {}
        for gen_list in gens:
            if gen_list:
                g = gen_list[0]["generation"]
                d[g] = len({(rec.get("strategy") or "").strip() for rec in gen_list})
        if d:
            runs.append(d)
    return runs


def adoption_rank_counts(folder, n_ranks=6):
    """Counts of inherited_rank 1..n_ranks across all gen-2+ agents, all runs."""
    counts = np.zeros(n_ranks)
    for p in sorted(glob.glob(os.path.join(folder, "**", "all_strategies_*.json"),
                              recursive=True)):
        with open(p) as f:
            gens = json.load(f)
        for gen_list in gens:
            for rec in gen_list:
                r = rec.get("inherited_rank")
                if rec.get("generation", 1) > 1 and r and 1 <= r <= n_ranks:
                    counts[r - 1] += 1
    return counts


def game_rule(folder):
    """Identify which endowment rule a dataset was run under, from the stored
    system_prompt. The rule changed mid-project: runs before the change gave a
    fresh 10 units every round (unspent units banked, so donating is cheap);
    runs after give one initial endowment that accumulates. Datasets on opposite
    sides of that change are NOT comparable."""
    for p in glob.glob(os.path.join(folder, "**", "*Final_Accum*.json"), recursive=True):
        try:
            sp = json.load(open(p))["hyperparameters"].get("system_prompt", "")
        except Exception:
            continue
        if "fixed new" in sp:
            return "fixed-per-round + savings"
        if "initial endowment" in sp:
            return "initial endowment"
        return "unknown"
    return None


def model_id(folder):
    """The `llm` string a dataset was run with (e.g. gpt-4o vs gpt-new)."""
    for p in glob.glob(os.path.join(folder, "**", "*Final_Accum*.json"), recursive=True):
        try:
            return json.load(open(p))["hyperparameters"].get("llm")
        except Exception:
            continue
    return None


def mean_sem_by_gen(runs):
    """(gens, means, sems) across runs for a list of {gen: value} dicts."""
    all_gens = sorted({g for r in runs for g in r})
    means, sems = [], []
    for g in all_gens:
        vals = np.array([r[g] for r in runs if g in r], float)
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    return np.array(all_gens), np.array(means), np.array(sems)


def plot_lines(ax, runs_by_cond, ylabel, title):
    for model in MODELS:
        for cond, ls in (("direct_evo", "-"), ("baseline", "--")):
            runs = runs_by_cond.get(model, {}).get(cond, [])
            if not runs:
                continue
            gens, means, sems = mean_sem_by_gen(runs)
            label = MODEL_LABELS[model] + (" — direct evo" if cond == "direct_evo"
                                           else " — LLM evo (baseline)")
            ax.plot(gens, means, ls, color=MODEL_COLORS[model], linewidth=2,
                    marker="o" if cond == "direct_evo" else None, markersize=3.5,
                    label=label)
            ax.fill_between(gens, means - sems, means + sems,
                            color=MODEL_COLORS[model], alpha=0.12, linewidth=0)
    ax.set_xlabel("Generation", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _finish(fig, args, rule_mismatch, rule_de, rule_bl, model_pairs,
            single_panel=False):
    """Title, validity banner, layout and save — shared by both figure shapes."""
    fig.suptitle("Direct evolution (softmax strategy adoption) vs LLM evolution"
                 + (f" — {args.subtitle}" if args.subtitle else ""),
                 fontsize=11 if single_panel else 13)
    notes = []
    if rule_mismatch:
        notes.append(f"INVALID — game rules differ (direct evo = {rule_de}, "
                     f"baseline = {rule_bl}); gaps exist already at gen 1")
    if model_pairs:
        notes.append("CONFOUNDED — models differ between conditions ("
                     + "; ".join(f"{m}: {a} vs {b}" for m, a, b in model_pairs) + ")")
    if notes:
        # The single-panel figure is narrower, so the banner needs to wrap
        # tighter or it runs off the canvas.
        text = "\n".join(textwrap.fill(n, 68) for n in notes) if single_panel \
            else "\n".join(notes)
        fig.text(0.5, 0.915, text, ha="center", va="top",
                 fontsize=7.5 if single_panel else 8.5,
                 color="#b3261e", fontweight="bold", linespacing=1.5)
    n_lines = sum(len(t.splitlines()) for t in
                  ([textwrap.fill(n, 68) for n in notes] if single_panel else notes))
    plt.tight_layout(rect=[0, 0, 1, 0.93 - 0.035 * n_lines])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Visualize direct-evolution runs.")
    parser.add_argument("--directevo_dir", default="simulations/direct_evo_new")
    parser.add_argument("--baseline_dir", default="simulations/neutral_baseline",
                        help="LLM-evolution runs to compare against. NOTE: the default "
                             "neutral_baseline used the ORIGINAL models "
                             "(claude-3-5-sonnet / gpt-4o / gemini-2.5-flash) while "
                             "direct_evo_new used the *_new models, so the contrast "
                             "confounds evolution mode with model generation.")
    parser.add_argument("--output", default="figures/direct_evolution_summary.png")
    parser.add_argument("--model_suffix", default="(new)",
                        help="Appended to model labels, e.g. '(new)'. Pass '' for bare names.")
    parser.add_argument("--subtitle", default="",
                        help="Extra text appended to the figure title (e.g. 'beta=5, original models')")
    parser.add_argument("--cooperation_only", action="store_true",
                        help="Plot only panel A (cooperation by generation) as a "
                             "single-panel figure. Skips the diversity and "
                             "adoption-rank passes entirely, so it is faster.")
    parser.add_argument("--legend_loc", default="lower right",
                        help="Legend position for --cooperation_only, e.g. "
                             "'lower right', 'upper left', 'center right'.")
    args = parser.parse_args()

    for m in MODELS:
        MODEL_LABELS[m] = f"{MODEL_BASE[m]} {args.model_suffix}".strip()

    # Guard: the two datasets must have been run under the same game rule, or
    # solid-vs-dashed differences are dominated by the rule change, not by the
    # evolution mechanism.
    rule_de = game_rule(os.path.join(args.directevo_dir, "claude")) or game_rule(args.directevo_dir)
    rule_bl = game_rule(os.path.join(args.baseline_dir, "claude")) or game_rule(args.baseline_dir)
    rule_mismatch = rule_de and rule_bl and rule_de != rule_bl
    if rule_mismatch:
        print(f"\n*** WARNING: game-rule mismatch — NOT a valid comparison ***\n"
              f"    direct evo : {rule_de}\n    baseline   : {rule_bl}\n"
              f"    Differences (including at generation 1, before evolution acts) "
              f"reflect the rule change, not the evolution mechanism.\n")

    # Guard 2: the two sides should also use the same underlying models, or the
    # contrast mixes evolution mode with model generation.
    model_pairs = [(m, model_id(os.path.join(args.directevo_dir, m)),
                    model_id(os.path.join(args.baseline_dir, m))) for m in MODELS]
    model_pairs = [(m, a, b) for m, a, b in model_pairs if a and b and a != b]
    if model_pairs:
        detail = ", ".join(f"{m}: {a} vs {b}" for m, a, b in model_pairs)
        print(f"\n*** WARNING: model mismatch between conditions ***\n    {detail}\n"
              f"    Solid-vs-dashed differences mix evolution mode with model generation.\n")

    coop, div = {}, {}
    ranks = {}
    for model in MODELS:
        de = os.path.join(args.directevo_dir, model)
        bl = os.path.join(args.baseline_dir, model)
        coop[model] = {"direct_evo": per_run_donation_by_gen(de),
                       "baseline":   per_run_donation_by_gen(bl)}
        if args.cooperation_only:
            print(f"[{model}] direct_evo runs: {len(coop[model]['direct_evo'])}, "
                  f"baseline runs: {len(coop[model]['baseline'])}")
            continue
        div[model] = {"direct_evo": per_run_diversity_by_gen(de),
                      "baseline":   per_run_diversity_by_gen(bl)}
        ranks[model] = adoption_rank_counts(de)
        print(f"[{model}] direct_evo runs: {len(coop[model]['direct_evo'])}, "
              f"baseline runs: {len(coop[model]['baseline'])}, "
              f"adoptions: {int(ranks[model].sum())}")

    if args.cooperation_only:
        fig, ax_a = plt.subplots(1, 1, figsize=(7.5, 5.5))
        # No axes title — the suptitle already names the comparison, and two
        # stacked titles on a single panel just eats vertical space.
        plot_lines(ax_a, coop, "Mean donation % (cooperation)", "")
        # Every corner of this plot tends to be occupied — Claude sits high,
        # GPT low, Gemini through the middle — so open up headroom on whichever
        # side the legend is going, rather than letting it cover the lines.
        lo, hi = ax_a.get_ylim()
        pad = 0.22 * (hi - lo)
        if "lower" in args.legend_loc:
            ax_a.set_ylim(lo - pad, hi)
        elif "upper" in args.legend_loc:
            ax_a.set_ylim(lo, hi + pad)
        ax_a.legend(fontsize=8, loc=args.legend_loc, framealpha=0.92,
                    borderpad=0.6, labelspacing=0.4)
        _finish(fig, args, rule_mismatch, rule_de, rule_bl, model_pairs,
                single_panel=True)
        return

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(17, 5))

    plot_lines(ax_a, coop, "Mean donation % (cooperation)",
               "A. Cooperation by generation")
    ax_a.legend(fontsize=7.5, ncol=1, loc="best")

    plot_lines(ax_b, div, "Distinct strategies (of 12 agents)",
               "B. Strategy diversity by generation")
    ax_b.set_ylim(0, 12.5)

    # C: adoption-rank distribution (direct evo only)
    n_ranks = 6
    xs = np.arange(1, n_ranks + 1)
    bar_w = 0.75 / len(MODELS)
    for i, model in enumerate(MODELS):
        c = ranks[model]
        if c.sum() == 0:
            continue
        ax_c.bar(xs + (i - 1) * bar_w, c / c.sum(), bar_w * 0.92,
                 color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                 edgecolor="white", linewidth=0.5)
    ax_c.axhline(1 / n_ranks, color="#555", linestyle=":", linewidth=1.2,
                 label="uniform (1/6)")
    ax_c.set_xticks(xs)
    ax_c.set_xlabel("Survivor rank copied (1 = best)", fontsize=10)
    ax_c.set_ylabel("Fraction of adoptions", fontsize=10)
    ax_c.set_title("C. Softmax adoption ranks (direct evo)", fontsize=11)
    ax_c.legend(fontsize=8)
    ax_c.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)

    _finish(fig, args, rule_mismatch, rule_de, rule_bl, model_pairs)


if __name__ == "__main__":
    main()
