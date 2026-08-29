#!/usr/bin/env python3
"""
How different are runs WITHIN each generation — AUC and text semantics.

Two per-generation analyses for each experiment/model:

1. AUC of coded strategies (auc_across_runs.png)
   Per agent: trapezoidal AUC of strategy_fn over pair_prev_pct in [0,100],
   normalised to [0,1] (same as compute_auc.py). Within each generation:
     - run-mean AUC per run (how far apart do runs sit?)
     - one-way ANOVA across runs on agent-level AUC (F, p)
     - eta^2 = share of agent-level AUC variance explained by run identity

2. Semantic similarity of text strategies (text_similarity_across_runs.png)
   Sentence embeddings (all-MiniLM-L6-v2, unit-norm). Within each generation:
     - cross-run similarity: pairwise cosine similarity between run centroids
     - within-run similarity: mean pairwise cosine similarity of agents in
       the same run (reference)
     - semantic eta^2: ||run centroid - grand centroid||^2 variance share

Output per experiment/model in --output_dir/<experiment>/<model>/:
  auc_across_runs.png, text_similarity_across_runs.png
Plus --output_dir/across_run_divergence.json and a console summary.

Usage:
    python figures/analyze_across_run_divergence.py \
        [--analysis_root figures/reactivity_plots] \
        [--experiments neutral_baseline neutral_claude ...] \
        [--models claude gpt gemini] \
        [--output_dir figures/run_variation]
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

X_VALUES = np.linspace(0, 100, 500)

MODEL_COLORS = {          # same as plot_coded_strategies.py
    "claude": "#FF6B35",
    "gpt":    "#10A37F",
    "gemini": "#4285F4",
}

_UNICODE_OPS = [
    ('≥', '>='), ('≤', '<='), ('≠', '!='),
    ('×', '*'),  ('÷', '/'),  ('−', '-'),
]


def _sanitize(code):
    for uni, asc in _UNICODE_OPS:
        code = code.replace(uni, asc)
    return code


def _fix_empty_blocks(code):
    lines = code.splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        result.append(line)
        code_part = line.split('#')[0].rstrip()
        if not code_part.endswith(':'):
            continue
        indent = len(line) - len(line.lstrip())
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith('#')):
            j += 1
        if j >= len(lines) or (len(lines[j]) - len(lines[j].lstrip())) <= indent:
            result.append(' ' * (indent + 4) + 'pass\n')
    return ''.join(result)


def eval_strategy(code):
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
    fn = ns.get('strategy_fn')
    if fn is None:
        return None
    try:
        return np.array([float(np.clip(fn(x), 0, 100)) for x in X_VALUES])
    except Exception:
        return None


def auc_normalised(ys):
    return float(np.trapz(ys, X_VALUES) / (100.0 * 100.0))


def load_agents(json_path):
    """List of dicts: run, generation, text, auc (auc None if not evaluable)."""
    with open(json_path) as f:
        data = json.load(f)
    agents, cache = [], {}
    for r in data.get("agent_records", []):
        text = (r.get("strategy") or "").strip()
        if not text:
            continue
        auc = None
        code = r.get("strategy_code")
        if r.get("is_specified", False) and code:
            if code not in cache:
                ys = eval_strategy(code)
                cache[code] = auc_normalised(ys) if ys is not None else None
            auc = cache[code]
        agents.append({"run": r["run"], "generation": r["generation"],
                       "text": text, "auc": auc})
    return agents


def eta_squared_1d(groups):
    """Share of variance explained by group identity (1-D values)."""
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_total = ((allv - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    return float(ss_between / ss_total)


def eta_squared_emb(groups):
    """Variance share of run identity in embedding space (squared distances)."""
    allv = np.vstack(groups)
    grand = allv.mean(axis=0)
    ss_total = ((allv - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(g) * ((g.mean(axis=0) - grand) ** 2).sum() for g in groups)
    return float(ss_between / ss_total)


def analyze_combo(agents, run_index, gens, label):
    """Per-generation AUC + text stats. Returns dict for JSON + plot data."""
    out = {"auc_by_gen": {}, "text_by_gen": {}}
    for g in gens:
        pool = [a for a in agents if a["generation"] == g]

        # ---- AUC across runs
        by_run = defaultdict(list)
        for a in pool:
            if a["auc"] is not None:
                by_run[a["run"]].append(a["auc"])
        groups = {r: np.array(v) for r, v in by_run.items() if len(v) >= 2}
        if len(groups) >= 3:
            garr = list(groups.values())
            F, p = sps.f_oneway(*garr)
            run_means = {run_index[r]: float(v.mean()) for r, v in groups.items()}
            out["auc_by_gen"][g] = {
                "run_mean_auc": run_means,
                "between_run_std": float(np.std(list(run_means.values()), ddof=1)),
                "run_mean_range": [float(min(run_means.values())),
                                   float(max(run_means.values()))],
                "anova_F": float(F), "anova_p": float(p),
                "eta_sq": eta_squared_1d(garr),
            }

        # ---- text similarity across runs
        by_run_e = defaultdict(list)
        for a in pool:
            by_run_e[a["run"]].append(a["emb"])
        eg = {r: np.vstack(v) for r, v in by_run_e.items() if len(v) >= 2}
        if len(eg) >= 3:
            cents = np.vstack([v.mean(axis=0) / np.linalg.norm(v.mean(axis=0))
                               for v in eg.values()])
            S = cents @ cents.T
            iu = np.triu_indices(len(cents), k=1)
            cross = S[iu]
            within = [float((E @ E.T)[np.triu_indices(len(E), k=1)].mean())
                      for E in eg.values()]
            out["text_by_gen"][g] = {
                "cross_run_sim_mean": float(cross.mean()),
                "cross_run_sim_std": float(cross.std()),
                "cross_run_sim_min": float(cross.min()),
                "within_run_sim_mean": float(np.mean(within)),
                "eta_sq": eta_squared_emb(list(eg.values())),
            }
    return out


def plot_auc(res, gens, out_path, label, color):
    cmap = plt.get_cmap("tab20")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True,
                                   height_ratios=[2, 1])
    rng = np.random.default_rng(0)
    for g in gens:
        d = res["auc_by_gen"].get(g)
        if not d:
            continue
        for rid, m in d["run_mean_auc"].items():
            ax1.scatter(g + rng.uniform(-0.22, 0.22), m,
                        color=cmap((rid - 1) % 20), s=22, alpha=0.85,
                        edgecolors="white", linewidths=0.4)
    grand = [np.mean(list(res["auc_by_gen"][g]["run_mean_auc"].values()))
             for g in gens if g in res["auc_by_gen"]]
    ax1.plot([g for g in gens if g in res["auc_by_gen"]], grand,
             color="#333333", lw=1.8, label="mean across runs")
    ax1.set_ylabel("run-mean AUC (0-1)", fontsize=10)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(f"{label} — AUC of coded strategies per run (color = run)",
                  fontsize=11)

    xs = [g for g in gens if g in res["auc_by_gen"]]
    ax2.plot(xs, [res["auc_by_gen"][g]["eta_sq"] for g in xs],
             color=color, lw=2.2, marker="o", ms=4)
    for g in xs:
        if res["auc_by_gen"][g]["anova_p"] < 0.05:
            ax2.annotate("*", (g, res["auc_by_gen"][g]["eta_sq"]),
                         textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=10)
    ax2.set_ylabel("eta$^2$ (run identity)", fontsize=10)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("generation", fontsize=11)
    ax2.set_xticks(gens)
    ax2.set_title("share of agent-level AUC variance explained by run "
                  "(* = ANOVA p<0.05)", fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_text(res, gens, out_path, label, color):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True,
                                   height_ratios=[2, 1])
    xs = [g for g in gens if g in res["text_by_gen"]]
    cross_m = np.array([res["text_by_gen"][g]["cross_run_sim_mean"] for g in xs])
    cross_s = np.array([res["text_by_gen"][g]["cross_run_sim_std"] for g in xs])
    within = [res["text_by_gen"][g]["within_run_sim_mean"] for g in xs]
    ax1.plot(xs, within, color="#2a78d6", lw=2.2, marker="o", ms=4,
             label="within run (agents, same run)")
    ax1.plot(xs, cross_m, color="#1baf7a", lw=2.2, marker="o", ms=4,
             label="across runs (run centroids)")
    ax1.fill_between(xs, cross_m - cross_s, cross_m + cross_s,
                     color="#1baf7a", alpha=0.15)
    ax1.set_ylabel("cosine similarity", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(f"{label} — text-strategy semantic similarity within each generation",
                  fontsize=11)

    ax2.plot(xs, [res["text_by_gen"][g]["eta_sq"] for g in xs],
             color=color, lw=2.2, marker="o", ms=4)
    ax2.set_ylabel("semantic eta$^2$", fontsize=10)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("generation", fontsize=11)
    ax2.set_xticks(gens)
    ax2.set_title("share of embedding variance explained by run identity", fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Across-run AUC + text divergence per generation.")
    ap.add_argument("--analysis_root", default="figures/reactivity_plots")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline", "neutral_claude", "neutral_gemini",
                             "neutral_gpt", "neutral_deepseek"])
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    all_stats = {}
    for exp in args.experiments:
        for model in args.models:
            json_path = os.path.join(args.analysis_root, exp, model,
                                     f"strategy_analysis_{model}.json")
            if not os.path.exists(json_path):
                continue
            label = f"{exp}/{model}"
            agents = load_agents(json_path)
            unique = sorted(set(a["text"] for a in agents))
            embs = st_model.encode(unique, batch_size=64, show_progress_bar=False,
                                   normalize_embeddings=True)
            lookup = dict(zip(unique, embs))
            for a in agents:
                a["emb"] = lookup[a["text"]]

            runs = sorted(set(a["run"] for a in agents))
            run_index = {r: i + 1 for i, r in enumerate(runs)}
            gens = sorted(set(a["generation"] for a in agents))
            res = analyze_combo(agents, run_index, gens, label)
            all_stats[label] = res

            out_dir = os.path.join(args.output_dir, exp, model)
            os.makedirs(out_dir, exist_ok=True)
            color = MODEL_COLORS.get(model, "#555555")
            plot_auc(res, gens, os.path.join(out_dir, "auc_across_runs.png"),
                     label, color)
            plot_text(res, gens, os.path.join(out_dir, "text_similarity_across_runs.png"),
                      label, color)

            a1 = res["auc_by_gen"].get(gens[0], {})
            aN = res["auc_by_gen"].get(gens[-1], {})
            t1 = res["text_by_gen"].get(gens[0], {})
            tN = res["text_by_gen"].get(gens[-1], {})
            print(f"  [{label}] AUC eta2 g1={a1.get('eta_sq', float('nan')):.2f} -> "
                  f"g{gens[-1]}={aN.get('eta_sq', float('nan')):.2f} "
                  f"(run-mean spread g{gens[-1]}: "
                  f"{aN.get('run_mean_range', [np.nan, np.nan])[0]:.2f}-"
                  f"{aN.get('run_mean_range', [np.nan, np.nan])[1]:.2f}) | "
                  f"text eta2 g1={t1.get('eta_sq', float('nan')):.2f} -> "
                  f"g{gens[-1]}={tN.get('eta_sq', float('nan')):.2f} | "
                  f"cross-run sim g{gens[-1]}={tN.get('cross_run_sim_mean', float('nan')):.2f}")

    stats_path = os.path.join(args.output_dir, "across_run_divergence.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved stats: {stats_path}")


if __name__ == "__main__":
    main()
