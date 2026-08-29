#!/usr/bin/env python3
"""
Analyze run-by-run variation of strategies (text + coded).

For each experiment/model strategy_analysis.json this script asks:
  1. Are agents within a single run more similar to each other than to
     agents in other runs (at the same generation)?
       - coded strategies: pairwise distance = area between curves / 100
         (mean absolute donation-% gap over pair_prev_pct in [0, 100])
       - text strategies: cosine similarity of sentence embeddings
         (sentence-transformers, all-MiniLM-L6-v2)
     Tested per generation with a label-permutation test.
  2. Do the generations of one run group together?
       - each (run, generation) becomes one unit (mean curve / mean embedding)
       - nearest-neighbor same-run purity, silhouette score (labels = run),
         and a 2-D MDS map with same-run generations connected.
  3. How similar is the FINAL generation's strategy set to earlier
     generations of its own run vs the same generations of other runs?

Outputs per experiment/model (in --output_dir/<experiment>/<model>/):
  curves_by_run_grid.png        coded gen-mean curves, one panel per run
  final_gen_curves_overlay.png  final-gen run-mean curves overlaid
  mds_code.png / mds_text.png   (run, gen) units in 2-D, same run connected
  within_between_code.png       within- vs between-run curve distance by gen
  within_between_text.png       within- vs between-run text similarity by gen
  final_lineage.png             final-gen similarity to earlier gens, own vs other runs
  stats.json                    all numbers incl. permutation p-values

Usage:
    python figures/run_variation_analysis.py \
        [--analysis_root figures/reactivity_plots] \
        [--experiments neutral_baseline neutral_claude ...] \
        [--models claude gpt gemini] \
        [--output_dir figures/run_variation] \
        [--n_perm 1000]
"""

import json
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score


X_VALUES = np.linspace(0, 100, 201)
FINAL_GEN = 10

# Chart chrome (validated palette; see dataviz refs)
INK       = "#0b0b0b"
MUTED     = "#898781"
GRIDLINE  = "#e1e0d9"
SURFACE   = "#fcfcfb"
C_WITHIN  = "#2a78d6"   # categorical slot 1 (blue)
C_BETWEEN = "#1baf7a"   # categorical slot 2 (aqua)
# Sequential blues, one step per generation 1..10 (light -> dark)
GEN_COLORS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   MUTED,
    "axes.labelcolor":  INK,
    "text.color":       INK,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "grid.color":       GRIDLINE,
    "grid.linewidth":   0.8,
})


# ---------------------------------------------------------------- code eval
# (same sanitizing/eval conventions as plot_coded_strategies.py / compute_auc.py)

_UNICODE_OPS = [
    ('≥', '>='), ('≤', '<='), ('≠', '!='),
    ('×', '*'), ('÷', '/'), ('−', '-'),
]

def _sanitize_code(code):
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
    """Evaluate strategy_fn over X_VALUES. Returns np.array or None."""
    namespace = {}
    sanitized = _sanitize_code(code)
    try:
        exec(sanitized, namespace)
    except SyntaxError:
        try:
            exec(_fix_empty_blocks(sanitized), namespace)
        except Exception:
            return None
    except Exception:
        return None
    fn = namespace.get('strategy_fn')
    if fn is None:
        return None
    try:
        return np.array([float(np.clip(fn(x), 0, 100)) for x in X_VALUES])
    except Exception:
        return None


def curve_distance_matrix(curves):
    """Pairwise area between curves / x-range -> mean |gap| in donation pp."""
    diffs = np.abs(curves[:, None, :] - curves[None, :, :])
    return np.trapz(diffs, X_VALUES, axis=2) / (X_VALUES[-1] - X_VALUES[0])


# ---------------------------------------------------------------- loading

def load_agents(json_path):
    """Return list of dicts: run, generation, text, curve (curve may be None)."""
    with open(json_path) as f:
        data = json.load(f)
    agents = []
    code_cache = {}
    for r in data.get("agent_records", []):
        text = (r.get("strategy") or "").strip()
        if not text:
            continue
        curve = None
        code = r.get("strategy_code")
        if r.get("is_specified", False) and code:
            if code not in code_cache:
                code_cache[code] = eval_strategy(code)
            curve = code_cache[code]
        agents.append({
            "run": r["run"],
            "generation": r["generation"],
            "text": text,
            "curve": curve,
        })
    return agents


def embed_texts(agents, st_model):
    """Attach a unit-norm embedding to every agent (deduplicated encoding)."""
    unique = sorted(set(a["text"] for a in agents))
    embs = st_model.encode(unique, batch_size=64, show_progress_bar=False,
                           normalize_embeddings=True)
    lookup = {t: e for t, e in zip(unique, embs)}
    for a in agents:
        a["emb"] = lookup[a["text"]]


# ---------------------------------------------------------------- statistics

def within_between(matrix, run_labels, higher_is_closer):
    """Mean within-run vs between-run pairwise value + permutation p.

    matrix: symmetric pairwise value matrix (distance or similarity).
    Returns (within_mean, between_mean, p_value_placeholder_masks).
    """
    labels = np.asarray(run_labels)
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(labels), dtype=bool)
    within_mask = same & off_diag
    between_mask = ~same
    if within_mask.sum() == 0 or between_mask.sum() == 0:
        return None
    return matrix[within_mask].mean(), matrix[between_mask].mean()


def permutation_test(matrix, run_labels, higher_is_closer, n_perm, rng):
    """P-value for 'within-run values are closer than between-run'."""
    labels = np.asarray(run_labels)
    off_diag = ~np.eye(len(labels), dtype=bool)

    def stat(lbl):
        same = lbl[:, None] == lbl[None, :]
        w = matrix[same & off_diag].mean()
        b = matrix[~same].mean()
        return (w - b) if higher_is_closer else (b - w)

    observed = stat(labels)
    count = 0
    for _ in range(n_perm):
        if stat(rng.permutation(labels)) >= observed:
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def nn_purity(dist, run_labels):
    """Fraction of units whose nearest neighbor is from the same run."""
    labels = np.asarray(run_labels)
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nearest = d.argmin(axis=1)
    purity = float((labels[nearest] == labels).mean())
    # chance: pick uniformly among the other units
    counts = np.array([np.sum(labels == l) for l in labels])
    chance = float(((counts - 1) / (len(labels) - 1)).mean())
    return purity, chance


# ---------------------------------------------------------------- figures

def _style_axis(ax):
    ax.grid(True, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_curves_by_run(units, runs, gens, out_path, title):
    n = len(runs)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for idx, run in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        for g in gens:
            key = (run, g)
            if key in units and units[key]["curve"] is not None:
                ax.plot(X_VALUES, units[key]["curve"],
                        color=GEN_COLORS[(g - 1) % len(GEN_COLORS)], lw=1.4)
        ax.set_title(f"run {idx + 1}", fontsize=9, color=MUTED)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        _style_axis(ax)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    sm = plt.cm.ScalarMappable(
        cmap=plt.matplotlib.colors.ListedColormap(GEN_COLORS),
        norm=plt.matplotlib.colors.BoundaryNorm(np.arange(0.5, len(gens) + 1.5), len(gens)))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01,
                        ticks=range(1, len(gens) + 1))
    cbar.set_label("generation", color=INK)
    cbar.ax.tick_params(color=MUTED, labelcolor=MUTED)
    fig.suptitle(title, fontsize=13, color=INK)
    fig.supxlabel("pair's donation % in previous round", fontsize=10, color=INK)
    fig.supylabel("my donation %", fontsize=10, color=INK)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_final_overlay(units, runs, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    curves = []
    for run in runs:
        key = (run, FINAL_GEN)
        if key in units and units[key]["curve"] is not None:
            ax.plot(X_VALUES, units[key]["curve"], color=C_WITHIN, alpha=0.30, lw=1.4)
            curves.append(units[key]["curve"])
    if curves:
        ax.plot(X_VALUES, np.mean(curves, axis=0), color=INK, lw=2.5,
                label=f"grand mean ({len(curves)} runs)")
    ax.set_xlabel("pair's donation % in previous round", fontsize=11)
    ax.set_ylabel("my donation %", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    _style_axis(ax)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_mds(dist, unit_keys, runs, out_path, title, subtitle):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
              n_init=4, normalized_stress="auto")
    xy = mds.fit_transform(dist)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    # connect the generations of each run in order (identity via connectivity,
    # not 20 categorical hues)
    for run in runs:
        idxs = [i for i, (r, _) in enumerate(unit_keys) if r == run]
        idxs.sort(key=lambda i: unit_keys[i][1])
        ax.plot(xy[idxs, 0], xy[idxs, 1], color=MUTED, lw=0.7, alpha=0.55, zorder=1)
    gens = np.array([g for _, g in unit_keys])
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=gens,
                    cmap=plt.matplotlib.colors.ListedColormap(GEN_COLORS),
                    vmin=0.5, vmax=len(GEN_COLORS) + 0.5,
                    s=42, edgecolors=SURFACE, linewidths=0.8, zorder=2)
    cbar = fig.colorbar(sc, ax=ax, ticks=range(1, len(GEN_COLORS) + 1), fraction=0.04)
    cbar.set_label("generation", color=INK)
    cbar.ax.tick_params(color=MUTED, labelcolor=MUTED)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRIDLINE)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_within_between(per_gen, gens, out_path, title, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, color, label in ((("within"), C_WITHIN, "within run"),
                               (("between"), C_BETWEEN, "between runs")):
        means = [per_gen[g][kind] for g in gens]
        ax.plot(gens, means, color=color, lw=2.2, marker="o", ms=5, label=label)
    for g in gens:
        if per_gen[g].get("p") is not None and per_gen[g]["p"] < 0.05:
            ymax = max(per_gen[g]["within"], per_gen[g]["between"])
            ax.annotate("*", (g, ymax), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=11, color=INK)
    ax.set_xlabel("generation", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(gens)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    _style_axis(ax)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_final_lineage(own, other, gens, out_path, title, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gens, [np.mean(own[g]) for g in gens], color=C_WITHIN, lw=2.2,
            marker="o", ms=5, label="same run")
    ax.fill_between(gens,
                    [np.mean(own[g]) - np.std(own[g]) for g in gens],
                    [np.mean(own[g]) + np.std(own[g]) for g in gens],
                    color=C_WITHIN, alpha=0.15)
    ax.plot(gens, [np.mean(other[g]) for g in gens], color=C_BETWEEN, lw=2.2,
            marker="o", ms=5, label="other runs")
    ax.fill_between(gens,
                    [np.mean(other[g]) - np.std(other[g]) for g in gens],
                    [np.mean(other[g]) + np.std(other[g]) for g in gens],
                    color=C_BETWEEN, alpha=0.15)
    ax.set_xlabel("earlier generation compared against", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(gens)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    _style_axis(ax)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- pipeline

def analyze(json_path, out_dir, label, st_model, n_perm, rng):
    os.makedirs(out_dir, exist_ok=True)
    agents = load_agents(json_path)
    if not agents:
        print(f"  [{label}] no usable records — skipping")
        return None
    embed_texts(agents, st_model)

    runs = sorted(set(a["run"] for a in agents))
    gens = sorted(set(a["generation"] for a in agents))
    n_code = sum(a["curve"] is not None for a in agents)
    print(f"  [{label}] {len(agents)} agents | {len(runs)} runs | "
          f"gens {gens[0]}-{gens[-1]} | {n_code} coded curves")

    stats = {"label": label, "n_agents": len(agents), "n_runs": len(runs),
             "n_coded": n_code, "generations": gens}

    # ---- (run, gen) units: mean curve + mean embedding
    units = {}
    for run in runs:
        for g in gens:
            group = [a for a in agents if a["run"] == run and a["generation"] == g]
            if not group:
                continue
            curves = [a["curve"] for a in group if a["curve"] is not None]
            emb = np.mean([a["emb"] for a in group], axis=0)
            emb /= np.linalg.norm(emb)
            units[(run, g)] = {
                "curve": np.mean(curves, axis=0) if curves else None,
                "emb": emb,
            }

    # ---- figures 1+2: coded curves per run
    fig_curves_by_run(units, runs, gens,
                      os.path.join(out_dir, "curves_by_run_grid.png"),
                      f"{label} — generation-mean coded strategies per run")
    fig_final_overlay(units, runs,
                      os.path.join(out_dir, "final_gen_curves_overlay.png"),
                      f"{label} — final-generation run-mean coded strategies")

    # ---- agent-level within vs between run, per generation
    per_gen_code, per_gen_text = {}, {}
    for g in gens:
        group = [a for a in agents if a["generation"] == g]
        labels = [a["run"] for a in group]
        # text
        E = np.array([a["emb"] for a in group])
        S = E @ E.T
        wb = within_between(S, labels, higher_is_closer=True)
        if wb:
            _, p = permutation_test(S, labels, True, n_perm, rng)
            per_gen_text[g] = {"within": float(wb[0]), "between": float(wb[1]), "p": p}
        # code
        coded = [a for a in group if a["curve"] is not None]
        labels_c = [a["run"] for a in coded]
        if len(set(labels_c)) > 1 and len(coded) > 2:
            D = curve_distance_matrix(np.array([a["curve"] for a in coded]))
            wb = within_between(D, labels_c, higher_is_closer=False)
            if wb:
                _, p = permutation_test(D, labels_c, False, n_perm, rng)
                per_gen_code[g] = {"within": float(wb[0]), "between": float(wb[1]), "p": p}

    fig_within_between(per_gen_code, sorted(per_gen_code),
                       os.path.join(out_dir, "within_between_code.png"),
                       f"{label} — coded-strategy distance (area between curves)",
                       "mean pairwise |gap| (donation pp)")
    fig_within_between(per_gen_text, sorted(per_gen_text),
                       os.path.join(out_dir, "within_between_text.png"),
                       f"{label} — text-strategy semantic similarity",
                       "mean pairwise cosine similarity")
    stats["within_between_code_by_gen"] = per_gen_code
    stats["within_between_text_by_gen"] = per_gen_text

    # ---- unit-level clustering: do a run's generations group together?
    unit_keys = sorted(units.keys())
    unit_runs = [r for r, _ in unit_keys]

    coded_keys = [k for k in unit_keys if units[k]["curve"] is not None]
    coded_runs = [r for r, _ in coded_keys]
    D_units = curve_distance_matrix(np.array([units[k]["curve"] for k in coded_keys]))
    E_units = np.array([units[k]["emb"] for k in unit_keys])
    T_units = np.clip(1.0 - E_units @ E_units.T, 0, None)
    np.fill_diagonal(T_units, 0.0)

    stats["run_grouping"] = {}
    for name, dist, keys, labels in (("code", D_units, coded_keys, coded_runs),
                                     ("text", T_units, unit_keys, unit_runs)):
        purity, chance = nn_purity(dist, labels)
        sil = float(silhouette_score(dist, labels, metric="precomputed"))
        stats["run_grouping"][name] = {"nn_same_run_purity": purity,
                                       "nn_chance": chance, "silhouette": sil}
        fig_mds(dist, keys, runs,
                os.path.join(out_dir, f"mds_{name}.png"),
                f"{label} — MDS of (run, generation) units [{name}]",
                f"lines connect generations of one run · NN same-run purity "
                f"{purity:.0%} (chance {chance:.0%}) · silhouette {sil:.2f}")

    # ---- final generation vs earlier generations, own run vs other runs
    earlier = [g for g in gens if g != FINAL_GEN]
    own_t, oth_t = {g: [] for g in earlier}, {g: [] for g in earlier}
    own_c, oth_c = {g: [] for g in earlier}, {g: [] for g in earlier}
    for run in runs:
        fin = units.get((run, FINAL_GEN))
        if fin is None:
            continue
        for g in earlier:
            for run2 in runs:
                u = units.get((run2, g))
                if u is None:
                    continue
                sim = float(fin["emb"] @ u["emb"])
                (own_t if run2 == run else oth_t)[g].append(sim)
                if fin["curve"] is not None and u["curve"] is not None:
                    d = float(np.trapz(np.abs(fin["curve"] - u["curve"]), X_VALUES)
                              / (X_VALUES[-1] - X_VALUES[0]))
                    (own_c if run2 == run else oth_c)[g].append(d)

    fig_final_lineage(own_t, oth_t, earlier,
                      os.path.join(out_dir, "final_lineage_text.png"),
                      f"{label} — final-gen text similarity to earlier generations",
                      "cosine similarity to final gen")
    fig_final_lineage(own_c, oth_c, earlier,
                      os.path.join(out_dir, "final_lineage_code.png"),
                      f"{label} — final-gen coded-strategy distance to earlier generations",
                      "curve distance to final gen (donation pp)")
    stats["final_lineage"] = {
        "text_own_by_gen":   {g: float(np.mean(own_t[g])) for g in earlier if own_t[g]},
        "text_other_by_gen": {g: float(np.mean(oth_t[g])) for g in earlier if oth_t[g]},
        "code_own_by_gen":   {g: float(np.mean(own_c[g])) for g in earlier if own_c[g]},
        "code_other_by_gen": {g: float(np.mean(oth_c[g])) for g in earlier if oth_c[g]},
    }

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run-by-run strategy variation analysis.")
    parser.add_argument("--analysis_root", default="figures/reactivity_plots")
    parser.add_argument("--experiments", nargs="+",
                        default=["neutral_baseline", "neutral_claude", "neutral_gemini",
                                 "neutral_gpt", "neutral_deepseek"])
    parser.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    parser.add_argument("--output_dir", default="figures/run_variation")
    parser.add_argument("--n_perm", type=int, default=1000)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    rng = np.random.default_rng(0)

    all_stats = {}
    for exp in args.experiments:
        for model in args.models:
            json_path = os.path.join(args.analysis_root, exp, model,
                                     f"strategy_analysis_{model}.json")
            if not os.path.exists(json_path):
                print(f"  [{exp}/{model}] {json_path} not found — skipping")
                continue
            out_dir = os.path.join(args.output_dir, exp, model)
            stats = analyze(json_path, out_dir, f"{exp}/{model}",
                            st_model, args.n_perm, rng)
            if stats:
                all_stats[f"{exp}/{model}"] = stats

    summary_path = os.path.join(args.output_dir, "summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # console summary
    print("\n=== Run-grouping summary (final gen within/between + unit clustering) ===")
    hdr = f"{'experiment/model':34s} {'code NN%':>8s} {'code sil':>9s} {'text NN%':>8s} {'text sil':>9s} {'W<B code p(g10)':>16s} {'W>B text p(g10)':>16s}"
    print(hdr)
    for key, s in all_stats.items():
        rg = s["run_grouping"]
        pc = s["within_between_code_by_gen"].get(FINAL_GEN, {}).get("p")
        pt = s["within_between_text_by_gen"].get(FINAL_GEN, {}).get("p")
        print(f"{key:34s} {rg['code']['nn_same_run_purity']*100:7.0f}% "
              f"{rg['code']['silhouette']:9.2f} "
              f"{rg['text']['nn_same_run_purity']*100:7.0f}% "
              f"{rg['text']['silhouette']:9.2f} "
              f"{pc if pc is not None else float('nan'):16.3f} "
              f"{pt if pt is not None else float('nan'):16.3f}")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
