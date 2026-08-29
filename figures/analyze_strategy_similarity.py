#!/usr/bin/env python3
"""
Within-run strategy homogeneity, measured two ways: lexically and semantically.

For one run at one generation the 12 agents hold a set S of strategy texts.
This takes every unordered pair in S (66 pairs at 12 agents), scores the two
texts for similarity, and histograms the result. A distribution pressed up
against 1.0 means the population converged on one strategy; a broad or
low-centred one means the agents are still saying different things.

Pairs are only ever formed WITHIN a run — two agents from different runs are
never compared, since a cross-run pair says nothing about convergence. Pairs
from all runs of a model are then pooled into one histogram.

METRICS  (both are similarities in [0, 1]; higher = more homogeneous)

  lexical   1 - lev(w_a, w_b) / max(|w_a|, |w_b|)

            w_x = the text lowercased and split into words by [a-z0-9']+ ;
            lev = Levenshtein distance over those WORD sequences (unit cost
            per word insertion, deletion, or substitution). Dividing by the
            longer token count bounds it in [0, 1] and stops the score from
            tracking strategy length, which grows several-fold across
            generations. 1.0 = identical word sequences.

  semantic  (e_a . e_b) / (||e_a|| ||e_b||)

            e_x = OpenAI text-embedding-3-small embedding of the text at
            dimensions=512. The API returns L2-normalised vectors, so this
            is just the dot product. Captures paraphrase: two strategies
            that say "match my partner's last donation" and "reciprocate
            whatever they gave me" score high here and low on lexical.

  words     raw lev(w_a, w_b), reported as a DISTANCE not a similarity, for
            the appendix note on why normalisation is necessary.

Embeddings are cached in --cache (sha1 of the text -> vector) so re-runs and
overlapping conditions cost nothing.

SOURCES  (--source auto picks whichever is present)

  raw       {data_dir}/{model}/**/all_strategies_*.json — the simulation's own
            per-generation strategy dump. Preferred: no classifier pass needed.
  analysis  {data_dir}/{model}/strategy_analysis_{model}.json — the coded-
            strategy analysis JSONs, which also carry the raw strategy text.

Outputs (to --output_dir, default figures/strategy_diversity/):
  {label}_similarity_hist.png     one row per metric, one column per model
  {label}_similarity_stats.json   formulas, per-cell stats, significance tests

Usage:
    python figures/analyze_strategy_similarity.py \
        [--data_dir figures/reactivity_plots/neutral_baseline] \
        [--label neutral_baseline] \
        [--source auto|raw|analysis] \
        [--metrics lexical semantic] \
        [--generations 1 10] \
        [--output_dir figures/strategy_diversity]
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_gen0_variants import (style_axes, MODEL_LABELS,
                                   SURFACE, INK, INK_2, MUTED)

MODELS = ["claude", "gpt", "gemini"]
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}
GEN0_COLOR = "#888888"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512
EMBED_BATCH = 128
MAX_CHARS = 20000          # ~5k tokens, well under the 8191-token input limit

_WORD_RE = re.compile(r"[a-z0-9']+")

# Printed in the run header and written verbatim into the stats JSON.
FORMULAS = {
    "lexical": {
        "name": "Normalised word-level lexical similarity",
        "formula": "sim_lex(a,b) = 1 - lev(w_a, w_b) / max(|w_a|, |w_b|)",
        "definition": (
            "w_x = text lowercased, tokenised by [a-z0-9']+ ; lev = Levenshtein "
            "distance over word sequences (unit cost per word insertion, "
            "deletion, substitution). Range [0, 1]; 1.0 = identical word "
            "sequences. Order-sensitive, no semantics."),
        "range": [0, 1],
        "higher_is": "more homogeneous",
    },
    "semantic": {
        "name": "Embedding cosine similarity",
        "formula": "sim_sem(a,b) = (e_a . e_b) / (||e_a|| ||e_b||)",
        "definition": (
            f"e_x = OpenAI {EMBED_MODEL} embedding of the text at "
            f"dimensions={EMBED_DIMS} (returned L2-normalised, so this reduces "
            "to a dot product). Captures paraphrase; insensitive to wording."),
        "range": [0, 1],
        "higher_is": "more homogeneous",
    },
    "words": {
        "name": "Raw word-level edit distance",
        "formula": "d_words(a,b) = lev(w_a, w_b)",
        "definition": (
            "Unnormalised word-level Levenshtein distance. A DISTANCE, not a "
            "similarity: it grows with strategy length, so it conflates "
            "verbosity with divergence. Reported only to justify normalising."),
        "range": None,
        "higher_is": "more diverse",
    },
}

METRIC_AXIS = {
    "lexical": "Lexical similarity",
    "semantic": "Semantic similarity (cosine)",
    "words": "Word-level edit distance (words)",
}


def gen_label(g):
    """Data generation number -> the label used in the paper (gen 1 = t=0)."""
    return "Generation 0" if g == 1 else f"Generation {g}"


def tokenize(text):
    return _WORD_RE.findall(text.lower())


def levenshtein(a, b):
    """Edit distance between two token sequences (two-row DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, tok_a in enumerate(a, 1):
        cur = [i]
        for j, tok_b in enumerate(b, 1):
            cur.append(min(prev[j] + 1,
                           cur[j - 1] + 1,
                           prev[j - 1] + (tok_a != tok_b)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def detect_source(data_dir, models):
    """'raw' if any model has all_strategies_*.json, else 'analysis'."""
    for m in models:
        if glob.glob(os.path.join(data_dir, m, "**", "all_strategies_*.json"),
                     recursive=True):
            return "raw"
    return "analysis"


_TS_RE = re.compile(r'_(\d{8}_\d{6})\.json$')


def load_raw(data_dir, model, generation):
    """{run_id: [text, ...]} from the simulation's all_strategies dumps."""
    paths = sorted(glob.glob(
        os.path.join(data_dir, model, "**", "all_strategies_*.json"),
        recursive=True))
    by_run = {}
    for p in paths:
        m = _TS_RE.search(os.path.basename(p))
        run = m.group(1) if m else os.path.basename(p)
        with open(p) as f:
            gens = json.load(f)
        for gen_list in gens:
            for rec in gen_list:
                # 'generation' is a string in these files.
                if int(rec.get("generation", -1)) != generation:
                    continue
                text = (rec.get("strategy") or "").strip()
                if text:
                    by_run.setdefault(run, []).append(text)
    return by_run


def load_analysis(data_dir, model, generation):
    """{run_id: [text, ...]} from strategy_analysis_{model}.json."""
    path = os.path.join(data_dir, model, f"strategy_analysis_{model}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        records = json.load(f).get("agent_records", [])
    by_run = {}
    for rec in records:
        if rec.get("generation") != generation:
            continue
        text = (rec.get("strategy") or "").strip()
        if text:
            by_run.setdefault(rec["run"], []).append(text)
    return by_run


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _load_api_key():
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = \
                    line.split("=", 1)[1].strip().strip('"').strip("'")
                return


def _key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """sha1(text) -> unit vector, persisted as a single compressed .npz."""

    def __init__(self, path):
        self.path = path
        self.vecs = {}
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            keys, mat = z["keys"], z["vecs"]
            self.vecs = {str(k): mat[i] for i, k in enumerate(keys)}
            print(f"  embedding cache: {len(self.vecs)} vectors loaded")
        self._client = None

    def client(self):
        if self._client is None:
            import openai
            _load_api_key()
            self._client = openai.OpenAI()
        return self._client

    def embed(self, texts):
        """Return {text: vector}, fetching only what is not already cached."""
        missing = sorted({t for t in texts if _key(t) not in self.vecs})
        if missing:
            print(f"  embedding {len(missing)} new strategy text(s) "
                  f"via {EMBED_MODEL} (dims={EMBED_DIMS})")
            client = self.client()
            for i in range(0, len(missing), EMBED_BATCH):
                chunk = missing[i:i + EMBED_BATCH]
                resp = client.embeddings.create(
                    model=EMBED_MODEL, dimensions=EMBED_DIMS,
                    input=[t[:MAX_CHARS] for t in chunk])
                for text, item in zip(chunk, resp.data):
                    v = np.asarray(item.embedding, dtype=np.float32)
                    self.vecs[_key(text)] = v / np.linalg.norm(v)
                print(f"    {min(i + EMBED_BATCH, len(missing))}/{len(missing)}")
            self.save()
        return {t: self.vecs[_key(t)] for t in texts}

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        keys = list(self.vecs)
        np.savez_compressed(self.path, keys=np.array(keys),
                            vecs=np.stack([self.vecs[k] for k in keys]))


# ---------------------------------------------------------------------------
# Pairwise scoring
# ---------------------------------------------------------------------------
def pairwise(by_run, metric, cache=None):
    """All within-run pairwise scores, plus one mean per run and the mean
    number of distinct strategy texts per run."""
    if metric == "semantic":
        all_texts = [t for texts in by_run.values() for t in texts]
        emb = cache.embed(all_texts) if all_texts else {}

    scores, run_means, n_pairs, n_distinct = [], [], [], []
    for _run, texts in sorted(by_run.items()):
        if len(texts) < 2:
            continue
        n_distinct.append(len(set(texts)))
        vals = []
        if metric == "semantic":
            mat = np.stack([emb[t] for t in texts])
            sim = mat @ mat.T
            iu = np.triu_indices(len(texts), k=1)
            vals = list(np.clip(sim[iu], 0.0, 1.0))
        else:
            toks = [tokenize(t) for t in texts]
            for a, b in combinations(toks, 2):
                d = levenshtein(a, b)
                if metric == "lexical":
                    denom = max(len(a), len(b))
                    vals.append(1.0 - d / denom if denom else 1.0)
                else:                       # words
                    vals.append(float(d))
        scores.extend(vals)
        run_means.append(float(np.mean(vals)))
        n_pairs.append(len(vals))
    return (np.array(scores), np.array(run_means), n_pairs,
            np.array(n_distinct))


def cell_summary(s, run_means, n_runs, n_pairs, n_distinct, metric):
    top = 1.0 if metric in ("lexical", "semantic") else None
    return {
        "n_runs": int(n_runs),
        "n_pairs": int(s.size),
        "pairs_per_run": sorted(set(n_pairs)),
        "mean": float(s.mean()),
        "median": float(np.median(s)),
        "std": float(s.std()),
        "p10": float(np.percentile(s, 10)),
        "p90": float(np.percentile(s, 90)),
        "frac_identical": float((s >= 0.99999).mean()) if top else None,
        "mean_distinct_strategies_per_run": float(n_distinct.mean()),
        "run_mean_of_means": float(run_means.mean()),
        "run_means": [float(m) for m in run_means],
    }


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    parser = argparse.ArgumentParser(
        description="Within-run strategy similarity (lexical + semantic), "
                    "generation 0 vs generation 10.")
    parser.add_argument("--data_dir",
                        default="figures/reactivity_plots/neutral_baseline")
    parser.add_argument("--label", default=None,
                        help="Output filename tag (default: basename of data_dir)")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "raw", "analysis"])
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--metrics", nargs="+", default=["lexical", "semantic"],
                        choices=["lexical", "semantic", "words"])
    parser.add_argument("--generations", nargs="+", type=int, default=[1, 10])
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--output_dir", default="figures/strategy_diversity")
    parser.add_argument("--cache",
                        default="figures/strategy_diversity/_embedding_cache.npz")
    args = parser.parse_args()

    label = args.label or os.path.basename(os.path.normpath(args.data_dir))
    source = (detect_source(args.data_dir, args.models)
              if args.source == "auto" else args.source)
    loader = load_raw if source == "raw" else load_analysis
    cache = (EmbeddingCache(args.cache) if "semantic" in args.metrics else None)

    # ── Definitions, printed with the results ────────────────────────────────
    print(f"\n{'=' * 74}")
    print(f"Within-run strategy similarity — {label}")
    print(f"{'=' * 74}")
    print(f"source   : {source}  ({args.data_dir})")
    print(f"pairs    : every unordered pair of the agents' strategies WITHIN a "
          f"run, pooled\n           across runs; cross-run pairs are never formed")
    for m in args.metrics:
        f = FORMULAS[m]
        print(f"\n{m:9}: {f['name']}")
        print(f"           {f['formula']}")
        for line in _wrap(f["definition"], 62):
            print(f"           {line}")

    data, summary, tests = {}, {}, {}
    for metric in args.metrics:
        data[metric], summary[metric], tests[metric] = {}, {}, {}
        print(f"\n{'-' * 74}\n{metric}\n{'-' * 74}")
        print(f"{'model':<8} {'gen':>4} {'runs':>5} {'pairs':>6} {'mean':>8} "
              f"{'median':>8} {'distinct/run':>13} {'%identical':>11}")
        for model in args.models:
            data[metric][model], summary[metric][model] = {}, {}
            for gen in args.generations:
                by_run = loader(args.data_dir, model, gen)
                if not by_run:
                    print(f"{model:<8} {gen:>4}   no strategies — skipped")
                    continue
                s, rm, npair, ndist = pairwise(by_run, metric, cache)
                if not s.size:
                    continue
                data[metric][model][gen] = (s, rm)
                cs = cell_summary(s, rm, len(by_run), npair, ndist, metric)
                summary[metric][model][gen] = cs
                ident = ("     n/a" if cs["frac_identical"] is None
                         else f"{100 * cs['frac_identical']:9.1f}%")
                print(f"{model:<8} {gen:>4} {cs['n_runs']:>5} {cs['n_pairs']:>6} "
                      f"{cs['mean']:>8.3f} {cs['median']:>8.3f} "
                      f"{cs['mean_distinct_strategies_per_run']:>13.1f} {ident:>11}")

            g_a, g_b = args.generations[0], args.generations[-1]
            if g_a in data[metric][model] and g_b in data[metric][model]:
                rm_a = data[metric][model][g_a][1]
                rm_b = data[metric][model][g_b][1]
                t = {}
                if rm_a.size > 1 and rm_b.size > 1:
                    u, p = sps.mannwhitneyu(rm_a, rm_b, alternative="two-sided")
                    t["run_mean_mannwhitney"] = {"U": float(u), "p": float(p)}
                    st, p2 = sps.ttest_ind(rm_a, rm_b, equal_var=False)
                    t["run_mean_welch_t"] = {"t": float(st), "p": float(p2)}
                    # Cliff's delta from U: effect size on run means.
                    n1, n2 = rm_a.size, rm_b.size
                    t["cliffs_delta_runs"] = float(
                        2 * t["run_mean_mannwhitney"]["U"] / (n1 * n2) - 1)
                t["delta_mean"] = float(rm_b.mean() - rm_a.mean())
                tests[metric][model] = t

    plotted = [m for m in args.models
               if any(data[k].get(m) for k in args.metrics)]
    if not plotted:
        print("\nNo data to plot.")
        return

    # ── Significant results summary ──────────────────────────────────────────
    print(f"\n{'=' * 74}")
    print(f"SIGNIFICANT RESULTS — generation {args.generations[0]} vs "
          f"{args.generations[-1]}, run-level Mann-Whitney (n = runs)")
    print(f"{'=' * 74}")
    print(f"{'metric':<9} {'model':<8} {'gen0':>7} {'gen10':>7} {'delta':>8} "
          f"{'cliff d':>8} {'p':>12}  sig")
    for metric in args.metrics:
        for model in plotted:
            t = tests[metric].get(model)
            if not t or "run_mean_mannwhitney" not in t:
                continue
            a = summary[metric][model][args.generations[0]]["run_mean_of_means"]
            b = summary[metric][model][args.generations[-1]]["run_mean_of_means"]
            p = t["run_mean_mannwhitney"]["p"]
            print(f"{metric:<9} {model:<8} {a:>7.3f} {b:>7.3f} "
                  f"{t['delta_mean']:>+8.3f} {t['cliffs_delta_runs']:>8.2f} "
                  f"{fmt_p(p):>12}  {sig_stars(p)}")

    # ── Figure: one row per metric, one column per model ─────────────────────
    nrow, ncol = len(args.metrics), len(plotted)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.3 * nrow),
                             sharey="row", facecolor=SURFACE, squeeze=False)
    for row, metric in enumerate(args.metrics):
        vals = [s for m in plotted for s, _ in data[metric].get(m, {}).values()]
        hi = 1.0 if metric in ("lexical", "semantic") else max(
            (s.max() for s in vals), default=1.0)
        # Cosine similarity between two strategies for the same game never
        # approaches 0, so anchoring the semantic row at 0 would leave most of
        # the panel empty. Start it just below the smallest observed pair.
        if metric == "semantic" and vals:
            lo = np.floor(min(s.min() for s in vals) * 20) / 20
        else:
            lo = 0.0
        edges = np.linspace(lo, hi, args.bins + 1)
        for col, model in enumerate(plotted):
            ax = axes[row][col]
            style_axes(ax, grid_axis="y")
            for i, gen in enumerate(args.generations):
                cell = data[metric].get(model, {}).get(gen)
                if cell is None:
                    continue
                s, _ = cell
                color = GEN0_COLOR if i == 0 else MODEL_COLORS[model]
                ax.hist(s, bins=edges, density=True, color=color,
                        alpha=0.45 if i == 0 else 0.65, zorder=2,
                        label=f"{gen_label(gen)} (n={s.size})")
                ax.axvline(np.median(s), color=color, linestyle="--",
                           linewidth=1.4, zorder=3)
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=11.5, color=INK)
            ax.set_xlabel(METRIC_AXIS[metric], fontsize=9.5, color=INK_2)
            ax.set_xlim(lo, hi)
            ax.legend(fontsize=8, loc="upper left", frameon=False,
                      labelcolor=INK_2)
        axes[row][0].set_ylabel("Density of pairs", fontsize=9.5, color=INK_2)

    fig.suptitle(f"Within-run strategy similarity — {label}",
                 fontsize=13, color=INK)
    fig.text(0.99, 0.005,
             "Every unordered pair of the agents' strategies within a run, "
             "pooled across runs · dashed line: median",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, f"{label}_similarity_hist.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"\nSaved: {out_png}")

    out_json = os.path.join(args.output_dir, f"{label}_similarity_stats.json")
    with open(out_json, "w") as f:
        json.dump({
            "label": label,
            "data_dir": args.data_dir,
            "source": source,
            "generations": args.generations,
            "note": "generation 1 in the data = initial state (t=0); pairs are "
                    "formed within a run only. Significance tests use one "
                    "value per run (the run's mean) because the 66 pairs "
                    "within a run share agents and are not independent.",
            "metrics": {m: FORMULAS[m] for m in args.metrics},
            "summary": summary, "tests": tests,
        }, f, indent=2)
    print(f"Saved: {out_json}")


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    main()
