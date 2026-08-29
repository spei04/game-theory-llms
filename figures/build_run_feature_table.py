#!/usr/bin/env python3
"""
Step 0 of the variability-prediction analysis: build one run-level table.

One row per run (prompt variant x executor model x run), with:

Identity        : experiment (prompt variant), model (executor)
Initial state   : gen-1 features only (no leakage into later outcomes)
    gen1_auc_mean / gen1_auc_std      coded-strategy AUC across the 12 agents
    gen1_first_round_mean             mean strategy_fn(None) donation %
    gen1_semantic_diversity           mean pairwise cosine DISTANCE between
                                      the initial strategy-text embeddings
    gen1_frac_underspecified          fraction of agents not fully specified
    gen1_text_len_mean                mean strategy text length (words)
Early dynamics  : gens 1-3
    coop_gen1, dcoop_12, dcoop_13     cooperation level + early changes
    dauc_12                           gen-2 minus gen-1 mean AUC
Outcomes        :
    coop_gen1..coop_gen10             cooperation rate per generation
    final_coop, mean_coop             last-generation / all-gen mean
    volatility_detrended              std of residuals around a linear
                                      coop-vs-generation fit (ddof=2)
    collapsed                         final_coop < 10%

Sources: simulations/<exp>/<model>/Final_Accum*.json (cooperation) and
figures/reactivity_plots/<exp>/<model>/strategy_analysis_<model>.json
(strategy text/code), joined on the run timestamp.

Output: figures/run_variation/run_feature_table.csv

Usage:
    python figures/build_run_feature_table.py \
        [--experiments neutral_baseline neutral_claude ...] \
        [--models claude gpt gemini] \
        [--output figures/run_variation/run_feature_table.csv]
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import eval_strategy, auc_normalised

_TS_RE = re.compile(r'_(\d{8}_\d{6})\.json$')
MAX_GEN = 10


def run_cooperation_by_gen(accum_path):
    """gen -> mean donation % across all donor rounds (repo convention)."""
    with open(accum_path) as f:
        data = json.load(f)
    donation_pcts = defaultdict(list)
    for r in data.get("agents_data", []):
        if not r.get("is_donor"):
            continue
        donated = r.get("donated", 0) or 0
        remaining = r.get("resources", 0) or 0
        denom = donated + remaining
        if denom > 0:
            donation_pcts[r["current_generation"]].append(100.0 * donated / denom)
    return {g: float(np.mean(v)) for g, v in donation_pcts.items() if v}


def eval_first_round(code):
    """strategy_fn(None) donation %, or None."""
    ns = {}
    try:
        exec(code, ns)
        fn = ns.get("strategy_fn")
        if fn is None:
            return None
        return float(np.clip(fn(None), 0, 100))
    except Exception:
        return None


def load_analysis(analysis_root, exp, model):
    """run -> gen -> list of agent records (text, code, is_specified)."""
    path = os.path.join(analysis_root, exp, model, f"strategy_analysis_{model}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        records = json.load(f).get("agent_records", [])
    out = defaultdict(lambda: defaultdict(list))
    for r in records:
        out[r["run"]][r["generation"]].append(r)
    return out


def gen_auc_stats(recs, curve_cache):
    """(auc_mean, auc_std, first_round_mean) for one run-generation."""
    aucs, firsts = [], []
    for r in recs:
        code = r.get("strategy_code")
        if not (r.get("is_specified") and code):
            continue
        if code not in curve_cache:
            ys = eval_strategy(code)
            curve_cache[code] = (auc_normalised(ys) if ys is not None else None,
                                 eval_first_round(code))
        auc, fr = curve_cache[code]
        if auc is not None:
            aucs.append(auc)
        if fr is not None:
            firsts.append(fr)
    return (float(np.mean(aucs)) if aucs else np.nan,
            float(np.std(aucs, ddof=1)) if len(aucs) > 1 else np.nan,
            float(np.mean(firsts)) if firsts else np.nan)


def semantic_diversity(texts, emb_lookup):
    """Mean pairwise cosine distance between the texts' embeddings."""
    E = np.array([emb_lookup[t] for t in texts])
    if len(E) < 2:
        return np.nan
    S = E @ E.T
    iu = np.triu_indices(len(E), k=1)
    return float(np.mean(1.0 - S[iu]))


def main():
    ap = argparse.ArgumentParser(description="Build the run-level feature table.")
    ap.add_argument("--sim_root", default="simulations")
    ap.add_argument("--analysis_root", default="figures/reactivity_plots")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline", "neutral_claude", "neutral_gemini",
                             "neutral_gpt", "neutral_deepseek"])
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--output", default="figures/run_variation/run_feature_table.csv")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    rows = []
    curve_cache = {}
    for exp in args.experiments:
        for model in args.models:
            sim_dir = os.path.join(args.sim_root, exp, model)
            accum_paths = sorted(glob.glob(os.path.join(sim_dir, "Final_Accum*.json")),
                                 key=lambda p: (_TS_RE.search(p).group(1)
                                                if _TS_RE.search(p) else p))
            analysis = load_analysis(args.analysis_root, exp, model)
            if not accum_paths:
                print(f"  [{exp}/{model}] no Final_Accum files — skipped")
                continue

            # embed all gen-1 strategy texts of this cell in one batch
            gen1_texts = sorted({(r.get("strategy") or "").strip()
                                 for recs in analysis.values() for r in recs.get(1, [])
                                 if (r.get("strategy") or "").strip()})
            embs = st_model.encode(gen1_texts, batch_size=64,
                                   show_progress_bar=False, normalize_embeddings=True)
            emb_lookup = dict(zip(gen1_texts, embs))

            n_rows = 0
            for path in accum_paths:
                m = _TS_RE.search(path)
                run_id = m.group(1) if m else os.path.basename(path)
                coop = run_cooperation_by_gen(path)
                if not coop:
                    continue
                gens = sorted(coop)
                vals = np.array([coop[g] for g in gens])

                row = {"experiment": exp, "model": model, "run": run_id,
                       "n_gens": len(gens)}
                for g in range(1, MAX_GEN + 1):
                    row[f"coop_gen{g}"] = coop.get(g, np.nan)

                # outcomes
                row["final_coop"] = coop[gens[-1]]
                row["mean_coop"] = float(vals.mean())
                if len(gens) >= 4:
                    slope, intercept = np.polyfit(gens, vals, 1)
                    resid = vals - (slope * np.array(gens) + intercept)
                    row["volatility_detrended"] = float(resid.std(ddof=2))
                    row["coop_slope"] = float(slope)
                else:
                    row["volatility_detrended"] = np.nan
                    row["coop_slope"] = np.nan
                row["collapsed"] = int(row["final_coop"] < 10.0)

                # gen-1 initial-state features
                recs1 = analysis.get(run_id, {}).get(1, [])
                if recs1:
                    a_mean, a_std, fr = gen_auc_stats(recs1, curve_cache)
                    texts = [(r.get("strategy") or "").strip() for r in recs1]
                    texts = [t for t in texts if t]
                    row["gen1_auc_mean"] = a_mean
                    row["gen1_auc_std"] = a_std
                    row["gen1_first_round_mean"] = fr
                    row["gen1_semantic_diversity"] = semantic_diversity(texts, emb_lookup)
                    row["gen1_frac_underspecified"] = float(
                        np.mean([not r.get("is_specified", False) for r in recs1]))
                    row["gen1_text_len_mean"] = float(
                        np.mean([len(t.split()) for t in texts])) if texts else np.nan
                else:
                    for k in ("gen1_auc_mean", "gen1_auc_std", "gen1_first_round_mean",
                              "gen1_semantic_diversity", "gen1_frac_underspecified",
                              "gen1_text_len_mean"):
                        row[k] = np.nan

                # early dynamics (gens 1-3)
                row["coop_gen1_feat"] = coop.get(1, np.nan)
                row["dcoop_12"] = (coop.get(2, np.nan) - coop.get(1, np.nan))
                row["dcoop_13"] = (coop.get(3, np.nan) - coop.get(1, np.nan))
                recs2 = analysis.get(run_id, {}).get(2, [])
                if recs1 and recs2:
                    a2_mean, _, _ = gen_auc_stats(recs2, curve_cache)
                    row["dauc_12"] = (a2_mean - row["gen1_auc_mean"]
                                      if not np.isnan(a2_mean) and
                                      not np.isnan(row["gen1_auc_mean"]) else np.nan)
                else:
                    row["dauc_12"] = np.nan

                rows.append(row)
                n_rows += 1
            print(f"  [{exp}/{model}] {n_rows} runs "
                  f"({len(analysis)} runs in analysis JSON)")

    df = pd.DataFrame(rows)
    # deviation from cell mean (needs full cell)
    df["cell_mean_final"] = df.groupby(["experiment", "model"])["final_coop"].transform("mean")
    df["abs_dev_from_cell"] = (df["final_coop"] - df["cell_mean_final"]).abs()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n{len(df)} runs x {len(df.columns)} columns -> {args.output}")
    feat_cols = [c for c in df.columns if c.startswith("gen1_") or c.startswith("dcoop")
                 or c in ("dauc_12", "coop_gen1_feat")]
    print("\nFeature coverage (non-NaN):")
    for c in feat_cols + ["final_coop", "volatility_detrended"]:
        print(f"  {c:28s} {df[c].notna().sum():4d}/{len(df)}")


if __name__ == "__main__":
    main()
