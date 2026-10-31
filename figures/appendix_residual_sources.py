#!/usr/bin/env python3
"""
What is inside the "run" term?

The variance decomposition reports a single within-cell residual and calls it
run stochasticity.  That label is a container, not a mechanism: the runs in a
prompt x model cell differ because (a) the initial strategy pool the executor
wrote at generation 1 differs from run to run, (b) selection and LLM-mediated
inheritance amplify small early differences over ten generations, and (c) the
execution itself is nondeterministic (decoding temperature, pairing order,
whatever drifts in a hosted model between May and August).  This script puts
numbers on how much of the within-cell variance each of these accounts for,
so the appendix can say which parts are measured and which are not.

Method.  Everything is an INCREMENT OVER THE CELL MEANS, evaluated
out-of-fold.  The baseline model is the 15 prompt x model cell dummies alone;
its out-of-fold residual sum of squares IS the run component.  Adding a block
of features and refitting gives the share of that run component the block
explains:

    explained_share = 1 - SSE_oof(cells + block) / SSE_oof(cells)

Folds are 5-fold, stratified so every cell is represented in every training
split, repeated over 20 seeds.  Nothing is tuned inside the loop (ridge with
alpha = 1 on standardised features), so there is no selection optimism.

Blocks
  init_pool      generation-1 strategy-pool features only, no behaviour:
                 gen1_auc_mean, gen1_auc_std, gen1_first_round_mean,
                 gen1_semantic_diversity, gen1_frac_underspecified,
                 gen1_text_len_mean
  coop_gen1      generation-1 realised cooperation
  through_gen_g  realised cooperation at generations 1..g

Also reported
  lock-in curve    the through_gen_g share as a function of g: the generation
                   at which a run's final position stops being open
  drift check      within-cell Spearman correlation between a run's wall-clock
                   order and its final cooperation, combined across cells --
                   tests whether hosted-model drift over the collection window
                   masquerades as run noise

Input : figures/run_variation/run_feature_table.csv
Output: figures/appendix_stats/residual_sources.json
        figures/appendix_stats/residual_sources.png

Usage:
    python3.10 figures/appendix_residual_sources.py
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

INIT_POOL = ["gen1_auc_mean", "gen1_auc_std", "gen1_first_round_mean",
             "gen1_semantic_diversity", "gen1_frac_underspecified",
             "gen1_text_len_mean"]

C_INIT, C_EARLY, C_REST = "#1baf7a", "#eda100", "#c3c2b7"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def cell_dummies(d):
    return pd.get_dummies(d["experiment"] + ":" + d["model"]).to_numpy(float)


def oof_sse(X, y, strata, n_rep, seed0=0):
    """Out-of-fold residual SS of ridge(alpha=1) on standardised X, averaged
    over n_rep repeats of stratified 5-fold CV."""
    out = []
    for rep in range(n_rep):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed0 + rep)
        pred = np.empty(len(y))
        for tr, te in skf.split(X, strata):
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict(sc.transform(X[te]))
        out.append(float(((y - pred) ** 2).sum()))
    return float(np.mean(out)), float(np.std(out, ddof=1))


def block_share(df, y, blocks, n_rep):
    """Share of the run (within-cell) component explained by each block."""
    C = cell_dummies(df)
    strata = (df["experiment"] + ":" + df["model"]).to_numpy()
    base, base_sd = oof_sse(C, y, strata, n_rep)
    rows = {}
    for name, cols in blocks.items():
        B = df[cols].to_numpy(float) if isinstance(cols, list) else cols
        aug, aug_sd = oof_sse(np.column_stack([C, B]), y, strata, n_rep)
        rows[name] = {"n_features": int(np.atleast_2d(B).shape[1]),
                      "sse_oof": aug, "sse_oof_sd": aug_sd,
                      "share_of_run_component": float(1.0 - aug / base)}
    return {"baseline_cells_only_sse_oof": base,
            "baseline_sse_oof_sd": base_sd,
            "n_repeats": int(n_rep), "blocks": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="figures/run_variation/run_feature_table.csv")
    ap.add_argument("--n_rep", type=int, default=20)
    ap.add_argument("--output_dir", default="figures/appendix_stats")
    args = ap.parse_args()

    df = pd.read_csv(args.table).dropna(subset=INIT_POOL + ["final_coop"])
    df = df.reset_index(drop=True)
    y = df["final_coop"].to_numpy(float)
    res = {"n_runs": int(len(df))}

    # ---- size of the run component ----------------------------------------
    cellmean = df.groupby(["experiment", "model"])["final_coop"].transform("mean")
    within = y - cellmean.to_numpy()
    res["run_component"] = {
        "within_cell_variance_pp2": float((within ** 2).sum() / (len(y) - 15)),
        "within_cell_sd_pp": float(np.sqrt((within ** 2).sum() / (len(y) - 15))),
        "total_variance_pp2": float(y.var(ddof=1)),
    }
    print("=" * 80)
    print("THE RUN COMPONENT — final-generation cooperation")
    print("=" * 80)
    print(f"  {len(df)} runs, 15 cells; within-cell SD = "
          f"{res['run_component']['within_cell_sd_pp']:.1f} pp "
          f"(total SD = {np.sqrt(y.var(ddof=1)):.1f} pp)")

    # ---- how much of it is measured initialisation? -----------------------
    blocks = {"init_pool (gen-1 strategy text/code only)": INIT_POOL,
              "coop_gen1 (gen-1 realised cooperation)": ["coop_gen1"],
              "init_pool + coop_gen1": INIT_POOL + ["coop_gen1"]}
    for g in range(2, 11):
        blocks[f"through_gen{g}"] = [f"coop_gen{k}" for k in range(1, g + 1)]
    blocks["init_pool + through_gen3"] = INIT_POOL + [f"coop_gen{k}"
                                                      for k in range(1, 4)]
    out = block_share(df, y, blocks, args.n_rep)
    res["explained_share_of_run_component"] = out

    print("\nShare of the run component explained, out-of-fold "
          f"(ridge, 5-fold stratified by cell, {args.n_rep} repeats):")
    print(f"  {'block':<42} {'k':>3} {'share':>8}")
    for name, r in out["blocks"].items():
        print(f"  {name:<42} {r['n_features']:3d} "
              f"{r['share_of_run_component']*100:7.1f}%")

    # ---- lock-in curve -----------------------------------------------------
    lock = {g: out["blocks"][f"through_gen{g}"]["share_of_run_component"]
            for g in range(2, 11)}
    lock[1] = out["blocks"]["coop_gen1 (gen-1 realised cooperation)"][
        "share_of_run_component"]
    res["lockin_curve"] = {str(g): float(lock[g]) for g in sorted(lock)}

    # ---- within-cell correlation of each generation with the final --------
    print("\nWithin-cell correlation between cooperation at gen g and the final "
          "generation:")
    corr = {}
    for g in range(1, 11):
        col = f"coop_gen{g}"
        cm = df.groupby(["experiment", "model"])[col].transform("mean")
        xg = df[col].to_numpy(float) - cm.to_numpy()
        r = float(np.corrcoef(xg, within)[0, 1])
        corr[g] = {"pearson_r": r, "r2": r ** 2}
        print(f"  gen {g:2d}: r = {r:+.3f}  (r^2 = {r**2:.3f})   "
              f"cumulative OOF share = {lock[g]*100:5.1f}%")
    res["within_cell_corr_with_final"] = {str(k): v for k, v in corr.items()}

    # ---- drift check -------------------------------------------------------
    print("\nDrift check — within-cell Spearman(run wall-clock order, final coop):")
    rhos, ps, ns = [], [], []
    for (e, m), g in df.groupby(["experiment", "model"]):
        if len(g) < 5:
            continue
        order = np.argsort(np.argsort(g["run"].to_numpy()))
        rho, p = sps.spearmanr(order, g["final_coop"].to_numpy(float))
        if np.isnan(rho):
            continue
        rhos.append(float(rho)); ps.append(float(p)); ns.append(len(g))
        print(f"  {e:<18} {m:<7} n={len(g):2d}  rho = {rho:+.3f}  p = {p:.3f}")
    chi2 = -2 * np.sum(np.log(np.clip(ps, 1e-300, 1)))
    p_comb = float(sps.chi2.sf(chi2, 2 * len(ps)))
    t_med = sps.wilcoxon(rhos).pvalue if len(rhos) > 5 else np.nan
    print(f"  Fisher-combined p = {p_comb:.3f} over {len(ps)} cells; "
          f"median rho = {np.median(rhos):+.3f}; "
          f"Wilcoxon(rho != 0) p = {t_med:.3f}")
    res["drift_check"] = {"cell_rho": rhos, "cell_p": ps, "cell_n": ns,
                          "fisher_combined_p": p_comb,
                          "median_rho": float(np.median(rhos)),
                          "wilcoxon_p": float(t_med)}

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    gs = list(range(1, 11))
    ys = [lock[g] * 100 for g in gs]
    init = out["blocks"]["init_pool (gen-1 strategy text/code only)"][
        "share_of_run_component"] * 100
    ax.fill_between(gs, 0, ys, color=C_EARLY, alpha=0.35, zorder=1)
    ax.plot(gs, ys, color=C_EARLY, linewidth=2.4, marker="o", markersize=5,
            zorder=3, label="cooperation observed through generation $g$")
    ax.axhline(init, color=C_INIT, linewidth=2.0, linestyle="--", zorder=2,
               label=f"generation-1 strategy pool alone ({init:.0f}%)")
    ax.axhline(100, color=MUTED, linewidth=1.0)
    ax.set_xticks(gs)
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 105)
    ax.set_xlabel("generation $g$ of the last observation used", fontsize=10.5, color=INK)
    ax.set_ylabel("share of the run component explained (%)", fontsize=10, color=INK)
    ax.set_title("What the within-cell run component is made of",
                 fontsize=12, color=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right", labelcolor=INK)
    fig.text(0.99, 0.01, "out-of-fold ridge increment over prompt x model cell "
             "means; 5-fold stratified by cell, 20 repeats",
             ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    png = os.path.join(args.output_dir, "residual_sources.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(args.output_dir, "residual_sources.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {png}")
    print(f"Saved: {os.path.join(args.output_dir, 'residual_sources.json')}")


if __name__ == "__main__":
    main()
