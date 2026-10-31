#!/usr/bin/env python3
"""
Full specification and validation of the run-fate prediction models.

The appendix previously reported only headline numbers.  This script pins
down, for every number in the prediction table: the estimand, the exact input
features, the learner and its hyper-parameters, the resampling scheme, the
baseline the score is measured against, and an uncertainty interval.  It also
adds the diagnostics a reader needs to interpret the collapse ROC-AUC, which
is easy to over-read because collapse is almost a cell-level property.

ESTIMANDS (one row per run, n = 283)
  final_coop   cooperation rate in generation 10, in percentage points
  collapse     1 if final_coop < 10, else 0

FEATURE BLOCKS
  identity  15 prompt x model cell indicator variables.  Under
            leave-one-cell-out the held-out cell's indicator is never active
            in training, so this block necessarily predicts the grand mean
            there; its LOCO score is reported to make that floor explicit,
            not as a competitive result.
  gen1      six features of the generation-1 strategy pool, computed from the
            strategy text and its extracted code BEFORE any donation is made:
              gen1_auc_mean            mean over the 12 agents of the
                                       normalised area under the agent's
                                       donation-vs-partner-history response
                                       curve (generosity of the coded policy)
              gen1_auc_std             standard deviation of that AUC across
                                       the 12 agents (pool heterogeneity)
              gen1_first_round_mean    mean donation the coded policy makes
                                       with no history, strategy_fn(None)
              gen1_semantic_diversity  mean pairwise cosine DISTANCE between
                                       the 12 strategy-text embeddings
              gen1_frac_underspecified fraction of agents whose strategy text
                                       did not yield a fully specified policy
              gen1_text_len_mean       mean strategy length in words
  early     four features from generations 1-3 of realised play:
              coop_gen1_feat  cooperation in generation 1
              dcoop_12        cooperation gen 2 minus gen 1
              dcoop_13        cooperation gen 3 minus gen 1
              dauc_12         mean coded-strategy AUC gen 2 minus gen 1
  Nested sets: identity, gen1, early, identity+gen1, identity+early, all.

LEARNERS (fixed, nothing tuned inside or outside the folds)
  ridge      Ridge(alpha=1.0) / LogisticRegression(C=1.0, lbfgs, max_iter=2000)
             on features standardised with the TRAINING fold's mean and SD
  forest     RandomForest{Regressor,Classifier}(n_estimators=400,
             min_samples_leaf=3, all other defaults, random_state fixed)
  Both are reported for every configuration.  The earlier table showed the
  better of the two, which is a selection on the same held-out data; the gap
  between "best of two" and each learner alone is reported here so the size of
  that optimism is visible.

RESAMPLING
  cv5   5-fold, shuffled, stratified by prompt x model cell, repeated over 20
        seeds; the table gives mean and SD across repeats.  Question: how well
        can a NEW run of an ALREADY-SEEN cell be predicted?
  loco  leave-one-cell-out: all runs of one prompt x model cell are held out
        at a time, 15 folds.  Question: do the behavioural features transfer
        to a prompt x model combination never seen in training?

BASELINES
  regression      out-of-fold R^2 is against the grand mean, so 0 is "no
                  better than predicting the overall average".  The
                  cell-mean-only model (= the identity block) is the
                  reference for how much the behavioural blocks add.
  classification  ROC-AUC against 0.5; the collapse base rate and the
                  per-cell collapse rates are printed because a pooled AUC
                  over cells with near-pure outcomes mostly measures
                  BETWEEN-cell discrimination.  A within-cell (cell-stratified)
                  AUC is therefore reported alongside, computed only over
                  the cells that contain both outcomes.

UNCERTAINTY
  cv5   SD over the 20 repeats.
  loco  cluster bootstrap over the 15 cells, 2000 resamples, 95 percentile
        interval, plus a label-permutation null (labels shuffled within cell
        for the within-cell AUC, across cells for the pooled one).

Input : figures/run_variation/run_feature_table.csv
Output: figures/appendix_stats/prediction_spec.json  (+ console tables)

Usage:
    python3.10 figures/appendix_prediction_spec.py
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

GEN1 = ["gen1_auc_mean", "gen1_auc_std", "gen1_first_round_mean",
        "gen1_semantic_diversity", "gen1_frac_underspecified",
        "gen1_text_len_mean"]
EARLY = ["coop_gen1_feat", "dcoop_12", "dcoop_13", "dauc_12"]

SETS = {
    "identity":       ("cells",),
    "gen1":           ("gen1",),
    "early":          ("early",),
    "identity+gen1":  ("cells", "gen1"),
    "identity+early": ("cells", "early"),
    "all":            ("cells", "gen1", "early"),
}


def build(df, spec):
    parts = []
    for p in spec:
        if p == "cells":
            parts.append(pd.get_dummies(
                df["experiment"] + ":" + df["model"]).to_numpy(float))
        elif p == "gen1":
            parts.append(df[GEN1].to_numpy(float))
        else:
            parts.append(df[EARLY].to_numpy(float))
    return np.column_stack(parts)


def make(name, clf, seed):
    if name == "ridge":
        return (LogisticRegression(max_iter=2000, C=1.0) if clf
                else Ridge(alpha=1.0))
    return (RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                   random_state=seed) if clf
            else RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                       random_state=seed))


def fit_predict(learner, X, y, tr, te, clf, seed):
    if clf and len(set(y[tr])) < 2:
        return np.full(len(te), float(y[tr].mean()))
    if learner == "ridge":
        sc = StandardScaler().fit(X[tr])
        m = make(learner, clf, seed).fit(sc.transform(X[tr]), y[tr])
        Xte = sc.transform(X[te])
        return m.predict_proba(Xte)[:, 1] if clf else m.predict(Xte)
    m = make(learner, clf, seed).fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1] if clf else m.predict(X[te])


def oof_cv5(X, y, strata, clf, learner, n_rep=20):
    scores, preds = [], []
    for rep in range(n_rep):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
        p = np.empty(len(y), float)
        for tr, te in skf.split(X, strata):
            p[te] = fit_predict(learner, X, y, tr, te, clf, rep)
        scores.append(roc_auc_score(y, p) if clf else r2_score(y, p))
        preds.append(p)
    return float(np.mean(scores)), float(np.std(scores, ddof=1)), np.mean(preds, 0)


def oof_loco(X, y, groups, clf, learner, seed=0):
    p = np.empty(len(y), float)
    per_fold = {}
    for c in sorted(set(groups)):
        te = np.where(groups == c)[0]
        tr = np.where(groups != c)[0]
        p[te] = fit_predict(learner, X, y, tr, te, clf, seed)
        if clf:
            per_fold[c] = (float(roc_auc_score(y[te], p[te]))
                           if len(set(y[te])) > 1 else None)
        else:
            per_fold[c] = float(np.mean((y[te] - p[te]) ** 2))
    score = roc_auc_score(y, p) if clf else r2_score(y, p)
    return float(score), p, per_fold


def cluster_bootstrap(y, pred, groups, clf, n_boot=2000, seed=0):
    """95% interval by resampling whole prompt x model cells with replacement."""
    rng = np.random.default_rng(seed)
    cells = sorted(set(groups))
    idx_by_cell = {c: np.where(groups == c)[0] for c in cells}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cells, len(cells), replace=True)
        idx = np.concatenate([idx_by_cell[c] for c in pick])
        if clf and len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], pred[idx]) if clf
                    else r2_score(y[idx], pred[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)),
            int(len(vals)))


def within_cell_auc(y, pred, groups):
    """Cell-stratified AUC: only pairs from the SAME cell are compared."""
    num = den = 0.0
    used = []
    for c in sorted(set(groups)):
        i = np.where(groups == c)[0]
        pos, neg = i[y[i] == 1], i[y[i] == 0]
        if not len(pos) or not len(neg):
            continue
        used.append(c)
        d = pred[pos][:, None] - pred[neg][None, :]
        num += (d > 0).sum() + 0.5 * (d == 0).sum()
        den += len(pos) * len(neg)
    return (float(num / den) if den else float("nan")), used, int(den)


def permutation_auc_null(y, pred, groups, within, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        if within:
            ys = y.copy()
            for c in set(groups):
                i = np.where(groups == c)[0]
                ys[i] = rng.permutation(y[i])
            v = within_cell_auc(ys, pred, groups)[0]
        else:
            v = roc_auc_score(rng.permutation(y), pred)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)), float(np.percentile(vals, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="figures/run_variation/run_feature_table.csv")
    ap.add_argument("--n_rep", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--output_dir", default="figures/appendix_stats")
    args = ap.parse_args()

    df = pd.read_csv(args.table).dropna(
        subset=GEN1 + EARLY + ["final_coop"]).reset_index(drop=True)
    groups = (df["experiment"] + ":" + df["model"]).to_numpy()
    res = {"n_runs": int(len(df)),
           "features": {"gen1": GEN1, "early": EARLY,
                        "identity": "15 prompt x model cell indicators"},
           "learners": {"ridge": "Ridge(alpha=1.0) / LogisticRegression(C=1.0), "
                                 "standardised on the training fold",
                        "forest": "RandomForest(n_estimators=400, "
                                  "min_samples_leaf=3)"},
           "cv": {"cv5": f"5-fold stratified by cell, {args.n_rep} repeats",
                  "loco": "leave-one-prompt-x-model-cell-out, 15 folds"}}

    outcomes = {
        "final cooperation (R^2)": (df["final_coop"].to_numpy(float), False),
        "collapse <10% (ROC-AUC)": (df["collapsed"].to_numpy(int), True),
    }

    # ---- collapse structure, printed before any model -----------------------
    yc = df["collapsed"].to_numpy(int)
    rate = pd.crosstab(df["experiment"], df["model"],
                       values=df["collapsed"], aggfunc="mean")
    mixed = [c for c in sorted(set(groups))
             if 0 < yc[groups == c].mean() < 1]
    print("=" * 86)
    print("COLLAPSE STRUCTURE — how much of 'collapse' is a cell-level property?")
    print("=" * 86)
    print(rate.round(2).to_string())
    print(f"\n  base rate {yc.mean()*100:.1f}% ({yc.sum()}/{len(yc)} runs); "
          f"{len(mixed)}/15 cells contain BOTH outcomes: {mixed}")
    print("  => a pooled ROC-AUC over all runs is dominated by between-cell "
          "discrimination.\n     The within-cell AUC below is the part that is "
          "about individual runs.")
    res["collapse_structure"] = {
        "base_rate": float(yc.mean()), "n_collapsed": int(yc.sum()),
        "cell_rate": {f"{e}:{m}": float(rate.loc[e, m])
                      for e in rate.index for m in rate.columns},
        "mixed_cells": mixed}

    # ---- main table ---------------------------------------------------------
    res["scores"] = {}
    for oname, (y, clf) in outcomes.items():
        res["scores"][oname] = {}
        print("\n" + "=" * 86)
        print(f"{oname}   (n = {len(y)})")
        print("=" * 86)
        print(f"{'features':<16} {'learner':<8} {'cv5':>16} {'loco':>8} "
              f"{'loco 95% CI':>18}")
        for sname, spec in SETS.items():
            X = build(df, spec)
            res["scores"][oname][sname] = {"n_columns": int(X.shape[1])}
            best_cv, best_lo = -np.inf, -np.inf
            for learner in ("ridge", "forest"):
                m, sd, _ = oof_cv5(X, y, groups, clf, learner, args.n_rep)
                s_lo, pred_lo, per_fold = oof_loco(X, y, groups, clf, learner)
                lo, hi, nb = cluster_bootstrap(y, pred_lo, groups, clf,
                                               args.n_boot)
                entry = {"cv5_mean": m, "cv5_sd": sd, "loco": s_lo,
                         "loco_ci95": [lo, hi], "loco_boot_n": nb,
                         "loco_per_fold": per_fold}
                if clf:
                    wa, used, npairs = within_cell_auc(y, pred_lo, groups)
                    entry["loco_within_cell_auc"] = wa
                    entry["loco_within_cell_cells"] = used
                    entry["loco_within_cell_pairs"] = npairs
                res["scores"][oname][sname][learner] = entry
                best_cv, best_lo = max(best_cv, m), max(best_lo, s_lo)
                print(f"{sname:<16} {learner:<8} {m:8.3f} +- {sd:.3f} "
                      f"{s_lo:8.3f} {f'[{lo:.3f}, {hi:.3f}]':>18}")
            res["scores"][oname][sname]["best_of_two_cv5"] = float(best_cv)
            res["scores"][oname][sname]["best_of_two_loco"] = float(best_lo)

    # ---- collapse: within-cell AUC and its null ----------------------------
    print("\n" + "=" * 86)
    print("COLLAPSE — pooled vs within-cell discrimination (leave-one-cell-out)")
    print("=" * 86)
    y = df["collapsed"].to_numpy(int)
    res["collapse_detail"] = {}
    for sname in ("gen1", "early", "all"):
        X = build(df, SETS[sname])
        for learner in ("ridge", "forest"):
            s_lo, pred, per_fold = oof_loco(X, y, groups, True, learner)
            wa, used, npairs = within_cell_auc(y, pred, groups)
            null_w = permutation_auc_null(y, pred, groups, True, 2000)
            null_p = permutation_auc_null(y, pred, groups, False, 2000)
            lo, hi, _ = cluster_bootstrap(y, pred, groups, True, args.n_boot)
            print(f"  {sname:<6} {learner:<7} pooled AUC {s_lo:.3f} "
                  f"[{lo:.3f},{hi:.3f}]  (null mean {null_p[0]:.3f}, "
                  f"95th {null_p[1]:.3f})")
            print(f"  {'':<6} {'':<7} within-cell AUC {wa:.3f} over "
                  f"{len(used)} mixed cells / {npairs} pairs  "
                  f"(null mean {null_w[0]:.3f}, 95th {null_w[1]:.3f})")
            res["collapse_detail"][f"{sname}:{learner}"] = {
                "pooled_auc": s_lo, "pooled_ci95": [lo, hi],
                "pooled_null_mean": null_p[0], "pooled_null_p95": null_p[1],
                "within_cell_auc": wa, "within_cell_cells": used,
                "within_cell_pairs": npairs,
                "within_cell_null_mean": null_w[0],
                "within_cell_null_p95": null_w[1],
                "per_fold_auc": per_fold}

    # ---- what the models actually use --------------------------------------
    print("\n" + "=" * 86)
    print("WHERE THE SIGNAL SITS — standardised ridge coefficients (all data) "
          "and\nheld-out permutation importance (forest, 5-fold, 20 repeats)")
    print("=" * 86)
    from sklearn.inspection import permutation_importance
    res["feature_effects"] = {}
    for oname, (y, clf) in outcomes.items():
        res["feature_effects"][oname] = {}
        for sname in ("gen1", "early"):
            cols = GEN1 if sname == "gen1" else EARLY
            X = df[cols].to_numpy(float)
            sc = StandardScaler().fit(X)
            m = make("ridge", clf, 0).fit(sc.transform(X), y)
            coef = (m.coef_[0] if clf else m.coef_)
            imps = []
            for rep in range(20):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
                for tr, te in skf.split(X, groups):
                    f = make("forest", clf, rep).fit(X[tr], y[tr])
                    r = permutation_importance(
                        f, X[te], y[te], n_repeats=5, random_state=rep,
                        scoring="roc_auc" if clf else "r2")
                    imps.append(r.importances_mean)
            imp = np.mean(imps, 0)
            res["feature_effects"][oname][sname] = {
                c: {"ridge_beta_standardised": float(b),
                    "forest_permutation_importance": float(i)}
                for c, b, i in zip(cols, coef, imp)}
            print(f"  {oname} / {sname}")
            for c, b, i in sorted(zip(cols, coef, imp), key=lambda t: -abs(t[2])):
                print(f"    {c:<26} beta = {b:+7.3f}   perm. importance = {i:+.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "prediction_spec.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
