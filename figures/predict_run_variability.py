#!/usr/bin/env python3
"""
Layer C of the variability-prediction analysis: can a run's fate be
predicted, and from what?

Nested feature sets, one row per run (from the run feature table):
  identity : the 15 prompt x model cell dummies
  gen1     : initial-state features (gen-1 only — AUC mean/std, first-round
             donation, semantic diversity, underspecified fraction, text len)
  early    : early-dynamics features (gens 1-3 — coop level/changes, dAUC)

Outcomes: final-generation cooperation, log10 within-run volatility,
collapse (final < 10%, classification).

Two evaluation regimes:
  5-fold CV (shuffled runs)  : "how predictable is a NEW RUN of a known cell?"
  leave-one-cell-out (LOCO)  : "do behavioral features generalize to an
                               UNSEEN prompt x model combination?" (identity
                               dummies cannot, by construction)

Models: ridge / logistic regression (standardized) and random forest;
the table reports the better of the two per configuration. Feature
importance: per-fold permutation importance on the held-out data,
averaged, for the all-features random forest.

Input : figures/run_variation/run_feature_table.csv
Output: figures/run_variation/prediction_r2.png, prediction_importance.png,
        prediction_results.json

Usage:
    python figures/predict_run_variability.py [--table ...]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

GEN1_FEATS = ["gen1_auc_mean", "gen1_auc_std", "gen1_first_round_mean",
              "gen1_semantic_diversity", "gen1_frac_underspecified",
              "gen1_text_len_mean"]
EARLY_FEATS = ["coop_gen1_feat", "dcoop_12", "dcoop_13", "dauc_12"]

C_ID    = "#2a78d6"
C_GEN1  = "#1baf7a"
C_EARLY = "#eda100"
C_COMBO = "#4a3aa7"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

FEATURE_SETS = [
    ("identity",        C_ID,    lambda d: (cell_dummies(d),)),
    ("gen1",            C_GEN1,  lambda d: (d[GEN1_FEATS].to_numpy(float),)),
    ("early",           C_EARLY, lambda d: (d[EARLY_FEATS].to_numpy(float),)),
    ("identity+gen1",   C_COMBO, lambda d: (cell_dummies(d),
                                            d[GEN1_FEATS].to_numpy(float))),
    ("identity+early",  C_COMBO, lambda d: (cell_dummies(d),
                                            d[EARLY_FEATS].to_numpy(float))),
    ("all",             C_COMBO, lambda d: (cell_dummies(d),
                                            d[GEN1_FEATS].to_numpy(float),
                                            d[EARLY_FEATS].to_numpy(float))),
]


def cell_dummies(d):
    return pd.get_dummies(d["experiment"] + ":" + d["model"]).to_numpy(float)


def build_X(df, spec):
    return np.column_stack(spec(df))


def cv_score(X, y, groups, classification, loco=False, seed=0):
    """Out-of-fold R^2 (or ROC-AUC). loco: folds = prompt x model cells."""
    if loco:
        uniq = sorted(set(groups))
        folds = [(np.where(groups != c)[0], np.where(groups == c)[0]) for c in uniq]
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(kf.split(X))

    oof = {}
    for name in ("linear", "forest"):
        pred = np.full(len(y), np.nan)
        for tr, te in folds:
            if classification and len(set(y[tr])) < 2:
                pred[te] = float(y[tr].mean())
                continue
            if name == "linear":
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                m = (LogisticRegression(max_iter=2000, C=1.0) if classification
                     else Ridge(alpha=1.0))
                m.fit(Xtr, y[tr])
                pred[te] = (m.predict_proba(Xte)[:, 1] if classification
                            else m.predict(Xte))
            else:
                m = (RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                            random_state=seed) if classification
                     else RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                random_state=seed))
                m.fit(X[tr], y[tr])
                pred[te] = (m.predict_proba(X[te])[:, 1]
                            if classification and m.n_classes_ > 1 else
                            m.predict(X[te]) if not classification else
                            float(y[tr].mean()))
        score = (roc_auc_score(y, pred) if classification else r2_score(y, pred))
        oof[name] = float(score)
    best = max(oof, key=oof.get)
    return oof[best], best, oof


def fold_permutation_importance(df, y, classification, feat_names, seed=0):
    """Permutation importance of the all-features RF, averaged over 5 held-out folds."""
    X = np.column_stack([df[c].to_numpy(float) for c in feat_names])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    imps = []
    for tr, te in kf.split(X):
        m = (RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                    random_state=seed) if classification
             else RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                        random_state=seed))
        m.fit(X[tr], y[tr])
        r = permutation_importance(m, X[te], y[te], n_repeats=20,
                                   random_state=seed,
                                   scoring="roc_auc" if classification else "r2")
        imps.append(r.importances_mean)
    return np.mean(imps, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Run-fate prediction (Layer C).")
    ap.add_argument("--table", default="figures/run_variation/run_feature_table.csv")
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    df = pd.read_csv(args.table).dropna(
        subset=GEN1_FEATS + EARLY_FEATS + ["final_coop", "volatility_detrended"])
    groups = (df["experiment"] + ":" + df["model"]).to_numpy()
    outcomes = {
        "final cooperation": (df["final_coop"].to_numpy(float), False),
        "log10 volatility": (np.log10(np.clip(
            df["volatility_detrended"].to_numpy(float), 0.1, None)), False),
        "collapse (<10%)": (df["collapsed"].to_numpy(int), True),
    }

    results = {}
    print(f"{'outcome':<20} {'features':<16} {'5-fold':>8} {'LOCO':>8}  best")
    for oname, (y, is_clf) in outcomes.items():
        results[oname] = {}
        for fname, _, spec in FEATURE_SETS:
            X = build_X(df, spec)
            s_cv, best_cv, _ = cv_score(X, y, groups, is_clf, loco=False)
            s_lo, _, _ = cv_score(X, y, groups, is_clf, loco=True)
            results[oname][fname] = {"cv5": s_cv, "loco": s_lo, "best_model": best_cv}
            print(f"{oname:<20} {fname:<16} {s_cv:8.3f} {s_lo:8.3f}  {best_cv}")

    # ---- figure 1: scores per feature set (5-fold and LOCO side by side)
    fig, axes = plt.subplots(1, len(outcomes), figsize=(5.0 * len(outcomes), 4.4),
                             facecolor=SURFACE)
    for ax, (oname, _) in zip(np.atleast_1d(axes), outcomes.items()):
        ax.set_facecolor(SURFACE)
        names = [f for f, _, _ in FEATURE_SETS]
        cv5 = [results[oname][f]["cv5"] for f in names]
        loco = [results[oname][f]["loco"] for f in names]
        xs = np.arange(len(names))
        ax.bar(xs - 0.19, cv5, width=0.38, color=C_ID, label="5-fold CV")
        ax.bar(xs + 0.19, loco, width=0.38, color=C_GEN1, label="leave-one-cell-out")
        metric = "ROC-AUC" if "collapse" in oname else "out-of-fold R²"
        ax.axhline(0.5 if "collapse" in oname else 0.0, color=MUTED,
                   linewidth=1.0, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{oname}  ({metric})", fontsize=10.5, color=INK)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle("Predicting a run's fate — nested feature sets (Layer C)",
                 fontsize=12.5, color=INK)
    fig.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out1 = os.path.join(args.output_dir, "prediction_r2.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # ---- figure 2: permutation importance (behavioral features, all outcomes)
    feat_names = GEN1_FEATS + EARLY_FEATS
    fig, axes = plt.subplots(1, len(outcomes), figsize=(5.0 * len(outcomes), 4.2),
                             facecolor=SURFACE)
    imp_json = {}
    for ax, (oname, (y, is_clf)) in zip(np.atleast_1d(axes), outcomes.items()):
        ax.set_facecolor(SURFACE)
        imp = fold_permutation_importance(df, y, is_clf, feat_names)
        order = np.argsort(imp)
        imp_json[oname] = {feat_names[i]: float(imp[i]) for i in order[::-1]}
        colors = [C_GEN1 if feat_names[i] in GEN1_FEATS else C_EARLY for i in order]
        ax.barh(np.arange(len(imp)), imp[order], color=colors, height=0.6)
        ax.set_yticks(np.arange(len(imp)))
        ax.set_yticklabels([feat_names[i] for i in order], fontsize=8, color=INK)
        ax.set_title(oname, fontsize=10.5, color=INK)
        ax.set_xlabel("permutation importance (held-out)", fontsize=8.5, color=MUTED)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_GEN1),
               plt.Rectangle((0, 0), 1, 1, color=C_EARLY)]
    np.atleast_1d(axes)[-1].legend(handles, ["gen-1 initial state", "early dynamics (gens 1-3)"],
                                   fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Which behavioral features carry the signal? (random forest, "
                 "importance on held-out folds)", fontsize=12, color=INK)
    fig.tight_layout()
    out2 = os.path.join(args.output_dir, "prediction_importance.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(args.output_dir, "prediction_results.json"), "w") as f:
        json.dump({"scores": results, "importance": imp_json}, f, indent=2)
    print(f"\nSaved: {out1}\nSaved: {out2}")


if __name__ == "__main__":
    main()
