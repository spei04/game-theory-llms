#!/usr/bin/env python3
"""
Appendix statistics for the variance decomposition.

Three things the short appendix description left implicit:

 1. PERMUTATION SCHEME.  The first version shuffled the raw cooperation values
    against all labels at once.  That null ("no structure of any kind") is the
    wrong reference for the interaction: data with strong main effects and NO
    interaction still produce a large SS_{MxP} relative to that null, so the
    test is anti-conservative.  This script uses the Freedman-Lane scheme --
    for each term, fit the REDUCED model holding everything except that term,
    freeze its fitted values, permute its residuals, rebuild a synthetic
    response and recompute the statistic.  For the interaction the reduced
    model is exactly the additive M + P model.  ter Braak's variant (permute
    FULL-model residuals) and the old raw shuffle are computed alongside, and
    a calibration study measures the type-I error of all three under a null
    that has real main effects but no interaction.

 2. BALANCE.  The design is NOT balanced (6-20 runs per cell).  With unequal n
    the factorial is non-orthogonal, so type-II components do not add to the
    total sum of squares.  Reported: cell counts, the size of the
    non-additivity, the type-III decomposition as an alternative weighting,
    and a balanced-subsample bootstrap that redoes the decomposition on
    equal-n draws.

 3. RESIDUAL TERM.  Its size, and whether its magnitude is itself
    heterogeneous across cells (Brown-Forsythe).  The decomposition of what
    is *inside* it lives in figures/appendix_residual_sources.py.

Input : figures/run_variation/run_feature_table.csv
Output: figures/appendix_stats/variance_stats.json  (+ console tables)

Usage:
    python3.10 figures/appendix_variance_stats.py [--n_perm 10000]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

MAX_GEN = 10
PARTS = ("model", "prompt", "interaction", "run")


# ---------------------------------------------------------------------------
# Fast OLS: orthonormal bases let every permutation reuse one factorisation
# ---------------------------------------------------------------------------
def basis(*blocks, n):
    """Orthonormal basis of the column space of [1, blocks...]."""
    X = np.column_stack([np.ones(n)] + [b for b in blocks if b.size])
    Q, _ = np.linalg.qr(X)
    keep = np.linalg.matrix_rank(X)
    if keep < Q.shape[1]:                       # drop redundant columns
        _, R = np.linalg.qr(X)
        piv = np.argsort(-np.abs(np.diag(R)))[:keep]
        Q, _ = np.linalg.qr(X[:, np.sort(piv)])
    return Q, keep


def sse_multi(Q, Y):
    """Residual SS of OLS onto span(Q) for each column of Y (n x B)."""
    return np.einsum("ij,ij->j", Y, Y) - np.einsum("ij,ij->j", Q.T @ Y, Q.T @ Y)


class Design:
    """All the fits the decomposition needs, factorised once."""

    def __init__(self, sub):
        n = len(sub)
        self.n = n
        M = pd.get_dummies(sub["model"], drop_first=True).to_numpy(float)
        P = pd.get_dummies(sub["experiment"], drop_first=True).to_numpy(float)
        C = pd.get_dummies(sub["model"] + ":" + sub["experiment"],
                           drop_first=True).to_numpy(float)
        self.Q_m, r_m = basis(M, n=n)
        self.Q_p, r_p = basis(P, n=n)
        self.Q_add, r_add = basis(M, P, n=n)
        self.Q_cell, r_cell = basis(C, n=n)
        self.df = {"model": r_m - 1, "prompt": r_p - 1,
                   "interaction": r_cell - r_add, "err": n - r_cell}
        # effect coding for type III
        Me, Pe, Ie = self._effect(sub)
        self.Q_full3, r_f3 = basis(Me, Pe, Ie, n=n)
        self.Q_pi, _ = basis(Pe, Ie, n=n)
        self.Q_mi, _ = basis(Me, Ie, n=n)
        self.Q_mp, _ = basis(Me, Pe, n=n)

    @staticmethod
    def _effect(sub):
        def ec(series):
            lv = sorted(series.unique())
            base, rest = lv[-1], lv[:-1]
            v = series.to_numpy()
            return np.column_stack([(v == l).astype(float) - (v == base).astype(float)
                                    for l in rest])
        M, P = ec(sub["model"]), ec(sub["experiment"])
        I = np.column_stack([M[:, [i]] * P[:, [j]]
                             for i in range(M.shape[1]) for j in range(P.shape[1])])
        return M, P, I

    # ---- type-II components for a matrix of responses ---------------------
    def type2(self, Y):
        """Y is n x B. Returns dict of component SS arrays of length B."""
        s_cell = sse_multi(self.Q_cell, Y)
        s_add = sse_multi(self.Q_add, Y)
        s_m = sse_multi(self.Q_m, Y)
        s_p = sse_multi(self.Q_p, Y)
        return {"model": np.maximum(s_p - s_add, 0.0),
                "prompt": np.maximum(s_m - s_add, 0.0),
                "interaction": np.maximum(s_add - s_cell, 0.0),
                "run": s_cell}

    def type3(self, Y):
        s_full = sse_multi(self.Q_full3, Y)
        return {"model": np.maximum(sse_multi(self.Q_pi, Y) - s_full, 0.0),
                "prompt": np.maximum(sse_multi(self.Q_mi, Y) - s_full, 0.0),
                "interaction": np.maximum(sse_multi(self.Q_mp, Y) - s_full, 0.0),
                "run": s_full}

    def F(self, Y):
        c = self.type2(Y)
        ms_err = c["run"] / self.df["err"]
        return {k: (c[k] / self.df[k]) / ms_err
                for k in ("model", "prompt", "interaction")}

    def project(self, Q, y):
        """(fitted, residuals) of y on span(Q)."""
        fit = Q @ (Q.T @ y)
        return fit, y - fit


def shares(comp_scalar):
    s = sum(comp_scalar[k] for k in PARTS)
    return {k: comp_scalar[k] / s for k in PARTS}


def scalarise(comp):
    return {k: float(np.atleast_1d(v)[0]) for k, v in comp.items()}


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------
REDUCED_Q = {"model": "Q_p", "prompt": "Q_m", "interaction": "Q_add"}


def permutation_tests(y, D, n_perm, rng, block=500):
    """
    Freedman-Lane, ter Braak and raw-shuffle p-values for the three tested
    terms, all using the type-II F statistic.  Permutations are applied in
    blocks so every scheme reuses one QR factorisation.
    """
    obs = {k: float(v[0]) for k, v in D.F(y[:, None]).items()}
    red = {k: D.project(getattr(D, REDUCED_Q[k]), y) for k in obs}
    _, resid_full = D.project(D.Q_cell, y)

    obs_ss = {k: float(v[0]) for k, v in D.type2(y[:, None]).items() if k in obs}
    schemes = ("freedman_lane", "ter_braak", "raw_shuffle", "legacy_raw_SS")
    ge = {s: {k: 0 for k in obs} for s in schemes}
    null = {s: {k: [] for k in obs} for s in schemes}
    done = 0
    while done < n_perm:
        b = min(block, n_perm - done)
        idx = np.array([rng.permutation(D.n) for _ in range(b)]).T      # n x b
        for k in obs:
            fitted_r, resid_r = red[k]
            for scheme, src in (("freedman_lane", resid_r), ("ter_braak", resid_full)):
                Y = fitted_r[:, None] + src[idx]
                f = D.F(Y)[k]
                ge[scheme][k] += int((f >= obs[k]).sum())
                null[scheme][k].append(f)
        Y = y[idx]
        f_all = D.F(Y)
        ss_all = D.type2(Y)
        for k in obs:
            ge["raw_shuffle"][k] += int((f_all[k] >= obs[k]).sum())
            null["raw_shuffle"][k].append(f_all[k])
            # legacy scheme: shuffle y, compare the RAW component SS
            ge["legacy_raw_SS"][k] += int((ss_all[k] >= obs_ss[k]).sum())
            null["legacy_raw_SS"][k].append(ss_all[k])
        done += b

    out = {"F_obs": obs, "SS_obs": obs_ss,
           "df": {k: int(v) for k, v in D.df.items()},
           "n_perm": int(n_perm), "p": {}, "null_summary": {},
           "statistic": {"freedman_lane": "type-II F", "ter_braak": "type-II F",
                         "raw_shuffle": "type-II F",
                         "legacy_raw_SS": "type-II SS (original scheme)"}}
    for s in ge:
        out["p"][s] = {k: (v + 1) / (n_perm + 1) for k, v in ge[s].items()}
        out["null_summary"][s] = {}
        for k in obs:
            arr = np.concatenate(null[s][k])
            out["null_summary"][s][k] = {
                "mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                "p95": float(np.percentile(arr, 95)), "max": float(arr.max())}
    out["p"]["parametric_F"] = {k: float(sps.f.sf(obs[k], D.df[k], D.df["err"]))
                                for k in obs}
    return out


def calibration_study(y, D, n_sim, n_perm, rng):
    """
    Type-I error of each scheme under two nulls built from this design:

      no_interaction : y = fitted(M + P) + resampled additive residuals
                       -> real main effects, interaction truly absent
      no_model       : y = fitted(P)     + resampled P-model residuals
                       -> real prompt effect, model effect truly absent

    The raw shuffle should over-reject on both; the residual schemes should
    hold roughly nominal size.
    """
    nulls = {"no_interaction": ("interaction", D.Q_add),
             "no_model":       ("model",       D.Q_p)}
    out = {}
    for null_name, (term, Q_gen) in nulls.items():
        fitted_gen, resid_gen = D.project(Q_gen, y)
        p_by_scheme = {s: [] for s in ("freedman_lane", "ter_braak",
                                       "raw_shuffle", "legacy_raw_SS")}
        for _ in range(n_sim):
            y_sim = fitted_gen + resid_gen[rng.permutation(D.n)]
            obs = float(D.F(y_sim[:, None])[term][0])
            obs_ss = float(D.type2(y_sim[:, None])[term][0])
            fitted_r, resid_r = D.project(getattr(D, REDUCED_Q[term]), y_sim)
            _, resid_f = D.project(D.Q_cell, y_sim)
            idx = np.array([rng.permutation(D.n) for _ in range(n_perm)]).T
            for scheme, src in (("freedman_lane", resid_r), ("ter_braak", resid_f)):
                f = D.F(fitted_r[:, None] + src[idx])[term]
                p_by_scheme[scheme].append((int((f >= obs).sum()) + 1) / (n_perm + 1))
            Ysh = y_sim[idx]
            f = D.F(Ysh)[term]
            p_by_scheme["raw_shuffle"].append((int((f >= obs).sum()) + 1) / (n_perm + 1))
            ss = D.type2(Ysh)[term]
            p_by_scheme["legacy_raw_SS"].append(
                (int((ss >= obs_ss).sum()) + 1) / (n_perm + 1))
        out[null_name] = {
            "tested_term": term, "n_sim": int(n_sim), "n_perm": int(n_perm),
            "type1_error": {s: {"alpha_0.05": float(np.mean(np.array(v) <= .05)),
                                "alpha_0.01": float(np.mean(np.array(v) <= .01))}
                            for s, v in p_by_scheme.items()},
            "median_p": {s: float(np.median(v)) for s, v in p_by_scheme.items()},
        }
    return out


# ---------------------------------------------------------------------------
def balanced_bootstrap(df, gen, n_per_cell, n_draws, rng, drop_cells=()):
    col = f"coop_gen{gen}"
    sub = df.dropna(subset=[col]).copy()
    sub["cell"] = sub["model"] + ":" + sub["experiment"]
    sub = sub[~sub["cell"].isin(drop_cells)]
    groups = {c: g.index.to_numpy() for c, g in sub.groupby("cell")}
    if any(len(v) < n_per_cell for v in groups.values()):
        return None
    draws = []
    for _ in range(n_draws):
        idx = np.concatenate([rng.choice(v, n_per_cell, replace=False)
                              for v in groups.values()])
        s = sub.loc[idx]
        Dz = Design(s)
        comp = scalarise(Dz.type2(s[col].to_numpy(float)[:, None]))
        draws.append([shares(comp)[k] for k in PARTS])
    A = np.array(draws) * 100
    return {"n_per_cell": int(n_per_cell), "n_cells": int(len(groups)),
            "n_draws": int(n_draws), "dropped_cells": list(drop_cells),
            "mean": {k: float(v) for k, v in zip(PARTS, A.mean(0))},
            "lo95": {k: float(v) for k, v in zip(PARTS, np.percentile(A, 2.5, 0))},
            "hi95": {k: float(v) for k, v in zip(PARTS, np.percentile(A, 97.5, 0))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="figures/run_variation/run_feature_table.csv")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--n_draws", type=int, default=2000)
    ap.add_argument("--n_sim", type=int, default=400)
    ap.add_argument("--sim_perm", type=int, default=400)
    ap.add_argument("--output_dir", default="figures/appendix_stats")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    rng = np.random.default_rng(0)
    res = {"source_table": args.table}

    # ---- 1. design balance -------------------------------------------------
    counts = pd.crosstab(df["experiment"], df["model"])
    print("=" * 84)
    print("1. DESIGN BALANCE — completed runs per prompt x model cell")
    print("=" * 84)
    print(counts.to_string())
    flat = counts.to_numpy().ravel()
    print(f"\ntotal = {flat.sum()} runs in {flat.size} cells; "
          f"min {flat.min()}, max {flat.max()}, "
          f"balanced = {bool(flat.min() == flat.max())}")
    res["balance"] = {
        "counts": {e: {m: int(counts.loc[e, m]) for m in counts.columns}
                   for e in counts.index},
        "total_runs": int(flat.sum()), "n_cells": int(flat.size),
        "min_per_cell": int(flat.min()), "max_per_cell": int(flat.max()),
        "is_balanced": bool(flat.min() == flat.max())}

    # ---- 2. decomposition + permutation tests ------------------------------
    print("\n" + "=" * 84)
    print("2. TYPE-II DECOMPOSITION AND FREEDMAN-LANE PERMUTATION TESTS")
    print("=" * 84)
    print(f"{'gen':>3} {'n':>4} | {'model':>6} {'prompt':>7} {'inter':>6} {'run':>6} "
          f"| {'sum/SStot':>9} | {'F_M':>8} {'F_P':>8} {'F_MxP':>8} | p(FL) all terms")
    res["by_generation"] = {}
    D_full = {}
    for g in range(1, MAX_GEN + 1):
        col = f"coop_gen{g}"
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col]).reset_index(drop=True)
        y = sub[col].to_numpy(float)
        D = Design(sub)
        D_full[g] = (D, y)
        c2 = scalarise(D.type2(y[:, None]))
        c3 = scalarise(D.type3(y[:, None]))
        ss_total = float(((y - y.mean()) ** 2).sum())
        s2, s3 = shares(c2), shares(c3)
        perm = permutation_tests(y, D, args.n_perm, rng)
        ratio = sum(c2[k] for k in PARTS) / ss_total
        fl = perm["p"]["freedman_lane"]
        pmax = max(fl.values())
        print(f"{g:3d} {len(sub):4d} | {s2['model']*100:5.1f}% {s2['prompt']*100:6.1f}% "
              f"{s2['interaction']*100:5.1f}% {s2['run']*100:5.1f}% | {ratio:9.4f} | "
              f"{perm['F_obs']['model']:8.1f} {perm['F_obs']['prompt']:8.1f} "
              f"{perm['F_obs']['interaction']:8.1f} | p <= {pmax:.5f}")
        res["by_generation"][g] = {
            "n_runs": int(len(sub)), "ss_total": ss_total,
            "type2_ss": c2, "type2_shares_pct": {k: v * 100 for k, v in s2.items()},
            "type3_ss": c3, "type3_shares_pct": {k: v * 100 for k, v in s3.items()},
            "component_sum_over_ss_total": float(ratio),
            "permutation": perm}

    print("\nPermutation p by scheme, and how far the observed F sits above each null:")
    for g in (1, 10):
        b = res["by_generation"][g]
        print(f"  gen {g}:  F_obs = " + ", ".join(
            f"{k} {b['permutation']['F_obs'][k]:.1f}" for k in
            ("model", "prompt", "interaction")))
        for scheme in ("freedman_lane", "ter_braak", "raw_shuffle", "legacy_raw_SS"):
            ns = b["permutation"]["null_summary"][scheme]
            p = b["permutation"]["p"][scheme]
            print(f"    {scheme:<14} " + "  ".join(
                f"{k}: p={p[k]:.5f} null95={ns[k]['p95']:.2f} nullmax={ns[k]['max']:.2f}"
                for k in ("model", "prompt", "interaction")))
        print(f"    {'parametric_F':<14} " + "  ".join(
            f"{k}: p={b['permutation']['p']['parametric_F'][k]:.3g}"
            for k in ("model", "prompt", "interaction")))

    print("\nType-II vs type-III shares (%), the two standard weightings:")
    for g in (1, 10):
        a = res["by_generation"][g]["type2_shares_pct"]
        b = res["by_generation"][g]["type3_shares_pct"]
        print(f"  gen {g:<2} II : " + " ".join(f"{k}={a[k]:5.1f}" for k in PARTS))
        print(f"        III: " + " ".join(f"{k}={b[k]:5.1f}" for k in PARTS))

    # ---- 2b. calibration of the three schemes ------------------------------
    print("\n" + "=" * 84)
    print("2b. SCHEME CALIBRATION — type-I error under nulls built from gen-10 data")
    print("=" * 84)
    D10, y10 = D_full[10]
    cal = calibration_study(y10, D10, args.n_sim, args.sim_perm, rng)
    res["calibration_gen10"] = cal
    for null_name, c in cal.items():
        print(f"  null = {null_name}  (testing the '{c['tested_term']}' term, "
              f"{c['n_sim']} simulated datasets x {c['n_perm']} permutations)")
        for s, v in c["type1_error"].items():
            print(f"    {s:<15} reject@.05 = {v['alpha_0.05']:.3f}   "
                  f"reject@.01 = {v['alpha_0.01']:.3f}   "
                  f"median p = {c['median_p'][s]:.3f}")

    # ---- 3. balanced-subsample robustness ---------------------------------
    print("\n" + "=" * 84)
    print("3. BALANCED-SUBSAMPLE ROBUSTNESS (equal n per cell, type-II shares %)")
    print("=" * 84)
    res["balanced_bootstrap"] = {}
    specs = [(6, ()), (18, ("gpt:neutral_deepseek",))]
    for g in (1, 10):
        res["balanced_bootstrap"][g] = {}
        for n_pc, drop in specs:
            bb = balanced_bootstrap(df, g, n_pc, args.n_draws, rng, drop_cells=drop)
            if bb is None:
                continue
            key = f"n{n_pc}" + ("_drop_gpt_deepseek" if drop else "_all15")
            res["balanced_bootstrap"][g][key] = bb
            desc = f"gen {g}  n={n_pc}/cell, {bb['n_cells']} cells"
            print(f"  {desc:<32} " + "  ".join(
                f"{k}={bb['mean'][k]:5.1f}[{bb['lo95'][k]:4.1f},{bb['hi95'][k]:4.1f}]"
                for k in PARTS))

    # ---- 4. heterogeneity of the run term ---------------------------------
    print("\n" + "=" * 84)
    print("4. IS THE RUN COMPONENT HOMOGENEOUS ACROSS CELLS? (Brown-Forsythe)")
    print("=" * 84)
    res["within_cell_spread"] = {}
    for g in (1, 10):
        col = f"coop_gen{g}"
        sub = df.dropna(subset=[col])
        groups = [v[col].to_numpy(float)
                  for _, v in sub.groupby(["experiment", "model"])]
        stat, p = sps.levene(*groups, center="median")
        sds = {f"{e}:{m}": float(v[col].std(ddof=1))
               for (e, m), v in sub.groupby(["experiment", "model"])}
        print(f"  gen {g:<2} W = {stat:.2f}, p = {p:.3g};  within-cell SD "
              f"{min(sds.values()):.1f}-{max(sds.values()):.1f} pp")
        res["within_cell_spread"][g] = {"brown_forsythe_W": float(stat),
                                        "p": float(p), "cell_sd": sds}

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "variance_stats.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
