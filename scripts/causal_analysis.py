"""
causal_analysis.py — Causal Forest analysis on a BanditDB Parquet export.

Answers four questions that standard OPE estimators cannot:
  1. What is the causal effect of each arm, controlling for context?
  2. Which arm is causally best for each user profile, and does the bandit agree?
  3. Which context features drive the heterogeneity?
  4. Is the bandit's selection policy stable over time?

Works with both LinUCB and Thompson Sampling campaigns. The estimator is
CausalForestDML (Double Machine Learning), which residualises both the outcome
and the treatment against the feature distribution before estimating causal
effects. The selection propensity is learned internally by model_t from
observed (arm, context) pairs — logged propensities from the Parquet file are
not used. This makes TS and LinUCB equivalent: both have their selection
mechanism recovered from data, which is more robust than using time-varying
logged propensities from a non-stationary bandit policy.

Requirements:
    pip install econml scikit-learn polars numpy pandas

Usage:
    python causal_analysis.py --parquet /data/exports/sleep.parquet \\
                               --features sex age_norm weight_norm activity bedtime_norm

Note on sample size: causal forests need at least ~200–300 observations per arm
for reliable CATE estimates. With fewer observations treat results as directional.
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_parquet(path: str):
    df = pl.read_parquet(path).to_pandas()
    null_pct = df["propensity"].isna().mean() * 100
    if null_pct == 100:
        print("  [info] Thompson Sampling campaign — propensity column is null. "
              "Selection probability will be estimated internally by model_t (DML).\n")
    elif null_pct > 0:
        print(f"  [info] {null_pct:.0f}% of rows have null propensity (mixed TS/LinUCB or "
              f"Progressive campaign). All rows are used — model_t estimates selection "
              f"probability from data.\n")
    return df


def feature_matrix(df):
    cols = sorted(c for c in df.columns if c.startswith("feature_"))
    return df[cols].values, cols


def _rate_block(r: float) -> str:
    if r > 0.6:  return "█"
    if r > 0.4:  return "▓"
    if r > 0.2:  return "▒"
    if r > 0.05: return "░"
    return "·"


# ── Section 0: Positivity & Confounding ──────────────────────────────────────

def print_positivity_check(df, arm_names):
    """
    Pre-flight DML validity check. Runs before the expensive causal forests.

    DML requires the positivity assumption: every arm must have some positive
    selection probability across the entire feature space. If an arm is never
    selected for a region of contexts (e.g. because the bandit has collapsed),
    the causal effect estimate for that arm in that region is extrapolation.

    Also surfaces how much confounding the DML is correcting for (AUC of model_t).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    X, _    = feature_matrix(df)
    arm_ids = df["arm_id"].values

    print("━" * 65)
    print("0. POSITIVITY & CONFOUNDING DIAGNOSTICS")
    print("   (pre-flight check — run before fitting causal forests)")
    print("━" * 65)
    print(f"  {'Arm':<32}  {'AUC':>5}  {'P<0.05':>7}  {'P>0.95':>7}")
    print(f"  {'─'*32}  {'─'*5}  {'─'*7}  {'─'*7}")

    any_violation = False
    for arm in arm_names:
        T   = (arm_ids == arm).astype(int)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        auc = cross_val_score(clf, X, T, cv=3, scoring="roc_auc").mean()
        clf.fit(X, T)
        p         = clf.predict_proba(X)[:, 1]
        near_zero = (p < 0.05).mean() * 100
        near_one  = (p > 0.95).mean() * 100
        violation = near_zero > 20 or near_one > 20
        if violation:
            any_violation = True
        flag = "  ⚠ positivity violation" if violation else ""
        print(f"  {arm:<32}  {auc:.3f}  {near_zero:6.1f}%  {near_one:6.1f}%{flag}")

    print()
    print("  AUC ≈ 0.5 → near-random selection  (strong causal validity)")
    print("  AUC > 0.8 → strong confounding      (DML is correcting for selection bias)")
    print("  P<0.05 > 20% → positivity violation — CATE estimates unreliable for that arm")
    if any_violation:
        print()
        print("  ⚠  Positivity violations often indicate exploration collapse.")
        print("     Check GET /campaign/:id/diagnostics for entropy_status.")
    print()


# ── Core analysis ─────────────────────────────────────────────────────────────

def run_causal_forest(df, feature_names=None):
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

    X, feat_cols = feature_matrix(df)
    Y            = df["reward"].values
    arms         = df["arm_id"].values
    arm_names    = sorted(df["arm_id"].unique())
    names        = feature_names if feature_names else feat_cols

    n_per_arm = {a: (arms == a).sum() for a in arm_names}
    low_data  = [a for a, n in n_per_arm.items() if n < 200]
    if low_data:
        print(f"  [warn] Low sample count for arms: {low_data}. "
              f"Treat CATE estimates as directional only.\n")

    results = {}

    for arm in arm_names:
        T = (arms == arm).astype(int)

        model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
            model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3),
            n_estimators=500,
            n_crossfit_splits=5,   # reduces first-stage overfitting bias vs default 2
            min_samples_leaf=10,
            random_state=42,
            inference=True,
        )
        model.fit(Y, T, X=X)

        cate              = model.effect(X)
        cate_lb, cate_ub  = model.effect_interval(X, alpha=0.1)
        ate               = model.ate(X)
        ate_lb, ate_ub    = model.ate_interval(X, alpha=0.1)
        importances       = model.feature_importances_
        cate_p25, cate_p50, cate_p75 = np.percentile(cate, [25, 50, 75])

        results[arm] = dict(
            ate=ate, ate_lb=ate_lb, ate_ub=ate_ub,
            cate=cate, cate_lb=cate_lb, cate_ub=cate_ub,
            cate_p25=cate_p25, cate_p50=cate_p50, cate_p75=cate_p75,
            importances=importances,
            n=int(T.sum()),
        )

    return results, arm_names, X, Y, arms, names


# ── Output formatters ─────────────────────────────────────────────────────────

def print_ate(results, arm_names):
    print("━" * 65)
    print("1. AVERAGE TREATMENT EFFECT  (arm vs rest, 90% CI)")
    print("   CATE distribution: p25 / median / p75 across individual users")
    print("━" * 65)
    print(f"  {'Arm':<32} {'ATE':>8}  {'90% CI':<24}    {'p25':>7} {'p50':>7} {'p75':>7}")
    print(f"  {'─'*32} {'─'*8}  {'─'*24}    {'─'*7} {'─'*7} {'─'*7}")
    for arm in arm_names:
        r   = results[arm]
        sig = "✓" if r["ate_lb"] > 0 else ("✗" if r["ate_ub"] < 0 else "~")
        print(f"  {arm:<32} {r['ate']:>+8.4f}  "
              f"[{r['ate_lb']:+.4f}, {r['ate_ub']:+.4f}]  {sig}  "
              f"{r['cate_p25']:>+7.4f} {r['cate_p50']:>+7.4f} {r['cate_p75']:>+7.4f}")
    print()
    print("  ✓ = significantly positive  ✗ = significantly negative  ~ = inconclusive")
    print("  Narrow p25–p75 → homogeneous effect (one answer fits all users)")
    print("  Wide  p25–p75 → heterogeneous effect (personalisation matters)")
    print()


def print_causal_assignment(results, arm_names, X, arms):
    print("━" * 65)
    print("2. CAUSAL ASSIGNMENT vs BANDIT SELECTION")
    print("   (what the causal model says vs what the bandit actually chose)")
    print("━" * 65)
    all_cates = np.column_stack([results[a]["cate"] for a in arm_names])
    best_idx  = np.argmax(all_cates, axis=1)

    print(f"  {'Arm':<32}  {'Causal':>7}  {'Bandit':>7}  {'Gap':>7}")
    print(f"  {'─'*32}  {'─'*7}  {'─'*7}  {'─'*7}")
    for i, arm in enumerate(arm_names):
        causal_pct = (best_idx == i).mean() * 100
        bandit_pct = (arms == arm).mean() * 100
        gap        = causal_pct - bandit_pct
        bar        = "█" * int(causal_pct / 2)
        print(f"  {arm:<32}  {causal_pct:6.1f}%  {bandit_pct:6.1f}%  {gap:+6.1f}%  {bar}")
    print()
    print("  Gap ≈ 0 → bandit has converged to the correct causal structure")
    print("  Gap > 0 → arm is underserved — bandit is under-routing this group")
    print("  Gap < 0 → arm is overserved  — bandit may have over-converged here")
    print()


def print_feature_importance(results, arm_names, feature_names):
    print("━" * 65)
    print("3. FEATURE IMPORTANCE FOR HETEROGENEOUS EFFECTS")
    print("   (which context dimensions drive arm selection)")
    print("━" * 65)
    for arm in arm_names:
        imp   = results[arm]["importances"]
        order = np.argsort(imp)[::-1]
        print(f"  {arm}:")
        for idx in order:
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            bar  = "█" * int(imp[idx] * 40)
            print(f"    {name:<22}  {imp[idx]:.3f}  {bar}")
        print()


def print_winning_segments(results, arm_names, X, feature_names):
    """
    For each arm, find the context profiles where it has the highest
    causal effect — described as simple high/low splits on top features.
    """
    print("━" * 65)
    print("4. WINNING SEGMENTS")
    print("   (user profiles where each arm has a strong causal advantage)")
    print("━" * 65)

    all_cates = np.column_stack([results[a]["cate"] for a in arm_names])
    best_idx  = np.argmax(all_cates, axis=1)

    for i, arm in enumerate(arm_names):
        mask = best_idx == i
        if mask.sum() < 10:
            print(f"  {arm}: too few users assigned — inconclusive\n")
            continue

        X_arm   = X[mask]
        X_other = X[~mask]

        lines = []
        for f_idx, name in enumerate(feature_names[:X.shape[1]]):
            mean_arm   = X_arm[:, f_idx].mean()
            mean_other = X_other[:, f_idx].mean() if len(X_other) > 0 else 0.5
            diff = mean_arm - mean_other
            if abs(diff) > 0.08:
                direction = "high" if diff > 0 else "low"
                lines.append(f"{name} ({direction}, Δ={diff:+.2f})")

        desc = ", ".join(lines) if lines else "no strong segment signal"
        print(f"  {arm}:")
        print(f"    {desc}")
        print()


def print_selection_stability(df, arm_names, n_buckets=5):
    """
    Split the campaign timeline into equal-sized buckets and show per-arm
    selection rate across time. Stable rates validate the DML IID assumption.
    A monotonic shift toward one arm indicates convergence or collapse.
    """
    if "predicted_at" not in df.columns:
        return

    print("━" * 65)
    print("5. SELECTION STABILITY OVER TIME")
    print(f"   (campaign split into {n_buckets} time buckets, oldest → newest)")
    print("━" * 65)

    df = df.copy()
    try:
        df["_bucket"] = pd.qcut(df["predicted_at"], n_buckets, labels=False, duplicates="drop")
    except Exception:
        return

    buckets     = sorted(df["_bucket"].dropna().unique())
    bucket_ns   = [len(df[df["_bucket"] == b]) for b in buckets]
    header_nums = "  " + "".join(f"  t{int(b)+1}({n}) " for b, n in zip(buckets, bucket_ns))
    print(header_nums)
    print()

    for arm in arm_names:
        rates = []
        for b in buckets:
            bucket = df[df["_bucket"] == b]
            rate   = (bucket["arm_id"] == arm).mean() if len(bucket) > 0 else 0.0
            rates.append(rate)

        blocks = "".join(_rate_block(r) for r in rates)
        pcts   = " ".join(f"{r:5.1%}" for r in rates)
        print(f"  {arm:<32}  {blocks}  [{pcts}]")

    print()
    print("  Stable across buckets  → IID assumption holds, DML estimates reliable")
    print("  Monotonic increase     → bandit converging or collapsing to this arm")
    print("  Sudden jump in t4/t5   → possible reward pipeline event or config change")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Causal Forest analysis on BanditDB Parquet export")
    parser.add_argument("--parquet",  required=True, help="Path to .parquet file")
    parser.add_argument("--features", nargs="*",     help="Human-readable feature names (in order)")
    args = parser.parse_args()

    print()
    print(f"Loading {args.parquet} ...")
    df = load_parquet(args.parquet)
    print(f"  {len(df)} interactions, {df['arm_id'].nunique()} arms, "
          f"{df['arm_id'].value_counts().to_dict()}")
    print()

    arm_names = sorted(df["arm_id"].unique())

    print_positivity_check(df, arm_names)
    print_selection_stability(df, arm_names)

    print("Fitting causal forests (one per arm) ...")
    print()

    results, arm_names, X, Y, arms, feature_names = run_causal_forest(df, args.features)

    print_ate(results, arm_names)
    print_causal_assignment(results, arm_names, X, arms)
    print_feature_importance(results, arm_names, feature_names)
    print_winning_segments(results, arm_names, X, feature_names)

    print("━" * 65)
    print("WHAT TO DO WITH THIS")
    print("━" * 65)
    print("""
  ATE significant & positive → arm is genuinely causing reward
  ATE near zero              → arm and reward are uncorrelated causally
  ATE significant & negative → arm is actively hurting reward

  CATE p25–p75 wide          → personalisation matters; consider finer segmentation
  CATE p25–p75 narrow        → effect is homogeneous; a simple rule may be enough

  Causal assignment gap > 0  → bandit is under-routing a group that would benefit
  Causal assignment gap < 0  → bandit has over-converged; may be missing better arms

  Positivity violation       → bandit has collapsed exploration for that arm;
                               check entropy_status in /campaign/:id/diagnostics

  Selection unstable         → non-stationarity in the data; DML estimates cover the
                               full history but may not reflect current policy

  Feature importance high    → that dimension is driving who gets which arm;
                               if it doesn't match domain knowledge, audit the feature
""")


if __name__ == "__main__":
    main()
