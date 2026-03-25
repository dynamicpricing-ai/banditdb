"""
causal_analysis.py — Causal Forest analysis on a BanditDB Parquet export.

Answers three questions that standard OPE estimators cannot:
  1. What is the causal effect of each arm, controlling for context?
  2. Which arm is causally best for each user profile?
  3. Which context features drive the heterogeneity?

Requirements:
    pip install econml scikit-learn polars numpy

Usage:
    python causal_analysis.py --parquet /data/exports/sleep.parquet \
                               --features sex age_norm weight_norm activity bedtime_norm

Note on sample size: causal forests need at least ~200–300 observations per arm
for reliable CATE estimates. With fewer observations treat results as directional.
"""

import argparse
import warnings
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_parquet(path: str):
    df = pl.read_parquet(path).to_pandas()
    if df["propensity"].isna().all():
        raise ValueError(
            "All propensity values are null. Causal analysis requires LinUCB campaigns "
            "(Thompson Sampling does not log propensities)."
        )
    # Drop rows with null propensity (TS records mixed into a LinUCB export)
    df = df[df["propensity"].notna()].reset_index(drop=True)
    return df


def feature_matrix(df):
    cols = sorted(c for c in df.columns if c.startswith("feature_"))
    return df[cols].values, cols


# ── Core analysis ─────────────────────────────────────────────────────────────

def run_causal_forest(df, feature_names=None):
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

    X, feat_cols = feature_matrix(df)
    Y = df["reward"].values
    arms = df["arm_id"].values
    arm_names = sorted(df["arm_id"].unique())

    # Use provided names or fall back to column names
    names = feature_names if feature_names else feat_cols

    n_per_arm = {a: (arms == a).sum() for a in arm_names}
    low_data = [a for a, n in n_per_arm.items() if n < 200]
    if low_data:
        print(f"  [warn] Low sample count for arms: {low_data}. "
              f"Treat CATE estimates as directional only.\n")

    results = {}

    for arm in arm_names:
        # Binary treatment: was this arm selected?
        T = (arms == arm).astype(int)

        model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
            model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3),
            n_estimators=500,
            min_samples_leaf=10,
            random_state=42,
            inference=True,
        )
        model.fit(Y, T, X=X)

        cate           = model.effect(X)
        cate_lb, cate_ub = model.effect_interval(X, alpha=0.1)
        ate            = model.ate(X)
        ate_lb, ate_ub = model.ate_interval(X, alpha=0.1)
        importances    = model.feature_importances_

        results[arm] = dict(
            ate=ate, ate_lb=ate_lb, ate_ub=ate_ub,
            cate=cate, cate_lb=cate_lb, cate_ub=cate_ub,
            importances=importances,
            n=int(T.sum()),
        )

    return results, arm_names, X, Y, arms, names


# ── Output formatters ─────────────────────────────────────────────────────────

def print_ate(results, arm_names):
    print("━" * 65)
    print("1. AVERAGE TREATMENT EFFECT  (arm vs rest, 90% CI)")
    print("━" * 65)
    print(f"  {'Arm':<32} {'ATE':>8}  {'90% CI'}")
    print(f"  {'─'*32} {'─'*8}  {'─'*20}")
    for arm in arm_names:
        r = results[arm]
        sig = "✓" if r["ate_lb"] > 0 else ("✗" if r["ate_ub"] < 0 else "~")
        print(f"  {arm:<32} {r['ate']:>+8.4f}  "
              f"[{r['ate_lb']:+.4f}, {r['ate_ub']:+.4f}]  {sig}")
    print()
    print("  ✓ = significantly positive  ✗ = significantly negative  ~ = inconclusive")
    print()


def print_causal_assignment(results, arm_names, X):
    print("━" * 65)
    print("2. CAUSAL ARM ASSIGNMENT")
    print("   (which arm is causally best for each user in the dataset)")
    print("━" * 65)
    all_cates  = np.column_stack([results[a]["cate"] for a in arm_names])
    best_idx   = np.argmax(all_cates, axis=1)
    n_total    = len(best_idx)

    for i, arm in enumerate(arm_names):
        pct = (best_idx == i).mean() * 100
        bar = "█" * int(pct / 2)
        print(f"  {arm:<32}  {pct:5.1f}%  {bar}")
    print()

    # Compare causal assignment to what the bandit actually picked
    print("  Interpretation: if the bandit's arm distribution (Campaigns tab)")
    print("  roughly matches the percentages above, the model has converged")
    print("  to the correct causal structure. Large mismatches suggest the")
    print("  bandit is still exploring or the context features need rethinking.")
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
            name   = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            bar    = "█" * int(imp[idx] * 40)
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

        # For each feature, check if the arm's users have a notably
        # different mean than the rest — simple but interpretable
        lines = []
        for f_idx, name in enumerate(feature_names[:X.shape[1]]):
            mean_arm   = X_arm[:, f_idx].mean()
            mean_other = X_other[:, f_idx].mean() if len(X_other) > 0 else 0.5
            diff = mean_arm - mean_other
            if abs(diff) > 0.08:   # only report meaningful differences
                direction = "high" if diff > 0 else "low"
                lines.append(f"{name} ({direction}, Δ={diff:+.2f})")

        desc = ", ".join(lines) if lines else "no strong segment signal"
        print(f"  {arm}:")
        print(f"    {desc}")
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
    print("Fitting causal forests (one per arm) ...")
    print()

    results, arm_names, X, Y, arms, feature_names = run_causal_forest(df, args.features)

    print_ate(results, arm_names)
    print_causal_assignment(results, arm_names, X)
    print_feature_importance(results, arm_names, feature_names)
    print_winning_segments(results, arm_names, X, feature_names)

    print("━" * 65)
    print("WHAT TO DO WITH THIS")
    print("━" * 65)
    print("""
  ATE significant & positive → arm is genuinely causing reward
  ATE near zero              → arm and reward are uncorrelated causally
  ATE significant & negative → arm is actively hurting reward

  Causal assignment ≠ bandit distribution → model hasn't converged yet,
    or context features don't capture the relevant signal.

  Feature importance high for a feature → that dimension is driving
    who gets which arm. If it doesn't match your domain knowledge,
    investigate whether the feature is measured correctly.

  Winning segments → use these to design a/b tests or to audit
    whether the bandit is correctly routing each profile.
""")


if __name__ == "__main__":
    main()
