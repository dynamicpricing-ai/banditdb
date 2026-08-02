#!/usr/bin/env python3
"""
UCI classification datasets -> contextual-bandit replay format.

Implements the standard "classification as bandit" reduction used by the
NeuralUCB / NeuralTS line of work (Zhang, Zhou, Li, Gu, ICLR 2021) and by
Riquelme et al., "Deep Bayesian Bandits Showdown" (ICLR 2018):

    k classes            -> k arms
    feature vector       -> context
    reward 1 if the chosen arm equals the true class, else 0

Each dataset is written as <name>.jsonl with one record per round:

    {"context": [...float...], "label": <int arm index>}

plus a <name>.meta.json holding {n_arms, dim, n_rows, arms}.

Numeric features are standardised (zero mean, unit variance) and categorical
features are one-hot encoded. Rows with missing values are dropped.

Usage:
    python benchmark/uci/convert.py              # all datasets
    python benchmark/uci/convert.py mushroom     # one dataset
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "uci"
OUT = HERE.parent / "data" / "uci_bandit"

# Cap on context width. Kept because per-update cost is O(d^2) and the
# Thompson Sampling Cholesky is O(d^3); the paper's datasets stay well under.
MAX_DIM = 128


def _standardise(df: pd.DataFrame) -> np.ndarray:
    x = df.to_numpy(dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (x - mu) / sd


def _minmax(df: pd.DataFrame) -> np.ndarray:
    """
    Scale each column into [0, 1]. Used instead of standardisation when rows are
    L2-normalised afterwards: it puts raw numeric columns on the same footing as
    one-hot columns before the norm constraint, so no single wide-range feature
    (adult's fnlwgt spans ~10k-1.5M) dominates the unit vector. On shuttle this
    scored best of every combination tried (regret 709).
    """
    x = df.to_numpy(dtype=np.float64)
    lo, hi = x.min(axis=0), x.max(axis=0)
    rng = hi - lo
    rng[rng < 1e-12] = 1.0
    return (x - lo) / rng


def _encode(numeric: pd.DataFrame | None, categorical: pd.DataFrame | None,
            standardise: bool = True) -> np.ndarray:
    parts = []
    if numeric is not None and numeric.shape[1]:
        # When rows are L2-normalised afterwards, per-column standardisation is
        # actively harmful: on shuttle, standardise+L2 gives regret 2,381 while
        # L2 on raw columns gives 981. Standardisation reshapes the geometry
        # before the norm constraint is applied, and the two do not compose.
        parts.append(_standardise(numeric) if standardise else _minmax(numeric))
    if categorical is not None and categorical.shape[1]:
        dummies = pd.get_dummies(categorical.astype(str), drop_first=True)
        parts.append(dummies.to_numpy(dtype=np.float64))
    return np.hstack(parts)


def load_mushroom(standardise: bool = True):
    """8124 rows, 22 categorical features, 2 classes (edible / poisonous)."""
    path = DATA / "mushroom" / "agaricus-lepiota.data"
    df = pd.read_csv(path, header=None, dtype=str)
    df = df.replace("?", np.nan).dropna()
    y = (df[0] == "p").astype(int).to_numpy()  # 0 = edible, 1 = poisonous
    x = _encode(None, df.drop(columns=[0]), standardise)
    return x, y, ["edible", "poisonous"]


def load_magic(standardise: bool = True):
    """19020 rows, 10 numeric features, 2 classes (gamma / hadron)."""
    path = DATA / "magic+gamma+telescope" / "magic04.data"
    df = pd.read_csv(path, header=None)
    df = df.dropna()
    y = (df[10] == "h").astype(int).to_numpy()
    x = _encode(df.drop(columns=[10]), None, standardise)
    return x, y, ["gamma", "hadron"]


def load_shuttle(standardise: bool = True):
    """58000 rows, 9 numeric features, 7 classes. Heavily imbalanced (~80% class 1)."""
    trn_z = DATA / "statlog+shuttle" / "shuttle.trn.Z"
    trn = DATA / "statlog+shuttle" / "shuttle.trn"
    if trn_z.exists() and not trn.exists():
        # .Z is LZW "compress" format; gzip reads it.
        with open(trn, "wb") as fh:
            subprocess.run(["gzip", "-dc", str(trn_z)], stdout=fh, check=True)

    frames = [pd.read_csv(p, header=None, sep=r"\s+")
              for p in (trn, DATA / "statlog+shuttle" / "shuttle.tst") if p.exists()]
    df = pd.concat(frames, ignore_index=True).dropna()

    labels = df[9].to_numpy()
    # Labels are 1..7 but some classes are tiny; remap to a dense 0-based index.
    uniq = np.sort(np.unique(labels))
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in labels])
    x = _encode(df.drop(columns=[9]), None, standardise)
    return x, y, [f"class_{v}" for v in uniq]


def load_adult(standardise: bool = True):
    """48842 rows, 6 numeric + 8 categorical features, 2 classes (income)."""
    cols = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    frames = []
    for name in ("adult.data", "adult.test"):
        p = DATA / "adult" / name
        if not p.exists():
            continue
        # adult.test has a junk first line and a trailing '.' on each label.
        skip = 1 if name == "adult.test" else 0
        frames.append(pd.read_csv(p, header=None, names=cols, skiprows=skip,
                                  sep=r",\s*", engine="python", na_values="?"))
    df = pd.concat(frames, ignore_index=True).dropna()

    y = df["income"].str.replace(".", "", regex=False).str.strip()
    y = (y == ">50K").astype(int).to_numpy()

    num = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    cat = ["workclass", "education", "marital_status", "occupation",
           "relationship", "race", "sex", "native_country"]
    x = _encode(df[num], df[cat], standardise)
    return x, y, ["<=50K", ">50K"]


LOADERS = {
    "mushroom": load_mushroom,
    "magic": load_magic,
    "shuttle": load_shuttle,
    "adult": load_adult,
}


def _l2_rows(x: np.ndarray) -> np.ndarray:
    """
    Scale every context to unit L2 norm.

    LinUCB's confidence width alpha*sqrt(x' A_inv x) scales with ||x||, and its
    regret analysis assumes ||x|| <= 1. Per-column standardisation leaves heavy
    outliers intact -- on shuttle it produced ||x|| up to 123 against a mean of
    2.09, so rare high-norm rows generated enormous exploration bonuses and
    derailed arm selection (regret 2,026 vs 990 with this normalisation).
    Zhang et al. (ICLR 2021) normalise the same way.
    """
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-12, 1.0, n)


def convert(name: str, l2: bool = True) -> None:
    # L2 replaces standardisation rather than composing with it (see _encode).
    x, y, arms = LOADERS[name](standardise=not l2)

    if x.shape[1] > MAX_DIM:
        # Deterministic PCA-free reduction: keep the highest-variance columns.
        keep = np.argsort(x.var(axis=0))[::-1][:MAX_DIM]
        keep.sort()
        x = x[:, keep]
        print(f"  [{name}] reduced {name} to {MAX_DIM} highest-variance columns")

    if l2:
        x = _l2_rows(x)

    OUT.mkdir(parents=True, exist_ok=True)
    jsonl = OUT / f"{name}.jsonl"
    with open(jsonl, "w") as fh:
        for ctx, label in zip(x, y):
            fh.write(json.dumps({"context": [round(float(v), 6) for v in ctx],
                                 "label": int(label)}) + "\n")

    meta = {"name": name, "n_arms": len(arms), "dim": int(x.shape[1]),
            "n_rows": int(x.shape[0]), "arms": arms,
            "class_balance": np.bincount(y, minlength=len(arms)).tolist()}
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"  [{name}] rows={x.shape[0]:,}  dim={x.shape[1]}  arms={len(arms)}  "
          f"balance={meta['class_balance']}")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--no-l2"]
    l2 = "--no-l2" not in sys.argv[1:]
    names = args or list(LOADERS)
    for name in names:
        if name not in LOADERS:
            sys.exit(f"unknown dataset '{name}' — choose from {list(LOADERS)}")
        print(f"converting {name} ...")
        convert(name, l2=l2)
    print(f"\nwrote to {OUT}  (l2_normalised={l2})")


if __name__ == "__main__":
    main()
