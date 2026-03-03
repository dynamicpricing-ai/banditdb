#!/usr/bin/env python3
"""
MovieLens 100K -> BanditDB WAL Converter
=========================================

Downloads MovieLens 100K and converts the rating log into two BanditDB WAL files:

  data/movielens_train.jsonl  -- first 90% of ratings (chronological)
  data/movielens_test.jsonl   -- remaining 10%, for replay evaluation

Bandit problem formulation
--------------------------
  Campaign:   movielens_recommendation
  Arms:       Drama | Comedy | Action | Romance | Thriller | Adventure
              (primary genre of each rated movie)
  Context:    [1.0, age_norm, is_male, occ_onehot×21]  (feature_dim=24, one-hot occupation + bias)
  Reward:     1.0 if rating >= 4 else 0.0  (binary: liked / not liked)

Loading the train WAL into BanditDB
------------------------------------
  docker cp benchmark/data/movielens_train.jsonl <container>:/data/bandit_wal.jsonl
  docker exec <container> rm -f /data/checkpoint.json
  docker compose restart

The server replays the WAL on startup, training the model on all 83K interactions.
Run benchmark/movielens/evaluate.py afterwards to measure CTR vs. random baseline.

Usage
-----
  pip install requests
  python benchmark/movielens/convert.py
"""

import json
import sys
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency — run:  pip install requests")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMPAIGN_ID = "movielens_recommendation"
ARMS        = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Adventure"]
N_OCC       = 21          # u.occupation has 21 occupations
FEATURE_DIM = 24          # 1 (bias) + 1 (age) + 1 (gender) + 21 (occ one-hot)
ALPHA       = 1.5
TRAIN_RATIO = 0.9

ML100K_URL  = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

HERE      = Path(__file__).parent
DATA_DIR  = HERE.parent / "data"
TRAIN_WAL = DATA_DIR / "movielens_train.jsonl"
TEST_WAL  = DATA_DIR / "movielens_test.jsonl"

# Genre column indices in u.item (0-indexed from the first genre column, which
# is column index 5 in the pipe-delimited file).
# Full order: unknown|Action|Adventure|Animation|Children's|Comedy|Crime|
#             Documentary|Drama|Fantasy|Film-Noir|Horror|Musical|Mystery|
#             Romance|Sci-Fi|Thriller|War|Western
GENRE_COLS = {
    "Action":    1,
    "Adventure": 2,
    "Comedy":    5,
    "Drama":     8,
    "Romance":   14,
    "Thriller":  16,
}

# Priority order when a movie belongs to multiple genres.
# Less common genres listed first to prevent Drama/Comedy monopolising arms.
GENRE_PRIORITY = ["Action", "Adventure", "Romance", "Thriller", "Comedy", "Drama"]


# ---------------------------------------------------------------------------
# Download & extract
# ---------------------------------------------------------------------------

def download_movielens() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-100k.zip"

    if not zip_path.exists():
        print(f"Downloading MovieLens 100K from {ML100K_URL} ...")
        r = requests.get(ML100K_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Saved {zip_path.stat().st_size / 1024:.0f} KB to {zip_path}")
    else:
        print("MovieLens 100K already downloaded — skipping.")

    extract_dir = DATA_DIR / "ml-100k-raw"
    if not extract_dir.exists():
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        print("  Done.")
    else:
        print("Already extracted — skipping.")

    return extract_dir / "ml-100k"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_occupations(ml_dir: Path) -> dict:
    """Returns {occupation_name: integer_id}."""
    return {
        line.strip(): i
        for i, line in enumerate((ml_dir / "u.occupation").read_text().strip().splitlines())
    }


def parse_users(ml_dir: Path, occupations: dict) -> dict:
    """
    Returns {user_id: context_vector} where context is the 24-dim one-hot+bias encoding:
      [1.0, age_norm, is_male, occ_0, occ_1, ..., occ_20]

    1.0        — bias term (lets each arm learn an intercept / base popularity)
    age_norm   — age / 73.0  (73 is the max age in the dataset)
    is_male    — 1.0 for M, 0.0 for F
    occ_0..20  — one-hot occupation (21 categories); avoids false ordinal proximity
    """
    users = {}
    for line in (ml_dir / "u.user").read_text().strip().splitlines():
        uid, age, gender, occ, _ = line.split("|")
        occ_id  = occupations.get(occ.strip(), 0)
        occ_oh  = [0.0] * N_OCC
        occ_oh[occ_id] = 1.0
        users[int(uid)] = [1.0, float(age) / 73.0, 1.0 if gender.strip() == "M" else 0.0] + occ_oh
    return users


def parse_movies(ml_dir: Path) -> dict:
    """
    Returns {movie_id: arm_id | None}.

    Assigns the first matching genre from GENRE_PRIORITY.
    Returns None for movies with no arm-eligible genre.
    """
    movies = {}
    for line in (ml_dir / "u.item").read_text(encoding="latin-1").strip().splitlines():
        parts = line.split("|")
        mid   = int(parts[0])
        flags = [int(x) for x in parts[5:]]
        arm   = next(
            (genre for genre in GENRE_PRIORITY if GENRE_COLS[genre] < len(flags) and flags[GENRE_COLS[genre]] == 1),
            None,
        )
        movies[mid] = arm
    return movies


def parse_ratings(ml_dir: Path) -> list:
    """Returns list of (user_id, movie_id, rating, timestamp), sorted by timestamp."""
    ratings = []
    for line in (ml_dir / "u.data").read_text().strip().splitlines():
        uid, mid, rating, ts = line.split("\t")
        ratings.append((int(uid), int(mid), int(rating), int(ts)))
    ratings.sort(key=lambda r: r[3])
    return ratings


# ---------------------------------------------------------------------------
# WAL writers
# ---------------------------------------------------------------------------

def write_campaign_created(f):
    f.write(json.dumps({"CampaignCreated": {
        "campaign_id": CAMPAIGN_ID,
        "arms":        ARMS,
        "feature_dim": FEATURE_DIM,
        "alpha":       ALPHA,
    }}) + "\n")


def write_interaction(f, arm_id: str, context: list, reward: float, ts: int):
    iid = str(uuid.uuid4())
    f.write(json.dumps({"Predicted": {
        "interaction_id": iid,
        "campaign_id":    CAMPAIGN_ID,
        "arm_id":         arm_id,
        "context":        [round(v, 6) for v in context],
        "timestamp_secs": ts,
    }}) + "\n")
    f.write(json.dumps({"Rewarded": {
        "interaction_id": iid,
        "reward":         round(reward, 4),
        "timestamp_secs": ts + 1,
    }}) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ml_dir      = download_movielens()
    occupations = parse_occupations(ml_dir)
    users       = parse_users(ml_dir, occupations)
    movies      = parse_movies(ml_dir)
    ratings     = parse_ratings(ml_dir)

    # Build interaction list — filter out ratings without user features or genre arm
    interactions = []
    skipped = 0
    for uid, mid, rating, ts in ratings:
        if uid not in users:
            skipped += 1
            continue
        arm = movies.get(mid)
        if arm is None:
            skipped += 1
            continue
        interactions.append((arm, users[uid], 1.0 if rating >= 4 else 0.0, ts))

    if skipped:
        print(f"Skipped {skipped:,} ratings (user or genre not in scope).")

    # Chronological 90/10 split
    split = int(len(interactions) * TRAIN_RATIO)
    train = interactions[:split]
    test  = interactions[split:]

    # Write train WAL (CampaignCreated + all train interactions)
    arm_counts   = defaultdict(int)
    reward_sums  = defaultdict(float)
    with open(TRAIN_WAL, "w") as f:
        write_campaign_created(f)
        for arm, context, reward, ts in train:
            write_interaction(f, arm, context, reward, ts)
            arm_counts[arm]  += 1
            reward_sums[arm] += reward

    # Write test WAL (interactions only — no CampaignCreated)
    with open(TEST_WAL, "w") as f:
        for arm, context, reward, ts in test:
            write_interaction(f, arm, context, reward, ts)

    # Summary
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  MovieLens 100K  ->  BanditDB WAL")
    print(sep)
    print(f"  Total usable interactions : {len(interactions):>8,}")
    print(f"  Train ({TRAIN_RATIO:.0%}) : {len(train):>8,}  ->  {TRAIN_WAL.name}")
    print(f"  Test  ({1-TRAIN_RATIO:.0%}) : {len(test):>8,}  ->  {TEST_WAL.name}")
    print(f"\n  Arm distribution in train set:")
    print(f"  {'Arm':<14} {'Count':>7}   {'Avg reward':>10}")
    print(f"  {'-'*36}")
    for arm in ARMS:
        count = arm_counts.get(arm, 0)
        avg   = reward_sums[arm] / count if count else 0
        print(f"  {arm:<14} {count:>7,}   {avg:>10.3f}")

    print(f"\n  Campaign : '{CAMPAIGN_ID}'")
    print(f"  Arms     : {ARMS}")
    print(f"  Features : [1.0 (bias), age_norm, is_male, occ_onehot×{N_OCC}]  (dim={FEATURE_DIM})")
    print(f"  Reward   : binary (1.0 if rating >= 4 else 0.0)")
    print(f"\n  Next steps:")
    print(f"  1. Load train WAL into BanditDB:")
    print(f"     docker cp {TRAIN_WAL} <container>:/data/bandit_wal.jsonl")
    print(f"     docker exec <container> rm -f /data/checkpoint.json")
    print(f"     docker compose restart")
    print(f"  2. Run replay evaluation:")
    print(f"     python benchmark/movielens/evaluate.py")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
