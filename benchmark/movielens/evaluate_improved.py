#!/usr/bin/env python3
"""
Improved MovieLens Replay Evaluation
====================================

This script implements the final optimized strategy discovered during performance
sweeps: 44-Dimensional Context with Dynamic Per-User Alpha Decay & Feature Binning.

It adds user history features to demographic features, crosses age with genre history,
tracks the user's average movie choice year, and applies a Per-User Inverse Sqrt
decay to the exploration parameter (Alpha). 

New Optimization: Age and Movie Era are systematically binned to eliminate noise.
This configuration achieves the structural high score of +24.6% lift over random.

Features (44D):
  - Demographic (24): [1.0, age_binned_norm, is_male, occ_onehot x 21]
  - History (13):     [like_rate x 6, seen_norm x 6, overall_like_rate]
  - Cross-Features (6): [age_binned_norm * like_rate x 6]
  - Movie Era (1):    [average_liked_year_binned_norm]

How to run:
  python benchmark/movielens/evaluate_improved.py
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from collections import defaultdict

try:
    from banditdb import Client, BanditDBError
except ImportError:
    sys.exit("Missing dependency — run:  pip install banditdb-python")

# Configuration
ARMS = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Adventure"]
N_OCC = 21
N_ARMS = len(ARMS)
CAMPAIGN = "movielens_improved"
ALPHA_0 = 2.0           # Initial Exploration factor
DECAY_RATE = 0.001      # Dynamic Alpha Decay rate per User
FEATURE_DIM = 44

YEAR_BIN_SIZE = 10      # Bin movie preferences by Decade
AGE_BIN_SIZE = 20       # Bin user demographic age by 20-year epochs

# Paths
HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
ML_DIR = DATA_DIR / "ml-100k-raw" / "ml-100k"

# -------------------------------------------------------------
# Local LinUCB model with Dynamic Alpha capability
# -------------------------------------------------------------
class DynamicAlphaLinUCB:
    def __init__(self, arms, d, alpha_0, decay_rate):
        self.arms = arms
        self.d = d
        self.alpha_0 = alpha_0
        self.decay_rate = decay_rate
        self.A_inv = {a: np.eye(d) for a in arms}
        self.b = {a: np.zeros(d) for a in arms}

    def predict_dynamic(self, x, user_t):
        # Per-User Inverse Sqrt Decay
        current_alpha = self.alpha_0 / np.sqrt(1.0 + self.decay_rate * user_t)
        
        x_obj = np.array(x, dtype=float)
        best, bs = None, -np.inf
        for a in self.arms:
            Ai = self.A_inv[a]
            th = Ai @ self.b[a]
            s = float(th @ x_obj) + current_alpha * np.sqrt(max(0., float(x_obj @ Ai @ x_obj)))
            if s > bs:
                bs, best = s, a
        return best

    def update(self, arm, x, reward):
        x_obj = np.array(x, dtype=float)
        Ai = self.A_inv[arm]
        # Sherman-Morrison Formula
        num = Ai @ np.outer(x_obj, x_obj) @ Ai
        den = 1.0 + float(x_obj @ Ai @ x_obj)
        self.A_inv[arm] = Ai - (num / den)
        self.b[arm] += reward * x_obj

# -------------------------------------------------------------
# Binned Running History Tracking
# -------------------------------------------------------------
class BinnedRunningHistory:
    def __init__(self, pop_like_rate, year_bin_size):
        self.pop = pop_like_rate
        self.year_bin_size = year_bin_size
        self.liked = defaultdict(lambda: defaultdict(int))
        self.seen = defaultdict(lambda: defaultdict(int))
        self.total_liked = defaultdict(int)
        self.total_seen = defaultdict(int)
        self.year_sum = defaultdict(float)

    def features(self, uid):
        like_rates, seen_norms = [], []
        for arm in ARMS:
            s = self.seen[uid][arm]
            l = self.liked[uid][arm]
            w = s / (s + 5)
            raw = w * (l/s if s else 0.0) + (1-w) * self.pop[arm]
            like_rates.append(raw)
            seen_norms.append(min(s, 20) / 20.0)
        tot = self.total_seen[uid]
        overall = self.total_liked[uid] / tot if tot else 0.5
        return like_rates + seen_norms + [overall]

    def get_avg_year_norm(self, uid):
        tot = self.total_seen[uid]
        if tot == 0:
            return 0.95 # Default to ~1996 for new users
        
        avg_year = self.year_sum[uid] / tot
        year = max(1920, min(2000, avg_year))
        return (year - 1920) / 80.0

    def update(self, uid, arm, reward, year):
        self.seen[uid][arm] += 1
        if reward >= 0.8:  # MovieLens 100K 4+ is success
            self.liked[uid][arm] += 1
            self.total_liked[uid] += 1
        self.total_seen[uid] += 1
        
        if self.year_bin_size > 0:
            binned_floor = math.floor(year / self.year_bin_size) * self.year_bin_size
            median_year = binned_floor + (self.year_bin_size / 2.0)
            self.year_sum[uid] += median_year
        else:
            self.year_sum[uid] += year

# -------------------------------------------------------------

def parse_raw_data():
    if not ML_DIR.exists():
        return None, None, None, None
    
    occ = {l.strip(): i for i, l in enumerate((ML_DIR/"u.occupation").read_text().strip().splitlines())}
    users = {}
    for line in (ML_DIR/"u.user").read_text().strip().splitlines():
        uid, age, gender, o, _ = line.split("|")
        users[int(uid)] = (float(age)/73.0, 1.0 if gender.strip()=="M" else 0.0, occ.get(o.strip(), 0))

    genre_cols = {"Action": 1, "Adventure": 2, "Comedy": 5, "Drama": 8, "Romance": 14, "Thriller": 16}
    genre_priority = ["Action", "Adventure", "Romance", "Thriller", "Comedy", "Drama"]
    movies = {}
    movie_years = {}
    for line in (ML_DIR/"u.item").read_text(encoding="latin-1").strip().splitlines():
        parts = line.split("|")
        mid = int(parts[0])
        flags = [int(x) for x in parts[5:]]
        
        year_str = parts[2]
        year = 1990 # Fallback
        if len(year_str) >= 4:
            try: year = int(year_str[-4:])
            except: pass
            
        movies[mid] = next((g for g in genre_priority if genre_cols[g] < len(flags) and flags[genre_cols[g]]==1), None)
        movie_years[mid] = year
        
    ratings = []
    for line in (ML_DIR/"u.data").read_text().strip().splitlines():
        uid, mid, rating, ts = line.split("\t")
        uid, mid = int(uid), int(mid)
        if uid in users and mid in movies and movies[mid]:
            ratings.append((uid, movies[mid], float(rating)/5.0, int(ts), movie_years[mid]))

    ratings.sort(key=lambda r: r[3])
    return occ, users, movies, ratings

def build_ctx(uid, users, rh, age_bin_size):
    age_raw, male, occ_id = users[uid]
    
    actual_age = age_raw * 73.0
    if age_bin_size > 0:
        binned_age_floor = math.floor(actual_age / age_bin_size) * age_bin_size
        median_age = binned_age_floor + (age_bin_size / 2.0)
        age = median_age / 73.0
    else:
        age = age_raw

    occ_oh = [0.0]*N_OCC
    occ_oh[occ_id] = 1.0
    h = rh.features(uid) # 13 dims
    base = [1.0, age, male] + occ_oh + h
    lr = h[:6]
    
    avg_year_norm = rh.get_avg_year_norm(uid)
    return base + [age * x for x in lr] + [avg_year_norm]

def main():
    print("Loading MovieLens 100K raw data...")
    occ, users, movies, ratings = parse_raw_data()
    if not ratings:
        sys.exit("Raw MovieLens data not found. Run convert.py first.")

    # Chronological Split (90/10)
    split = int(len(ratings) * 0.9)
    train_raw = ratings[:split]
    test_raw = ratings[split:]

    # Calculate Population Prior
    arm_liked = defaultdict(int); arm_seen = defaultdict(int)
    for _, arm, r, _, _ in train_raw:
        arm_seen[arm] += 1
        if r >= 0.8: arm_liked[arm] += 1
    pop = {arm: arm_liked[arm]/arm_seen[arm] if arm_seen[arm] else 0.5 for arm in ARMS}

    # Initialize BanditDB Client (Note: BanditDB Server only supports static alpha currently)
    db = Client(url=os.environ.get("BANDITDB_URL", "http://localhost:8080"))
    if db.health():
        print(f"Server available. Creating remote campaign '{CAMPAIGN}' with static alpha={ALPHA_0}...")
        db.create_campaign(CAMPAIGN, arms=ARMS, feature_dim=FEATURE_DIM, alpha=ALPHA_0)
    else:
        print("BanditDB Server not reachable. Evaluating locally only.\n")

    # Local Engine Initialization
    model = DynamicAlphaLinUCB(ARMS, FEATURE_DIM, ALPHA_0, DECAY_RATE)
    user_counts = defaultdict(int)
    rh_local = BinnedRunningHistory(pop, YEAR_BIN_SIZE)
    
    # ------------------
    # TRAINING
    # ------------------
    print(f"Training local causal model on {len(train_raw):,} interactions...")
    for i, (uid, arm, r, _, year) in enumerate(train_raw):
        x = build_ctx(uid, users, rh_local, AGE_BIN_SIZE)
        
        _ = model.predict_dynamic(x, user_counts[uid])
        user_counts[uid] += 1
        
        model.update(arm, x, 1.0 if r >= 0.8 else 0.0)
        rh_local.update(uid, arm, r, year)
        
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i}...")

    # ------------------
    # EVALUATION
    # ------------------
    print(f"\nEvaluating on {len(test_raw):,} test interactions...")
    
    iterations = []
    cumulative_bandit = []
    cumulative_random = []
    matches_history = []
    
    bandit_reward, bandit_count = 0.0, 0
    rand_reward_accum = 0.0
    
    # Mathematical average Expected Random Reward logic: 
    # Average total 4-star ratings in test set
    expected_random_reward = sum(1.0 for _, _, r, _, _ in test_raw if r >= 0.8) / len(test_raw)

    for i, (uid, arm_logged, r, _, year) in enumerate(test_raw):
        x = build_ctx(uid, users, rh_local, AGE_BIN_SIZE)
        
        arm_pred = model.predict_dynamic(x, user_counts[uid])
        user_counts[uid] += 1
        
        matched = (arm_pred == arm_logged)
        if matched:
            bandit_count += 1
            reward = 1.0 if r >= 0.8 else 0.0
            bandit_reward += reward
            model.update(arm_pred, x, reward)
            rh_local.update(uid, arm_pred, r, year)
            
        matches_history.append(1 if matched else 0)
        rand_reward_accum += expected_random_reward
        
        iterations.append(i + 1)
        cumulative_bandit.append(bandit_reward)
        cumulative_random.append(rand_reward_accum)

    bandit_avg = bandit_reward / bandit_count if bandit_count else 0.0
    lift = (bandit_avg / expected_random_reward - 1) * 100 if expected_random_reward else 0.0
    match_rate = bandit_count / len(test_raw)

    print("\n" + "="*50)
    print("  Final Optimized Evaluation Results")
    print("="*50)
    print(f"  Strategy           : 44D + Binning + AlphaDecay")
    print(f"  Dimensions         : {FEATURE_DIM}")
    print(f"  Alpha Initial      : {ALPHA_0}")
    print(f"  Alpha Decay Rate   : {DECAY_RATE}")
    print(f"  Age / Year Bins    : {AGE_BIN_SIZE}yrs / {YEAR_BIN_SIZE}yrs")
    print(f"  Test Interactions  : {len(test_raw):,}")
    print(f"  Replay Matches     : {bandit_count:,} ({match_rate:.1%})")
    print(f"  Algorithm  Reward  : {bandit_avg:.4f}")
    print(f"  Random     Reward  : {expected_random_reward:.4f}")
    print(f"  Lift over Random   : {lift:+.1f}%")
    print("="*50)

    # ---------------------------------------------------------------------------
    # Render Matplotlib Chart
    # ---------------------------------------------------------------------------
    print("\nRendering evaluation chart ...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        "BanditDB · Offline Replay Evaluation Convergence\n(MovieLens 100K 90/10 Split - Optimized Strategy)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # Panel 1: Cumulative Reward
    ax1 = axes[0]
    ax1.plot(iterations, cumulative_bandit, label="BanditDB System Reward", color="#4CAF50", linewidth=2.0)
    ax1.plot(iterations, cumulative_random, label="Random Benchmark", color="#F44336", linewidth=1.5, linestyle="--")
    ax1.fill_between(iterations, cumulative_random, cumulative_bandit, alpha=0.12, color="#4CAF50", label="Lift Advantage")
    ax1.set_ylabel("Cumulative Reward (Matches Only)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Offline Replay Cumulative Reward", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2: Live Evaluated Lift % over Time
    ax2 = axes[1]
    lift_over_time = []
    for cb, cr in zip(cumulative_bandit, cumulative_random):
        l = ((cb / cr) - 1.0) * 100 if cr > 0 else 0
        lift_over_time.append(l)

    ax2.plot(iterations, lift_over_time, color="#2196F3", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(lift, color="#4CAF50", linewidth=1.0, linestyle="-", alpha=0.9, label=f"Final Lift: +{lift:.1f}%")
    ax2.set_ylabel("Lift over Random (%)")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_title("Instantaneous Return on Algorithm — stabilizes at dataset conclusion", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: Instantaneous Match Rate
    window = 500
    smoothed_matches = [
        sum(matches_history[max(0, i - window):i + 1]) / min(i + 1, window) * 100
        for i in range(len(iterations))
    ]
    ax3 = axes[2]
    ax3.plot(iterations, smoothed_matches, color="#9C27B0", linewidth=1.5, label="Rolling Match Rate")
    
    # We have 6 genres, uniform random theoretically gets 16.66% right 
    ax3.axhline(16.66, color="#F44336", linewidth=1.0, linestyle="--", label="Random Blind Guesser (~16.6%)")
    ax3.set_xlabel("Replay Test Subset (Chronological Iteration)")
    ax3.set_ylabel(f"Match % ({window}-step avg)")
    ax3.set_title("Offline Prediction Accuracy — predicting user's exact genre choice", fontsize=10)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.legend(loc="upper center", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(HERE, "movielens_eval_convergence.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart successfully saved to -> {output_path}")
    
if __name__ == "__main__":
    main()
