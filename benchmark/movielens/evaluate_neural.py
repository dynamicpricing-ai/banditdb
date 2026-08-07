#!/usr/bin/env python3
"""
NeuralLinUCB MovieLens Replay Evaluation
========================================

This script compares BanditDB's NeuralLinUCB implementation against the
previously optimized LinUCB results (+24.6% lift).

It uses the same 44-dimensional context as evaluate_improved.py but delegates
the learning and prediction to the BanditDB server running with --features neural.

Usage:
  # 1. Build and start server:
  # cargo build --release --features neural
  # ./target/release/banditdb
  
  # 2. Run evaluation:
  # python benchmark/movielens/evaluate_neural.py
"""

import os
import sys
import math
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
CAMPAIGN = "movielens_neural"
DB_URL   = os.environ.get("BANDITDB_URL", "http://localhost:8080")
API_KEY  = os.environ.get("BANDITDB_API_KEY", "dev-key")

ARMS = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Adventure"]
N_OCC = 21
N_ARMS = len(ARMS)
FEATURE_DIM = 44

YEAR_BIN_SIZE = 10
AGE_BIN_SIZE = 20

# Paths
HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
ML_DIR = DATA_DIR / "ml-100k-raw" / "ml-100k"

# --- Helper functions ---

def parse_raw_data():
    occ = {}
    occ_file = ML_DIR / "u.occupation"
    if not occ_file.exists(): return None, None, None, None
    for i, line in enumerate(occ_file.read_text().splitlines()):
        occ[line.strip()] = i

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

class BinnedRunningHistory:
    def __init__(self, pop_like_rate, year_bin_size):
        self.pop = pop_like_rate
        self.year_bin_size = year_bin_size
        self.liked = defaultdict(lambda: defaultdict(int))
        self.seen = defaultdict(lambda: defaultdict(int))
        self.user_years = defaultdict(list)

    def update(self, uid, arm, r, year):
        self.seen[uid][arm] += 1
        if r >= 0.8:
            self.liked[uid][arm] += 1
            self.user_years[uid].append(year)

    def features(self, uid):
        res = []
        tot_liked = 0
        tot_seen = 0
        for a in ARMS:
            l = self.liked[uid][a]
            s = self.seen[uid][a]
            tot_liked += l
            tot_seen += s
            res.append(l/s if s > 0 else self.pop[a])
        for a in ARMS:
            res.append(min(1.0, self.seen[uid][a]/10.0))
        res.append(tot_liked/tot_seen if tot_seen > 0 else sum(self.pop.values())/len(ARMS))
        return res

    def get_avg_year_norm(self, uid):
        years = self.user_years[uid]
        if not years: return 0.5
        avg_year = sum(years) / len(years)
        binned = math.floor(avg_year / self.year_bin_size) * self.year_bin_size
        return (binned - 1920) / 80.0

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
    h = rh.features(uid)
    base = [1.0, age, male] + occ_oh + h
    lr = h[:6]
    avg_year_norm = rh.get_avg_year_norm(uid)
    return base + [age * x for x in lr] + [avg_year_norm]

# --- BanditDB Client wrapper ---

class BanditClient:
    def __init__(self, url, api_key):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({"X-Api-Key": api_key})

    def delete_campaign(self, cid):
        self.session.delete(f"{self.url}/campaign/{cid}")

    def create_neural_campaign(self, cid, arms, context_dim):
        payload = {
            "campaign_id": cid,
            "arms": arms,
            "feature_dim": 0,
            "alpha": 0.1, # Tighten exploration
            "algorithm": {
                "neural_lin_ucb": {
                    "context_dim": context_dim,
                    "embed_dim": 12,
                    "hidden_dim": 32,
                    "hidden_layers": 2,
                    "retrain_every": 200,
                    "retrain_steps": 1000,
                    "learning_rate": 0.0001,
                    "lambda": 0.001,
                }
            }
        }
        r = self.session.post(f"{self.url}/campaign", json=payload)
        r.raise_for_status()

    def predict(self, cid, context):
        r = self.session.post(f"{self.url}/predict", json={"campaign_id": cid, "context": context})
        r.raise_for_status()
        return r.json()["arm_id"], r.json()["interaction_id"]

    def reward(self, iid, reward):
        self.session.post(f"{self.url}/reward", json={"interaction_id": iid, "reward": reward})

    def interact(self, cid, arm_id, context, reward):
        payload = {"arm_id": arm_id, "context": context, "reward": reward}
        r = self.session.post(f"{self.url}/campaign/{cid}/interact", json=payload)
        r.raise_for_status()

    def checkpoint(self):
        self.session.post(f"{self.url}/checkpoint")

# --- Main script ---

def main():
    print("Loading MovieLens 100K raw data...")
    occ, users, movies, ratings = parse_raw_data()
    if not ratings:
        sys.exit("Raw MovieLens data not found. Run convert.py first.")

    split = int(len(ratings) * 0.9)
    train_raw = ratings[:split]
    test_raw = ratings[split:]

    arm_liked = defaultdict(int); arm_seen = defaultdict(int)
    for _, arm, r, _, _ in train_raw:
        arm_seen[arm] += 1
        if r >= 0.8: arm_liked[arm] += 1
    pop = {arm: arm_liked[arm]/arm_seen[arm] if arm_seen[arm] else 0.5 for arm in ARMS}

    client = BanditClient(DB_URL, API_KEY)
    print(f"Resetting campaign '{CAMPAIGN}'...")
    client.delete_campaign(CAMPAIGN)
    client.create_neural_campaign(CAMPAIGN, ARMS, FEATURE_DIM)

    rh = BinnedRunningHistory(pop, YEAR_BIN_SIZE)
    
    # ------------------
    # TRAINING
    # ------------------
    print(f"Training NeuralLinUCB on {len(train_raw):,} interactions...")
    start_t = time.time()
    for i, (uid, arm, r, _, year) in enumerate(train_raw):
        x = build_ctx(uid, users, rh, AGE_BIN_SIZE)
        
        # Use the new /interact endpoint for forced offline training
        client.interact(CAMPAIGN, arm, x, 1.0 if r >= 0.8 else 0.0)
        rh.update(uid, arm, r, year)
        
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_t
            print(f"  Processed {i}... ({i/elapsed:.1f} req/s)")
            if i % 2000 == 0:
                client.checkpoint() # Trigger retrain

    client.checkpoint() # Final training checkpoint

    # ------------------
    # EVALUATION
    # ------------------
    print(f"\nEvaluating on {len(test_raw):,} test interactions...")
    
    bandit_reward, bandit_count = 0.0, 0
    expected_random_reward = sum(1.0 for _, _, r, _, _ in test_raw if r >= 0.8) / len(test_raw)
    
    start_t = time.time()
    for i, (uid, arm_logged, r, _, year) in enumerate(test_raw):
        x = build_ctx(uid, users, rh, AGE_BIN_SIZE)
        
        arm_pred, iid = client.predict(CAMPAIGN, x)
        
        matched = (arm_pred == arm_logged)
        if matched:
            bandit_count += 1
            reward = 1.0 if r >= 0.8 else 0.0
            bandit_reward += reward
            client.reward(iid, reward)
            rh.update(uid, arm_pred, r, year)
            
        if i % 200 == 0 and i > 0:
            client.checkpoint() # Trigger retrain during evaluation too
            elapsed = time.time() - start_t
            print(f"  Evaluated {i}/{len(test_raw)}... ({i/elapsed:.1f} req/s)")

    bandit_avg = bandit_reward / bandit_count if bandit_count else 0.0
    lift = (bandit_avg / expected_random_reward - 1) * 100 if expected_random_reward else 0.0
    
    print("\n" + "="*50)
    print("  NeuralLinUCB Evaluation Results")
    print("="*50)
    print(f"  Test Interactions  : {len(test_raw):,}")
    print(f"  Replay Matches     : {bandit_count:,} ({bandit_count/len(test_raw):.1%})")
    print(f"  Random Avg Reward  : {expected_random_reward:.4f}")
    print(f"  Neural Avg Reward  : {bandit_avg:.4f}")
    print(f"  LIFT OVER RANDOM   : {lift:+.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
