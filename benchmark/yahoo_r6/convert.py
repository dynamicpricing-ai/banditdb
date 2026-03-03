#!/usr/bin/env python3
"""
Yahoo! R6 Dataset -> BanditDB WAL Converter
============================================

The R6 dataset is the gold standard for contextual bandit benchmarking.
It is the exact dataset used in the original LinUCB paper (Li et al. 2010)
and contains 45 million real news recommendation events from Yahoo! Front Page.

How to get access
-----------------
1. Go to https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=49
2. Sign in with a Yahoo account and request access (free, approval takes 1-2 days)
3. Download:
     ydata-fp-td-clicks-v1_0.20090501.gz
     ydata-fp-td-clicks-v1_0.20090502.gz
     ...
     ydata-fp-td-clicks-v1_0.20090510.gz   (10 days of data)
4. Extract all files into:  benchmark/data/yahoo_r6/

Dataset format
--------------
Each line in the raw file:

  timestamp  article_id  click  |user  f1:v1 f2:v2 f3:v3 f4:v4 f5:v5 f6:v6  |article_id  f1:v1 ...

Fields:
  timestamp:   Unix epoch seconds
  article_id:  ID of the article that was actually displayed
  click:       1 if the user clicked, 0 if not  (this is the reward)
  |user:       6 user feature values (including a constant 1.0)
  |article_id: 6 article feature values per candidate article

Bandit problem formulation
--------------------------
  Campaign:   yahoo_news_recommendation
  Arms:       top-20 article IDs by frequency in the log
              (use the first day's data to identify the top arms)
  Context:    6 user features  (feature_dim=6)
  Reward:     click  ->  1.0 if clicked, 0.0 if not

Evaluation: the Replay Method (Li et al. 2010)
-----------------------------------------------
The logging policy in R6 is uniform random — each article is equally likely
to be shown. This enables the UNBIASED replay estimator:

  For each event in the log:
    1. Ask BanditDB which article to show:  arm = db.predict(context)
    2. If arm == logged article_id: COUNT this event (use the logged reward)
    3. If arm != logged article_id: SKIP (do not reward, do not count)

  Estimate CTR = total_reward / total_counted_events

Because the logging policy is uniform random, the probability of a match is
1/K (K = number of arms). You typically need ~10M events to get a stable
estimate with 20 arms. Use all 10 days.

Comparison baseline
-------------------
  Random policy CTR = average click rate in the raw log (~0.03-0.05)
  LinUCB paper reports ~5% improvement over random after convergence.
  BanditDB should match or exceed that.

Usage (once you have the data in benchmark/data/yahoo_r6/)
-----------------------------------------------------------
  python benchmark/yahoo_r6/convert.py --data-dir benchmark/data/yahoo_r6/
"""

import sys


def main():
    print("Yahoo R6 converter — waiting for dataset access.")
    print()
    print("Follow the instructions in the docstring at the top of this file:")
    print("  https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=49")
    print()
    print("Once you have the data, this script will be implemented to:")
    print("  1. Parse the raw click logs")
    print("  2. Identify the top-20 articles by frequency")
    print("  3. Write data/yahoo_r6_train.jsonl and data/yahoo_r6_test.jsonl")
    print("  4. Run the unbiased replay evaluation against BanditDB")
    sys.exit(0)


if __name__ == "__main__":
    main()
