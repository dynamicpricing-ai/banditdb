# BanditDB as a Pretrained Decision Model

Technical report on prior-shaped contextual bandits, eligibility-aware off-policy
evaluation, and capacity-constrained serving — with the code and raw results behind
every number in it.

**[BanditDB_Pretrained_Decision_Model.pdf](BanditDB_Pretrained_Decision_Model.pdf)** — 18 pages, 9 theorems/propositions with proofs, 5 experiments, 5 figures.

## What it argues

Pretraining enters a contextual bandit at exactly one place: the prior mean `μ₀` and
its strength `λ`. Theorem 1 shows a warm-started ridge bandit on `θ*` is exactly a
cold-started one on `θ* − μ₀`, so everything a pretrained model contributes compresses
into the single scalar `‖θ* − μ₀‖`. That makes the value of pretraining measurable
before any transformer is trained, and gives a sharp help/harm criterion (Corollary 2).

Two results worth knowing before building on this:

- **A unit-strength prior changes no exploration at all** (Proposition 3), so warm
  starting at `λ = 1` buys nothing in regret. Measured twice, once at 20 seeds.
  The benefit is real but has to be purchased with `λ > 1`, which is exactly when a
  mis-specified prior becomes expensive.
- **Propensities computed after eligibility filtering bias IPS by +106%**, while SNIPS
  silently absorbs the same error (−0.93%). A tournament evaluated with SNIPS passes
  its own checks while its logs are wrong (Theorem 5, Proposition 6).

## Layout

```
paper.html          source of the report; {{FIG_*}} placeholders are filled at build
build_paper.py      inlines figures, renders to PDF via headless Chrome
experiments/        one script per experiment, plus the reference implementation
results/            raw JSON output of the runs cited in the paper
figures/            figures as rendered from results/
```

## Reproducing

Experiments 8.1 and 8.5 drive a live server over HTTP; the rest run standalone.

```bash
# live-engine experiments need a running instance
cargo build
DATA_DIR=/tmp/paper_run PORT=8099 BANDITDB_API_KEYS="sim-admin-key=admin" \
  BANDITDB_RATE_LIMIT_PER_SEC=200000 ./target/debug/banditdb &

cd docs/pretrained_decision_model/experiments
OUT=../results/validate.json python3 exp_validate.py      # 8.1  reference validation
OUT=../results/prior.json    python3 exp_prior.py         # 8.2  prior sweep (offline)
OUT=../results/ope3.json     python3 exp_ope3.py          # 8.3  off-policy evaluation
OUT=../results/capacity.json python3 exp_capacity.py      # 8.4  allocation under capacity
SIM_SEED=7 SIM_TAG=s7 python3 sim_dynamic_arms.py ../results/results2.json   # 8.5

python3 figures.py            # regenerate figures/ from results/
cd .. && python3 build_paper.py
```

`exp_ope.py` and `exp_ope2.py` are the two superseded designs discussed in §8.3. They
are kept because their failures are findings in their own right: LinUCB is
deterministic, so its softmax propensities are a heuristic rather than action
probabilities, and Monte-Carlo propensity estimates carry a ratio bias (+11.7% at
N=400) independent of any filtering question.

Requirements: `numpy`, `scipy`, `matplotlib`, `requests`; Chrome or Chromium for the
PDF step, which also needs network access to fetch MathJax.

## Caveat

Every reward in these experiments is synthetic, with a known `θ*` — necessary to
measure bias and regret against ground truth, and insufficient to establish that
pretraining helps in production. §9 of the paper states the bar: evaluation on real
logged bandit data with real propensities, plus a negative-transfer test, before any
pretrained prior ships.
