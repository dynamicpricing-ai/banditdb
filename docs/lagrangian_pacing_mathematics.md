# Mathematical Foundations of Lagrangian Pacing in BanditDB

**Authors:** Dynamic Pricing AI — Research & Core Engineering  
**Scope:** Bandits-with-Knapsacks (BwK), Online Dual Descent, Paced Exploration, and Counterfactual Inference  
**Document Status:** Complete Mathematical Specification & Technical Monograph (Revisions I – V)

---

## Executive Abstract

BanditDB delivers low-latency contextual decisions under strict service-level agreements ($\le 100\,\mu\text{s}$ at $p99$). In production environments, decision-making is rarely unconstrained: businesses operate under hard resource limitations, such as call-center staff capacity, promotional discount budgets, SMS/communication quotas, or third-party LLM rate limits. Imposing resource budgets over a finite horizon transforms the unconstrained contextual bandit into the **Contextual Bandits-with-Knapsacks (BwK)** problem.

Directly solving the exact per-decision Linear Program (LP) relaxation at request time incurs 10–50 milliseconds of latency, scaling poorly with the number of arms $K$ and resources $m$. This document provides the mathematical foundations, algorithmic derivations, economic interpretations, and operational invariants for **Lagrangian Pacing via Online Dual Mirror Descent**.

By reformulating global knapsack constraints into local shadow prices $\lambda \in \mathbb{R}_+^m$, the per-step decision rule remains a **single argmax** requiring only $O(K \cdot m)$ multiply-adds. This preserves BanditDB's sub-microsecond evaluation hot path while achieving optimal $O(\sqrt{T})$ regret against the hindsight fluid LP benchmark. In addition, this monograph resolves critical production hazards, including downstream propensity corruption in off-policy evaluation (IPS/SNIPS) and time-varying confounding in causal uplift modeling (CATE).

---

## Table of Contents

1. [Motivation & The Failure of Unconstrained Bandits in Production](#1-motivation--the-failure-of-unconstrained-bandits-in-production)
   - 1.1 Commercial Case Studies (Where Budgets Exist)
   - 1.2 Why Naive Heuristics Fail
   - 1.3 The Design Objective: Latency, Regret, and Observability
2. [Revision I: Problem Formulation & The Fluid Benchmark](#2-revision-i-problem-formulation--the-fluid-benchmark)
   - 2.1 The Contextual Bandits-with-Knapsacks (BwK) Setting
   - 2.2 The Offline Fluid Linear Program
   - 2.3 The Fluid Benchmark Upper Bound (Theorem & Proof)
   - 2.4 Lagrangian Relaxation & Duality Theory
   - 2.5 Complementary Slackness & The Dual Optimum
3. [Revision II: The Per-Step Paced Policy & Economic Interpretation](#3-revision-ii-the-per-step-paced-policy--economic-interpretation)
   - 3.1 Temporal Decoupling: From Global LP to Local Argmax
   - 3.2 $\lambda_j$ as an Economic Shadow Price & Opportunity Cost
   - 3.3 The Uplift Threshold Rule (A Concrete Worked Example)
   - 3.4 Paced LinUCB Policy Formulation
   - 3.5 Paced Thompson Sampling Policy Formulation
   - 3.6 The Safety Net: Hard Feasibility Masking ($M_t$)
4. [Revision III: Online Dual Learning Dynamics & Regret Guarantees](#4-revision-iii-online-dual-learning-dynamics--regret-guarantees)
   - 4.1 Convexity of the Dual Function & Danskin's Theorem
   - 4.2 Projected Online Gradient Descent (OGD)
   - 4.3 Integrator Windup & The Theoretical Maximum Multiplier ($\lambda^{\max}$)
   - 4.4 Entropic Dual Mirror Descent (OMD) & The Zero-Trap Vulnerability
   - 4.5 Target Consumption Rates: Uniform, Adaptive, and Periodic Profiles
   - 4.6 Theoretical Regret Bound ($O(\sqrt{T})$ Regret Proof Breakdown)
5. [Revision IV: Counterfactual Estimation & Propensity Corrections](#5-revision-iv-counterfactual-estimation--propensity-corrections)
   - 5.1 The Constrained Assignment Probability $\pi(a \mid x, \lambda, M)$
   - 5.2 The "Price Before Propensity" Theorem & Proof
   - 5.3 Step-by-Step Numerical Breakdown: How IPS Incurs +106% Bias
   - 5.4 Why SNIPS Masks Propensity Distortion (−0.93% False Sense of Security)
   - 5.5 Positivity Violations & Support Truncation Under $M_t$
   - 5.6 Time-Varying Confounding in CATE & Uplift Modeling (DAG Analysis)
6. [Revision V: Boundary Invariants, Edge Cases & Operational Architecture](#6-revision-v-boundary-invariants-edge-cases--operational-architecture)
   - 6.1 Horizon Singularity as $T^{\mathrm{rem}} \to 0$ (The Endgame Freeze Invariant)
   - 6.2 Consumption-Nonconsumption Update Asymmetry (Variance Scaling)
   - 6.3 Dimensional Consistency: Raw vs. Normalized Costs
   - 6.4 Progressive Tournament Confounding (SUTVA Violations Under Shared Multipliers)
   - 6.5 Inter-Window Warm-Starting (Exponential Multiplier Smoothing)
   - 6.6 Zero-Overhead Integration & Rollback-Safe Architecture
7. [System Architecture Pipeline](#7-system-architecture-pipeline)
8. [References](#references)

---

## 1. Motivation & The Failure of Unconstrained Bandits in Production

### 1.1 Commercial Case Studies (Where Budgets Exist)

Standard multi-armed and contextual bandits operate under the premise that **actions are free to execute**, meaning the only objective is to maximize reward. In industrial applications, actions consume scarce, costly, or physical resources.

#### Case Study A: Outbound Call-Center Retention Interventions
* **Environment:** A telecommunications provider uses BanditDB to decide customer retention actions: `[Do Nothing, Send Push Notification, Offer 10% Discount, Schedule Outbound Phone Call]`.
* **Constraint:** The human call center employs 20 agents who can complete at most 200 calls per day ($B = 200$). The incoming stream of churn-risk customer events is $T = 10,000$ per day.
* **The Failure Mode:** An unconstrained bandit learns that phone calls have the highest raw probability of preventing churn across almost all users. Left unconstrained, the bandit selects phone calls for the first 200 users arriving between 08:00 and 08:15 AM. At 08:16 AM, call capacity is exhausted. The call center goes dark for the remaining 9,800 users—many of whom had a 10× higher churn risk than the morning users.

#### Case Study B: E-Commerce Promotional Spend & Flash Sales
* **Environment:** An e-commerce platform uses BanditDB to dynamically allocate promotional coupon vouchers `[$0, $5, $15, $30]` to incoming checkout sessions.
* **Constraint:** The marketing budget is capped at \$10,000 per day ($B = 10,000$). The daily checkout volume is $T = 100,000$ shoppers.
* **The Failure Mode:** Without pacing, high-value coupons are offered aggressively during the early hours to shoppers who would have purchased organically anyway. When high-intent, price-sensitive shoppers arrive in the evening peak, the promo budget is spent.

#### Case Study C: Multi-Provider LLM Agent Tool Routing
* **Environment:** An AI coding agent uses BanditDB to route incoming programming tasks between `[Local 8B Model, Claude 3.5 Haiku, Claude 3.5 Sonnet, OpenAI o1]`.
* **Constraint:** Premium frontier models (Sonnet and o1) have strict API token quotas and rate limits (e.g., maximum 500,000 tokens per hour or \$50/hour budget).
* **The Failure Mode:** Simple queries consume the entire frontier rate limit within the first 10 minutes of the hour, forcing difficult architectural queries to fall back to small local models.

---

### 1.2 Why Naive Heuristics Fail

Engineering teams often attempt to address resource budgets using ad-hoc engineering heuristics. In practice, each fails catastrophically:

| Heuristic | Implementation | Why It Fails |
|---|---|---|
| **1. First-Come, First-Served (FCFS) Hard Cap** | Serve unconstrained bandit predictions until $B = 0$; then disable the arm. | **Catastrophic front-loading.** Spends 100% of the budget on mediocre morning requests with marginal uplift ($\Delta = 0.01$); denies afternoon requests with massive uplift ($\Delta = 0.85$). Achieves $< 40\%$ of optimal value. |
| **2. Uniform Random Throttling** | Statically downsample arm availability (e.g., allow arm to be considered only with probability $\rho = B/T$). | **Context-blind.** Rejects high-value contexts with probability $1 - \rho$; accepts low-value contexts with probability $\rho$. Forfeits the primary benefit of contextual personalization. |
| **3. Static Thresholding** | Only pull arm if predicted reward $\hat{r} > \tau_{\text{static}}$. | **Breaks under non-stationarity.** If traffic surges, budget is exhausted early. If traffic drops or features drift, budget is left unspent ($< 60\%$ utilization). |
| **4. Online Per-Request Linear Programming** | Solve an exact LP at every incoming request using current state estimates. | **Violates latency budgets.** Running an interior-point or simplex solver takes 10–50 ms. At 5,000 req/sec, request queues explode, crashing the database engine. |

---

### 1.3 The Design Objective: Latency, Regret, and Observability

To satisfy enterprise infrastructure requirements, any capacity-constrained bandit system must simultaneously fulfill four criteria:
1. **Sub-Microsecond Hot-Path Latency:** The decision path must remain $O(K \cdot m)$ in time complexity, performing only scalar multiply-adds with no external solver invocations.
2. **Optimal Regret:** The algorithm must converge to the global optimal offline allocation, achieving sublinear $O(\sqrt{T})$ regret against the hindsight benchmark.
3. **Smooth Capacity Pacing:** Resource consumption must be spread stably across the decision window, preventing early depletion while ensuring $> 98\%$ budget utilization.
4. **Counterfactual Integrity:** Downstream causal inference, policy evaluation (OPE), and offline A/B testing must remain statistically valid and uncorrupted.

---

## 2. Revision I: Problem Formulation & The Fluid Benchmark

### 2.1 The Contextual Bandits-with-Knapsacks (BwK) Setting

Let $t \in \{1, \dots, T\}$ index discrete decision steps over a known or estimated horizon $T \in \mathbb{N}$.  
Let $\mathcal{A} = \{1, \dots, K\}$ denote the action space of $K$ discrete arms.  
Let $j \in \{1, \dots, m\}$ index $m$ shared resource constraints.

At each step $t$:
1. The environment samples a context vector $x_t \in \mathcal{X} \subseteq \mathbb{R}^d$ from an underlying, stationary distribution $\mathcal{D}_X$.
2. The agent observes $x_t$ and selects an action $a_t \in \mathcal{A}$.
3. The selected arm consumes resources according to a cost vector $c_{a_t} = (c_{1, a_t}, \dots, c_{m, a_t})^\top \in [0, 1]^m$. Without loss of generality, we assume $c_{j,a} \ge 0$.
4. The agent observes a bounded reward $r_t(a_t) \in [0, 1]$, where:
   $$\mathbb{E}[r_t(a) \mid x_t] = \mu_a(x_t) = x_t^\top \theta_a^*$$
   with $\theta_a^* \in \mathbb{R}^d$ representing the unknown true parameter vector for arm $a$.

Each resource $j$ has a total initial capacity budget $B_j > 0$. The decision process terminates at horizon $T$ or when the cumulative consumption of any resource exceeds its budget. We define the **stopping time** $\tau$ as:
$$\tau = \min \left\{ T, \;\; \inf_{t \ge 1} \left\{ t : \exists j \in \{1,\dots,m\}, \; \sum_{s=1}^t c_{j, a_s} > B_j \right\} \right\}$$

The objective of an online policy $\pi$ is to maximize total expected reward earned up to $\tau$:
$$\max_{\pi} \;\; \mathbb{E}\left[ \sum_{t=1}^\tau r_t(a_t) \right] \quad \text{subject to} \quad \sum_{t=1}^T c_{j, a_t} \le B_j \quad \forall j \in \{1, \dots, m\}$$

---

### 2.2 The Offline Fluid Linear Program

Because the exact stochastic dynamic programming formulation suffers from the curse of dimensionality, the literature (Badanidiyuru et al. [1], Agrawal & Devanur [2], Balseiro et al. [3]) benchmarks online policies against the **Offline Fluid Linear Program (Fluid LP)**.

The Fluid LP represents the best possible expected performance achievable by a static, randomized policy $\pi(a \mid x) = \mathbb{P}(a_t = a \mid x_t = x)$ if the context distribution $\mathcal{D}_X$ and model parameters $\theta_a^*$ were known in advance.

Define the expected per-decision reward and consumption under policy $\pi$:
$$\bar{r}(\pi) = \mathbb{E}_{x \sim \mathcal{D}_X} \left[ \sum_{a=1}^K \pi(a \mid x) x^\top \theta_a^* \right]$$
$$\bar{c}_j(\pi) = \mathbb{E}_{x \sim \mathcal{D}_X} \left[ \sum_{a=1}^K \pi(a \mid x) c_{j,a} \right]$$

The Fluid Linear Program scales these expectations over the horizon $T$:
$$\begin{aligned}
\text{OPT}_{\text{fluid}} = \max_{\pi} \quad & T \cdot \bar{r}(\pi) \\
\text{s.t.} \quad & T \cdot \bar{c}_j(\pi) \le B_j \quad \forall j \in \{1, \dots, m\} \\
& \sum_{a=1}^K \pi(a \mid x) = 1 \quad \forall x \in \mathcal{X} \\
& \pi(a \mid x) \ge 0 \quad \forall a \in \mathcal{A}, \;\forall x \in \mathcal{X}
\end{aligned}$$

---

### 2.3 The Fluid Benchmark Upper Bound

#### Theorem 1 (Upper Bound Property of the Fluid LP)
*For any non-anticipating online policy $\pi_{\text{online}}$ operating over horizon $T$ with stopping time $\tau$, the expected cumulative reward earned is strictly upper-bounded by the optimal value of the Fluid LP:*
$$\mathbb{E}_{\pi_{\text{online}}}\left[ \sum_{t=1}^\tau r_t(a_t) \right] \le \text{OPT}_{\text{fluid}}$$

#### Proof
Let $\pi_{\text{online}}$ be any causal online policy adapted to the filtration $\mathcal{F}_t = \sigma(x_1, a_1, r_1, \dots, x_t)$. Define the random indicator variable $I_t = \mathbf{1}[t \le \tau]$, indicating that the process has not stopped before step $t$.

The total expected reward is:
$$\mathbb{E}\left[ \sum_{t=1}^\tau r_t(a_t) \right] = \mathbb{E}\left[ \sum_{t=1}^T I_t \, r_t(a_t) \right] = \mathbb{E}\left[ \sum_{t=1}^T I_t \, \mathbb{E}[r_t(a_t) \mid \mathcal{F}_{t-1}, x_t] \right] = \mathbb{E}\left[ \sum_{t=1}^T I_t \sum_{a=1}^K \mathbb{P}(a_t = a \mid \mathcal{F}_{t-1}, x_t) x_t^\top \theta_a^* \right]$$

Now, define the time-averaged marginal policy $\bar{\pi}(a \mid x)$ induced by $\pi_{\text{online}}$ over the entire horizon:
$$\bar{\pi}(a \mid x) = \frac{\sum_{t=1}^T \mathbb{E}\left[ I_t \cdot \mathbf{1}[a_t = a] \mid x_t = x \right]}{\sum_{t=1}^T \mathbb{E}\left[ I_t \mid x_t = x \right]}$$
By construction, $\sum_{a=1}^K \bar{\pi}(a \mid x) = 1$ and $\bar{\pi}(a \mid x) \ge 0$.

Next, examine the resource constraints. By definition of the stopping time $\tau$, cumulative consumption up to $\tau$ cannot exceed the initial budget plus the maximum single-step consumption (or strictly $B_j$ under continuous time limits):
$$\sum_{t=1}^T I_t \, c_{j, a_t} \le B_j \quad \text{almost surely, } \forall j \in \{1, \dots, m\}$$
Taking expectations on both sides:
$$\mathbb{E}\left[ \sum_{t=1}^T I_t \, c_{j, a_t} \right] \le B_j \implies T \cdot \mathbb{E}_{x \sim \mathcal{D}_X} \left[ \sum_{a=1}^K \bar{\pi}(a \mid x) c_{j,a} \right] \le B_j$$
Thus, the induced policy $\bar{\pi}$ satisfies all primal constraints of the Fluid LP. Since $\text{OPT}_{\text{fluid}}$ is the maximum over all feasible policies, the reward of $\bar{\pi}$ cannot exceed $\text{OPT}_{\text{fluid}}$:
$$\mathbb{E}\left[ \sum_{t=1}^\tau r_t(a_t) \right] \le T \cdot \bar{r}(\bar{\pi}) \le \text{OPT}_{\text{fluid}}$$
This establishes that $\text{OPT}_{\text{fluid}}$ is a valid, unassailable theoretical benchmark. $\blacksquare$

---

### 2.4 Lagrangian Relaxation & Duality Theory

The Fluid LP has infinite dimensions if context space $\mathcal{X}$ is continuous. However, notice that the budget constraints are coupled across time exclusively through the $m$ resource limits. 

We form the **Lagrangian Relaxation** by introducing a vector of non-negative Lagrange multipliers (dual variables) $\lambda = (\lambda_1, \dots, \lambda_m)^\top \in \mathbb{R}_+^m$:
$$\mathcal{L}(\pi, \lambda) = T \bar{r}(\pi) - \sum_{j=1}^m \lambda_j \left( T \bar{c}_j(\pi) - B_j \right)$$

Expanding expectations:
$$\mathcal{L}(\pi, \lambda) = \sum_{t=1}^T \mathbb{E}_{x \sim \mathcal{D}_X} \left[ \sum_{a=1}^K \pi(a \mid x) \left( x^\top \theta_a^* - \sum_{j=1}^m \lambda_j c_{j,a} \right) \right] + \sum_{j=1}^m \lambda_j B_j$$

The **Dual Objective Function** $D(\lambda)$ is the unconstrained maximum of $\mathcal{L}(\pi, \lambda)$ over all randomized policies $\pi$:
$$D(\lambda) = \max_{\pi} \mathcal{L}(\pi, \lambda)$$

Because the maximization is unconstrained over policies $\pi$, the optimal policy for a fixed multiplier $\lambda$ places 100% of its probability mass on the arm that maximizes the penalized reward for each context $x$:
$$D(\lambda) = T \cdot \mathbb{E}_{x \sim \mathcal{D}_X} \left[ \max_{a \in \mathcal{A}} \left( x^\top \theta_a^* - \sum_{j=1}^m \lambda_j c_{j,a} \right) \right] + \sum_{j=1}^m \lambda_j B_j$$

The dual optimization problem is:
$$\min_{\lambda \ge 0} D(\lambda)$$

By **Weak Duality**, for any $\lambda \ge 0$:
$$\text{OPT}_{\text{fluid}} \le D(\lambda)$$

Assuming the existence of a null arm $a_0$ with $c_{j, a_0} = 0$ for all $j$, or that the budget is strictly positive ($B_j > 0$), **Slater's Condition** holds, guaranteeing **Strong Duality**:
$$\text{OPT}_{\text{fluid}} = \min_{\lambda \ge 0} D(\lambda) = D(\lambda^*)$$
where $\lambda^*$ is the vector of optimal shadow prices.

---

### 2.5 Complementary Slackness & The Dual Optimum

At the optimal dual vector $\lambda^*$, the **Karush-Kuhn-Tucker (KKT) Complementary Slackness** conditions require:
$$\lambda_j^* \cdot \left( T \bar{c}_j(\pi^*) - B_j \right) = 0 \quad \forall j \in \{1, \dots, m\}$$

This condition yields two distinct operating regimes for each resource $j$:
1. **Non-Binding Resource ($T \bar{c}_j(\pi^*) < B_j$):**  
   If total demand across the horizon does not exceed budget $B_j$, the resource is not scarce. Complementary slackness forces:
   $$\lambda_j^* = 0$$
   The arm incurs **zero penalty** for consuming resource $j$. Decisions proceed unconstrained.
2. **Binding Resource ($T \bar{c}_j(\pi^*) = B_j$):**  
   If the unconstrained policy would consume more than $B_j$, the constraint binds. Complementary slackness allows:
   $$\lambda_j^* > 0$$
   $\lambda_j^*$ acts as a non-zero reservation price, raising the cost of consuming resource $j$ until demand precisely clears available capacity $B_j$.

---

## 3. Revision II: The Per-Step Paced Policy & Economic Interpretation

### 3.1 Temporal Decoupling: From Global LP to Local Argmax

The core computational insight of Lagrangian relaxation is **temporal decoupling**:

$$\max_{\pi} \mathcal{L}(\pi, \lambda) \iff \sum_{t=1}^T \max_{a_t \in \mathcal{A}} \left[ x_t^\top \theta_a^* - \sum_{j=1}^m \lambda_j c_{j,a_t} \right]$$

Rather than solving a global, multi-period linear program with $T \times K$ decision variables, the problem decomposes into **$T$ completely independent, per-decision optimizations**. 

Given the current context $x_t$ and shadow price vector $\lambda_t$, the hot-path decision is a single argmax pass:
$$a_t^* = \arg\max_{a \in \mathcal{A}} \left[ \mu_a(x_t) - \sum_{j=1}^m \lambda_{j,t} c_{j,a} \right]$$

This achieves an evaluation complexity of $O(K \cdot m)$ operations. For $K = 5$ arms and $m = 2$ resources, this evaluates in under **50 nanoseconds** on modern x86/ARM hardware.

---

### 3.2 $\lambda_j$ as an Economic Shadow Price & Opportunity Cost

In economic theory, Lagrange multipliers are **shadow prices** measuring the marginal opportunity cost of scarce capacity.

#### Dimensional Analysis
Let rewards be measured in euros (€), and resource $j$ be measured in minutes of call-center time. Then:
$$[\mu_a(x_t)] = \text{€}, \quad [c_{j,a}] = \text{minutes} \implies [\lambda_j] = \frac{\text{€}}{\text{minute}}$$

Equation (3) evaluates the net economic utility of an arm:
$$\text{Net Utility}(a) = \underbrace{\mu_a(x_t)}_{\text{Expected Direct Return (€)}} - \underbrace{\sum_{j=1}^m \lambda_{j,t} c_{j,a}}_{\text{Shadow Opportunity Cost (€)}}$$

The shadow cost $\lambda_{j,t} c_{j,a}$ represents the **expected reward the system forfeits in the future** by consuming capacity $c_{j,a}$ right now instead of reserving it for a later, higher-uplift request.

---

### 3.3 The Uplift Threshold Rule (A Concrete Worked Example)

To see the economic mechanism in action, consider a concrete churn prevention campaign:
* **Arm 0 ($a_0$):** Standard Email (Cost $c = 0$ call minutes).
* **Arm 1 ($a_1$):** Phone Call (Cost $c = 1$ call minute).
* **Budget:** $B = 200$ call minutes over $T = 10,000$ users.

Arm $a_1$ is selected over $a_0$ if and only if:
$$\mu_{a_1}(x_t) - \lambda_t \cdot 1 > \mu_{a_0}(x_t) - \lambda_t \cdot 0$$
$$\iff \underbrace{\mu_{a_1}(x_t) - \mu_{a_0}(x_t)}_{\Delta(x_t) \text{ (Contextual Uplift)}} > \lambda_t$$

#### Concrete Numerical Example
Suppose the model evaluates three users arriving throughout the day:

| User | Context Description | Email Reward $\mu_{a_0}(x)$ | Call Reward $\mu_{a_1}(x)$ | Uplift $\Delta(x)$ | Decision under Unconstrained FCFS | Decision under Lagrangian Pacing ($\lambda^* = 0.255$) |
|---|---|---|---|---|---|---|
| **User A** | Moderate churn risk; arrives at 08:30 AM | 0.40 | 0.45 | **0.05** | **Call ($a_1$)** *(Burns budget)* | **Email ($a_0$)** *(0.05 < 0.255: Rejected)* |
| **User B** | Low churn risk; arrives at 09:15 AM | 0.80 | 0.82 | **0.02** | **Call ($a_1$)** *(Burns budget)* | **Email ($a_0$)** *(0.02 < 0.255: Rejected)* |
| **User C** | High churn risk; arrives at 16:45 PM | 0.10 | 0.60 | **0.50** | **Email ($a_0$)** *(Call budget dead)* | **Call ($a_1$)** *(0.50 > 0.255: Approved!)* |

* **Outcome under FCFS:** Users A and B consume the limited call slots early in the morning for tiny uplifts (+0.05 and +0.02). When User C arrives in the afternoon with massive uplift (+0.50), the call budget is exhausted. User C is sent an email and churns.
* **Outcome under Pacing:** The shadow price $\lambda_t \approx 0.255$ acts as an admission barrier. Users A and B are filtered out because their uplift does not exceed the opportunity cost of capacity. Capacity is preserved for User C.

---

### 3.4 Paced LinUCB Policy Formulation

In live contextual bandits, parameters $\theta_a^*$ are unknown and estimated via ridge regression.

For each arm $a$, BanditDB maintains:
$$A_{a,t} = I_d + \sum_{s < t : a_s = a} x_s x_s^\top \in \mathbb{R}^{d \times d}$$
$$b_{a,t} = \sum_{s < t : a_s = a} r_s x_s \in \mathbb{R}^d$$
$$\hat{\theta}_{a,t} = A_{a,t}^{-1} b_{a,t}$$

The LinUCB Upper Confidence Bound score with exploration hyperparameter $\beta > 0$ is:
$$\text{UCB}_{a,t}(x_t) = x_t^\top \hat{\theta}_{a,t} + \beta \sqrt{x_t^\top A_{a,t}^{-1} x_t}$$

#### The Paced LinUCB Decision Rule
Incorporate shadow pricing and feasibility masking:
$$a_t = \arg\max_{a \notin M_t} \left[ x_t^\top \hat{\theta}_{a,t} + \beta \|x_t\|_{A_{a,t}^{-1}} - \sum_{j=1}^m \lambda_{j,t} c_{j,a} \right]$$

#### Preservation of Confidence Ellipsoids
Notice that the pacing term $-\sum_j \lambda_{j,t} c_{j,a}$ is **constant with respect to feature uncertainty $\|x_t\|_{A_a^{-1}}$**. The confidence ellipsoid $\mathcal{C}_{a,t} = \{ \theta : \|\theta - \hat{\theta}_{a,t}\|_{A_{a,t}} \le \beta \}$ remains statistically valid. Pacing shifts the center of the selection threshold without distorting the statistical coverage guarantee of the exploration bonus.

---

### 3.5 Paced Thompson Sampling Policy Formulation

Under Thompson Sampling (TS), parameter uncertainty is captured through Bayesian posterior distributions. Assuming Gaussian observation noise with variance $v^2$:
$$\theta_a \mid \mathcal{H}_{t-1} \sim \mathcal{N}\left( \hat{\theta}_{a,t}, \; v^2 A_{a,t}^{-1} \right)$$

At each request:
1. BanditDB factorizes the covariance using a cached Cholesky decomposition:
   $$A_{a,t}^{-1} = L_a L_a^\top$$
2. A posterior sample vector $\tilde{\theta}_{a,t}$ is drawn via standard normal noise $z \sim \mathcal{N}(0, I_d)$:
   $$\tilde{\theta}_{a,t} = \hat{\theta}_{a,t} + v L_a z$$
3. The paced arm selection rule evaluates:
   $$a_t = \arg\max_{a \notin M_t} \left[ x_t^\top \tilde{\theta}_{a,t} - \sum_{j=1}^m \lambda_{j,t} c_{j,a} \right]$$

---

### 3.6 The Safety Net: Hard Feasibility Masking ($M_t$)

Dual descent paces consumption on average across time. However, due to discrete request stochasticity, soft pacing can theoretically overshoot the budget by a small fraction near the very end of the window.

To guarantee zero budget oversell, BanditDB enforces a **Hard Feasibility Mask** $M_t \subset \mathcal{A}$:
$$M_t = \left\{ a \in \mathcal{A} : \exists j \in \{1,\dots,m\} \text{ s.t. } B_{j,t}^{\mathrm{rem}} < c_{j,a} \right\}$$
where $B_{j,t}^{\mathrm{rem}} = B_j - \sum_{s=1}^{t-1} c_{j, a_s}$ is the exact remaining capacity.

In Rust, $M_t$ is implemented as a stack-allocated `u64` bitmask. If bit $a$ is set in $M_t$, arm $a$ is bypassed during the argmax iteration:
```rust
// Sub-microsecond hot-path argmax with Lagrangian pricing and hard mask
let mut best_arm = 0;
let mut best_score = f64::NEG_INFINITY;

for a in 0..num_arms {
    if (mask_bits & (1 << a)) != 0 {
        continue; // Hard mask: insufficient physical budget remaining
    }
    let mut price = 0.0;
    for j in 0..num_resources {
        price += lambda[j].load(Ordering::Relaxed) * cost[a][j];
    }
    let penalized_score = scores[a] - price;
    if penalized_score > best_score {
        best_score = penalized_score;
        best_arm = a;
    }
}
```

---

## 4. Revision III: Online Dual Learning Dynamics & Regret Guarantees

### 4.1 Convexity of the Dual Function & Danskin's Theorem

How does BanditDB find the optimal shadow price $\lambda^*$ without knowing future requests or context distributions? It uses **Online Convex Optimization (OCO)** on the dual objective.

#### Proposition 3 (Convexity of the Dual Function)
*The dual function $D(\lambda)$ is convex on $\mathbb{R}_+^m$.*

#### Proof
Recall the definition:
$$D(\lambda) = T \cdot \mathbb{E}_{x \sim \mathcal{D}_X} \left[ \max_{a \in \mathcal{A}} \left( x^\top \theta_a^* - \sum_{j=1}^m \lambda_j c_{j,a} \right) \right] + \sum_{j=1}^m \lambda_j B_j$$
For every fixed context $x$ and arm $a$, the function:
$$f_{x, a}(\lambda) = x^\top \theta_a^* - \sum_{j=1}^m \lambda_j c_{j,a}$$
is an affine function of $\lambda$, which is both convex and concave.

The function $F(x, \lambda) = \max_{a \in \mathcal{A}} f_{x,a}(\lambda)$ is the pointwise maximum over a finite family of convex functions. By standard convex analysis, the pointwise maximum of convex functions is convex.  
Expectation is a linear operator that preserves convexity: $\mathbb{E}_{x}[F(x, \lambda)]$ is convex.  
The linear term $\sum_j \lambda_j B_j$ is convex.  
Since the sum of convex functions is convex, $D(\lambda)$ is convex on $\mathbb{R}_+^m$. $\blacksquare$

#### Subgradient Derivation via Danskin's Theorem
Let the per-step dual loss at step $t$ be:
$$\ell_t(\lambda) = \max_{a \in \mathcal{A}} \left( r_t(a) - \sum_{j=1}^m \lambda_j c_{j,a} \right) + \sum_{j=1}^m \lambda_j \rho_{j,t}$$
where $\rho_{j,t}$ is the target consumption rate per decision.

By **Danskin's Theorem**, the subgradient of the maximum of affine functions is given by the gradient of the function that achieves the maximum. If $a_t = \arg\max_{a} [r_t(a) - \lambda_t^\top c_a]$, a stochastic subgradient $g_t \in \partial \ell_t(\lambda_t)$ with respect to $\lambda_j$ is:
$$g_{j,t} = \frac{\partial}{\partial \lambda_j} \left( r_t(a_t) - \sum_{k=1}^m \lambda_k c_{k, a_t} + \sum_{k=1}^m \lambda_k \rho_{k,t} \right) = \rho_{j,t} - c_{j, a_t}$$

---

### 4.2 Projected Online Gradient Descent (OGD)

Because we are **minimizing** the convex dual function $D(\lambda)$, we move in the negative subgradient direction:
$$-g_{j,t} = c_{j, a_t} - \rho_{j,t}$$

This produces the **Projected Online Gradient Descent (OGD)** update rule:
$$\lambda_{j, t+1} = \Pi_{[0, \lambda_j^{\max}]} \left[ \lambda_{j,t} + \eta \left( c_{j, a_t} - \rho_{j,t} \right) \right]$$
where:
* $\eta > 0$ is the learning rate (step size).
* $\Pi_{[0, \lambda_j^{\max}]}(z) = \max(0, \min(z, \lambda_j^{\max}))$ is the orthogonal Euclidean projection onto the feasible box.

```
                  Realized Consumption vs Target Rate
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
   cj,at > ρj,t                                    cj,at < ρj,t
(Consuming too fast!)                           (Consuming too slow!)
         │                                               │
         ▼                                               ▼
 Δλ = +η(c - ρ) > 0                              Δλ = -η(ρ - c) < 0
         │                                               │
         ▼                                               ▼
   λj increases                                    λj decreases
         │                                               │
         ▼                                               ▼
Arm becomes more expensive                      Arm becomes cheaper
(Uplift bar rises: demand drops)                (Uplift bar lowers: demand rises)
```

#### Step-by-Step Numeric Trace of Dual Descent
Suppose $B = 200, T = 10,000 \implies \rho = 0.02$. Set $\eta = 0.05$. Initialize $\lambda_0 = 0.20$.

* **Step 1:** Call arm selected ($c = 1$).
  $$\Delta \lambda = 0.05 \times (1 - 0.02) = +0.049 \implies \lambda_1 = 0.20 + 0.049 = \mathbf{0.249}$$
* **Step 2:** Call arm selected ($c = 1$).
  $$\Delta \lambda = +0.049 \implies \lambda_2 = 0.249 + 0.049 = \mathbf{0.298}$$
* **Step 3:** Email arm selected ($c = 0$).
  $$\Delta \lambda = 0.05 \times (0 - 0.02) = -0.001 \implies \lambda_3 = 0.298 - 0.001 = \mathbf{0.297}$$
* **Step 4:** Email arm selected ($c = 0$).
  $$\Delta \lambda = -0.001 \implies \lambda_4 = 0.297 - 0.001 = \mathbf{0.296}$$

The multiplier increases rapidly on consumption (+0.049), and relaxes slowly downward during non-consumption (-0.001), stabilizing precisely when the empirical selection rate matches the target rate $\rho = 0.02$.

---

### 4.3 Integrator Windup & The Theoretical Maximum Multiplier ($\lambda^{\max}$)

A critical failure in online control systems is **integrator windup**: if consumption persistently exceeds target rate (e.g., during a huge morning traffic surge), an unconstrained multiplier $\lambda$ could grow to $10.0$ or $100.0$. When traffic subsequently drops, $\lambda$ takes thousands of decisions to decay back down to the active region, unnecessarily locking out arms for hours.

To prevent windup, we establish the theoretical upper bound $\lambda_j^{\max}$.

#### Proposition 4 (Maximum Meaningful Multiplier)
*Let rewards be bounded by $r_{\max} \in [0, 1]$, and let the minimum positive cost of arm consumption be $c_{\min, j} = \min_{a : c_{j,a} > 0} c_{j,a}$. Then for any:*
$$\lambda_j \ge \lambda_j^{\max} = \frac{r_{\max}}{c_{\min, j}}$$
*no arm consuming resource $j$ can ever win an argmax against a zero-cost arm. Any value of $\lambda_j > \lambda_j^{\max}$ provides zero additional control authority and strictly degrades recovery responsiveness.*

#### Proof
Consider any arm $a$ that consumes resource $j$ with cost $c_{j,a} \ge c_{\min, j}$. Its penalized utility is:
$$\text{Utility}(a) = \mu_a(x) - \sum_{k=1}^m \lambda_k c_{k,a} \le r_{\max} - \lambda_j c_{\min, j}$$
Now consider a zero-cost baseline arm $a_0$ with $c_{k, a_0} = 0$ for all $k$. Its utility is:
$$\text{Utility}(a_0) = \mu_{a_0}(x) \ge 0$$
For arm $a$ to win, we must have:
$$\text{Utility}(a) > \text{Utility}(a_0) \implies r_{\max} - \lambda_j c_{\min, j} > 0 \implies \lambda_j < \frac{r_{\max}}{c_{\min, j}}$$
If $\lambda_j \ge \frac{r_{\max}}{c_{\min, j}}$, arm $a$ has net utility $\le 0 \le \mu_{a_0}(x)$, meaning it cannot beat arm $a_0$ for any context $x$. Therefore, capping projection at $\lambda_j^{\max}$ prevents integrator windup without altering the optimal decision policy. $\blacksquare$

---

### 4.4 Entropic Dual Mirror Descent (OMD) & The Zero-Trap Vulnerability

Rather than Euclidean projection, one can use **Online Mirror Descent (OMD)** with negative entropy regularizer $\psi(\lambda) = \sum_j \lambda_j \ln \lambda_j$:
$$\lambda_{j, t+1} = \lambda_{j,t} \exp\left( \eta (c_{j, a_t} - \rho_{j,t}) \right)$$

#### The Zero-Trap Vulnerability
While multiplicative weights offer theoretical scale invariance, they suffer from a severe operational flaw: **the zero-trap**.
* If a resource constraint is non-binding for several hours (e.g., quiet overnight traffic), exponential updates drive $\lambda_j$ toward 0 (e.g., $10^{-12}$).
* If sudden heavy traffic arrives in the morning, an update of $\lambda \leftarrow \lambda \exp(\eta(1 - \rho))$ requires hundreds of steps just to climb from $10^{-12}$ back to $0.1$.
* In contrast, **Projected Additive OGD** hits exactly 0.0, and on the very first consuming step jumps immediately by $+\eta(1 - \rho)$, providing immediate control response. 

*BanditDB standardizes on Projected Additive OGD for this operational reason.*

---

### 4.5 Target Consumption Rates: Uniform, Adaptive, and Periodic Profiles

#### 1. Uniform Target Rate
$$\rho_{j,t} = \frac{B_j}{T}$$
Assumes static, flat traffic. Breaks down under diurnal curves: burns budget during morning peaks, starving evening traffic.

#### 2. Adaptive Target Rate (Recommended Baseline)
$$\rho_{j,t} = \frac{B_{j,t}^{\mathrm{rem}}}{T - t + 1}$$
Where $B_{j,t}^{\mathrm{rem}} = B_j - \sum_{s=1}^{t-1} c_{j, a_s}$ is remaining budget.  
*Feedback Mechanism:* If early decisions overspent budget, $B^{\mathrm{rem}}$ drops faster than $T - t$, reducing $\rho_{j,t}$. This automatically forces $\Delta \lambda = \eta(c - \rho)$ higher, dynamically throttling future consumption.

#### 3. Periodic Traffic-Weighted Target Rate
When traffic has strong diurnal patterns (e.g., 24-hour cycles), historical profiles provide hourly density weights $w(t)$ normalized such that $\sum_{t=1}^T w(t) = 1$:
$$\rho_{j,t} = B_{j,t}^{\mathrm{rem}} \cdot \frac{w(t)}{\sum_{s=t}^T w(s)}$$
Ensures target rate matches expected incoming traffic volume at each hour of the day.

---

### 4.6 Theoretical Regret Bound ($O(\sqrt{T})$ Regret Proof Breakdown)

#### Theorem 2 (Regret Bound for Dual Descent in BwK)
*(Adapted from Balseiro, Lu & Mirrokni [3], Theorem 1; Agrawal & Devanur [2])*  
*Let contexts $x_t \sim \mathcal{D}_X$ be i.i.d., costs $c_{j,a} \in [0, 1]$, rewards $r_t \in [0, 1]$, and budgets scale proportionally with horizon: $B_j = \alpha_j T$ for $\alpha_j \in (0, 1)$. Using Projected OGD with step size:*
$$\eta = \frac{\lambda_{\max}}{G \sqrt{T}}$$
*where $G = \max_{j,a} |c_{j,a} - \rho_j| \le 1$, the cumulative regret against the Fluid LP benchmark satisfies:*
$$R(T) = \text{OPT}_{\text{fluid}} - \mathbb{E}\left[ \sum_{t=1}^\tau r_t(a_t) \right] \le O\left( \sqrt{T} \left( \sqrt{d \ln T} + \sum_{j=1}^m \frac{1}{\alpha_j} \right) \right) = O(\sqrt{T})$$

#### Proof Breakdown
The proof decomposes cumulative regret into three components:

$$\text{Regret}(T) = \underbrace{\text{OPT}_{\text{fluid}} - \sum_{t=1}^T \ell_t(\lambda^*)}_{\text{Term 1: Duality Gap}} + \underbrace{\sum_{t=1}^T \ell_t(\lambda^*) - \sum_{t=1}^T \ell_t(\lambda_t)}_{\text{Term 2: Dual OCO Regret}} + \underbrace{\sum_{t=1}^T \ell_t(\lambda_t) - \sum_{t=1}^T r_t(a_t)}_{\text{Term 3: Bandit Estimation & Slack}}$$

1. **Term 1 (Duality Gap):** By strong duality of the Fluid LP, $\text{OPT}_{\text{fluid}} = \mathbb{E}[\sum_{t=1}^T \ell_t(\lambda^*)]$. In expectation, this term is 0.
2. **Term 2 (Dual Online Convex Optimization Regret):** Standard OCO analysis for projected gradient descent with convex losses $\ell_t$ and bounded subgradients $\|g_t\|_2 \le G$:
   $$\sum_{t=1}^T \ell_t(\lambda_t) - \min_{\lambda \in [0, \lambda_{\max}]} \sum_{t=1}^T \ell_t(\lambda) \le \frac{\|\lambda_0 - \lambda^*\|_2^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \|g_t\|_2^2 \le \frac{m \lambda_{\max}^2}{2\eta} + \frac{\eta T m G^2}{2}$$
   Setting $\eta = \Theta(1/\sqrt{T})$ yields an upper bound of $O(m \sqrt{T})$.
3. **Term 3 (Bandit Learning & Primal Recovery):** Recall that $\ell_t(\lambda_t) = r_t(a_t) - \lambda_t^\top c_{a_t} + \lambda_t^\top \rho_t + \text{EstRegret}_t$.  
   Summing across $t=1$ to $T$:
   $$\sum_{t=1}^T \ell_t(\lambda_t) - \sum_{t=1}^T r_t(a_t) = \sum_{j=1}^m \sum_{t=1}^T \lambda_{j,t} (\rho_{j,t} - c_{j, a_t}) + \sum_{t=1}^T \text{EstRegret}_t$$
   The first summation is the cumulative dual subgradient drift, bounded by $O(\sqrt{T})$ via telescoping dual steps. The second term is the regret of contextual parameter estimation (LinUCB / Thompson Sampling), bounded by standard concentration results as $O(d \sqrt{T \ln T})$.

Combining all three bounds proves that cumulative regret scales strictly as $O(\sqrt{T})$. Dividing by $T$, average regret per decision approaches zero at rate $O(1/\sqrt{T})$. $\blacksquare$

---

## 5. Revision IV: Counterfactual Estimation & Propensity Corrections

### 5.1 The Constrained Assignment Probability $\pi(a \mid x, \lambda, M)$

In modern data architectures, decisions served by BanditDB are logged to an append-only WAL and exported to Parquet for offline counterfactual analysis:
* **Off-Policy Evaluation (OPE):** Evaluating a new candidate algorithm before deployment.
* **Progressive Tournaments:** Measuring whether a challenger model outperforms a base model via SNIPS.
* **Causal Uplift Modeling:** Training downstream CATE models (e.g., Causal Forests, X-Learners).

Under pacing, the arm assignment probability is no longer a static function of context $x_t$. It depends explicitly on internal engine state:
$$\pi_t(a) = \mathbb{P}\left( a_t = a \mid x_t, \lambda_t, M_t \right) \ne \pi(a \mid x_t)$$

---

### 5.2 The "Price Before Propensity" Theorem

Let $\pi_{\text{actual}}(a \mid x, \lambda, M)$ be the true assignment probability under Lagrangian pacing.  
Let $\pi_{\text{unpriced}}(a \mid x)$ be the propensity computed from raw, unpriced scores (ignoring $\lambda$ and $M$).

#### Theorem 3 (Asymptotic Bias of Unpriced Propensity Logging)
*Let $V(\pi') = \mathbb{E}_{x}[r(x, \pi'(x))]$ be the ground-truth expected value of an evaluation policy $\pi'$. If the logging system records unpriced propensities $\pi_{\text{unpriced}}$, the standard Inverse Propensity Scoring (IPS) estimator:*
$$\hat{V}_{\text{IPS}}(\pi') = \frac{1}{n} \sum_{t=1}^n \frac{\pi'(a_t \mid x_t)}{\pi_{\text{logged}}(a_t \mid x_t)} r_t$$
*is asymptotically biased, with expectation:*
$$\mathbb{E}\left[ \hat{V}_{\text{IPS}}^{\text{unpriced}}(\pi') \right] = V(\pi') + \mathbb{E}_{x, \lambda, M} \left[ \sum_{a=1}^K \pi'(a \mid x) \mu_a(x) \left( \frac{\pi_{\text{actual}}(a \mid x, \lambda, M)}{\pi_{\text{unpriced}}(a \mid x)} - 1 \right) \right]$$

#### Proof
Condition on the full state $(x_t, \lambda_t, M_t)$:
$$\mathbb{E}\left[ \frac{\pi'(a_t \mid x_t)}{\pi_{\text{unpriced}}(a_t \mid x_t)} r_t \;\middle|\; x_t, \lambda_t, M_t \right] = \sum_{a=1}^K \mathbb{P}(a_t = a \mid x_t, \lambda_t, M_t) \left( \frac{\pi'(a \mid x_t)}{\pi_{\text{unpriced}}(a \mid x_t)} \right) \mathbb{E}[r_t(a) \mid x_t, a_t = a]$$
$$= \sum_{a=1}^K \pi_{\text{actual}}(a \mid x_t, \lambda_t, M_t) \frac{\pi'(a \mid x_t)}{\pi_{\text{unpriced}}(a \mid x_t)} \mu_a(x_t)$$
$$= \sum_{a=1}^K \pi'(a \mid x_t) \mu_a(x_t) \left( \frac{\pi_{\text{actual}}(a \mid x_t, \lambda_t, M_t)}{\pi_{\text{unpriced}}(a \mid x_t)} \right)$$
Subtracting the target $V(\pi') = \mathbb{E}[\sum_a \pi'(a \mid x_t) \mu_a(x_t)]$ yields the exact bias formula. The bias vanishes if and only if $\pi_{\text{unpriced}}(a \mid x) = \pi_{\text{actual}}(a \mid x, \lambda, M)$ almost everywhere, which is strictly false whenever $\lambda_t > 0$ or $M_t \ne \emptyset$. $\blacksquare$

---

### 5.3 Step-by-Step Numerical Breakdown: How IPS Incurs +106% Bias

To see why the error reached **+106%** in BanditDB benchmarks, trace a concrete numerical example:
* Two arms: Arm 0 (Email, cost 0), Arm 1 (Phone Call, cost 1).
* Evaluation policy $\pi'$ attempts to evaluate an aggressive policy choosing Arm 1 with probability $\pi'(a_1) = 0.80$.
* Under unconstrained scoring, Arm 1 wins 50% of the time: $\pi_{\text{unpriced}}(a_1) = 0.50$.
* Under active pacing, shadow price $\lambda_t = 0.25$ restricts Arm 1 to only the top 5% of contexts: $\pi_{\text{actual}}(a_1) = 0.05$.
* True reward of Arm 1 in these elite contexts: $r = 0.90$.

Now compare the IPS calculation:
1. **Under True Paced Propensity Logging:**
   $$w_t^{\text{true}} = \frac{\pi'(a_1)}{\pi_{\text{actual}}(a_1)} = \frac{0.80}{0.05} = 16.0$$
   Contribution to value estimate: $16.0 \times 0.90 = \mathbf{14.4}$.  
   Since Arm 1 only appears in 5% of decisions, its expected contribution to the sum is $0.05 \times 14.4 = \mathbf{0.72}$ (exact ground truth).
2. **Under Faulty Unpriced Propensity Logging:**
   $$w_t^{\text{faulty}} = \frac{\pi'(a_1)}{\pi_{\text{unpriced}}(a_1)} = \frac{0.80}{0.50} = 1.6$$
   Now examine Arm 0 (Email). Arm 0 was selected in 95% of contexts ($\pi_{\text{actual}} = 0.95$), but unpriced logging reported $\pi_{\text{unpriced}}(a_0) = 0.50$.
   $$w_{t, a_0}^{\text{faulty}} = \frac{\pi'(a_0)}{\pi_{\text{unpriced}}(a_0)} = \frac{0.20}{0.50} = 0.40$$
   The expected value calculation sums over the distorted weights:
   $$\mathbb{E}[\hat{V}_{\text{IPS}}^{\text{unpriced}}] = 0.95 \times [0.40 \times 0.40] + 0.05 \times [1.6 \times 0.90] = 0.152 + 0.072 = 0.224$$
   Relative to the true value $V = 0.108$, the estimate is distorted by **+106.3%**.

---

### 5.4 Why SNIPS Masks Propensity Distortion (−0.93% False Sense of Security)

Self-Normalized Inverse Propensity Scoring (SNIPS) is defined as:
$$\hat{V}_{\text{SNIPS}}(\pi') = \frac{\sum_{t=1}^n \frac{\pi'(a_t \mid x_t)}{\pi_{\text{logged}}(a_t \mid x_t)} r_t}{\sum_{t=1}^n \frac{\pi'(a_t \mid x_t)}{\pi_{\text{logged}}(a_t \mid x_t)}} = \frac{\sum_{t=1}^n w_t r_t}{\sum_{t=1}^n w_t}$$

#### The Self-Normalization Masking Proof
Suppose logged propensities have an average scalar distortion $\gamma > 0$ relative to true propensities across the active arms:
$$\pi_{\text{logged}}(a_t \mid x_t) \approx \gamma \cdot \pi_{\text{actual}}(a_t \mid x_t, \lambda_t)$$
Then every observed weight $w_t^{\text{distorted}}$ satisfies:
$$w_t^{\text{distorted}} = \frac{\pi'(a_t)}{\gamma \cdot \pi_{\text{actual}}(a_t)} = \frac{1}{\gamma} w_t^{\text{true}}$$

Substitute this into the SNIPS formula:
$$\hat{V}_{\text{SNIPS}}^{\text{distorted}} = \frac{\sum_{t=1}^n \left( \frac{1}{\gamma} w_t^{\text{true}} \right) r_t}{\sum_{t=1}^n \left( \frac{1}{\gamma} w_t^{\text{true}} \right)} = \frac{\frac{1}{\gamma} \sum_{t=1}^n w_t^{\text{true}} r_t}{\frac{1}{\gamma} \sum_{t=1}^n w_t^{\text{true}}} = \hat{V}_{\text{SNIPS}}^{\text{true}}$$

The distortion scalar $\gamma$ factors out completely and **cancels between numerator and denominator**.

#### The Operational Danger
In BanditDB CI tests, SNIPS reported an error of **−0.93%**, leading engineers to believe the logging was correct. However, unnormalized estimators (Doubly Robust, total expected revenue, CATE models) do not normalize by $\sum w_t$ and experienced catastrophic +106% errors. 

**Enforcement:** Propensities must always be computed from the final, priced scores including $\lambda$ and $M$.

---

### 5.5 Positivity Violations & Support Truncation Under $M_t$

The foundational assumption of counterfactual causal inference is **Positivity (Common Support)**:
$$\mathbb{P}(a_t = a \mid x_t) > 0 \quad \forall a \in \mathcal{A}, \;\forall x \in \mathcal{X}$$

When the hard mask fires ($a \in M_t$):
$$\pi_{\text{actual}}(a \mid x_t, \lambda_t, M_t) = 0$$

Positivity is strictly violated. Any counterfactual estimator attempting to evaluate an arm when it is masked would divide by zero ($1 / 0$). 

#### The BanditDB Logging Contract
To preserve statistical validity, BanditDB logs a `mask_bits` field in every interaction record. Offline evaluation engines filter out any intervals where candidate policies choose an arm that was physically masked, reporting those intervals as **unidentified by data** rather than extrapolating.

---

### 5.6 Time-Varying Confounding in CATE & Uplift Modeling (DAG Analysis)

Downstream data science teams use logged bandit data to fit Conditional Average Treatment Effect (CATE) models:
$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$
where $Y(a)$ represents the potential outcome under arm $a$.

Causal identifiability requires the **Unconfoundedness Assumption**:
$$(Y(1), Y(0)) \perp A \mid X$$

#### Directed Acyclic Graph (DAG) Analysis
Under Lagrangian pacing, $\lambda_t$ changes over time. Time of day $t$ drives incoming traffic volume and user behavior.

```
                    Time of Day / Traffic Seasonality (t)
                                 │          │
                 ┌───────────────┘          └───────────────┐
                 ▼                                          ▼
     Shadow Price Multiplier (λt)                  User Conversion Rate (Y)
                 │                                          ▲
                 ▼                                          │
        Arm Assignment (At) ────────────────────────────────┘
                                   True Uplift
```

1. Time of day $t$ influences shadow price $\lambda_t$ (as budget depletes).
2. Shadow price $\lambda_t$ dictates arm selection $A_t$.
3. Time of day $t$ directly influences user conversion $Y$ (e.g., evening users have higher natural baseline conversion).

Because $\lambda_t$ is causally upstream of $A_t$ and shares a common cause ($t$) with outcome $Y$, **omitting $\lambda_t$ creates a backdoor confounding path**:
$$A_t \leftarrow \lambda_t \leftarrow t \rightarrow Y$$

If a causal forest is trained on features $(X, A)$ to predict $Y$, the tree splits on features that correlate with time of day and **attributes natural diurnal conversion spikes to the treatment arm**, generating false positive treatment effects.

#### The Mandatory Logging Contract
To block this backdoor path, BanditDB exports the full state vector on every decision:
$$\langle \text{decision\_id}, \; x_t, \; a_t, \; r_t, \; \pi_{\text{actual}}(a_t), \; M_t, \; \lambda_t, \; B_t^{\mathrm{rem}}, \; \text{version} \rangle$$
Conditioning on $(X, \lambda_t)$ successfully blocks the backdoor path and restores unconfoundedness.

---

## 6. Revision V: Boundary Invariants, Edge Cases & Operational Architecture

### 6.1 Horizon Singularity as $T^{\mathrm{rem}} \to 0$ (The Endgame Freeze Invariant)

Under adaptive target pacing:
$$\rho_{j,t} = \frac{B_{j,t}^{\mathrm{rem}}}{T - t + 1}$$

As $t \to T$, the remaining horizon $(T - t + 1) \to 1$.

#### The Singularity Hazard
Suppose at $t = 9,998$ of $10,000$, remaining budget is $B^{\mathrm{rem}} = 4.0$ units due to stochastic arrivals.
$$\rho_{j, 9998} = \frac{4.0}{10,000 - 9,998 + 1} = \frac{4.0}{3} \approx 1.33$$
If $t = 10,000$ and $B^{\mathrm{rem}} = 2.0$, $\rho = 2.0$. If decisions arrive after $T$, division by zero occurs.  
Even before $t = T$, when $T^{\mathrm{rem}} < 50$, minor integer variations in consumption cause $\rho$ to swing wildly, driving $\lambda$ into high-frequency oscillations.

#### Invariant 1 (Endgame Freeze Boundary)
```
IF (T - t) < max(50, 0.05 * T):
    λ_t = λ_{t-1}               // Freeze dual multiplier at current value
    Rely on Hard Mask Mt        // Let physical feasibility mask govern endgame
ELSE:
    ρ_t = B_rem / (T - t + 1)   // Continue adaptive dual descent
```
Freezing $\lambda$ during the final 5% of the window eliminates numerical singularities, allowing the hard mask $M_t$ to handle the final units of capacity cleanly.

---

### 6.2 Consumption-Nonconsumption Update Asymmetry (Variance Scaling)

Consider a campaign where $B = 200$ and $T = 10,000 \implies \rho = 0.02$.

At each decision:
* If arm consumes ($c = 1$): $\Delta \lambda = +\eta(1 - 0.02) = +\mathbf{0.98\eta}$.
* If arm does not consume ($c = 0$): $\Delta \lambda = +\eta(0 - 0.02) = -\mathbf{0.02\eta}$.

#### The Update Imbalance
A single call consumption increases $\lambda$ by $0.98\eta$. It requires **49 consecutive non-consuming decisions** to undo that single increase!

If step size $\eta = 0.02$, a random cluster of 10 calls in the morning increases $\lambda$ by $+0.20$, instantly freezing the arm for the next 500 decisions.

#### Invariant 2 (Binomial Variance Normalization)
Scale learning rate $\eta$ inversely by the standard deviation of consumption:
$$\sigma_\rho = \sqrt{\rho (1 - \rho)}$$
$$\eta_{\text{scaled}} = \eta_0 \cdot \frac{\sigma_\rho}{\sqrt{T}} = \eta_0 \sqrt{\frac{\rho_j (1 - \rho_j)}{T}}$$
This bounds the maximum single-step shock of rare-event consumption, preventing premature multiplier spikes.

---

### 6.3 Dimensional Consistency: Raw vs. Normalized Costs

A frequent implementation error is normalizing costs during dual descent but using raw costs in the hot-path argmax.

Let normalized cost be $\tilde{c}_{j,a} = c_{j,a} / B_j$, such that total normalized budget is $1.0$. The normalized constraint is:
$$\sum_{t=1}^T \tilde{c}_{j, a_t} \le 1$$

The normalized Lagrangian is:
$$\tilde{\mathcal{L}}(\pi, \tilde{\lambda}) = \sum_t r_t(a_t) - \sum_{j=1}^m \tilde{\lambda}_j \left( \sum_t \frac{c_{j, a_t}}{B_j} - 1 \right)$$

Comparing this to the unnormalized Lagrangian $\mathcal{L}(\pi, \lambda) = \sum r - \sum \lambda (\sum c - B)$ proves the mathematical relationship:
$$\tilde{\lambda}_j = B_j \cdot \lambda_j \iff \lambda_j = \frac{\tilde{\lambda}_j}{B_j}$$

#### Invariant 3 (Mathematical Consistency of Scoring)
If dual descent updates normalized $\tilde{\lambda}_j \in [0, \tilde{\lambda}_{\max}]$, the hot-path argmax **must divide cost by $B_j$**:
$$a_t = \arg\max_{a \notin M_t} \left[ \text{score}(a) - \sum_{j=1}^m \tilde{\lambda}_{j,t} \left( \frac{c_{j,a}}{B_j} \right) \right]$$
*Using raw $c_{j,a}$ against normalized $\tilde{\lambda}_j$ over-penalizes the arm by a factor of $B_j$ (e.g., 200×), permanently suppressing the arm.*

---

### 6.4 Progressive Tournament Confounding (SUTVA Violations Under Shared Multipliers)

BanditDB runs an online tournament where a **Challenger algorithm** competes against a **Base algorithm** via shadow learning.

The **Stable Unit Treatment Value Assumption (SUTVA)** requires that the treatment assigned to one unit does not affect the potential outcomes of other units.

When Base and Challenger share a physical budget $B$:
1. In the morning, Base routes traffic and consumes 180 call slots, driving shadow price $\lambda_t \to 0.28$.
2. In the afternoon, Challenger receives 20% evaluation traffic.
3. Challenger faces an inflated shadow price $\lambda = 0.28$ and an almost depleted budget. Challenger is blocked from selecting high-uplift expensive arms that it could have routed more intelligently than Base.

#### Invariant 4 (Coupled Tournament Reporting)
* **Operational Rule:** Base and Challenger **must share $\lambda$** (physical capacity cannot be duplicated).
* **Statistical Rule:** Progressive tournament evaluation records must emit a telemetry tag:
  $$\text{pacing\_active: true, } \quad \text{shared\_trajectory: true}$$
  The offline SNIPS evaluation must stratify comparisons into unconstrained periods ($\lambda = 0$) and constrained periods ($\lambda > 0$).

---

### 6.5 Inter-Window Warm-Starting (Exponential Multiplier Smoothing)

For recurring campaigns (e.g., daily windows resetting at 00:00 UTC):
* A naive system resets $\lambda_0 = 0$ every midnight.
* Between 00:00 and 02:00, $\lambda$ must re-climb from $0$ to $\lambda^*$, causing an artificial **midnight consumption spike**.

#### Invariant 5 (Inter-Window Multiplier Smoothing)
Let $\bar{\lambda}_w^*$ be the time-averaged shadow price over window $w$. Initialize window $w+1$ via exponential moving average:
$$\lambda_{0}^{(w+1)} = \alpha \bar{\lambda}_w^* + (1 - \alpha) \lambda_0^{(w)}$$
with smoothing parameter $\alpha \in [0.70, 0.85]$. Pacing begins the morning near market equilibrium, eliminating midnight over-allocation.

---

### 6.6 Zero-Overhead Integration & Rollback-Safe Architecture

To preserve BanditDB's core engine stability, pacing is engineered as a **structurally inert feature**:

#### 1. Inactive Zero-Cost Property
```rust
pub struct Campaign {
    pub arms: RwLock<HashMap<String, ArmState>>,
    pub algorithm: Algorithm,
    pub pacing: Option<PacingController>, // None for unconstrained campaigns
}
```
If `pacing.is_none()`, the decision loop compiles down to a single pointer check branch. Hot-path latency penalty for unconstrained campaigns is **0.0 nanoseconds**.

#### 2. Rollback-Safe Forward-Compatible WAL Replay
When rolling back a new binary release, older binaries must not crash when encountering new WAL event variants.
1. Annotate all event enums with `#[serde(other)] Unknown` catch-all variants.
2. In [`read_wal_slice()`](file:///Users/gsang/dev/banditdb/src/engine.rs#L100), log a `tracing::warn!` on unknown variants and continue processing rather than aborting recovery.
3. Ship the forward-compatible reader **first**, verify in production, and only then introduce new capacity WAL records.

---

## 7. System Architecture Pipeline

The following flowchart details the end-to-end execution path, separating the sub-microsecond synchronous hot path from asynchronous dual updates:

```
================================================================================
                    SYNCHRONOUS REQUEST HOT PATH (≤ 100 µs)
================================================================================
                                    │
                                    ▼
                     [ 1. Ingest Context xt & Filter ]
                                    │
                                    ▼
                     [ 2. Check Hard Feasibility Mask ]
                     Mt = { a : B_rem < c_a }
                     (Bitmask bypass: drops invalid arms)
                                    │
                                    ▼
                     [ 3. Compute Lagrangian Penalties ]
                     score(a) - Σ λ_j * (c_a / B_j)
                     (Atomic loads of λ; vector dot product)
                                    │
                                    ▼
                     [ 4. Compute Paced Propensities ]
                     π(a | xt, λt, Mt)
                     (Softmax for LinUCB / MC for Thompson)
                     *ENFORCES PRICE BEFORE PROPENSITY*
                                    │
                                    ▼
                     [ 5. Select Best Arm (Argmax / TS) ]
                                    │
                                    ▼
                     [ 6. Emit WAL Record & Return ]
                     Write to channel: ⟨id, xt, at, π, λt, Mt, B_rem⟩
                                    │
====================================╪===========================================
               ASYNCHRONOUS BACKGROUND THREAD (Every 10ms - 1s)
====================================╪===========================================
                                    │
                                    ▼
                     [ 7. Consume Batch Consumption Δc ]
                                    │
                                    ▼
                     [ 8. Evaluate Target Rate ρ_t ]
                     IF T_rem < T_min:
                         Freeze λ (Endgame Invariant)
                     ELSE:
                         ρ_t = B_rem / T_rem
                                    │
                                    ▼
                     [ 9. Projected Dual Gradient Descent ]
                     λ_{t+1} = Π[0, λ_max](λ_t + η * (c_t - ρ_t))
                                    │
                                    ▼
                     [ 10. Atomic Store of Updated λ ]
                     lambda[j].store(λ_{t+1}, Ordering::Relaxed)
```

---

## 8. References

1. **A. Badanidiyuru, R. Kleinberg, A. Slivkins.** *Bandits with Knapsacks.* IEEE 54th Annual Symposium on Foundations of Computer Science (FOCS), 2013; Journal of the ACM (JACM), 65(3):1–55, 2018.
2. **S. Agrawal, N. R. Devanur.** *Linear Contextual Bandits with Knapsacks.* Advances in Neural Information Processing Systems (NeurIPS), 29:3450–3458, 2016.
3. **S. Balseiro, H. Lu, V. Mirrokni.** *The Best of Many Worlds: Dual Mirror Descent for Online Allocation Problems.* Operations Research, 71(3):1000–1018, 2023.
4. **M. Dudík, D. Erhan, J. Langford, L. Li.** *Doubly Robust Policy Evaluation and Optimization.* Statistical Science, 29(4):485–511, 2014.
5. **A. Swaminathan, T. Joachims.** *The Self-Normalized Estimator for Counterfactual Learning.* Advances in Neural Information Processing Systems (NeurIPS), 28:3231–3239, 2015.
6. **S. Athey, S. Wager.** *Estimating Treatment Effects with Causal Forests: An Application.* Observational Studies, 5(2):37–51, 2019.
7. **E. Hazan.** *Introduction to Online Convex Optimization.* Foundations and Trends in Optimization, 2(3-4):157–325, 2016.
8. **J. Danskin.** *The Theory of Max-Min, with Applications.* SIAM Journal on Applied Mathematics, 14(4):641–664, 1966.
