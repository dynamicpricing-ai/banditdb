# BanditDB: GCP Managed Service Launch Strategy

This document outlines the minimum viable strategy for productizing and launching BanditDB as a public managed service on Google Cloud Platform (GCP).

## 1. The Core Architecture: "Dedicated by Default"

Given BanditDB's current single-writer architecture and the complexities of multi-tenant noisy-neighbor isolation, the most robust and fastest path to market is a **Dedicated Instance Model** managed via Kubernetes (GKE).

*   **The Data Plane:** Each paying customer gets a dedicated Kubernetes namespace on a shared GKE cluster. Inside that namespace is a dedicated BanditDB Helm release: a `StatefulSet` (1 replica), a dedicated PersistentVolumeClaim (PVC), and a dedicated `Service`.
*   **The Benefit:** This provides hard data isolation by default, eliminates noisy-neighbor performance degradation (crucial for latency-sensitive RL), and makes cost attribution and incident response per customer trivial.

## 2. Minimum Viable Product (MVP) Steps for Launch

To launch the service and start accepting revenue, you need the following components:

### A. The Control Plane (Provisioning & Management)
You need a lightweight control plane to automate onboarding.
1.  **Customer Portal/API:** A simple web dashboard (or API) where a user can sign up, enter a credit card (via Stripe), and click "Create Database".
2.  **The Provisioner:** A backend service that intercepts the "Create" request and executes a Helm install against the GKE cluster:
    ```bash
    helm install banditdb-cust123 ./helm/banditdb --namespace cust123 \
      --set auth.apiKey="<generated-key>" \
      --set persistence.size=10Gi
    ```
3.  **Routing:** A central Ingress controller (e.g., NGINX or Google Cloud Gateway) that routes traffic based on the host header or path prefix to the correct customer's service (e.g., `cust123.banditdb.com` -> `Service` in namespace `cust123`).

### B. Observability & Billing
1.  **Metrics:** Deploy Prometheus/Grafana in the GKE cluster to scrape the `/metrics` endpoint of every BanditDB pod.
2.  **Billing Metrics:** You need to bill based on usage. The most straightforward metric is **API Requests** (Predictions + Rewards). You can scrape the `req_2xx` counters from the Prometheus metrics and push them to your billing provider (e.g., Stripe Metered Billing) daily.

### C. Automated Backups & Disaster Recovery
1.  **Volume Snapshots:** Use GCP's native CSI volume snapshotting to snapshot the PVCs of all `StatefulSets` daily.
2.  **The "Restore Drill" Mandate:** As suggested, backups are useless if untested. Build a simple script that runs weekly: it takes a random customer snapshot, provisions a temporary namespace, restores the snapshot to a new PVC, boots a BanditDB pod, and hits the `/health` endpoint to verify the checkpoint loaded successfully.

## 3. Product Tiering Strategy

Start simple and scale up.

### Tier 1: Developer / Prototype (Free or $29/mo)
*   **Architecture:** Multi-tenant (logical separation).
*   **Implementation:** Use a single, large BanditDB instance with `tenantMode: true`. Customers share the compute and memory.
*   **Limits:** Strict rate limits (e.g., 10 req/sec) and volume limits.
*   **Goal:** Frictionless onboarding and experimentation. Let developers prove the value before paying.

### Tier 2: Pro / Mid-Market ($299 - $999/mo)
*   **Architecture:** Dedicated Data Plane.
*   **Implementation:** The "Dedicated by Default" model described above (1 Namespace, 1 Pod, 1 PVC).
*   **SLA:** 99.9% uptime.
*   **Goal:** The core revenue driver. Provides predictable performance and isolation for production workloads.

### Tier 3: Enterprise (Custom Pricing)
*   *Defer this until Stage 2 (Post-Launch).*
*   **Features to build later:** VPC Peering (so their services can talk to BanditDB privately), cross-region replication (active-passive), Bring-Your-Own-Key (BYOK) encryption, SOC2 compliance, and dedicated support SLAs.

## 4. Go-To-Market & Positioning

*   **The Pitch:** "The world's first ultra-fast, managed Contextual Bandit Database. Stop building brittle RL infrastructure; start optimizing in minutes."
*   **Target Audience:** Data scientists and ML engineers at mid-market tech companies who understand the value of Reinforcement Learning (personalization, dynamic pricing, ad routing) but lack the backend engineering resources to build and maintain the real-time serving infrastructure.
*   **The Wedge:** Offer SDKs (Python, TypeScript, Go) that make integration trivial. The value of BanditDB is that it abstracts away the complex math and concurrency; the SDKs must reflect that simplicity.

## Summary Checklist for Launch:
1. [ ] Finalize Stripe integration for metered billing.
2. [ ] Build the automated Helm provisioner (Control Plane).
3. [ ] Configure GKE with NGINX Ingress for dynamic routing.
4. [ ] Set up Prometheus for global metric aggregation.
5. [ ] Automate daily PVC snapshots via GCP CSI.
6. [ ] Publish documentation and SDKs.