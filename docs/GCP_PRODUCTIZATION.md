# BanditDB: GCP Productization Roadmap

This document outlines the architecture for deploying and productizing BanditDB as a managed or semi-managed service on **Google Cloud Platform (GCP)**.

---

## 1. Enterprise Architecture

To move beyond a single-node Rust binary, BanditDB on GCP utilizes a "Sidecar Durability" pattern combined with Google's world-class analytics stack.

### Compute Layer: Google Compute Engine (GCE)
*   **Instance Type:** `c3-standard-4` (Compute-optimized) or `t2d-standard-4` (Tau T2D for best price/performance).
*   **Security:** Deploy on **Confidential VMs**. This ensures that the agent's "intuition" (the model matrices) is encrypted even while in use in RAM.
*   **Persistence:** Attach a **Local SSD** (375GB) for the Write-Ahead Log (WAL) to ensure 300μs latency. 

### Storage Layer: Google Cloud Storage (GCS)
*   **Durability:** A lightweight sidecar process (or an engine hook) monitors the `data/` directory.
*   **Sync Logic:** Every time `/checkpoint` is called, the `checkpoint.json` and any new `.safetensors` or `.parquet` files are synced to a private GCS bucket.
*   **Disaster Recovery:** If the GCE instance dies, the new instance pulls the latest checkpoint from GCS and replays the WAL tail.

### Analytics Layer: BigQuery Omni
*   **Zero-ETL Causal Inference:** Parquet files in GCS are registered as **BigQuery External Tables**.
*   **Workflow:** Data Scientists run SQL directly on production interaction data without needing a data engineering pipeline.

---

## 2. The "Vertex AI" Integration

BanditDB should be positioned as the **Memory Extension** for Vertex AI Reasoning Engine.

1.  **Vertex Extensions:** Package the BanditDB API as an OpenAPI spec and register it as a Vertex AI Extension.
2.  **Gemini Tool Use:** Provide a system prompt template that allows Gemini models to autonomously call `predict` and `reward`.
3.  **Confidential Reasoning:** Market the combination of Gemini + BanditDB + Confidential Space as the only way to build self-learning agents for highly regulated sectors (Banking, Gov).

---

## 3. Implementation Plan (Scripts)

### Task A: `scripts/gcp_bigquery_setup.py`
A script that:
1.  Creates a GCS bucket for BanditDB exports.
2.  Creates a BigQuery Dataset.
3.  Creates an External Table mapped to the GCS bucket.
4.  Defines the schema automatically based on the BanditDB Parquet structure.

### Task B: `scripts/gcp_sync_sidecar.sh`
A simple bash loop using `gsutil -m rsync` that triggers on file changes to ensure the WAL is backed up to the cloud every few seconds.

---

## 4. Product Tiers

| Feature | Community (OSS) | Enterprise (GCP) |
| :--- | :--- | :--- |
| **Logic** | LinUCB / Thompson Sampling | Progressive (Auto-Tournament) |
| **Durability** | Local Disk WAL | GCS Continuous Sync |
| **Analytics** | Raw Parquet files | BigQuery SQL Dashboard |
| **Security** | API Key | IAM + Confidential VMs |
| **Support** | GitHub Issues | 99.9% Uptime SLA |

---

## 5. Deployment Pitch for GCP Customers

> *"You are using Vertex AI for reasoning and Vector Search for facts. But your agents are still stateless. BanditDB adds the **Active Learning Layer** to your GCP stack. It lives next to your agents on GCE, syncs its soul to GCS, and lets your Data Scientists audit every decision in BigQuery. Zero latency, infinite learning."*
