# BanditDB Distributed Architecture on GCP

## Objective
Deploy a highly available, 3-node distributed installation of BanditDB on Google Cloud Platform (GCP) using Campaign-Level Sharding.

## Architectural Approach vs. Manual Prefixing

You suggested prefixing the campaign ID with the node name (e.g., `node-1_campaign_A`). While this creates a sharding mechanism, it creates a **brittle system**:
*   If `node-1` crashes and `node-2` takes over its volume, the system gets confused.
*   Scaling from 3 nodes to 5 nodes requires manually renaming campaigns in your application code.

**The Solution: Consistent Hashing at the Load Balancer**
Instead of changing the `campaign_id`, we let the **Google Cloud Global Load Balancer (GXLB)** do the math. The Load Balancer hashes the `campaign_id` and automatically routes it to a specific node. If a node fails, the Load Balancer seamlessly recalculates the hash and routes traffic to a surviving node.

## Phase 1: Database Code Updates (The "Update")

To make the Load Balancer's job possible, the Load Balancer needs to "see" the `campaign_id` on every request. Currently, the GCP Load Balancer cannot inspect JSON bodies; it routes based on **Headers** or **URLs**.

1.  **API Restructuring:** 
    Move `campaign_id` from the JSON body to the URL path for all primary endpoints:
    *   `POST /campaign/:campaign_id/predict`
    *   `POST /campaign/:campaign_id/reward` (This is the most critical change. Currently, `/reward` only takes `interaction_id`. It must be updated to include the campaign ID so the Load Balancer routes the reward to the exact node holding the matrix state).
2.  **Stateless Reward Lookups:**
    Ensure that when a reward reaches a node, if that node recently took over from a crashed peer and doesn't have the `interaction_id` in its local RAM cache, it can still safely process the reward by pulling the context from the shared WAL.

## Phase 2: GCP Infrastructure Setup

1.  **Compute (Google Kubernetes Engine - GKE):**
    Create a 3-node GKE cluster to host the BanditDB pods.
2.  **Shared Storage (Cloud Filestore):**
    Provision a basic Cloud Filestore (NFS) instance. Mount this to `/data` across all 3 BanditDB pods. This shared storage guarantees that if Pod A dies, Pod B can instantly read its WAL and Checkpoints.
3.  **Load Balancer (GXLB):**
    Deploy a Global External Application Load Balancer.
    *   Configure a **Backend Service** pointing to your GKE cluster.
    *   Set the **Locality Load Balancing Policy** to `MAGLEV` (Google's consistent hashing algorithm).
    *   Set the **Hash Key** to extract the `:campaign_id` from the URL path.

## Phase 3: Deployment & Verification

1.  Update the existing `docker-compose.yml` and `helm` charts to reflect the new API paths.
2.  Deploy the Helm chart to GKE.
3.  Simulate a node crash (delete a pod) and verify that the Load Balancer instantly re-routes traffic and the new pod recovers state from the shared Filestore WAL within seconds.
