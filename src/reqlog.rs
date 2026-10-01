//! Per-tenant ring of recent requests, for a console's request inspector.
//!
//! When something is wrong, the question a developer asks is "what did my last
//! few calls actually do?" — and until now the only answer was the process log,
//! which a tenant cannot see. This keeps a small window per tenant so the console
//! can show status, latency and the error text the engine returned.
//!
//! ## Why in memory, and why small
//!
//! This is a debugging aid, not an audit trail — `BANDITDB_AUDIT_LOG` already
//! covers the durable write-path record. A bounded in-memory ring costs nothing
//! to maintain, disappears on restart (which is fine for "what just happened?"),
//! and cannot grow into a disk-space problem on a busy tenant.
//!
//! ## Cost on the request path
//!
//! Recording is a short critical section on a per-tenant lock: push, and pop the
//! oldest when full. Tenants have their own rings, so two tenants never contend,
//! and a single tenant's requests are already serialised through the network
//! anyway. Measured against a decision costing tens of microseconds, the push is
//! noise — but it is still only done for provisioned tenants, since a
//! single-tenant install has the process log right there.

use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};

/// Entries kept per tenant. Enough to cover "what did I just try?" without
/// turning into storage.
const RING_CAPACITY: usize = 50;

#[derive(Serialize, Debug, Clone)]
pub struct RequestRecord {
    /// Unix seconds.
    pub at: u64,
    pub method: String,
    /// Path with the tenant's namespace already stripped, so it reads the way the
    /// caller wrote it.
    pub path: String,
    pub status: u16,
    pub latency_us: u64,
}

#[derive(Default)]
pub struct RequestLog {
    /// tenant id → ring. A read lock covers lookup; the per-tenant mutex covers
    /// the push, so recording never blocks another tenant.
    tenants: RwLock<HashMap<String, Mutex<VecDeque<RequestRecord>>>>,
}

impl RequestLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, tenant_id: &str, record: RequestRecord) {
        // Fast path: the tenant already has a ring.
        {
            let guard = self.tenants.read();
            if let Some(ring) = guard.get(tenant_id) {
                let mut ring = ring.lock();
                if ring.len() == RING_CAPACITY {
                    ring.pop_front();
                }
                ring.push_back(record);
                return;
            }
        }
        // First request for this tenant: take the write lock once, then never
        // again for the life of the process.
        let mut guard = self.tenants.write();
        let ring = guard
            .entry(tenant_id.to_string())
            .or_insert_with(|| Mutex::new(VecDeque::with_capacity(RING_CAPACITY)));
        let mut ring = ring.lock();
        if ring.len() == RING_CAPACITY {
            ring.pop_front();
        }
        ring.push_back(record);
    }

    /// Most recent first, so a console renders it without reversing.
    pub fn recent(&self, tenant_id: &str, limit: usize) -> Vec<RequestRecord> {
        let guard = self.tenants.read();
        match guard.get(tenant_id) {
            Some(ring) => ring.lock().iter().rev().take(limit).cloned().collect(),
            None => Vec::new(),
        }
    }

    /// Drop a tenant's ring — called when the tenant is removed, so a deleted
    /// customer leaves nothing behind in memory.
    pub fn forget(&self, tenant_id: &str) {
        self.tenants.write().remove(tenant_id);
    }
}
