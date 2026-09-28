//! Runtime tenant and API key provisioning.
//!
//! Keys configured through `BANDITDB_API_KEYS` are fixed at startup, which is
//! correct for a single-tenant install and unusable for a hosted one: a signup at
//! 3am cannot require a restart. This module holds the tenants a control plane
//! creates at runtime, persists them beside the data directory, and answers the
//! authentication question in constant work regardless of how many exist.
//!
//! ## Why hashes, and why SHA-256 specifically
//!
//! Keys are stored as SHA-256 digests, never in the clear: a leaked snapshot of
//! the store should not be a leak of working credentials.
//!
//! SHA-256 rather than a password KDF is a deliberate choice. Argon2 and bcrypt
//! are slow *on purpose* — tens of milliseconds — because passwords carry perhaps
//! 30 bits of entropy and an attacker can enumerate them. An API key issued here
//! carries 128+ bits from a CSPRNG, so enumeration is not on the table, and a slow
//! KDF on the authentication path would cost far more than the decision it guards
//! (a decision is measured in microseconds).
//!
//! ## Why a map lookup is not a timing leak
//!
//! The legacy registry compares the presented key against every configured key in
//! constant time, so that no comparison reveals how many leading bytes matched.
//! That is O(keys) per request — at a thousand tenants it costs more than scoring.
//!
//! Hashing first removes the need. The lookup key is a digest of the secret, and
//! an attacker cannot walk a digest backwards a byte at a time: near-misses in the
//! input produce unrelated digests, so probe timing carries no gradient to follow.
//! This is the same construction GitHub and Stripe use for token lookup.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Quotas a control plane assigns to a tenant. Stored here so the engine can
/// answer "what is this tenant allowed?" without calling back to the control
/// plane on the request path — the control plane must never be a dependency of a
/// decision.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct TenantQuotas {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_campaigns:      Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_campaign_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_feature_dim:    Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit_per_sec: Option<u32>,
}

/// One API key as stored: the digest plus what it authorises.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TenantKey {
    /// Lowercase hex SHA-256 of the key. The key itself is never stored.
    pub hash:   String,
    /// "admin" | "writer" | "reader", matching the engine's role ladder.
    pub role:   String,
    /// Leading characters of the key, for display in a console. Not a secret and
    /// not used for authentication.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Tenant {
    pub id:     String,
    #[serde(default)]
    pub keys:   Vec<TenantKey>,
    #[serde(default)]
    pub quotas: TenantQuotas,
    /// active | suspended. A suspended tenant authenticates but is refused, so
    /// non-payment reads as a clear error rather than an invalid key.
    #[serde(default = "default_status")]
    pub status: String,
    #[serde(default)]
    pub updated_at: u64,
}

fn default_status() -> String { "active".to_string() }

impl Tenant {
    pub fn is_active(&self) -> bool { self.status == "active" }
}

/// What a successful lookup yields.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyMatch {
    pub tenant_id: String,
    pub role:      String,
    pub active:    bool,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct StoreFile {
    #[serde(default)]
    tenants: Vec<Tenant>,
}

/// In-memory tenants plus a digest index, persisted as JSON in the data directory.
///
/// The control plane is the source of truth; this is a local replica so the engine
/// can authenticate offline. Writes are whole-file and atomic (temp + rename +
/// fsync), which is fine because they happen on provisioning, not on the request
/// path.
#[derive(Debug, Default)]
pub struct TenantStore {
    tenants: HashMap<String, Tenant>,
    /// key digest → (tenant id, role). Rebuilt whenever tenants change.
    index:   HashMap<String, (String, String)>,
    path:    Option<PathBuf>,
}

impl TenantStore {
    pub fn new() -> Self { Self::default() }

    /// Load from `<data_dir>/tenants.json`, or start empty if absent.
    ///
    /// A corrupt file is a hard error rather than an empty start: coming up with
    /// no tenants would authenticate nobody while looking healthy, and the next
    /// write would overwrite the evidence.
    pub fn load(data_dir: &str) -> Result<Self, String> {
        let path = Path::new(data_dir).join("tenants.json");
        let mut store = Self { tenants: HashMap::new(), index: HashMap::new(), path: Some(path.clone()) };
        if !path.exists() {
            return Ok(store);
        }
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("tenants.json unreadable: {e}"))?;
        let parsed: StoreFile = serde_json::from_str(&raw)
            .map_err(|e| format!("tenants.json is corrupt: {e}"))?;
        for t in parsed.tenants {
            store.tenants.insert(t.id.clone(), t);
        }
        store.reindex();
        Ok(store)
    }

    fn reindex(&mut self) {
        self.index.clear();
        for t in self.tenants.values() {
            for k in &t.keys {
                self.index.insert(k.hash.to_lowercase(), (t.id.clone(), k.role.clone()));
            }
        }
    }

    fn persist(&self) -> Result<(), String> {
        let Some(path) = &self.path else { return Ok(()) };
        let mut tenants: Vec<&Tenant> = self.tenants.values().collect();
        tenants.sort_by(|a, b| a.id.cmp(&b.id));      // stable file, readable diffs
        let file = StoreFile { tenants: tenants.into_iter().cloned().collect() };
        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| format!("tenant serialisation failed: {e}"))?;

        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json).map_err(|e| format!("tenant write failed: {e}"))?;
        // fsync the file, then rename, then fsync the directory — the rename is
        // only durable once the directory entry is.
        if let Ok(f) = std::fs::File::open(&tmp) { let _ = f.sync_all(); }
        std::fs::rename(&tmp, path).map_err(|e| format!("tenant rename failed: {e}"))?;
        if let Some(dir) = path.parent() {
            if let Ok(d) = std::fs::File::open(dir) { let _ = d.sync_all(); }
        }
        Ok(())
    }

    /// Create or replace a tenant. Idempotent: the control plane retries this
    /// until it succeeds, so applying the same payload twice must be a no-op.
    pub fn upsert(&mut self, mut tenant: Tenant) -> Result<(), String> {
        if tenant.id.is_empty() || tenant.id.len() > 128 {
            return Err("tenant id must be 1–128 characters".into());
        }
        if !tenant.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err("tenant id may only contain ASCII letters, digits, '-', and '_'".into());
        }
        for k in &tenant.keys {
            if k.hash.len() != 64 || !k.hash.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(format!("key hash must be 64 hex characters, got {:?}", k.hash));
            }
            if !matches!(k.role.as_str(), "admin" | "writer" | "reader") {
                return Err(format!("unknown role {:?}", k.role));
            }
        }
        // Reject a digest already claimed by a different tenant: whoever presented
        // that key would otherwise authenticate as whichever tenant indexed last.
        for k in &tenant.keys {
            if let Some((owner, _)) = self.index.get(&k.hash.to_lowercase()) {
                if owner != &tenant.id {
                    return Err(format!("key already assigned to tenant '{owner}'"));
                }
            }
        }
        tenant.updated_at = now_secs();
        self.tenants.insert(tenant.id.clone(), tenant);
        self.reindex();
        self.persist()
    }

    pub fn remove(&mut self, tenant_id: &str) -> Result<bool, String> {
        let existed = self.tenants.remove(tenant_id).is_some();
        if existed {
            self.reindex();
            self.persist()?;
        }
        Ok(existed)
    }

    /// Authenticate a presented key. O(1) in the number of tenants.
    pub fn authenticate(&self, presented: &str) -> Option<KeyMatch> {
        let digest = hash_key(presented);
        let (tenant_id, role) = self.index.get(&digest)?;
        let active = self.tenants.get(tenant_id).map(|t| t.is_active()).unwrap_or(false);
        Some(KeyMatch { tenant_id: tenant_id.clone(), role: role.clone(), active })
    }

    pub fn get(&self, tenant_id: &str) -> Option<&Tenant> { self.tenants.get(tenant_id) }
    pub fn len(&self) -> usize { self.tenants.len() }
    pub fn is_empty(&self) -> bool { self.tenants.is_empty() }
    pub fn ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tenants.keys().cloned().collect();
        v.sort();
        v
    }
}

/// Lowercase hex SHA-256 of an API key. The control plane computes the same value
/// when it provisions, so the key itself never crosses the wire to the engine.
pub fn hash_key(key: &str) -> String {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    format!("{:x}", h.finalize())
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
