//! Tenant changes must reach disk before they take effect, so memory and disk
//! never disagree.
//!
//! `upsert` and `remove` used to change the in-memory tenants first and persist
//! after. A failed write returned an error but left the change live: a revocation
//! that "failed" still revoked until the next restart brought the key back. Worse,
//! a retried `remove` found nothing left in memory, skipped the write and reported
//! success — so the tenant and every key it held came back at the next restart.
//!
//! Writes are made to fail for real by making the data directory read-only.

#![cfg(unix)]

use banditdb::tenancy::{hash_key, Tenant, TenantKey, TenantQuotas, TenantStore};
use std::fs;
use std::os::unix::fs::PermissionsExt;

fn tenant(id: &str, keys: &[&str]) -> Tenant {
    Tenant {
        id: id.into(),
        keys: keys.iter().map(|k| TenantKey {
            hash: hash_key(k), role: "admin".into(), prefix: None, last_used_at: 0,
        }).collect(),
        quotas: TenantQuotas::default(),
        status: "active".into(),
        updated_at: 0,
    }
}

fn set_writable(dir: &str, writable: bool) {
    fs::set_permissions(dir, fs::Permissions::from_mode(if writable { 0o755 } else { 0o555 })).unwrap();
}

/// A fresh data dir, or None when this process can write to a read-only
/// directory anyway (root) and the failure cannot be produced.
fn fresh(dir: &str) -> Option<()> {
    let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o755));
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    set_writable(dir, false);
    let bypass = fs::write(format!("{dir}/probe"), b"x").is_ok();
    set_writable(dir, true);
    if bypass {
        eprintln!("SKIPPED: read-only directories are writable to this user");
        return None;
    }
    Some(())
}

#[test]
fn failed_upsert_changes_nothing() {
    let dir = "/tmp/banditdb_tenant_store_upsert";
    if fresh(dir).is_none() { return; }
    let mut store = TenantStore::load(dir).unwrap();
    store.upsert(tenant("t", &["key-t"])).unwrap();

    set_writable(dir, false);
    assert!(store.upsert(tenant("t", &[])).is_err(), "the write must fail");
    assert!(store.upsert(tenant("u", &["key-u"])).is_err(), "the write must fail");
    assert!(store.authenticate("key-t").is_some(),
        "a revocation that failed to persist must not take effect — disk still has the key");
    assert!(store.authenticate("key-u").is_none(),
        "a grant that failed to persist must not take effect");

    // The control plane's retry succeeds once the disk does.
    set_writable(dir, true);
    store.upsert(tenant("t", &[])).unwrap();
    assert!(store.authenticate("key-t").is_none());
    assert!(TenantStore::load(dir).unwrap().authenticate("key-t").is_none(),
        "the revocation must survive a reload");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn retried_remove_after_a_failed_write_still_persists() {
    let dir = "/tmp/banditdb_tenant_store_remove";
    if fresh(dir).is_none() { return; }
    let mut store = TenantStore::load(dir).unwrap();
    store.upsert(tenant("t", &["key-t"])).unwrap();

    set_writable(dir, false);
    assert!(store.remove("t").is_err(), "the write must fail");
    assert!(store.authenticate("key-t").is_some(),
        "a removal that failed to persist must not take effect");

    set_writable(dir, true);
    assert_eq!(store.remove("t"), Ok(true), "the retry must find the tenant and remove it");
    let reloaded = TenantStore::load(dir).unwrap();
    assert!(reloaded.get("t").is_none() && reloaded.authenticate("key-t").is_none(),
        "a removal reported as successful came back after a reload");
    let _ = fs::remove_dir_all(dir);
}
