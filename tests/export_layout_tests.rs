//! Parquet export layout: where shards are written, what pruning deletes.
//!
//! Tenant campaigns are stored as `{tenant}/{campaign}`. Before per-tenant
//! directories the writer kept only `Path::file_name()` of the shard name, so:
//!   * every tenant's shards landed flat as `{campaign}_{micros}.parquet`, and two
//!     tenants' `checkout` shards could not be told apart;
//!   * pruning looked for `{tenant}/{campaign}_` among flat names, never matched,
//!     and tenant shards grew without bound;
//!   * pruning `checkout` by prefix also deleted `checkout_v2`'s shards.
//!
//! Each of those is pinned below, along with the path checks that keep a
//! WAL-sourced id from writing outside `exports/`.

use banditdb::engine::{export_location, prune_export_shards, shard_stem, write_campaign_parquet};
use banditdb::state::{Algorithm, CompletedInteraction};
use banditdb::BanditDB;
use polars::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// A fresh directory per test, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "banditdb_export_layout_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    fn exports(&self) -> String {
        self.0.join("exports").to_string_lossy().to_string()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn rows(prefix: &str, n: usize) -> Vec<CompletedInteraction> {
    (0..n)
        .map(|i| CompletedInteraction {
            interaction_id: format!("{prefix}-{i}"),
            arm_id: "a".into(),
            context: vec![i as f64, 1.0],
            reward: 1.0,
            predicted_at: 1_700_000_000,
            rewarded_at: 1_700_000_001,
            propensity: Some(0.5),
        })
        .collect()
}

/// File names directly in `dir` (no recursion), sorted.
fn files_in(dir: impl AsRef<Path>) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .filter(|e| e.file_type().unwrap().is_file())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn interaction_ids(path: impl AsRef<Path>) -> Vec<String> {
    let df = ParquetReader::new(fs::File::open(path).unwrap())
        .finish()
        .unwrap();
    let col = df.column("interaction_id").unwrap().str().unwrap().clone();
    let mut ids: Vec<String> = col.into_iter().map(|v| v.unwrap().to_string()).collect();
    ids.sort();
    ids
}

/// Write `n` shards with distinct microsecond timestamps.
fn write_n(exports: &str, campaign: &str, n: usize) {
    for i in 0..n {
        write_campaign_parquet(
            exports,
            campaign,
            &rows(&format!("{campaign}-{i}"), 1),
            None,
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(2));
    }
}

// ---------------------------------------------------------------------------
// Naming
// ---------------------------------------------------------------------------

#[test]
fn location_is_a_directory_per_tenant() {
    let (dir, stem) = export_location("/x/exports", "org_01H/checkout").unwrap();
    assert_eq!(dir, PathBuf::from("/x/exports/org_01H"));
    assert_eq!(stem, "checkout");
}

#[test]
fn location_without_tenant_stays_flat() {
    let (dir, stem) = export_location("/x/exports", "checkout").unwrap();
    assert_eq!(dir, PathBuf::from("/x/exports"));
    assert_eq!(stem, "checkout");
}

#[test]
fn location_refuses_anything_but_plain_ids() {
    let long = "a".repeat(129);
    for bad in [
        "",
        "..",
        ".",
        "../evil",
        "t/..",
        "t/../evil",
        "../t/c",
        "/c",
        "t/",
        "t/c/d",
        "t//c",
        "a b",
        "t/a.b",
        "t\\c",
        "t/c\0",
        long.as_str(),
    ] {
        assert!(
            export_location("/x/exports", bad).is_err(),
            "accepted {bad:?}"
        );
    }
    assert!(
        export_location("/x/exports", &"a".repeat(128)).is_ok(),
        "128 chars is the limit, not over it"
    );
}

#[test]
fn stem_takes_the_last_underscore_part_as_the_timestamp() {
    assert_eq!(
        shard_stem("checkout_1700000000000000.parquet"),
        Some("checkout")
    );
    assert_eq!(
        shard_stem("checkout_v2_1700000000000000.parquet"),
        Some("checkout_v2")
    );
    assert_eq!(
        shard_stem("promo_2024_1700000000000000.parquet"),
        Some("promo_2024")
    );
    for not_a_shard in [
        "checkout.parquet",
        "_1700000000000000.parquet",
        "checkout_.parquet",
        "checkout_17000x.parquet",
        "checkout_1700000000000000.parquet.tmp",
        "checkout_1700000000000000.csv",
    ] {
        assert_eq!(shard_stem(not_a_shard), None, "{not_a_shard}");
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

#[test]
fn tenant_shard_is_written_in_its_tenant_directory() {
    let tmp = TempDir::new("tenant_write");
    let exports = tmp.exports();
    write_campaign_parquet(&exports, "tenant_a/checkout", &rows("a", 3), None).unwrap();

    assert!(
        files_in(&exports).is_empty(),
        "nothing flat in exports/: {:?}",
        files_in(&exports)
    );
    let files = files_in(format!("{exports}/tenant_a"));
    assert_eq!(files.len(), 1, "{files:?}");
    assert_eq!(shard_stem(&files[0]), Some("checkout"));
    assert_eq!(
        interaction_ids(format!("{exports}/tenant_a/{}", files[0])),
        vec!["a-0", "a-1", "a-2"]
    );
}

#[test]
fn same_campaign_in_two_tenants_never_mixes() {
    let tmp = TempDir::new("two_tenants");
    let exports = tmp.exports();
    write_campaign_parquet(&exports, "tenant_a/checkout", &rows("a", 2), None).unwrap();
    write_campaign_parquet(&exports, "tenant_b/checkout", &rows("b", 3), None).unwrap();

    let a = files_in(format!("{exports}/tenant_a"));
    let b = files_in(format!("{exports}/tenant_b"));
    assert_eq!((a.len(), b.len()), (1, 1));
    assert_eq!(
        interaction_ids(format!("{exports}/tenant_a/{}", a[0])),
        vec!["a-0", "a-1"]
    );
    assert_eq!(
        interaction_ids(format!("{exports}/tenant_b/{}", b[0])),
        vec!["b-0", "b-1", "b-2"]
    );
}

#[test]
fn campaign_without_tenant_is_written_flat() {
    let tmp = TempDir::new("flat_write");
    let exports = tmp.exports();
    write_campaign_parquet(&exports, "solo", &rows("s", 1), None).unwrap();

    let files = files_in(&exports);
    assert_eq!(files.len(), 1, "{files:?}");
    assert_eq!(shard_stem(&files[0]), Some("solo"));
}

#[test]
fn no_tmp_file_is_left_behind() {
    let tmp = TempDir::new("no_tmp");
    let exports = tmp.exports();
    write_campaign_parquet(&exports, "tenant_a/checkout", &rows("a", 1), None).unwrap();
    let files = files_in(format!("{exports}/tenant_a"));
    assert!(files.iter().all(|f| f.ends_with(".parquet")), "{files:?}");
}

#[test]
fn traversal_ids_write_nothing_anywhere() {
    let tmp = TempDir::new("traversal");
    let exports = tmp.exports();
    fs::create_dir_all(&exports).unwrap();
    for bad in ["../evil", "t/../../evil", "t/../evil", "../evil/c", "t/c/d"] {
        assert!(
            write_campaign_parquet(&exports, bad, &rows("x", 1), None).is_err(),
            "wrote {bad:?}"
        );
    }
    // Nothing in exports/, nothing beside it, no stray directories.
    let mut all = Vec::new();
    for entry in walk(&tmp.0) {
        all.push(entry);
    }
    assert_eq!(
        all,
        vec![PathBuf::from(&exports)],
        "unexpected paths: {all:?}"
    );
}

fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in fs::read_dir(root).unwrap().filter_map(|e| e.ok()) {
        out.push(e.path());
        if e.file_type().unwrap().is_dir() {
            out.extend(walk(&e.path()));
        }
    }
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// Pruning
// ---------------------------------------------------------------------------

#[test]
fn prune_keeps_the_newest_tenant_shards() {
    let tmp = TempDir::new("prune_tenant");
    let exports = tmp.exports();
    write_n(&exports, "tenant_a/checkout", 5);
    let before = files_in(format!("{exports}/tenant_a"));

    prune_export_shards(&exports, "tenant_a/checkout", 2);

    let after = files_in(format!("{exports}/tenant_a"));
    assert_eq!(after, before[3..].to_vec(), "kept the two newest");
}

#[test]
fn prune_leaves_another_tenants_same_campaign_alone() {
    let tmp = TempDir::new("prune_other_tenant");
    let exports = tmp.exports();
    write_n(&exports, "tenant_a/checkout", 4);
    write_n(&exports, "tenant_b/checkout", 4);

    prune_export_shards(&exports, "tenant_a/checkout", 1);

    assert_eq!(files_in(format!("{exports}/tenant_a")).len(), 1);
    assert_eq!(
        files_in(format!("{exports}/tenant_b")).len(),
        4,
        "tenant_b untouched"
    );
}

#[test]
fn prune_leaves_a_campaign_sharing_the_prefix_alone() {
    let tmp = TempDir::new("prune_prefix");
    let exports = tmp.exports();
    write_n(&exports, "tenant_a/checkout", 3);
    write_n(&exports, "tenant_a/checkout_v2", 3);

    prune_export_shards(&exports, "tenant_a/checkout", 1);

    let left = files_in(format!("{exports}/tenant_a"));
    let count = |stem: &str| left.iter().filter(|f| shard_stem(f) == Some(stem)).count();
    assert_eq!(count("checkout"), 1);
    assert_eq!(count("checkout_v2"), 3, "checkout_v2 untouched: {left:?}");
}

#[test]
fn prune_still_works_for_flat_campaigns_and_skips_tenant_dirs() {
    let tmp = TempDir::new("prune_flat");
    let exports = tmp.exports();
    write_n(&exports, "checkout", 3);
    write_n(&exports, "tenant_a/checkout", 3);

    prune_export_shards(&exports, "checkout", 1);

    assert_eq!(files_in(&exports).len(), 1);
    assert_eq!(
        files_in(format!("{exports}/tenant_a")).len(),
        3,
        "tenant dir untouched"
    );
}

#[test]
fn prune_with_zero_retention_keeps_everything() {
    let tmp = TempDir::new("prune_zero");
    let exports = tmp.exports();
    write_n(&exports, "tenant_a/checkout", 3);
    prune_export_shards(&exports, "tenant_a/checkout", 0);
    assert_eq!(files_in(format!("{exports}/tenant_a")).len(), 3);
}

#[test]
fn prune_refuses_traversal_ids() {
    let tmp = TempDir::new("prune_traversal");
    let exports = tmp.exports();
    // A victim directory beside exports/ that a `..` id would otherwise reach.
    let victim = tmp.0.join("victim");
    fs::create_dir_all(&victim).unwrap();
    for i in 0..3 {
        fs::write(victim.join(format!("x_17000000000000{i}.parquet")), b"keep").unwrap();
    }
    fs::create_dir_all(&exports).unwrap();

    prune_export_shards(&exports, "../victim/x", 1);
    prune_export_shards(&exports, "../victim", 1);

    assert_eq!(
        files_in(&victim).len(),
        3,
        "nothing outside exports/ was deleted"
    );
}

// ---------------------------------------------------------------------------
// End to end: checkpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn checkpoint_writes_each_tenants_shards_in_its_own_directory() {
    let tmp = TempDir::new("checkpoint");
    let data_dir = tmp.0.to_string_lossy().to_string();
    let wal = format!("{data_dir}/bandit_wal.jsonl");
    let db = BanditDB::new(&wal, &data_dir);

    // Stored ids exactly as the HTTP layer namespaces them (`ns` in main.rs).
    let campaigns = [
        ("tenant_a/checkout", 2),
        ("tenant_b/checkout", 3),
        ("solo", 1),
    ];
    for (id, _) in campaigns {
        db.add_campaign(
            id,
            vec!["x".into(), "y".into()],
            2,
            1.0,
            Algorithm::Linucb,
            None,
            None,
        )
        .await
        .unwrap();
    }
    for (id, n) in campaigns {
        for _ in 0..n {
            let (_, iid) = db.predict(id, vec![0.1, 0.2]).unwrap();
            db.reward(&iid, 1.0).await.unwrap();
        }
    }
    db.checkpoint().await.unwrap();

    let exports = tmp.exports();
    let a = files_in(format!("{exports}/tenant_a"));
    let b = files_in(format!("{exports}/tenant_b"));
    let flat = files_in(&exports);
    assert_eq!(a.len(), 1, "tenant_a: {a:?}");
    assert_eq!(b.len(), 1, "tenant_b: {b:?}");
    assert_eq!(
        flat.len(),
        1,
        "only the tenantless campaign is flat: {flat:?}"
    );
    assert_eq!(shard_stem(&a[0]), Some("checkout"));
    assert_eq!(shard_stem(&b[0]), Some("checkout"));
    assert_eq!(shard_stem(&flat[0]), Some("solo"));
    assert_eq!(
        interaction_ids(format!("{exports}/tenant_a/{}", a[0])).len(),
        2
    );
    assert_eq!(
        interaction_ids(format!("{exports}/tenant_b/{}", b[0])).len(),
        3
    );
}
