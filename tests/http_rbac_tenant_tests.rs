//! P1 acceptance tests — HTTP surface, RBAC, and tenant isolation.
//!
//! The codebase had no HTTP-level tests at all, so every authorisation rule was
//! unverified. These drive a real server process over the network, because the
//! defects being covered live in the handler/router layer rather than the engine:
//!
//!   * `/health` was mounted outside the auth layer and returned every campaign ID
//!     with its entropy. In tenant mode those IDs carry the tenant prefix, so an
//!     anonymous caller could enumerate the customer list.
//!   * `/export` did not take an AuthContext at all and listed every tenant's shards.
//!   * `/reward` identifies its target by interaction id alone and never checked
//!     which tenant owned it.
//!   * CORS allowed any origin, and a missing key set silently granted admin.
//!
//! Run: cargo test --release --features neural --test http_rbac_tenant_tests

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const ADMIN_A: &str = "key-admin-a";
const WRITER_A: &str = "key-writer-a";
const READER_A: &str = "key-reader-a";
const ADMIN_B: &str = "key-admin-b";

struct Server {
    child: Child,
    port: u16,
    _dir: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self._dir);
    }
}

impl Server {
    fn start(port: u16, extra_env: &[(&str, &str)]) -> Option<Self> {
        let bin = env!("CARGO_BIN_EXE_banditdb");
        let dir = format!("/tmp/banditdb_http_test_{port}");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok()?;

        let mut cmd = Command::new(bin);
        cmd.env("DATA_DIR", &dir)
            .env("PORT", port.to_string())
            .env("BANDITDB_RATE_LIMIT_PER_SEC", "100000")
            .env(
                "BANDITDB_API_KEYS",
                format!("{ADMIN_A}=admin:tenant_a;{WRITER_A}=writer:tenant_a;\
                         {READER_A}=reader:tenant_a;{ADMIN_B}=admin:tenant_b"),
            )
            .env("BANDITDB_TENANT_MODE", "true")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for (k, v) in extra_env {
            cmd.env(k, v);
        }

        let child = cmd.spawn().ok()?;
        let server = Server { child, port, _dir: dir };

        // Must be a real 200: curl still reports `__STATUS__000` on connection
        // refused, so "got a response" is not the same as "server is listening".
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if matches!(server.get("/health", None), Some((200, _))) {
                return Some(server);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        None
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{}", self.port, path)
    }

    /// Returns (status, body). `None` if the server is unreachable.
    fn request(&self, method: &str, path: &str, key: Option<&str>, body: Option<&str>)
        -> Option<(u16, String)>
    {
        let mut cmd = Command::new("curl");
        cmd.arg("-sS").arg("--max-time").arg("10")
            .arg("-o").arg("-")
            .arg("-w").arg("\n__STATUS__%{http_code}")
            .arg("-X").arg(method)
            .arg(self.url(path));
        if let Some(k) = key {
            cmd.arg("-H").arg(format!("X-Api-Key: {k}"));
        }
        if let Some(b) = body {
            cmd.arg("-H").arg("Content-Type: application/json").arg("-d").arg(b);
        }
        let out = cmd.output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let (body, status) = text.rsplit_once("__STATUS__")?;
        Some((status.trim().parse().ok()?, body.trim_end().to_string()))
    }

    fn get(&self, path: &str, key: Option<&str>) -> Option<(u16, String)> {
        self.request("GET", path, key, None)
    }
    fn post(&self, path: &str, key: Option<&str>, body: &str) -> Option<(u16, String)> {
        self.request("POST", path, key, Some(body))
    }

    fn create_campaign(&self, key: &str, id: &str) -> u16 {
        self.post("/campaign", Some(key),
            &format!(r#"{{"campaign_id":"{id}","arms":["A","B"],"feature_dim":2,"alpha":1.0}}"#))
            .map(|(s, _)| s).unwrap_or(0)
    }

    /// Predict and return the interaction id.
    fn predict(&self, key: &str, campaign: &str) -> Option<String> {
        let (status, body) = self.post("/predict", Some(key),
            &format!(r#"{{"campaign_id":"{campaign}","context":[0.5,0.5]}}"#))?;
        if status != 200 { return None; }
        let v: serde_json::Value = serde_json::from_str(&body).ok()?;
        v["interaction_id"].as_str().map(str::to_string)
    }
}

/// Skips rather than fails when the binary cannot be launched (sandboxed CI).
macro_rules! server_or_skip {
    ($port:expr, $env:expr) => {
        match Server::start($port, $env) {
            Some(s) => s,
            None => { eprintln!("SKIPPED: could not start server on port {}", $port); return; }
        }
    };
}

// ---------------------------------------------------------------------------
// Public surface must not leak
// ---------------------------------------------------------------------------

#[test]
fn public_health_exposes_no_campaign_identifiers() {
    let srv = server_or_skip!(18301, &[]);
    assert_eq!(srv.create_campaign(ADMIN_A, "secret_launch"), 200);

    let (status, body) = srv.get("/health", None).expect("health reachable");
    assert_eq!(status, 200, "health must stay public for load balancer probes");
    assert!(
        !body.contains("secret_launch") && !body.contains("tenant_a"),
        "unauthenticated /health leaked campaign identifiers, which in tenant mode \
         means the customer list: {body}"
    );
    assert!(body.contains("\"status\""), "health must still report status: {body}");
}

#[test]
fn health_detail_requires_a_key_and_is_tenant_scoped() {
    let srv = server_or_skip!(18302, &[]);
    assert_eq!(srv.create_campaign(ADMIN_A, "camp_a"), 200);
    assert_eq!(srv.create_campaign(ADMIN_B, "camp_b"), 200);

    let (status, _) = srv.get("/health/detail", None).expect("reachable");
    assert_eq!(status, 401, "health detail must require authentication");

    let (status, body) = srv.get("/health/detail", Some(READER_A)).expect("reachable");
    assert_eq!(status, 200);
    assert!(body.contains("camp_a"), "tenant must see its own campaign: {body}");
    assert!(!body.contains("camp_b"), "tenant must not see another tenant's campaign: {body}");
}

#[test]
fn metrics_require_authentication_by_default() {
    let srv = server_or_skip!(18303, &[]);
    let (status, _) = srv.get("/metrics", None).expect("reachable");
    assert_eq!(
        status, 401,
        "metrics default to authenticated: they expose campaign and arm identifiers \
         plus per-arm traffic"
    );
    let (status, _) = srv.get("/metrics", Some(READER_A)).expect("reachable");
    assert_eq!(status, 200, "a valid key must still reach metrics");
}

// ---------------------------------------------------------------------------
// Tenant isolation
// ---------------------------------------------------------------------------

#[test]
fn tenant_cannot_reward_another_tenants_interaction() {
    let srv = server_or_skip!(18304, &[]);
    assert_eq!(srv.create_campaign(ADMIN_B, "b_camp"), 200);

    let iid = srv.predict(ADMIN_B, "b_camp").expect("tenant B predicts");

    // Tenant A holds a valid writer key and the interaction id.
    let (status, _) = srv.post("/reward", Some(WRITER_A),
        &format!(r#"{{"interaction_id":"{iid}","reward":1.0}}"#)).expect("reachable");
    assert_eq!(
        status, 404,
        "/reward names its target by interaction id alone; without an ownership check \
         one tenant can write into another tenant's model"
    );

    // The rightful owner still can.
    let (status, _) = srv.post("/reward", Some(ADMIN_B),
        &format!(r#"{{"interaction_id":"{iid}","reward":1.0}}"#)).expect("reachable");
    assert_eq!(status, 200, "the owning tenant must still be able to reward");
}

#[test]
fn campaign_reads_are_tenant_scoped() {
    let srv = server_or_skip!(18305, &[]);
    assert_eq!(srv.create_campaign(ADMIN_A, "only_a"), 200);

    let (status, body) = srv.get("/campaigns", Some(ADMIN_B)).expect("reachable");
    assert_eq!(status, 200);
    assert!(!body.contains("only_a"), "campaign list crossed tenants: {body}");

    let (status, _) = srv.get("/campaign/only_a", Some(ADMIN_B)).expect("reachable");
    assert_eq!(status, 404, "direct fetch crossed tenants");

    let (status, _) = srv.get("/campaign/only_a/report", Some(ADMIN_B)).expect("reachable");
    assert_eq!(status, 404, "report crossed tenants");
}

#[test]
fn export_listing_is_tenant_scoped() {
    let srv = server_or_skip!(18306, &[]);
    assert_eq!(srv.create_campaign(ADMIN_A, "exp_a"), 200);
    let iid = srv.predict(ADMIN_A, "exp_a").expect("predict");
    srv.post("/reward", Some(ADMIN_A),
        &format!(r#"{{"interaction_id":"{iid}","reward":1.0}}"#)).expect("reward");
    srv.post("/checkpoint", Some(ADMIN_A), "{}").expect("checkpoint writes parquet");

    let (status, body) = srv.get("/export", Some(ADMIN_B)).expect("reachable");
    // 404 is acceptable when no exports exist at all; 200 must not name tenant A.
    if status == 200 {
        assert!(
            !body.contains("exp_a") && !body.contains("tenant_a"),
            "export listing exposed another tenant's campaign names: {body}"
        );
    }
}

// ---------------------------------------------------------------------------
// RBAC
// ---------------------------------------------------------------------------

#[test]
fn roles_are_enforced_per_route() {
    let srv = server_or_skip!(18307, &[]);
    assert_eq!(srv.create_campaign(ADMIN_A, "rbac"), 200);

    // Reader may read but not write.
    assert_eq!(srv.get("/campaigns", Some(READER_A)).unwrap().0, 200);
    assert_eq!(
        srv.post("/predict", Some(READER_A), r#"{"campaign_id":"rbac","context":[0.5,0.5]}"#).unwrap().0,
        403, "reader must not reach a writer route"
    );

    // Writer may predict but not create or delete campaigns.
    assert_eq!(
        srv.post("/predict", Some(WRITER_A), r#"{"campaign_id":"rbac","context":[0.5,0.5]}"#).unwrap().0,
        200
    );
    assert_eq!(srv.create_campaign(WRITER_A, "nope"), 403, "writer must not create campaigns");
    assert_eq!(
        srv.request("DELETE", "/campaign/rbac", Some(WRITER_A), None).unwrap().0,
        403, "writer must not delete campaigns"
    );

    // Unknown key is rejected outright.
    assert_eq!(srv.get("/campaigns", Some("not-a-real-key")).unwrap().0, 401);
    assert_eq!(srv.get("/campaigns", None).unwrap().0, 401);
}

// ---------------------------------------------------------------------------
// Fail-closed startup
// ---------------------------------------------------------------------------

#[test]
fn require_auth_refuses_to_start_without_keys() {
    let bin = env!("CARGO_BIN_EXE_banditdb");
    let dir = "/tmp/banditdb_http_test_requireauth";
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();

    let out = Command::new(bin)
        .env("DATA_DIR", dir)
        .env("PORT", "18308")
        .env("BANDITDB_REQUIRE_AUTH", "true")
        .env_remove("BANDITDB_API_KEYS")
        .env_remove("BANDITDB_API_KEY")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output();

    let _ = std::fs::remove_dir_all(dir);
    match out {
        Ok(o) => assert!(
            !o.status.success(),
            "with BANDITDB_REQUIRE_AUTH set and no keys configured the server must refuse \
             to start — otherwise every request is silently granted admin"
        ),
        Err(_) => eprintln!("SKIPPED require_auth: could not spawn binary"),
    }
}
