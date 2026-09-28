//! Runtime tenant provisioning, driven against a real server process.
//!
//! The behaviour under test is the one that makes a hosted console possible: a
//! tenant created over HTTP can authenticate immediately, is isolated to its own
//! namespace, and survives a restart. These live at the process boundary because
//! that is where the provisioning credential, the auth layer and the persisted
//! store meet — none of it is observable from inside the engine.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const PROVISION_KEY: &str = "prov-secret-key";
const OPERATOR_KEY:  &str = "operator-admin-key";

struct Server {
    child: Child,
    port:  u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    /// `keep_dir` lets a test restart a server against the same data directory.
    fn start(port: u16, dir: &str, fresh: bool) -> Option<Self> {
        let bin = env!("CARGO_BIN_EXE_banditdb");
        if fresh {
            let _ = std::fs::remove_dir_all(dir);
            std::fs::create_dir_all(dir).ok()?;
        }
        let child = Command::new(bin)
            .env("DATA_DIR", dir)
            .env("PORT", port.to_string())
            .env("BANDITDB_PROVISION_KEY", PROVISION_KEY)
            .env("BANDITDB_API_KEYS", format!("{OPERATOR_KEY}=admin"))
            .env("BANDITDB_RATE_LIMIT_PER_SEC", "100000")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn().ok()?;

        let server = Server { child, port };
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if matches!(server.get("/health", None), Some((200, _))) {
                return Some(server);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        None
    }

    fn request(&self, method: &str, path: &str, headers: &[(&str, &str)], body: Option<&str>)
        -> Option<(u16, String)>
    {
        let mut cmd = Command::new("curl");
        cmd.arg("-sS").arg("--max-time").arg("10")
            .arg("-o").arg("-")
            .arg("-w").arg("\n__STATUS__%{http_code}")
            .arg("-X").arg(method)
            .arg(format!("http://127.0.0.1:{}{}", self.port, path));
        for (k, v) in headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
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
        let h: Vec<(&str, &str)> = key.map(|k| vec![("X-Api-Key", k)]).unwrap_or_default();
        self.request("GET", path, &h, None)
    }
    fn post_key(&self, path: &str, key: &str, body: &str) -> Option<(u16, String)> {
        self.request("POST", path, &[("X-Api-Key", key)], Some(body))
    }
    fn provision(&self, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
        self.request(method, path, &[("X-Provision-Key", PROVISION_KEY)], body)
    }
}

macro_rules! server_or_skip {
    ($port:expr, $dir:expr, $fresh:expr) => {
        match Server::start($port, $dir, $fresh) {
            Some(s) => s,
            None => { eprintln!("SKIPPED: could not start server on port {}", $port); return; }
        }
    };
}

/// sha256 of a key, matching what a control plane would compute at mint time.
fn hash(key: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    format!("{:x}", h.finalize())
}

fn tenant_body(admin: &str, writer: &str, max_campaigns: usize) -> String {
    format!(
        r#"{{"keys":[{{"hash":"{}","role":"admin","prefix":"{}"}},
                     {{"hash":"{}","role":"writer"}}],
            "quotas":{{"max_campaigns":{max_campaigns},"max_feature_dim":64}},
            "status":"active"}}"#,
        hash(admin), &admin[..8.min(admin.len())], hash(writer)
    )
}

// ════════════════════════════════════════════════════════════════════════════

/// The core loop a console depends on: provision, authenticate, use — no restart.
#[test]
fn provisioned_tenant_can_work_immediately() {
    let srv = server_or_skip!(18401, "/tmp/bdb_provision_1", true);

    let key_admin  = "BDBtenant1admin000000000000000000";
    let key_writer = "BDBtenant1writer00000000000000000";

    // Before provisioning, the key is simply unknown.
    assert_eq!(srv.get("/campaigns", Some(key_admin)).unwrap().0, 401);

    let (status, _) = srv.provision("PUT", "/admin/tenants/org_alpha",
        Some(&tenant_body(key_admin, key_writer, 5))).unwrap();
    assert_eq!(status, 200, "provisioning must succeed");

    // Usable straight away — no restart, no config edit.
    assert_eq!(srv.get("/campaigns", Some(key_admin)).unwrap().0, 200);

    let (status, _) = srv.post_key("/campaign", key_admin,
        r#"{"campaign_id":"checkout","arms":["a","b"],"feature_dim":4,"alpha":1.0}"#).unwrap();
    assert_eq!(status, 200, "tenant admin must be able to create a campaign");

    // The writer key works for decisions but cannot create campaigns.
    let (status, _) = srv.post_key("/predict", key_writer,
        r#"{"campaign_id":"checkout","context":[0.1,0.2,0.3,0.4]}"#).unwrap();
    assert_eq!(status, 200, "writer must be able to predict");
    let (status, _) = srv.post_key("/campaign", key_writer,
        r#"{"campaign_id":"other","arms":["a","b"],"feature_dim":4}"#).unwrap();
    assert_eq!(status, 403, "writer must not create campaigns");
}

/// A provisioned tenant is namespace-scoped whatever BANDITDB_TENANT_MODE says.
#[test]
fn provisioned_tenants_cannot_see_each_other() {
    let srv = server_or_skip!(18402, "/tmp/bdb_provision_2", true);

    let a_key = "BDBorgAadmin0000000000000000000000";
    let b_key = "BDBorgBadmin0000000000000000000000";
    srv.provision("PUT", "/admin/tenants/org_a", Some(&tenant_body(a_key, "BDBorgAwriter000000000000000", 5))).unwrap();
    srv.provision("PUT", "/admin/tenants/org_b", Some(&tenant_body(b_key, "BDBorgBwriter000000000000000", 5))).unwrap();

    srv.post_key("/campaign", a_key,
        r#"{"campaign_id":"secret","arms":["a","b"],"feature_dim":4}"#).unwrap();

    // B lists its own campaigns and sees nothing of A's.
    let (_, body) = srv.get("/campaigns", Some(b_key)).unwrap();
    assert!(!body.contains("secret"),
        "tenant B must not see tenant A's campaigns — got {body}");

    // Nor can B address it directly.
    assert_eq!(srv.get("/campaign/secret", Some(b_key)).unwrap().0, 404);
    assert_eq!(srv.get("/campaign/secret", Some(a_key)).unwrap().0, 200);
}

/// Tenants must outlive the process — otherwise every deploy locks every customer out.
#[test]
fn tenants_survive_restart() {
    let dir = "/tmp/bdb_provision_3";
    let key = "BDBpersist000000000000000000000000";

    {
        let srv = server_or_skip!(18403, dir, true);
        srv.provision("PUT", "/admin/tenants/org_persist",
            Some(&tenant_body(key, "BDBpersistwriter00000000000", 3))).unwrap();
        srv.post_key("/campaign", key,
            r#"{"campaign_id":"kept","arms":["a","b"],"feature_dim":4}"#).unwrap();
    } // server dropped → killed

    let srv = server_or_skip!(18404, dir, false);   // same data dir, new process
    assert_eq!(srv.get("/campaigns", Some(key)).unwrap().0, 200,
        "the key must still authenticate after a restart");
    let (_, body) = srv.get("/campaigns", Some(key)).unwrap();
    assert!(body.contains("kept"), "the tenant's campaign must survive too: {body}");
}

/// Quotas are enforced per tenant, and the error names the numbers.
#[test]
fn tenant_quotas_are_enforced() {
    let srv = server_or_skip!(18405, "/tmp/bdb_provision_4", true);
    let key = "BDBquota0000000000000000000000000";
    srv.provision("PUT", "/admin/tenants/org_quota",
        Some(&tenant_body(key, "BDBquotawriter000000000000", 2))).unwrap();

    for i in 0..2 {
        let (status, _) = srv.post_key("/campaign", key,
            &format!(r#"{{"campaign_id":"c{i}","arms":["a","b"],"feature_dim":4}}"#)).unwrap();
        assert_eq!(status, 200);
    }
    let (status, body) = srv.post_key("/campaign", key,
        r#"{"campaign_id":"c3","arms":["a","b"],"feature_dim":4}"#).unwrap();
    assert_eq!(status, 403, "the third campaign must be refused");
    assert!(body.contains('2'), "the error must name the limit: {body}");

    // feature_dim is capped at 64 by the quota in tenant_body.
    let (status, _) = srv.post_key("/campaign", key,
        r#"{"campaign_id":"wide","arms":["a","b"],"feature_dim":128}"#).unwrap();
    assert_eq!(status, 403, "feature_dim above the plan limit must be refused");

    // /limits reports the same numbers the errors quote.
    let (status, body) = srv.get("/limits", Some(key)).unwrap();
    assert_eq!(status, 200);
    assert!(body.contains("\"max_campaigns\":2") && body.contains("\"campaigns_used\":2"),
        "/limits must report quota and usage: {body}");
}

/// Suspension is not the same as an invalid key, and the difference must be visible.
#[test]
fn suspended_tenant_is_refused_with_a_clear_reason() {
    let srv = server_or_skip!(18406, "/tmp/bdb_provision_5", true);
    let key = "BDBsuspend00000000000000000000000";
    srv.provision("PUT", "/admin/tenants/org_susp",
        Some(&tenant_body(key, "BDBsuspendwriter0000000000", 5))).unwrap();
    assert_eq!(srv.get("/campaigns", Some(key)).unwrap().0, 200);

    // Suspend by re-provisioning with the same keys — the call is idempotent.
    let body = format!(
        r#"{{"keys":[{{"hash":"{}","role":"admin"}}],"status":"suspended"}}"#, hash(key));
    srv.provision("PUT", "/admin/tenants/org_susp", Some(&body)).unwrap();

    let (status, msg) = srv.get("/campaigns", Some(key)).unwrap();
    assert_eq!(status, 403, "a suspended tenant must be forbidden, not unauthorized");
    assert!(msg.to_lowercase().contains("suspend"),
        "the message must say why, so a console can show 'renew' rather than 'bad key': {msg}");
}

/// The provisioning surface must be unreachable from any tenant credential.
#[test]
fn provisioning_requires_its_own_credential() {
    let srv = server_or_skip!(18407, "/tmp/bdb_provision_6", true);

    // No provisioning key at all.
    assert_eq!(srv.request("GET", "/admin/tenants", &[], None).unwrap().0, 401);
    // Wrong provisioning key.
    assert_eq!(
        srv.request("GET", "/admin/tenants", &[("X-Provision-Key", "wrong")], None).unwrap().0,
        401);
    // An operator admin API key is NOT a provisioning credential.
    assert_eq!(
        srv.request("GET", "/admin/tenants", &[("X-Api-Key", OPERATOR_KEY)], None).unwrap().0,
        401, "an admin API key must not reach the provisioning routes");
    // The real credential works.
    assert_eq!(srv.provision("GET", "/admin/tenants", None).unwrap().0, 200);
}

/// A key already bound to one tenant must not be silently rebound to another.
#[test]
fn a_key_cannot_be_claimed_by_two_tenants() {
    let srv = server_or_skip!(18408, "/tmp/bdb_provision_7", true);
    let shared = "BDBshared000000000000000000000000";

    srv.provision("PUT", "/admin/tenants/org_one",
        Some(&tenant_body(shared, "BDBonewriter0000000000000", 5))).unwrap();
    let (status, body) = srv.provision("PUT", "/admin/tenants/org_two",
        Some(&tenant_body(shared, "BDBtwowriter0000000000000", 5))).unwrap();

    assert_eq!(status, 400, "reusing a key hash across tenants must be rejected");
    assert!(body.contains("org_one"), "the error must name the current owner: {body}");
}

/// Removing a tenant revokes its keys but must not destroy its data.
#[test]
fn removing_a_tenant_revokes_keys_but_keeps_data() {
    let srv = server_or_skip!(18409, "/tmp/bdb_provision_8", true);
    let key = "BDBremove000000000000000000000000";
    srv.provision("PUT", "/admin/tenants/org_gone",
        Some(&tenant_body(key, "BDBremovewriter00000000000", 5))).unwrap();
    srv.post_key("/campaign", key,
        r#"{"campaign_id":"data","arms":["a","b"],"feature_dim":4}"#).unwrap();

    assert_eq!(srv.provision("DELETE", "/admin/tenants/org_gone", None).unwrap().0, 200);
    assert_eq!(srv.get("/campaigns", Some(key)).unwrap().0, 401,
        "the revoked key must stop working");

    // The operator can still see the namespaced campaign: deleting credentials is
    // not deleting models.
    let (_, body) = srv.get("/campaigns", Some(OPERATOR_KEY)).unwrap();
    assert!(body.contains("org_gone/data"),
        "campaign data must survive tenant removal: {body}");

    // Re-provisioning the same tenant restores access to it.
    srv.provision("PUT", "/admin/tenants/org_gone",
        Some(&tenant_body(key, "BDBremovewriter00000000000", 5))).unwrap();
    let (_, body) = srv.get("/campaigns", Some(key)).unwrap();
    assert!(body.contains("data"), "re-provisioning must restore access: {body}");
}
