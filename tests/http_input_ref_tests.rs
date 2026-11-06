//! `input_ref` over HTTP: accepted on `/predict`, `/batch_predict` and
//! `/campaign/:id/interact`, written to the WAL, and refused with a 400 when the
//! URI carries credentials.
//!
//! Drives the real server binary, like `http_rbac_tenant_tests`.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const KEY: &str = "key-admin";
const PORT: u16 = 18601;

struct Server {
    child: Child,
    dir: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Server {
    fn start() -> Self {
        let dir = format!("/tmp/banditdb_http_input_ref_{PORT}");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_banditdb"))
            .env("DATA_DIR", &dir)
            .env("PORT", PORT.to_string())
            .env("BANDITDB_API_KEYS", format!("{KEY}=admin"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let server = Server { child, dir };
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if server.request("GET", "/health", None).0 == 200 {
                return server;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("server did not start");
    }

    fn request(&self, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
        let mut cmd = Command::new("curl");
        cmd.args([
            "-sS",
            "--max-time",
            "10",
            "-o",
            "-",
            "-w",
            "\n__STATUS__%{http_code}",
        ])
        .args(["-X", method, &format!("http://127.0.0.1:{PORT}{path}")])
        .args(["-H", &format!("X-Api-Key: {KEY}")]);
        if let Some(b) = body {
            cmd.args(["-H", "Content-Type: application/json", "-d", b]);
        }
        let out = cmd.output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let (body, status) = text.rsplit_once("__STATUS__").unwrap_or(("", "0"));
        (
            status.trim().parse().unwrap_or(0),
            body.trim_end().to_string(),
        )
    }

    fn post(&self, path: &str, body: &str) -> (u16, String) {
        self.request("POST", path, Some(body))
    }

    fn wal(&self) -> String {
        std::fs::read_to_string(format!("{}/bandit_wal.jsonl", self.dir)).unwrap()
    }
}

fn json(body: &str) -> serde_json::Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

#[test]
fn input_ref_over_http() {
    if Command::new("curl").arg("--version").output().is_err() {
        eprintln!("curl not available; skipping");
        return;
    }
    let s = Server::start();
    let (st, body) = s.post(
        "/campaign",
        r#"{"campaign_id":"c","arms":["A","B"],"feature_dim":2}"#,
    );
    assert_eq!(st, 200, "{body}");

    // /predict accepts it.
    let (st, body) = s.post(
        "/predict",
        r#"{"campaign_id":"c","context":[0.5,0.5],
            "input_ref":{"uri":"postgres://crm/tickets/4412","sha256":"ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB","media_type":"application/json"}}"#,
    );
    assert_eq!(st, 200, "{body}");
    let iid = json(&body)["interaction_id"].as_str().unwrap().to_string();

    // A credential-bearing URI is a 400 naming the problem.
    let (st, body) = s.post(
        "/predict",
        r#"{"campaign_id":"c","context":[0.5,0.5],
            "input_ref":{"uri":"https://b.s3.amazonaws.com/k?X-Amz-Signature=abc"}}"#,
    );
    assert_eq!(st, 400, "{body}");
    assert!(body.contains("x-amz-signature"), "{body}");

    // A missing uri is rejected by the request parser.
    let (st, _) = s.post(
        "/predict",
        r#"{"campaign_id":"c","context":[0.5,0.5],"input_ref":{"sha256":"ab"}}"#,
    );
    assert!(
        st == 400 || st == 422,
        "missing uri must be refused, got {st}"
    );

    // /batch_predict: per-item, so one bad ref fails only its own item.
    let (st, body) = s.post(
        "/batch_predict",
        r#"{"predictions":[
            {"campaign_id":"c","context":[0.5,0.5],"input_ref":{"uri":"s3://b/ok.json"}},
            {"campaign_id":"c","context":[0.5,0.5],"input_ref":{"uri":"mongodb://u:p@h/db/c/1"}}
        ]}"#,
    );
    assert_eq!(st, 200, "{body}");
    let items = json(&body);
    assert!(items[0]["interaction_id"].is_string(), "{body}");
    assert!(
        items[1]["error"].as_str().unwrap().contains("credentials"),
        "{body}"
    );

    // /campaign/:id/interact accepts it.
    let (st, body) = s.post(
        "/campaign/c/interact",
        r#"{"arm_id":"B","context":[0.5,0.5],"reward":1.0,
            "input_ref":{"uri":"s3://b/history/7.json"}}"#,
    );
    assert_eq!(st, 200, "{body}");

    // A reward is fsynced, and so is everything logged before it.
    let (st, body) = s.post(
        "/reward",
        &format!(r#"{{"interaction_id":"{iid}","reward":1.0}}"#),
    );
    assert_eq!(st, 200, "{body}");

    let wal = s.wal();
    assert!(wal.contains(&format!(
        r#""input_ref":{{"uri":"postgres://crm/tickets/4412","sha256":"{}","media_type":"application/json"}}"#,
        "ab".repeat(32)
    )));
    assert!(wal.contains(r#""uri":"s3://b/ok.json""#));
    assert!(wal.contains(r#""uri":"s3://b/history/7.json""#));
    assert!(!wal.contains("X-Amz-Signature") && !wal.contains("u:p@"));
}
