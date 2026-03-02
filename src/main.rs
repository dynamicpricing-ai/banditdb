use axum::{
    extract::{Path, State, Json},
    http::StatusCode,
    middleware::{self, Next},
    extract::Request,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Router,
};
use banditdb::BanditDB;
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::net::TcpListener;

// --- Unified error type ---
// Every handler returns AppError on failure, which serialises to {"error": "..."}
// instead of a bare string. Keeps all error responses structurally identical.
struct AppError(StatusCode, String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct Body { error: String }
        (self.0, Json(Body { error: self.1 })).into_response()
    }
}

// --- Request / response structs ---
#[derive(Deserialize)]
struct CreateCampaignRequest {
    campaign_id: String,
    arms: Vec<String>,
    feature_dim: usize,
}

#[derive(Deserialize)]
struct PredictRequest {
    campaign_id: String,
    context: Vec<f64>,
}

#[derive(Serialize)]
struct PredictResponse {
    arm_id: String,
    interaction_id: String,
}

#[derive(Deserialize)]
struct RewardRequest {
    interaction_id: String,
    reward: f64,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

// --- Route handlers ---
async fn handle_health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn handle_create_campaign(
    State(db): State<Arc<BanditDB>>,
    Json(payload): Json<CreateCampaignRequest>,
) -> Json<&'static str> {
    db.add_campaign(&payload.campaign_id, payload.arms, payload.feature_dim);
    Json("Campaign Created")
}

async fn handle_delete_campaign(
    State(db): State<Arc<BanditDB>>,
    Path(campaign_id): Path<String>,
) -> Result<Json<&'static str>, AppError> {
    match db.delete_campaign(&campaign_id) {
        true  => Ok(Json("Campaign Deleted")),
        false => Err(AppError(
            StatusCode::NOT_FOUND,
            format!("Campaign '{}' not found", campaign_id),
        )),
    }
}

async fn handle_predict(
    State(db): State<Arc<BanditDB>>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, AppError> {
    db.predict(&payload.campaign_id, payload.context)
        .map(|(arm_id, interaction_id)| Json(PredictResponse { arm_id, interaction_id }))
        .ok_or_else(|| AppError(
            StatusCode::NOT_FOUND,
            format!("Campaign '{}' not found", payload.campaign_id),
        ))
}

async fn handle_reward(
    State(db): State<Arc<BanditDB>>,
    Json(payload): Json<RewardRequest>,
) -> Json<&'static str> {
    db.reward(&payload.interaction_id, payload.reward);
    Json("OK")
}

async fn handle_checkpoint(State(db): State<Arc<BanditDB>>) -> Result<Json<String>, AppError> {
    db.checkpoint()
        .await
        .map(Json)
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e))
}

async fn handle_export(State(db): State<Arc<BanditDB>>) -> Result<Json<String>, AppError> {
    let export_dir = db.export_dir();
    let entries = fs::read_dir(&export_dir)
        .map_err(|e| AppError(StatusCode::NOT_FOUND, format!("No exports yet: {}", e)))?;
    let files: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.ends_with(".parquet"))
        .collect();
    Ok(Json(format!("Parquet files in {}: {:?}", export_dir, files)))
}

// --- Main server boot ---
#[tokio::main]
async fn main() {
    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| ".".to_string());
    let wal_path = format!("{}/bandit_wal.jsonl", data_dir);
    println!("📂 Using Data Directory: {}", data_dir);

    // #5 — Configurable reward TTL (read inside BanditDB::new via env var)
    let db = Arc::new(BanditDB::new(&wal_path, &data_dir));

    // #1 — API key authentication
    // Set BANDITDB_API_KEY to enable. If unset, the server runs open (dev mode).
    let api_key = std::env::var("BANDITDB_API_KEY").ok();
    match &api_key {
        Some(k) => println!("🔒 Authentication enabled (key: {}...)", &k[..k.len().min(4)]),
        None    => println!("⚠️  BANDITDB_API_KEY not set — running without authentication"),
    }

    // Protected routes — all require the API key when one is configured
    let protected = Router::new()
        .route("/campaign",     post(handle_create_campaign))
        .route("/campaign/:id", delete(handle_delete_campaign))
        .route("/predict",      post(handle_predict))
        .route("/reward",       post(handle_reward))
        .route("/checkpoint",   post(handle_checkpoint))
        .route("/export",       get(handle_export))
        .layer(middleware::from_fn(move |req: Request, next: Next| {
            let key = api_key.clone();
            async move {
                if let Some(expected) = key {
                    let provided = req
                        .headers()
                        .get("X-Api-Key")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("");
                    if provided != expected {
                        return AppError(StatusCode::UNAUTHORIZED, "Unauthorized".to_string())
                            .into_response();
                    }
                }
                next.run(req).await
            }
        }));

    // #7 — /health is always public — load balancers and probes must reach it
    let app = Router::new()
        .route("/health", get(handle_health))
        .merge(protected)
        .with_state(Arc::clone(&db));

    // Auto-checkpoint: optional background task triggered by rewarded event count.
    // Clone before with_state() consumes db.
    let checkpoint_interval: Option<u64> = std::env::var("BANDITDB_CHECKPOINT_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0);

    match checkpoint_interval {
        Some(n) => println!("⏱️  Auto-checkpoint enabled every {} rewarded events", n),
        None    => println!("💡 Auto-checkpoint disabled (set BANDITDB_CHECKPOINT_INTERVAL to enable)"),
    }

    let max_wal_bytes: Option<u64> = std::env::var("BANDITDB_MAX_WAL_SIZE_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .map(|mb| mb * 1024 * 1024);

    match max_wal_bytes {
        Some(b) => println!("💾 WAL size limit: {} MB", b / 1024 / 1024),
        None    => println!("💾 WAL size limit: unlimited"),
    }

    if checkpoint_interval.is_some() || max_wal_bytes.is_some() {
        let db_bg = Arc::clone(&db);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

                let count_exceeded = checkpoint_interval
                    .map_or(false, |n| db_bg.rewarded_count.load(Ordering::Relaxed) >= n);

                let size_exceeded = max_wal_bytes.map_or(false, |max| {
                    fs::metadata(&db_bg.wal_path)
                        .map(|m| m.len() > max)
                        .unwrap_or(false)
                });

                if count_exceeded || size_exceeded {
                    db_bg.rewarded_count.store(0, Ordering::Relaxed);
                    if let Err(e) = db_bg.checkpoint().await {
                        println!("[checkpoint] Auto-checkpoint failed: {}", e);
                    }
                }
            }
        });
    }

    let listener = TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("🚀 BanditDB running on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}
