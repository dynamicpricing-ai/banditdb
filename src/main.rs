use axum::{
    extract::{State, Json},
    routing::post,
    Router,
};
use axum::routing::get;
use banditdb::BanditDB;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;

// --- JSON Payloads ---
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

// --- Route Handlers ---
async fn handle_create_campaign(
    State(db): State<Arc<BanditDB>>,
    Json(payload): Json<CreateCampaignRequest>,
) -> Json<&'static str> {
    db.add_campaign(&payload.campaign_id, payload.arms, payload.feature_dim);
    Json("Campaign Created")
}

async fn handle_predict(
    State(db): State<Arc<BanditDB>>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, String> {
    match db.predict(&payload.campaign_id, payload.context) {
        Some((arm_id, interaction_id)) => Ok(Json(PredictResponse { arm_id, interaction_id })),
        None => Err("Campaign not found".to_string()),
    }
}

async fn handle_reward(
    State(db): State<Arc<BanditDB>>,
    Json(payload): Json<RewardRequest>,
) -> Json<&'static str> {
    db.reward(&payload.interaction_id, payload.reward);
    Json("OK")
}

async fn handle_export(State(db): State<Arc<BanditDB>>) -> Result<Json<String>, String> {
    // In production, we'd timestamp this file (e.g., bandit_logs_2026-02-26.parquet)
    let output_file = "bandit_logs_latest.parquet";
    
    match db.export_to_parquet("bandit_wal.jsonl", output_file) {
        Ok(msg) => Ok(Json(msg)),
        Err(e) => Err(e),
    }
}

// --- Main Server Boot ---
#[tokio::main]
async fn main() {
    // Read the DATA_DIR environment variable (Default to current directory if not set)
    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| ".".to_string());
    
    let wal_path = format!("{}/bandit_wal.jsonl", data_dir);
    println!("📂 Using Data Directory: {}", data_dir);

    // Initialize the Database Engine with the dynamic path
    let db = Arc::new(BanditDB::new(&wal_path));

    let app = Router::new()
        .route("/campaign", post(handle_create_campaign))
        .route("/predict", post(handle_predict))
        .route("/reward", post(handle_reward))
        .route("/export", get(handle_export))
        .with_state(db);

    let listener = TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("🚀 BanditDB running on http://0.0.0.0:8080");
    axum::serve(listener, app).await.unwrap();
}