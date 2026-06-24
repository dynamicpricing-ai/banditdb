use axum::{
    extract::{ConnectInfo, DefaultBodyLimit, Extension, Path, State, Json},
    http::{HeaderMap, StatusCode, Method},
    middleware::{self, Next},
    extract::Request,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Router,
};
use tower_http::cors::{CorsLayer, Any};
use banditdb::state::{Algorithm, CampaignReport, EngineError, EntropyStatus, DEFAULT_ALPHA};
use banditdb::BanditDB;
use governor::{Quota, RateLimiter, state::keyed::DefaultKeyedStateStore, clock::DefaultClock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use subtle::ConstantTimeEq;
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// RBAC + multi-tenancy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role { Reader = 1, Writer = 2, Admin = 3 }

/// Per-request auth context — injected by `auth_middleware`, read by handlers.
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub role:      Role,
    /// None = global access (admin with no tenant restriction).
    /// Some(t) = restricted to the `t/` namespace when tenant mode is active.
    pub tenant_id: Option<String>,
}

/// Registry of API keys, roles, and optional tenant scopes.
///
/// BANDITDB_API_KEYS format:
///   "key1=admin;key2=writer;key3=reader"                    (no tenants)
///   "key1=admin;key2=admin:tenant_a;key3=writer:tenant_b"   (with tenants)
///
/// BANDITDB_API_KEY (legacy) — treated as an admin key with no tenant restriction.
pub struct KeyRegistry {
    keys:        Vec<(Vec<u8>, Role, Option<String>)>,
    tenant_mode: bool,
}

impl KeyRegistry {
    pub fn from_env() -> Self {
        let tenant_mode = std::env::var("BANDITDB_TENANT_MODE")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        let mut keys: Vec<(Vec<u8>, Role, Option<String>)> = Vec::new();

        if let Ok(raw) = std::env::var("BANDITDB_API_KEYS") {
            for pair in raw.split(';') {
                if let Some((k, spec)) = pair.split_once('=') {
                    // spec = "role" or "role:tenant_id"
                    let (role_str, tenant_id) = match spec.split_once(':') {
                        Some((r, t)) => (r.trim(), Some(t.trim().to_string())),
                        None         => (spec.trim(), None),
                    };
                    let role = match role_str {
                        "admin"  => Role::Admin,
                        "writer" => Role::Writer,
                        "reader" => Role::Reader,
                        _        => continue,
                    };
                    if !k.trim().is_empty() {
                        keys.push((k.trim().as_bytes().to_vec(), role, tenant_id));
                    }
                }
            }
        }

        // Backward compat: BANDITDB_API_KEY = admin, no tenant restriction.
        if let Ok(k) = std::env::var("BANDITDB_API_KEY") {
            if !k.is_empty() && !keys.iter().any(|(b, _, _)| b == k.as_bytes()) {
                keys.push((k.into_bytes(), Role::Admin, None));
            }
        }

        Self { keys, tenant_mode }
    }

    /// Authenticate in constant time — no branch on comparison result position.
    pub fn authenticate(&self, provided: &str) -> Option<AuthContext> {
        if self.keys.is_empty() {
            return Some(AuthContext { role: Role::Admin, tenant_id: None });
        }
        let provided = provided.as_bytes();
        let mut found:     u8              = 0u8;
        let mut role_val:  u8              = 0u8;
        let mut tenant_id: Option<String>  = None;

        for (key_bytes, role, tid) in &self.keys {
            let matched = provided.ct_eq(key_bytes.as_slice()).unwrap_u8();
            let update  = matched & (!found & 1);
            role_val   |= update.wrapping_neg() & (*role as u8);
            if matched == 1 && found == 0 {
                tenant_id = tid.clone();
            }
            found |= matched;
        }

        if found == 1 {
            let role = match role_val {
                3 => Role::Admin,
                2 => Role::Writer,
                _ => Role::Reader,
            };
            Some(AuthContext { role, tenant_id: if self.tenant_mode { tenant_id } else { None } })
        } else {
            None
        }
    }

    pub fn is_open(&self) -> bool { self.keys.is_empty() }
    pub fn key_count(&self) -> usize { self.keys.len() }
}

// ---------------------------------------------------------------------------
// Namespace helpers
// ---------------------------------------------------------------------------

/// Prepend tenant prefix when tenant mode is active.
fn ns(auth: &AuthContext, campaign_id: &str) -> String {
    match &auth.tenant_id {
        Some(t) => format!("{t}/{campaign_id}"),
        None    => campaign_id.to_string(),
    }
}

/// Strip tenant prefix from a stored campaign ID for API responses.
fn strip_ns(auth: &AuthContext, stored_id: &str) -> String {
    match &auth.tenant_id {
        Some(t) => stored_id.strip_prefix(&format!("{t}/")).unwrap_or(stored_id).to_string(),
        None    => stored_id.to_string(),
    }
}

/// Check whether a stored campaign ID belongs to this tenant.
fn owns(auth: &AuthContext, stored_id: &str) -> bool {
    match &auth.tenant_id {
        Some(t) => stored_id.starts_with(&format!("{t}/")),
        None    => true,
    }
}

// ---------------------------------------------------------------------------
// Rate limiter
// ---------------------------------------------------------------------------

type ApiKeyLimiter = RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock>;

// ---------------------------------------------------------------------------
// HTTP metrics
// ---------------------------------------------------------------------------

const LATENCY_BOUNDS: [f64; 10] = [0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0];

const EP_PREDICT:  usize = 0;
const EP_BATCH:    usize = 1;
const EP_REWARD:   usize = 2;
const EP_CAMPAIGN: usize = 3;
const EP_OTHER:    usize = 4;
const N_EP: usize = 5;
const EP_NAMES: [&str; N_EP] = ["predict", "batch_predict", "reward", "campaign", "other"];

pub struct EndpointMetrics {
    pub req_2xx:      AtomicU64,
    pub req_4xx:      AtomicU64,
    pub req_5xx:      AtomicU64,
    pub lat_bucket:   [AtomicU64; 10],
    pub lat_sum_bits: AtomicU64,
    pub lat_count:    AtomicU64,
}

impl Default for EndpointMetrics {
    fn default() -> Self {
        Self {
            req_2xx:      AtomicU64::new(0),
            req_4xx:      AtomicU64::new(0),
            req_5xx:      AtomicU64::new(0),
            lat_bucket:   std::array::from_fn(|_| AtomicU64::new(0)),
            lat_sum_bits: AtomicU64::new(0),
            lat_count:    AtomicU64::new(0),
        }
    }
}

pub struct HttpMetrics {
    pub by_endpoint: [EndpointMetrics; N_EP],
}

impl Default for HttpMetrics {
    fn default() -> Self {
        Self { by_endpoint: std::array::from_fn(|_| EndpointMetrics::default()) }
    }
}

fn classify_endpoint(path: &str) -> usize {
    match path {
        "/predict"       => EP_PREDICT,
        "/batch_predict" => EP_BATCH,
        "/reward"        => EP_REWARD,
        _ if path.starts_with("/campaign") => EP_CAMPAIGN,
        _ => EP_OTHER,
    }
}

fn atomic_add_f64(atom: &AtomicU64, val: f64) {
    loop {
        let old = atom.load(Ordering::Relaxed);
        let new = (f64::from_bits(old) + val).to_bits();
        if atom.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

pub struct AppState {
    pub db:             Arc<BanditDB>,
    pub registry:       Arc<KeyRegistry>,
    pub rate_limiter:   Option<Arc<ApiKeyLimiter>>,
    pub metrics_public: bool,
    pub http_metrics:   HttpMetrics,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

struct AppError(StatusCode, String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct Body { error: String }
        (self.0, Json(Body { error: self.1 })).into_response()
    }
}

fn map_engine_err(e: EngineError) -> AppError {
    let status = match &e {
        EngineError::NotFound(_)      => StatusCode::NOT_FOUND,
        EngineError::AlreadyExists(_) => StatusCode::CONFLICT,
        EngineError::Archived(_)      => StatusCode::NOT_FOUND,
        EngineError::WalFull          => StatusCode::SERVICE_UNAVAILABLE,
        EngineError::WalUnavailable   => StatusCode::SERVICE_UNAVAILABLE,
        EngineError::BadRequest(_)    => StatusCode::BAD_REQUEST,
        EngineError::Internal(_)      => StatusCode::INTERNAL_SERVER_ERROR,
    };
    AppError(status, e.to_string())
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

fn validate_id(s: &str, field: &str) -> Result<(), AppError> {
    if s.is_empty() || s.len() > 128 {
        return Err(AppError(StatusCode::BAD_REQUEST,
            format!("{field} must be 1–128 characters")));
    }
    if !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(AppError(StatusCode::BAD_REQUEST,
            format!("{field} may only contain ASCII letters, digits, '-', and '_'")));
    }
    Ok(())
}

fn prom_label(s: &str) -> String {
    s.chars().filter(|&c| c != '"' && c != '\n' && c != '\\').collect()
}

// ---------------------------------------------------------------------------
// Request / response structs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CreateCampaignRequest {
    campaign_id: String,
    arms:        Vec<String>,
    feature_dim: usize,
    #[serde(default = "default_alpha")]
    alpha:       f64,
    #[serde(default)]
    algorithm:   Algorithm,
    metadata:             Option<serde_json::Value>,
    decay_half_life_hours: Option<f64>,
}

fn default_alpha() -> f64 { DEFAULT_ALPHA }

#[derive(Deserialize)]
struct PredictRequest {
    campaign_id: String,
    context:     Vec<f64>,
}

#[derive(Serialize)]
struct PredictResponse { arm_id: String, interaction_id: String }

#[derive(Deserialize)]
struct BatchPredictItem { campaign_id: String, context: Vec<f64> }

#[derive(Deserialize)]
struct BatchPredictRequest { predictions: Vec<BatchPredictItem> }

#[derive(Serialize)]
struct BatchPredictResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    arm_id:         Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interaction_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error:          Option<String>,
}

#[derive(Deserialize)]
struct RewardRequest { interaction_id: String, reward: f64 }

#[derive(Deserialize)]
struct InteractRequest {
    context: Vec<f64>,
    arm_id:  String,
    reward:  f64,
}

async fn handle_interact(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(campaign_id): Path<String>,
    Json(payload): Json<InteractRequest>,
) -> Result<Json<PredictResponse>, AppError> {
    validate_id(&campaign_id, "campaign_id")?;
    validate_id(&payload.arm_id, "arm_id")?;
    
    let interaction_id = state.db.interact(
        &ns(&auth, &campaign_id),
        &payload.arm_id,
        payload.context,
        payload.reward,
    )
    .map_err(map_engine_err)?;

    Ok(Json(PredictResponse { arm_id: payload.arm_id, interaction_id }))
}

#[derive(Serialize)]
struct CampaignEntropyHealth {
    entropy: f64,
    status:  EntropyStatus,
}

#[derive(Serialize)]
struct HealthResponse {
    status:    &'static str,
    campaigns: HashMap<String, CampaignEntropyHealth>,
}

#[derive(Serialize)]
struct CampaignSummary {
    campaign_id: String,
    alpha:       f64,
    algorithm:   Algorithm,
    arm_count:   usize,
    archived:    bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata:    Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ArmInfo {
    theta: Vec<f64>, theta_norm: f64,
    prediction_count: u64, reward_count: u64, avg_reward: Option<f64>,
}

#[derive(Serialize)]
struct CampaignInfo {
    campaign_id: String, alpha: f64, algorithm: Algorithm, archived: bool,
    total_predictions: u64, total_rewards: u64,
    arms: HashMap<String, ArmInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ExportResponse { export_dir: String, shards: HashMap<String, Vec<String>> }

// ---------------------------------------------------------------------------
// Auth + rate-limit middleware
// ---------------------------------------------------------------------------

async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    mut req: Request,
    next: Next,
) -> Response {
    let provided = req.headers().get("X-Api-Key")
        .and_then(|v| v.to_str().ok()).unwrap_or("");

    let auth = match state.registry.authenticate(provided) {
        Some(a) => a,
        None    => return AppError(StatusCode::UNAUTHORIZED, "Unauthorized".into()).into_response(),
    };

    if let Some(limiter) = &state.rate_limiter {
        let key = if provided.is_empty() { addr.ip().to_string() } else { provided.to_string() };
        if limiter.check_key(&key).is_err() {
            return AppError(StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded — retry after 1 second".into()).into_response();
        }
    }

    req.extensions_mut().insert(auth);
    next.run(req).await
}

async fn require_role(min: Role, Extension(auth): Extension<AuthContext>, req: Request, next: Next) -> Response {
    if auth.role >= min { next.run(req).await }
    else { AppError(StatusCode::FORBIDDEN, "Insufficient permissions".into()).into_response() }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async fn handle_health(State(state): State<Arc<AppState>>) -> (StatusCode, Json<HealthResponse>) {
    let wal_ok   = state.db.wal_healthy.load(Ordering::Relaxed);
    let statuses = state.db.entropy_status_all();
    let degraded = statuses.iter().any(|(_, _, s)| !matches!(s, EntropyStatus::Ok));
    let campaigns = statuses.into_iter()
        .map(|(id, entropy, status)| (id, CampaignEntropyHealth { entropy, status }))
        .collect();

    let (http_status, overall) = if !wal_ok {
        (StatusCode::SERVICE_UNAVAILABLE, "degraded: wal unavailable")
    } else if degraded {
        (StatusCode::OK, "degraded")
    } else {
        (StatusCode::OK, "ok")
    };
    (http_status, Json(HealthResponse { status: overall, campaigns }))
}

async fn handle_create_campaign(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(payload): Json<CreateCampaignRequest>,
) -> Result<Json<&'static str>, AppError> {
    validate_id(&payload.campaign_id, "campaign_id")?;
    if payload.arms.is_empty() {
        return Err(AppError(StatusCode::BAD_REQUEST, "Campaign must have at least one arm".into()));
    }
    if payload.arms.len() > state.db.max_arms {
        return Err(AppError(StatusCode::BAD_REQUEST,
            format!("arm count {} exceeds BANDITDB_MAX_ARMS={}", payload.arms.len(), state.db.max_arms)));
    }
    for arm in &payload.arms { validate_id(arm, "arm_id")?; }
    if let Some(meta) = &payload.metadata {
        if serde_json::to_string(meta).map(|s| s.len()).unwrap_or(0) > 64 * 1024 {
            return Err(AppError(StatusCode::BAD_REQUEST, "metadata exceeds 64 KB limit".into()));
        }
    }
    if let Some(hl) = payload.decay_half_life_hours {
        if hl <= 0.0 {
            return Err(AppError(StatusCode::BAD_REQUEST, "decay_half_life_hours must be > 0".into()));
        }
    }

    let arm_dim = match &payload.algorithm {
        Algorithm::NeuralLinUCB(cfg) | Algorithm::NeuralThompsonSampling(cfg) => cfg.embed_dim,
        _ => {
            if payload.feature_dim == 0 {
                return Err(AppError(StatusCode::BAD_REQUEST, "feature_dim must be > 0".into()));
            }
            payload.feature_dim
        }
    };
    if arm_dim > state.db.max_feature_dim {
        return Err(AppError(StatusCode::BAD_REQUEST,
            format!("feature_dim {arm_dim} exceeds BANDITDB_MAX_FEATURE_DIM={}", state.db.max_feature_dim)));
    }

    state.db.add_campaign(
        &ns(&auth, &payload.campaign_id),
        payload.arms, arm_dim, payload.alpha, payload.algorithm, payload.metadata,
        payload.decay_half_life_hours,
    )
    .map(|_| Json("Campaign Created"))
    .map_err(map_engine_err)
}

async fn handle_delete_campaign(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(campaign_id): Path<String>,
) -> Result<Json<&'static str>, AppError> {
    validate_id(&campaign_id, "campaign_id")?;
    state.db.delete_campaign(&ns(&auth, &campaign_id))
        .map(|_| Json("Campaign Deleted"))
        .map_err(map_engine_err)
}

async fn handle_archive_campaign(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(campaign_id): Path<String>,
) -> Result<Json<&'static str>, AppError> {
    validate_id(&campaign_id, "campaign_id")?;
    state.db.archive_campaign(&ns(&auth, &campaign_id))
        .map(|_| Json("Campaign Archived"))
        .map_err(map_engine_err)
}

async fn handle_restore_campaign(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(campaign_id): Path<String>,
) -> Result<Json<&'static str>, AppError> {
    validate_id(&campaign_id, "campaign_id")?;
    state.db.restore_campaign(&ns(&auth, &campaign_id))
        .map(|_| Json("Campaign Restored"))
        .map_err(map_engine_err)
}

async fn handle_predict(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, AppError> {
    if payload.context.len() > state.db.max_feature_dim {
        return Err(AppError(StatusCode::BAD_REQUEST,
            format!("context length {} exceeds BANDITDB_MAX_FEATURE_DIM={}", payload.context.len(), state.db.max_feature_dim)));
    }
    let db  = Arc::clone(&state.db);
    let cid = ns(&auth, &payload.campaign_id);
    tokio::task::spawn_blocking(move || db.predict(&cid, payload.context))
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, format!("scoring task failed: {e}")))?
        .map(|(arm_id, interaction_id)| Json(PredictResponse { arm_id, interaction_id }))
        .map_err(map_engine_err)
}

async fn handle_batch_predict(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(payload): Json<BatchPredictRequest>,
) -> Result<Json<Vec<BatchPredictResult>>, AppError> {
    const MAX_BATCH: usize = 100;
    if payload.predictions.len() > MAX_BATCH {
        return Err(AppError(StatusCode::BAD_REQUEST,
            format!("batch size {} exceeds maximum {MAX_BATCH}", payload.predictions.len())));
    }
    let db      = Arc::clone(&state.db);
    let max_dim = state.db.max_feature_dim;
    // Namespace campaign IDs upfront before moving into spawn_blocking.
    let namespaced: Vec<(String, Vec<f64>)> = payload.predictions.into_iter()
        .map(|item| (ns(&auth, &item.campaign_id), item.context))
        .collect();

    let results = tokio::task::spawn_blocking(move || {
        namespaced.into_iter().map(|(cid, context)| {
            if context.len() > max_dim {
                return BatchPredictResult { arm_id: None, interaction_id: None,
                    error: Some(format!("context length {} exceeds limit", context.len())) };
            }
            match db.predict(&cid, context) {
                Ok((arm_id, iid)) => BatchPredictResult { arm_id: Some(arm_id), interaction_id: Some(iid), error: None },
                Err(e)            => BatchPredictResult { arm_id: None, interaction_id: None, error: Some(e.to_string()) },
            }
        }).collect::<Vec<_>>()
    }).await.map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, format!("batch task failed: {e}")))?;

    Ok(Json(results))
}

async fn handle_reward(
    State(state): State<Arc<AppState>>,
    Extension(_auth): Extension<AuthContext>,
    Json(payload): Json<RewardRequest>,
) -> Result<Json<&'static str>, AppError> {
    if !(0.0..=1.0).contains(&payload.reward) {
        return Err(AppError(StatusCode::BAD_REQUEST,
            format!("reward {} is outside required range [0.0, 1.0]", payload.reward)));
    }
    let db = Arc::clone(&state.db);
    tokio::task::spawn_blocking(move || db.reward(&payload.interaction_id, payload.reward))
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, format!("reward task failed: {e}")))?
        .map(|_| Json("OK"))
        .map_err(map_engine_err)
}

async fn handle_list_campaigns(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Json<Vec<CampaignSummary>> {
    let campaigns = state.db.campaigns.read();
    let mut list: Vec<CampaignSummary> = campaigns
        .iter()
        .filter(|(id, _)| owns(&auth, id))
        .map(|(id, campaign)| CampaignSummary {
            campaign_id: strip_ns(&auth, id),
            alpha:       campaign.alpha,
            algorithm:   campaign.algorithm.clone(),
            arm_count:   campaign.arms.read().len(),
            archived:    campaign.archived.load(Ordering::Relaxed),
            metadata:    campaign.metadata.clone(),
        })
        .collect();
    list.sort_by(|a, b| a.campaign_id.cmp(&b.campaign_id));
    Json(list)
}

async fn handle_campaign_info(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(campaign_id): Path<String>,
) -> Result<Json<CampaignInfo>, AppError> {
    let stored_id = ns(&auth, &campaign_id);
    let campaigns  = state.db.campaigns.read();
    let campaign   = campaigns.get(&stored_id).ok_or_else(|| {
        AppError(StatusCode::NOT_FOUND, format!("Campaign '{campaign_id}' not found"))
    })?;

    let arms_guard = campaign.arms.read();
    let mut total_preds   = 0u64;
    let mut total_rewards = 0u64;
    let mut arms          = HashMap::new();

    for (arm_id, s) in arms_guard.iter() {
        let p  = s.prediction_count.load(Ordering::Relaxed);
        let r  = s.reward_count.load(Ordering::Relaxed);
        let tr = f64::from_bits(s.total_reward.load(Ordering::Relaxed));
        total_preds   += p;
        total_rewards += r;
        arms.insert(arm_id.clone(), ArmInfo {
            theta: s.theta.to_vec(), theta_norm: s.theta.dot(&s.theta).sqrt(),
            prediction_count: p, reward_count: r,
            avg_reward: if r > 0 { Some(tr / r as f64) } else { None },
        });
    }

    Ok(Json(CampaignInfo {
        campaign_id, // return caller-supplied name, not namespaced
        alpha:       campaign.alpha,
        algorithm:   campaign.algorithm.clone(),
        archived:    campaign.archived.load(Ordering::Relaxed),
        total_predictions: total_preds,
        total_rewards,
        arms,
        metadata: campaign.metadata.clone(),
    }))
}

async fn handle_campaign_report(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(campaign_id): Path<String>,
) -> Result<Json<CampaignReport>, AppError> {
    let stored_id = ns(&auth, &campaign_id);
    state.db.campaign_report(&stored_id)
        .map(|mut r| { r.campaign_id = campaign_id; Json(r) }) // strip namespace from response
        .map_err(map_engine_err)
}

async fn handle_campaign_diagnostics(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(campaign_id): Path<String>,
) -> Result<Json<banditdb::state::CampaignDiagnosticsData>, AppError> {
    let stored_id = ns(&auth, &campaign_id);
    state.db.campaign_diagnostics(&stored_id)
        .map(|mut d| { d.campaign_id = campaign_id; Json(d) })
        .map_err(map_engine_err)
}

async fn handle_checkpoint(State(state): State<Arc<AppState>>) -> Result<Json<String>, AppError> {
    state.db.checkpoint().await
        .map(Json)
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e))
}

async fn handle_export(State(state): State<Arc<AppState>>) -> Result<Json<ExportResponse>, AppError> {
    let export_dir = state.db.export_dir();
    let entries = std::fs::read_dir(&export_dir)
        .map_err(|e| AppError(StatusCode::NOT_FOUND, format!("No exports yet: {e}")))?;

    let mut shards: HashMap<String, Vec<String>> = HashMap::new();
    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".parquet") { continue; }
        let stem = name.strip_suffix(".parquet").unwrap_or(&name);
        let cid  = stem.rfind('_')
            .filter(|&pos| stem[pos + 1..].chars().all(|c| c.is_ascii_digit()))
            .map(|pos| stem[..pos].to_string())
            .unwrap_or_else(|| stem.to_string());
        shards.entry(cid).or_default().push(name);
    }
    for v in shards.values_mut() { v.sort(); }
    Ok(Json(ExportResponse { export_dir, shards }))
}

async fn metrics_middleware(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let ep_idx = classify_endpoint(req.uri().path());
    let t0 = Instant::now();
    let resp = next.run(req).await;
    let elapsed = t0.elapsed().as_secs_f64();
    let status = resp.status().as_u16();

    let ep = &state.http_metrics.by_endpoint[ep_idx];
    if status < 400      { ep.req_2xx.fetch_add(1, Ordering::Relaxed); }
    else if status < 500 { ep.req_4xx.fetch_add(1, Ordering::Relaxed); }
    else                 { ep.req_5xx.fetch_add(1, Ordering::Relaxed); }

    for (i, &bound) in LATENCY_BOUNDS.iter().enumerate() {
        if elapsed <= bound { ep.lat_bucket[i].fetch_add(1, Ordering::Relaxed); }
    }
    atomic_add_f64(&ep.lat_sum_bits, elapsed);
    ep.lat_count.fetch_add(1, Ordering::Relaxed);

    resp
}

async fn handle_metrics(State(state): State<Arc<AppState>>) -> (HeaderMap, String) {
    let mut out = String::with_capacity(8192);

    // WAL health
    let wal_ok = if state.db.wal_healthy.load(Ordering::Relaxed) { 1 } else { 0 };
    out.push_str("# HELP banditdb_wal_healthy WAL writer health (1=ok, 0=degraded)\n");
    out.push_str("# TYPE banditdb_wal_healthy gauge\n");
    out.push_str(&format!("banditdb_wal_healthy {wal_ok}\n\n"));
    out.push_str("# HELP banditdb_wal_channel_available Remaining WAL channel slots\n");
    out.push_str("# TYPE banditdb_wal_channel_available gauge\n");
    out.push_str(&format!("banditdb_wal_channel_available {}\n\n", state.db.event_tx.capacity()));

    // Per-arm counters + campaign gauges (single lock acquisition)
    struct ArmCounts { p: u64, r: u64 }
    let campaigns = state.db.campaigns.read();

    let mut active = 0u64;
    let mut archived = 0u64;
    let arm_data: Vec<(String, String, ArmCounts)> = campaigns.iter()
        .inspect(|(_, c)| {
            if c.archived.load(Ordering::Relaxed) { archived += 1; } else { active += 1; }
        })
        .flat_map(|(cid, campaign)| {
            let safe_cid = prom_label(cid);
            campaign.arms.read().iter().map(|(aid, s)| (
                safe_cid.clone(), prom_label(aid),
                ArmCounts { p: s.prediction_count.load(Ordering::Relaxed),
                             r: s.reward_count.load(Ordering::Relaxed) },
            )).collect::<Vec<_>>()
        }).collect();

    out.push_str("# HELP banditdb_campaigns_active Active (non-archived) campaigns\n");
    out.push_str("# TYPE banditdb_campaigns_active gauge\n");
    out.push_str(&format!("banditdb_campaigns_active {active}\n\n"));
    out.push_str("# HELP banditdb_campaigns_archived Archived campaigns\n");
    out.push_str("# TYPE banditdb_campaigns_archived gauge\n");
    out.push_str(&format!("banditdb_campaigns_archived {archived}\n\n"));

    out.push_str("# HELP banditdb_arm_predictions_total Predictions served per arm\n");
    out.push_str("# TYPE banditdb_arm_predictions_total counter\n");
    for (cid, aid, c) in &arm_data {
        out.push_str(&format!("banditdb_arm_predictions_total{{campaign=\"{cid}\",arm=\"{aid}\"}} {}\n", c.p));
    }
    out.push('\n');
    out.push_str("# HELP banditdb_arm_rewards_total Rewards recorded per arm\n");
    out.push_str("# TYPE banditdb_arm_rewards_total counter\n");
    for (cid, aid, c) in &arm_data {
        out.push_str(&format!("banditdb_arm_rewards_total{{campaign=\"{cid}\",arm=\"{aid}\"}} {}\n", c.r));
    }
    out.push('\n');
    out.push_str("# HELP banditdb_tournament_traffic_bps Challenger traffic in basis points (Progressive)\n");
    out.push_str("# TYPE banditdb_tournament_traffic_bps gauge\n");
    for (cid, campaign) in campaigns.iter() {
        if matches!(&campaign.algorithm, Algorithm::Progressive(_)) {
            let bps = campaign.challenger_traffic_bps.load(Ordering::Relaxed);
            out.push_str(&format!("banditdb_tournament_traffic_bps{{campaign=\"{}\"}} {bps}\n", prom_label(cid)));
        }
    }
    out.push('\n');
    drop(campaigns);

    // HTTP request counters
    out.push_str("# HELP banditdb_http_requests_total HTTP requests by endpoint and status class\n");
    out.push_str("# TYPE banditdb_http_requests_total counter\n");
    for (i, name) in EP_NAMES.iter().enumerate() {
        let ep = &state.http_metrics.by_endpoint[i];
        let r2 = ep.req_2xx.load(Ordering::Relaxed);
        let r4 = ep.req_4xx.load(Ordering::Relaxed);
        let r5 = ep.req_5xx.load(Ordering::Relaxed);
        out.push_str(&format!("banditdb_http_requests_total{{endpoint=\"{name}\",status=\"2xx\"}} {r2}\n"));
        out.push_str(&format!("banditdb_http_requests_total{{endpoint=\"{name}\",status=\"4xx\"}} {r4}\n"));
        out.push_str(&format!("banditdb_http_requests_total{{endpoint=\"{name}\",status=\"5xx\"}} {r5}\n"));
    }
    out.push('\n');

    // HTTP latency histograms
    out.push_str("# HELP banditdb_http_request_duration_seconds HTTP request latency\n");
    out.push_str("# TYPE banditdb_http_request_duration_seconds histogram\n");
    for (i, name) in EP_NAMES.iter().enumerate() {
        let ep = &state.http_metrics.by_endpoint[i];
        for (j, &bound) in LATENCY_BOUNDS.iter().enumerate() {
            let count = ep.lat_bucket[j].load(Ordering::Relaxed);
            out.push_str(&format!(
                "banditdb_http_request_duration_seconds_bucket{{endpoint=\"{name}\",le=\"{bound}\"}} {count}\n"
            ));
        }
        let total = ep.lat_count.load(Ordering::Relaxed);
        let sum   = f64::from_bits(ep.lat_sum_bits.load(Ordering::Relaxed));
        out.push_str(&format!(
            "banditdb_http_request_duration_seconds_bucket{{endpoint=\"{name}\",le=\"+Inf\"}} {total}\n"
        ));
        out.push_str(&format!(
            "banditdb_http_request_duration_seconds_sum{{endpoint=\"{name}\"}} {sum:.6}\n"
        ));
        out.push_str(&format!(
            "banditdb_http_request_duration_seconds_count{{endpoint=\"{name}\"}} {total}\n"
        ));
    }
    out.push('\n');

    let mut headers = HeaderMap::new();
    headers.insert(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8".parse().unwrap());
    (headers, out)
}

async fn handle_openapi() -> (HeaderMap, &'static str) {
    let mut headers = HeaderMap::new();
    headers.insert(axum::http::header::CONTENT_TYPE, "application/yaml".parse().unwrap());
    (headers, include_str!("../docs/openapi.yaml"))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // Print version and exit before booting the runtime, so `banditdb --version`
    // works as an ops check instead of falling through to a full server start.
    if std::env::args().skip(1).any(|a| a == "--version" || a == "-V" || a == "version") {
        println!("banditdb {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    let filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(tracing::Level::INFO.into());
    let use_json = std::env::var("LOG_FORMAT").map(|v| v == "json").unwrap_or(false);
    if use_json { tracing_subscriber::fmt().json().with_env_filter(filter).init(); }
    else        { tracing_subscriber::fmt().with_env_filter(filter).init(); }

    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| ".".to_string());
    let wal_path = format!("{data_dir}/bandit_wal.jsonl");
    tracing::info!(data_dir = %data_dir, "BanditDB starting");

    let db       = Arc::new(BanditDB::new(&wal_path, &data_dir));
    let registry = Arc::new(KeyRegistry::from_env());

    if registry.is_open() {
        tracing::warn!("running in open mode — set BANDITDB_API_KEYS to enable authentication");
    } else {
        tracing::info!(key_count = registry.key_count(),
            tenant_mode = registry.tenant_mode, "API key authentication enabled");
    }

    let rps       = std::env::var("BANDITDB_RATE_LIMIT_PER_SEC")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1000u32);
    let rate_limiter = Some(Arc::new(
        RateLimiter::<String, _, _>::keyed(Quota::per_second(NonZeroU32::new(rps).expect("rps > 0")))
    ));
    tracing::info!(rps, "rate limiting enabled");

    let metrics_public = std::env::var("BANDITDB_METRICS_PUBLIC")
        .map(|v| v == "true" || v == "1").unwrap_or(false);

    let app_state = Arc::new(AppState {
        db: Arc::clone(&db), registry, rate_limiter, metrics_public,
        http_metrics: HttpMetrics::default(),
    });

    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

    let checkpoint_interval: Option<u64> = std::env::var("BANDITDB_CHECKPOINT_INTERVAL")
        .ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0);
    let max_wal_bytes: Option<u64> = std::env::var("BANDITDB_MAX_WAL_SIZE_MB")
        .ok().and_then(|v| v.parse::<u64>().ok()).filter(|&n| n > 0)
        .map(|mb| mb * 1024 * 1024);

    if checkpoint_interval.is_some() || max_wal_bytes.is_some() {
        let db_bg = Arc::clone(&db);
        let mut cancel = cancel_rx.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(10)) => {}
                    _ = cancel.changed() => { break; }
                }
                let count_exceeded = checkpoint_interval.is_some_and(|n| db_bg.rewarded_count.load(Ordering::Relaxed) >= n);
                let size_exceeded  = max_wal_bytes.is_some_and(|max| {
                    std::fs::metadata(&db_bg.wal_path).map(|m| m.len() > max).unwrap_or(false)
                });
                if count_exceeded || size_exceeded {
                    db_bg.rewarded_count.store(0, Ordering::Relaxed);
                    if let Err(e) = db_bg.checkpoint().await {
                        tracing::error!(error = %e, "auto-checkpoint failed");
                    }
                }
            }
            tracing::info!("auto-checkpoint task stopped");
        });
    }

    let state = Arc::clone(&app_state);

    let reader_routes = Router::new()
        .route("/campaigns",                get(handle_list_campaigns))
        .route("/campaign/:id",             get(handle_campaign_info))
        .route("/campaign/:id/report",      get(handle_campaign_report))
        .route("/campaign/:id/diagnostics", get(handle_campaign_diagnostics))
        .route("/export",                   get(handle_export));

    let writer_routes = Router::new()
        .route("/predict",       post(handle_predict))
        .route("/batch_predict", post(handle_batch_predict))
        .route("/reward",        post(handle_reward))
        .route("/campaign/:id/interact", post(handle_interact))
        .layer(middleware::from_fn(|ext: Extension<AuthContext>, req: Request, next: Next| {
            require_role(Role::Writer, ext, req, next)
        }));

    let admin_state = Arc::clone(&app_state);
    let admin_routes = Router::new()
        .route("/campaign",               post(handle_create_campaign))
        .route("/campaign/:id",           delete(handle_delete_campaign))
        .route("/campaign/:id/archive",   post(handle_archive_campaign))
        .route("/campaign/:id/restore",   post(handle_restore_campaign))
        .route("/checkpoint",             post(handle_checkpoint))
        .layer(middleware::from_fn(|ext: Extension<AuthContext>, req: Request, next: Next| {
            require_role(Role::Admin, ext, req, next)
        }))
        .with_state(Arc::clone(&admin_state));

    let protected = Router::new()
        .merge(reader_routes)
        .merge(writer_routes)
        .merge(admin_routes)
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .layer(middleware::from_fn_with_state(Arc::clone(&state), auth_middleware));

    let metrics_state = Arc::clone(&app_state);
    let metrics_route = if app_state.metrics_public {
        Router::new().route("/metrics", get(handle_metrics))
    } else {
        Router::new().route("/metrics", get(handle_metrics))
            .layer(DefaultBodyLimit::max(1024))
            .layer(middleware::from_fn_with_state(Arc::clone(&metrics_state), auth_middleware))
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers(Any);

    let app = Router::new()
        .route("/health",       get(handle_health))
        .route("/openapi.yaml", get(handle_openapi))
        .merge(metrics_route)
        .merge(protected)
        .layer(cors)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(middleware::from_fn_with_state(Arc::clone(&app_state), metrics_middleware))
        .with_state(app_state);

    let shutdown = async {
        #[cfg(unix)] {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = signal(SignalKind::terminate()).expect("SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {},
                _ = sigterm.recv()          => {},
            }
        }
        #[cfg(not(unix))] tokio::signal::ctrl_c().await.ok();
    };

    let port     = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr     = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).await.unwrap();
    tracing::info!(addr = %addr, "BanditDB listening");

    let db_shutdown = Arc::clone(&db);
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(shutdown)
        .await
        .unwrap();

    let _ = cancel_tx.send(true);
    tracing::info!("shutdown signal received — running final checkpoint");
    match tokio::time::timeout(std::time::Duration::from_secs(30), db_shutdown.checkpoint()).await {
        Ok(Ok(msg)) => tracing::info!(msg = %msg, "final checkpoint complete"),
        Ok(Err(e))  => tracing::error!(error = %e, "final checkpoint failed"),
        Err(_)      => tracing::error!("final checkpoint timed out after 30s"),
    }
    tracing::info!("shutdown complete");
}
