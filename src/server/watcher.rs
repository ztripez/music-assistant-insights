//! Watcher API endpoints.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};

use super::AppState;
use crate::watcher::{WatchedFolder, WatcherCommand, WatcherState};

/// Response for watcher status endpoint
#[derive(Debug, Serialize)]
pub struct WatcherStatusResponse {
    pub state: WatcherState,
}

/// Request to add a folder
#[derive(Debug, Deserialize)]
pub struct AddFolderRequest {
    pub path: String,
    #[serde(default = "default_recursive")]
    pub recursive: bool,
    pub label: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_recursive() -> bool {
    true
}

fn default_enabled() -> bool {
    true
}

/// Request to trigger a scan
#[derive(Debug, Deserialize)]
pub struct TriggerScanRequest {
    #[serde(default)]
    pub force: bool,
}

/// Simple success response
#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    pub success: bool,
    pub message: String,
}

/// Get watcher status
pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    let watcher_guard = state.watcher.read().await;

    match watcher_guard.as_ref() {
        Some(watcher) => {
            let watcher_state = watcher.state().await;
            Json(WatcherStatusResponse { state: watcher_state }).into_response()
        }
        None => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "Watcher not initialized",
                "status": "not_initialized"
            }))).into_response()
        }
    }
}

/// Start the watcher
pub async fn start(State(state): State<AppState>) -> impl IntoResponse {
    let mut watcher_guard = state.watcher.write().await;

    match watcher_guard.as_mut() {
        Some(watcher) => {
            match watcher.start().await {
                Ok(()) => Json(SuccessResponse {
                    success: true,
                    message: "Watcher started".to_string(),
                }).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                    "error": e.to_string()
                }))).into_response(),
            }
        }
        None => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "Watcher not initialized"
            }))).into_response()
        }
    }
}

/// Stop the watcher
pub async fn stop(State(state): State<AppState>) -> impl IntoResponse {
    let mut watcher_guard = state.watcher.write().await;

    match watcher_guard.as_mut() {
        Some(watcher) => {
            match watcher.stop().await {
                Ok(()) => Json(SuccessResponse {
                    success: true,
                    message: "Watcher stopped".to_string(),
                }).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                    "error": e.to_string()
                }))).into_response(),
            }
        }
        None => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "Watcher not initialized"
            }))).into_response()
        }
    }
}

/// Pause file processing
pub async fn pause(State(state): State<AppState>) -> impl IntoResponse {
    let watcher_guard = state.watcher.read().await;

    match watcher_guard.as_ref() {
        Some(watcher) => {
            match watcher.send_command(WatcherCommand::Pause).await {
                Ok(()) => Json(SuccessResponse {
                    success: true,
                    message: "Watcher paused".to_string(),
                }).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                    "error": e.to_string()
                }))).into_response(),
            }
        }
        None => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "Watcher not initialized"
            }))).into_response()
        }
    }
}

/// Resume file processing
pub async fn resume(State(state): State<AppState>) -> impl IntoResponse {
    let watcher_guard = state.watcher.read().await;

    match watcher_guard.as_ref() {
        Some(watcher) => {
            match watcher.send_command(WatcherCommand::Resume).await {
                Ok(()) => Json(SuccessResponse {
                    success: true,
                    message: "Watcher resumed".to_string(),
                }).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                    "error": e.to_string()
                }))).into_response(),
            }
        }
        None => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "Watcher not initialized"
            }))).into_response()
        }
    }
}

/// Trigger a manual scan
pub async fn trigger_scan(
    State(state): State<AppState>,
    Json(request): Json<TriggerScanRequest>,
) -> impl IntoResponse {
    let watcher_guard = state.watcher.read().await;

    match watcher_guard.as_ref() {
        Some(watcher) => {
            match watcher.send_command(WatcherCommand::TriggerScan { force: request.force }).await {
                Ok(()) => Json(SuccessResponse {
                    success: true,
                    message: format!("Scan triggered (force={})", request.force),
                }).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                    "error": e.to_string()
                }))).into_response(),
            }
        }
        None => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "Watcher not initialized"
            }))).into_response()
        }
    }
}

/// List watched folders
pub async fn list_folders(State(state): State<AppState>) -> impl IntoResponse {
    let watcher_guard = state.watcher.read().await;

    match watcher_guard.as_ref() {
        Some(watcher) => {
            let watcher_state = watcher.state().await;
            Json(watcher_state.folders).into_response()
        }
        None => {
            // Return from config if watcher not started
            let config = &state.config;
            #[cfg(feature = "watcher")]
            {
                Json(&config.watcher.folders).into_response()
            }
            #[cfg(not(feature = "watcher"))]
            {
                Json::<Vec<()>>(vec![]).into_response()
            }
        }
    }
}

/// Add a folder to watch
pub async fn add_folder(
    State(state): State<AppState>,
    Json(request): Json<AddFolderRequest>,
) -> impl IntoResponse {
    let watcher_guard = state.watcher.read().await;

    let folder = WatchedFolder {
        path: request.path.clone(),
        recursive: request.recursive,
        label: request.label,
        enabled: request.enabled,
    };

    match watcher_guard.as_ref() {
        Some(watcher) => {
            match watcher.send_command(WatcherCommand::AddFolder(folder)).await {
                Ok(()) => (StatusCode::CREATED, Json(SuccessResponse {
                    success: true,
                    message: format!("Folder added: {}", request.path),
                })).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                    "error": e.to_string()
                }))).into_response(),
            }
        }
        None => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "Watcher not started. Start the watcher first."
            }))).into_response()
        }
    }
}

/// Remove a folder from watching
pub async fn remove_folder(
    State(state): State<AppState>,
    Path(path): Path<String>,
) -> impl IntoResponse {
    let watcher_guard = state.watcher.read().await;

    // URL decode the path
    let decoded_path = urlencoding::decode(&path)
        .map(|s| s.into_owned())
        .unwrap_or(path);

    match watcher_guard.as_ref() {
        Some(watcher) => {
            match watcher.send_command(WatcherCommand::RemoveFolder(decoded_path.clone())).await {
                Ok(()) => Json(SuccessResponse {
                    success: true,
                    message: format!("Folder removed: {}", decoded_path),
                }).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                    "error": e.to_string()
                }))).into_response(),
            }
        }
        None => {
            (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "Watcher not started"
            }))).into_response()
        }
    }
}
