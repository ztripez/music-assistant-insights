//! HTTP route handlers.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::types::{
    AudioInfo, ConfigResponse, HealthResponse, HealthStatus, ModelInfo, ServerInfo,
};

use super::AppState;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// MessagePack response wrapper
pub struct MsgPack<T>(pub T);

impl<T: serde::Serialize> IntoResponse for MsgPack<T> {
    fn into_response(self) -> Response {
        match rmp_serde::to_vec(&self.0) {
            Ok(bytes) => (
                StatusCode::OK,
                [("content-type", "application/msgpack")],
                bytes,
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to serialize response: {e}"),
            )
                .into_response(),
        }
    }
}

/// Health check endpoint
///
/// GET /api/v1/health
pub async fn health() -> MsgPack<HealthResponse> {
    MsgPack(HealthResponse {
        status: HealthStatus::Healthy,
        version: VERSION.to_string(),
    })
}

/// Configuration endpoint
///
/// GET /api/v1/config
pub async fn config(State(state): State<AppState>) -> MsgPack<ConfigResponse> {
    let config = &state.config;

    MsgPack(ConfigResponse {
        model: ModelInfo {
            name: config.model.name.clone(),
            cuda_enabled: config.model.enable_cuda,
        },
        audio: AudioInfo {
            window_size_s: config.audio.window_size_s,
            hop_size_s: config.audio.hop_size_s,
        },
        server: ServerInfo {
            host: config.server.host.clone(),
            port: config.server.port,
        },
    })
}
