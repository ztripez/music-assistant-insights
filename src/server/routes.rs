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
pub async fn health(State(_state): State<AppState>) -> MsgPack<HealthResponse> {
    #[cfg(feature = "inference")]
    let model_loaded = _state.model.is_some();
    #[cfg(not(feature = "inference"))]
    let model_loaded = false;

    // Degraded if inference feature enabled but model not loaded
    #[cfg(feature = "inference")]
    let status = if model_loaded {
        HealthStatus::Healthy
    } else {
        HealthStatus::Degraded
    };
    #[cfg(not(feature = "inference"))]
    let status = HealthStatus::Healthy;

    MsgPack(HealthResponse {
        status,
        version: VERSION.to_string(),
        model_loaded,
    })
}

/// Configuration endpoint
///
/// GET /api/v1/config
pub async fn config(State(state): State<AppState>) -> MsgPack<ConfigResponse> {
    let config = &state.config;

    #[cfg(feature = "inference")]
    let (loaded, device) = if let Some(ref model) = state.model {
        (true, Some(model.device().to_string()))
    } else {
        (false, None)
    };
    #[cfg(not(feature = "inference"))]
    let (loaded, device) = (false, None);

    MsgPack(ConfigResponse {
        model: ModelInfo {
            name: config.model.name.clone(),
            cuda_enabled: config.model.enable_cuda,
            loaded,
            device,
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
