//! HTTP server setup and routing.

mod routes;

use axum::{routing::get, Router};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use crate::config::AppConfig;

#[cfg(feature = "inference")]
use crate::inference::ClapModel;

/// Shared application state passed to all handlers
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    #[cfg(feature = "inference")]
    pub model: Option<Arc<ClapModel>>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config: Arc::new(config),
            #[cfg(feature = "inference")]
            model: None,
        }
    }

    /// Create AppState with a loaded model
    #[cfg(feature = "inference")]
    pub fn with_model(config: AppConfig, model: ClapModel) -> Self {
        Self {
            config: Arc::new(config),
            model: Some(Arc::new(model)),
        }
    }

    /// Check if a model is loaded
    #[cfg(feature = "inference")]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }
}

/// Creates the application router with all routes configured
pub fn create_router(state: AppState) -> Router {
    let api_routes = Router::new()
        .route("/health", get(routes::health))
        .route("/config", get(routes::config));

    Router::new()
        .nest("/api/v1", api_routes)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
