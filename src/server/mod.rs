//! HTTP server setup and routing.

mod routes;

use axum::{routing::get, Router};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use crate::config::AppConfig;

/// Shared application state passed to all handlers
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
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
