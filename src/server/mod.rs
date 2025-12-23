//! HTTP server setup and routing.

mod extractors;
mod routes;
mod tracks;

use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use crate::config::AppConfig;

#[cfg(feature = "inference")]
use crate::inference::ClapModel;

#[cfg(feature = "storage")]
use crate::storage::QdrantStorage;

/// Shared application state passed to all handlers
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    #[cfg(feature = "inference")]
    pub model: Option<Arc<ClapModel>>,
    #[cfg(feature = "storage")]
    pub storage: Option<Arc<QdrantStorage>>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config: Arc::new(config),
            #[cfg(feature = "inference")]
            model: None,
            #[cfg(feature = "storage")]
            storage: None,
        }
    }

    /// Create AppState with a loaded model
    #[cfg(feature = "inference")]
    pub fn with_model(config: AppConfig, model: ClapModel) -> Self {
        Self {
            config: Arc::new(config),
            model: Some(Arc::new(model)),
            #[cfg(feature = "storage")]
            storage: None,
        }
    }

    /// Create AppState with storage
    #[cfg(feature = "storage")]
    pub fn with_storage(config: AppConfig, storage: QdrantStorage) -> Self {
        Self {
            config: Arc::new(config),
            #[cfg(feature = "inference")]
            model: None,
            storage: Some(Arc::new(storage)),
        }
    }

    /// Create AppState with both model and storage
    #[cfg(all(feature = "inference", feature = "storage"))]
    pub fn with_model_and_storage(config: AppConfig, model: ClapModel, storage: QdrantStorage) -> Self {
        Self {
            config: Arc::new(config),
            model: Some(Arc::new(model)),
            storage: Some(Arc::new(storage)),
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
        .route("/config", get(routes::config))
        // Track embedding endpoints
        .route("/tracks/upsert", post(tracks::upsert))
        .route("/tracks/search", post(tracks::search))
        .route(
            "/tracks/{id}",
            get(tracks::get_track).delete(tracks::delete_track),
        );

    Router::new()
        .nest("/api/v1", api_routes)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
