//! HTTP server setup and routing.

mod embed;
mod extractors;
mod management;
mod mood;
mod routes;
#[cfg(feature = "inference")]
mod stream;
mod taste;
mod tracks;
#[cfg(feature = "watcher")]
mod watcher;

use axum::{
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;
use std::time::Instant;
#[cfg(any(feature = "inference", feature = "watcher"))]
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;

use crate::config::AppConfig;

#[cfg(feature = "inference")]
use crate::inference::{ClapModel, DownloadManager};

#[cfg(any(feature = "storage", feature = "storage-file"))]
use crate::storage::VectorStorage;

#[cfg(feature = "watcher")]
use crate::watcher::WatcherService;

#[cfg(feature = "inference")]
use stream::{SharedStreamManager, StreamSessionManager};

#[cfg(feature = "inference")]
pub use stream::spawn_session_cleanup_task;

/// Type alias for boxed storage implementation
#[cfg(any(feature = "storage", feature = "storage-file"))]
pub type BoxedStorage = Box<dyn VectorStorage>;

/// Shared application state passed to all handlers
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    #[cfg(feature = "inference")]
    pub model: Arc<RwLock<Option<Arc<ClapModel>>>>,
    #[cfg(feature = "inference")]
    pub download_manager: DownloadManager,
    #[cfg(feature = "inference")]
    pub current_model_id: Arc<RwLock<Option<String>>>,
    #[cfg(any(feature = "storage", feature = "storage-file"))]
    pub storage: Option<Arc<BoxedStorage>>,
    #[cfg(feature = "watcher")]
    pub watcher: Arc<RwLock<Option<WatcherService>>>,
    /// Streaming session manager
    #[cfg(feature = "inference")]
    pub stream_manager: SharedStreamManager,
    /// Server start time for uptime calculation
    pub started_at: Instant,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config: Arc::new(config),
            #[cfg(feature = "inference")]
            model: Arc::new(RwLock::new(None)),
            #[cfg(feature = "inference")]
            download_manager: DownloadManager::new(),
            #[cfg(feature = "inference")]
            current_model_id: Arc::new(RwLock::new(None)),
            #[cfg(any(feature = "storage", feature = "storage-file"))]
            storage: None,
            #[cfg(feature = "watcher")]
            watcher: Arc::new(RwLock::new(None)),
            #[cfg(feature = "inference")]
            stream_manager: Arc::new(StreamSessionManager::new()),
            started_at: Instant::now(),
        }
    }

    /// Create AppState with a loaded model
    #[cfg(feature = "inference")]
    pub fn with_model(config: AppConfig, model: ClapModel, model_id: String) -> Self {
        Self {
            config: Arc::new(config),
            model: Arc::new(RwLock::new(Some(Arc::new(model)))),
            download_manager: DownloadManager::new(),
            current_model_id: Arc::new(RwLock::new(Some(model_id))),
            #[cfg(any(feature = "storage", feature = "storage-file"))]
            storage: None,
            #[cfg(feature = "watcher")]
            watcher: Arc::new(RwLock::new(None)),
            stream_manager: Arc::new(StreamSessionManager::new()),
            started_at: Instant::now(),
        }
    }

    /// Create AppState with storage (any backend)
    #[cfg(any(feature = "storage", feature = "storage-file"))]
    pub fn with_storage(config: AppConfig, storage: BoxedStorage) -> Self {
        Self {
            config: Arc::new(config),
            #[cfg(feature = "inference")]
            model: Arc::new(RwLock::new(None)),
            #[cfg(feature = "inference")]
            download_manager: DownloadManager::new(),
            #[cfg(feature = "inference")]
            current_model_id: Arc::new(RwLock::new(None)),
            storage: Some(Arc::new(storage)),
            #[cfg(feature = "watcher")]
            watcher: Arc::new(RwLock::new(None)),
            #[cfg(feature = "inference")]
            stream_manager: Arc::new(StreamSessionManager::new()),
            started_at: Instant::now(),
        }
    }

    /// Create AppState with both model and storage
    #[cfg(all(
        feature = "inference",
        any(feature = "storage", feature = "storage-file")
    ))]
    pub fn with_model_and_storage(
        config: AppConfig,
        model: ClapModel,
        model_id: String,
        storage: BoxedStorage,
    ) -> Self {
        Self {
            config: Arc::new(config),
            model: Arc::new(RwLock::new(Some(Arc::new(model)))),
            download_manager: DownloadManager::new(),
            current_model_id: Arc::new(RwLock::new(Some(model_id))),
            storage: Some(Arc::new(storage)),
            #[cfg(feature = "watcher")]
            watcher: Arc::new(RwLock::new(None)),
            stream_manager: Arc::new(StreamSessionManager::new()),
            started_at: Instant::now(),
        }
    }

    /// Check if a model is loaded
    #[cfg(feature = "inference")]
    pub async fn has_model(&self) -> bool {
        self.model.read().await.is_some()
    }

    /// Load a new model, replacing any existing one
    #[cfg(feature = "inference")]
    pub async fn load_model(
        &self,
        model: ClapModel,
        model_id: String,
    ) -> Result<(), crate::error::AppError> {
        // Take write lock to swap the model
        let mut model_guard = self.model.write().await;
        let mut id_guard = self.current_model_id.write().await;

        // Drop old model and set new one
        *model_guard = Some(Arc::new(model));
        *id_guard = Some(model_id);

        Ok(())
    }

    /// Unload the current model
    #[cfg(feature = "inference")]
    pub async fn unload_model(&self) {
        let mut model_guard = self.model.write().await;
        let mut id_guard = self.current_model_id.write().await;
        *model_guard = None;
        *id_guard = None;
    }

    /// Get the current model ID if a model is loaded
    #[cfg(feature = "inference")]
    pub async fn get_current_model_id(&self) -> Option<String> {
        self.current_model_id.read().await.clone()
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }
}

/// Creates the application router with all routes configured
#[allow(unused_mut)]
pub fn create_router(state: AppState) -> Router {
    let mut api_routes = Router::new()
        .route("/health", get(routes::health))
        .route("/config", get(routes::config))
        // Management endpoints
        .route("/status", get(management::status))
        .route("/models", get(management::list_models))
        .route("/models/download", post(management::start_download))
        .route("/models/downloads", get(management::list_downloads))
        .route("/models/{model_id}/load", post(management::load_model))
        .route("/models/{model_id}", delete(management::delete_model))
        .route("/storage/stats", get(management::storage_stats))
        // Embedding generation endpoints
        .route("/embed/text", post(embed::text_embed))
        .route("/embed/audio", post(embed::audio_embed))
        // Track storage endpoints
        .route("/tracks/upsert", post(tracks::upsert))
        .route("/tracks/search", post(tracks::search))
        .route("/tracks/embed-text", post(tracks::embed_text_and_store))
        .route(
            "/tracks/:id",
            get(tracks::get_track).delete(tracks::delete_track),
        )
        // Batch operations
        .route("/tracks/batch-upsert", post(tracks::batch_upsert))
        .route("/tracks/batch-embed-text", post(tracks::batch_embed_text))
        // Mood classification endpoints
        .route("/mood/classify", post(mood::classify_mood))
        .route("/mood/list", get(mood::list_moods))
        // Taste profile endpoints
        .route("/users/:user_id/profile/compute", post(taste::compute_profile))
        .route("/users/:user_id/recommend", post(taste::get_recommendations))
        .route("/users/:user_id/profile/vector", get(taste::get_taste_vector))
        .route("/users/:user_id/profile", delete(taste::delete_profile))
        .route("/users/:user_id/profiles", delete(taste::delete_all_profiles));

    // Streaming ingestion endpoints (when inference feature is enabled)
    #[cfg(feature = "inference")]
    {
        api_routes = api_routes
            .route("/stream/start", post(stream::start_stream))
            .route("/stream/:session_id/frames", post(stream::stream_frames))
            .route("/stream/:session_id/end", post(stream::end_stream))
            .route("/stream/:session_id", delete(stream::abort_stream))
            .route("/stream/:session_id/status", get(stream::stream_status));
    }

    // Watcher endpoints (when feature is enabled)
    #[cfg(feature = "watcher")]
    {
        api_routes = api_routes
            .route("/watcher/status", get(watcher::status))
            .route("/watcher/start", post(watcher::start))
            .route("/watcher/stop", post(watcher::stop))
            .route("/watcher/pause", post(watcher::pause))
            .route("/watcher/resume", post(watcher::resume))
            .route("/watcher/scan", post(watcher::trigger_scan))
            .route(
                "/watcher/folders",
                get(watcher::list_folders).post(watcher::add_folder),
            )
            .route("/watcher/folders/:path", delete(watcher::remove_folder));
    }

    Router::new()
        .nest("/api/v1", api_routes)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
