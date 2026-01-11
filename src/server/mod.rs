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
mod websocket;

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
use crate::queue::AudioQueue;

#[cfg(feature = "inference")]
use crate::inference::{ClapModel, DownloadManager};

#[cfg(feature = "inference")]
use crate::mood::MoodClassifier;

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
    /// Streaming session manager (legacy REST-based)
    #[cfg(feature = "inference")]
    pub stream_manager: SharedStreamManager,
    /// Mood classifier (cached prompt embeddings)
    #[cfg(feature = "inference")]
    pub mood_classifier: Arc<RwLock<Option<MoodClassifier>>>,
    /// Audio queue for WebSocket streaming (crash-resistant)
    pub audio_queue: Option<Arc<AudioQueue>>,
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
            #[cfg(feature = "inference")]
            mood_classifier: Arc::new(RwLock::new(None)),
            audio_queue: None,
            started_at: Instant::now(),
        }
    }

    /// Create AppState with a loaded model.
    /// Note: Call `init_classifier()` after construction to initialize mood classification.
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
            mood_classifier: Arc::new(RwLock::new(None)),
            audio_queue: None,
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
            #[cfg(feature = "inference")]
            mood_classifier: Arc::new(RwLock::new(None)),
            audio_queue: None,
            started_at: Instant::now(),
        }
    }

    /// Create AppState with both model and storage.
    /// Note: Call `init_classifier()` after construction to initialize mood classification.
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
            mood_classifier: Arc::new(RwLock::new(None)),
            audio_queue: None,
            started_at: Instant::now(),
        }
    }

    /// Set the audio queue for WebSocket streaming
    pub fn with_audio_queue(mut self, queue: AudioQueue) -> Self {
        self.audio_queue = Some(Arc::new(queue));
        self
    }

    /// Check if a model is loaded
    #[cfg(feature = "inference")]
    pub async fn has_model(&self) -> bool {
        self.model.read().await.is_some()
    }

    /// Load a new model, replacing any existing one.
    /// Mood classifier initialization is done in a blocking task to avoid blocking the async runtime.
    #[cfg(feature = "inference")]
    pub async fn load_model(
        &self,
        model: ClapModel,
        model_id: String,
    ) -> Result<(), crate::error::AppError> {
        // Wrap model in Arc first so we can share it
        let model = Arc::new(model);

        // Pre-compute mood classifier embeddings in a blocking task
        let model_for_classifier = model.clone();
        let classifier = tokio::task::spawn_blocking(move || {
            MoodClassifier::new(&model_for_classifier)
        })
        .await
        .map_err(|e| crate::error::AppError::Internal(format!("Join error: {}", e)))?;

        // Take write locks to swap model and classifier
        let mut model_guard = self.model.write().await;
        let mut id_guard = self.current_model_id.write().await;
        let mut classifier_guard = self.mood_classifier.write().await;

        // Drop old model/classifier and set new ones
        *model_guard = Some(model);
        *id_guard = Some(model_id);
        *classifier_guard = Some(classifier);

        Ok(())
    }

    /// Initialize the mood classifier from the currently loaded model.
    /// This should be called after constructing AppState with `with_model` or `with_model_and_storage`.
    /// Runs in a blocking task to avoid blocking the async runtime.
    #[cfg(feature = "inference")]
    pub async fn init_classifier(&self) -> Result<(), crate::error::AppError> {
        // Get the current model
        let model_guard = self.model.read().await;
        let model = model_guard
            .as_ref()
            .ok_or_else(|| crate::error::AppError::Internal("No model loaded".to_string()))?
            .clone();
        drop(model_guard);

        // Pre-compute mood classifier embeddings in a blocking task
        let classifier = tokio::task::spawn_blocking(move || MoodClassifier::new(&model))
            .await
            .map_err(|e| crate::error::AppError::Internal(format!("Join error: {}", e)))?;

        // Store the classifier
        let mut classifier_guard = self.mood_classifier.write().await;
        *classifier_guard = Some(classifier);

        Ok(())
    }

    /// Unload the current model
    #[cfg(feature = "inference")]
    pub async fn unload_model(&self) {
        let mut model_guard = self.model.write().await;
        let mut id_guard = self.current_model_id.write().await;
        let mut classifier_guard = self.mood_classifier.write().await;
        *model_guard = None;
        *id_guard = None;
        *classifier_guard = None;
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
        .route("/tracks/search-text", post(tracks::text_search))
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

    // WebSocket audio streaming endpoint
    api_routes = api_routes.route("/ws/audio", get(websocket::audio_stream_handler));

    Router::new()
        .nest("/api/v1", api_routes)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
