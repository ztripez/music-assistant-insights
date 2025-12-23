//! Music Assistant Insight Sidecar - Entry Point

use anyhow::Context;
use tokio::signal;
#[cfg(any(feature = "inference", feature = "storage"))]
use tracing::error;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use insight_sidecar::{config::AppConfig, server};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    init_logging();

    info!("Starting Music Assistant Insight Sidecar");

    // Load configuration
    let config = AppConfig::load().unwrap_or_else(|e| {
        warn!("Failed to load config from environment: {e}, using defaults");
        AppConfig {
            model: Default::default(),
            audio: Default::default(),
            storage: Default::default(),
            server: Default::default(),
        }
    });

    info!(
        model = %config.model.name,
        cuda = config.model.enable_cuda,
        storage_url = %config.storage.url,
        "Configuration loaded"
    );

    // Create app state (with optional model and storage)
    let state = create_app_state(config.clone()).await;

    // Create router
    let app = server::create_router(state);

    // Bind to socket
    let addr = config.server.socket_addr();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("Failed to bind to address")?;

    info!(%addr, "Server listening");

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server error")?;

    info!("Server shutdown complete");
    Ok(())
}

/// Create application state, loading the ML model and storage if features are enabled
#[cfg(all(feature = "inference", feature = "storage"))]
async fn create_app_state(config: AppConfig) -> server::AppState {
    use insight_sidecar::inference::{download_model, ClapModel};
    use insight_sidecar::storage::{QdrantStorage, VectorStorage};

    // Load model first
    let model = load_model(&config).await;

    // Then connect to storage
    let storage = connect_storage(&config).await;

    match (model, storage) {
        (Some(m), Some(s)) => server::AppState::with_model_and_storage(config, m, s),
        (Some(m), None) => server::AppState::with_model(config, m),
        (None, Some(s)) => server::AppState::with_storage(config, s),
        (None, None) => server::AppState::new(config),
    }
}

#[cfg(all(feature = "inference", feature = "storage"))]
async fn load_model(config: &AppConfig) -> Option<insight_sidecar::inference::ClapModel> {
    use insight_sidecar::inference::{download_model, ClapModel};

    info!(model = %config.model.name, "Downloading/loading CLAP model...");

    match download_model(&config.model.name, None).await {
        Ok(paths) => {
            info!(?paths, "Model files ready, loading into ONNX Runtime...");
            let use_cuda = config.model.enable_cuda;
            match tokio::task::spawn_blocking(move || ClapModel::load(&paths, use_cuda)).await {
                Ok(Ok(model)) => {
                    info!(device = %model.device(), "CLAP model loaded successfully");
                    Some(model)
                }
                Ok(Err(e)) => {
                    error!("Failed to load model: {e}");
                    warn!("Starting without ML inference capability");
                    None
                }
                Err(e) => {
                    error!("Model loading task panicked: {e}");
                    warn!("Starting without ML inference capability");
                    None
                }
            }
        }
        Err(e) => {
            error!("Failed to download model: {e}");
            warn!("Starting without ML inference capability");
            None
        }
    }
}

#[cfg(all(feature = "inference", feature = "storage"))]
async fn connect_storage(config: &AppConfig) -> Option<insight_sidecar::storage::QdrantStorage> {
    use insight_sidecar::storage::{QdrantStorage, VectorStorage};

    if !config.storage.enabled {
        info!("Storage disabled in configuration");
        return None;
    }

    info!(url = %config.storage.url, "Connecting to Qdrant...");

    match QdrantStorage::new(&config.storage.url, config.storage.collection_prefix.clone()).await {
        Ok(storage) => {
            if let Err(e) = storage.initialize().await {
                error!("Failed to initialize storage collections: {e}");
                warn!("Starting without vector storage capability");
                return None;
            }
            info!("Qdrant storage connected and initialized");
            Some(storage)
        }
        Err(e) => {
            error!("Failed to connect to Qdrant: {e}");
            warn!("Starting without vector storage capability");
            None
        }
    }
}

/// Create application state with inference only
#[cfg(all(feature = "inference", not(feature = "storage")))]
async fn create_app_state(config: AppConfig) -> server::AppState {
    use insight_sidecar::inference::{download_model, ClapModel};

    info!(model = %config.model.name, "Downloading/loading CLAP model...");

    match download_model(&config.model.name, None).await {
        Ok(paths) => {
            info!(?paths, "Model files ready, loading into ONNX Runtime...");
            let use_cuda = config.model.enable_cuda;
            match tokio::task::spawn_blocking(move || ClapModel::load(&paths, use_cuda)).await {
                Ok(Ok(model)) => {
                    info!(device = %model.device(), "CLAP model loaded successfully");
                    server::AppState::with_model(config, model)
                }
                Ok(Err(e)) => {
                    error!("Failed to load model: {e}");
                    warn!("Starting without ML inference capability");
                    server::AppState::new(config)
                }
                Err(e) => {
                    error!("Model loading task panicked: {e}");
                    warn!("Starting without ML inference capability");
                    server::AppState::new(config)
                }
            }
        }
        Err(e) => {
            error!("Failed to download model: {e}");
            warn!("Starting without ML inference capability");
            server::AppState::new(config)
        }
    }
}

/// Create application state with storage only
#[cfg(all(feature = "storage", not(feature = "inference")))]
async fn create_app_state(config: AppConfig) -> server::AppState {
    use insight_sidecar::storage::{QdrantStorage, VectorStorage};

    if !config.storage.enabled {
        info!("Storage disabled in configuration");
        return server::AppState::new(config);
    }

    info!(url = %config.storage.url, "Connecting to Qdrant...");

    match QdrantStorage::new(&config.storage.url, config.storage.collection_prefix.clone()).await {
        Ok(storage) => {
            if let Err(e) = storage.initialize().await {
                error!("Failed to initialize storage collections: {e}");
                warn!("Starting without vector storage capability");
                return server::AppState::new(config);
            }
            info!("Qdrant storage connected and initialized");
            server::AppState::with_storage(config, storage)
        }
        Err(e) => {
            error!("Failed to connect to Qdrant: {e}");
            warn!("Starting without vector storage capability");
            server::AppState::new(config)
        }
    }
}

/// Create application state without optional features
#[cfg(not(any(feature = "inference", feature = "storage")))]
async fn create_app_state(config: AppConfig) -> server::AppState {
    info!("No optional features enabled");
    server::AppState::new(config)
}

/// Initialize the tracing subscriber for logging
fn init_logging() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "insight_sidecar=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        }
        () = terminate => {
            info!("Received SIGTERM, starting graceful shutdown");
        }
    }
}
