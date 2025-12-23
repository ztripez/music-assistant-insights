//! Music Assistant Insight Sidecar - Entry Point

use anyhow::Context;
use tokio::signal;
#[cfg(feature = "inference")]
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
        "Configuration loaded"
    );

    // Create app state (with or without model)
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

/// Create application state, loading the ML model if inference feature is enabled
#[cfg(feature = "inference")]
async fn create_app_state(config: AppConfig) -> server::AppState {
    use insight_sidecar::inference::{download_model, ClapModel};

    info!(model = %config.model.name, "Downloading/loading CLAP model...");

    // Download model files
    match download_model(&config.model.name, None).await {
        Ok(paths) => {
            info!(?paths, "Model files ready, loading into ONNX Runtime...");

            // Load model into ONNX Runtime (blocking operation)
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

#[cfg(not(feature = "inference"))]
async fn create_app_state(config: AppConfig) -> server::AppState {
    info!("Inference feature not enabled, starting without ML model");
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
