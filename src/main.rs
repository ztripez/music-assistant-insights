//! Music Assistant Insight Sidecar - Entry Point

use anyhow::Context;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
#[cfg(any(feature = "inference", feature = "storage", feature = "storage-file"))]
use tracing::error;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use insight_sidecar::{config::AppConfig, queue::AudioQueue, server, worker};

/// Music Assistant Insight Sidecar
///
/// ML inference sidecar providing audio/text embeddings using CLAP models
/// with vector storage for similarity search.
#[derive(Parser, Debug)]
#[command(name = "insight-sidecar")]
#[command(version, about, long_about = None)]
struct Cli {
    /// Server port to listen on
    #[arg(short, long, env = "INSIGHT_SERVER__PORT")]
    port: Option<u16>,

    /// Server host to bind to
    #[arg(long, env = "INSIGHT_SERVER__HOST")]
    host: Option<String>,

    /// Storage mode: "file" for embedded usearch (default), "qdrant" for hosted Qdrant
    #[arg(long, env = "INSIGHT_STORAGE__MODE", value_parser = clap::value_parser!(String))]
    storage_mode: Option<String>,

    /// Data directory for file-based storage (default: ~/.local/share/insight-sidecar)
    #[arg(long, env = "INSIGHT_STORAGE__DATA_DIR")]
    data_dir: Option<String>,

    /// Qdrant server URL (e.g., <http://localhost:6334> or <https://xxx.cloud.qdrant.io:6333>)
    #[arg(long, env = "INSIGHT_STORAGE__URL")]
    qdrant_url: Option<String>,

    /// Qdrant API key for authenticated instances (e.g., Qdrant Cloud)
    #[arg(long, env = "INSIGHT_STORAGE__API_KEY")]
    qdrant_api_key: Option<String>,

    /// Collection name prefix for multi-tenant setups
    #[arg(long, env = "INSIGHT_STORAGE__COLLECTION_PREFIX")]
    collection_prefix: Option<String>,

    /// Disable vector storage (run inference-only mode)
    #[arg(long)]
    no_storage: bool,

    /// CLAP model to use (Hugging Face model ID)
    #[arg(long, env = "INSIGHT_MODEL__NAME")]
    model: Option<String>,

    /// Enable CUDA acceleration for model inference (NVIDIA GPUs)
    #[arg(long, env = "INSIGHT_MODEL__ENABLE_CUDA")]
    cuda: bool,

    /// Enable `ROCm` acceleration for model inference (AMD GPUs)
    #[arg(long, env = "INSIGHT_MODEL__ENABLE_ROCM")]
    rocm: bool,

    /// Enable `CoreML` acceleration for model inference (Apple Silicon/macOS)
    #[arg(long, env = "INSIGHT_MODEL__ENABLE_COREML")]
    coreml: bool,

    /// Enable `DirectML` acceleration for model inference (Windows GPU)
    #[arg(long, env = "INSIGHT_MODEL__ENABLE_DIRECTML")]
    directml: bool,

    /// Enable `OpenVINO` acceleration for model inference (Intel CPUs/GPUs/VPUs)
    #[arg(long, env = "INSIGHT_MODEL__ENABLE_OPENVINO")]
    openvino: bool,

    /// Increase logging verbosity (-v for debug, -vv for trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Quiet mode (only show warnings and errors)
    #[arg(short, long)]
    quiet: bool,

    /// Enable file logging (writes to ~/.local/share/insight-sidecar/logs/)
    #[arg(long, env = "INSIGHT_LOG_FILE")]
    log_file: bool,

    /// Custom log directory (default: ~/.local/share/insight-sidecar/logs)
    #[arg(long, env = "INSIGHT_LOG_DIR")]
    log_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    init_logging(cli.verbose, cli.quiet, cli.log_file, cli.log_dir)?;

    print_banner();

    // Load configuration (env vars) then merge with CLI args
    let mut config = AppConfig::load().unwrap_or_else(|e| {
        warn!("Failed to load config from environment: {e}, using defaults");
        AppConfig::default()
    });

    // CLI args override environment/defaults
    if let Some(port) = cli.port {
        config.server.port = port;
    }
    if let Some(host) = cli.host {
        config.server.host = host;
    }
    if let Some(mode) = cli.storage_mode {
        config.storage.mode = mode.parse().unwrap_or_else(|e| {
            warn!("Invalid storage mode '{}': {}, using default", mode, e);
            insight_sidecar::config::StorageMode::default()
        });
    }
    if let Some(data_dir) = cli.data_dir {
        config.storage.data_dir = data_dir;
    }
    if let Some(url) = cli.qdrant_url {
        config.storage.url = url;
    }
    if let Some(api_key) = cli.qdrant_api_key {
        config.storage.api_key = Some(api_key);
    }
    if let Some(prefix) = cli.collection_prefix {
        config.storage.collection_prefix = Some(prefix);
    }
    if cli.no_storage {
        config.storage.enabled = false;
    }
    if let Some(model) = cli.model {
        config.model.name = model;
    }
    if cli.cuda {
        config.model.enable_cuda = true;
    }
    if cli.rocm {
        config.model.enable_rocm = true;
    }
    if cli.coreml {
        config.model.enable_coreml = true;
    }
    if cli.directml {
        config.model.enable_directml = true;
    }
    if cli.openvino {
        config.model.enable_openvino = true;
    }

    info!(
        model = %config.model.name,
        cuda = config.model.enable_cuda,
        rocm = config.model.enable_rocm,
        coreml = config.model.enable_coreml,
        directml = config.model.enable_directml,
        openvino = config.model.enable_openvino,
        storage_mode = %config.storage.mode,
        storage_enabled = config.storage.enabled,
        "Configuration loaded"
    );

    // Create app state (with optional model and storage)
    let mut state = create_app_state(config.clone()).await;

    // Initialize audio queue for WebSocket streaming
    let data_dir = PathBuf::from(&config.storage.data_dir);
    let queue_db_path = data_dir.join("queue.redb");
    let audio_sessions_dir = data_dir.join("audio_sessions");

    match AudioQueue::new(&queue_db_path, &audio_sessions_dir) {
        Ok(queue) => {
            info!(
                queue_db = %queue_db_path.display(),
                audio_dir = %audio_sessions_dir.display(),
                "Audio queue initialized"
            );

            // Startup recovery: reset any "Processing" sessions to "Pending"
            match queue.reset_processing_to_pending() {
                Ok(count) if count > 0 => {
                    info!(count, "Reset interrupted sessions to pending");
                }
                Err(e) => {
                    warn!(error = %e, "Failed to reset processing sessions");
                }
                _ => {}
            }

            // Clean up orphaned PCM files (not in queue)
            match queue.cleanup_orphaned_files() {
                Ok(count) if count > 0 => {
                    info!(count, "Cleaned up orphaned audio files");
                }
                Err(e) => {
                    warn!(error = %e, "Failed to clean up orphaned files");
                }
                _ => {}
            }

            // Clean up old PCM files (>24h, handles db corruption edge cases)
            match queue.cleanup_old_files(std::time::Duration::from_secs(24 * 60 * 60)) {
                Ok(count) if count > 0 => {
                    info!(count, "Cleaned up old audio files (>24h)");
                }
                Err(e) => {
                    warn!(error = %e, "Failed to clean up old files");
                }
                _ => {}
            }

            // Report pending sessions
            match queue.list_pending() {
                Ok(pending) if !pending.is_empty() => {
                    info!(count = pending.len(), "Pending audio sessions to process");
                }
                Err(e) => {
                    warn!(error = %e, "Failed to list pending sessions");
                }
                _ => {}
            }

            // Attach queue to state
            state = state.with_audio_queue(queue);
        }
        Err(e) => {
            warn!(error = %e, "Failed to initialize audio queue, WebSocket streaming disabled");
        }
    }

    // Start background tasks
    #[cfg(feature = "inference")]
    {
        // Start audio processing worker if we have queue and model
        if let Some(ref queue) = state.audio_queue {
            let worker_state = Arc::new(worker::WorkerState {
                queue: queue.clone(),
                model: state.model.clone(),
                #[cfg(any(feature = "storage", feature = "storage-file"))]
                storage: state.storage.clone(),
                config: worker::WorkerConfig::default(),
            });

            let _worker_handle = worker::spawn_worker(worker_state);
            info!("Audio processing worker started");
            // Note: worker_handle could be stored for graceful shutdown
        }
    }

    // Create router
    let app = server::create_router(state);

    // Bind to socket
    let addr = config.server.socket_addr();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("Failed to bind to address")?;

    info!(%addr, "Server listening");
    info!("API available at http://{}/api/v1/health", addr);

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server error")?;

    info!("Server shutdown complete");
    Ok(())
}

/// Print startup banner
fn print_banner() {
    let version = env!("CARGO_PKG_VERSION");
    info!("╔═══════════════════════════════════════════════════════╗");
    info!(
        "║     Music Assistant Insight Sidecar v{}          ║",
        version
    );
    info!("║     ML-powered audio/text embeddings for MA           ║");
    info!("╚═══════════════════════════════════════════════════════╝");
}

/// Create application state, loading the ML model and storage if features are enabled
#[cfg(all(
    feature = "inference",
    any(feature = "storage", feature = "storage-file")
))]
async fn create_app_state(config: AppConfig) -> server::AppState {
    // Load model first
    let model_id = config.model.name.clone();
    let model = load_model(&config).await;
    let has_model = model.is_some();

    // Then connect to storage
    let storage = connect_storage(&config).await;

    let state = match (model, storage) {
        (Some(m), Some(s)) => server::AppState::with_model_and_storage(config, m, model_id, s),
        (Some(m), None) => server::AppState::with_model(config, m, model_id),
        (None, Some(s)) => server::AppState::with_storage(config, s),
        (None, None) => server::AppState::new(config),
    };

    // Initialize mood classifier in a blocking task (if model was loaded)
    if has_model {
        info!("Initializing mood classifier...");
        if let Err(e) = state.init_classifier().await {
            error!("Failed to initialize mood classifier: {e}");
        } else {
            info!("Mood classifier initialized");
        }
    }

    state
}

#[cfg(all(
    feature = "inference",
    any(feature = "storage", feature = "storage-file")
))]
async fn load_model(config: &AppConfig) -> Option<insight_sidecar::inference::ClapModel> {
    use insight_sidecar::inference::{download_model, ClapModel, DeviceConfig};

    info!(model = %config.model.name, "Downloading/loading CLAP model...");

    match download_model(&config.model.name, None).await {
        Ok(paths) => {
            info!(?paths, "Model files ready, loading into ONNX Runtime...");
            let device_config = DeviceConfig {
                cuda: config.model.enable_cuda,
                rocm: config.model.enable_rocm,
                coreml: config.model.enable_coreml,
                directml: config.model.enable_directml,
                openvino: config.model.enable_openvino,
            };
            match tokio::task::spawn_blocking(move || {
                ClapModel::load_with_config(&paths, device_config)
            })
            .await
            {
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

#[cfg(all(
    feature = "inference",
    any(feature = "storage", feature = "storage-file")
))]
async fn connect_storage(config: &AppConfig) -> Option<server::BoxedStorage> {
    use insight_sidecar::config::StorageMode;
    use insight_sidecar::storage::VectorStorage;

    if !config.storage.enabled {
        info!("Storage disabled in configuration");
        return None;
    }

    match config.storage.mode {
        #[cfg(feature = "storage-file")]
        StorageMode::File => {
            use insight_sidecar::storage::UsearchStorage;
            use std::path::PathBuf;

            let data_dir = PathBuf::from(&config.storage.data_dir);
            info!(?data_dir, "Initializing file-based usearch storage...");

            match UsearchStorage::new(data_dir) {
                Ok(storage) => {
                    if let Err(e) = storage.initialize().await {
                        error!("Failed to initialize usearch storage: {e}");
                        warn!("Starting without vector storage capability");
                        return None;
                    }
                    info!("usearch file storage initialized");
                    Some(Box::new(storage) as server::BoxedStorage)
                }
                Err(e) => {
                    error!("Failed to create usearch storage: {e}");
                    warn!("Starting without vector storage capability");
                    None
                }
            }
        }
        #[cfg(not(feature = "storage-file"))]
        StorageMode::File => {
            error!("File storage requested but 'storage-file' feature not enabled");
            warn!("Starting without vector storage capability");
            None
        }

        #[cfg(feature = "storage")]
        StorageMode::Qdrant => {
            use insight_sidecar::storage::QdrantStorage;

            info!(url = %config.storage.url, "Connecting to Qdrant...");

            match QdrantStorage::new(
                &config.storage.url,
                config.storage.api_key.clone(),
                config.storage.collection_prefix.clone(),
            )
            .await
            {
                Ok(storage) => {
                    if let Err(e) = storage.initialize().await {
                        error!("Failed to initialize storage collections: {e}");
                        warn!("Starting without vector storage capability");
                        return None;
                    }
                    info!("Qdrant storage connected and initialized");
                    Some(Box::new(storage) as server::BoxedStorage)
                }
                Err(e) => {
                    error!("Failed to connect to Qdrant: {e}");
                    warn!("Starting without vector storage capability");
                    None
                }
            }
        }
        #[cfg(not(feature = "storage"))]
        StorageMode::Qdrant => {
            error!("Qdrant storage requested but 'storage' feature not enabled");
            warn!("Starting without vector storage capability");
            None
        }
    }
}

/// Create application state with inference only (no storage features)
#[cfg(all(
    feature = "inference",
    not(any(feature = "storage", feature = "storage-file"))
))]
async fn create_app_state(config: AppConfig) -> server::AppState {
    use insight_sidecar::inference::{download_model, ClapModel, DeviceConfig};

    let model_id = config.model.name.clone();
    info!(model = %config.model.name, "Downloading/loading CLAP model...");

    match download_model(&config.model.name, None).await {
        Ok(paths) => {
            info!(?paths, "Model files ready, loading into ONNX Runtime...");
            let device_config = DeviceConfig {
                cuda: config.model.enable_cuda,
                rocm: config.model.enable_rocm,
                coreml: config.model.enable_coreml,
                directml: config.model.enable_directml,
                openvino: config.model.enable_openvino,
            };
            match tokio::task::spawn_blocking(move || {
                ClapModel::load_with_config(&paths, device_config)
            })
            .await
            {
                Ok(Ok(model)) => {
                    info!(device = %model.device(), "CLAP model loaded successfully");
                    let state = server::AppState::with_model(config, model, model_id);
                    // Initialize mood classifier in a blocking task
                    info!("Initializing mood classifier...");
                    if let Err(e) = state.init_classifier().await {
                        error!("Failed to initialize mood classifier: {e}");
                    } else {
                        info!("Mood classifier initialized");
                    }
                    state
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

/// Create application state with storage only (no inference)
#[cfg(all(
    any(feature = "storage", feature = "storage-file"),
    not(feature = "inference")
))]
async fn create_app_state(config: AppConfig) -> server::AppState {
    use insight_sidecar::config::StorageMode;
    use insight_sidecar::storage::VectorStorage;

    if !config.storage.enabled {
        info!("Storage disabled in configuration");
        return server::AppState::new(config);
    }

    let storage: Option<server::BoxedStorage> = match config.storage.mode {
        #[cfg(feature = "storage-file")]
        StorageMode::File => {
            use insight_sidecar::storage::UsearchStorage;
            use std::path::PathBuf;

            let data_dir = PathBuf::from(&config.storage.data_dir);
            info!(?data_dir, "Initializing file-based usearch storage...");

            match UsearchStorage::new(data_dir) {
                Ok(storage) => {
                    if let Err(e) = storage.initialize().await {
                        error!("Failed to initialize usearch storage: {e}");
                        None
                    } else {
                        info!("usearch file storage initialized");
                        Some(Box::new(storage) as server::BoxedStorage)
                    }
                }
                Err(e) => {
                    error!("Failed to create usearch storage: {e}");
                    None
                }
            }
        }
        #[cfg(not(feature = "storage-file"))]
        StorageMode::File => {
            error!("File storage requested but 'storage-file' feature not enabled");
            None
        }

        #[cfg(feature = "storage")]
        StorageMode::Qdrant => {
            use insight_sidecar::storage::QdrantStorage;

            info!(url = %config.storage.url, "Connecting to Qdrant...");

            match QdrantStorage::new(
                &config.storage.url,
                config.storage.api_key.clone(),
                config.storage.collection_prefix.clone(),
            )
            .await
            {
                Ok(storage) => {
                    if let Err(e) = storage.initialize().await {
                        error!("Failed to initialize storage collections: {e}");
                        None
                    } else {
                        info!("Qdrant storage connected and initialized");
                        Some(Box::new(storage) as server::BoxedStorage)
                    }
                }
                Err(e) => {
                    error!("Failed to connect to Qdrant: {e}");
                    None
                }
            }
        }
        #[cfg(not(feature = "storage"))]
        StorageMode::Qdrant => {
            error!("Qdrant storage requested but 'storage' feature not enabled");
            None
        }
    };

    match storage {
        Some(s) => server::AppState::with_storage(config, s),
        None => {
            warn!("Starting without vector storage capability");
            server::AppState::new(config)
        }
    }
}

/// Create application state without optional features
#[cfg(not(any(feature = "inference", feature = "storage", feature = "storage-file")))]
#[allow(clippy::unused_async)]
async fn create_app_state(config: AppConfig) -> server::AppState {
    info!("No optional features enabled");
    server::AppState::new(config)
}

/// Initialize the tracing subscriber for logging
fn init_logging(
    verbose: u8,
    quiet: bool,
    log_file: bool,
    log_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let level = if quiet {
        "warn"
    } else {
        match verbose {
            0 => "insight_sidecar=info,tower_http=info",
            1 => "insight_sidecar=debug,tower_http=debug",
            _ => "insight_sidecar=trace,tower_http=trace",
        }
    };

    let env_filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| level.into());

    let registry = tracing_subscriber::registry().with(env_filter);

    if log_file {
        // Determine log directory
        let log_path = log_dir.unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("insight-sidecar")
                .join("logs")
        });

        // Create log directory if it doesn't exist
        std::fs::create_dir_all(&log_path).with_context(|| {
            format!("Failed to create log directory: {}", log_path.display())
        })?;

        let log_file_path = log_path.join("insight-sidecar.log");

        // Rotate on startup: insight-sidecar.log -> insight-sidecar.log.1
        if log_file_path.exists() {
            let backup_path = log_path.join("insight-sidecar.log.1");
            if backup_path.exists() {
                let _ = std::fs::remove_file(&backup_path);
            }
            let _ = std::fs::rename(&log_file_path, &backup_path);
        }

        // Create file appender (no rotation, we handle it manually)
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file_path)
            .with_context(|| {
                format!("Failed to open log file: {}", log_file_path.display())
            })?;

        let (non_blocking, _guard) = tracing_appender::non_blocking(file);

        // Leak the guard to keep it alive for the program lifetime
        Box::leak(Box::new(_guard));

        // Dual output: console + file
        registry
            .with(tracing_subscriber::fmt::layer())
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(non_blocking)
                    .with_ansi(false), // No ANSI colors in file
            )
            .init();

        info!("File logging enabled at: {}", log_file_path.display());
    } else {
        // Console only
        registry.with(tracing_subscriber::fmt::layer()).init();
    }

    Ok(())
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
