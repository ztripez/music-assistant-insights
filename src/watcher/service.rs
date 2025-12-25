//! Background watcher service.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::inference::ClapModel;
use crate::storage::VectorStorage;

use super::config::{WatchedFolder, WatcherConfig};
use super::processor::TrackProcessor;
use super::scanner::FolderScanner;
use super::state::{FileRegistry, FolderState, ScanProgress, WatcherState, WatcherStats, WatcherStatus};
use super::watcher::{FileEvent, FolderWatcher};
use super::WatcherError;

/// Commands that can be sent to the watcher service
#[derive(Debug, Clone)]
pub enum WatcherCommand {
    /// Start the watcher
    Start,
    /// Stop the watcher
    Stop,
    /// Pause file processing
    Pause,
    /// Resume file processing
    Resume,
    /// Trigger a full scan of all folders
    TriggerScan { force: bool },
    /// Add a new folder to watch
    AddFolder(WatchedFolder),
    /// Remove a folder from watching
    RemoveFolder(String),
    /// Process a specific file
    ProcessFile(PathBuf),
}

/// Events emitted by the watcher service
#[derive(Debug, Clone)]
pub enum WatcherEvent {
    /// Watcher started
    Started,
    /// Watcher stopped
    Stopped,
    /// Watcher paused
    Paused,
    /// Watcher resumed
    Resumed,
    /// Scan started
    ScanStarted {
        folder: String,
        total_files: usize,
    },
    /// Scan progress update
    ScanProgress {
        folder: String,
        processed: usize,
        total: usize,
    },
    /// Scan completed
    ScanCompleted {
        folder: String,
        succeeded: usize,
        failed: usize,
    },
    /// File processed successfully
    FileProcessed {
        path: PathBuf,
        track_id: String,
    },
    /// File processing failed
    FileError {
        path: PathBuf,
        error: String,
    },
    /// Folder added
    FolderAdded(String),
    /// Folder removed
    FolderRemoved(String),
}

/// Background watcher service
pub struct WatcherService {
    config: Arc<RwLock<WatcherConfig>>,
    state: Arc<RwLock<WatcherState>>,
    registry: Arc<RwLock<FileRegistry>>,
    model: Arc<RwLock<Option<Arc<ClapModel>>>>,
    storage: Option<Arc<dyn VectorStorage + Send + Sync>>,

    /// Channel for sending commands to the service
    command_tx: Option<mpsc::Sender<WatcherCommand>>,
    /// Channel for broadcasting events from the service
    event_tx: broadcast::Sender<WatcherEvent>,

    /// Handle to the background task
    task_handle: Option<JoinHandle<()>>,
}

impl WatcherService {
    /// Create a new watcher service
    pub fn new(
        config: WatcherConfig,
        model: Arc<RwLock<Option<Arc<ClapModel>>>>,
        storage: Option<Arc<dyn VectorStorage + Send + Sync>>,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(256);

        // Initialize state from config
        let folders: Vec<FolderState> = config
            .folders
            .iter()
            .map(|f| FolderState::new(&f.path, f.label.clone(), f.enabled))
            .collect();

        let state = WatcherState {
            status: WatcherStatus::Stopped,
            folders,
            current_scan: None,
            stats: WatcherStats::default(),
            last_error: None,
        };

        Self {
            config: Arc::new(RwLock::new(config)),
            state: Arc::new(RwLock::new(state)),
            registry: Arc::new(RwLock::new(FileRegistry::new())),
            model,
            storage,
            command_tx: None,
            event_tx,
            task_handle: None,
        }
    }

    /// Get the current state
    pub async fn state(&self) -> WatcherState {
        self.state.read().await.clone()
    }

    /// Subscribe to watcher events
    pub fn subscribe(&self) -> broadcast::Receiver<WatcherEvent> {
        self.event_tx.subscribe()
    }

    /// Send a command to the watcher
    pub async fn send_command(&self, cmd: WatcherCommand) -> Result<(), WatcherError> {
        if let Some(ref tx) = self.command_tx {
            tx.send(cmd)
                .await
                .map_err(|e| WatcherError::ChannelError(e.to_string()))?;
            Ok(())
        } else {
            Err(WatcherError::NotStarted)
        }
    }

    /// Start the watcher service
    pub async fn start(&mut self) -> Result<(), WatcherError> {
        // Check if already running
        {
            let state = self.state.read().await;
            if state.status == WatcherStatus::Running || state.status == WatcherStatus::Scanning {
                return Err(WatcherError::AlreadyRunning);
            }
        }

        // Update status
        {
            let mut state = self.state.write().await;
            state.status = WatcherStatus::Starting;
        }

        let config = self.config.read().await.clone();
        let state = self.state.clone();
        let registry = self.registry.clone();
        let model = self.model.clone();
        let storage = self.storage.clone();
        let event_tx = self.event_tx.clone();

        let (cmd_tx, mut cmd_rx) = mpsc::channel::<WatcherCommand>(100);
        self.command_tx = Some(cmd_tx);

        // Spawn the background task
        let handle = tokio::spawn(async move {
            info!("Watcher service starting");

            // Create file watcher
            let mut watcher = match FolderWatcher::new(&config) {
                Ok(w) => w,
                Err(e) => {
                    error!(error = %e, "Failed to create file watcher");
                    let mut state = state.write().await;
                    state.status = WatcherStatus::Error;
                    state.last_error = Some(e.to_string());
                    return;
                }
            };

            // Setup watched folders
            for folder in &config.folders {
                if folder.enabled {
                    if let Err(e) = watcher.watch(folder) {
                        warn!(folder = %folder.path, error = %e, "Failed to watch folder");
                    }
                }
            }

            // Create processor
            let processor = TrackProcessor::new(model, storage);

            // Create scanner
            let scanner = FolderScanner::new(config.extensions.clone());

            // Update status to running
            {
                let mut state = state.write().await;
                state.status = WatcherStatus::Running;
            }
            let _ = event_tx.send(WatcherEvent::Started);

            // Initial scan if configured
            if config.scan_on_startup {
                Self::perform_scan(
                    &config,
                    &state,
                    &registry,
                    &scanner,
                    &processor,
                    &event_tx,
                    false,
                )
                .await;
            }

            // Main event loop
            let mut paused = false;

            loop {
                tokio::select! {
                    // Handle commands
                    Some(cmd) = cmd_rx.recv() => {
                        match cmd {
                            WatcherCommand::Stop => {
                                info!("Stopping watcher service");
                                break;
                            }
                            WatcherCommand::Pause => {
                                paused = true;
                                let mut state = state.write().await;
                                state.status = WatcherStatus::Paused;
                                let _ = event_tx.send(WatcherEvent::Paused);
                                info!("Watcher paused");
                            }
                            WatcherCommand::Resume => {
                                paused = false;
                                let mut state = state.write().await;
                                state.status = WatcherStatus::Running;
                                let _ = event_tx.send(WatcherEvent::Resumed);
                                info!("Watcher resumed");
                            }
                            WatcherCommand::TriggerScan { force } => {
                                if !paused {
                                    Self::perform_scan(
                                        &config,
                                        &state,
                                        &registry,
                                        &scanner,
                                        &processor,
                                        &event_tx,
                                        force,
                                    )
                                    .await;
                                }
                            }
                            WatcherCommand::AddFolder(folder) => {
                                if let Err(e) = watcher.watch(&folder) {
                                    warn!(folder = %folder.path, error = %e, "Failed to add folder");
                                } else {
                                    let mut state = state.write().await;
                                    state.folders.push(FolderState::new(&folder.path, folder.label.clone(), folder.enabled));
                                    let _ = event_tx.send(WatcherEvent::FolderAdded(folder.path));
                                }
                            }
                            WatcherCommand::RemoveFolder(path) => {
                                if let Err(e) = watcher.unwatch(&path) {
                                    warn!(path = %path, error = %e, "Failed to remove folder");
                                } else {
                                    let mut state = state.write().await;
                                    state.folders.retain(|f| f.path != path);
                                    let _ = event_tx.send(WatcherEvent::FolderRemoved(path));
                                }
                            }
                            WatcherCommand::ProcessFile(path) => {
                                if !paused {
                                    Self::process_single_file(&path, &config, &registry, &processor, &state, &event_tx).await;
                                }
                            }
                            WatcherCommand::Start => {
                                // Already running, ignore
                            }
                        }
                    }

                    // Handle file events
                    Some(event) = watcher.next_event() => {
                        if paused {
                            continue;
                        }

                        match event {
                            FileEvent::Created(path) => {
                                debug!(path = ?path, "File created/modified");
                                Self::process_single_file(&path, &config, &registry, &processor, &state, &event_tx).await;
                            }
                            FileEvent::Removed(path) => {
                                debug!(path = ?path, "File removed");
                                // TODO: Handle file removal (delete from storage)
                                let mut reg = registry.write().await;
                                if let Some(track_id) = reg.unregister_file(&path) {
                                    info!(track_id = %track_id, path = ?path, "Unregistered removed file");
                                }
                            }
                        }
                    }
                }
            }

            // Cleanup
            {
                let mut state = state.write().await;
                state.status = WatcherStatus::Stopped;
            }
            let _ = event_tx.send(WatcherEvent::Stopped);
            info!("Watcher service stopped");
        });

        self.task_handle = Some(handle);
        Ok(())
    }

    /// Stop the watcher service
    pub async fn stop(&mut self) -> Result<(), WatcherError> {
        self.send_command(WatcherCommand::Stop).await?;

        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }

        self.command_tx = None;
        Ok(())
    }

    /// Perform a scan of all folders
    async fn perform_scan(
        config: &WatcherConfig,
        state: &Arc<RwLock<WatcherState>>,
        registry: &Arc<RwLock<FileRegistry>>,
        scanner: &FolderScanner,
        processor: &TrackProcessor,
        event_tx: &broadcast::Sender<WatcherEvent>,
        force: bool,
    ) {
        // Update status
        {
            let mut state = state.write().await;
            state.status = WatcherStatus::Scanning;
        }

        let reg = registry.read().await;
        let files = scanner.scan_all(&config.folders, &reg, force);
        drop(reg);

        let total = files.len();
        info!(total_files = total, force = force, "Starting scan");

        // Initialize progress
        {
            let mut state = state.write().await;
            state.current_scan = Some(ScanProgress::new("all", total));
        }

        let _ = event_tx.send(WatcherEvent::ScanStarted {
            folder: "all".to_string(),
            total_files: total,
        });

        let mut succeeded = 0;
        let mut failed = 0;

        for (i, scanned_file) in files.iter().enumerate() {
            // Update progress
            {
                let mut state = state.write().await;
                if let Some(ref mut progress) = state.current_scan {
                    progress.current_file = Some(scanned_file.path.to_string_lossy().to_string());
                }
            }

            match processor.process_file(&scanned_file.path, &scanned_file.base_path).await {
                Ok(result) => {
                    // Register in file registry
                    let mut reg = registry.write().await;
                    reg.register_file(scanned_file.path.clone(), result.track_id.clone());

                    let _ = event_tx.send(WatcherEvent::FileProcessed {
                        path: scanned_file.path.clone(),
                        track_id: result.track_id,
                    });
                    succeeded += 1;
                }
                Err(e) => {
                    warn!(path = ?scanned_file.path, error = %e, "Failed to process file");
                    let _ = event_tx.send(WatcherEvent::FileError {
                        path: scanned_file.path.clone(),
                        error: e.to_string(),
                    });
                    failed += 1;
                }
            }

            // Update progress
            {
                let mut state = state.write().await;
                if let Some(ref mut progress) = state.current_scan {
                    progress.update(failed == 0);
                }
            }

            // Emit progress event periodically
            if (i + 1) % 10 == 0 || i + 1 == total {
                let _ = event_tx.send(WatcherEvent::ScanProgress {
                    folder: "all".to_string(),
                    processed: i + 1,
                    total,
                });
            }
        }

        // Finalize
        {
            let mut state = state.write().await;
            state.current_scan = None;
            state.status = WatcherStatus::Running;
            state.stats.total_files_processed += succeeded as u64;
            state.stats.total_files_failed += failed as u64;
            state.stats.total_scans += 1;
        }

        let _ = event_tx.send(WatcherEvent::ScanCompleted {
            folder: "all".to_string(),
            succeeded,
            failed,
        });

        info!(succeeded = succeeded, failed = failed, "Scan completed");
    }

    /// Process a single file
    async fn process_single_file(
        path: &PathBuf,
        config: &WatcherConfig,
        registry: &Arc<RwLock<FileRegistry>>,
        processor: &TrackProcessor,
        state: &Arc<RwLock<WatcherState>>,
        event_tx: &broadcast::Sender<WatcherEvent>,
    ) {
        // Find the base path for this file from configured folders
        let base_path = config
            .folders
            .iter()
            .filter(|f| f.enabled)
            .find(|f| path.starts_with(&f.path))
            .map(|f| PathBuf::from(&f.path))
            .unwrap_or_else(|| {
                // Fallback: use parent directory
                path.parent().map(|p| p.to_path_buf()).unwrap_or_default()
            });

        match processor.process_file(path, &base_path).await {
            Ok(result) => {
                // Register in file registry
                let mut reg = registry.write().await;
                reg.register_file(path.clone(), result.track_id.clone());

                // Update stats
                {
                    let mut state = state.write().await;
                    state.stats.total_files_processed += 1;
                }

                let _ = event_tx.send(WatcherEvent::FileProcessed {
                    path: path.clone(),
                    track_id: result.track_id,
                });
            }
            Err(e) => {
                warn!(path = ?path, error = %e, "Failed to process file");

                // Update stats
                {
                    let mut state = state.write().await;
                    state.stats.total_files_failed += 1;
                }

                let _ = event_tx.send(WatcherEvent::FileError {
                    path: path.clone(),
                    error: e.to_string(),
                });
            }
        }
    }
}

impl Drop for WatcherService {
    fn drop(&mut self) {
        if let Some(handle) = self.task_handle.take() {
            handle.abort();
        }
    }
}
