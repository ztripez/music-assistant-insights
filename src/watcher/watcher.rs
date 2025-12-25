//! File system watcher using notify.

use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{new_debouncer, DebouncedEventKind, Debouncer};
use tokio::sync::mpsc;

use super::config::{WatchedFolder, WatcherConfig};
use super::{is_audio_file, WatcherError};

/// File system event types
#[derive(Debug, Clone)]
pub enum FileEvent {
    /// File was created or modified
    Created(PathBuf),
    /// File was removed
    Removed(PathBuf),
}

/// File system watcher for detecting changes in watched folders
pub struct FolderWatcher {
    /// The debouncer wrapping the watcher
    _debouncer: Debouncer<RecommendedWatcher>,
    /// Channel receiver for file events
    event_rx: mpsc::UnboundedReceiver<FileEvent>,
}

impl FolderWatcher {
    /// Create a new folder watcher
    pub fn new(config: &WatcherConfig) -> Result<Self, WatcherError> {
        let (tx, rx) = mpsc::unbounded_channel();
        let debounce_duration = Duration::from_millis(config.debounce_ms);

        // Create a standard mpsc channel for notify
        let (notify_tx, notify_rx) = std_mpsc::channel();

        // Create the debouncer
        let debouncer = new_debouncer(debounce_duration, notify_tx)
            .map_err(|e| WatcherError::NotifyError(e.to_string()))?;

        // Spawn a thread to forward events from notify to tokio channel
        let tx_clone = tx.clone();
        std::thread::spawn(move || {
            while let Ok(events) = notify_rx.recv() {
                match events {
                    Ok(events) => {
                        for event in events {
                            // Filter for audio files only
                            if !is_audio_file(&event.path) {
                                continue;
                            }

                            let file_event = match event.kind {
                                DebouncedEventKind::Any
                                | DebouncedEventKind::AnyContinuous => {
                                    FileEvent::Created(event.path)
                                }
                                _ => continue, // Skip unknown event kinds
                            };

                            if tx_clone.send(file_event).is_err() {
                                // Receiver dropped, exit thread
                                break;
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!(error = %error, "File watcher error");
                    }
                }
            }
        });

        Ok(Self {
            _debouncer: debouncer,
            event_rx: rx,
        })
    }

    /// Start watching a folder
    pub fn watch(&mut self, folder: &WatchedFolder) -> Result<(), WatcherError> {
        let mode = if folder.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        let path = Path::new(&folder.path);
        if !path.exists() {
            return Err(WatcherError::FileNotFound(path.to_path_buf()));
        }

        self._debouncer
            .watcher()
            .watch(path, mode)
            .map_err(|e| WatcherError::NotifyError(e.to_string()))?;

        tracing::info!(
            path = %folder.path,
            recursive = folder.recursive,
            "Started watching folder"
        );

        Ok(())
    }

    /// Stop watching a folder
    pub fn unwatch(&mut self, path: &str) -> Result<(), WatcherError> {
        self._debouncer
            .watcher()
            .unwatch(Path::new(path))
            .map_err(|e| WatcherError::NotifyError(e.to_string()))?;

        tracing::info!(path = %path, "Stopped watching folder");
        Ok(())
    }

    /// Get the next file event (async)
    pub async fn next_event(&mut self) -> Option<FileEvent> {
        self.event_rx.recv().await
    }

    /// Try to get a file event without blocking
    pub fn try_next_event(&mut self) -> Option<FileEvent> {
        self.event_rx.try_recv().ok()
    }
}
