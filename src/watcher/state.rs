//! Watcher state management and progress tracking.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// Current status of the watcher service
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WatcherStatus {
    /// Watcher is not running
    Stopped,
    /// Watcher is starting up
    Starting,
    /// Watcher is running and monitoring folders
    Running,
    /// Watcher is paused (not processing new files)
    Paused,
    /// Watcher is performing an initial/full scan
    Scanning,
    /// Watcher encountered an error
    Error,
}

impl Default for WatcherStatus {
    fn default() -> Self {
        Self::Stopped
    }
}

impl std::fmt::Display for WatcherStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => write!(f, "stopped"),
            Self::Starting => write!(f, "starting"),
            Self::Running => write!(f, "running"),
            Self::Paused => write!(f, "paused"),
            Self::Scanning => write!(f, "scanning"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// Overall state of the watcher service
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WatcherState {
    /// Current status
    pub status: WatcherStatus,
    /// State of each watched folder
    pub folders: Vec<FolderState>,
    /// Current scan progress (if scanning)
    pub current_scan: Option<ScanProgress>,
    /// Cumulative statistics
    pub stats: WatcherStats,
    /// Last error message (if status is Error)
    pub last_error: Option<String>,
}

/// State of a single watched folder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderState {
    /// Path to the folder
    pub path: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Whether this folder is enabled
    pub enabled: bool,
    /// Unix timestamp of last successful scan
    pub last_scan: Option<i64>,
    /// Number of files in this folder
    pub file_count: usize,
    /// Number of files that failed processing
    pub error_count: usize,
}

impl FolderState {
    /// Create a new folder state
    pub fn new(path: impl Into<String>, label: Option<String>, enabled: bool) -> Self {
        Self {
            path: path.into(),
            label,
            enabled,
            last_scan: None,
            file_count: 0,
            error_count: 0,
        }
    }
}

/// Progress of an ongoing scan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanProgress {
    /// Folder being scanned
    pub folder: String,
    /// Total files to process
    pub total_files: usize,
    /// Files processed so far
    pub processed: usize,
    /// Successfully processed files
    pub succeeded: usize,
    /// Failed files
    pub failed: usize,
    /// Current file being processed
    pub current_file: Option<String>,
    /// Unix timestamp when scan started
    pub started_at: i64,
    /// Estimated remaining time in seconds
    pub estimated_remaining_s: Option<f32>,
}

impl ScanProgress {
    /// Create a new scan progress tracker
    pub fn new(folder: impl Into<String>, total_files: usize) -> Self {
        Self {
            folder: folder.into(),
            total_files,
            processed: 0,
            succeeded: 0,
            failed: 0,
            current_file: None,
            started_at: chrono::Utc::now().timestamp(),
            estimated_remaining_s: None,
        }
    }

    /// Update progress after processing a file
    pub fn update(&mut self, success: bool) {
        self.processed += 1;
        if success {
            self.succeeded += 1;
        } else {
            self.failed += 1;
        }

        // Calculate estimated remaining time
        if self.processed > 0 {
            let elapsed = chrono::Utc::now().timestamp() - self.started_at;
            if elapsed > 0 {
                let rate = self.processed as f32 / elapsed as f32;
                let remaining = self.total_files.saturating_sub(self.processed);
                self.estimated_remaining_s = Some(remaining as f32 / rate);
            }
        }
    }

    /// Get progress as a percentage
    pub fn percentage(&self) -> f32 {
        if self.total_files == 0 {
            100.0
        } else {
            (self.processed as f32 / self.total_files as f32) * 100.0
        }
    }
}

/// Cumulative statistics for the watcher
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WatcherStats {
    /// Total files processed since startup
    pub total_files_processed: u64,
    /// Total files that failed processing
    pub total_files_failed: u64,
    /// Total scans performed
    pub total_scans: u64,
    /// Uptime in seconds
    pub uptime_s: u64,
}

/// Registry of processed files for incremental scanning.
///
/// Tracks which files have been processed and their last modification time,
/// so we can skip unchanged files on subsequent scans.
#[derive(Debug, Default)]
pub struct FileRegistry {
    /// Map of file path -> (track_id, last_modified, file_size)
    files: HashMap<PathBuf, FileEntry>,
}

#[derive(Debug, Clone)]
struct FileEntry {
    track_id: String,
    last_modified: SystemTime,
    file_size: u64,
}

impl FileRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a file has changed since it was last registered
    pub fn is_file_changed(&self, path: &PathBuf) -> bool {
        if let Some(entry) = self.files.get(path) {
            if let Ok(metadata) = path.metadata() {
                let current_mod = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                let current_size = metadata.len();
                return current_mod != entry.last_modified || current_size != entry.file_size;
            }
        }
        // File not in registry or couldn't get metadata = treat as new/changed
        true
    }

    /// Register a file after successful processing
    pub fn register_file(&mut self, path: PathBuf, track_id: String) {
        if let Ok(metadata) = path.metadata() {
            let last_modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let file_size = metadata.len();
            self.files.insert(
                path,
                FileEntry {
                    track_id,
                    last_modified,
                    file_size,
                },
            );
        }
    }

    /// Remove a file from the registry
    pub fn unregister_file(&mut self, path: &PathBuf) -> Option<String> {
        self.files.remove(path).map(|e| e.track_id)
    }

    /// Get the track ID for a registered file
    pub fn get_track_id(&self, path: &PathBuf) -> Option<&str> {
        self.files.get(path).map(|e| e.track_id.as_str())
    }

    /// Get the number of registered files
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.files.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watcher_status_display() {
        assert_eq!(WatcherStatus::Running.to_string(), "running");
        assert_eq!(WatcherStatus::Scanning.to_string(), "scanning");
    }

    #[test]
    fn test_scan_progress() {
        let mut progress = ScanProgress::new("/music", 100);
        assert_eq!(progress.percentage(), 0.0);

        progress.update(true);
        progress.update(true);
        progress.update(false);

        assert_eq!(progress.processed, 3);
        assert_eq!(progress.succeeded, 2);
        assert_eq!(progress.failed, 1);
        assert_eq!(progress.percentage(), 3.0);
    }

    #[test]
    fn test_file_registry() {
        let mut registry = FileRegistry::new();
        assert!(registry.is_empty());

        // Create a real temporary file for testing
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_file_registry.mp3");
        std::fs::write(&path, b"test content").expect("Failed to create temp file");

        registry.register_file(path.clone(), "track-123".to_string());

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get_track_id(&path), Some("track-123"));

        let removed = registry.unregister_file(&path);
        assert_eq!(removed, Some("track-123".to_string()));
        assert!(registry.is_empty());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
