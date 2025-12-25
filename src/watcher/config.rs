//! Configuration types for the folder watcher.

use serde::{Deserialize, Serialize};

/// Configuration for the folder watcher service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherConfig {
    /// Enable/disable the watcher service
    #[serde(default)]
    pub enabled: bool,

    /// Folders to watch for audio files
    #[serde(default)]
    pub folders: Vec<WatchedFolder>,

    /// Debounce interval for file events in milliseconds.
    /// Multiple rapid events for the same file will be coalesced.
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,

    /// Number of concurrent file processing tasks.
    /// Higher values process files faster but use more resources.
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,

    /// Whether to perform an initial scan of all folders on startup.
    /// If false, only new/modified files will be processed.
    #[serde(default = "default_true")]
    pub scan_on_startup: bool,

    /// File extensions to process. Empty means all supported extensions.
    /// Example: ["mp3", "flac", "wav"]
    #[serde(default)]
    pub extensions: Vec<String>,

    /// Retry configuration for failed file processing
    #[serde(default)]
    pub retry: RetryConfig,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            folders: Vec::new(),
            debounce_ms: default_debounce_ms(),
            concurrency: default_concurrency(),
            scan_on_startup: default_true(),
            extensions: Vec::new(),
            retry: RetryConfig::default(),
        }
    }
}

/// Configuration for a single watched folder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchedFolder {
    /// Absolute path to the folder to watch
    pub path: String,

    /// Whether to watch subdirectories recursively
    #[serde(default = "default_true")]
    pub recursive: bool,

    /// Optional human-readable label for this folder
    #[serde(default)]
    pub label: Option<String>,

    /// Whether this folder is currently enabled for watching
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl WatchedFolder {
    /// Create a new watched folder with default settings
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            recursive: true,
            label: None,
            enabled: true,
        }
    }

    /// Set the label for this folder
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set whether to watch recursively
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }
}

/// Configuration for retry behavior on failed file processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts for failed files
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Initial backoff delay in seconds before first retry
    #[serde(default = "default_retry_delay")]
    pub initial_delay_s: u64,

    /// Exponential backoff multiplier for subsequent retries
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            initial_delay_s: default_retry_delay(),
            backoff_multiplier: default_backoff_multiplier(),
        }
    }
}

// Default value functions
fn default_debounce_ms() -> u64 {
    500
}

fn default_concurrency() -> usize {
    4
}

fn default_true() -> bool {
    true
}

fn default_max_retries() -> u32 {
    3
}

fn default_retry_delay() -> u64 {
    5
}

fn default_backoff_multiplier() -> f32 {
    2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_watcher_config() {
        let config = WatcherConfig::default();
        assert!(!config.enabled);
        assert!(config.folders.is_empty());
        assert_eq!(config.debounce_ms, 500);
        assert_eq!(config.concurrency, 4);
        assert!(config.scan_on_startup);
    }

    #[test]
    fn test_watched_folder_builder() {
        let folder = WatchedFolder::new("/music")
            .with_label("Music Library")
            .with_recursive(true);

        assert_eq!(folder.path, "/music");
        assert_eq!(folder.label, Some("Music Library".to_string()));
        assert!(folder.recursive);
        assert!(folder.enabled);
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_delay_s, 5);
        assert_eq!(config.backoff_multiplier, 2.0);
    }
}
