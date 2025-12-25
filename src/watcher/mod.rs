//! Background folder watcher for automatic music library ingestion.
//!
//! This module monitors configured directories for audio files, automatically
//! decodes them using symphonia, extracts metadata (ID3/Vorbis tags), generates
//! embeddings via CLAP, and stores them in the vector database.

mod config;
mod decoder;
mod metadata;
mod processor;
mod scanner;
mod service;
mod state;
mod watcher;

pub use config::{RetryConfig, WatchedFolder, WatcherConfig};
pub use decoder::{AudioDecoder, DecodedAudio};
pub use metadata::ExtractedMetadata;
pub use processor::TrackProcessor;
pub use scanner::FolderScanner;
pub use service::{WatcherCommand, WatcherEvent, WatcherService};
pub use state::{FileRegistry, FolderState, ScanProgress, WatcherState, WatcherStats, WatcherStatus};
pub use watcher::{FileEvent, FolderWatcher};

use std::path::Path;
use uuid::Uuid;

/// Namespace UUID for music-assistant-insights track IDs (DNS namespace)
const TRACK_ID_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
]);

/// Generate a deterministic track ID from a file path using UUID v5.
///
/// The track ID is stable across runs as long as the file path doesn't change.
/// Uses the canonical (absolute) path to ensure consistency.
pub fn generate_track_id(file_path: &Path) -> String {
    // Use canonical/absolute path for consistency
    let canonical = file_path
        .canonicalize()
        .unwrap_or_else(|_| file_path.to_path_buf());
    let path_str = canonical.to_string_lossy();

    let uuid = Uuid::new_v5(&TRACK_ID_NAMESPACE, path_str.as_bytes());
    uuid.to_string()
}

/// Supported audio file extensions
pub const AUDIO_EXTENSIONS: &[&str] = &[
    "mp3", "flac", "wav", "ogg", "m4a", "aac", "aiff", "aif", "wma", "opus", "wv", "ape", "alac",
];

/// Check if a file path has a supported audio extension
pub fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| AUDIO_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Watcher error types
#[derive(Debug, thiserror::Error)]
pub enum WatcherError {
    #[error("Watcher not started")]
    NotStarted,

    #[error("Watcher already running")]
    AlreadyRunning,

    #[error("File not found: {0}")]
    FileNotFound(std::path::PathBuf),

    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("Audio decode error: {0}")]
    DecodeError(String),

    #[error("No audio track found in file")]
    NoAudioTrack,

    #[error("Missing sample rate in audio file")]
    MissingSampleRate,

    #[error("Metadata extraction failed: {0}")]
    MetadataError(String),

    #[error("Embedding generation failed: {0}")]
    EmbeddingError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Notify error: {0}")]
    NotifyError(String),

    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Storage not available")]
    StorageNotAvailable,

    #[error("Channel send error: {0}")]
    ChannelError(String),

    #[error("Resampling error: {0}")]
    ResamplingError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_generate_track_id_deterministic() {
        let path = PathBuf::from("/music/artist/album/track.mp3");
        let id1 = generate_track_id(&path);
        let id2 = generate_track_id(&path);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_generate_track_id_different_paths() {
        let path1 = PathBuf::from("/music/artist1/track.mp3");
        let path2 = PathBuf::from("/music/artist2/track.mp3");
        let id1 = generate_track_id(&path1);
        let id2 = generate_track_id(&path2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_is_audio_file() {
        assert!(is_audio_file(Path::new("song.mp3")));
        assert!(is_audio_file(Path::new("song.FLAC")));
        assert!(is_audio_file(Path::new("song.wav")));
        assert!(is_audio_file(Path::new("song.ogg")));
        assert!(is_audio_file(Path::new("song.m4a")));
        assert!(!is_audio_file(Path::new("document.pdf")));
        assert!(!is_audio_file(Path::new("image.jpg")));
        assert!(!is_audio_file(Path::new("noextension")));
    }
}
