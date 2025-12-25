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

/// Generate a track ID from file path relative to a base directory.
///
/// The track ID is the relative path from base_path to file_path, using
/// forward slashes as separators (for cross-platform consistency).
/// This matches how Music Assistant identifies local tracks.
///
/// # Example
/// ```ignore
/// let base = Path::new("/music");
/// let file = Path::new("/music/Artist/Album/song.mp3");
/// assert_eq!(generate_track_id(file, base), "Artist/Album/song.mp3");
/// ```
pub fn generate_track_id(file_path: &Path, base_path: &Path) -> String {
    // Get relative path from base
    let relative = file_path
        .strip_prefix(base_path)
        .unwrap_or(file_path);

    // Convert to string with forward slashes for cross-platform consistency
    let path_str = relative.to_string_lossy();

    // Normalize path separators to forward slashes
    path_str.replace('\\', "/")
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
    fn test_generate_track_id_relative_path() {
        let base = PathBuf::from("/music");
        let file = PathBuf::from("/music/Artist/Album/track.mp3");
        let id = generate_track_id(&file, &base);
        assert_eq!(id, "Artist/Album/track.mp3");
    }

    #[test]
    fn test_generate_track_id_deterministic() {
        let base = PathBuf::from("/music");
        let file = PathBuf::from("/music/artist/album/track.mp3");
        let id1 = generate_track_id(&file, &base);
        let id2 = generate_track_id(&file, &base);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_generate_track_id_different_paths() {
        let base = PathBuf::from("/music");
        let path1 = PathBuf::from("/music/artist1/track.mp3");
        let path2 = PathBuf::from("/music/artist2/track.mp3");
        let id1 = generate_track_id(&path1, &base);
        let id2 = generate_track_id(&path2, &base);
        assert_ne!(id1, id2);
        assert_eq!(id1, "artist1/track.mp3");
        assert_eq!(id2, "artist2/track.mp3");
    }

    #[test]
    fn test_generate_track_id_normalizes_slashes() {
        let base = PathBuf::from("/music");
        // Even with backslashes, output should use forward slashes
        let file = PathBuf::from("/music/Artist/Album/track.mp3");
        let id = generate_track_id(&file, &base);
        assert!(!id.contains('\\'));
        assert!(id.contains('/'));
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
