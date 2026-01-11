//! Shared types for the insight sidecar API.
//!
//! These types are used across the application for request/response handling
//! and internal data representation.
//!
//! # Module Organization
//!
//! API types are split into domain-specific modules:
//! - [`tracks`] - Track upsert, search, delete, and batch operations
//! - [`embed`] - Direct embedding generation from text and audio
//! - [`models`] - Model management (list, download, load, delete)
//! - [`mood`] - Mood classification operations
//! - [`taste`] - Taste profiles and recommendations
//!
//! The [`api`] module re-exports all types for backward compatibility.

pub mod api;
pub mod embed;
pub mod models;
pub mod mood;
pub mod taste;
pub mod tracks;

use serde::{Deserialize, Serialize};

pub use api::*;

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub version: String,
    #[serde(default)]
    pub model_loaded: bool,
    #[serde(default)]
    pub storage_ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Configuration response (subset of config safe to expose)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigResponse {
    pub model: ModelInfo,
    pub audio: AudioInfo,
    pub server: ServerInfo,
    pub storage: StorageInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub cuda_enabled: bool,
    pub loaded: bool,
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInfo {
    pub window_size_s: f32,
    pub hop_size_s: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    pub url: String,
    pub enabled: bool,
    pub connected: bool,
    #[serde(default)]
    pub mode: String,
}

// ============================================================================
// Management API Types
// ============================================================================

/// Model status for management API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    /// Model not downloaded yet
    NotDownloaded,
    /// Model is currently downloading
    Downloading,
    /// Model downloaded but not loaded
    Downloaded,
    /// Model loaded and ready for inference
    Loaded,
    /// Model loading failed
    Failed,
}

/// Download status for tracking downloads
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    /// Download is queued
    Pending,
    /// Download in progress
    Downloading,
    /// Download completed successfully
    Completed,
    /// Download failed
    Failed,
    /// Download was cancelled
    Cancelled,
}

/// Detailed model information for management API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetail {
    /// Model ID (e.g., "Xenova/clap-htsat-unfused")
    pub model_id: String,
    /// Display name
    pub name: String,
    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Current status
    pub status: ModelStatus,
    /// Estimated file size in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_size_bytes: Option<u64>,
    /// Actual file size in bytes (if downloaded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_size_bytes: Option<u64>,
    /// Cache path (if downloaded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_path: Option<String>,
    /// Whether this is a recommended/tested model
    #[serde(default)]
    pub recommended: bool,
    /// Whether this is the currently loaded model
    #[serde(default)]
    pub is_current: bool,
    /// Device it's loaded on (if loaded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// Download progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Unique download ID
    pub download_id: String,
    /// Model being downloaded
    pub model_id: String,
    /// Current status
    pub status: DownloadStatus,
    /// Bytes downloaded so far
    pub bytes_downloaded: u64,
    /// Total bytes (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<u64>,
    /// Progress percentage (0-100)
    pub progress_percent: f32,
    /// Unix timestamp when download started
    pub started_at: i64,
    /// Unix timestamp when download completed/failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Current file being downloaded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_file: Option<String>,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Storage mode (file or qdrant)
    pub mode: String,
    /// Whether storage is connected
    pub connected: bool,
    /// Number of tracks in text collection
    pub text_collection_count: u64,
    /// Number of tracks in audio collection
    pub audio_collection_count: u64,
    /// Estimated total unique tracks
    pub total_tracks: u64,
}

/// Comprehensive system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    /// Version string
    pub version: String,
    /// Health status
    pub health: HealthStatus,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Model information (if loaded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelDetail>,
    /// Storage statistics
    pub storage: StorageStats,
    /// Enabled feature flags
    pub features: Vec<String>,
}

// ============================================================================
// Taste Profile Types
// ============================================================================

/// Type of taste profile
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(tag = "type", content = "value")]
pub enum ProfileType {
    /// Global profile across all listening
    #[default]
    Global,
    /// Mood-based profile (energetic, peaceful, aggressive, melancholic)
    Mood(String),
    /// Context-based profile (weekday_morning, weekend_evening, etc.)
    Context(String),
}

impl std::fmt::Display for ProfileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileType::Global => write!(f, "global"),
            ProfileType::Mood(mood) => write!(f, "mood:{}", mood),
            ProfileType::Context(ctx) => write!(f, "context:{}", ctx),
        }
    }
}

/// User taste profile with embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasteProfile {
    /// User ID this profile belongs to
    pub user_id: String,
    /// Type of profile
    pub profile_type: ProfileType,
    /// 512-dimensional taste vector (normalized)
    pub embedding: Vec<f32>,
    /// Number of tracks that contributed to this profile
    pub track_count: u32,
    /// Confidence score (0.0-1.0), higher with more data
    pub confidence: f32,
    /// Unix timestamp when profile was last updated
    pub updated_at: i64,
}

/// Type of user interaction signal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    /// Track played to completion
    FullPlay,
    /// Track partially played (>50%)
    PartialPlay,
    /// Track skipped early (<30s or <25% completion)
    Skip,
    /// Track played again within same day
    Repeat,
    /// Track explicitly liked/favorited
    Favorite,
    /// Track explicitly disliked
    Dislike,
    /// Track saved to playlist
    Save,
}

/// User interaction with a track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteraction {
    /// Track ID (MA item_id)
    pub track_id: String,
    /// Unix timestamp of interaction
    pub timestamp: i64,
    /// Type of signal
    pub signal_type: SignalType,
    /// Seconds of track played
    pub seconds_played: f32,
    /// Total track duration
    pub duration: f32,
}

/// Result of taste vector computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasteVector {
    /// 512-dimensional embedding (normalized)
    pub embedding: Vec<f32>,
    /// Number of tracks used
    pub track_count: u32,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Total weight sum (for debugging)
    pub total_weight: f32,
}
