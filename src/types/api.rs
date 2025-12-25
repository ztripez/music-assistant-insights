//! API request and response types for track operations.

use serde::{Deserialize, Serialize};

#[cfg(feature = "storage")]
use crate::storage::{SearchFilter, SearchResult, TrackMetadata};

#[cfg(feature = "inference")]
use crate::inference::{AudioData, AudioFormat};

/// Request to upsert track embedding(s)
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRequest {
    /// Track ID (Music Assistant item_id)
    pub track_id: String,
    /// Track metadata
    pub metadata: TrackMetadataInput,
    /// Text embedding (512-dimensional)
    #[serde(default)]
    pub text_embedding: Option<Vec<f32>>,
    /// Audio embedding (512-dimensional)
    #[serde(default)]
    pub audio_embedding: Option<Vec<f32>>,
}

/// Input metadata for upsert operations
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMetadataInput {
    /// Track name
    pub name: String,
    /// Artist names
    #[serde(default)]
    pub artists: Vec<String>,
    /// Album name
    #[serde(default)]
    pub album: Option<String>,
    /// Genre tags
    #[serde(default)]
    pub genres: Vec<String>,
}

#[cfg(feature = "storage")]
impl TrackMetadataInput {
    /// Convert to storage TrackMetadata
    pub fn into_storage(self, track_id: String) -> TrackMetadata {
        let mut metadata = TrackMetadata::new(track_id, self.name)
            .with_artists(self.artists)
            .with_genres(self.genres);

        if let Some(album) = self.album {
            metadata = metadata.with_album(album);
        }

        metadata
    }
}

/// Response from upsert operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    /// Track ID that was upserted
    pub track_id: String,
    /// Whether text embedding was stored
    pub text_stored: bool,
    /// Whether audio embedding was stored
    pub audio_stored: bool,
}

/// Request to search for similar tracks
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Query embedding (512-dimensional)
    pub embedding: Vec<f32>,
    /// Collection to search: "text" or "audio"
    pub collection: String,
    /// Maximum number of results (default: 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Optional filter
    #[serde(default)]
    pub filter: Option<SearchFilterInput>,
}

#[cfg(feature = "storage")]
fn default_limit() -> usize {
    10
}

/// Input filter for search operations
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilterInput {
    /// Filter by artist names (any match)
    #[serde(default)]
    pub artists: Option<Vec<String>>,
    /// Filter by genres (any match)
    #[serde(default)]
    pub genres: Option<Vec<String>>,
    /// Filter by album name
    #[serde(default)]
    pub album: Option<String>,
    /// Exclude specific track IDs
    #[serde(default)]
    pub exclude_ids: Option<Vec<String>>,
}

#[cfg(feature = "storage")]
impl From<SearchFilterInput> for SearchFilter {
    fn from(input: SearchFilterInput) -> Self {
        let mut filter = SearchFilter::new();

        if let Some(artists) = input.artists {
            filter = filter.with_artists(artists);
        }
        if let Some(genres) = input.genres {
            filter = filter.with_genres(genres);
        }
        if let Some(album) = input.album {
            filter = filter.with_album(album);
        }
        if let Some(exclude_ids) = input.exclude_ids {
            filter = filter.exclude(exclude_ids);
        }

        filter
    }
}

/// Response from search operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results sorted by similarity
    pub results: Vec<SearchResult>,
    /// Number of results returned
    pub count: usize,
}

/// Request to delete a track's embeddings
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    /// Delete from text collection
    #[serde(default = "default_true")]
    pub text: bool,
    /// Delete from audio collection
    #[serde(default = "default_true")]
    pub audio: bool,
}

#[cfg(feature = "storage")]
fn default_true() -> bool {
    true
}

#[cfg(feature = "storage")]
impl Default for DeleteRequest {
    fn default() -> Self {
        Self {
            text: true,
            audio: true,
        }
    }
}

/// Response from delete operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    /// Track ID that was deleted
    pub track_id: String,
    /// Whether text embedding was deleted
    pub text_deleted: bool,
    /// Whether audio embedding was deleted
    pub audio_deleted: bool,
}

/// Response for get track operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTrackResponse {
    /// Track ID
    pub track_id: String,
    /// Track metadata (from either collection)
    pub metadata: Option<TrackMetadata>,
    /// Whether text embedding exists
    pub has_text: bool,
    /// Whether audio embedding exists
    pub has_audio: bool,
    /// Text embedding if requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_embedding: Option<Vec<f32>>,
    /// Audio embedding if requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_embedding: Option<Vec<f32>>,
}

#[cfg(all(test, feature = "storage"))]
mod tests {
    use super::*;

    #[test]
    fn test_upsert_request_serialize() {
        let req = UpsertRequest {
            track_id: "track_123".to_string(),
            metadata: TrackMetadataInput {
                name: "Test Song".to_string(),
                artists: vec!["Artist 1".to_string()],
                album: Some("Album".to_string()),
                genres: vec!["rock".to_string()],
            },
            text_embedding: Some(vec![0.1; 512]),
            audio_embedding: None,
        };

        let bytes = rmp_serde::to_vec(&req).unwrap();
        let decoded: UpsertRequest = rmp_serde::from_slice(&bytes).unwrap();

        assert_eq!(decoded.track_id, "track_123");
        assert!(decoded.text_embedding.is_some());
        assert!(decoded.audio_embedding.is_none());
    }

    #[test]
    fn test_search_filter_conversion() {
        let input = SearchFilterInput {
            artists: Some(vec!["Artist".to_string()]),
            genres: Some(vec!["rock".to_string()]),
            album: Some("Album".to_string()),
            exclude_ids: Some(vec!["track_1".to_string()]),
        };

        let filter: SearchFilter = input.into();

        assert!(filter.artists.is_some());
        assert!(filter.genres.is_some());
        assert!(filter.album.is_some());
        assert!(filter.exclude_ids.is_some());
    }
}

// ============================================================================
// Embedding generation types (inference feature)
// ============================================================================

/// Request to generate text embedding
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbedRequest {
    /// Raw text to embed (optional if metadata provided)
    #[serde(default)]
    pub text: Option<String>,
    /// Track metadata to format and embed (optional if text provided)
    #[serde(default)]
    pub metadata: Option<TextEmbedMetadata>,
}

/// Track metadata for text embedding generation
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbedMetadata {
    /// Track name
    pub name: String,
    /// Artist names
    #[serde(default)]
    pub artists: Vec<String>,
    /// Album name
    #[serde(default)]
    pub album: Option<String>,
    /// Genre tags
    #[serde(default)]
    pub genres: Vec<String>,
}

/// Response from text embedding generation
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbedResponse {
    /// Generated embedding (512-dimensional)
    pub embedding: Vec<f32>,
    /// Text that was embedded (for verification)
    pub text: String,
}

/// Request to generate audio embedding
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEmbedRequest {
    /// Audio format
    pub format: AudioFormat,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u8,
    /// Raw PCM bytes
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[cfg(feature = "inference")]
impl From<AudioEmbedRequest> for AudioData {
    fn from(req: AudioEmbedRequest) -> Self {
        AudioData {
            format: req.format,
            sample_rate: req.sample_rate,
            channels: req.channels,
            data: req.data,
        }
    }
}

/// Response from audio embedding generation
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEmbedResponse {
    /// Generated embedding (512-dimensional)
    pub embedding: Vec<f32>,
    /// Duration of audio in seconds
    pub duration_s: f32,
}

// ============================================================================
// Combined embedding + storage types (requires both inference and storage)
// ============================================================================

/// Request to generate text embedding from metadata and store it
#[cfg(all(feature = "inference", feature = "storage"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedTextAndStoreRequest {
    /// Track ID (Music Assistant item_id)
    pub track_id: String,
    /// Track metadata for embedding generation and storage
    pub metadata: EmbedTextAndStoreMetadata,
}

/// Track metadata for combined embed + store operation
#[cfg(all(feature = "inference", feature = "storage"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedTextAndStoreMetadata {
    /// Track name
    pub name: String,
    /// Artist names
    #[serde(default)]
    pub artists: Vec<String>,
    /// Album name
    #[serde(default)]
    pub album: Option<String>,
    /// Genre tags
    #[serde(default)]
    pub genres: Vec<String>,
}

/// Response from combined embed + store operation
#[cfg(all(feature = "inference", feature = "storage"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedTextAndStoreResponse {
    /// Track ID that was processed
    pub track_id: String,
    /// Whether the embedding was stored successfully
    pub stored: bool,
    /// The text that was embedded (for verification)
    pub text: String,
}

// ============================================================================
// Batch operation types (storage feature)
// ============================================================================

/// Request to batch upsert multiple track embeddings
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchUpsertRequest {
    /// List of tracks to upsert
    pub tracks: Vec<UpsertRequest>,
}

/// Response from batch upsert operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchUpsertResponse {
    /// Results for each track
    pub results: Vec<BatchUpsertResult>,
    /// Total number of tracks processed
    pub total: usize,
    /// Number of successful upserts
    pub succeeded: usize,
    /// Number of failed upserts
    pub failed: usize,
}

/// Result for a single track in batch upsert
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchUpsertResult {
    /// Track ID
    pub track_id: String,
    /// Whether the operation succeeded
    pub success: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Whether text embedding was stored
    pub text_stored: bool,
    /// Whether audio embedding was stored
    pub audio_stored: bool,
}

// ============================================================================
// Batch embed + store types (requires both inference and storage)
// ============================================================================

/// Request to batch generate text embeddings and store them
#[cfg(all(feature = "inference", feature = "storage"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbedTextRequest {
    /// List of tracks to embed and store
    pub tracks: Vec<EmbedTextAndStoreRequest>,
}

/// Response from batch embed text operation
#[cfg(all(feature = "inference", feature = "storage"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbedTextResponse {
    /// Results for each track
    pub results: Vec<BatchEmbedTextResult>,
    /// Total number of tracks processed
    pub total: usize,
    /// Number of successful operations
    pub succeeded: usize,
    /// Number of failed operations
    pub failed: usize,
}

/// Result for a single track in batch embed text
#[cfg(all(feature = "inference", feature = "storage"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbedTextResult {
    /// Track ID
    pub track_id: String,
    /// Whether the operation succeeded
    pub success: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The text that was embedded (for verification)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

// ============================================================================
// Management API Request/Response Types
// ============================================================================

use crate::types::{DownloadProgress, ModelDetail, StorageStats, SystemStatus};

/// Response for listing models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    /// List of all models (known + cached)
    pub models: Vec<ModelDetail>,
    /// Currently loaded model ID (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_model: Option<String>,
}

/// Request to download a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadModelRequest {
    /// Model ID to download (HuggingFace format: owner/model-name)
    pub model_id: String,
}

/// Response from starting a download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadModelResponse {
    /// Unique download ID for tracking progress (None if already exists)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_id: Option<String>,
    /// Model being downloaded
    pub model_id: String,
    /// Message
    pub message: String,
    /// Whether the model already exists
    #[serde(default)]
    pub already_exists: bool,
}

/// Response for listing active downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDownloadsResponse {
    /// Active and recent downloads
    pub downloads: Vec<DownloadProgress>,
}

/// Request to load a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelRequest {
    /// Model ID to load (must be downloaded first)
    pub model_id: String,
}

/// Response from loading a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelResponse {
    /// Model that was loaded
    pub model_id: String,
    /// Whether load was successful
    pub loaded: bool,
    /// Status message
    pub message: String,
    /// Device model is running on
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// Response from deleting a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteModelResponse {
    /// Model that was deleted
    pub model_id: String,
    /// Whether delete was successful
    pub deleted: bool,
    /// Status message
    pub message: String,
}

/// Request to update configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfigRequest {
    /// Model configuration updates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<UpdateModelConfig>,
    /// Storage configuration updates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<UpdateStorageConfig>,
    /// Server configuration updates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server: Option<UpdateServerConfig>,
}

/// Model configuration updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateModelConfig {
    /// Model name/ID to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Enable CUDA acceleration (NVIDIA GPUs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_cuda: Option<bool>,
    /// Enable ROCm acceleration (AMD GPUs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_rocm: Option<bool>,
    /// Enable CoreML acceleration (Apple Silicon/macOS)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_coreml: Option<bool>,
    /// Enable DirectML acceleration (Windows GPU)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_directml: Option<bool>,
    /// Enable OpenVINO acceleration (Intel CPUs/GPUs/VPUs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_openvino: Option<bool>,
}

/// Storage configuration updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStorageConfig {
    /// Storage mode (file or qdrant)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Data directory for file storage
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<String>,
    /// Qdrant URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Qdrant API key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// Server configuration updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateServerConfig {
    /// Host to bind to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    /// Port to listen on
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
}

/// Response from updating configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfigResponse {
    /// Whether update was successful
    pub success: bool,
    /// Message describing what was updated
    pub message: String,
    /// Fields that require restart to take effect
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub requires_restart: Vec<String>,
}

/// Response for storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStatsResponse {
    /// Storage statistics
    pub stats: StorageStats,
}

/// Response for system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatusResponse {
    /// Full system status
    #[serde(flatten)]
    pub status: SystemStatus,
}

// ============================================================================
// Mood Classification API Types
// ============================================================================

use crate::mood::{MoodClassification, MoodTier};

/// Request to classify mood of audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodClassifyRequest {
    /// Pre-computed audio embedding (512-dimensional)
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,

    /// Track ID to lookup embedding from storage
    #[serde(default)]
    pub track_id: Option<String>,

    /// Which mood tiers to include in results
    #[serde(default = "default_mood_tiers")]
    pub tiers: Vec<MoodTier>,

    /// Number of top moods to return per tier
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Include valence-arousal coordinates
    #[serde(default = "default_true_bool")]
    pub include_va: bool,
}

fn default_mood_tiers() -> Vec<MoodTier> {
    vec![MoodTier::Primary, MoodTier::Refined]
}

fn default_top_k() -> usize {
    3
}

fn default_true_bool() -> bool {
    true
}

/// Response from mood classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodClassifyResponse {
    /// Classification results
    #[serde(flatten)]
    pub classification: MoodClassification,
}

/// Request to list available moods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListMoodsRequest {
    /// Filter by tier (optional)
    #[serde(default)]
    pub tier: Option<MoodTier>,
}

/// Info about a single mood definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodInfo {
    /// Mood identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Classification tier
    pub tier: MoodTier,
    /// Valence hint (-1 to 1)
    pub valence_hint: f32,
    /// Arousal hint (-1 to 1)
    pub arousal_hint: f32,
}

/// Response listing available moods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListMoodsResponse {
    /// Available moods
    pub moods: Vec<MoodInfo>,
    /// Count by tier
    pub counts: std::collections::HashMap<String, usize>,
}

// ============================================================================
// Shared Ingestion Types
// ============================================================================

/// Track metadata for ingestion (used by streaming)
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestMetadata {
    /// Track name
    pub name: String,
    /// Artist names
    #[serde(default)]
    pub artists: Vec<String>,
    /// Album name
    #[serde(default)]
    pub album: Option<String>,
    /// Genre tags
    #[serde(default)]
    pub genres: Vec<String>,
}

// ============================================================================
// Streaming Ingestion API Types
// ============================================================================

/// Request to start a streaming ingestion session.
///
/// MA opens a session when playback begins, then streams audio frames
/// as the user listens.
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartStreamRequest {
    /// Track ID (Music Assistant item_id)
    pub track_id: String,

    /// Track metadata for text embedding generation
    pub metadata: IngestMetadata,

    /// Audio format of incoming frames
    pub format: AudioFormat,

    /// Sample rate of incoming audio (will be resampled to 48kHz)
    pub sample_rate: u32,

    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u8,
}

/// Response from starting a streaming session
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartStreamResponse {
    /// Unique session ID for subsequent calls
    pub session_id: String,

    /// Number of samples needed per embedding window (at 48kHz)
    /// This is 480,000 samples = 10 seconds
    pub window_samples: usize,
}

/// Response from sending audio frames
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramesResponse {
    /// Seconds of audio buffered (at 48kHz, after resampling)
    pub buffered_seconds: f32,

    /// Number of 10-second windows that have been processed
    pub windows_completed: usize,
}

/// Request to end a streaming session and finalize embeddings
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndStreamRequest {
    /// Whether to store embeddings in the vector database
    #[serde(default = "default_true_bool")]
    pub store: bool,

    /// Minimum duration (in seconds) required to generate embeddings
    /// Shorter audio will be discarded
    #[serde(default = "default_min_duration")]
    pub min_duration_s: f32,
}

#[cfg(feature = "inference")]
fn default_min_duration() -> f32 {
    3.0
}

#[cfg(feature = "inference")]
impl Default for EndStreamRequest {
    fn default() -> Self {
        Self {
            store: true,
            min_duration_s: 3.0,
        }
    }
}

/// Response from ending a streaming session
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndStreamResponse {
    /// Track ID that was processed
    pub track_id: String,

    /// Total duration of audio received (in seconds)
    pub duration_s: f32,

    /// Number of 10-second windows processed
    pub windows_processed: usize,

    /// Whether text embedding was stored
    pub text_stored: bool,

    /// Whether audio embedding was stored
    pub audio_stored: bool,
}

/// Status of a streaming session
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamSessionStatus {
    /// Session is active and receiving frames
    Active,
    /// Session is processing final embeddings
    Finalizing,
    /// Session completed successfully
    Completed,
    /// Session was aborted
    Aborted,
    /// Session encountered an error
    Error,
}

/// Response from getting session status
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStatusResponse {
    /// Session ID
    pub session_id: String,

    /// Track ID being processed
    pub track_id: String,

    /// Current session status
    pub status: StreamSessionStatus,

    /// Seconds of audio buffered
    pub buffered_seconds: f32,

    /// Windows completed so far
    pub windows_completed: usize,

    /// Session age in seconds
    pub age_seconds: f32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod stream_api_tests {
    use super::*;

    #[test]
    #[cfg(feature = "inference")]
    fn test_start_stream_request() {
        let json = r#"{
            "track_id": "spotify:track:abc123",
            "metadata": {
                "name": "Bohemian Rhapsody",
                "artists": ["Queen"],
                "album": "A Night at the Opera",
                "genres": ["Rock"]
            },
            "format": "pcm_s16_le",
            "sample_rate": 44100,
            "channels": 2
        }"#;

        let req: StartStreamRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.track_id, "spotify:track:abc123");
        assert_eq!(req.metadata.name, "Bohemian Rhapsody");
        assert_eq!(req.sample_rate, 44100);
        assert_eq!(req.channels, 2);
    }

    #[test]
    #[cfg(feature = "inference")]
    fn test_start_stream_response() {
        let resp = StartStreamResponse {
            session_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            window_samples: 480000,
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("550e8400"));
        assert!(json.contains("480000"));
    }

    #[test]
    #[cfg(feature = "inference")]
    fn test_end_stream_request_defaults() {
        let json = r#"{}"#;
        let req: EndStreamRequest = serde_json::from_str(json).unwrap();

        assert!(req.store);
        assert_eq!(req.min_duration_s, 3.0);
    }

    #[test]
    #[cfg(feature = "inference")]
    fn test_end_stream_response() {
        let resp = EndStreamResponse {
            track_id: "track_123".to_string(),
            duration_s: 45.2,
            windows_processed: 4,
            text_stored: true,
            audio_stored: true,
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("45.2"));
        assert!(json.contains("\"windows_processed\":4"));
    }

    #[test]
    #[cfg(feature = "inference")]
    fn test_stream_status_response() {
        let resp = StreamStatusResponse {
            session_id: "session_123".to_string(),
            track_id: "track_456".to_string(),
            status: StreamSessionStatus::Active,
            buffered_seconds: 4.5,
            windows_completed: 0,
            age_seconds: 12.3,
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("active"));
        assert!(json.contains("4.5"));

        let decoded: StreamStatusResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, StreamSessionStatus::Active);
    }
}

#[cfg(test)]
mod mood_api_tests {
    use super::*;

    #[test]
    fn test_mood_classify_request_defaults() {
        // Test that defaults work correctly
        let json = r#"{"embedding": [0.1, 0.2, 0.3]}"#;
        let req: MoodClassifyRequest = serde_json::from_str(json).unwrap();

        assert!(req.embedding.is_some());
        assert!(req.track_id.is_none());
        // Check defaults
        assert_eq!(req.tiers.len(), 2); // primary + refined
        assert_eq!(req.top_k, 3);
        assert!(req.include_va);
    }

    #[test]
    fn test_mood_classify_request_with_track_id() {
        let json = r#"{"track_id": "track_123", "tiers": ["primary"], "top_k": 5}"#;
        let req: MoodClassifyRequest = serde_json::from_str(json).unwrap();

        assert!(req.embedding.is_none());
        assert_eq!(req.track_id, Some("track_123".to_string()));
        assert_eq!(req.tiers.len(), 1);
        assert_eq!(req.top_k, 5);
    }

    #[test]
    fn test_mood_info_serialization() {
        let info = MoodInfo {
            id: "happy".to_string(),
            name: "Happy".to_string(),
            tier: MoodTier::Refined,
            valence_hint: 0.8,
            arousal_hint: 0.6,
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("happy"));
        assert!(json.contains("refined"));
        assert!(json.contains("0.8"));

        let decoded: MoodInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "happy");
        assert_eq!(decoded.tier, MoodTier::Refined);
    }

    #[test]
    fn test_list_moods_response() {
        let mut counts = std::collections::HashMap::new();
        counts.insert("primary".to_string(), 4);
        counts.insert("refined".to_string(), 14);

        let resp = ListMoodsResponse {
            moods: vec![MoodInfo {
                id: "energetic".to_string(),
                name: "Energetic".to_string(),
                tier: MoodTier::Primary,
                valence_hint: 0.5,
                arousal_hint: 0.9,
            }],
            counts,
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("energetic"));
        assert!(json.contains("primary"));

        let decoded: ListMoodsResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.moods.len(), 1);
        assert_eq!(decoded.counts.get("primary"), Some(&4));
    }
}
