//! API types for embedding generation.
//!
//! This module contains request/response types for direct embedding generation
//! from text and audio, as well as combined embed+store operations.

use serde::{Deserialize, Serialize};

#[cfg(feature = "inference")]
use crate::inference::{AudioData, AudioFormat};

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
// Combined embedding + storage types
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
    /// Hash of metadata for change detection
    #[serde(default)]
    pub metadata_hash: Option<String>,
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
// Batch embed + store types
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
