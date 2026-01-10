//! API types for streaming audio ingestion.
//!
//! This module contains request/response types for the real-time audio
//! streaming pipeline where Music Assistant streams audio frames during playback.

use serde::{Deserialize, Serialize};

#[cfg(feature = "inference")]
use crate::inference::AudioFormat;

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

    /// If true, replace any existing session for this track (default: true)
    #[serde(default = "default_replace_existing")]
    pub replace_existing: bool,
}

#[cfg(feature = "inference")]
fn default_replace_existing() -> bool {
    true
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
fn default_true_bool() -> bool {
    true
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

#[cfg(test)]
#[cfg(feature = "inference")]
mod tests {
    use super::*;

    #[test]
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
    fn test_start_stream_response() {
        let resp = StartStreamResponse {
            session_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            window_samples: 480_000,
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("550e8400"));
        assert!(json.contains("480000"));
    }

    #[test]
    fn test_end_stream_request_defaults() {
        let json = r"{}";
        let req: EndStreamRequest = serde_json::from_str(json).unwrap();

        assert!(req.store);
        assert_eq!(req.min_duration_s, 3.0);
    }

    #[test]
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
