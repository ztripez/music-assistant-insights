//! Streaming audio ingestion for real-time track processing.
//!
//! This module provides session-based streaming ingestion that allows MA to
//! send audio frames as a user listens, with the sidecar buffering and
//! generating embeddings when 10-second windows are complete.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use rubato::{FftFixedIn, Resampler};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::extractors::MsgPackExtractor;
use super::routes::MsgPack;
use super::AppState;
use crate::inference::{AudioFormat, ClapModel, EMBEDDING_DIM};
use crate::types::api::{
    EndStreamRequest, EndStreamResponse, FramesResponse, IngestMetadata, StartStreamRequest,
    StartStreamResponse, StreamSessionStatus, StreamStatusResponse,
};
use serde::Serialize;

/// Error response for streaming endpoints
#[derive(Debug, Serialize)]
struct StreamErrorResponse {
    error: String,
}

/// Target sample rate for CLAP models
const TARGET_SAMPLE_RATE: u32 = 48_000;

/// Window size in samples (10 seconds at 48kHz)
const WINDOW_SAMPLES: usize = 480_000;

/// Minimum samples required to generate an embedding (3 seconds at 48kHz)
#[allow(dead_code)]
const MIN_SAMPLES_FOR_EMBEDDING: usize = 144_000;

/// Session timeout in seconds (5 minutes)
const SESSION_TIMEOUT_SECS: u64 = 300;

/// Maximum resampling errors before failing the session
const MAX_RESAMPLING_ERRORS: usize = 10;

/// Error type for streaming operations
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Session already exists for track: {0}")]
    SessionExists(String),

    #[error("Invalid audio format: {0}")]
    InvalidAudioFormat(String),

    #[error("Resampler error: {0}")]
    ResamplerError(String),

    #[error("Too many resampling errors ({0}), audio quality compromised")]
    TooManyResamplingErrors(usize),

    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Storage error: {0}")]
    StorageError(String),
}

/// A single streaming ingestion session
pub struct StreamSession {
    /// Unique session ID
    pub id: Uuid,
    /// Track ID being processed
    pub track_id: String,
    /// Track metadata for text embedding
    pub metadata: IngestMetadata,
    /// Audio format of incoming frames
    pub format: AudioFormat,
    /// Source sample rate
    pub source_sample_rate: u32,
    /// Number of channels (1 or 2)
    pub channels: u8,

    // Buffers
    /// Buffer for incoming samples before resampling (mono f32)
    mono_buffer: Vec<f32>,
    /// Buffer for resampled audio at 48kHz
    resampled_buffer: Vec<f32>,
    /// Resampler instance (None if source is already 48kHz)
    resampler: Option<FftFixedIn<f32>>,

    // State
    /// Embeddings from completed windows
    window_embeddings: Vec<Vec<f32>>,
    /// Count of resampling errors (for quality tracking)
    resampling_errors: usize,
    /// Current session status
    pub status: StreamSessionStatus,

    // Tracking
    /// When session was created
    pub created_at: Instant,
    /// Last activity time
    pub last_activity: Instant,
}

impl StreamSession {
    /// Create a new streaming session
    pub fn new(
        track_id: String,
        metadata: IngestMetadata,
        format: AudioFormat,
        sample_rate: u32,
        channels: u8,
    ) -> Result<Self, StreamError> {
        let id = Uuid::new_v4();

        // Create resampler if source rate differs from target
        let resampler = if sample_rate != TARGET_SAMPLE_RATE {
            Some(
                FftFixedIn::<f32>::new(
                    sample_rate as usize,
                    TARGET_SAMPLE_RATE as usize,
                    1024,
                    1,
                    1,
                )
                .map_err(|e| StreamError::ResamplerError(e.to_string()))?,
            )
        } else {
            None
        };

        let now = Instant::now();

        Ok(Self {
            id,
            track_id,
            metadata,
            format,
            source_sample_rate: sample_rate,
            channels,
            mono_buffer: Vec::with_capacity(sample_rate as usize * 10), // ~10s buffer
            resampled_buffer: Vec::with_capacity(WINDOW_SAMPLES),
            resampler,
            window_embeddings: Vec::new(),
            resampling_errors: 0,
            status: StreamSessionStatus::Active,
            created_at: now,
            last_activity: now,
        })
    }

    /// Process incoming audio frames and return number of new complete windows
    pub fn process_frames(&mut self, data: &[u8], model: &ClapModel) -> Result<usize, StreamError> {
        self.last_activity = Instant::now();

        // Convert bytes to f32 samples
        let samples = self.bytes_to_f32(data)?;

        // Convert to mono if stereo
        let mono = if self.channels == 2 {
            samples
                .chunks_exact(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect()
        } else {
            samples
        };

        // Add to mono buffer
        self.mono_buffer.extend(mono);

        // Resample and process windows
        let initial_windows = self.window_embeddings.len();
        self.process_resampling_and_windows(model)?;

        Ok(self.window_embeddings.len() - initial_windows)
    }

    /// Convert raw bytes to f32 samples based on format
    fn bytes_to_f32(&self, data: &[u8]) -> Result<Vec<f32>, StreamError> {
        match self.format {
            AudioFormat::PcmF32Le => {
                if data.len() % 4 != 0 {
                    return Err(StreamError::InvalidAudioFormat(
                        "F32 data length not multiple of 4".into(),
                    ));
                }
                Ok(data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
            AudioFormat::PcmS16Le => {
                if data.len() % 2 != 0 {
                    return Err(StreamError::InvalidAudioFormat(
                        "S16 data length not multiple of 2".into(),
                    ));
                }
                Ok(data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect())
            }
            AudioFormat::PcmS24Le => {
                if data.len() % 3 != 0 {
                    return Err(StreamError::InvalidAudioFormat(
                        "S24 data length not multiple of 3".into(),
                    ));
                }
                Ok(data
                    .chunks_exact(3)
                    .map(|chunk| {
                        let sample = if chunk[2] & 0x80 != 0 {
                            i32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0xFF])
                        } else {
                            i32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0x00])
                        };
                        sample as f32 / 8388608.0
                    })
                    .collect())
            }
        }
    }

    /// Process resampling and extract windows
    fn process_resampling_and_windows(&mut self, model: &ClapModel) -> Result<(), StreamError> {
        if let Some(ref mut resampler) = self.resampler {
            // Incremental resampling
            let chunk_size = resampler.input_frames_max();

            while self.mono_buffer.len() >= chunk_size {
                // Take a chunk from mono_buffer
                let chunk: Vec<f32> = self.mono_buffer.drain(..chunk_size).collect();
                let input = vec![chunk];

                match resampler.process(&input, None) {
                    Ok(output) => {
                        if !output.is_empty() {
                            self.resampled_buffer.extend(&output[0]);
                        }
                    }
                    Err(e) => {
                        self.resampling_errors += 1;
                        warn!(
                            error = %e,
                            errors = self.resampling_errors,
                            max = MAX_RESAMPLING_ERRORS,
                            "Resampling chunk failed"
                        );
                        if self.resampling_errors >= MAX_RESAMPLING_ERRORS {
                            return Err(StreamError::TooManyResamplingErrors(
                                self.resampling_errors,
                            ));
                        }
                    }
                }
            }
        } else {
            // No resampling needed, move directly to resampled buffer
            self.resampled_buffer.append(&mut self.mono_buffer);
        }

        // Extract complete windows and generate embeddings
        while self.resampled_buffer.len() >= WINDOW_SAMPLES {
            let window: Vec<f32> = self.resampled_buffer.drain(..WINDOW_SAMPLES).collect();
            let embedding = self.generate_window_embedding(&window, model)?;
            self.window_embeddings.push(embedding);

            debug!(
                session_id = %self.id,
                windows = self.window_embeddings.len(),
                "Completed embedding window"
            );
        }

        Ok(())
    }

    /// Generate embedding for a single window
    fn generate_window_embedding(
        &self,
        window: &[f32],
        model: &ClapModel,
    ) -> Result<Vec<f32>, StreamError> {
        // The model expects [batch, samples] format
        // We call the raw audio inference directly since we already have processed samples

        // Create AudioData with our preprocessed samples
        let data: Vec<u8> = window.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let audio_data = crate::inference::AudioData {
            format: AudioFormat::PcmF32Le,
            sample_rate: TARGET_SAMPLE_RATE,
            channels: 1,
            data,
        };

        let embedding = model
            .audio_embedding(&audio_data)
            .map_err(|e| StreamError::InferenceError(e.to_string()))?;

        Ok(embedding.data().to_vec())
    }

    /// Finalize session and get averaged embedding
    pub fn finalize(
        &mut self,
        model: &ClapModel,
        min_duration_s: f32,
    ) -> Result<Option<Vec<f32>>, StreamError> {
        self.status = StreamSessionStatus::Finalizing;

        // Process any remaining mono buffer
        if !self.mono_buffer.is_empty() {
            self.flush_resampler()?;
        }

        // Calculate minimum samples needed
        let min_samples = (min_duration_s * TARGET_SAMPLE_RATE as f32) as usize;

        // Check if we have enough remaining samples to process
        if self.resampled_buffer.len() >= min_samples {
            // Take ownership of buffer since we're finalizing (avoids clone)
            let mut final_window = std::mem::take(&mut self.resampled_buffer);
            if final_window.len() < WINDOW_SAMPLES {
                final_window.resize(WINDOW_SAMPLES, 0.0);
            } else {
                final_window.truncate(WINDOW_SAMPLES);
            }

            let embedding = self.generate_window_embedding(&final_window, model)?;
            self.window_embeddings.push(embedding);
        }

        // Average all window embeddings
        if self.window_embeddings.is_empty() {
            self.status = StreamSessionStatus::Completed;
            return Ok(None);
        }

        let mut averaged = vec![0.0f32; EMBEDDING_DIM];
        for emb in &self.window_embeddings {
            for (i, &v) in emb.iter().enumerate() {
                if i < EMBEDDING_DIM {
                    averaged[i] += v;
                }
            }
        }
        let count = self.window_embeddings.len() as f32;
        for v in &mut averaged {
            *v /= count;
        }

        // Normalize
        let norm: f32 = averaged.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut averaged {
                *v /= norm;
            }
        }

        self.status = StreamSessionStatus::Completed;
        Ok(Some(averaged))
    }

    /// Flush any remaining samples through the resampler
    fn flush_resampler(&mut self) -> Result<(), StreamError> {
        if let Some(ref mut resampler) = self.resampler {
            if !self.mono_buffer.is_empty() {
                // Pad to chunk size
                let chunk_size = resampler.input_frames_max();
                let mut final_chunk = self.mono_buffer.drain(..).collect::<Vec<_>>();
                final_chunk.resize(chunk_size, 0.0);

                let input = vec![final_chunk];
                if let Ok(output) = resampler.process(&input, None) {
                    if !output.is_empty() {
                        self.resampled_buffer.extend(&output[0]);
                    }
                }
            }
        } else {
            self.resampled_buffer.append(&mut self.mono_buffer);
        }
        Ok(())
    }

    /// Get buffered duration in seconds (at 48kHz)
    pub fn buffered_seconds(&self) -> f32 {
        self.resampled_buffer.len() as f32 / TARGET_SAMPLE_RATE as f32
    }

    /// Get total duration processed including windows
    pub fn total_duration_seconds(&self) -> f32 {
        let window_samples = self.window_embeddings.len() * WINDOW_SAMPLES;
        let total_samples = window_samples + self.resampled_buffer.len();
        total_samples as f32 / TARGET_SAMPLE_RATE as f32
    }

    /// Get number of windows completed
    pub fn windows_completed(&self) -> usize {
        self.window_embeddings.len()
    }

    /// Get session age in seconds
    pub fn age_seconds(&self) -> f32 {
        self.created_at.elapsed().as_secs_f32()
    }

    /// Check if session has timed out
    pub fn is_timed_out(&self) -> bool {
        self.last_activity.elapsed().as_secs() > SESSION_TIMEOUT_SECS
    }
}

/// Manager for all active streaming sessions
#[derive(Default)]
pub struct StreamSessionManager {
    sessions: HashMap<Uuid, StreamSession>,
    /// Map track_id -> session_id for duplicate detection
    track_sessions: HashMap<String, Uuid>,
}

impl StreamSessionManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new session for a track
    pub fn create_session(
        &mut self,
        track_id: String,
        metadata: IngestMetadata,
        format: AudioFormat,
        sample_rate: u32,
        channels: u8,
    ) -> Result<Uuid, StreamError> {
        // Check for existing session for this track
        if self.track_sessions.contains_key(&track_id) {
            return Err(StreamError::SessionExists(track_id));
        }

        let session =
            StreamSession::new(track_id.clone(), metadata, format, sample_rate, channels)?;
        let session_id = session.id;

        self.track_sessions.insert(track_id, session_id);
        self.sessions.insert(session_id, session);

        Ok(session_id)
    }

    /// Get a mutable reference to a session
    pub fn get_session_mut(&mut self, session_id: &Uuid) -> Option<&mut StreamSession> {
        self.sessions.get_mut(session_id)
    }

    /// Get a reference to a session
    pub fn get_session(&self, session_id: &Uuid) -> Option<&StreamSession> {
        self.sessions.get(session_id)
    }

    /// Remove a session
    pub fn remove_session(&mut self, session_id: &Uuid) -> Option<StreamSession> {
        if let Some(session) = self.sessions.remove(session_id) {
            self.track_sessions.remove(&session.track_id);
            Some(session)
        } else {
            None
        }
    }

    /// Clean up timed-out sessions
    pub fn cleanup_timed_out(&mut self) -> usize {
        let timed_out: Vec<Uuid> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.is_timed_out())
            .map(|(id, _)| *id)
            .collect();

        let count = timed_out.len();
        for id in timed_out {
            if let Some(session) = self.sessions.remove(&id) {
                self.track_sessions.remove(&session.track_id);
                info!(
                    session_id = %id,
                    track_id = %session.track_id,
                    "Cleaned up timed out session"
                );
            }
        }
        count
    }

    /// Get number of active sessions
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

/// Type alias for shared session manager
pub type SharedStreamManager = Arc<RwLock<StreamSessionManager>>;

/// Interval for session cleanup checks (60 seconds)
const CLEANUP_INTERVAL_SECS: u64 = 60;

/// Spawn a background task to periodically clean up timed-out sessions
pub fn spawn_session_cleanup_task(manager: SharedStreamManager) {
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(CLEANUP_INTERVAL_SECS));

        loop {
            interval.tick().await;

            let cleaned = {
                let mut mgr = manager.write().await;
                mgr.cleanup_timed_out()
            };

            if cleaned > 0 {
                info!(count = cleaned, "Cleaned up timed-out streaming sessions");
            }
        }
    });

    info!(
        interval_secs = CLEANUP_INTERVAL_SECS,
        timeout_secs = SESSION_TIMEOUT_SECS,
        "Started streaming session cleanup task"
    );
}

// ============================================================================
// HTTP Handlers
// ============================================================================

/// Start a new streaming session
pub async fn start_stream(
    State(state): State<AppState>,
    MsgPackExtractor(request): MsgPackExtractor<StartStreamRequest>,
) -> impl IntoResponse {
    // Check if model is loaded
    let model_guard = state.model.read().await;
    if model_guard.is_none() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            MsgPack(StreamErrorResponse {
                error: "Model not loaded".to_string(),
            }),
        )
            .into_response();
    }
    drop(model_guard);

    // Get stream manager
    let mut manager = state.stream_manager.write().await;

    match manager.create_session(
        request.track_id.clone(),
        request.metadata,
        request.format,
        request.sample_rate,
        request.channels,
    ) {
        Ok(session_id) => {
            info!(
                session_id = %session_id,
                track_id = %request.track_id,
                sample_rate = request.sample_rate,
                channels = request.channels,
                "Started streaming session"
            );

            MsgPack(StartStreamResponse {
                session_id: session_id.to_string(),
                window_samples: WINDOW_SAMPLES,
            })
            .into_response()
        }
        Err(StreamError::SessionExists(track_id)) => (
            StatusCode::CONFLICT,
            MsgPack(StreamErrorResponse {
                error: format!("Session already exists for track: {}", track_id),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            MsgPack(StreamErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

/// Receive audio frames for a session
pub async fn stream_frames(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    body: Bytes,
) -> impl IntoResponse {
    let session_uuid = match Uuid::parse_str(&session_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                MsgPack(StreamErrorResponse {
                    error: "Invalid session ID".to_string(),
                }),
            )
                .into_response();
        }
    };

    // Get model reference
    let model_guard = state.model.read().await;
    let model = match model_guard.as_ref() {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                MsgPack(StreamErrorResponse {
                    error: "Model not loaded".to_string(),
                }),
            )
                .into_response();
        }
    };
    drop(model_guard);

    // Get session and process frames
    let mut manager = state.stream_manager.write().await;
    let session = match manager.get_session_mut(&session_uuid) {
        Some(s) => s,
        None => {
            return (
                StatusCode::NOT_FOUND,
                MsgPack(StreamErrorResponse {
                    error: "Session not found".to_string(),
                }),
            )
                .into_response();
        }
    };

    match session.process_frames(&body, &model) {
        Ok(_new_windows) => MsgPack(FramesResponse {
            buffered_seconds: session.buffered_seconds(),
            windows_completed: session.windows_completed(),
        })
        .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            MsgPack(StreamErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

/// End a streaming session and finalize embeddings
pub async fn end_stream(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    MsgPackExtractor(request): MsgPackExtractor<EndStreamRequest>,
) -> impl IntoResponse {
    let session_uuid = match Uuid::parse_str(&session_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                MsgPack(StreamErrorResponse {
                    error: "Invalid session ID".to_string(),
                }),
            )
                .into_response();
        }
    };

    // Get model reference
    let model_guard = state.model.read().await;
    let model = match model_guard.as_ref() {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                MsgPack(StreamErrorResponse {
                    error: "Model not loaded".to_string(),
                }),
            )
                .into_response();
        }
    };
    drop(model_guard);

    // Remove session from manager
    let mut manager = state.stream_manager.write().await;
    let mut session = match manager.remove_session(&session_uuid) {
        Some(s) => s,
        None => {
            return (
                StatusCode::NOT_FOUND,
                MsgPack(StreamErrorResponse {
                    error: "Session not found".to_string(),
                }),
            )
                .into_response();
        }
    };
    drop(manager);

    let track_id = session.track_id.clone();
    let duration_s = session.total_duration_seconds();
    let windows_processed = session.windows_completed();

    // Finalize and get audio embedding
    let audio_embedding = match session.finalize(&model, request.min_duration_s) {
        Ok(emb) => emb,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                MsgPack(StreamErrorResponse {
                    error: format!("Failed to finalize: {}", e),
                }),
            )
                .into_response();
        }
    };

    // Generate text embedding from metadata (needs reference first)
    let text = format_metadata_text(&session.metadata);
    let text_embedding = match model.text_embedding(&text) {
        Ok(emb) => emb,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                MsgPack(StreamErrorResponse {
                    error: format!("Failed to generate text embedding: {}", e),
                }),
            )
                .into_response();
        }
    };

    let mut text_stored = false;
    let mut audio_stored = false;

    // Store embeddings if requested
    if request.store {
        if let Some(ref storage) = state.storage {
            use crate::storage::{TrackMetadata, AUDIO_COLLECTION, TEXT_COLLECTION};

            // Move metadata fields from session (avoids clones since we own session)
            let IngestMetadata {
                name,
                artists,
                album,
                genres,
            } = session.metadata;

            // Build metadata for storage - move fields instead of cloning
            let mut metadata = TrackMetadata::new(track_id.clone(), name)
                .with_artists(artists)
                .with_genres(genres);

            if let Some(album_name) = album {
                metadata = metadata.with_album(album_name);
            }

            // Store audio embedding first (moves metadata ownership)
            if let Some(ref audio_emb) = audio_embedding {
                // Clone metadata only if we need it for text embedding too
                let audio_metadata = metadata.clone();
                if let Err(e) = storage
                    .upsert(AUDIO_COLLECTION, &track_id, audio_emb, audio_metadata)
                    .await
                {
                    warn!(error = %e, "Failed to store audio embedding");
                } else {
                    audio_stored = true;
                }
            }

            // Store text embedding (moves metadata, no clone needed)
            if let Err(e) = storage
                .upsert(TEXT_COLLECTION, &track_id, text_embedding.data(), metadata)
                .await
            {
                warn!(error = %e, "Failed to store text embedding");
            } else {
                text_stored = true;
            }
        }
    }

    info!(
        track_id = %track_id,
        duration_s = duration_s,
        windows = windows_processed,
        text_stored = text_stored,
        audio_stored = audio_stored,
        "Streaming session completed"
    );

    MsgPack(EndStreamResponse {
        track_id,
        duration_s,
        windows_processed,
        text_stored,
        audio_stored,
    })
    .into_response()
}

/// Abort a streaming session
pub async fn abort_stream(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    let session_uuid = match Uuid::parse_str(&session_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                MsgPack(StreamErrorResponse {
                    error: "Invalid session ID".to_string(),
                }),
            )
                .into_response();
        }
    };

    let mut manager = state.stream_manager.write().await;
    match manager.remove_session(&session_uuid) {
        Some(session) => {
            info!(
                session_id = %session_uuid,
                track_id = %session.track_id,
                "Session aborted"
            );
            (StatusCode::NO_CONTENT, ()).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            MsgPack(StreamErrorResponse {
                error: "Session not found".to_string(),
            }),
        )
            .into_response(),
    }
}

/// Get status of a streaming session
pub async fn stream_status(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    let session_uuid = match Uuid::parse_str(&session_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                MsgPack(StreamErrorResponse {
                    error: "Invalid session ID".to_string(),
                }),
            )
                .into_response();
        }
    };

    let manager = state.stream_manager.read().await;
    match manager.get_session(&session_uuid) {
        Some(session) => MsgPack(StreamStatusResponse {
            session_id: session.id.to_string(),
            track_id: session.track_id.clone(),
            status: session.status,
            buffered_seconds: session.buffered_seconds(),
            windows_completed: session.windows_completed(),
            age_seconds: session.age_seconds(),
        })
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            MsgPack(StreamErrorResponse {
                error: "Session not found".to_string(),
            }),
        )
            .into_response(),
    }
}

/// Format metadata into text for embedding
fn format_metadata_text(metadata: &IngestMetadata) -> String {
    let mut parts = vec![metadata.name.clone()];

    if !metadata.artists.is_empty() {
        parts.push(format!("by {}", metadata.artists.join(", ")));
    }

    if let Some(ref album) = metadata.album {
        parts.push(format!("from {}", album));
    }

    if !metadata.genres.is_empty() {
        parts.push(format!("({})", metadata.genres.join(", ")));
    }

    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_metadata_text() {
        let metadata = IngestMetadata {
            name: "Bohemian Rhapsody".to_string(),
            artists: vec!["Queen".to_string()],
            album: Some("A Night at the Opera".to_string()),
            genres: vec!["Rock".to_string(), "Progressive Rock".to_string()],
        };

        let text = format_metadata_text(&metadata);
        assert!(text.contains("Bohemian Rhapsody"));
        assert!(text.contains("by Queen"));
        assert!(text.contains("from A Night at the Opera"));
        assert!(text.contains("Rock"));
    }

    #[test]
    fn test_session_manager_create() {
        let mut manager = StreamSessionManager::new();

        let metadata = IngestMetadata {
            name: "Test Song".to_string(),
            artists: vec![],
            album: None,
            genres: vec![],
        };

        let result = manager.create_session(
            "track_123".to_string(),
            metadata.clone(),
            AudioFormat::PcmS16Le,
            44100,
            2,
        );

        assert!(result.is_ok());
        assert_eq!(manager.session_count(), 1);

        // Duplicate should fail
        let result2 = manager.create_session(
            "track_123".to_string(),
            metadata,
            AudioFormat::PcmS16Le,
            44100,
            2,
        );

        assert!(matches!(result2, Err(StreamError::SessionExists(_))));
    }

    #[test]
    fn test_bytes_to_f32_s16() {
        let metadata = IngestMetadata {
            name: "Test".to_string(),
            artists: vec![],
            album: None,
            genres: vec![],
        };

        let session =
            StreamSession::new("test".into(), metadata, AudioFormat::PcmS16Le, 44100, 1).unwrap();

        // 16384 in little-endian = 0.5 when normalized
        let data = vec![0x00u8, 0x40];
        let samples = session.bytes_to_f32(&data).unwrap();

        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.5).abs() < 0.001);
    }
}
