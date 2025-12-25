//! Streaming audio ingestion for real-time track processing.
//!
//! This module provides session-based streaming ingestion that allows MA to
//! send audio frames as a user listens, with the sidecar buffering and
//! generating embeddings when 10-second windows are complete.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use dashmap::DashMap;
use ndarray::Array2;
use rubato::{FftFixedIn, Resampler};
use tokio::sync::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

use super::extractors::MsgPackExtractor;
use super::routes::MsgPack;
use super::AppState;
use crate::inference::{
    AudioFormat, ClapModel, MelFeatures, EMBEDDING_DIM,
    audio::{CLAP_N_FFT, CLAP_HOP_LENGTH, CLAP_N_MELS, CLAP_F_MIN, CLAP_F_MAX, CLAP_SPEC_SIZE, resize_mel_spectrogram},
};
use mel_spec::mel::mel;
use mel_spec::stft::Spectrogram;
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

/// Hop size in samples (5 seconds = 50% overlap)
const HOP_SAMPLES: usize = 240_000;

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

    // Mel spectrogram computation
    /// Mel filterbank matrix [n_mels, n_fft/2+1]
    mel_filterbank: Array2<f32>,

    // State
    /// Mel spectrograms from completed windows (computed during streaming, cheap)
    /// Inference is deferred to finalize() for efficiency
    window_mels: Vec<MelFeatures>,
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

        // Create mel filterbank (same as AudioProcessor)
        let mel_fb_f64 = mel(
            TARGET_SAMPLE_RATE as f64,
            CLAP_N_FFT,
            CLAP_N_MELS,
            Some(CLAP_F_MIN as f64),
            Some(CLAP_F_MAX as f64),
            false, // htk-style
            true,  // normalize (slaney style)
        );
        let mel_filterbank = mel_fb_f64.mapv(|x| x as f32);

        let now = Instant::now();

        Ok(Self {
            id,
            track_id,
            metadata,
            format,
            source_sample_rate: sample_rate,
            channels,
            mono_buffer: Vec::with_capacity(sample_rate as usize * 10), // ~10s buffer
            resampled_buffer: Vec::with_capacity(WINDOW_SAMPLES + HOP_SAMPLES), // Extra for overlap
            resampler,
            mel_filterbank,
            window_mels: Vec::new(),
            resampling_errors: 0,
            status: StreamSessionStatus::Active,
            created_at: now,
            last_activity: now,
        })
    }

    /// Process incoming audio frames and return number of new complete windows
    ///
    /// This method is cheap - it only computes mel spectrograms (STFT + filterbank).
    /// Model inference is deferred to finalize() for efficiency.
    pub fn process_frames(&mut self, data: &[u8]) -> Result<usize, StreamError> {
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

        // Resample and extract mel spectrograms for complete windows
        let initial_windows = self.window_mels.len();
        self.process_resampling_and_windows()?;

        Ok(self.window_mels.len() - initial_windows)
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
                        sample as f32 / 8_388_608.0
                    })
                    .collect())
            }
        }
    }

    /// Process resampling and extract mel spectrograms for complete windows
    ///
    /// Uses sliding window with 50% overlap (hop = 5s, window = 10s).
    /// Only computes mel spectrograms (cheap), inference is deferred to finalize().
    fn process_resampling_and_windows(&mut self) -> Result<(), StreamError> {
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

        // Extract complete windows with sliding window overlap
        // Copy window (don't drain), then drain only HOP_SAMPLES
        while self.resampled_buffer.len() >= WINDOW_SAMPLES {
            let window: Vec<f32> = self.resampled_buffer[..WINDOW_SAMPLES].to_vec();

            // Compute mel spectrogram for this window (cheap operation)
            let mel = self.compute_mel_spectrogram(&window)?;
            self.window_mels.push(mel);

            // Only remove hop samples (keep overlap for next window)
            self.resampled_buffer.drain(..HOP_SAMPLES);

            info!(
                session_id = %self.id,
                track_id = %self.track_id,
                window = self.window_mels.len(),
                total_duration_s = format!("{:.1}", (self.window_mels.len() as f32 * HOP_SAMPLES as f32 / TARGET_SAMPLE_RATE as f32) + 10.0),
                "Mel spectrogram computed"
            );
        }

        Ok(())
    }

    /// Compute mel spectrogram for a single audio window (cheap operation)
    fn compute_mel_spectrogram(&self, samples: &[f32]) -> Result<MelFeatures, StreamError> {
        // Create STFT processor
        let mut stft = Spectrogram::new(CLAP_N_FFT, CLAP_HOP_LENGTH);

        // Collect all STFT frames as power spectra
        let mut power_frames: Vec<Vec<f32>> = Vec::new();

        for chunk in samples.chunks(CLAP_HOP_LENGTH) {
            if let Some(fft_result) = stft.add(chunk) {
                // Compute power spectrum (magnitude squared)
                let power: Vec<f32> = fft_result
                    .iter()
                    .map(|c| (c.re * c.re + c.im * c.im) as f32)
                    .collect();
                power_frames.push(power);
            }
        }

        if power_frames.is_empty() {
            return Err(StreamError::InvalidAudioFormat(
                "No STFT frames produced".to_string(),
            ));
        }

        // Apply mel filterbank to each power spectrum frame
        let n_freqs = CLAP_N_FFT / 2 + 1;
        let time_frames = power_frames.len();

        // Build mel spectrogram: [n_mels, time_frames]
        let mut mel_spec_raw = vec![0.0f32; CLAP_N_MELS * time_frames];

        for (t, power_frame) in power_frames.iter().enumerate() {
            let frame_len = power_frame.len().min(n_freqs);

            for m in 0..CLAP_N_MELS {
                let mut sum = 0.0f32;
                for f in 0..frame_len {
                    sum += self.mel_filterbank[[m, f]] * power_frame[f];
                }
                // Apply log scaling
                mel_spec_raw[m * time_frames + t] = (sum + 1e-10).ln();
            }
        }

        // Resize to [CLAP_SPEC_SIZE, CLAP_N_MELS] for CLAP model
        let resized = resize_mel_spectrogram(&mel_spec_raw, CLAP_N_MELS, time_frames, CLAP_SPEC_SIZE);

        Ok(MelFeatures {
            data: resized,
            height: CLAP_SPEC_SIZE,
            width: CLAP_N_MELS,
        })
    }

    /// Finalize session: run inference on all buffered mel spectrograms and average
    ///
    /// This is where all the expensive model inference happens - at the end of the session,
    /// not during streaming. This ensures we have all the data before committing.
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

        // Check if we have enough remaining samples to process as a final window
        if self.resampled_buffer.len() >= min_samples {
            let mut final_window = std::mem::take(&mut self.resampled_buffer);
            if final_window.len() < WINDOW_SAMPLES {
                final_window.resize(WINDOW_SAMPLES, 0.0);
            } else {
                final_window.truncate(WINDOW_SAMPLES);
            }

            // Compute mel spectrogram for the final window
            let mel = self.compute_mel_spectrogram(&final_window)?;
            self.window_mels.push(mel);
        }

        // No windows? Return None
        if self.window_mels.is_empty() {
            self.status = StreamSessionStatus::Completed;
            return Ok(None);
        }

        info!(
            session_id = %self.id,
            track_id = %self.track_id,
            num_windows = self.window_mels.len(),
            "Running batch inference on mel spectrograms"
        );

        // Run inference on all mel spectrograms and collect embeddings
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(self.window_mels.len());

        for mel in &self.window_mels {
            let embedding = model
                .run_audio_inference_from_mel(mel)
                .map_err(|e| StreamError::InferenceError(e.to_string()))?;
            embeddings.push(embedding);
        }

        // Average all window embeddings
        let mut averaged = vec![0.0f32; EMBEDDING_DIM];
        for emb in &embeddings {
            for (i, &v) in emb.iter().enumerate() {
                if i < EMBEDDING_DIM {
                    averaged[i] += v;
                }
            }
        }
        let count = embeddings.len() as f32;
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
    ///
    /// With 50% overlap, each window after the first adds HOP_SAMPLES worth of new audio.
    /// First window = WINDOW_SAMPLES, subsequent windows add HOP_SAMPLES each.
    pub fn total_duration_seconds(&self) -> f32 {
        let window_samples = if self.window_mels.is_empty() {
            0
        } else {
            // First window covers WINDOW_SAMPLES, each additional covers HOP_SAMPLES of new audio
            WINDOW_SAMPLES + (self.window_mels.len().saturating_sub(1)) * HOP_SAMPLES
        };
        let total_samples = window_samples + self.resampled_buffer.len();
        total_samples as f32 / TARGET_SAMPLE_RATE as f32
    }

    /// Get number of windows completed
    pub fn windows_completed(&self) -> usize {
        self.window_mels.len()
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

/// Wrapper for a session with its own lock for concurrent access
pub type LockedSession = Arc<Mutex<StreamSession>>;

/// Manager for all active streaming sessions using lock-free concurrent maps.
///
/// This design allows multiple streams to process frames concurrently:
/// - DashMap provides sharded locking for the session registry
/// - Each session has its own Mutex for frame processing
/// - Different sessions never block each other
pub struct StreamSessionManager {
    /// Map session_id -> locked session
    sessions: DashMap<Uuid, LockedSession>,
    /// Map track_id -> session_id for duplicate detection
    track_sessions: DashMap<String, Uuid>,
}

impl Default for StreamSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamSessionManager {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            track_sessions: DashMap::new(),
        }
    }

    /// Create a new session for a track
    ///
    /// This is a quick operation - just creates the session and registers it.
    /// The expensive mel filterbank creation happens here but doesn't block other sessions.
    pub fn create_session(
        &self,
        track_id: String,
        metadata: IngestMetadata,
        format: AudioFormat,
        sample_rate: u32,
        channels: u8,
        replace_existing: bool,
    ) -> Result<Uuid, StreamError> {
        // Check for existing session for this track
        if let Some(existing_entry) = self.track_sessions.get(&track_id) {
            let existing_id = *existing_entry;
            drop(existing_entry); // Release the read lock before modifying

            if replace_existing {
                // Remove the existing session
                self.sessions.remove(&existing_id);
                self.track_sessions.remove(&track_id);
                info!(
                    old_session_id = %existing_id,
                    track_id = %track_id,
                    "Replaced existing session for track"
                );
            } else {
                return Err(StreamError::SessionExists(track_id));
            }
        }

        let session =
            StreamSession::new(track_id.clone(), metadata, format, sample_rate, channels)?;
        let session_id = session.id;

        self.track_sessions.insert(track_id, session_id);
        self.sessions.insert(session_id, Arc::new(Mutex::new(session)));

        Ok(session_id)
    }

    /// Get a locked session for processing
    ///
    /// Returns an Arc<Mutex<StreamSession>> that can be locked independently.
    /// This allows the caller to hold the session lock without blocking the manager.
    pub fn get_session(&self, session_id: &Uuid) -> Option<LockedSession> {
        self.sessions.get(session_id).map(|entry| entry.clone())
    }

    /// Remove a session and return it (already locked for finalization)
    ///
    /// Returns the session wrapped in Arc<Mutex> - caller should lock it for final processing.
    pub fn remove_session(&self, session_id: &Uuid) -> Option<LockedSession> {
        if let Some((_, session)) = self.sessions.remove(session_id) {
            // We need to get the track_id to clean up the reverse mapping
            // Try to lock briefly just to read track_id
            if let Ok(guard) = session.try_lock() {
                self.track_sessions.remove(&guard.track_id);
            }
            // If we can't lock, the track_sessions entry will be stale but harmless
            // (cleanup will handle it, or next create with replace_existing will)
            Some(session)
        } else {
            None
        }
    }

    /// Clean up timed-out sessions
    ///
    /// This iterates all sessions and removes those that have timed out.
    /// Uses try_lock to avoid blocking on active sessions.
    pub fn cleanup_timed_out(&self) -> usize {
        let mut timed_out = Vec::new();

        // First pass: identify timed-out sessions (non-blocking)
        for entry in self.sessions.iter() {
            let session_id = *entry.key();
            if let Ok(guard) = entry.value().try_lock() {
                if guard.is_timed_out() {
                    timed_out.push((session_id, guard.track_id.clone()));
                }
            }
            // If we can't lock, session is actively being used - not timed out
        }

        // Second pass: remove timed-out sessions
        let count = timed_out.len();
        for (session_id, track_id) in timed_out {
            self.sessions.remove(&session_id);
            self.track_sessions.remove(&track_id);
            info!(
                session_id = %session_id,
                track_id = %track_id,
                "Cleaned up timed out session"
            );
        }

        count
    }

    /// Get number of active sessions
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

/// Type alias for shared session manager
///
/// No RwLock needed - DashMap handles internal concurrency,
/// and each session has its own Mutex for frame processing.
pub type SharedStreamManager = Arc<StreamSessionManager>;

/// Interval for session cleanup checks (60 seconds)
const CLEANUP_INTERVAL_SECS: u64 = 60;

/// Spawn a background task to periodically clean up timed-out sessions
pub fn spawn_session_cleanup_task(manager: SharedStreamManager) {
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(CLEANUP_INTERVAL_SECS));

        loop {
            interval.tick().await;

            // No lock needed - cleanup_timed_out uses DashMap's internal concurrency
            let cleaned = manager.cleanup_timed_out();

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

    // Create session - no write lock needed, DashMap handles concurrency
    match state.stream_manager.create_session(
        request.track_id.clone(),
        request.metadata,
        request.format,
        request.sample_rate,
        request.channels,
        request.replace_existing,
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
///
/// This endpoint is lightweight - it only buffers audio and computes mel spectrograms.
/// Model inference is deferred to end_stream for efficiency.
///
/// Concurrent streams don't block each other - each session has its own lock.
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

    // Get session (quick DashMap lookup, doesn't block other sessions)
    let locked_session = match state.stream_manager.get_session(&session_uuid) {
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

    // Lock this specific session for frame processing
    // Other sessions can process concurrently
    let mut session = locked_session.lock().await;

    match session.process_frames(&body) {
        Ok(_new_windows) => MsgPack(FramesResponse {
            buffered_seconds: session.buffered_seconds(),
            windows_completed: session.windows_completed(),
        })
        .into_response(),
        Err(e) => {
            warn!(
                session_id = %session_id,
                body_len = body.len(),
                error = %e,
                "Failed to process audio frames"
            );
            (
                StatusCode::BAD_REQUEST,
                MsgPack(StreamErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response()
        }
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

    // Remove session from manager (quick DashMap operation)
    let locked_session = match state.stream_manager.remove_session(&session_uuid) {
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

    // Lock the session for finalization
    let mut session = locked_session.lock().await;

    let track_id = session.track_id.clone();
    let duration_s = session.total_duration_seconds();
    let windows_processed = session.windows_completed();

    // Finalize and get audio embedding
    #[allow(unused_variables)]
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
    #[allow(unused_variables)]
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

    #[allow(unused_mut)]
    let mut text_stored = false;
    #[allow(unused_mut)]
    let mut audio_stored = false;

    // Store embeddings if requested
    #[cfg(any(feature = "storage", feature = "storage-file"))]
    if request.store {
        if let Some(ref storage) = state.storage {
            use crate::storage::{TrackMetadata, AUDIO_COLLECTION, TEXT_COLLECTION};

            // Clone metadata fields from session (can't move from MutexGuard)
            let name = session.metadata.name.clone();
            let artists = session.metadata.artists.clone();
            let album = session.metadata.album.clone();
            let genres = session.metadata.genres.clone();

            // Build metadata for storage
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
                    info!(
                        track_id = %track_id,
                        collection = AUDIO_COLLECTION,
                        "Audio embedding saved"
                    );
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
                info!(
                    track_id = %track_id,
                    collection = TEXT_COLLECTION,
                    "Text embedding saved"
                );
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

    // Remove session (quick DashMap operation)
    match state.stream_manager.remove_session(&session_uuid) {
        Some(locked_session) => {
            // Try to get track_id for logging (non-blocking)
            let track_id = locked_session
                .try_lock()
                .map(|s| s.track_id.clone())
                .unwrap_or_else(|_| "<locked>".to_string());
            info!(
                session_id = %session_uuid,
                track_id = %track_id,
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

    // Get session (quick DashMap lookup)
    let locked_session = match state.stream_manager.get_session(&session_uuid) {
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

    // Lock briefly to read status
    let session = locked_session.lock().await;
    MsgPack(StreamStatusResponse {
        session_id: session.id.to_string(),
        track_id: session.track_id.clone(),
        status: session.status,
        buffered_seconds: session.buffered_seconds(),
        windows_completed: session.windows_completed(),
        age_seconds: session.age_seconds(),
    })
    .into_response()
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
        let manager = StreamSessionManager::new();

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
            false, // replace_existing
        );

        assert!(result.is_ok());
        assert_eq!(manager.session_count(), 1);

        // Duplicate should fail (replace_existing = false)
        let result2 = manager.create_session(
            "track_123".to_string(),
            metadata,
            AudioFormat::PcmS16Le,
            44100,
            2,
            false, // replace_existing
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
