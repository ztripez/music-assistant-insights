//! WebSocket-based audio streaming for crash-resistant ingestion.
//!
//! This module provides a WebSocket endpoint for streaming audio from Music Assistant.
//! Audio is persisted to disk as it arrives, and queued for processing.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tracing::{debug, error, info, warn};

use super::AppState;
use crate::queue::{SessionMetadata, SessionRecord, SessionStatus};

/// Maximum size of a single audio frame (64KB - reasonable for ~0.5s of audio)
const MAX_FRAME_SIZE: usize = 64 * 1024;

/// Maximum total file size for a session (100MB - ~10 minutes of 48kHz stereo)
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Header message sent at the start of a WebSocket audio stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamHeader {
    /// Unique session ID from Music Assistant (queue_item_id)
    pub queue_item_id: String,
    /// Track ID from Music Assistant library
    pub track_id: String,
    /// Player/queue ID (identifies which player)
    pub queue_id: String,
    /// Sample rate of incoming PCM audio
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u8,
    /// Track metadata for embedding generation
    pub metadata: SessionMetadata,
}

/// Response sent back after successful session start
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamAck {
    /// Whether the session was accepted
    pub accepted: bool,
    /// Error message if not accepted
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Handle WebSocket upgrade for audio streaming
pub async fn audio_stream_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_audio_socket(socket, state))
}

/// Handle an audio streaming WebSocket connection
async fn handle_audio_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    // Wait for header message
    let header: StreamHeader = match receiver.next().await {
        Some(Ok(Message::Binary(data))) => {
            match rmp_serde::from_slice(&data) {
                Ok(h) => h,
                Err(e) => {
                    error!(error = %e, "Failed to parse stream header");
                    let ack = StreamAck {
                        accepted: false,
                        error: Some(format!("Invalid header: {}", e)),
                    };
                    let _ = sender
                        .send(Message::Binary(rmp_serde::to_vec(&ack).unwrap().into()))
                        .await;
                    return;
                }
            }
        }
        Some(Ok(Message::Text(text))) => {
            // Also accept JSON for easier debugging
            match serde_json::from_str(&text) {
                Ok(h) => h,
                Err(e) => {
                    error!(error = %e, "Failed to parse stream header (JSON)");
                    let ack = StreamAck {
                        accepted: false,
                        error: Some(format!("Invalid header: {}", e)),
                    };
                    let _ = sender
                        .send(Message::Text(serde_json::to_string(&ack).unwrap().into()))
                        .await;
                    return;
                }
            }
        }
        Some(Ok(msg)) => {
            error!(?msg, "Expected binary header message");
            let ack = StreamAck {
                accepted: false,
                error: Some("Expected binary header message".to_string()),
            };
            let _ = sender
                .send(Message::Binary(rmp_serde::to_vec(&ack).unwrap().into()))
                .await;
            return;
        }
        Some(Err(e)) => {
            error!(error = %e, "WebSocket error receiving header");
            return;
        }
        None => {
            debug!("WebSocket closed before header received");
            return;
        }
    };

    info!(
        queue_item_id = %header.queue_item_id,
        track_id = %header.track_id,
        queue_id = %header.queue_id,
        sample_rate = header.sample_rate,
        channels = header.channels,
        "Audio stream session started"
    );

    // Check if queue is available
    let queue = match &state.audio_queue {
        Some(q) => q.clone(),
        None => {
            error!("Audio queue not initialized");
            let ack = StreamAck {
                accepted: false,
                error: Some("Audio queue not available".to_string()),
            };
            let _ = sender
                .send(Message::Binary(rmp_serde::to_vec(&ack).unwrap().into()))
                .await;
            return;
        }
    };

    // Create PCM file path
    let pcm_path = queue.audio_dir().join(format!("{}.pcm", header.queue_item_id));

    // Open file for writing
    let mut file = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&pcm_path)
        .await
    {
        Ok(f) => f,
        Err(e) => {
            error!(error = %e, path = %pcm_path.display(), "Failed to create PCM file");
            let ack = StreamAck {
                accepted: false,
                error: Some(format!("Failed to create audio file: {}", e)),
            };
            let _ = sender
                .send(Message::Binary(rmp_serde::to_vec(&ack).unwrap().into()))
                .await;
            return;
        }
    };

    // Send acknowledgment
    let ack = StreamAck {
        accepted: true,
        error: None,
    };
    if let Err(e) = sender
        .send(Message::Binary(rmp_serde::to_vec(&ack).unwrap().into()))
        .await
    {
        error!(error = %e, "Failed to send ack");
        return;
    }

    // Process incoming audio frames
    let mut total_bytes: u64 = 0;
    let mut frame_count: u64 = 0;

    loop {
        match receiver.next().await {
            Some(Ok(Message::Binary(data))) => {
                // Validate frame size
                if data.len() > MAX_FRAME_SIZE {
                    warn!(
                        queue_item_id = %header.queue_item_id,
                        frame_size = data.len(),
                        max_size = MAX_FRAME_SIZE,
                        "Frame too large, closing session"
                    );
                    break;
                }

                // Check total file size limit
                if total_bytes + data.len() as u64 > MAX_FILE_SIZE {
                    warn!(
                        queue_item_id = %header.queue_item_id,
                        total_bytes,
                        max_size = MAX_FILE_SIZE,
                        "Session exceeded max file size, closing"
                    );
                    break;
                }

                // Write PCM data to file
                if let Err(e) = file.write_all(&data).await {
                    error!(error = %e, "Failed to write PCM data");
                    break;
                }
                total_bytes += data.len() as u64;
                frame_count += 1;

                if frame_count % 100 == 0 {
                    debug!(
                        queue_item_id = %header.queue_item_id,
                        frames = frame_count,
                        bytes = total_bytes,
                        "Streaming progress"
                    );
                }
            }
            Some(Ok(Message::Close(_))) => {
                debug!(
                    queue_item_id = %header.queue_item_id,
                    "Client closed connection"
                );
                break;
            }
            Some(Ok(Message::Ping(data))) => {
                if let Err(e) = sender.send(Message::Pong(data)).await {
                    warn!(error = %e, "Failed to send pong");
                }
            }
            Some(Ok(_)) => {
                // Ignore other message types
            }
            Some(Err(e)) => {
                error!(error = %e, "WebSocket error");
                break;
            }
            None => {
                debug!("WebSocket stream ended");
                break;
            }
        }
    }

    // Flush and close file
    if let Err(e) = file.flush().await {
        warn!(error = %e, "Failed to flush PCM file");
    }
    drop(file);

    // Only queue if we received some data
    if total_bytes > 0 {
        // Create session record and queue it
        let record = SessionRecord {
            queue_item_id: header.queue_item_id.clone(),
            track_id: header.track_id.clone(),
            queue_id: header.queue_id.clone(),
            sample_rate: header.sample_rate,
            channels: header.channels,
            pcm_path: pcm_path.clone(),
            status: SessionStatus::Pending,
            created_at: chrono::Utc::now().timestamp(),
            metadata: header.metadata,
        };

        match queue.push(record) {
            Ok(()) => {
                info!(
                    queue_item_id = %header.queue_item_id,
                    track_id = %header.track_id,
                    bytes = total_bytes,
                    frames = frame_count,
                    pcm_path = %pcm_path.display(),
                    "Audio session queued for processing"
                );
            }
            Err(e) => {
                error!(
                    error = %e,
                    queue_item_id = %header.queue_item_id,
                    "Failed to queue audio session"
                );
                // Clean up the PCM file since we couldn't queue it
                if let Err(e) = tokio::fs::remove_file(&pcm_path).await {
                    warn!(error = %e, "Failed to clean up PCM file after queue error");
                }
            }
        }
    } else {
        info!(
            queue_item_id = %header.queue_item_id,
            "No audio data received, not queuing"
        );
        // Clean up empty PCM file
        let _ = tokio::fs::remove_file(&pcm_path).await;
    }
}
