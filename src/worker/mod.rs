//! Background worker for processing queued audio sessions.
//!
//! The worker polls the redb queue for pending sessions, processes them through
//! CLAP inference, and stores the resulting embeddings in the vector database.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::queue::{AudioQueue, SessionRecord};

#[cfg(feature = "inference")]
use crate::inference::ClapModel;

#[cfg(any(feature = "storage", feature = "storage-file"))]
use crate::storage::VectorStorage;

/// Configuration for the audio worker
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// How often to poll the queue when empty (in milliseconds)
    pub poll_interval_ms: u64,
    /// Minimum audio duration in seconds to process (shorter is discarded)
    pub min_duration_s: f32,
    /// Number of concurrent workers (default: 1)
    pub concurrency: usize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 1000,
            min_duration_s: 3.0,
            concurrency: 1,
        }
    }
}

/// Shared state for the worker
pub struct WorkerState {
    pub queue: Arc<AudioQueue>,
    #[cfg(feature = "inference")]
    pub model: Arc<RwLock<Option<Arc<ClapModel>>>>,
    #[cfg(any(feature = "storage", feature = "storage-file"))]
    pub storage: Option<Arc<Box<dyn VectorStorage>>>,
    pub config: WorkerConfig,
}

/// Handle for controlling the worker
pub struct WorkerHandle {
    shutdown_tx: tokio::sync::watch::Sender<bool>,
}

impl WorkerHandle {
    /// Signal the worker to shut down gracefully
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }
}

/// Spawn the background worker task
#[cfg(feature = "inference")]
pub fn spawn_worker(state: Arc<WorkerState>) -> WorkerHandle {
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    for worker_id in 0..state.config.concurrency {
        let state = state.clone();
        let mut rx = shutdown_rx.clone();

        tokio::spawn(async move {
            info!(worker_id, "Audio processing worker started");

            loop {
                // Check for shutdown signal
                if *rx.borrow() {
                    info!(worker_id, "Worker received shutdown signal");
                    break;
                }

                // Try to get a pending session
                match state.queue.pop_pending() {
                    Ok(Some(session)) => {
                        info!(
                            worker_id,
                            queue_item_id = %session.queue_item_id,
                            track_id = %session.track_id,
                            "Processing audio session"
                        );

                        if let Err(e) = process_session(&state, session).await {
                            error!(worker_id, error = %e, "Failed to process session");
                        }
                    }
                    Ok(None) => {
                        // Queue is empty, wait before polling again
                        tokio::select! {
                            _ = tokio::time::sleep(Duration::from_millis(state.config.poll_interval_ms)) => {}
                            _ = rx.changed() => {
                                if *rx.borrow() {
                                    info!(worker_id, "Worker received shutdown signal during sleep");
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!(worker_id, error = %e, "Error polling queue");
                        tokio::time::sleep(Duration::from_millis(state.config.poll_interval_ms)).await;
                    }
                }
            }

            info!(worker_id, "Audio processing worker stopped");
        });
    }

    WorkerHandle { shutdown_tx }
}

/// Process a single audio session
#[cfg(feature = "inference")]
async fn process_session(state: &WorkerState, session: SessionRecord) -> Result<(), String> {
    use crate::inference::audio::{AudioData, AudioFormat};
    use crate::storage::{TrackMetadata, AUDIO_COLLECTION};

    let queue_item_id = session.queue_item_id.clone();
    let track_id = session.track_id.clone();
    let pcm_path = session.pcm_path.clone();

    // Read PCM file
    let pcm_data = tokio::fs::read(&pcm_path)
        .await
        .map_err(|e| format!("Failed to read PCM file: {}", e))?;

    if pcm_data.is_empty() {
        // Clean up empty file and remove from queue
        let _ = tokio::fs::remove_file(&pcm_path).await;
        state.queue.remove(&queue_item_id).map_err(|e| e.to_string())?;
        return Ok(());
    }

    // Calculate duration
    let bytes_per_sample = 2; // s16le
    let samples = pcm_data.len() / bytes_per_sample / session.channels as usize;
    let duration_s = samples as f32 / session.sample_rate as f32;

    if duration_s < state.config.min_duration_s {
        info!(
            queue_item_id = %queue_item_id,
            duration_s,
            min_duration_s = state.config.min_duration_s,
            "Audio too short, skipping"
        );
        // Clean up and remove from queue
        let _ = tokio::fs::remove_file(&pcm_path).await;
        state.queue.remove(&queue_item_id).map_err(|e| e.to_string())?;
        return Ok(());
    }

    // Get model
    let model_guard = state.model.read().await;
    let model = model_guard
        .as_ref()
        .ok_or_else(|| "Model not loaded".to_string())?
        .clone();
    drop(model_guard);

    // Create AudioData struct for the inference API
    let audio_data = AudioData {
        format: AudioFormat::PcmS16Le,
        sample_rate: session.sample_rate,
        channels: session.channels,
        data: pcm_data,
    };

    // Run inference in blocking task
    let embedding = tokio::task::spawn_blocking(move || {
        model.audio_embedding(&audio_data)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
    .map_err(|e| format!("Inference error: {}", e))?;

    info!(
        queue_item_id = %queue_item_id,
        track_id = %track_id,
        duration_s,
        "Generated audio embedding"
    );

    // Store in vector database
    #[cfg(any(feature = "storage", feature = "storage-file"))]
    if let Some(ref storage) = state.storage {
        let metadata = TrackMetadata::new(track_id.clone(), session.metadata.name.clone())
            .with_artists(session.metadata.artists.clone())
            .with_genres(session.metadata.genres.clone());

        let metadata = if let Some(ref album) = session.metadata.album {
            metadata.with_album(album.clone())
        } else {
            metadata
        };

        storage
            .upsert(AUDIO_COLLECTION, &track_id, embedding.data(), metadata)
            .await
            .map_err(|e| format!("Storage error: {}", e))?;

        info!(
            queue_item_id = %queue_item_id,
            track_id = %track_id,
            collection = AUDIO_COLLECTION,
            "Audio embedding stored"
        );
    }

    // Clean up PCM file
    if let Err(e) = tokio::fs::remove_file(&pcm_path).await {
        warn!(error = %e, path = %pcm_path.display(), "Failed to remove PCM file");
    }

    // Remove from queue
    state.queue.remove(&queue_item_id).map_err(|e| e.to_string())?;

    info!(
        queue_item_id = %queue_item_id,
        track_id = %track_id,
        "Audio session processed successfully"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.poll_interval_ms, 1000);
        assert_eq!(config.min_duration_s, 3.0);
        assert_eq!(config.concurrency, 1);
    }
}
