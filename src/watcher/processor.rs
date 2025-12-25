//! Track processing pipeline: decode -> embed -> store.

use std::path::Path;
use std::sync::Arc;

use rubato::{FftFixedIn, Resampler};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::inference::{format_track_metadata, AudioData, AudioFormat, ClapModel, Embedding, TextTrackMetadata};
use crate::storage::{TrackMetadata, VectorStorage, TEXT_COLLECTION, AUDIO_COLLECTION};

use super::decoder::AudioDecoder;
use super::metadata::ExtractedMetadata;
use super::{generate_track_id, WatcherError};

/// Target sample rate for CLAP model (48kHz)
const TARGET_SAMPLE_RATE: u32 = 48000;

/// Window size in samples at target sample rate (10 seconds)
const WINDOW_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * 10;

/// Track processor that handles the decode -> embed -> store pipeline
pub struct TrackProcessor {
    model: Arc<RwLock<Option<Arc<ClapModel>>>>,
    storage: Option<Arc<dyn VectorStorage + Send + Sync>>,
}

/// Result of processing a single track
#[derive(Debug)]
pub struct ProcessedTrack {
    /// Generated track ID
    pub track_id: String,
    /// Extracted metadata
    pub metadata: ExtractedMetadata,
    /// Text embedding (if generated)
    pub text_embedding: Option<Embedding>,
    /// Audio embedding (if generated)
    pub audio_embedding: Option<Embedding>,
    /// Audio duration in seconds
    pub duration_s: f32,
}

impl TrackProcessor {
    /// Create a new track processor
    pub fn new(
        model: Arc<RwLock<Option<Arc<ClapModel>>>>,
        storage: Option<Arc<dyn VectorStorage + Send + Sync>>,
    ) -> Self {
        Self { model, storage }
    }

    /// Process a single audio file
    pub async fn process_file(&self, path: &Path) -> Result<ProcessedTrack, WatcherError> {
        let path_str = path.to_string_lossy().to_string();
        info!(path = %path_str, "Processing audio file");

        // Generate track ID from path
        let track_id = generate_track_id(path);
        debug!(track_id = %track_id, "Generated track ID");

        // Extract metadata
        let metadata = ExtractedMetadata::from_file(path)?;
        debug!(
            title = ?metadata.title,
            artists = ?metadata.artists,
            album = ?metadata.album,
            "Extracted metadata"
        );

        // Decode audio
        let decoded = AudioDecoder::decode_file(path)?;
        debug!(
            duration_s = decoded.duration_s,
            sample_rate = decoded.sample_rate,
            channels = decoded.channels,
            "Decoded audio"
        );

        // Convert to mono
        let mono_samples = decoded.to_mono();

        // Resample to 48kHz
        let resampled = resample_audio(&mono_samples, decoded.sample_rate, TARGET_SAMPLE_RATE)?;
        debug!(samples = resampled.len(), "Resampled audio to 48kHz");

        // Get model (if available)
        let model_guard = self.model.read().await;
        let model = model_guard
            .as_ref()
            .ok_or(WatcherError::ModelNotLoaded)?
            .clone();
        drop(model_guard);

        // Generate text embedding
        let text_metadata = TextTrackMetadata {
            name: metadata.title.clone().unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Unknown")
                    .to_string()
            }),
            artists: if metadata.artists.is_empty() {
                vec!["Unknown Artist".to_string()]
            } else {
                metadata.artists.clone()
            },
            album: metadata.album.clone(),
            genres: metadata.genres.clone(),
            mood: None,
        };

        let formatted_text = format_track_metadata(&text_metadata);
        let text_embedding = tokio::task::spawn_blocking({
            let model = model.clone();
            let text = formatted_text.clone();
            move || model.text_embedding(&text)
        })
        .await
        .map_err(|e| WatcherError::EmbeddingError(e.to_string()))?
        .map_err(|e| WatcherError::EmbeddingError(e.to_string()))?;

        debug!("Generated text embedding");

        // Generate audio embedding
        let audio_embedding = tokio::task::spawn_blocking({
            let model = model.clone();
            let samples = resampled.clone();
            move || generate_audio_embedding(&model, &samples)
        })
        .await
        .map_err(|e| WatcherError::EmbeddingError(e.to_string()))?
        .map_err(|e| WatcherError::EmbeddingError(e.to_string()))?;

        debug!("Generated audio embedding");

        // Store embeddings (if storage is available)
        if let Some(ref storage) = self.storage {
            let mut track_meta = TrackMetadata::new(track_id.clone(), text_metadata.name.clone())
                .with_artists(text_metadata.artists.clone())
                .with_genres(text_metadata.genres.clone());

            if let Some(ref album) = text_metadata.album {
                track_meta = track_meta.with_album(album.clone());
            }

            // Store text embedding
            storage
                .upsert(TEXT_COLLECTION, &track_id, text_embedding.data(), track_meta.clone())
                .await
                .map_err(|e| WatcherError::StorageError(e.to_string()))?;

            // Store audio embedding
            storage
                .upsert(AUDIO_COLLECTION, &track_id, audio_embedding.data(), track_meta)
                .await
                .map_err(|e| WatcherError::StorageError(e.to_string()))?;

            debug!(track_id = %track_id, "Stored embeddings");
        } else {
            warn!("No storage configured, embeddings not persisted");
        }

        info!(
            track_id = %track_id,
            title = ?metadata.title,
            duration_s = decoded.duration_s,
            "Successfully processed track"
        );

        Ok(ProcessedTrack {
            track_id,
            metadata,
            text_embedding: Some(text_embedding),
            audio_embedding: Some(audio_embedding),
            duration_s: decoded.duration_s,
        })
    }
}

/// Resample audio to target sample rate using rubato
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, WatcherError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    // Calculate resampling parameters
    let chunk_size = 1024;

    // Create resampler
    let mut resampler = FftFixedIn::<f32>::new(from_rate as usize, to_rate as usize, chunk_size, 2, 1)
        .map_err(|e| WatcherError::ResamplingError(e.to_string()))?;

    let mut output = Vec::new();
    let mut position = 0;

    // Process in chunks
    while position < samples.len() {
        let end = (position + chunk_size).min(samples.len());
        let chunk = &samples[position..end];

        // Pad if necessary
        let input = if chunk.len() < chunk_size {
            let mut padded = chunk.to_vec();
            padded.resize(chunk_size, 0.0);
            vec![padded]
        } else {
            vec![chunk.to_vec()]
        };

        let result = resampler
            .process(&input, None)
            .map_err(|e| WatcherError::ResamplingError(e.to_string()))?;

        if !result.is_empty() {
            output.extend_from_slice(&result[0]);
        }

        position += chunk_size;
    }

    Ok(output)
}

/// Generate audio embedding from resampled samples
fn generate_audio_embedding(model: &ClapModel, samples: &[f32]) -> Result<Embedding, WatcherError> {
    // Process in windows
    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    for chunk in samples.chunks(WINDOW_SAMPLES) {
        // Pad short chunks
        let window = if chunk.len() < WINDOW_SAMPLES {
            let mut padded = chunk.to_vec();
            padded.resize(WINDOW_SAMPLES, 0.0);
            padded
        } else {
            chunk.to_vec()
        };

        // Create AudioData for the model (convert f32 samples to bytes)
        let data: Vec<u8> = window.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let audio_data = AudioData {
            format: AudioFormat::PcmF32Le,
            sample_rate: TARGET_SAMPLE_RATE,
            channels: 1,
            data,
        };

        let embedding = model
            .audio_embedding(&audio_data)
            .map_err(|e| WatcherError::EmbeddingError(e.to_string()))?;

        embeddings.push(embedding.into_data());
    }

    if embeddings.is_empty() {
        return Err(WatcherError::EmbeddingError(
            "No audio windows processed".to_string(),
        ));
    }

    // Average embeddings
    let dim = embeddings[0].len();
    let mut averaged = vec![0.0f32; dim];
    for emb in &embeddings {
        for (i, &v) in emb.iter().enumerate() {
            averaged[i] += v;
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

    Embedding::new(averaged).map_err(|e| WatcherError::EmbeddingError(e.to_string()))
}
