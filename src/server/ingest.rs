//! Unified track ingestion handler.
//!
//! Combines text embedding, audio embedding, storage, and mood classification
//! into a single efficient operation.

use axum::extract::State;

use crate::error::AppError;

#[cfg(all(feature = "inference", any(feature = "storage", feature = "storage-file")))]
use crate::types::{
    BatchIngestRequest, BatchIngestResponse, BatchIngestResult, IngestRequest, IngestResponse,
};

#[cfg(all(feature = "inference", any(feature = "storage", feature = "storage-file")))]
use super::extractors::MsgPackExtractor;
#[cfg(all(feature = "inference", any(feature = "storage", feature = "storage-file")))]
use super::routes::MsgPack;
use super::AppState;

#[cfg(all(feature = "inference", any(feature = "storage", feature = "storage-file")))]
use crate::inference::{format_track_metadata, AudioData, AudioFormat, TextTrackMetadata};

#[cfg(all(feature = "inference", any(feature = "storage", feature = "storage-file")))]
use crate::mood::MoodClassifier;

#[cfg(all(feature = "inference", any(feature = "storage", feature = "storage-file")))]
use crate::storage::{TrackMetadata, AUDIO_COLLECTION, TEXT_COLLECTION};

#[cfg(all(feature = "inference", any(feature = "storage", feature = "storage-file")))]
use tracing::{debug, error, info};

/// POST /api/v1/tracks/ingest
///
/// Unified track ingestion endpoint that performs all processing in one call:
/// 1. Generate text embedding from metadata
/// 2. Generate audio embedding (if audio provided)
/// 3. Store embeddings (unless skip_storage is true)
/// 4. Classify mood (if classify_mood is true)
#[cfg(all(feature = "inference", any(feature = "storage", feature = "storage-file")))]
pub async fn ingest(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<IngestRequest>,
) -> Result<MsgPack<IngestResponse>, AppError> {
    info!(track_id = %req.track_id, has_audio = req.audio.is_some(), "Ingesting track");

    // Get model for embedding generation
    let model = {
        let guard = state.model.read().await;
        guard
            .as_ref()
            .ok_or_else(|| AppError::Internal("Model not loaded".to_string()))?
            .clone()
    };

    // Format metadata for text embedding
    let track_meta = TextTrackMetadata {
        name: req.metadata.name.clone(),
        artists: req.metadata.artists.clone(),
        album: req.metadata.album.clone(),
        genres: req.metadata.genres.clone(),
        mood: None,
    };
    let formatted_text = format_track_metadata(&track_meta);

    // Generate text embedding
    let text_embedding = tokio::task::spawn_blocking({
        let model = model.clone();
        let text = formatted_text.clone();
        move || model.text_embedding(&text)
    })
    .await
    .map_err(|e| {
        error!(error = %e, "Text embedding task panicked");
        AppError::Internal(e.to_string())
    })?
    .map_err(|e| {
        error!(error = %e, "Text embedding failed");
        AppError::from(e)
    })?;

    debug!(track_id = %req.track_id, "Text embedding generated");

    // Generate audio embedding if audio provided
    let (audio_embedding, audio_duration_s) = if let Some(audio) = req.audio {
        // Validate audio parameters
        if audio.channels == 0 {
            return Err(AppError::BadRequest(
                "Channel count must be at least 1".to_string(),
            ));
        }
        if audio.sample_rate == 0 {
            return Err(AppError::BadRequest(
                "Sample rate must be greater than 0".to_string(),
            ));
        }

        // Calculate duration
        let samples_per_channel = match audio.format {
            AudioFormat::PcmF32Le => audio.data.len() / 4,
            AudioFormat::PcmS16Le => audio.data.len() / 2,
            AudioFormat::PcmS24Le => audio.data.len() / 3,
        };
        let total_samples = samples_per_channel / audio.channels as usize;
        let duration_s = total_samples as f32 / audio.sample_rate as f32;

        let audio_data: AudioData = audio.into();

        let embedding = tokio::task::spawn_blocking({
            let model = model.clone();
            move || model.audio_embedding(&audio_data)
        })
        .await
        .map_err(|e| {
            error!(error = %e, "Audio embedding task panicked");
            AppError::Internal(e.to_string())
        })?
        .map_err(|e| {
            error!(error = %e, "Audio embedding failed");
            AppError::from(e)
        })?;

        debug!(track_id = %req.track_id, duration_s, "Audio embedding generated");
        (Some(embedding.into_data()), Some(duration_s))
    } else {
        (None, None)
    };

    // Store embeddings unless skip_storage is true
    let mut text_stored = false;
    let mut audio_stored = false;

    if !req.skip_storage {
        if let Some(ref storage) = state.storage {
            // Build storage metadata
            let storage_metadata = TrackMetadata::new(req.track_id.clone(), req.metadata.name.clone())
                .with_artists(req.metadata.artists.clone())
                .with_genres(req.metadata.genres.clone());
            let storage_metadata = if let Some(ref album) = req.metadata.album {
                storage_metadata.with_album(album.clone())
            } else {
                storage_metadata
            };

            // Store text embedding
            storage
                .upsert(
                    TEXT_COLLECTION,
                    &req.track_id,
                    text_embedding.data(),
                    storage_metadata.clone(),
                )
                .await
                .map_err(|e| {
                    error!(error = %e, "Failed to store text embedding");
                    AppError::Internal(e.to_string())
                })?;
            text_stored = true;
            debug!(track_id = %req.track_id, "Text embedding stored");

            // Store audio embedding if present
            if let Some(ref audio_emb) = audio_embedding {
                storage
                    .upsert(AUDIO_COLLECTION, &req.track_id, audio_emb, storage_metadata)
                    .await
                    .map_err(|e| {
                        error!(error = %e, "Failed to store audio embedding");
                        AppError::Internal(e.to_string())
                    })?;
                audio_stored = true;
                debug!(track_id = %req.track_id, "Audio embedding stored");
            }
        }
    }

    // Classify mood if requested
    let mood = if req.classify_mood {
        // Use text embedding for mood classification
        let classifier = MoodClassifier::new(&model);
        let classification =
            classifier.classify(text_embedding.data(), &req.mood_tiers, req.mood_top_k, req.include_va);
        Some(classification)
    } else {
        None
    };

    Ok(MsgPack(IngestResponse {
        track_id: req.track_id,
        text_embedding: text_embedding.into_data(),
        audio_embedding,
        formatted_text,
        audio_duration_s,
        text_stored,
        audio_stored,
        mood,
    }))
}

/// POST /api/v1/tracks/batch-ingest
///
/// Batch ingest multiple tracks with all processing in one call.
#[cfg(all(feature = "inference", any(feature = "storage", feature = "storage-file")))]
pub async fn batch_ingest(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<BatchIngestRequest>,
) -> Result<MsgPack<BatchIngestResponse>, AppError> {
    let total = req.tracks.len();
    info!(count = total, "Batch ingesting tracks");

    // Get model for embedding generation
    let model = {
        let guard = state.model.read().await;
        guard
            .as_ref()
            .ok_or_else(|| AppError::Internal("Model not loaded".to_string()))?
            .clone()
    };

    let mut results = Vec::with_capacity(total);
    let mut succeeded = 0;
    let mut failed = 0;

    for track in req.tracks {
        let track_id = track.track_id.clone();
        let should_classify = track.classify_mood.unwrap_or(req.classify_mood);

        // Format metadata for text embedding
        let track_meta = TextTrackMetadata {
            name: track.metadata.name.clone(),
            artists: track.metadata.artists.clone(),
            album: track.metadata.album.clone(),
            genres: track.metadata.genres.clone(),
            mood: None,
        };
        let formatted_text = format_track_metadata(&track_meta);

        // Generate text embedding
        let text_result = tokio::task::spawn_blocking({
            let model = model.clone();
            let text = formatted_text.clone();
            move || model.text_embedding(&text)
        })
        .await;

        let text_embedding = match text_result {
            Err(e) => {
                error!(error = %e, track_id = %track_id, "Text embedding task panicked");
                failed += 1;
                results.push(BatchIngestResult {
                    track_id,
                    success: false,
                    error: Some(format!("Task panicked: {}", e)),
                    text_embedding: None,
                    audio_embedding: None,
                    text_stored: false,
                    audio_stored: false,
                    mood: None,
                });
                continue;
            }
            Ok(Err(e)) => {
                error!(error = %e, track_id = %track_id, "Text embedding failed");
                failed += 1;
                results.push(BatchIngestResult {
                    track_id,
                    success: false,
                    error: Some(e.to_string()),
                    text_embedding: None,
                    audio_embedding: None,
                    text_stored: false,
                    audio_stored: false,
                    mood: None,
                });
                continue;
            }
            Ok(Ok(emb)) => emb,
        };

        // Generate audio embedding if provided
        let (audio_embedding, _audio_duration) = if let Some(audio) = track.audio {
            if audio.channels == 0 || audio.sample_rate == 0 {
                failed += 1;
                results.push(BatchIngestResult {
                    track_id,
                    success: false,
                    error: Some("Invalid audio parameters".to_string()),
                    text_embedding: None,
                    audio_embedding: None,
                    text_stored: false,
                    audio_stored: false,
                    mood: None,
                });
                continue;
            }

            let audio_data: AudioData = audio.into();
            let audio_result = tokio::task::spawn_blocking({
                let model = model.clone();
                move || model.audio_embedding(&audio_data)
            })
            .await;

            match audio_result {
                Err(e) => {
                    error!(error = %e, track_id = %track_id, "Audio embedding task panicked");
                    failed += 1;
                    results.push(BatchIngestResult {
                        track_id,
                        success: false,
                        error: Some(format!("Audio task panicked: {}", e)),
                        text_embedding: None,
                        audio_embedding: None,
                        text_stored: false,
                        audio_stored: false,
                        mood: None,
                    });
                    continue;
                }
                Ok(Err(e)) => {
                    error!(error = %e, track_id = %track_id, "Audio embedding failed");
                    failed += 1;
                    results.push(BatchIngestResult {
                        track_id,
                        success: false,
                        error: Some(e.to_string()),
                        text_embedding: None,
                        audio_embedding: None,
                        text_stored: false,
                        audio_stored: false,
                        mood: None,
                    });
                    continue;
                }
                Ok(Ok(emb)) => (Some(emb.into_data()), None::<f32>),
            }
        } else {
            (None, None)
        };

        // Store embeddings unless skip_storage
        let mut text_stored = false;
        let mut audio_stored = false;

        if !req.skip_storage {
            if let Some(ref storage) = state.storage {
                let storage_metadata =
                    TrackMetadata::new(track_id.clone(), track.metadata.name.clone())
                        .with_artists(track.metadata.artists.clone())
                        .with_genres(track.metadata.genres.clone());
                let storage_metadata = if let Some(ref album) = track.metadata.album {
                    storage_metadata.with_album(album.clone())
                } else {
                    storage_metadata
                };

                // Store text embedding
                if let Err(e) = storage
                    .upsert(
                        TEXT_COLLECTION,
                        &track_id,
                        text_embedding.data(),
                        storage_metadata.clone(),
                    )
                    .await
                {
                    error!(error = %e, track_id = %track_id, "Failed to store text embedding");
                    failed += 1;
                    results.push(BatchIngestResult {
                        track_id,
                        success: false,
                        error: Some(e.to_string()),
                        text_embedding: None,
                        audio_embedding: None,
                        text_stored: false,
                        audio_stored: false,
                        mood: None,
                    });
                    continue;
                }
                text_stored = true;

                // Store audio embedding if present
                if let Some(ref audio_emb) = audio_embedding {
                    if let Err(e) = storage
                        .upsert(AUDIO_COLLECTION, &track_id, audio_emb, storage_metadata)
                        .await
                    {
                        error!(error = %e, track_id = %track_id, "Failed to store audio embedding");
                        // Continue anyway, text was stored
                    } else {
                        audio_stored = true;
                    }
                }
            }
        }

        // Classify mood if requested
        let mood = if should_classify {
            let classifier = MoodClassifier::new(&model);
            Some(classifier.classify(
                text_embedding.data(),
                &req.mood_tiers,
                req.mood_top_k,
                req.include_va,
            ))
        } else {
            None
        };

        succeeded += 1;
        results.push(BatchIngestResult {
            track_id,
            success: true,
            error: None,
            text_embedding: Some(text_embedding.into_data()),
            audio_embedding,
            text_stored,
            audio_stored,
            mood,
        });
    }

    info!(total, succeeded, failed, "Batch ingest complete");

    Ok(MsgPack(BatchIngestResponse {
        results,
        total,
        succeeded,
        failed,
    }))
}

/// Fallback handler when required features are disabled
#[cfg(not(all(feature = "inference", any(feature = "storage", feature = "storage-file"))))]
pub async fn ingest(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Both inference and storage features must be enabled".to_string(),
    ))
}

#[cfg(not(all(feature = "inference", any(feature = "storage", feature = "storage-file"))))]
pub async fn batch_ingest(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Both inference and storage features must be enabled".to_string(),
    ))
}
