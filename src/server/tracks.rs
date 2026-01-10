//! Track embedding route handlers.

use axum::extract::{Path, Query, State};
use serde::Deserialize;

use crate::error::AppError;

#[cfg(feature = "storage")]
use crate::types::{
    BatchUpsertRequest, BatchUpsertResponse, BatchUpsertResult, DeleteRequest, DeleteResponse,
    GetTrackResponse, SearchRequest, SearchResponse, UpsertRequest, UpsertResponse,
};

#[cfg(all(feature = "inference", feature = "storage"))]
use crate::types::{
    BatchEmbedTextRequest, BatchEmbedTextResponse, BatchEmbedTextResult, EmbedTextAndStoreRequest,
    EmbedTextAndStoreResponse, TextSearchRequest,
};

#[cfg(feature = "storage")]
use super::extractors::MsgPackExtractor;
#[cfg(feature = "storage")]
use super::routes::MsgPack;
use super::AppState;

#[cfg(feature = "storage")]
use crate::storage::{AUDIO_COLLECTION, EMBEDDING_DIM, TEXT_COLLECTION};

/// Maximum batch size for batch operations (prevents memory exhaustion)
#[cfg(feature = "storage")]
const MAX_BATCH_SIZE: usize = 100;

#[cfg(all(feature = "inference", feature = "storage"))]
use crate::inference::{format_track_metadata, TextTrackMetadata};

#[cfg(all(feature = "inference", feature = "storage"))]
use crate::storage::TrackMetadata;

#[cfg(all(feature = "inference", feature = "storage"))]
use crate::mood::MoodTier;

#[cfg(feature = "storage")]
use tracing::{debug, error, info};

#[cfg(feature = "storage")]
use std::collections::HashMap;

/// Audio embedding score boost factor (audio matches are more reliable)
#[cfg(feature = "storage")]
const AUDIO_SCORE_BOOST: f32 = 1.15;

/// Search both text and audio collections and merge results.
/// Audio results get a score boost since they're more reliable.
/// Results are deduplicated by track_id, keeping the highest score.
#[cfg(feature = "storage")]
async fn merged_search(
    storage: &std::sync::Arc<super::BoxedStorage>,
    embedding: &[f32],
    limit: usize,
    filter: Option<crate::storage::SearchFilter>,
) -> Result<Vec<crate::storage::SearchResult>, crate::error::AppError> {
    use crate::storage::SearchResult;

    // Search both collections in parallel
    let text_filter = filter.clone();
    let audio_filter = filter;

    let (text_results, audio_results) = tokio::join!(
        storage.search(TEXT_COLLECTION, embedding, limit * 2, text_filter),
        storage.search(AUDIO_COLLECTION, embedding, limit * 2, audio_filter)
    );

    // Merge results into a HashMap by track_id
    let mut merged: HashMap<String, SearchResult> = HashMap::new();

    // Add text results
    if let Ok(results) = text_results {
        for result in results {
            merged.insert(result.track_id.clone(), result);
        }
    }

    // Add audio results with score boost (overwrite if higher score)
    if let Ok(results) = audio_results {
        for mut result in results {
            // Apply audio score boost
            result.score = (result.score * AUDIO_SCORE_BOOST).min(1.0);

            // Keep result with higher score
            if let Some(existing) = merged.get(&result.track_id) {
                if result.score > existing.score {
                    merged.insert(result.track_id.clone(), result);
                }
            } else {
                merged.insert(result.track_id.clone(), result);
            }
        }
    }

    // Sort by score descending and take limit
    let mut results: Vec<SearchResult> = merged.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    Ok(results)
}

/// POST /api/v1/tracks/upsert
///
/// Upsert track embedding(s) to the vector store.
#[cfg(feature = "storage")]
pub async fn upsert(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<UpsertRequest>,
) -> Result<MsgPack<UpsertResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?;

    info!(track_id = %req.track_id, "Upserting track embeddings");

    let mut text_stored = false;
    let mut audio_stored = false;

    let metadata = req.metadata.into_storage(req.track_id.clone());

    // Store text embedding if provided
    if let Some(ref embedding) = req.text_embedding {
        if embedding.len() != EMBEDDING_DIM {
            return Err(AppError::BadRequest(format!(
                "Text embedding dimension mismatch: expected {}, got {}",
                EMBEDDING_DIM,
                embedding.len()
            )));
        }

        storage
            .upsert(TEXT_COLLECTION, &req.track_id, embedding, metadata.clone())
            .await
            .map_err(|e| {
                error!(error = %e, "Failed to upsert text embedding");
                AppError::Internal(e.to_string())
            })?;

        text_stored = true;
        debug!(track_id = %req.track_id, "Text embedding stored");
    }

    // Store audio embedding if provided
    if let Some(ref embedding) = req.audio_embedding {
        if embedding.len() != EMBEDDING_DIM {
            return Err(AppError::BadRequest(format!(
                "Audio embedding dimension mismatch: expected {}, got {}",
                EMBEDDING_DIM,
                embedding.len()
            )));
        }

        storage
            .upsert(AUDIO_COLLECTION, &req.track_id, embedding, metadata)
            .await
            .map_err(|e| {
                error!(error = %e, "Failed to upsert audio embedding");
                AppError::Internal(e.to_string())
            })?;

        audio_stored = true;
        debug!(track_id = %req.track_id, "Audio embedding stored");
    }

    if !text_stored && !audio_stored {
        return Err(AppError::BadRequest(
            "At least one embedding (text or audio) must be provided".to_string(),
        ));
    }

    Ok(MsgPack(UpsertResponse {
        track_id: req.track_id,
        text_stored,
        audio_stored,
    }))
}

/// POST /api/v1/tracks/search
///
/// Search for similar tracks by embedding.
#[cfg(feature = "storage")]
pub async fn search(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<SearchRequest>,
) -> Result<MsgPack<SearchResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?;

    // Validate embedding dimension
    if req.embedding.len() != EMBEDDING_DIM {
        return Err(AppError::BadRequest(format!(
            "Embedding dimension mismatch: expected {}, got {}",
            EMBEDDING_DIM,
            req.embedding.len()
        )));
    }

    // Determine collection
    let collection = match req.collection.as_str() {
        "text" => TEXT_COLLECTION,
        "audio" => AUDIO_COLLECTION,
        _ => {
            return Err(AppError::BadRequest(format!(
                "Invalid collection '{}': must be 'text' or 'audio'",
                req.collection
            )))
        }
    };

    let limit = req.limit.min(100); // Cap at 100 results

    debug!(collection, limit, "Searching for similar tracks");

    let filter = req.filter.map(Into::into);

    let results = storage
        .search(collection, &req.embedding, limit, filter)
        .await
        .map_err(|e| {
            error!(error = %e, "Search failed");
            AppError::Internal(e.to_string())
        })?;

    let count = results.len();

    Ok(MsgPack(SearchResponse { results, count }))
}

/// POST /api/v1/tracks/search-text
///
/// Search for similar tracks using a text query.
/// Searches both text and audio collections, with audio results getting a score boost.
#[cfg(all(feature = "inference", feature = "storage"))]
pub async fn text_search(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<TextSearchRequest>,
) -> Result<MsgPack<SearchResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?;

    // Acquire read lock and clone model
    let model = {
        let guard = state.model.read().await;
        guard
            .as_ref()
            .ok_or_else(|| AppError::Internal("Model not loaded".to_string()))?
            .clone()
    };

    debug!(query = %req.query, "Text search request");

    // Generate embedding from query text
    let query_text = req.query.clone();
    let embedding = tokio::task::spawn_blocking(move || model.text_embedding(&query_text))
        .await
        .map_err(|e| AppError::Internal(format!("Task join error: {}", e)))?
        .map_err(|e| {
            error!(error = %e, "Failed to generate query embedding");
            AppError::from(e)
        })?;

    let limit = req.limit.min(100);
    let filter = req.filter.map(Into::into);

    // Search both text and audio collections with merged results
    let results = merged_search(storage, embedding.data(), limit, filter).await?;

    let count = results.len();

    info!(query = %req.query, count, "Text search completed");

    Ok(MsgPack(SearchResponse { results, count }))
}

/// Query parameters for GET /api/v1/tracks/{id}
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct GetTrackQuery {
    /// Include text embedding in response
    #[serde(default)]
    pub include_text: bool,
    /// Include audio embedding in response
    #[serde(default)]
    pub include_audio: bool,
}

/// GET /api/v1/tracks/{id}
///
/// Get track embedding information.
#[cfg(feature = "storage")]
pub async fn get_track(
    State(state): State<AppState>,
    Path(track_id): Path<String>,
    Query(query): Query<GetTrackQuery>,
) -> Result<MsgPack<GetTrackResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?;

    debug!(track_id = %track_id, "Getting track embeddings");

    // Check text collection
    let text_result = storage.get(TEXT_COLLECTION, &track_id).await.map_err(|e| {
        error!(error = %e, "Failed to get text embedding");
        AppError::Internal(e.to_string())
    })?;

    // Check audio collection
    let audio_result = storage
        .get(AUDIO_COLLECTION, &track_id)
        .await
        .map_err(|e| {
            error!(error = %e, "Failed to get audio embedding");
            AppError::Internal(e.to_string())
        })?;

    let has_text = text_result.is_some();
    let has_audio = audio_result.is_some();

    if !has_text && !has_audio {
        return Err(AppError::NotFound(format!(
            "Track '{}' not found",
            track_id
        )));
    }

    // Get metadata from whichever collection has data
    let metadata = text_result
        .as_ref()
        .map(|r| r.metadata.clone())
        .or_else(|| audio_result.as_ref().map(|r| r.metadata.clone()));

    // Include embeddings if requested
    let text_embedding = if query.include_text {
        text_result.map(|r| r.embedding)
    } else {
        None
    };

    let audio_embedding = if query.include_audio {
        audio_result.map(|r| r.embedding)
    } else {
        None
    };

    Ok(MsgPack(GetTrackResponse {
        track_id,
        metadata,
        has_text,
        has_audio,
        text_embedding,
        audio_embedding,
    }))
}

/// DELETE /api/v1/tracks/:id
///
/// Delete track embeddings. Request body is required to specify which
/// collections to delete from (text, audio, or both).
#[cfg(feature = "storage")]
pub async fn delete_track(
    State(state): State<AppState>,
    Path(track_id): Path<String>,
    MsgPackExtractor(req): MsgPackExtractor<DeleteRequest>,
) -> Result<MsgPack<DeleteResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?;

    info!(track_id = %track_id, text = req.text, audio = req.audio, "Deleting track embeddings");

    let mut text_deleted = false;
    let mut audio_deleted = false;

    if req.text {
        // Check if exists first
        let exists = storage
            .exists(TEXT_COLLECTION, &track_id)
            .await
            .map_err(|e| AppError::Internal(e.to_string()))?;

        if exists {
            storage
                .delete(TEXT_COLLECTION, &track_id)
                .await
                .map_err(|e| {
                    error!(error = %e, "Failed to delete text embedding");
                    AppError::Internal(e.to_string())
                })?;
            text_deleted = true;
        }
    }

    if req.audio {
        let exists = storage
            .exists(AUDIO_COLLECTION, &track_id)
            .await
            .map_err(|e| AppError::Internal(e.to_string()))?;

        if exists {
            storage
                .delete(AUDIO_COLLECTION, &track_id)
                .await
                .map_err(|e| {
                    error!(error = %e, "Failed to delete audio embedding");
                    AppError::Internal(e.to_string())
                })?;
            audio_deleted = true;
        }
    }

    Ok(MsgPack(DeleteResponse {
        track_id,
        text_deleted,
        audio_deleted,
    }))
}

/// POST /api/v1/tracks/embed-text
///
/// Generate text embedding from metadata and store it in one operation.
/// Skips embedding generation if metadata_hash matches the stored hash.
/// Requires both inference and storage features.
#[cfg(all(feature = "inference", feature = "storage"))]
pub async fn embed_text_and_store(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<EmbedTextAndStoreRequest>,
) -> Result<MsgPack<EmbedTextAndStoreResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?;

    // Check if hash matches existing - skip if unchanged
    if let Some(ref new_hash) = req.metadata.metadata_hash {
        if let Ok(Some(existing)) = storage.get(TEXT_COLLECTION, &req.track_id).await {
            if let Some(ref stored_hash) = existing.metadata.metadata_hash {
                if stored_hash == new_hash {
                    debug!(track_id = %req.track_id, "Hash unchanged, skipping");
                    return Ok(MsgPack(EmbedTextAndStoreResponse {
                        track_id: req.track_id,
                        stored: false,
                        text: String::new(),
                    }));
                }
            }
        }
    }

    // Acquire read lock and clone model for use in blocking task
    let model = {
        let guard = state.model.read().await;
        guard
            .as_ref()
            .ok_or_else(|| AppError::Internal("Model not loaded".to_string()))?
            .clone()
    };

    info!(track_id = %req.track_id, "Generating and storing text embedding");

    // Format metadata into text for embedding
    let track_meta = TextTrackMetadata {
        name: req.metadata.name.clone(),
        artists: req.metadata.artists.clone(),
        album: req.metadata.album.clone(),
        genres: req.metadata.genres.clone(),
        mood: None,
    };
    let text = format_track_metadata(&track_meta);

    // Generate embedding using the model
    let embedding = tokio::task::spawn_blocking({
        let text = text.clone();
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

    debug!(
        embedding_dim = embedding.data().len(),
        "Text embedding generated"
    );

    // Classify mood using the embedding
    let classification = {
        let classifier_guard = state.mood_classifier.read().await;
        if let Some(ref classifier) = *classifier_guard {
            let tiers = &[MoodTier::Primary, MoodTier::Refined];
            Some(classifier.classify(embedding.data(), tiers, 3, true))
        } else {
            None
        }
    };

    // Build storage metadata
    let mut storage_metadata = TrackMetadata::new(req.track_id.clone(), req.metadata.name)
        .with_artists(req.metadata.artists)
        .with_genres(req.metadata.genres);
    if let Some(album) = req.metadata.album {
        storage_metadata = storage_metadata.with_album(album);
    }
    if let Some(hash) = req.metadata.metadata_hash {
        storage_metadata.metadata_hash = Some(hash);
    }

    // Add mood classification to metadata
    if let Some(ref cls) = classification {
        if let Some(ref primary) = cls.primary_mood {
            storage_metadata.primary_mood = Some(primary.clone());
        }
        storage_metadata.moods = Some(cls.moods.iter().map(|m| m.mood.clone()).collect());
        storage_metadata.mood_scores = Some(
            cls.moods
                .iter()
                .map(|m| (m.mood.clone(), m.confidence))
                .collect(),
        );
        storage_metadata.valence = cls.valence;
        storage_metadata.arousal = cls.arousal;

        debug!(
            track_id = %req.track_id,
            primary_mood = ?cls.primary_mood,
            valence = ?cls.valence,
            arousal = ?cls.arousal,
            "Mood classification complete"
        );
    }

    // Store the embedding
    storage
        .upsert(
            TEXT_COLLECTION,
            &req.track_id,
            embedding.data(),
            storage_metadata,
        )
        .await
        .map_err(|e| {
            error!(error = %e, "Failed to store text embedding");
            AppError::Internal(e.to_string())
        })?;

    debug!(track_id = %req.track_id, "Text embedding stored");

    Ok(MsgPack(EmbedTextAndStoreResponse {
        track_id: req.track_id,
        stored: true,
        text,
    }))
}

/// Maximum concurrency for storage-only batch operations
#[cfg(feature = "storage")]
const BATCH_STORAGE_CONCURRENCY: usize = 16;

/// POST /api/v1/tracks/batch-upsert
///
/// Batch upsert multiple track embeddings.
/// Uses concurrent processing with bounded parallelism for better throughput.
#[cfg(feature = "storage")]
pub async fn batch_upsert(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<BatchUpsertRequest>,
) -> Result<MsgPack<BatchUpsertResponse>, AppError> {
    use futures::stream::{self, StreamExt};

    // Validate batch size
    if req.tracks.len() > MAX_BATCH_SIZE {
        return Err(AppError::BadRequest(format!(
            "Batch size {} exceeds maximum of {}",
            req.tracks.len(),
            MAX_BATCH_SIZE
        )));
    }

    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?
        .clone();

    let total = req.tracks.len();
    info!(count = total, "Batch upserting tracks (concurrent)");

    // Process tracks concurrently with bounded parallelism
    let results: Vec<BatchUpsertResult> = stream::iter(req.tracks)
        .map(|track| {
            let storage = storage.clone();
            async move {
                let track_id = track.track_id.clone();
                let mut text_stored = false;
                let mut audio_stored = false;
                let mut error_msg: Option<String> = None;

                let metadata = track.metadata.into_storage(track_id.clone());

                // Validate and store text embedding
                if let Some(ref embedding) = track.text_embedding {
                    if embedding.len() != EMBEDDING_DIM {
                        error_msg = Some(format!(
                            "Text embedding dimension mismatch: expected {}, got {}",
                            EMBEDDING_DIM,
                            embedding.len()
                        ));
                    } else {
                        match storage
                            .upsert(TEXT_COLLECTION, &track_id, embedding, metadata.clone())
                            .await
                        {
                            Ok(_) => {
                                text_stored = true;
                                debug!(track_id = %track_id, "Text embedding stored");
                            }
                            Err(e) => {
                                error!(error = %e, track_id = %track_id, "Failed to upsert text embedding");
                                error_msg = Some(e.to_string());
                            }
                        }
                    }
                }

                // Validate and store audio embedding (only if no error yet)
                if error_msg.is_none() {
                    if let Some(ref embedding) = track.audio_embedding {
                        if embedding.len() != EMBEDDING_DIM {
                            error_msg = Some(format!(
                                "Audio embedding dimension mismatch: expected {}, got {}",
                                EMBEDDING_DIM,
                                embedding.len()
                            ));
                        } else {
                            match storage
                                .upsert(AUDIO_COLLECTION, &track_id, embedding, metadata)
                                .await
                            {
                                Ok(_) => {
                                    audio_stored = true;
                                    debug!(track_id = %track_id, "Audio embedding stored");
                                }
                                Err(e) => {
                                    error!(error = %e, track_id = %track_id, "Failed to upsert audio embedding");
                                    error_msg = Some(e.to_string());
                                }
                            }
                        }
                    }
                }

                // Check if at least one embedding was provided
                if error_msg.is_none() && !text_stored && !audio_stored {
                    error_msg =
                        Some("At least one embedding (text or audio) must be provided".to_string());
                }

                let success = error_msg.is_none() && (text_stored || audio_stored);
                BatchUpsertResult {
                    track_id,
                    success,
                    error: error_msg,
                    text_stored,
                    audio_stored,
                }
            }
        })
        .buffer_unordered(BATCH_STORAGE_CONCURRENCY)
        .collect()
        .await;

    // Count successes and failures
    let succeeded = results.iter().filter(|r| r.success).count();
    let failed = total - succeeded;

    info!(total, succeeded, failed, "Batch upsert complete");

    Ok(MsgPack(BatchUpsertResponse {
        results,
        total,
        succeeded,
        failed,
    }))
}

/// Maximum concurrency for batch operations (prevents thread pool exhaustion)
#[cfg(all(feature = "inference", feature = "storage"))]
const BATCH_CONCURRENCY: usize = 8;

/// POST /api/v1/tracks/batch-embed-text
///
/// Batch generate text embeddings from metadata and store them.
/// Uses concurrent processing with bounded parallelism for better throughput.
#[cfg(all(feature = "inference", feature = "storage"))]
pub async fn batch_embed_text(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<BatchEmbedTextRequest>,
) -> Result<MsgPack<BatchEmbedTextResponse>, AppError> {
    use futures::stream::{self, StreamExt};

    // Validate batch size
    if req.tracks.len() > MAX_BATCH_SIZE {
        return Err(AppError::BadRequest(format!(
            "Batch size {} exceeds maximum of {}",
            req.tracks.len(),
            MAX_BATCH_SIZE
        )));
    }

    // Acquire read lock and clone model for use in blocking tasks
    let model = {
        let guard = state.model.read().await;
        guard
            .as_ref()
            .ok_or_else(|| AppError::Internal("Model not loaded".to_string()))?
            .clone()
    };

    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?
        .clone();

    let total = req.tracks.len();
    info!(count = total, "Batch embedding and storing tracks (concurrent)");

    // Process tracks concurrently with bounded parallelism
    let results: Vec<BatchEmbedTextResult> = stream::iter(req.tracks)
        .map(|track| {
            let model = model.clone();
            let storage = storage.clone();
            async move {
                let track_id = track.track_id.clone();

                // Format metadata into text for embedding
                let track_meta = TextTrackMetadata {
                    name: track.metadata.name.clone(),
                    artists: track.metadata.artists.clone(),
                    album: track.metadata.album.clone(),
                    genres: track.metadata.genres.clone(),
                    mood: None,
                };
                let text = format_track_metadata(&track_meta);

                // Generate embedding using the model
                let embedding_result = tokio::task::spawn_blocking({
                    let model = model.clone();
                    let text = text.clone();
                    move || model.text_embedding(&text)
                })
                .await;

                match embedding_result {
                    Err(e) => {
                        error!(error = %e, track_id = %track_id, "Text embedding task panicked");
                        BatchEmbedTextResult {
                            track_id,
                            success: false,
                            error: Some(format!("Task panicked: {}", e)),
                            text: None,
                        }
                    }
                    Ok(Err(e)) => {
                        error!(error = %e, track_id = %track_id, "Text embedding failed");
                        BatchEmbedTextResult {
                            track_id,
                            success: false,
                            error: Some(e.to_string()),
                            text: None,
                        }
                    }
                    Ok(Ok(embedding)) => {
                        // Build storage metadata
                        let storage_metadata =
                            TrackMetadata::new(track_id.clone(), track.metadata.name.clone())
                                .with_artists(track.metadata.artists.clone())
                                .with_genres(track.metadata.genres.clone());
                        let storage_metadata = if let Some(album) = track.metadata.album.clone() {
                            storage_metadata.with_album(album)
                        } else {
                            storage_metadata
                        };

                        // Store the embedding
                        match storage
                            .upsert(
                                TEXT_COLLECTION,
                                &track_id,
                                embedding.data(),
                                storage_metadata,
                            )
                            .await
                        {
                            Ok(_) => {
                                debug!(track_id = %track_id, "Text embedding stored");
                                BatchEmbedTextResult {
                                    track_id,
                                    success: true,
                                    error: None,
                                    text: Some(text),
                                }
                            }
                            Err(e) => {
                                error!(error = %e, track_id = %track_id, "Failed to store text embedding");
                                BatchEmbedTextResult {
                                    track_id,
                                    success: false,
                                    error: Some(e.to_string()),
                                    text: None,
                                }
                            }
                        }
                    }
                }
            }
        })
        .buffer_unordered(BATCH_CONCURRENCY)
        .collect()
        .await;

    // Count successes and failures
    let succeeded = results.iter().filter(|r| r.success).count();
    let failed = total - succeeded;

    info!(total, succeeded, failed, "Batch embed-text complete");

    Ok(MsgPack(BatchEmbedTextResponse {
        results,
        total,
        succeeded,
        failed,
    }))
}

/// Fallback handler when inference or storage feature is disabled
#[cfg(not(all(feature = "inference", feature = "storage")))]
pub async fn embed_text_and_store(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Both inference and storage features must be enabled".to_string(),
    ))
}

#[cfg(not(all(feature = "inference", feature = "storage")))]
pub async fn batch_embed_text(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Both inference and storage features must be enabled".to_string(),
    ))
}

/// Fallback handlers when storage feature is disabled
#[cfg(not(feature = "storage"))]
pub async fn upsert(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Storage feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "storage"))]
pub async fn search(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Storage feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "storage"))]
pub async fn get_track(
    State(_state): State<AppState>,
    Path(_track_id): Path<String>,
    Query(_query): Query<GetTrackQuery>,
) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Storage feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "storage"))]
pub async fn delete_track(
    State(_state): State<AppState>,
    Path(_track_id): Path<String>,
) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Storage feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "storage"))]
pub async fn batch_upsert(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Storage feature not enabled".to_string(),
    ))
}
