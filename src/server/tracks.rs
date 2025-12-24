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
    EmbedTextAndStoreResponse,
};

use super::extractors::MsgPackExtractor;
use super::routes::MsgPack;
use super::AppState;

#[cfg(feature = "storage")]
use crate::storage::{AUDIO_COLLECTION, EMBEDDING_DIM, TEXT_COLLECTION};

#[cfg(all(feature = "inference", feature = "storage"))]
use crate::inference::{format_track_metadata, TextTrackMetadata};

#[cfg(all(feature = "inference", feature = "storage"))]
use crate::storage::TrackMetadata;

#[cfg(feature = "storage")]
use tracing::{debug, error, info};

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
    let audio_result = storage.get(AUDIO_COLLECTION, &track_id).await.map_err(|e| {
        error!(error = %e, "Failed to get audio embedding");
        AppError::Internal(e.to_string())
    })?;

    let has_text = text_result.is_some();
    let has_audio = audio_result.is_some();

    if !has_text && !has_audio {
        return Err(AppError::NotFound(format!("Track '{}' not found", track_id)));
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
/// Requires both inference and storage features.
#[cfg(all(feature = "inference", feature = "storage"))]
pub async fn embed_text_and_store(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<EmbedTextAndStoreRequest>,
) -> Result<MsgPack<EmbedTextAndStoreResponse>, AppError> {
    // Acquire read lock and clone model for use in blocking task
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
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?;

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

    debug!(embedding_dim = embedding.data().len(), "Text embedding generated");

    // Build storage metadata
    let storage_metadata = TrackMetadata::new(req.track_id.clone(), req.metadata.name)
        .with_artists(req.metadata.artists)
        .with_genres(req.metadata.genres);
    let storage_metadata = if let Some(album) = req.metadata.album {
        storage_metadata.with_album(album)
    } else {
        storage_metadata
    };

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

/// POST /api/v1/tracks/batch-upsert
///
/// Batch upsert multiple track embeddings.
#[cfg(feature = "storage")]
pub async fn batch_upsert(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<BatchUpsertRequest>,
) -> Result<MsgPack<BatchUpsertResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?;

    let total = req.tracks.len();
    info!(count = total, "Batch upserting tracks");

    let mut results = Vec::with_capacity(total);
    let mut succeeded = 0;
    let mut failed = 0;

    for track in req.tracks {
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
            error_msg = Some("At least one embedding (text or audio) must be provided".to_string());
        }

        let success = error_msg.is_none() && (text_stored || audio_stored);
        if success {
            succeeded += 1;
        } else {
            failed += 1;
        }

        results.push(BatchUpsertResult {
            track_id,
            success,
            error: error_msg,
            text_stored,
            audio_stored,
        });
    }

    info!(total, succeeded, failed, "Batch upsert complete");

    Ok(MsgPack(BatchUpsertResponse {
        results,
        total,
        succeeded,
        failed,
    }))
}

/// POST /api/v1/tracks/batch-embed-text
///
/// Batch generate text embeddings from metadata and store them.
#[cfg(all(feature = "inference", feature = "storage"))]
pub async fn batch_embed_text(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<BatchEmbedTextRequest>,
) -> Result<MsgPack<BatchEmbedTextResponse>, AppError> {
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
        .ok_or_else(|| AppError::Internal("Storage not configured".to_string()))?;

    let total = req.tracks.len();
    info!(count = total, "Batch embedding and storing tracks");

    let mut results = Vec::with_capacity(total);
    let mut succeeded = 0;
    let mut failed = 0;

    for track in req.tracks {
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

        let result = match embedding_result {
            Err(e) => {
                error!(error = %e, track_id = %track_id, "Text embedding task panicked");
                failed += 1;
                BatchEmbedTextResult {
                    track_id,
                    success: false,
                    error: Some(format!("Task panicked: {}", e)),
                    text: None,
                }
            }
            Ok(Err(e)) => {
                error!(error = %e, track_id = %track_id, "Text embedding failed");
                failed += 1;
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
                    .upsert(TEXT_COLLECTION, &track_id, embedding.data(), storage_metadata)
                    .await
                {
                    Ok(_) => {
                        debug!(track_id = %track_id, "Text embedding stored");
                        succeeded += 1;
                        BatchEmbedTextResult {
                            track_id,
                            success: true,
                            error: None,
                            text: Some(text),
                        }
                    }
                    Err(e) => {
                        error!(error = %e, track_id = %track_id, "Failed to store text embedding");
                        failed += 1;
                        BatchEmbedTextResult {
                            track_id,
                            success: false,
                            error: Some(e.to_string()),
                            text: None,
                        }
                    }
                }
            }
        };

        results.push(result);
    }

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
