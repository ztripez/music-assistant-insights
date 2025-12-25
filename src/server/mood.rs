//! Mood classification API route handlers.

use axum::extract::State;
use std::collections::HashMap;

use crate::error::AppError;
use crate::mood::{get_all_moods, get_moods_by_tier, MoodTier};
use crate::types::{ListMoodsResponse, MoodClassifyRequest, MoodClassifyResponse, MoodInfo};

use super::extractors::MsgPackExtractor;
use super::routes::MsgPack;
use super::AppState;

#[cfg(feature = "inference")]
use crate::mood::MoodClassifier;

/// POST /api/v1/mood/classify
///
/// Classify the mood of audio from an embedding or track ID.
#[cfg(feature = "inference")]
pub async fn classify_mood(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<MoodClassifyRequest>,
) -> Result<MsgPack<MoodClassifyResponse>, AppError> {
    // Get the embedding from request or storage
    let embedding = if let Some(emb) = req.embedding {
        emb
    } else if let Some(track_id) = req.track_id {
        // Lookup from storage
        #[cfg(any(feature = "storage", feature = "storage-file"))]
        {
            use crate::storage::TEXT_COLLECTION;

            if let Some(ref storage) = state.storage {
                let stored = storage
                    .get(TEXT_COLLECTION, &track_id)
                    .await
                    .map_err(|e| AppError::Internal(format!("Storage error: {e}")))?;

                stored
                    .map(|s| s.embedding)
                    .ok_or_else(|| AppError::NotFound(format!("Track {} not found", track_id)))?
            } else {
                return Err(AppError::Internal("Storage not configured".to_string()));
            }
        }
        #[cfg(not(any(feature = "storage", feature = "storage-file")))]
        {
            return Err(AppError::BadRequest(
                "track_id requires storage feature".to_string(),
            ));
        }
    } else {
        return Err(AppError::BadRequest(
            "Either embedding or track_id must be provided".to_string(),
        ));
    };

    // Get or create mood classifier
    let model_guard = state.model.read().await;
    let model = model_guard
        .as_ref()
        .ok_or_else(|| AppError::Internal("Model not loaded".to_string()))?;

    // Create classifier (TODO: cache this in AppState)
    let classifier = MoodClassifier::new(model);

    // Classify
    let classification = classifier.classify(&embedding, &req.tiers, req.top_k, req.include_va);

    Ok(MsgPack(MoodClassifyResponse { classification }))
}

/// GET /api/v1/mood/list
///
/// List all available mood definitions.
pub async fn list_moods(
    State(_state): State<AppState>,
) -> MsgPack<ListMoodsResponse> {
    let all_moods = get_all_moods();

    let moods: Vec<MoodInfo> = all_moods
        .iter()
        .map(|m| MoodInfo {
            id: m.id.to_string(),
            name: m.name.to_string(),
            tier: m.tier,
            valence_hint: m.valence_hint,
            arousal_hint: m.arousal_hint,
        })
        .collect();

    let mut counts = HashMap::new();
    counts.insert(
        "primary".to_string(),
        get_moods_by_tier(MoodTier::Primary).len(),
    );
    counts.insert(
        "refined".to_string(),
        get_moods_by_tier(MoodTier::Refined).len(),
    );
    counts.insert(
        "contextual".to_string(),
        get_moods_by_tier(MoodTier::Contextual).len(),
    );

    MsgPack(ListMoodsResponse { moods, counts })
}

// Fallback handler when inference is disabled
#[cfg(not(feature = "inference"))]
pub async fn classify_mood(
    State(_state): State<AppState>,
    MsgPackExtractor(_req): MsgPackExtractor<MoodClassifyRequest>,
) -> Result<MsgPack<MoodClassifyResponse>, AppError> {
    Err(AppError::Internal(
        "Inference feature not enabled".to_string(),
    ))
}
