//! Taste profile API endpoints.

use axum::extract::{Path, Query, State};
use std::collections::HashMap;
use std::time::SystemTime;

use crate::error::AppError;
use crate::storage::{SearchFilter, AUDIO_COLLECTION, TEXT_COLLECTION};
use crate::taste::TasteVectorComputer;
use crate::types::taste::{AnalyzeInteractionsRequest, AnalyzeInteractionsResponse};
use crate::types::{
    ComputeProfileRequest, ComputeProfileResponse, DeleteProfileRequest, DeleteProfileResponse,
    GetTasteVectorRequest, GetTasteVectorResponse, ProfileMetadata, ProfileType,
    ProfileTypeRequest, RecommendRequest, RecommendResponse, RecommendedTrack, TasteProfile,
};

use super::extractors::MsgPackExtractor;
use super::routes::MsgPack;
use super::AppState;

/// Audio embedding score boost factor (audio matches are more reliable)
const AUDIO_SCORE_BOOST: f32 = 1.15;

/// Search both text and audio collections and merge results.
/// Audio results get a score boost since they're more reliable.
/// Results are deduplicated by track_id, keeping the highest score.
async fn merged_search(
    storage: &std::sync::Arc<super::BoxedStorage>,
    embedding: &[f32],
    limit: usize,
    filter: Option<SearchFilter>,
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

/// Compute taste profile from user interactions
///
/// POST /api/v1/users/:user_id/profile/compute
pub async fn compute_profile(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    MsgPackExtractor(req): MsgPackExtractor<ComputeProfileRequest>,
) -> Result<MsgPack<ComputeProfileResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::BadRequest("Storage not configured".to_string()))?;

    if req.interactions.is_empty() {
        return Err(AppError::BadRequest(
            "No interactions provided".to_string(),
        ));
    }

    // Fetch track embeddings from storage (try text collection first, fall back to audio)
    let mut track_embeddings = HashMap::new();
    for interaction in &req.interactions {
        // Try text collection first (Phase 1)
        if let Ok(Some(stored)) = storage.get(TEXT_COLLECTION, &interaction.track_id).await {
            track_embeddings.insert(interaction.track_id.clone(), stored.embedding);
        }
        // Fall back to audio collection if no text embedding found
        else if let Ok(Some(stored)) = storage.get(AUDIO_COLLECTION, &interaction.track_id).await
        {
            track_embeddings.insert(interaction.track_id.clone(), stored.embedding);
        }
    }

    if track_embeddings.is_empty() {
        return Err(AppError::BadRequest(
            "No embeddings found for provided track IDs".to_string(),
        ));
    }

    let computer = TasteVectorComputer::new();
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let mut profiles = Vec::new();

    match req.profile_type {
        ProfileTypeRequest::Global => {
            // Compute single global profile
            let taste = computer
                .compute_taste_vector(&req.interactions, &track_embeddings, req.cutoff_days)
                .map_err(|e| AppError::Internal(e.to_string()))?;

            let profile = TasteProfile {
                user_id: user_id.clone(),
                profile_type: ProfileType::Global,
                embedding: taste.embedding,
                track_count: taste.track_count,
                confidence: taste.confidence,
                updated_at: now,
            };

            storage.store_taste_profile(profile.clone()).await?;

            profiles.push(ProfileMetadata {
                profile_type: ProfileType::Global,
                track_count: taste.track_count,
                confidence: taste.confidence,
                updated_at: now,
            });
        }
        ProfileTypeRequest::Mood | ProfileTypeRequest::Context => {
            // For now, just compute global profile
            // TODO: Implement mood and context profile computation in future phases
            return Err(AppError::BadRequest(
                "Mood and context profiles not yet implemented".to_string(),
            ));
        }
    }

    Ok(MsgPack(ComputeProfileResponse { user_id, profiles }))
}

/// Get personalized recommendations based on taste profile
///
/// POST /api/v1/users/:user_id/recommend
pub async fn get_recommendations(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    MsgPackExtractor(req): MsgPackExtractor<RecommendRequest>,
) -> Result<MsgPack<RecommendResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::BadRequest("Storage not configured".to_string()))?;

    // Load taste profile
    let profile: TasteProfile = storage
        .get_taste_profile(&user_id, &req.profile_type)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!(
                "No {} taste profile found for user {}",
                req.profile_type, user_id
            ))
        })?;

    // Convert filter
    let filter = req.filter.map(|f| f.into()).map(|mut f: SearchFilter| {
        // Add exclude_ids to filter
        if !req.exclude_ids.is_empty() {
            f.exclude_ids = Some(req.exclude_ids.clone());
        }
        f
    });

    // Search both text and audio collections with merged results
    let results = merged_search(storage, &profile.embedding, req.limit, filter).await?;

    // Convert to RecommendedTrack
    let tracks = results
        .into_iter()
        .map(|r| RecommendedTrack {
            track_id: r.track_id,
            score: r.score,
            metadata: r.metadata,
        })
        .collect();

    Ok(MsgPack(RecommendResponse {
        tracks,
        profile_confidence: profile.confidence,
    }))
}

/// Get taste vector for debugging/analysis
///
/// GET /api/v1/users/:user_id/profile/vector
pub async fn get_taste_vector(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(req): Query<GetTasteVectorRequest>,
) -> Result<MsgPack<GetTasteVectorResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::BadRequest("Storage not configured".to_string()))?;

    let profile: TasteProfile = storage
        .get_taste_profile(&user_id, &req.profile_type)
        .await?
        .ok_or_else(|| {
            AppError::NotFound(format!(
                "No {} taste profile found for user {}",
                req.profile_type, user_id
            ))
        })?;

    Ok(MsgPack(GetTasteVectorResponse { profile }))
}

/// Delete a taste profile
///
/// DELETE /api/v1/users/:user_id/profile
pub async fn delete_profile(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    MsgPackExtractor(req): MsgPackExtractor<DeleteProfileRequest>,
) -> Result<MsgPack<DeleteProfileResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::BadRequest("Storage not configured".to_string()))?;

    storage
        .delete_taste_profile(&user_id, &req.profile_type)
        .await?;

    Ok(MsgPack(DeleteProfileResponse {
        deleted: true,
        message: format!("Deleted {} profile for user {}", req.profile_type, user_id),
    }))
}

/// Delete all profiles for a user
///
/// DELETE /api/v1/users/:user_id/profiles
pub async fn delete_all_profiles(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<MsgPack<DeleteProfileResponse>, AppError> {
    let storage = state
        .storage
        .as_ref()
        .ok_or_else(|| AppError::BadRequest("Storage not configured".to_string()))?;

    storage.delete_user_profiles(&user_id).await?;

    Ok(MsgPack(DeleteProfileResponse {
        deleted: true,
        message: format!("Deleted all profiles for user {}", user_id),
    }))
}

/// Analyze interaction weights for debugging/visualization
///
/// Returns detailed weight breakdown for each interaction,
/// showing how time decay, signal types, and completion bonuses
/// affect the final weights used in taste profile computation.
///
/// POST /api/v1/users/:user_id/interactions/analyze
pub async fn analyze_interactions(
    Path(_user_id): Path<String>,
    MsgPackExtractor(req): MsgPackExtractor<AnalyzeInteractionsRequest>,
) -> Result<MsgPack<AnalyzeInteractionsResponse>, AppError> {
    if req.interactions.is_empty() {
        return Err(AppError::BadRequest(
            "No interactions provided".to_string(),
        ));
    }

    let computer = TasteVectorComputer::new();
    let response = computer.analyze_interactions(&req.interactions, req.cutoff_days);

    Ok(MsgPack(response))
}
