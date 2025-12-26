//! Taste profile API endpoints.

use axum::extract::{Path, Query, State};
use std::collections::HashMap;
use std::time::SystemTime;

use crate::error::AppError;
use crate::storage::{SearchFilter, AUDIO_COLLECTION, TEXT_COLLECTION};
use crate::taste::TasteVectorComputer;
use crate::types::{
    ComputeProfileRequest, ComputeProfileResponse, DeleteProfileRequest, DeleteProfileResponse,
    GetTasteVectorRequest, GetTasteVectorResponse, ProfileMetadata, ProfileType,
    ProfileTypeRequest, RecommendRequest, RecommendResponse, RecommendedTrack, TasteProfile,
};

use super::extractors::MsgPackExtractor;
use super::routes::MsgPack;
use super::AppState;

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

    // Search for similar tracks using the taste vector
    let results: Vec<crate::storage::SearchResult> = storage
        .search(
            AUDIO_COLLECTION,
            &profile.embedding,
            req.limit,
            filter,
        )
        .await?;

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
