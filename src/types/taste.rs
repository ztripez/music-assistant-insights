//! API types for taste profile and recommendation operations.
//!
//! This module contains request/response types for computing user taste profiles
//! from listening history and generating personalized recommendations.

use serde::{Deserialize, Serialize};

use super::tracks::SearchFilterInput;
use crate::types::{ProfileType, TasteProfile, UserInteraction};

/// Request to compute a taste profile from user interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeProfileRequest {
    /// List of user interactions with tracks
    pub interactions: Vec<UserInteraction>,

    /// How many days of history to consider
    #[serde(default = "default_cutoff_days")]
    pub cutoff_days: u32,

    /// Type of profile to compute (global, mood, context)
    #[serde(default)]
    pub profile_type: ProfileTypeRequest,
}

fn default_cutoff_days() -> u32 {
    21
}

/// Profile type to compute in request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProfileTypeRequest {
    /// Compute single global profile
    Global,
    /// Compute mood-based sub-profiles
    Mood,
    /// Compute context-based sub-profiles
    Context,
}

impl Default for ProfileTypeRequest {
    fn default() -> Self {
        Self::Global
    }
}

/// Response from computing a taste profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeProfileResponse {
    /// User ID this profile belongs to
    pub user_id: String,
    /// Profiles that were computed
    pub profiles: Vec<ProfileMetadata>,
}

/// Metadata about a computed profile (without embedding)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileMetadata {
    /// Profile type
    pub profile_type: ProfileType,
    /// Number of tracks that contributed
    pub track_count: u32,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Timestamp when profile was updated
    pub updated_at: i64,
}

/// Request to get personalized recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendRequest {
    /// Maximum number of results
    #[serde(default = "default_recommend_limit")]
    pub limit: usize,

    /// Profile type to use for recommendations
    #[serde(default)]
    pub profile_type: ProfileType,

    /// Track IDs to exclude from results
    #[serde(default)]
    pub exclude_ids: Vec<String>,

    /// Optional metadata filter
    #[serde(default)]
    pub filter: Option<SearchFilterInput>,
}

fn default_recommend_limit() -> usize {
    25
}

/// Response from getting recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendResponse {
    /// Recommended tracks with similarity scores
    pub tracks: Vec<RecommendedTrack>,

    /// Confidence of the profile used
    pub profile_confidence: f32,
}

/// A recommended track with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedTrack {
    /// Track ID
    pub track_id: String,
    /// Similarity score to user's taste (0.0-1.0)
    pub score: f32,
    /// Track metadata
    #[cfg(feature = "storage")]
    pub metadata: crate::storage::TrackMetadata,
}

/// Request to get a taste vector (for debugging/analysis)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTasteVectorRequest {
    /// Profile type to retrieve
    #[serde(default)]
    pub profile_type: ProfileType,
}

/// Response containing a taste vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTasteVectorResponse {
    /// The full taste profile
    pub profile: TasteProfile,
}

/// Request to delete a taste profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteProfileRequest {
    /// Profile type to delete
    pub profile_type: ProfileType,
}

/// Response from deleting a profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteProfileResponse {
    /// Whether the deletion was successful
    pub deleted: bool,
    /// Message
    pub message: String,
}
