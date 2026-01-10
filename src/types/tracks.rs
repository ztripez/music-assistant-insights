//! API types for track storage operations.
//!
//! This module contains request/response types for upserting, searching,
//! and deleting track embeddings.

use serde::{Deserialize, Serialize};

#[cfg(feature = "storage")]
use crate::storage::{SearchFilter, SearchResult, TrackMetadata};

/// Request to upsert track embedding(s)
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRequest {
    /// Track ID (Music Assistant item_id)
    pub track_id: String,
    /// Track metadata
    pub metadata: TrackMetadataInput,
    /// Text embedding (512-dimensional)
    #[serde(default)]
    pub text_embedding: Option<Vec<f32>>,
    /// Audio embedding (512-dimensional)
    #[serde(default)]
    pub audio_embedding: Option<Vec<f32>>,
}

/// Input metadata for upsert operations
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMetadataInput {
    /// Track name
    pub name: String,
    /// Artist names
    #[serde(default)]
    pub artists: Vec<String>,
    /// Album name
    #[serde(default)]
    pub album: Option<String>,
    /// Genre tags
    #[serde(default)]
    pub genres: Vec<String>,
}

#[cfg(feature = "storage")]
impl TrackMetadataInput {
    /// Convert to storage TrackMetadata
    pub fn into_storage(self, track_id: String) -> TrackMetadata {
        let mut metadata = TrackMetadata::new(track_id, self.name)
            .with_artists(self.artists)
            .with_genres(self.genres);

        if let Some(album) = self.album {
            metadata = metadata.with_album(album);
        }

        metadata
    }
}

/// Response from upsert operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    /// Track ID that was upserted
    pub track_id: String,
    /// Whether text embedding was stored
    pub text_stored: bool,
    /// Whether audio embedding was stored
    pub audio_stored: bool,
}

/// Request to search for similar tracks
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Query embedding (512-dimensional)
    pub embedding: Vec<f32>,
    /// Collection to search: "text" or "audio"
    pub collection: String,
    /// Maximum number of results (default: 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Optional filter
    #[serde(default)]
    pub filter: Option<SearchFilterInput>,
}

#[cfg(feature = "storage")]
fn default_limit() -> usize {
    10
}

/// Input filter for search operations
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilterInput {
    /// Filter by artist names (any match)
    #[serde(default)]
    pub artists: Option<Vec<String>>,
    /// Filter by genres (any match)
    #[serde(default)]
    pub genres: Option<Vec<String>>,
    /// Filter by album name
    #[serde(default)]
    pub album: Option<String>,
    /// Exclude specific track IDs
    #[serde(default)]
    pub exclude_ids: Option<Vec<String>>,
    /// Include tracks with any of these moods
    #[serde(default)]
    pub moods: Option<Vec<String>>,
    /// Exclude tracks with any of these moods
    #[serde(default)]
    pub exclude_moods: Option<Vec<String>>,
    /// Minimum valence (-1.0 to 1.0)
    #[serde(default)]
    pub min_valence: Option<f32>,
    /// Maximum valence (-1.0 to 1.0)
    #[serde(default)]
    pub max_valence: Option<f32>,
    /// Minimum arousal (-1.0 to 1.0)
    #[serde(default)]
    pub min_arousal: Option<f32>,
    /// Maximum arousal (-1.0 to 1.0)
    #[serde(default)]
    pub max_arousal: Option<f32>,
}

#[cfg(feature = "storage")]
impl From<SearchFilterInput> for SearchFilter {
    fn from(input: SearchFilterInput) -> Self {
        let mut filter = SearchFilter::new();

        if let Some(artists) = input.artists {
            filter = filter.with_artists(artists);
        }
        if let Some(genres) = input.genres {
            filter = filter.with_genres(genres);
        }
        if let Some(album) = input.album {
            filter = filter.with_album(album);
        }
        if let Some(exclude_ids) = input.exclude_ids {
            filter = filter.exclude(exclude_ids);
        }
        if let Some(moods) = input.moods {
            filter = filter.with_moods(moods);
        }
        if let Some(exclude_moods) = input.exclude_moods {
            filter = filter.exclude_moods(exclude_moods);
        }
        if input.min_valence.is_some() || input.max_valence.is_some() {
            filter = filter.with_valence_range(input.min_valence, input.max_valence);
        }
        if input.min_arousal.is_some() || input.max_arousal.is_some() {
            filter = filter.with_arousal_range(input.min_arousal, input.max_arousal);
        }

        filter
    }
}

/// Request to search for similar tracks using text query
#[cfg(all(feature = "inference", feature = "storage"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchRequest {
    /// Text query to search for (will be embedded)
    pub query: String,
    /// Maximum number of results (default: 10)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Optional filter
    #[serde(default)]
    pub filter: Option<SearchFilterInput>,
}

/// Response from search operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results sorted by similarity
    pub results: Vec<SearchResult>,
    /// Number of results returned
    pub count: usize,
}

/// Request to delete a track's embeddings
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    /// Delete from text collection
    #[serde(default = "default_true")]
    pub text: bool,
    /// Delete from audio collection
    #[serde(default = "default_true")]
    pub audio: bool,
}

#[cfg(feature = "storage")]
fn default_true() -> bool {
    true
}

#[cfg(feature = "storage")]
impl Default for DeleteRequest {
    fn default() -> Self {
        Self {
            text: true,
            audio: true,
        }
    }
}

/// Response from delete operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    /// Track ID that was deleted
    pub track_id: String,
    /// Whether text embedding was deleted
    pub text_deleted: bool,
    /// Whether audio embedding was deleted
    pub audio_deleted: bool,
}

/// Response for get track operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTrackResponse {
    /// Track ID
    pub track_id: String,
    /// Track metadata (from either collection)
    pub metadata: Option<TrackMetadata>,
    /// Whether text embedding exists
    pub has_text: bool,
    /// Whether audio embedding exists
    pub has_audio: bool,
    /// Text embedding if requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_embedding: Option<Vec<f32>>,
    /// Audio embedding if requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_embedding: Option<Vec<f32>>,
}

// ============================================================================
// Batch operation types
// ============================================================================

/// Request to batch upsert multiple track embeddings
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchUpsertRequest {
    /// List of tracks to upsert
    pub tracks: Vec<UpsertRequest>,
}

/// Response from batch upsert operation
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchUpsertResponse {
    /// Results for each track
    pub results: Vec<BatchUpsertResult>,
    /// Total number of tracks processed
    pub total: usize,
    /// Number of successful upserts
    pub succeeded: usize,
    /// Number of failed upserts
    pub failed: usize,
}

/// Result for a single track in batch upsert
#[cfg(feature = "storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchUpsertResult {
    /// Track ID
    pub track_id: String,
    /// Whether the operation succeeded
    pub success: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Whether text embedding was stored
    pub text_stored: bool,
    /// Whether audio embedding was stored
    pub audio_stored: bool,
}

#[cfg(all(test, feature = "storage"))]
mod tests {
    use super::*;

    #[test]
    fn test_upsert_request_serialize() {
        let req = UpsertRequest {
            track_id: "track_123".to_string(),
            metadata: TrackMetadataInput {
                name: "Test Song".to_string(),
                artists: vec!["Artist 1".to_string()],
                album: Some("Album".to_string()),
                genres: vec!["rock".to_string()],
            },
            text_embedding: Some(vec![0.1; 512]),
            audio_embedding: None,
        };

        let bytes = rmp_serde::to_vec(&req).unwrap();
        let decoded: UpsertRequest = rmp_serde::from_slice(&bytes).unwrap();

        assert_eq!(decoded.track_id, "track_123");
        assert!(decoded.text_embedding.is_some());
        assert!(decoded.audio_embedding.is_none());
    }

    #[test]
    fn test_search_filter_conversion() {
        let input = SearchFilterInput {
            artists: Some(vec!["Artist".to_string()]),
            genres: Some(vec!["rock".to_string()]),
            album: Some("Album".to_string()),
            exclude_ids: Some(vec!["track_1".to_string()]),
            moods: Some(vec!["energetic".to_string()]),
            exclude_moods: None,
            min_valence: Some(0.0),
            max_valence: None,
            min_arousal: None,
            max_arousal: Some(0.5),
        };

        let filter: SearchFilter = input.into();

        assert!(filter.artists.is_some());
        assert!(filter.genres.is_some());
        assert!(filter.moods.is_some());
        assert!(filter.min_valence.is_some());
        assert!(filter.max_arousal.is_some());
        assert!(filter.album.is_some());
        assert!(filter.exclude_ids.is_some());
    }
}
