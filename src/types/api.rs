//! API request and response types for track operations.

use serde::{Deserialize, Serialize};

use crate::storage::{SearchFilter, SearchResult, TrackMetadata};

/// Request to upsert track embedding(s)
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

fn default_limit() -> usize {
    10
}

/// Input filter for search operations
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
}

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

        filter
    }
}

/// Response from search operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results sorted by similarity
    pub results: Vec<SearchResult>,
    /// Number of results returned
    pub count: usize,
}

/// Request to delete a track's embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    /// Delete from text collection
    #[serde(default = "default_true")]
    pub text: bool,
    /// Delete from audio collection
    #[serde(default = "default_true")]
    pub audio: bool,
}

fn default_true() -> bool {
    true
}

impl Default for DeleteRequest {
    fn default() -> Self {
        Self {
            text: true,
            audio: true,
        }
    }
}

/// Response from delete operation
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

#[cfg(test)]
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
        };

        let filter: SearchFilter = input.into();

        assert!(filter.artists.is_some());
        assert!(filter.genres.is_some());
        assert!(filter.album.is_some());
        assert!(filter.exclude_ids.is_some());
    }
}
