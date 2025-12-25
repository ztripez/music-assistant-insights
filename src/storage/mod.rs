//! Vector storage module for track embeddings.
//!
//! This module provides a storage abstraction for vector embeddings,
//! with implementations for:
//! - Qdrant (hosted/docker vector database)
//! - usearch (embedded file-based storage)

#[cfg(feature = "storage")]
mod qdrant;
#[cfg(feature = "storage-file")]
mod usearch_store;

#[cfg(feature = "storage")]
pub use qdrant::QdrantStorage;
#[cfg(feature = "storage-file")]
pub use usearch_store::UsearchStorage;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Error type for storage operations
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Point not found: {0}")]
    PointNotFound(String),

    #[error("Invalid embedding dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    #[error("Storage operation failed: {0}")]
    OperationFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),
}

/// Expected embedding dimension for CLAP models
pub const EMBEDDING_DIM: usize = 512;

/// Collection names for different embedding types
pub const TEXT_COLLECTION: &str = "tracks_text";
pub const AUDIO_COLLECTION: &str = "tracks_audio";

/// Metadata stored with each embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMetadata {
    /// Music Assistant `item_id` (relative path for local files)
    pub track_id: String,
    /// Track name
    pub name: String,
    /// Artist names
    pub artists: Vec<String>,
    /// Album name
    pub album: Option<String>,
    /// Genre tags
    pub genres: Vec<String>,
    /// Absolute file path (for scanner-ingested tracks)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    /// Unix timestamp of last update
    pub updated_at: i64,

    // Mood classification fields (optional, populated by mood classification)
    /// Primary mood (top tier-1 mood)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_mood: Option<String>,
    /// All detected moods
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moods: Option<Vec<String>>,
    /// Mood confidence scores (`mood_id` -> confidence 0.0-1.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mood_scores: Option<std::collections::HashMap<String, f32>>,
    /// Valence (-1.0 negative to 1.0 positive)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valence: Option<f32>,
    /// Arousal (-1.0 calm to 1.0 energetic)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arousal: Option<f32>,
}

impl TrackMetadata {
    /// Create new track metadata
    pub fn new(track_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            track_id: track_id.into(),
            name: name.into(),
            artists: Vec::new(),
            album: None,
            genres: Vec::new(),
            file_path: None,
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            primary_mood: None,
            moods: None,
            mood_scores: None,
            valence: None,
            arousal: None,
        }
    }

    /// Builder method to add artists
    pub fn with_artists(mut self, artists: Vec<String>) -> Self {
        self.artists = artists;
        self
    }

    /// Builder method to add album
    pub fn with_album(mut self, album: impl Into<String>) -> Self {
        self.album = Some(album.into());
        self
    }

    /// Builder method to add genres
    pub fn with_genres(mut self, genres: Vec<String>) -> Self {
        self.genres = genres;
        self
    }

    /// Builder method to add file path
    pub fn with_file_path(mut self, path: impl Into<String>) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Builder method to set primary mood
    pub fn with_primary_mood(mut self, mood: impl Into<String>) -> Self {
        self.primary_mood = Some(mood.into());
        self
    }

    /// Builder method to set detected moods
    pub fn with_moods(mut self, moods: Vec<String>) -> Self {
        self.moods = Some(moods);
        self
    }

    /// Builder method to set mood scores
    pub fn with_mood_scores(mut self, scores: std::collections::HashMap<String, f32>) -> Self {
        self.mood_scores = Some(scores);
        self
    }

    /// Builder method to set valence-arousal coordinates
    pub fn with_valence_arousal(mut self, valence: f32, arousal: f32) -> Self {
        self.valence = Some(valence);
        self.arousal = Some(arousal);
        self
    }
}

/// Result of a similarity search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Track ID
    pub track_id: String,
    /// Similarity score (0.0 to 1.0 for cosine similarity)
    pub score: f32,
    /// Track metadata
    pub metadata: TrackMetadata,
}

/// A stored embedding with its metadata
#[derive(Debug, Clone)]
pub struct StoredEmbedding {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Associated metadata
    pub metadata: TrackMetadata,
}

/// Filter for search operations
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    /// Filter by artist names (any match)
    pub artists: Option<Vec<String>>,
    /// Filter by genres (any match)
    pub genres: Option<Vec<String>>,
    /// Filter by album name
    pub album: Option<String>,
    /// Exclude specific track IDs
    pub exclude_ids: Option<Vec<String>>,

    // Mood filters
    /// Include tracks with any of these moods
    pub moods: Option<Vec<String>>,
    /// Exclude tracks with any of these moods
    pub exclude_moods: Option<Vec<String>>,
    /// Minimum valence (-1.0 to 1.0)
    pub min_valence: Option<f32>,
    /// Maximum valence (-1.0 to 1.0)
    pub max_valence: Option<f32>,
    /// Minimum arousal (-1.0 to 1.0)
    pub min_arousal: Option<f32>,
    /// Maximum arousal (-1.0 to 1.0)
    pub max_arousal: Option<f32>,
}

impl SearchFilter {
    /// Create a new empty filter
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by artists
    pub fn with_artists(mut self, artists: Vec<String>) -> Self {
        self.artists = Some(artists);
        self
    }

    /// Filter by genres
    pub fn with_genres(mut self, genres: Vec<String>) -> Self {
        self.genres = Some(genres);
        self
    }

    /// Filter by album
    pub fn with_album(mut self, album: impl Into<String>) -> Self {
        self.album = Some(album.into());
        self
    }

    /// Exclude specific track IDs
    pub fn exclude(mut self, ids: Vec<String>) -> Self {
        self.exclude_ids = Some(ids);
        self
    }

    /// Filter by moods (include tracks with any of these moods)
    pub fn with_moods(mut self, moods: Vec<String>) -> Self {
        self.moods = Some(moods);
        self
    }

    /// Exclude tracks with any of these moods
    pub fn exclude_moods(mut self, moods: Vec<String>) -> Self {
        self.exclude_moods = Some(moods);
        self
    }

    /// Filter by valence range
    pub fn with_valence_range(mut self, min: Option<f32>, max: Option<f32>) -> Self {
        self.min_valence = min;
        self.max_valence = max;
        self
    }

    /// Filter by arousal range
    pub fn with_arousal_range(mut self, min: Option<f32>, max: Option<f32>) -> Self {
        self.min_arousal = min;
        self.max_arousal = max;
        self
    }
}

/// Trait for vector storage backends
#[async_trait]
pub trait VectorStorage: Send + Sync {
    /// Initialize storage and create collections if needed
    async fn initialize(&self) -> Result<(), StorageError>;

    /// Upsert a single embedding
    async fn upsert(
        &self,
        collection: &str,
        track_id: &str,
        embedding: &[f32],
        metadata: TrackMetadata,
    ) -> Result<(), StorageError>;

    /// Upsert multiple embeddings in a batch
    async fn upsert_batch(
        &self,
        collection: &str,
        items: Vec<(String, Vec<f32>, TrackMetadata)>,
    ) -> Result<(), StorageError>;

    /// Search for similar embeddings
    async fn search(
        &self,
        collection: &str,
        query: &[f32],
        limit: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, StorageError>;

    /// Delete an embedding by track ID
    async fn delete(&self, collection: &str, track_id: &str) -> Result<(), StorageError>;

    /// Delete multiple embeddings by track IDs
    async fn delete_batch(
        &self,
        collection: &str,
        track_ids: &[String],
    ) -> Result<(), StorageError>;

    /// Get an embedding by track ID
    async fn get(
        &self,
        collection: &str,
        track_id: &str,
    ) -> Result<Option<StoredEmbedding>, StorageError>;

    /// Check if a track exists in the collection
    async fn exists(&self, collection: &str, track_id: &str) -> Result<bool, StorageError>;

    /// Get the count of embeddings in a collection
    async fn count(&self, collection: &str) -> Result<u64, StorageError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_metadata_builder() {
        let metadata = TrackMetadata::new("track_123", "Test Song")
            .with_artists(vec!["Artist 1".to_string(), "Artist 2".to_string()])
            .with_album("Test Album")
            .with_genres(vec!["rock".to_string(), "indie".to_string()]);

        assert_eq!(metadata.track_id, "track_123");
        assert_eq!(metadata.name, "Test Song");
        assert_eq!(metadata.artists.len(), 2);
        assert_eq!(metadata.album, Some("Test Album".to_string()));
        assert_eq!(metadata.genres.len(), 2);
        assert!(metadata.updated_at > 0);
    }

    #[test]
    fn test_search_filter_builder() {
        let filter = SearchFilter::new()
            .with_artists(vec!["Artist 1".to_string()])
            .with_genres(vec!["rock".to_string()])
            .with_album("Album")
            .exclude(vec!["track_1".to_string()]);

        assert!(filter.artists.is_some());
        assert!(filter.genres.is_some());
        assert!(filter.album.is_some());
        assert!(filter.exclude_ids.is_some());
    }
}
