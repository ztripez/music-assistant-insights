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
}

/// Expected embedding dimension for CLAP models
pub const EMBEDDING_DIM: usize = 512;

/// Collection names for different embedding types
pub const TEXT_COLLECTION: &str = "tracks_text";
pub const AUDIO_COLLECTION: &str = "tracks_audio";

/// Metadata stored with each embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMetadata {
    /// Music Assistant item_id
    pub track_id: String,
    /// Track name
    pub name: String,
    /// Artist names
    pub artists: Vec<String>,
    /// Album name
    pub album: Option<String>,
    /// Genre tags
    pub genres: Vec<String>,
    /// Unix timestamp of last update
    pub updated_at: i64,
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
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
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
    async fn delete_batch(&self, collection: &str, track_ids: &[String]) -> Result<(), StorageError>;

    /// Get an embedding by track ID
    async fn get(&self, collection: &str, track_id: &str) -> Result<Option<StoredEmbedding>, StorageError>;

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
