//! usearch-based embedded vector storage implementation.
//!
//! This provides a file-based vector storage using usearch for HNSW indexing.
//! Vectors and metadata are stored in local files, no external database needed.

use async_trait::async_trait;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::task::spawn_blocking;
use tracing::{debug, info, warn};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// Helper trait to recover from poisoned RwLocks
trait RecoverableLock<T> {
    fn read_or_recover(&self) -> RwLockReadGuard<'_, T>;
    fn write_or_recover(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T> RecoverableLock<T> for RwLock<T> {
    fn read_or_recover(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap_or_else(|poisoned| {
            warn!("RwLock was poisoned during read, recovering");
            poisoned.into_inner()
        })
    }

    fn write_or_recover(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap_or_else(|poisoned| {
            warn!("RwLock was poisoned during write, recovering");
            poisoned.into_inner()
        })
    }
}

use super::{
    SearchFilter, SearchResult, StorageError, StoredEmbedding, TrackMetadata, VectorStorage,
    AUDIO_COLLECTION, EMBEDDING_DIM, TEXT_COLLECTION,
};

/// File-based vector storage using usearch HNSW index
pub struct UsearchStorage {
    /// Base directory for storage files
    data_dir: PathBuf,
    /// Index for text embeddings
    text_index: RwLock<Index>,
    /// Index for audio embeddings
    audio_index: RwLock<Index>,
    /// Metadata storage (track_id -> metadata)
    text_metadata: RwLock<HashMap<String, TrackMetadata>>,
    audio_metadata: RwLock<HashMap<String, TrackMetadata>>,
    /// Mapping from track_id to internal usearch key
    text_id_map: RwLock<HashMap<String, u64>>,
    audio_id_map: RwLock<HashMap<String, u64>>,
    /// Next available key for each collection
    text_next_key: RwLock<u64>,
    audio_next_key: RwLock<u64>,
}

impl std::fmt::Debug for UsearchStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsearchStorage")
            .field("data_dir", &self.data_dir)
            .finish()
    }
}

impl UsearchStorage {
    /// Create a new usearch storage instance
    pub fn new(data_dir: PathBuf) -> Result<Self, StorageError> {
        info!(?data_dir, "Initializing usearch file storage");

        // Create data directory if it doesn't exist
        fs::create_dir_all(&data_dir).map_err(|e| {
            StorageError::ConnectionFailed(format!("Failed to create data directory: {}", e))
        })?;

        // Create index options for HNSW with cosine similarity
        let options = IndexOptions {
            dimensions: EMBEDDING_DIM,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 16,     // M parameter for HNSW
            expansion_add: 128,   // ef_construction
            expansion_search: 64, // ef for search
            multi: false,
        };

        // Create indices
        let text_index =
            Index::new(&options).map_err(|e| StorageError::OperationFailed(e.to_string()))?;
        let audio_index =
            Index::new(&options).map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        let storage = Self {
            data_dir,
            text_index: RwLock::new(text_index),
            audio_index: RwLock::new(audio_index),
            text_metadata: RwLock::new(HashMap::new()),
            audio_metadata: RwLock::new(HashMap::new()),
            text_id_map: RwLock::new(HashMap::new()),
            audio_id_map: RwLock::new(HashMap::new()),
            text_next_key: RwLock::new(0),
            audio_next_key: RwLock::new(0),
        };

        // Try to load existing data
        storage.load_from_disk()?;

        Ok(storage)
    }

    /// Get paths for index and metadata files
    fn get_paths(&self, collection: &str) -> (PathBuf, PathBuf, PathBuf) {
        let index_path = self.data_dir.join(format!("{}.usearch", collection));
        let metadata_path = self.data_dir.join(format!("{}.meta.bin", collection));
        let idmap_path = self.data_dir.join(format!("{}.idmap.bin", collection));
        (index_path, metadata_path, idmap_path)
    }

    /// Load indices and metadata from disk
    fn load_from_disk(&self) -> Result<(), StorageError> {
        // Load text collection
        let (text_index_path, text_meta_path, text_idmap_path) = self.get_paths(TEXT_COLLECTION);
        if text_index_path.exists() {
            info!("Loading text index from {:?}", text_index_path);
            let index = self.text_index.write_or_recover();
            index
                .load(text_index_path.to_string_lossy().as_ref())
                .map_err(|e| {
                    StorageError::OperationFailed(format!("Failed to load index: {}", e))
                })?;

            if text_meta_path.exists() {
                let data = fs::read(&text_meta_path).map_err(|e| {
                    StorageError::OperationFailed(format!("Failed to read metadata: {}", e))
                })?;
                let metadata: HashMap<String, TrackMetadata> = bincode::deserialize(&data)
                    .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                *self.text_metadata.write_or_recover() = metadata;
            }

            if text_idmap_path.exists() {
                let data = fs::read(&text_idmap_path).map_err(|e| {
                    StorageError::OperationFailed(format!("Failed to read id map: {}", e))
                })?;
                let (id_map, next_key): (HashMap<String, u64>, u64) =
                    bincode::deserialize(&data)
                        .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                *self.text_id_map.write_or_recover() = id_map;
                *self.text_next_key.write_or_recover() = next_key;
            }
        }

        // Load audio collection
        let (audio_index_path, audio_meta_path, audio_idmap_path) =
            self.get_paths(AUDIO_COLLECTION);
        if audio_index_path.exists() {
            info!("Loading audio index from {:?}", audio_index_path);
            let index = self.audio_index.write_or_recover();
            index
                .load(audio_index_path.to_string_lossy().as_ref())
                .map_err(|e| {
                    StorageError::OperationFailed(format!("Failed to load index: {}", e))
                })?;

            if audio_meta_path.exists() {
                let data = fs::read(&audio_meta_path).map_err(|e| {
                    StorageError::OperationFailed(format!("Failed to read metadata: {}", e))
                })?;
                let metadata: HashMap<String, TrackMetadata> = bincode::deserialize(&data)
                    .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                *self.audio_metadata.write_or_recover() = metadata;
            }

            if audio_idmap_path.exists() {
                let data = fs::read(&audio_idmap_path).map_err(|e| {
                    StorageError::OperationFailed(format!("Failed to read id map: {}", e))
                })?;
                let (id_map, next_key): (HashMap<String, u64>, u64) =
                    bincode::deserialize(&data)
                        .map_err(|e| StorageError::SerializationError(e.to_string()))?;
                *self.audio_id_map.write_or_recover() = id_map;
                *self.audio_next_key.write_or_recover() = next_key;
            }
        }

        Ok(())
    }

    /// Save indices and metadata to disk (async to avoid blocking executor)
    async fn save_to_disk(&self, collection: &str) -> Result<(), StorageError> {
        let (index_path, meta_path, idmap_path) = self.get_paths(collection);

        // Serialize data while holding locks (fast operation)
        let (index_path_str, meta_data, idmap_data) = {
            let (index, metadata, id_map, next_key) = if collection == TEXT_COLLECTION {
                (
                    self.text_index.read_or_recover(),
                    self.text_metadata.read_or_recover(),
                    self.text_id_map.read_or_recover(),
                    *self.text_next_key.read_or_recover(),
                )
            } else {
                (
                    self.audio_index.read_or_recover(),
                    self.audio_metadata.read_or_recover(),
                    self.audio_id_map.read_or_recover(),
                    *self.audio_next_key.read_or_recover(),
                )
            };

            // Save index (blocking but usearch requires it)
            let index_path_str = index_path.to_string_lossy().into_owned();
            index.save(&index_path_str).map_err(|e| {
                StorageError::OperationFailed(format!("Failed to save index: {}", e))
            })?;

            // Serialize metadata and id map
            let meta_data = bincode::serialize(&*metadata)
                .map_err(|e| StorageError::SerializationError(e.to_string()))?;
            let idmap_data = bincode::serialize(&(&*id_map, next_key))
                .map_err(|e| StorageError::SerializationError(e.to_string()))?;

            (index_path_str, meta_data, idmap_data)
        };
        // Locks released here

        // Write files in blocking thread pool to avoid blocking async executor
        let meta_path_clone = meta_path.clone();
        let idmap_path_clone = idmap_path.clone();

        spawn_blocking(move || {
            fs::write(&meta_path_clone, meta_data).map_err(|e| {
                StorageError::OperationFailed(format!("Failed to write metadata: {}", e))
            })?;
            fs::write(&idmap_path_clone, idmap_data).map_err(|e| {
                StorageError::OperationFailed(format!("Failed to write id map: {}", e))
            })?;
            Ok::<(), StorageError>(())
        })
        .await
        .map_err(|e| StorageError::OperationFailed(format!("Spawn blocking failed: {}", e)))??;

        debug!(collection, index_path = %index_path_str, "Saved to disk");
        Ok(())
    }

    /// Get or create a key for a track ID
    ///
    /// Uses a single write lock with entry API to avoid TOCTOU race condition.
    fn get_or_create_key(&self, collection: &str, track_id: &str) -> u64 {
        let (id_map, next_key) = if collection == TEXT_COLLECTION {
            (&self.text_id_map, &self.text_next_key)
        } else {
            (&self.audio_id_map, &self.audio_next_key)
        };

        // Use write lock from the start to avoid race condition
        let mut map = id_map.write_or_recover();

        // Use entry API for atomic check-and-insert
        *map.entry(track_id.to_string()).or_insert_with(|| {
            let mut next = next_key.write_or_recover();
            let key = *next;
            *next += 1;
            key
        })
    }

    /// Apply filter to search results
    fn apply_filter(
        &self,
        results: Vec<(u64, f32)>,
        collection: &str,
        filter: Option<SearchFilter>,
    ) -> Vec<SearchResult> {
        let metadata_map = if collection == TEXT_COLLECTION {
            self.text_metadata.read_or_recover()
        } else {
            self.audio_metadata.read_or_recover()
        };

        let id_map = if collection == TEXT_COLLECTION {
            self.text_id_map.read_or_recover()
        } else {
            self.audio_id_map.read_or_recover()
        };

        // Build reverse map (key -> track_id)
        let reverse_map: HashMap<u64, &String> = id_map.iter().map(|(k, v)| (*v, k)).collect();

        results
            .into_iter()
            .filter_map(|(key, distance)| {
                let track_id = reverse_map.get(&key)?;
                let metadata = metadata_map.get(*track_id)?;

                // Apply filters
                if let Some(ref f) = filter {
                    // Exclude IDs
                    if let Some(ref exclude) = f.exclude_ids {
                        if exclude.contains(track_id) {
                            return None;
                        }
                    }

                    // Filter by artists
                    if let Some(ref artists) = f.artists {
                        if !artists.iter().any(|a| metadata.artists.contains(a)) {
                            return None;
                        }
                    }

                    // Filter by genres
                    if let Some(ref genres) = f.genres {
                        if !genres.iter().any(|g| metadata.genres.contains(g)) {
                            return None;
                        }
                    }

                    // Filter by album
                    if let Some(ref album) = f.album {
                        if metadata.album.as_ref() != Some(album) {
                            return None;
                        }
                    }

                    // Filter by moods (include)
                    if let Some(ref moods) = f.moods {
                        if let Some(ref track_moods) = metadata.moods {
                            if !moods.iter().any(|m| track_moods.contains(m)) {
                                return None;
                            }
                        } else {
                            // No moods on track, filter it out
                            return None;
                        }
                    }

                    // Filter by moods (exclude)
                    if let Some(ref exclude_moods) = f.exclude_moods {
                        if let Some(ref track_moods) = metadata.moods {
                            if exclude_moods.iter().any(|m| track_moods.contains(m)) {
                                return None;
                            }
                        }
                    }

                    // Filter by valence range
                    if let Some(min_valence) = f.min_valence {
                        if metadata.valence.map_or(true, |v| v < min_valence) {
                            return None;
                        }
                    }
                    if let Some(max_valence) = f.max_valence {
                        if metadata.valence.map_or(true, |v| v > max_valence) {
                            return None;
                        }
                    }

                    // Filter by arousal range
                    if let Some(min_arousal) = f.min_arousal {
                        if metadata.arousal.map_or(true, |a| a < min_arousal) {
                            return None;
                        }
                    }
                    if let Some(max_arousal) = f.max_arousal {
                        if metadata.arousal.map_or(true, |a| a > max_arousal) {
                            return None;
                        }
                    }
                }

                // Convert distance to similarity score (cosine: 1 - distance)
                let score = 1.0 - distance;

                Some(SearchResult {
                    track_id: (*track_id).clone(),
                    score,
                    metadata: metadata.clone(),
                })
            })
            .collect()
    }

    /// Internal upsert without saving to disk (for batch operations)
    fn upsert_one(
        &self,
        collection: &str,
        track_id: &str,
        embedding: &[f32],
        metadata: TrackMetadata,
    ) -> Result<(), StorageError> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(StorageError::InvalidDimension {
                expected: EMBEDDING_DIM,
                got: embedding.len(),
            });
        }

        let key = self.get_or_create_key(collection, track_id);

        // Get the appropriate index and metadata map
        let (index, metadata_map) = if collection == TEXT_COLLECTION {
            (&self.text_index, &self.text_metadata)
        } else {
            (&self.audio_index, &self.audio_metadata)
        };

        // Remove existing if present, then add
        {
            let idx = index.write_or_recover();
            let _ = idx.remove(key); // Ignore error if not exists
            idx.add(key, embedding)
                .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
        }

        // Update metadata
        metadata_map
            .write_or_recover()
            .insert(track_id.to_string(), metadata);

        Ok(())
    }

    /// Internal delete without saving to disk (for batch operations)
    fn delete_one(&self, collection: &str, track_id: &str) -> Result<bool, StorageError> {
        let (index, metadata_map, id_map) = if collection == TEXT_COLLECTION {
            (&self.text_index, &self.text_metadata, &self.text_id_map)
        } else {
            (&self.audio_index, &self.audio_metadata, &self.audio_id_map)
        };

        // Get key for track_id
        let key = {
            let map = id_map.read_or_recover();
            map.get(track_id).copied()
        };

        if let Some(key) = key {
            // Remove from index
            let idx = index.write_or_recover();
            idx.remove(key)
                .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
            drop(idx);

            // Remove from metadata
            metadata_map.write_or_recover().remove(track_id);

            // Remove from id map
            id_map.write_or_recover().remove(track_id);

            Ok(true) // Was deleted
        } else {
            Ok(false) // Nothing to delete
        }
    }
}

#[async_trait]
impl VectorStorage for UsearchStorage {
    async fn initialize(&self) -> Result<(), StorageError> {
        // Reserve capacity for indices
        let text_index = self.text_index.write_or_recover();
        text_index
            .reserve(10000)
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
        drop(text_index);

        let audio_index = self.audio_index.write_or_recover();
        audio_index
            .reserve(10000)
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
        drop(audio_index);

        info!("usearch storage initialized");
        Ok(())
    }

    async fn upsert(
        &self,
        collection: &str,
        track_id: &str,
        embedding: &[f32],
        metadata: TrackMetadata,
    ) -> Result<(), StorageError> {
        self.upsert_one(collection, track_id, embedding, metadata)?;

        // Save to disk after single upsert
        self.save_to_disk(collection).await?;

        debug!(track_id, collection, "Upserted embedding");
        Ok(())
    }

    async fn upsert_batch(
        &self,
        collection: &str,
        items: Vec<(String, Vec<f32>, TrackMetadata)>,
    ) -> Result<(), StorageError> {
        if items.is_empty() {
            return Ok(());
        }

        // Upsert all items without saving
        for (track_id, embedding, metadata) in &items {
            self.upsert_one(collection, track_id, embedding, metadata.clone())?;
        }

        // Save once after all upserts
        self.save_to_disk(collection).await?;

        debug!(collection, count = items.len(), "Batch upserted embeddings");
        Ok(())
    }

    async fn search(
        &self,
        collection: &str,
        query: &[f32],
        limit: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        if query.len() != EMBEDDING_DIM {
            return Err(StorageError::InvalidDimension {
                expected: EMBEDDING_DIM,
                got: query.len(),
            });
        }

        let index = if collection == TEXT_COLLECTION {
            self.text_index.read_or_recover()
        } else {
            self.audio_index.read_or_recover()
        };

        // Search with extra results to account for filtering
        let search_limit = if filter.is_some() { limit * 3 } else { limit };

        let results = index
            .search(query, search_limit)
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        drop(index);

        // Convert to (key, distance) pairs
        let pairs: Vec<(u64, f32)> = results.keys.into_iter().zip(results.distances).collect();

        // Apply filters and convert to SearchResults
        let mut filtered = self.apply_filter(pairs, collection, filter);
        filtered.truncate(limit);

        Ok(filtered)
    }

    async fn delete(&self, collection: &str, track_id: &str) -> Result<(), StorageError> {
        let deleted = self.delete_one(collection, track_id)?;

        if deleted {
            // Save to disk after single delete
            self.save_to_disk(collection).await?;
        }

        debug!(track_id, collection, "Deleted embedding");
        Ok(())
    }

    async fn delete_batch(
        &self,
        collection: &str,
        track_ids: &[String],
    ) -> Result<(), StorageError> {
        if track_ids.is_empty() {
            return Ok(());
        }

        let mut any_deleted = false;

        // Delete all items without saving
        for track_id in track_ids {
            if self.delete_one(collection, track_id)? {
                any_deleted = true;
            }
        }

        // Save once after all deletes
        if any_deleted {
            self.save_to_disk(collection).await?;
        }

        debug!(
            collection,
            count = track_ids.len(),
            "Batch deleted embeddings"
        );
        Ok(())
    }

    async fn get(
        &self,
        collection: &str,
        track_id: &str,
    ) -> Result<Option<StoredEmbedding>, StorageError> {
        let metadata_map = if collection == TEXT_COLLECTION {
            self.text_metadata.read_or_recover()
        } else {
            self.audio_metadata.read_or_recover()
        };

        let metadata = match metadata_map.get(track_id) {
            Some(m) => m.clone(),
            None => return Ok(None),
        };

        drop(metadata_map);

        // Get the key
        let id_map = if collection == TEXT_COLLECTION {
            self.text_id_map.read_or_recover()
        } else {
            self.audio_id_map.read_or_recover()
        };

        let _key = match id_map.get(track_id) {
            Some(&k) => k,
            None => return Ok(None),
        };

        drop(id_map);

        // usearch doesn't have a direct "get vector by key" method
        // We return metadata only with a placeholder embedding
        let embedding = vec![0.0f32; EMBEDDING_DIM];

        Ok(Some(StoredEmbedding {
            embedding,
            metadata,
        }))
    }

    async fn exists(&self, collection: &str, track_id: &str) -> Result<bool, StorageError> {
        let metadata_map = if collection == TEXT_COLLECTION {
            self.text_metadata.read_or_recover()
        } else {
            self.audio_metadata.read_or_recover()
        };

        Ok(metadata_map.contains_key(track_id))
    }

    async fn count(&self, collection: &str) -> Result<u64, StorageError> {
        let index = if collection == TEXT_COLLECTION {
            self.text_index.read_or_recover()
        } else {
            self.audio_index.read_or_recover()
        };

        Ok(index.size() as u64)
    }
}
