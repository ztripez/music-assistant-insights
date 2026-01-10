//! Qdrant vector database implementation.

use async_trait::async_trait;
use qdrant_client::qdrant::{
    point_id::PointIdOptions, Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance,
    Filter, GetPointsBuilder, PointId, PointStruct, PointsIdsList, SearchPointsBuilder,
    UpsertPointsBuilder, Value as QdrantValue, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use super::{
    SearchFilter, SearchResult, StorageError, StoredEmbedding, TrackMetadata, VectorStorage,
    AUDIO_COLLECTION, EMBEDDING_DIM, PROFILE_COLLECTION, TEXT_COLLECTION,
};

/// Qdrant-based vector storage implementation
pub struct QdrantStorage {
    client: Arc<Qdrant>,
    /// Prefix for collection names
    collection_prefix: String,
}

impl std::fmt::Debug for QdrantStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantStorage")
            .field("collection_prefix", &self.collection_prefix)
            .finish()
    }
}

impl QdrantStorage {
    /// Create a new Qdrant storage client
    pub async fn new(
        url: &str,
        api_key: Option<String>,
        collection_prefix: Option<String>,
    ) -> Result<Self, StorageError> {
        info!(%url, has_api_key = api_key.is_some(), "Connecting to Qdrant");

        let mut builder = Qdrant::from_url(url);

        if let Some(key) = api_key {
            builder = builder.api_key(key);
        }

        let client = builder
            .build()
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))?;

        Ok(Self {
            client: Arc::new(client),
            collection_prefix: collection_prefix.unwrap_or_default(),
        })
    }

    /// Get the full collection name with prefix
    fn collection_name(&self, base: &str) -> String {
        if self.collection_prefix.is_empty() {
            base.to_string()
        } else {
            format!("{}_{}", self.collection_prefix, base)
        }
    }

    /// Create a collection if it doesn't exist
    async fn ensure_collection(&self, name: &str) -> Result<(), StorageError> {
        let full_name = self.collection_name(name);

        let exists = self
            .client
            .collection_exists(&full_name)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        if !exists {
            info!(collection = %full_name, dim = EMBEDDING_DIM, "Creating collection");

            self.client
                .create_collection(CreateCollectionBuilder::new(&full_name).vectors_config(
                    VectorParamsBuilder::new(EMBEDDING_DIM as u64, Distance::Cosine),
                ))
                .await
                .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

            info!(collection = %full_name, "Collection created");
        } else {
            debug!(collection = %full_name, "Collection already exists");
        }

        Ok(())
    }

    /// Convert TrackMetadata to Qdrant Payload
    fn metadata_to_payload(metadata: &TrackMetadata) -> Payload {
        let mut payload = json!({
            "track_id": metadata.track_id,
            "name": metadata.name,
            "artists": metadata.artists,
            "album": metadata.album,
            "genres": metadata.genres,
            "updated_at": metadata.updated_at,
        });

        // Add file_path if present
        if let Some(ref file_path) = metadata.file_path {
            payload["file_path"] = json!(file_path);
        }

        // Add mood fields if present
        if let Some(ref primary_mood) = metadata.primary_mood {
            payload["primary_mood"] = json!(primary_mood);
        }
        if let Some(ref moods) = metadata.moods {
            payload["moods"] = json!(moods);
        }
        if let Some(ref mood_scores) = metadata.mood_scores {
            payload["mood_scores"] = json!(mood_scores);
        }
        if let Some(valence) = metadata.valence {
            payload["valence"] = json!(valence);
        }
        if let Some(arousal) = metadata.arousal {
            payload["arousal"] = json!(arousal);
        }

        Payload::try_from(payload).unwrap_or_default()
    }

    /// Convert Qdrant payload to TrackMetadata
    fn payload_to_metadata(
        payload: &HashMap<String, QdrantValue>,
    ) -> Result<TrackMetadata, StorageError> {
        let track_id = payload
            .get("track_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| StorageError::SerializationError("Missing track_id".to_string()))?;

        let name = payload
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default();

        let artists = payload
            .get("artists")
            .and_then(|v| v.as_list())
            .map(|list| {
                list.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        let album = payload
            .get("album")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let genres = payload
            .get("genres")
            .and_then(|v| v.as_list())
            .map(|list| {
                list.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        let updated_at = payload
            .get("updated_at")
            .and_then(|v| v.as_integer())
            .unwrap_or(0);

        let file_path = payload
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Parse mood fields
        let primary_mood = payload
            .get("primary_mood")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let moods = payload.get("moods").and_then(|v| v.as_list()).map(|list| {
            list.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        });

        let mood_scores = payload.get("mood_scores").and_then(|v| {
            v.as_struct().map(|s| {
                s.fields
                    .iter()
                    .filter_map(|(k, v)| v.as_double().map(|d| (k.clone(), d as f32)))
                    .collect()
            })
        });

        let valence = payload
            .get("valence")
            .and_then(|v| v.as_double())
            .map(|d| d as f32);

        let arousal = payload
            .get("arousal")
            .and_then(|v| v.as_double())
            .map(|d| d as f32);

        let metadata_hash = payload
            .get("metadata_hash")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(TrackMetadata {
            track_id,
            name,
            artists,
            album,
            genres,
            file_path,
            updated_at,
            primary_mood,
            moods,
            mood_scores,
            valence,
            arousal,
            metadata_hash,
        })
    }

    /// Build a Qdrant filter from SearchFilter
    fn build_filter(filter: &SearchFilter) -> Option<Filter> {
        use qdrant_client::qdrant::Range;

        let mut conditions: Vec<Condition> = Vec::new();

        if let Some(ref artists) = filter.artists {
            for artist in artists {
                conditions.push(Condition::matches("artists", artist.clone()));
            }
        }

        if let Some(ref genres) = filter.genres {
            for genre in genres {
                conditions.push(Condition::matches("genres", genre.clone()));
            }
        }

        if let Some(ref album) = filter.album {
            conditions.push(Condition::matches("album", album.clone()));
        }

        // Mood inclusion filter - match any of the specified moods
        if let Some(ref moods) = filter.moods {
            for mood in moods {
                conditions.push(Condition::matches("moods", mood.clone()));
            }
        }

        // Valence range filter
        if filter.min_valence.is_some() || filter.max_valence.is_some() {
            conditions.push(Condition::range(
                "valence",
                Range {
                    gte: filter.min_valence.map(|v| v as f64),
                    lte: filter.max_valence.map(|v| v as f64),
                    ..Default::default()
                },
            ));
        }

        // Arousal range filter
        if filter.min_arousal.is_some() || filter.max_arousal.is_some() {
            conditions.push(Condition::range(
                "arousal",
                Range {
                    gte: filter.min_arousal.map(|v| v as f64),
                    lte: filter.max_arousal.map(|v| v as f64),
                    ..Default::default()
                },
            ));
        }

        // Handle exclusions
        let mut must_not: Vec<Condition> = Vec::new();
        if let Some(ref exclude_ids) = filter.exclude_ids {
            for id in exclude_ids {
                must_not.push(Condition::matches("track_id", id.clone()));
            }
        }

        // Exclude tracks with any of these moods
        if let Some(ref exclude_moods) = filter.exclude_moods {
            for mood in exclude_moods {
                must_not.push(Condition::matches("moods", mood.clone()));
            }
        }

        if conditions.is_empty() && must_not.is_empty() {
            None
        } else if must_not.is_empty() {
            // Use must (AND) - all conditions must match
            Some(Filter::must(conditions))
        } else if conditions.is_empty() {
            Some(Filter::must_not(must_not))
        } else {
            // Both positive conditions (AND) and exclusions
            Some(Filter {
                must: conditions,
                must_not,
                ..Default::default()
            })
        }
    }

    /// Generate a deterministic point ID from track_id
    fn track_id_to_point_id(track_id: &str) -> PointId {
        // Use the track_id string directly as the point ID
        PointId {
            point_id_options: Some(PointIdOptions::Uuid(
                uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, track_id.as_bytes()).to_string(),
            )),
        }
    }
}

#[async_trait]
impl VectorStorage for QdrantStorage {
    async fn initialize(&self) -> Result<(), StorageError> {
        info!("Initializing Qdrant storage");

        // Create both collections
        self.ensure_collection(TEXT_COLLECTION).await?;
        self.ensure_collection(AUDIO_COLLECTION).await?;

        info!("Qdrant storage initialized");
        Ok(())
    }

    async fn upsert(
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

        let full_name = self.collection_name(collection);
        let point_id = Self::track_id_to_point_id(track_id);
        let payload = Self::metadata_to_payload(&metadata);

        debug!(collection = %full_name, %track_id, "Upserting point");

        let point = PointStruct::new(point_id, embedding.to_vec(), payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(&full_name, vec![point]).wait(true))
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

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

        // Validate dimensions
        for (_track_id, embedding, _) in &items {
            if embedding.len() != EMBEDDING_DIM {
                return Err(StorageError::InvalidDimension {
                    expected: EMBEDDING_DIM,
                    got: embedding.len(),
                });
            }
        }

        let full_name = self.collection_name(collection);
        debug!(collection = %full_name, count = items.len(), "Batch upserting points");

        let points: Vec<PointStruct> = items
            .into_iter()
            .map(|(track_id, embedding, metadata)| {
                let point_id = Self::track_id_to_point_id(&track_id);
                let payload = Self::metadata_to_payload(&metadata);
                PointStruct::new(point_id, embedding, payload)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(&full_name, points).wait(true))
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

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

        let full_name = self.collection_name(collection);
        debug!(collection = %full_name, %limit, "Searching points");

        let mut search_builder =
            SearchPointsBuilder::new(&full_name, query.to_vec(), limit as u64).with_payload(true);

        if let Some(ref f) = filter {
            if let Some(qdrant_filter) = Self::build_filter(f) {
                search_builder = search_builder.filter(qdrant_filter);
            }
        }

        let response = self
            .client
            .search_points(search_builder)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        let results = response
            .result
            .into_iter()
            .filter_map(|scored_point| {
                let payload = scored_point.payload;
                let metadata = Self::payload_to_metadata(&payload).ok()?;
                Some(SearchResult {
                    track_id: metadata.track_id.clone(),
                    score: scored_point.score,
                    metadata,
                })
            })
            .collect();

        Ok(results)
    }

    async fn delete(&self, collection: &str, track_id: &str) -> Result<(), StorageError> {
        let full_name = self.collection_name(collection);
        let point_id = Self::track_id_to_point_id(track_id);

        debug!(collection = %full_name, %track_id, "Deleting point");

        self.client
            .delete_points(
                DeletePointsBuilder::new(&full_name)
                    .points(PointsIdsList {
                        ids: vec![point_id],
                    })
                    .wait(true),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

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

        let full_name = self.collection_name(collection);
        debug!(collection = %full_name, count = track_ids.len(), "Batch deleting points");

        let point_ids: Vec<PointId> = track_ids
            .iter()
            .map(|id| Self::track_id_to_point_id(id))
            .collect();

        self.client
            .delete_points(
                DeletePointsBuilder::new(&full_name)
                    .points(PointsIdsList { ids: point_ids })
                    .wait(true),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        Ok(())
    }

    async fn get(
        &self,
        collection: &str,
        track_id: &str,
    ) -> Result<Option<StoredEmbedding>, StorageError> {
        let full_name = self.collection_name(collection);
        let point_id = Self::track_id_to_point_id(track_id);

        debug!(collection = %full_name, %track_id, "Getting point");

        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(&full_name, vec![point_id])
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        if let Some(point) = response.result.into_iter().next() {
            let metadata = Self::payload_to_metadata(&point.payload)?;

            // Extract vector using the vectors_output module
            #[allow(deprecated)]
            let embedding = point
                .vectors
                .and_then(|v| v.vectors_options)
                .and_then(|opts| {
                    use qdrant_client::qdrant::vectors_output::VectorsOptions;
                    match opts {
                        VectorsOptions::Vector(v) => Some(v.data.clone()),
                        _ => None,
                    }
                })
                .unwrap_or_default();

            Ok(Some(StoredEmbedding {
                embedding,
                metadata,
            }))
        } else {
            Ok(None)
        }
    }

    async fn exists(&self, collection: &str, track_id: &str) -> Result<bool, StorageError> {
        let full_name = self.collection_name(collection);
        let point_id = Self::track_id_to_point_id(track_id);

        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(&full_name, vec![point_id])
                    .with_payload(false)
                    .with_vectors(false),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        Ok(!response.result.is_empty())
    }

    async fn count(&self, collection: &str) -> Result<u64, StorageError> {
        let full_name = self.collection_name(collection);

        let info = self
            .client
            .collection_info(&full_name)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        Ok(info
            .result
            .map(|r| r.points_count.unwrap_or(0))
            .unwrap_or(0))
    }

    async fn store_taste_profile(
        &self,
        profile: crate::types::TasteProfile,
    ) -> Result<(), StorageError> {
        self.ensure_collection(PROFILE_COLLECTION).await?;

        // Create a unique identifier for this profile
        let profile_key = format!("{}::{}", profile.user_id, profile.profile_type);

        // Convert to UUID-based PointId (same pattern as tracks)
        let point_id = PointId {
            point_id_options: Some(PointIdOptions::Uuid(
                uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, profile_key.as_bytes()).to_string(),
            )),
        };

        let mut payload = Payload::new();
        payload.insert("user_id", profile.user_id.clone());
        payload.insert(
            "profile_type",
            serde_json::to_value(&profile.profile_type)
                .map_err(|e| StorageError::SerializationError(e.to_string()))?,
        );
        payload.insert("track_count", profile.track_count as i64);
        payload.insert("confidence", profile.confidence as f64);
        payload.insert("updated_at", profile.updated_at);

        let point = PointStruct::new(point_id, profile.embedding, payload);

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(
                    self.collection_name(PROFILE_COLLECTION),
                    vec![point],
                )
                .wait(true),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        Ok(())
    }

    async fn get_taste_profile(
        &self,
        user_id: &str,
        profile_type: &crate::types::ProfileType,
    ) -> Result<Option<crate::types::TasteProfile>, StorageError> {
        let full_name = self.collection_name(PROFILE_COLLECTION);

        let exists = self
            .client
            .collection_exists(&full_name)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        if !exists {
            return Ok(None);
        }

        // Create the same UUID-based point ID used during storage
        let profile_key = format!("{}::{}", user_id, profile_type);
        let point_id = PointId {
            point_id_options: Some(PointIdOptions::Uuid(
                uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, profile_key.as_bytes()).to_string(),
            )),
        };

        let response = self
            .client
            .get_points(
                GetPointsBuilder::new(&full_name, vec![point_id])
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        if response.result.is_empty() {
            return Ok(None);
        }

        let point = &response.result[0];
        let embedding = match &point.vectors {
            Some(vectors) => {
                let vec = vectors
                    .vectors_options
                    .as_ref()
                    .and_then(|v| match v {
                        qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(vec) => {
                            Some(vec.clone().into_vector())
                        }
                        _ => None,
                    })
                    .ok_or_else(|| {
                        StorageError::OperationFailed("No vector data found".to_string())
                    })?;

                match vec {
                    qdrant_client::qdrant::vector_output::Vector::Dense(dense) => dense.data,
                    _ => {
                        return Err(StorageError::OperationFailed(
                            "Expected dense vector".to_string(),
                        ))
                    }
                }
            }
            None => {
                return Err(StorageError::OperationFailed(
                    "No vectors in point".to_string(),
                ))
            }
        };

        let payload = &point.payload;
        let profile = crate::types::TasteProfile {
            user_id: payload
                .get("user_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| user_id.to_string()),
            profile_type: serde_json::from_value(
                payload
                    .get("profile_type")
                    .ok_or_else(|| {
                        StorageError::SerializationError("Missing profile_type".to_string())
                    })?
                    .clone()
                    .into(),
            )
            .map_err(|e| StorageError::SerializationError(e.to_string()))?,
            embedding,
            track_count: payload
                .get("track_count")
                .and_then(|v| v.as_integer())
                .unwrap_or(0) as u32,
            confidence: payload
                .get("confidence")
                .and_then(|v| v.as_double())
                .unwrap_or(0.0) as f32,
            updated_at: payload
                .get("updated_at")
                .and_then(|v| v.as_integer())
                .unwrap_or(0),
        };

        Ok(Some(profile))
    }

    async fn list_user_profiles(
        &self,
        user_id: &str,
    ) -> Result<Vec<crate::types::TasteProfile>, StorageError> {
        let full_name = self.collection_name(PROFILE_COLLECTION);

        let exists = self
            .client
            .collection_exists(&full_name)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        if !exists {
            return Ok(Vec::new());
        }

        // Use scroll to get all profiles for a user
        let filter = Filter::must(vec![Condition::matches(
            "user_id",
            user_id.to_string(),
        )]);

        let response = self
            .client
            .scroll(
                qdrant_client::qdrant::ScrollPointsBuilder::new(&full_name)
                    .filter(filter)
                    .with_payload(true)
                    .with_vectors(true)
                    .limit(100),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        let mut profiles = Vec::new();

        for point in response.result {
            let embedding = match &point.vectors {
                Some(vectors) => {
                    let vec = vectors
                        .vectors_options
                        .as_ref()
                        .and_then(|v| match v {
                            qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(vec) => {
                                Some(vec.clone().into_vector())
                            }
                            _ => None,
                        })
                        .ok_or_else(|| {
                            StorageError::OperationFailed("No vector data found".to_string())
                        })?;

                    match vec {
                        qdrant_client::qdrant::vector_output::Vector::Dense(dense) => dense.data,
                        _ => continue, // Skip non-dense vectors
                    }
                }
                None => continue,
            };

            let payload = &point.payload;
            let profile = crate::types::TasteProfile {
                user_id: payload
                    .get("user_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| user_id.to_string()),
                profile_type: serde_json::from_value(
                    payload
                        .get("profile_type")
                        .ok_or_else(|| {
                            StorageError::SerializationError("Missing profile_type".to_string())
                        })?
                        .clone()
                        .into(),
                )
                .map_err(|e| StorageError::SerializationError(e.to_string()))?,
                embedding,
                track_count: payload
                    .get("track_count")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(0) as u32,
                confidence: payload
                    .get("confidence")
                    .and_then(|v| v.as_double())
                    .unwrap_or(0.0) as f32,
                updated_at: payload
                    .get("updated_at")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(0),
            };

            profiles.push(profile);
        }

        Ok(profiles)
    }

    async fn delete_taste_profile(
        &self,
        user_id: &str,
        profile_type: &crate::types::ProfileType,
    ) -> Result<(), StorageError> {
        let full_name = self.collection_name(PROFILE_COLLECTION);

        let exists = self
            .client
            .collection_exists(&full_name)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        if !exists {
            return Ok(());
        }

        // Create the same UUID-based point ID used during storage
        let profile_key = format!("{}::{}", user_id, profile_type);
        let point_id = PointId {
            point_id_options: Some(PointIdOptions::Uuid(
                uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, profile_key.as_bytes()).to_string(),
            )),
        };

        self.client
            .delete_points(
                DeletePointsBuilder::new(&full_name)
                    .points(PointsIdsList {
                        ids: vec![point_id],
                    })
                    .wait(true),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        Ok(())
    }

    async fn delete_user_profiles(&self, user_id: &str) -> Result<(), StorageError> {
        let full_name = self.collection_name(PROFILE_COLLECTION);

        let exists = self
            .client
            .collection_exists(&full_name)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        if !exists {
            return Ok(());
        }

        let filter = Filter::must(vec![Condition::matches(
            "user_id",
            user_id.to_string(),
        )]);

        self.client
            .delete_points(
                DeletePointsBuilder::new(&full_name)
                    .points(filter)
                    .wait(true),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        Ok(())
    }
}
