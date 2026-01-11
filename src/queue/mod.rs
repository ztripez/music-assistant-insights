//! Crash-resistant audio session queue using redb.
//!
//! This module provides persistent storage for audio sessions that are pending
//! processing. Sessions survive sidecar restarts and crashes.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use redb::{Database, DatabaseError, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Table for pending audio sessions
const SESSIONS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("sessions");

/// Status of a queued session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Session is queued and waiting to be processed
    Pending,
    /// Session is currently being processed
    Processing,
}

/// A queued audio session record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    /// Unique session ID (MA queue_item_id)
    pub queue_item_id: String,
    /// Track ID from Music Assistant
    pub track_id: String,
    /// Player/queue ID from Music Assistant
    pub queue_id: String,
    /// Sample rate of the PCM audio
    pub sample_rate: u32,
    /// Number of channels (1 or 2)
    pub channels: u8,
    /// Path to the PCM file
    pub pcm_path: PathBuf,
    /// Current status
    pub status: SessionStatus,
    /// Unix timestamp when session was created
    pub created_at: i64,
    /// Track metadata for embedding
    pub metadata: SessionMetadata,
}

/// Track metadata stored with the session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
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

/// Errors that can occur during queue operations
#[derive(Debug, Error)]
pub enum QueueError {
    #[error("Database error: {0}")]
    Database(#[from] redb::Error),

    #[error("Database creation error: {0}")]
    DatabaseCreation(#[from] DatabaseError),

    #[error("Transaction error: {0}")]
    Transaction(#[from] redb::TransactionError),

    #[error("Table error: {0}")]
    Table(#[from] redb::TableError),

    #[error("Storage error: {0}")]
    Storage(#[from] redb::StorageError),

    #[error("Commit error: {0}")]
    Commit(#[from] redb::CommitError),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Session not found: {0}")]
    NotFound(String),
}

/// Persistent queue for audio sessions
pub struct AudioQueue {
    db: Database,
    /// Directory where PCM files are stored
    audio_dir: PathBuf,
}

impl AudioQueue {
    /// Create or open the queue database
    pub fn new(db_path: impl AsRef<Path>, audio_dir: impl AsRef<Path>) -> Result<Self, QueueError> {
        let db_path = db_path.as_ref();
        let audio_dir = audio_dir.as_ref();

        // Ensure directories exist
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                QueueError::Serialization(format!("Failed to create db directory: {}", e))
            })?;
        }
        std::fs::create_dir_all(audio_dir).map_err(|e| {
            QueueError::Serialization(format!("Failed to create audio directory: {}", e))
        })?;

        let db = Database::create(db_path)?;

        // Ensure table exists
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(SESSIONS_TABLE)?;
        }
        write_txn.commit()?;

        info!(
            db_path = %db_path.display(),
            audio_dir = %audio_dir.display(),
            "Audio queue initialized"
        );

        Ok(Self {
            db,
            audio_dir: audio_dir.to_path_buf(),
        })
    }

    /// Get the audio directory path
    pub fn audio_dir(&self) -> &Path {
        &self.audio_dir
    }

    /// Push a new session record to the queue
    pub fn push(&self, record: SessionRecord) -> Result<(), QueueError> {
        let data = rmp_serde::to_vec(&record)
            .map_err(|e| QueueError::Serialization(e.to_string()))?;

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(SESSIONS_TABLE)?;
            table.insert(record.queue_item_id.as_str(), data.as_slice())?;
        }
        write_txn.commit()?;

        debug!(
            queue_item_id = %record.queue_item_id,
            track_id = %record.track_id,
            "Session queued"
        );

        Ok(())
    }

    /// Get a session by ID without removing it
    pub fn get(&self, queue_item_id: &str) -> Result<Option<SessionRecord>, QueueError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SESSIONS_TABLE)?;

        match table.get(queue_item_id)? {
            Some(data) => {
                let record: SessionRecord = rmp_serde::from_slice(data.value())
                    .map_err(|e| QueueError::Serialization(e.to_string()))?;
                Ok(Some(record))
            }
            None => Ok(None),
        }
    }

    /// Pop a pending session and mark it as processing
    ///
    /// Returns the first pending session found, or None if queue is empty.
    pub fn pop_pending(&self) -> Result<Option<SessionRecord>, QueueError> {
        let write_txn = self.db.begin_write()?;
        let mut result = None;

        {
            let mut table = write_txn.open_table(SESSIONS_TABLE)?;

            // Find first pending session and deserialize it
            let mut pending_record: Option<(String, SessionRecord)> = None;
            for entry in table.iter()? {
                let (key, value) = entry?;
                let record: SessionRecord = rmp_serde::from_slice(value.value())
                    .map_err(|e| QueueError::Serialization(e.to_string()))?;

                if record.status == SessionStatus::Pending {
                    pending_record = Some((key.value().to_string(), record));
                    break;
                }
            }

            // Update status to Processing (no re-read needed, we have the record)
            if let Some((key, mut record)) = pending_record {
                record.status = SessionStatus::Processing;

                let updated_data = rmp_serde::to_vec(&record)
                    .map_err(|e| QueueError::Serialization(e.to_string()))?;

                table.insert(key.as_str(), updated_data.as_slice())?;
                result = Some(record);
            }
        }

        write_txn.commit()?;
        Ok(result)
    }

    /// Remove a session from the queue
    pub fn remove(&self, queue_item_id: &str) -> Result<Option<SessionRecord>, QueueError> {
        let write_txn = self.db.begin_write()?;
        let result;

        {
            let mut table = write_txn.open_table(SESSIONS_TABLE)?;

            result = match table.remove(queue_item_id)? {
                Some(data) => {
                    let record: SessionRecord = rmp_serde::from_slice(data.value())
                        .map_err(|e| QueueError::Serialization(e.to_string()))?;
                    Some(record)
                }
                None => None,
            };
        }

        write_txn.commit()?;

        if let Some(ref record) = result {
            debug!(
                queue_item_id = %queue_item_id,
                track_id = %record.track_id,
                "Session removed from queue"
            );
        }

        Ok(result)
    }

    /// List all pending sessions (for startup recovery)
    pub fn list_pending(&self) -> Result<Vec<SessionRecord>, QueueError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SESSIONS_TABLE)?;

        let mut pending = Vec::new();
        for entry in table.iter()? {
            let (_, value) = entry?;
            let record: SessionRecord = rmp_serde::from_slice(value.value())
                .map_err(|e| QueueError::Serialization(e.to_string()))?;

            if record.status == SessionStatus::Pending {
                pending.push(record);
            }
        }

        Ok(pending)
    }

    /// List all sessions (pending and processing)
    pub fn list_all(&self) -> Result<Vec<SessionRecord>, QueueError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SESSIONS_TABLE)?;

        let mut sessions = Vec::new();
        for entry in table.iter()? {
            let (_, value) = entry?;
            let record: SessionRecord = rmp_serde::from_slice(value.value())
                .map_err(|e| QueueError::Serialization(e.to_string()))?;
            sessions.push(record);
        }

        Ok(sessions)
    }

    /// Get the number of sessions in the queue
    pub fn len(&self) -> Result<usize, QueueError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SESSIONS_TABLE)?;
        Ok(table.len()? as usize)
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> Result<bool, QueueError> {
        Ok(self.len()? == 0)
    }

    /// Reset any "Processing" sessions back to "Pending"
    ///
    /// Called on startup to recover from crashes during processing.
    pub fn reset_processing_to_pending(&self) -> Result<usize, QueueError> {
        let write_txn = self.db.begin_write()?;
        let mut reset_count = 0;

        {
            let mut table = write_txn.open_table(SESSIONS_TABLE)?;

            // Collect processing sessions with their full data
            let mut processing_sessions: Vec<(String, SessionRecord)> = Vec::new();
            for entry in table.iter()? {
                let (key, value) = entry?;
                let record: SessionRecord = rmp_serde::from_slice(value.value())
                    .map_err(|e| QueueError::Serialization(e.to_string()))?;

                if record.status == SessionStatus::Processing {
                    processing_sessions.push((key.value().to_string(), record));
                }
            }

            // Reset each to pending (no re-read needed)
            for (key, mut record) in processing_sessions {
                record.status = SessionStatus::Pending;

                let updated_data = rmp_serde::to_vec(&record)
                    .map_err(|e| QueueError::Serialization(e.to_string()))?;

                table.insert(key.as_str(), updated_data.as_slice())?;
                reset_count += 1;

                info!(
                    queue_item_id = %key,
                    track_id = %record.track_id,
                    "Reset processing session to pending"
                );
            }
        }

        write_txn.commit()?;
        Ok(reset_count)
    }

    /// Clean up orphaned PCM files (files in audio_dir not in queue)
    pub fn cleanup_orphaned_files(&self) -> Result<usize, QueueError> {
        // Get all PCM paths from queue
        let sessions = self.list_all()?;
        let queued_paths: std::collections::HashSet<_> =
            sessions.iter().map(|s| s.pcm_path.clone()).collect();

        let mut cleaned = 0;

        // Scan audio directory
        if let Ok(entries) = std::fs::read_dir(&self.audio_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "pcm") {
                    if !queued_paths.contains(&path) {
                        // Orphaned file - delete it
                        if let Err(e) = std::fs::remove_file(&path) {
                            warn!(path = %path.display(), error = %e, "Failed to remove orphaned PCM file");
                        } else {
                            info!(path = %path.display(), "Removed orphaned PCM file");
                            cleaned += 1;
                        }
                    }
                }
            }
        }

        Ok(cleaned)
    }

    /// Clean up old PCM files regardless of queue state.
    ///
    /// Removes any .pcm files in audio_dir older than `max_age`.
    /// This handles edge cases like database corruption or reset.
    pub fn cleanup_old_files(&self, max_age: Duration) -> Result<usize, QueueError> {
        let now = SystemTime::now();
        let mut cleaned = 0;

        if let Ok(entries) = std::fs::read_dir(&self.audio_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.extension().map_or(false, |ext| ext == "pcm") {
                    continue;
                }

                // Check file modification time
                let should_delete = match entry.metadata() {
                    Ok(meta) => match meta.modified() {
                        Ok(modified) => {
                            now.duration_since(modified).unwrap_or(Duration::ZERO) > max_age
                        }
                        Err(_) => false,
                    },
                    Err(_) => false,
                };

                if should_delete {
                    if let Err(e) = std::fs::remove_file(&path) {
                        warn!(path = %path.display(), error = %e, "Failed to remove old PCM file");
                    } else {
                        info!(path = %path.display(), "Removed old PCM file");
                        cleaned += 1;
                    }
                }
            }
        }

        Ok(cleaned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_queue() -> (AudioQueue, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("queue.redb");
        let audio_dir = temp_dir.path().join("audio");
        let queue = AudioQueue::new(&db_path, &audio_dir).unwrap();
        (queue, temp_dir)
    }

    fn create_test_record(id: &str) -> SessionRecord {
        SessionRecord {
            queue_item_id: id.to_string(),
            track_id: format!("track_{}", id),
            queue_id: "player_1".to_string(),
            sample_rate: 48000,
            channels: 2,
            pcm_path: PathBuf::from(format!("/tmp/audio/{}.pcm", id)),
            status: SessionStatus::Pending,
            created_at: chrono::Utc::now().timestamp(),
            metadata: SessionMetadata {
                name: "Test Track".to_string(),
                artists: vec!["Test Artist".to_string()],
                album: Some("Test Album".to_string()),
                genres: vec!["Rock".to_string()],
            },
        }
    }

    #[test]
    fn test_push_and_get() {
        let (queue, _temp) = create_test_queue();
        let record = create_test_record("session_1");

        queue.push(record.clone()).unwrap();

        let retrieved = queue.get("session_1").unwrap().unwrap();
        assert_eq!(retrieved.queue_item_id, "session_1");
        assert_eq!(retrieved.track_id, "track_session_1");
        assert_eq!(retrieved.status, SessionStatus::Pending);
    }

    #[test]
    fn test_pop_pending() {
        let (queue, _temp) = create_test_queue();

        queue.push(create_test_record("session_1")).unwrap();
        queue.push(create_test_record("session_2")).unwrap();

        // Pop first pending
        let popped = queue.pop_pending().unwrap().unwrap();
        assert_eq!(popped.status, SessionStatus::Processing);

        // Verify it's now marked as processing in queue
        let still_there = queue.get(&popped.queue_item_id).unwrap().unwrap();
        assert_eq!(still_there.status, SessionStatus::Processing);

        // Pop next pending
        let popped2 = queue.pop_pending().unwrap().unwrap();
        assert_ne!(popped.queue_item_id, popped2.queue_item_id);
        assert_eq!(popped2.status, SessionStatus::Processing);

        // No more pending
        assert!(queue.pop_pending().unwrap().is_none());
    }

    #[test]
    fn test_remove() {
        let (queue, _temp) = create_test_queue();

        queue.push(create_test_record("session_1")).unwrap();
        assert_eq!(queue.len().unwrap(), 1);

        let removed = queue.remove("session_1").unwrap().unwrap();
        assert_eq!(removed.queue_item_id, "session_1");
        assert_eq!(queue.len().unwrap(), 0);

        // Remove non-existent
        assert!(queue.remove("session_1").unwrap().is_none());
    }

    #[test]
    fn test_list_pending() {
        let (queue, _temp) = create_test_queue();

        queue.push(create_test_record("session_1")).unwrap();
        queue.push(create_test_record("session_2")).unwrap();

        // Pop one to make it processing
        queue.pop_pending().unwrap();

        let pending = queue.list_pending().unwrap();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_reset_processing_to_pending() {
        let (queue, _temp) = create_test_queue();

        queue.push(create_test_record("session_1")).unwrap();
        queue.push(create_test_record("session_2")).unwrap();

        // Pop both to processing
        queue.pop_pending().unwrap();
        queue.pop_pending().unwrap();

        // Verify both are processing
        let pending_before = queue.list_pending().unwrap();
        assert_eq!(pending_before.len(), 0);

        // Reset
        let reset_count = queue.reset_processing_to_pending().unwrap();
        assert_eq!(reset_count, 2);

        // Verify both are pending again
        let pending_after = queue.list_pending().unwrap();
        assert_eq!(pending_after.len(), 2);
    }

    #[test]
    fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("queue.redb");
        let audio_dir = temp_dir.path().join("audio");

        // Create queue and push
        {
            let queue = AudioQueue::new(&db_path, &audio_dir).unwrap();
            queue.push(create_test_record("session_1")).unwrap();
        }

        // Reopen and verify
        {
            let queue = AudioQueue::new(&db_path, &audio_dir).unwrap();
            let record = queue.get("session_1").unwrap().unwrap();
            assert_eq!(record.queue_item_id, "session_1");
        }
    }
}
