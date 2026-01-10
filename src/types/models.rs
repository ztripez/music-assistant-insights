//! API types for model management operations.
//!
//! This module contains request/response types for listing, downloading,
//! loading, and deleting ML models.

use serde::{Deserialize, Serialize};

use crate::types::{DownloadProgress, ModelDetail, StorageStats, SystemStatus};

/// Response for listing models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    /// List of all models (known + cached)
    pub models: Vec<ModelDetail>,
    /// Currently loaded model ID (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_model: Option<String>,
}

/// Request to download a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadModelRequest {
    /// Model ID to download (`HuggingFace` format: owner/model-name)
    pub model_id: String,
}

/// Response from starting a download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadModelResponse {
    /// Unique download ID for tracking progress (None if already exists)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_id: Option<String>,
    /// Model being downloaded
    pub model_id: String,
    /// Message
    pub message: String,
    /// Whether the model already exists
    #[serde(default)]
    pub already_exists: bool,
}

/// Response for listing active downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDownloadsResponse {
    /// Active and recent downloads
    pub downloads: Vec<DownloadProgress>,
}

/// Request to load a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelRequest {
    /// Model ID to load (must be downloaded first)
    pub model_id: String,
}

/// Response from loading a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelResponse {
    /// Model that was loaded
    pub model_id: String,
    /// Whether load was successful
    pub loaded: bool,
    /// Status message
    pub message: String,
    /// Device model is running on
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// Response from deleting a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteModelResponse {
    /// Model that was deleted
    pub model_id: String,
    /// Whether delete was successful
    pub deleted: bool,
    /// Status message
    pub message: String,
}

// ============================================================================
// Configuration types
// ============================================================================

/// Request to update configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfigRequest {
    /// Model configuration updates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<UpdateModelConfig>,
    /// Storage configuration updates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<UpdateStorageConfig>,
    /// Server configuration updates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server: Option<UpdateServerConfig>,
}

/// Model configuration updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateModelConfig {
    /// Model name/ID to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Enable CUDA acceleration (NVIDIA GPUs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_cuda: Option<bool>,
    /// Enable `ROCm` acceleration (AMD GPUs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_rocm: Option<bool>,
    /// Enable `CoreML` acceleration (Apple Silicon/macOS)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_coreml: Option<bool>,
    /// Enable `DirectML` acceleration (Windows GPU)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_directml: Option<bool>,
    /// Enable `OpenVINO` acceleration (Intel CPUs/GPUs/VPUs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_openvino: Option<bool>,
}

/// Storage configuration updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStorageConfig {
    /// Storage mode (file or qdrant)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Data directory for file storage
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<String>,
    /// Qdrant URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Qdrant API key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// Server configuration updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateServerConfig {
    /// Host to bind to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    /// Port to listen on
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
}

/// Response from updating configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfigResponse {
    /// Whether update was successful
    pub success: bool,
    /// Message describing what was updated
    pub message: String,
    /// Fields that require restart to take effect
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub requires_restart: Vec<String>,
}

// ============================================================================
// Status types
// ============================================================================

/// Response for storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStatsResponse {
    /// Storage statistics
    pub stats: StorageStats,
}

/// Response for system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatusResponse {
    /// Full system status
    #[serde(flatten)]
    pub status: SystemStatus,
}
