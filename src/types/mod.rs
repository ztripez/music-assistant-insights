//! Shared types for the insight sidecar API.
//!
//! These types are used across the application for request/response handling
//! and internal data representation.

pub mod api;

use serde::{Deserialize, Serialize};

pub use api::*;

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub version: String,
    #[serde(default)]
    pub model_loaded: bool,
    #[serde(default)]
    pub storage_ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Configuration response (subset of config safe to expose)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigResponse {
    pub model: ModelInfo,
    pub audio: AudioInfo,
    pub server: ServerInfo,
    pub storage: StorageInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub cuda_enabled: bool,
    pub loaded: bool,
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInfo {
    pub window_size_s: f32,
    pub hop_size_s: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    pub url: String,
    pub enabled: bool,
    pub connected: bool,
}
