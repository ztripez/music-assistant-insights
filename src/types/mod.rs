//! Shared types for the insight sidecar API.
//!
//! These types are used across the application for request/response handling
//! and internal data representation.

use serde::{Deserialize, Serialize};

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub version: String,
    pub model_loaded: bool,
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
