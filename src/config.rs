use config::{Config, ConfigError, Environment};
use serde::Deserialize;
use std::net::SocketAddr;

/// Application configuration loaded from environment variables.
///
/// All settings can be configured via environment variables with the `INSIGHT_` prefix.
/// For example: `INSIGHT_PORT=8096`, `INSIGHT_ENABLE_CUDA=true`
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    /// Model configuration
    #[serde(default)]
    pub model: ModelConfig,

    /// Audio processing configuration
    #[serde(default)]
    pub audio: AudioConfig,

    /// Storage configuration
    #[serde(default)]
    pub storage: StorageConfig,

    /// Server configuration
    #[serde(default)]
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// CLAP model to use (Hugging Face model ID)
    #[serde(default = "default_model")]
    pub name: String,

    /// Enable CUDA acceleration
    #[serde(default)]
    pub enable_cuda: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: default_model(),
            enable_cuda: false,
        }
    }
}

fn default_model() -> String {
    "Xenova/clap-htsat-unfused".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    /// Window size in seconds for audio processing
    #[serde(default = "default_window_size")]
    pub window_size_s: f32,

    /// Hop size in seconds between windows
    #[serde(default = "default_hop_size")]
    pub hop_size_s: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            window_size_s: default_window_size(),
            hop_size_s: default_hop_size(),
        }
    }
}

fn default_window_size() -> f32 {
    10.0
}

fn default_hop_size() -> f32 {
    10.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct StorageConfig {
    /// Path to Qdrant storage directory
    #[serde(default = "default_storage_path")]
    pub path: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: default_storage_path(),
        }
    }
}

fn default_storage_path() -> String {
    "./data/qdrant".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to
    #[serde(default = "default_host")]
    pub host: String,

    /// Port to listen on
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8096
}

impl ServerConfig {
    /// Returns the socket address for binding the server
    pub fn socket_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("Invalid socket address")
    }
}

impl AppConfig {
    /// Load configuration from environment variables.
    ///
    /// Environment variables should be prefixed with `INSIGHT_` and use
    /// double underscores for nested values:
    /// - `INSIGHT_MODEL__NAME` -> model.name
    /// - `INSIGHT_MODEL__ENABLE_CUDA` -> model.enable_cuda
    /// - `INSIGHT_SERVER__PORT` -> server.port
    pub fn load() -> Result<Self, ConfigError> {
        let config = Config::builder()
            .add_source(
                Environment::with_prefix("INSIGHT")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?;

        config.try_deserialize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig {
            model: ModelConfig::default(),
            audio: AudioConfig::default(),
            storage: StorageConfig::default(),
            server: ServerConfig::default(),
        };

        assert_eq!(config.model.name, "Xenova/clap-htsat-unfused");
        assert!(!config.model.enable_cuda);
        assert_eq!(config.audio.window_size_s, 10.0);
        assert_eq!(config.server.port, 8096);
    }

    #[test]
    fn test_socket_addr() {
        let server = ServerConfig::default();
        let addr = server.socket_addr();
        assert_eq!(addr.port(), 8096);
    }
}
