use config::{Config, ConfigError, Environment};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;

/// Application configuration loaded from environment variables.
///
/// All settings can be configured via environment variables with the `INSIGHT_` prefix.
/// For example: `INSIGHT_PORT=8096`, `INSIGHT_ENABLE_CUDA=true`
#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Storage mode - either file-based (usearch) or hosted (Qdrant)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageMode {
    /// File-based storage using usearch HNSW index
    #[default]
    File,
    /// Hosted Qdrant vector database
    Qdrant,
}

impl std::fmt::Display for StorageMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageMode::File => write!(f, "file"),
            StorageMode::Qdrant => write!(f, "qdrant"),
        }
    }
}

impl std::str::FromStr for StorageMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "file" | "usearch" => Ok(StorageMode::File),
            "qdrant" => Ok(StorageMode::Qdrant),
            _ => Err(format!("Unknown storage mode: {}. Use 'file' or 'qdrant'", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage mode: "file" for embedded usearch, "qdrant" for hosted Qdrant
    #[serde(default)]
    pub mode: StorageMode,

    /// Data directory for file-based storage (default: ~/.local/share/insight-sidecar)
    #[serde(default = "default_data_dir")]
    pub data_dir: String,

    /// Qdrant server URL (e.g., http://localhost:6334 or https://xxx.cloud.qdrant.io:6333)
    #[serde(default = "default_qdrant_url")]
    pub url: String,

    /// API key for Qdrant Cloud or authenticated instances
    #[serde(default)]
    pub api_key: Option<String>,

    /// Prefix for collection names (useful for multi-tenant setups)
    #[serde(default)]
    pub collection_prefix: Option<String>,

    /// Enable storage (set to false to run without vector storage)
    #[serde(default = "default_storage_enabled")]
    pub enabled: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            mode: StorageMode::default(),
            data_dir: default_data_dir(),
            url: default_qdrant_url(),
            api_key: None,
            collection_prefix: None,
            enabled: default_storage_enabled(),
        }
    }
}

fn default_data_dir() -> String {
    directories::ProjectDirs::from("com", "music-assistant", "insight-sidecar")
        .map(|dirs| dirs.data_dir().to_string_lossy().to_string())
        .unwrap_or_else(|| "./data".to_string())
}

fn default_qdrant_url() -> String {
    "http://localhost:6334".to_string()
}

fn default_storage_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Get the path to the config file
    pub fn config_file_path() -> PathBuf {
        directories::ProjectDirs::from("com", "music-assistant", "insight-sidecar")
            .map(|dirs| dirs.config_dir().join("config.toml"))
            .unwrap_or_else(|| PathBuf::from("./config.toml"))
    }

    /// Load configuration from file, falling back to defaults
    pub fn load_from_file() -> Result<Self, ConfigError> {
        let config_path = Self::config_file_path();

        let mut builder = Config::builder();

        // Load from file if it exists
        if config_path.exists() {
            builder = builder.add_source(config::File::from(config_path));
        }

        // Environment variables override file config
        builder = builder.add_source(
            Environment::with_prefix("INSIGHT")
                .separator("__")
                .try_parsing(true),
        );

        let config = builder.build()?;
        config.try_deserialize()
    }

    /// Save configuration to file
    pub fn save(&self) -> std::io::Result<()> {
        let config_path = Self::config_file_path();

        // Create parent directory if it doesn't exist
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Serialize to TOML
        let toml_string = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        std::fs::write(&config_path, toml_string)?;

        Ok(())
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
        assert_eq!(config.storage.mode, StorageMode::File);
    }

    #[test]
    fn test_socket_addr() {
        let server = ServerConfig::default();
        let addr = server.socket_addr();
        assert_eq!(addr.port(), 8096);
    }

    #[test]
    fn test_storage_mode_from_str() {
        assert_eq!("file".parse::<StorageMode>().unwrap(), StorageMode::File);
        assert_eq!("usearch".parse::<StorageMode>().unwrap(), StorageMode::File);
        assert_eq!("qdrant".parse::<StorageMode>().unwrap(), StorageMode::Qdrant);
        assert!("invalid".parse::<StorageMode>().is_err());
    }

    #[test]
    fn test_storage_mode_display() {
        assert_eq!(StorageMode::File.to_string(), "file");
        assert_eq!(StorageMode::Qdrant.to_string(), "qdrant");
    }
}
