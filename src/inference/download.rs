//! Model downloading from Hugging Face Hub.

use std::path::{Path, PathBuf};

use directories::ProjectDirs;
use reqwest::Client;
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

use super::InferenceError;

/// Paths to downloaded model files
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Path to the text encoder ONNX model
    pub text_model: PathBuf,
    /// Path to the audio encoder ONNX model
    pub audio_model: PathBuf,
    /// Path to the tokenizer config (if any)
    pub tokenizer_config: Option<PathBuf>,
}

/// Known CLAP model configurations
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: String,
    pub text_model_file: String,
    pub audio_model_file: String,
    pub tokenizer_file: Option<String>,
}

impl ModelConfig {
    /// Get configuration for known models
    pub fn from_model_id(model_id: &str) -> Self {
        match model_id {
            "Xenova/clap-htsat-unfused" => Self {
                model_id: model_id.to_string(),
                text_model_file: "text_model.onnx".to_string(),
                audio_model_file: "audio_model.onnx".to_string(),
                tokenizer_file: Some("tokenizer.json".to_string()),
            },
            "laion/larger_clap_music" => Self {
                model_id: model_id.to_string(),
                text_model_file: "text_model.onnx".to_string(),
                audio_model_file: "audio_model.onnx".to_string(),
                tokenizer_file: Some("tokenizer.json".to_string()),
            },
            _ => Self {
                model_id: model_id.to_string(),
                text_model_file: "text_model.onnx".to_string(),
                audio_model_file: "audio_model.onnx".to_string(),
                tokenizer_file: None,
            },
        }
    }
}

/// Get the default cache directory for models
pub fn default_cache_dir() -> PathBuf {
    if let Some(proj_dirs) = ProjectDirs::from("com", "music-assistant", "insight-sidecar") {
        proj_dirs.cache_dir().join("models")
    } else {
        PathBuf::from("./cache/models")
    }
}

/// Download a model from Hugging Face Hub if not already cached
pub async fn download_model(
    model_id: &str,
    cache_dir: Option<&Path>,
) -> Result<ModelPaths, InferenceError> {
    let config = ModelConfig::from_model_id(model_id);
    let cache_dir = cache_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_cache_dir);

    // Create model-specific cache directory
    let model_cache = cache_dir.join(model_id.replace('/', "__"));
    fs::create_dir_all(&model_cache).await?;

    info!(model_id, ?model_cache, "Checking model cache");

    let client = Client::builder()
        .user_agent("insight-sidecar/0.1.0")
        .build()
        .map_err(|e| InferenceError::DownloadFailed(e.to_string()))?;

    // Download text model
    let text_model_path = model_cache.join(&config.text_model_file);
    if !text_model_path.exists() {
        download_file(
            &client,
            &config.model_id,
            &config.text_model_file,
            &text_model_path,
        )
        .await?;
    } else {
        debug!(?text_model_path, "Text model already cached");
    }

    // Download audio model
    let audio_model_path = model_cache.join(&config.audio_model_file);
    if !audio_model_path.exists() {
        download_file(
            &client,
            &config.model_id,
            &config.audio_model_file,
            &audio_model_path,
        )
        .await?;
    } else {
        debug!(?audio_model_path, "Audio model already cached");
    }

    // Download tokenizer if available
    let tokenizer_path = if let Some(tokenizer_file) = &config.tokenizer_file {
        let path = model_cache.join(tokenizer_file);
        if !path.exists() {
            match download_file(&client, &config.model_id, tokenizer_file, &path).await {
                Ok(()) => Some(path),
                Err(e) => {
                    warn!("Failed to download tokenizer (optional): {e}");
                    None
                }
            }
        } else {
            Some(path)
        }
    } else {
        None
    };

    Ok(ModelPaths {
        text_model: text_model_path,
        audio_model: audio_model_path,
        tokenizer_config: tokenizer_path,
    })
}

/// Download a single file from Hugging Face Hub
async fn download_file(
    client: &Client,
    model_id: &str,
    filename: &str,
    dest: &Path,
) -> Result<(), InferenceError> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/onnx/{}",
        model_id, filename
    );

    info!(%url, ?dest, "Downloading model file");

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| InferenceError::DownloadFailed(format!("Request failed: {e}")))?;

    if !response.status().is_success() {
        // Try alternative path without onnx/ prefix
        let alt_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_id, filename
        );
        debug!(%alt_url, "Trying alternative URL");

        let response = client
            .get(&alt_url)
            .send()
            .await
            .map_err(|e| InferenceError::DownloadFailed(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(InferenceError::DownloadFailed(format!(
                "HTTP {}: {}",
                response.status(),
                alt_url
            )));
        }

        return download_response(response, dest).await;
    }

    download_response(response, dest).await
}

async fn download_response(response: reqwest::Response, dest: &Path) -> Result<(), InferenceError> {
    // Create temp file for atomic write
    let temp_path = dest.with_extension("tmp");
    let mut file = fs::File::create(&temp_path).await?;

    let mut hasher = Sha256::new();

    // Download entire response at once (simpler approach)
    let bytes = response
        .bytes()
        .await
        .map_err(|e| InferenceError::DownloadFailed(format!("Download failed: {e}")))?;

    hasher.update(&bytes);
    file.write_all(&bytes).await?;
    let downloaded = bytes.len() as u64;

    file.flush().await?;
    drop(file);

    // Rename temp file to final destination
    fs::rename(&temp_path, dest).await?;

    let hash = hex::encode(hasher.finalize());
    info!(
        ?dest,
        bytes = downloaded,
        sha256 = %hash,
        "Download complete"
    );

    Ok(())
}

/// Convert bytes to hex string (simple implementation)
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}
