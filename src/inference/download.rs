//! Model downloading from Hugging Face Hub.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use directories::ProjectDirs;
use futures_util::StreamExt;
use reqwest::Client;
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::InferenceError;
use crate::types::{DownloadProgress, DownloadStatus};

/// Manages active downloads with progress tracking
#[derive(Clone)]
pub struct DownloadManager {
    /// Active downloads indexed by download_id
    downloads: Arc<RwLock<HashMap<String, DownloadProgress>>>,
    /// HTTP client for downloads
    client: Client,
}

impl Default for DownloadManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DownloadManager {
    /// Create a new download manager
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent("insight-sidecar/0.1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            downloads: Arc::new(RwLock::new(HashMap::new())),
            client,
        }
    }

    /// Start downloading a model, returns download ID
    pub async fn start_download(&self, model_id: String) -> Result<String, InferenceError> {
        let download_id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp();

        let progress = DownloadProgress {
            download_id: download_id.clone(),
            model_id: model_id.clone(),
            status: DownloadStatus::Pending,
            bytes_downloaded: 0,
            bytes_total: None,
            progress_percent: 0.0,
            started_at: now,
            completed_at: None,
            error: None,
            current_file: None,
        };

        {
            let mut downloads = self.downloads.write().await;
            downloads.insert(download_id.clone(), progress);
        }

        // Spawn download task
        let manager = self.clone();
        let id = download_id.clone();
        tokio::spawn(async move {
            let result = manager.run_download(&id, &model_id).await;
            if let Err(e) = result {
                manager.mark_failed(&id, &e.to_string()).await;
            }
        });

        Ok(download_id)
    }

    /// Get current download progress
    pub async fn get_progress(&self, download_id: &str) -> Option<DownloadProgress> {
        let downloads = self.downloads.read().await;
        downloads.get(download_id).cloned()
    }

    /// Get all downloads
    pub async fn list_downloads(&self) -> Vec<DownloadProgress> {
        let downloads = self.downloads.read().await;
        downloads.values().cloned().collect()
    }

    /// Get active downloads (pending or downloading)
    pub async fn active_downloads(&self) -> Vec<DownloadProgress> {
        let downloads = self.downloads.read().await;
        downloads
            .values()
            .filter(|d| {
                matches!(
                    d.status,
                    DownloadStatus::Pending | DownloadStatus::Downloading
                )
            })
            .cloned()
            .collect()
    }

    /// Cancel a download
    pub async fn cancel_download(&self, download_id: &str) -> bool {
        let mut downloads = self.downloads.write().await;
        if let Some(progress) = downloads.get_mut(download_id) {
            if matches!(
                progress.status,
                DownloadStatus::Pending | DownloadStatus::Downloading
            ) {
                progress.status = DownloadStatus::Cancelled;
                progress.completed_at = Some(chrono::Utc::now().timestamp());
                return true;
            }
        }
        false
    }

    /// Clear completed/failed/cancelled downloads from the list
    pub async fn clear_finished(&self) {
        let mut downloads = self.downloads.write().await;
        downloads.retain(|_, d| {
            matches!(
                d.status,
                DownloadStatus::Pending | DownloadStatus::Downloading
            )
        });
    }

    /// Run the actual download
    async fn run_download(&self, download_id: &str, model_id: &str) -> Result<(), InferenceError> {
        let config = ModelConfig::from_model_id(model_id);
        let model_cache = get_model_dir(model_id);
        fs::create_dir_all(&model_cache).await?;

        // Update status to downloading
        {
            let mut downloads = self.downloads.write().await;
            if let Some(progress) = downloads.get_mut(download_id) {
                progress.status = DownloadStatus::Downloading;
            }
        }

        // Download text model
        let text_model_path = model_cache.join(&config.text_model_file);
        if !text_model_path.exists() {
            self.download_file_with_progress(
                download_id,
                &config.model_id,
                &config.text_model_file,
                &text_model_path,
            )
            .await?;
        }

        // Check if cancelled
        if self.is_cancelled(download_id).await {
            return Err(InferenceError::DownloadFailed("Cancelled".to_string()));
        }

        // Download audio model
        let audio_model_path = model_cache.join(&config.audio_model_file);
        if !audio_model_path.exists() {
            self.download_file_with_progress(
                download_id,
                &config.model_id,
                &config.audio_model_file,
                &audio_model_path,
            )
            .await?;
        }

        // Check if cancelled
        if self.is_cancelled(download_id).await {
            return Err(InferenceError::DownloadFailed("Cancelled".to_string()));
        }

        // Download tokenizer if available
        if let Some(tokenizer_file) = &config.tokenizer_file {
            let path = model_cache.join(tokenizer_file);
            if !path.exists() {
                if let Err(e) = self
                    .download_file_with_progress(download_id, &config.model_id, tokenizer_file, &path)
                    .await
                {
                    warn!("Failed to download tokenizer (optional): {e}");
                }
            }
        }

        // Mark as completed
        self.mark_completed(download_id).await;
        Ok(())
    }

    async fn is_cancelled(&self, download_id: &str) -> bool {
        let downloads = self.downloads.read().await;
        downloads
            .get(download_id)
            .is_some_and(|d| d.status == DownloadStatus::Cancelled)
    }

    async fn mark_completed(&self, download_id: &str) {
        let mut downloads = self.downloads.write().await;
        if let Some(progress) = downloads.get_mut(download_id) {
            progress.status = DownloadStatus::Completed;
            progress.progress_percent = 100.0;
            progress.completed_at = Some(chrono::Utc::now().timestamp());
            progress.current_file = None;
        }
    }

    async fn mark_failed(&self, download_id: &str, error: &str) {
        let mut downloads = self.downloads.write().await;
        if let Some(progress) = downloads.get_mut(download_id) {
            progress.status = DownloadStatus::Failed;
            progress.error = Some(error.to_string());
            progress.completed_at = Some(chrono::Utc::now().timestamp());
        }
    }

    async fn update_progress(
        &self,
        download_id: &str,
        bytes_downloaded: u64,
        bytes_total: Option<u64>,
        current_file: &str,
    ) {
        let mut downloads = self.downloads.write().await;
        if let Some(progress) = downloads.get_mut(download_id) {
            progress.bytes_downloaded = bytes_downloaded;
            progress.bytes_total = bytes_total;
            progress.current_file = Some(current_file.to_string());
            if let Some(total) = bytes_total {
                if total > 0 {
                    progress.progress_percent = (bytes_downloaded as f32 / total as f32) * 100.0;
                }
            }
        }
    }

    async fn download_file_with_progress(
        &self,
        download_id: &str,
        model_id: &str,
        filename: &str,
        dest: &Path,
    ) -> Result<(), InferenceError> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/onnx/{}",
            model_id, filename
        );

        info!(%url, ?dest, "Downloading model file");

        let response = self
            .client
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

            let response = self
                .client
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

            return self
                .download_response_with_progress(download_id, response, filename, dest)
                .await;
        }

        self.download_response_with_progress(download_id, response, filename, dest)
            .await
    }

    async fn download_response_with_progress(
        &self,
        download_id: &str,
        response: reqwest::Response,
        filename: &str,
        dest: &Path,
    ) -> Result<(), InferenceError> {
        let total_size = response.content_length();

        // Create temp file for atomic write
        let temp_path = dest.with_extension("tmp");
        let mut file = fs::File::create(&temp_path).await?;

        let mut hasher = Sha256::new();
        let mut downloaded: u64 = 0;

        // Stream the response with progress updates
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            // Check for cancellation
            if self.is_cancelled(download_id).await {
                // Clean up temp file
                let _ = fs::remove_file(&temp_path).await;
                return Err(InferenceError::DownloadFailed("Cancelled".to_string()));
            }

            let chunk =
                chunk.map_err(|e| InferenceError::DownloadFailed(format!("Download failed: {e}")))?;

            hasher.update(&chunk);
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            // Update progress
            self.update_progress(download_id, downloaded, total_size, filename)
                .await;
        }

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
}

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

/// Alias for default_cache_dir for consistency with management API
pub fn get_cache_dir() -> PathBuf {
    default_cache_dir()
}

/// Get the cache directory for a specific model
pub fn get_model_dir(model_id: &str) -> PathBuf {
    default_cache_dir().join(model_id.replace('/', "__"))
}

/// Check if a model is downloaded (has required files)
pub fn is_model_downloaded(model_id: &str) -> bool {
    let model_dir = get_model_dir(model_id);
    if !model_dir.exists() {
        return false;
    }

    let config = ModelConfig::from_model_id(model_id);
    let text_model = model_dir.join(&config.text_model_file);
    let audio_model = model_dir.join(&config.audio_model_file);

    text_model.exists() && audio_model.exists()
}

/// Get total size of cached model files in bytes
pub fn get_model_size(model_id: &str) -> Option<u64> {
    let model_dir = get_model_dir(model_id);
    if !model_dir.exists() {
        return None;
    }

    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(&model_dir) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }

    if total > 0 { Some(total) } else { None }
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
