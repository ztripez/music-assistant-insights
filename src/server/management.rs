//! Management API route handlers.
//!
//! Provides endpoints for:
//! - System status and statistics
//! - Model management (list, download, load, delete)
//! - Download progress tracking
//! - Storage statistics

use axum::extract::{Path, State};
#[cfg(feature = "inference")]
use tracing::info;

use crate::error::AppError;
#[cfg(any(feature = "storage", feature = "storage-file"))]
use crate::types::StorageStatsResponse;
#[cfg(feature = "inference")]
use crate::types::{
    DeleteModelResponse, DownloadModelRequest, DownloadModelResponse, ListDownloadsResponse,
    ListModelsResponse, LoadModelRequest, LoadModelResponse, ModelStatus,
};
use crate::types::{HealthStatus, ModelDetail, StorageStats, SystemStatus, SystemStatusResponse};

#[cfg(feature = "inference")]
use super::extractors::MsgPackExtractor;
use super::routes::MsgPack;
use super::AppState;

#[cfg(feature = "inference")]
use crate::inference::{
    download_model, get_cache_dir, get_model_dir, get_model_size, is_model_downloaded, ClapModel,
    KnownModel, KNOWN_MODELS,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// GET /api/v1/status
///
/// Get comprehensive system status including model, storage, and system info.
pub async fn status(State(state): State<AppState>) -> MsgPack<SystemStatusResponse> {
    #[cfg(feature = "inference")]
    let model_loaded = state.has_model().await;
    #[cfg(not(feature = "inference"))]
    let model_loaded = false;
    let _ = model_loaded; // Used conditionally below

    #[cfg(any(feature = "storage", feature = "storage-file"))]
    let _storage_connected = state.storage.is_some();
    #[cfg(not(any(feature = "storage", feature = "storage-file")))]
    let _storage_connected = false;

    // Build health status
    #[cfg(feature = "inference")]
    let health = if model_loaded {
        HealthStatus::Healthy
    } else {
        HealthStatus::Degraded
    };
    #[cfg(not(feature = "inference"))]
    let health = HealthStatus::Healthy;

    // Build model detail
    #[cfg(feature = "inference")]
    let model = {
        let guard = state.model.read().await;
        let model_id = state.get_current_model_id().await;
        if let (Some(ref m), Some(ref id)) = (&*guard, model_id) {
            Some(ModelDetail {
                model_id: id.clone(),
                name: KnownModel::get(id)
                    .map(|k| k.name.to_string())
                    .unwrap_or_else(|| id.clone()),
                description: KnownModel::get(id).map(|k| k.description.to_string()),
                status: ModelStatus::Loaded,
                estimated_size_bytes: KnownModel::get(id).map(|k| k.estimated_size_bytes()),
                actual_size_bytes: get_model_size(id),
                cache_path: Some(get_model_dir(id).to_string_lossy().to_string()),
                recommended: KnownModel::is_known(id),
                is_current: true,
                device: Some(m.device().to_string()),
            })
        } else {
            None
        }
    };
    #[cfg(not(feature = "inference"))]
    let model: Option<ModelDetail> = None;

    // Build storage stats
    #[cfg(any(feature = "storage", feature = "storage-file"))]
    let storage = {
        if let Some(ref storage) = state.storage {
            let text_count = storage.count("text").await.unwrap_or(0);
            let audio_count = storage.count("audio").await.unwrap_or(0);
            StorageStats {
                mode: state.config.storage.mode.to_string(),
                connected: true,
                text_collection_count: text_count,
                audio_collection_count: audio_count,
                total_tracks: text_count.max(audio_count),
            }
        } else {
            StorageStats {
                mode: state.config.storage.mode.to_string(),
                connected: false,
                text_collection_count: 0,
                audio_collection_count: 0,
                total_tracks: 0,
            }
        }
    };
    #[cfg(not(any(feature = "storage", feature = "storage-file")))]
    let storage = StorageStats {
        mode: "disabled".to_string(),
        connected: false,
        text_collection_count: 0,
        audio_collection_count: 0,
        total_tracks: 0,
    };

    // Build features list
    let mut features = Vec::new();
    #[cfg(feature = "inference")]
    features.push("inference".to_string());
    #[cfg(feature = "storage")]
    features.push("storage".to_string());
    #[cfg(feature = "storage-file")]
    features.push("storage-file".to_string());
    #[cfg(feature = "audio")]
    features.push("audio".to_string());
    #[cfg(feature = "cuda")]
    features.push("cuda".to_string());

    MsgPack(SystemStatusResponse {
        status: SystemStatus {
            version: VERSION.to_string(),
            health,
            uptime_seconds: state.uptime_seconds(),
            model,
            storage,
            features,
        },
    })
}

/// GET /api/v1/models
///
/// List all available models (known + locally cached).
#[cfg(feature = "inference")]
pub async fn list_models(State(state): State<AppState>) -> MsgPack<ListModelsResponse> {
    let current_model_id = state.get_current_model_id().await;
    let model_guard = state.model.read().await;

    let mut models = Vec::new();

    // Add known/recommended models
    for known in KNOWN_MODELS {
        let downloaded = is_model_downloaded(known.model_id);
        let is_current = current_model_id
            .as_ref()
            .is_some_and(|id| id == known.model_id);

        let status = if is_current && model_guard.is_some() {
            ModelStatus::Loaded
        } else if downloaded {
            ModelStatus::Downloaded
        } else {
            ModelStatus::NotDownloaded
        };

        let device = if is_current {
            model_guard.as_ref().map(|m| m.device().to_string())
        } else {
            None
        };

        models.push(ModelDetail {
            model_id: known.model_id.to_string(),
            name: known.name.to_string(),
            description: Some(known.description.to_string()),
            status,
            estimated_size_bytes: Some(known.estimated_size_bytes()),
            actual_size_bytes: get_model_size(known.model_id),
            cache_path: if downloaded {
                Some(get_model_dir(known.model_id).to_string_lossy().to_string())
            } else {
                None
            },
            recommended: true,
            is_current,
            device,
        });
    }

    // Scan cache directory for additional models
    let cache_dir = get_cache_dir();
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                let dir_name = entry.file_name().to_string_lossy().to_string();
                // Convert directory name back to model ID (__ -> /)
                let model_id = dir_name.replace("__", "/");

                // Skip if already in list (known model)
                if models.iter().any(|m| m.model_id == model_id) {
                    continue;
                }

                // Check if it's a valid model directory
                if is_model_downloaded(&model_id) {
                    let is_current = current_model_id.as_ref().is_some_and(|id| id == &model_id);

                    let status = if is_current && model_guard.is_some() {
                        ModelStatus::Loaded
                    } else {
                        ModelStatus::Downloaded
                    };

                    let device = if is_current {
                        model_guard.as_ref().map(|m| m.device().to_string())
                    } else {
                        None
                    };

                    models.push(ModelDetail {
                        model_id: model_id.clone(),
                        name: model_id.clone(),
                        description: None,
                        status,
                        estimated_size_bytes: None,
                        actual_size_bytes: get_model_size(&model_id),
                        cache_path: Some(get_model_dir(&model_id).to_string_lossy().to_string()),
                        recommended: false,
                        is_current,
                        device,
                    });
                }
            }
        }
    }

    MsgPack(ListModelsResponse {
        models,
        current_model: current_model_id,
    })
}

/// POST /api/v1/models/download
///
/// Start downloading a model. Returns immediately with download ID.
#[cfg(feature = "inference")]
pub async fn start_download(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<DownloadModelRequest>,
) -> Result<MsgPack<DownloadModelResponse>, AppError> {
    info!(model_id = %req.model_id, "Starting model download");

    // Check if already downloaded
    if is_model_downloaded(&req.model_id) {
        return Ok(MsgPack(DownloadModelResponse {
            download_id: None,
            model_id: req.model_id,
            message: "Model already downloaded".to_string(),
            already_exists: true,
        }));
    }

    // Check for active download
    let active = state.download_manager.active_downloads().await;
    if active.iter().any(|d| d.model_id == req.model_id) {
        return Err(AppError::BadRequest(format!(
            "Model {} is already being downloaded",
            req.model_id
        )));
    }

    // Start download
    let download_id = state
        .download_manager
        .start_download(req.model_id.clone())
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(MsgPack(DownloadModelResponse {
        download_id: Some(download_id),
        model_id: req.model_id,
        message: "Download started".to_string(),
        already_exists: false,
    }))
}

/// GET /api/v1/models/downloads
///
/// Get status of all downloads (active and recent).
#[cfg(feature = "inference")]
pub async fn list_downloads(State(state): State<AppState>) -> MsgPack<ListDownloadsResponse> {
    let downloads = state.download_manager.list_downloads().await;
    MsgPack(ListDownloadsResponse { downloads })
}

/// POST /api/v1/models/{model_id}/load
///
/// Load a downloaded model (hot-swap).
#[cfg(feature = "inference")]
pub async fn load_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    MsgPackExtractor(_req): MsgPackExtractor<LoadModelRequest>,
) -> Result<MsgPack<LoadModelResponse>, AppError> {
    info!(model_id = %model_id, "Loading model");

    // Check if already loaded
    if let Some(current) = state.get_current_model_id().await {
        if current == model_id {
            return Ok(MsgPack(LoadModelResponse {
                model_id,
                loaded: true,
                message: "Model already loaded".to_string(),
                device: state
                    .model
                    .read()
                    .await
                    .as_ref()
                    .map(|m| m.device().to_string()),
            }));
        }
    }

    // Check if model is downloaded
    if !is_model_downloaded(&model_id) {
        return Err(AppError::BadRequest(format!(
            "Model {} is not downloaded",
            model_id
        )));
    }

    // Download model files (will use cache)
    let paths = download_model(&model_id, None)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    // Load model
    let use_cuda = state.config.model.enable_cuda;
    let loaded_model = tokio::task::spawn_blocking(move || ClapModel::load(&paths, use_cuda))
        .await
        .map_err(|e| AppError::Internal(format!("Model loading task panicked: {e}")))?
        .map_err(|e| AppError::Internal(format!("Failed to load model: {e}")))?;

    let device = loaded_model.device().to_string();

    // Swap in the new model
    state
        .load_model(loaded_model, model_id.clone())
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    info!(model_id = %model_id, device = %device, "Model loaded successfully");

    Ok(MsgPack(LoadModelResponse {
        model_id,
        loaded: true,
        message: "Model loaded successfully".to_string(),
        device: Some(device),
    }))
}

/// DELETE /api/v1/models/{model_id}
///
/// Delete a cached model. Cannot delete the currently loaded model.
#[cfg(feature = "inference")]
pub async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<MsgPack<DeleteModelResponse>, AppError> {
    info!(model_id = %model_id, "Deleting model");

    // Check if it's the current model
    if let Some(current) = state.get_current_model_id().await {
        if current == model_id {
            return Err(AppError::BadRequest(
                "Cannot delete currently loaded model".to_string(),
            ));
        }
    }

    let model_dir = get_model_dir(&model_id);
    if !model_dir.exists() {
        return Err(AppError::NotFound(format!(
            "Model {} is not cached",
            model_id
        )));
    }

    // Delete the directory
    tokio::fs::remove_dir_all(&model_dir)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to delete model: {e}")))?;

    info!(model_id = %model_id, "Model deleted");

    Ok(MsgPack(DeleteModelResponse {
        model_id,
        deleted: true,
        message: "Model deleted successfully".to_string(),
    }))
}

/// GET /api/v1/storage/stats
///
/// Get storage statistics.
#[cfg(any(feature = "storage", feature = "storage-file"))]
pub async fn storage_stats(State(state): State<AppState>) -> MsgPack<StorageStatsResponse> {
    let stats = if let Some(ref storage) = state.storage {
        let text_count = storage.count("text").await.unwrap_or(0);
        let audio_count = storage.count("audio").await.unwrap_or(0);
        StorageStats {
            mode: state.config.storage.mode.to_string(),
            connected: true,
            text_collection_count: text_count,
            audio_collection_count: audio_count,
            total_tracks: text_count.max(audio_count),
        }
    } else {
        StorageStats {
            mode: state.config.storage.mode.to_string(),
            connected: false,
            text_collection_count: 0,
            audio_collection_count: 0,
            total_tracks: 0,
        }
    };

    MsgPack(StorageStatsResponse { stats })
}

// Fallback handlers for when features are disabled

#[cfg(not(feature = "inference"))]
pub async fn list_models(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Inference feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "inference"))]
pub async fn start_download(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Inference feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "inference"))]
pub async fn list_downloads(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Inference feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "inference"))]
pub async fn load_model(
    State(_state): State<AppState>,
    Path(_model_id): Path<String>,
) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Inference feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "inference"))]
pub async fn delete_model(
    State(_state): State<AppState>,
    Path(_model_id): Path<String>,
) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Inference feature not enabled".to_string(),
    ))
}

#[cfg(not(any(feature = "storage", feature = "storage-file")))]
pub async fn storage_stats(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Storage feature not enabled".to_string(),
    ))
}
