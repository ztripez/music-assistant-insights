//! Embedding generation route handlers.

use axum::extract::State;

use crate::error::AppError;

#[cfg(feature = "inference")]
use crate::types::{AudioEmbedRequest, AudioEmbedResponse, TextEmbedRequest, TextEmbedResponse};

#[cfg(feature = "inference")]
use super::extractors::MsgPackExtractor;
#[cfg(feature = "inference")]
use super::routes::MsgPack;
use super::AppState;

#[cfg(feature = "inference")]
use crate::inference::{format_track_metadata, AudioData, TextTrackMetadata};

#[cfg(feature = "inference")]
use tracing::{debug, error, info};

/// POST /api/v1/embed/text
///
/// Generate text embedding from track metadata or raw text.
#[cfg(feature = "inference")]
pub async fn text_embed(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<TextEmbedRequest>,
) -> Result<MsgPack<TextEmbedResponse>, AppError> {
    let model = state
        .model
        .as_ref()
        .ok_or_else(|| AppError::Internal("Model not loaded".to_string()))?;

    // Determine the text to embed
    let text = if let Some(text) = req.text {
        // Use raw text if provided
        text
    } else if let Some(metadata) = req.metadata {
        // Format metadata into text
        let track_meta = TextTrackMetadata {
            name: metadata.name,
            artists: metadata.artists,
            album: metadata.album,
            genres: metadata.genres,
            mood: None,
        };
        format_track_metadata(&track_meta)
    } else {
        return Err(AppError::BadRequest(
            "Either 'text' or 'metadata' must be provided".to_string(),
        ));
    };

    info!(text_len = text.len(), "Generating text embedding");

    // Generate embedding using the model
    let embedding = tokio::task::spawn_blocking({
        let model = model.clone();
        let text = text.clone();
        move || model.text_embedding(&text)
    })
    .await
    .map_err(|e| {
        error!(error = %e, "Text embedding task panicked");
        AppError::Internal(e.to_string())
    })?
    .map_err(|e| {
        error!(error = %e, "Text embedding failed");
        AppError::from(e)
    })?;

    debug!(embedding_dim = embedding.data().len(), "Text embedding generated");

    Ok(MsgPack(TextEmbedResponse {
        embedding: embedding.into_data(),
        text,
    }))
}

/// POST /api/v1/embed/audio
///
/// Generate audio embedding from raw PCM audio data.
#[cfg(feature = "inference")]
pub async fn audio_embed(
    State(state): State<AppState>,
    MsgPackExtractor(req): MsgPackExtractor<AudioEmbedRequest>,
) -> Result<MsgPack<AudioEmbedResponse>, AppError> {
    let model = state
        .model
        .as_ref()
        .ok_or_else(|| AppError::Internal("Model not loaded".to_string()))?;

    // Convert request to AudioData
    let audio_data: AudioData = req.into();

    // Calculate duration for response
    let samples_per_channel = match audio_data.format {
        crate::inference::AudioFormat::PcmF32Le => audio_data.data.len() / 4,
        crate::inference::AudioFormat::PcmS16Le => audio_data.data.len() / 2,
        crate::inference::AudioFormat::PcmS24Le => audio_data.data.len() / 3,
    };
    let total_samples = samples_per_channel / audio_data.channels as usize;
    let duration_s = total_samples as f32 / audio_data.sample_rate as f32;

    info!(
        format = ?audio_data.format,
        sample_rate = audio_data.sample_rate,
        channels = audio_data.channels,
        duration_s,
        "Generating audio embedding"
    );

    // Generate embedding using the model
    let embedding = tokio::task::spawn_blocking({
        let model = model.clone();
        move || model.audio_embedding(&audio_data)
    })
    .await
    .map_err(|e| {
        error!(error = %e, "Audio embedding task panicked");
        AppError::Internal(e.to_string())
    })?
    .map_err(|e| {
        error!(error = %e, "Audio embedding failed");
        AppError::from(e)
    })?;

    debug!(embedding_dim = embedding.data().len(), "Audio embedding generated");

    Ok(MsgPack(AudioEmbedResponse {
        embedding: embedding.into_data(),
        duration_s,
    }))
}

/// Fallback handlers when inference feature is disabled
#[cfg(not(feature = "inference"))]
pub async fn text_embed(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Inference feature not enabled".to_string(),
    ))
}

#[cfg(not(feature = "inference"))]
pub async fn audio_embed(State(_state): State<AppState>) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Inference feature not enabled".to_string(),
    ))
}
