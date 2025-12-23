//! ML inference module for CLAP model embeddings.
//!
//! This module provides text and audio embedding generation using CLAP
//! (Contrastive Language-Audio Pretraining) models via ONNX Runtime.

mod audio;
mod download;
mod model;
mod text;

pub use audio::{AudioData, AudioFormat, AudioProcessor};
pub use download::{download_model, ModelPaths};
pub use model::{ClapModel, Device};
pub use text::format_track_metadata;

use crate::error::AppError;

/// 512-dimensional embedding vector (CLAP output dimension)
pub const EMBEDDING_DIM: usize = 512;

/// Embedding vector type
#[derive(Debug, Clone)]
pub struct Embedding {
    data: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding from a vector of floats
    pub fn new(data: Vec<f32>) -> Result<Self, AppError> {
        if data.len() != EMBEDDING_DIM {
            return Err(AppError::Internal(format!(
                "Invalid embedding dimension: expected {}, got {}",
                EMBEDDING_DIM,
                data.len()
            )));
        }
        Ok(Self { data })
    }

    /// Create an embedding from raw data without validation (internal use)
    pub(crate) fn from_raw(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Get the raw embedding data
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Consume and return the raw embedding data
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let dot: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// L2 normalize the embedding in place
    pub fn normalize(&mut self) {
        let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut self.data {
                *x /= norm;
            }
        }
    }

    /// Return a normalized copy of this embedding
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }
}

impl serde::Serialize for Embedding {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as raw bytes for efficiency
        let bytes: Vec<u8> = self
            .data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> serde::Deserialize<'de> for Embedding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        if bytes.len() != EMBEDDING_DIM * 4 {
            return Err(serde::de::Error::custom(format!(
                "Invalid embedding byte length: expected {}, got {}",
                EMBEDDING_DIM * 4,
                bytes.len()
            )));
        }

        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Self { data })
    }
}

/// Inference error types
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Model download failed: {0}")]
    DownloadFailed(String),

    #[error("Invalid audio format: {0}")]
    InvalidAudioFormat(String),

    #[error("Audio processing error: {0}")]
    AudioProcessing(String),

    #[error("ONNX runtime error: {0}")]
    Onnx(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<InferenceError> for AppError {
    fn from(err: InferenceError) -> Self {
        match err {
            InferenceError::InvalidAudioFormat(msg) => AppError::BadRequest(msg),
            _ => AppError::Internal(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_cosine_similarity() {
        let mut data1 = vec![0.0; EMBEDDING_DIM];
        data1[0] = 1.0;
        let emb1 = Embedding::from_raw(data1);

        let mut data2 = vec![0.0; EMBEDDING_DIM];
        data2[0] = 1.0;
        let emb2 = Embedding::from_raw(data2);

        let similarity = emb1.cosine_similarity(&emb2);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_orthogonal() {
        let mut data1 = vec![0.0; EMBEDDING_DIM];
        data1[0] = 1.0;
        let emb1 = Embedding::from_raw(data1);

        let mut data2 = vec![0.0; EMBEDDING_DIM];
        data2[1] = 1.0;
        let emb2 = Embedding::from_raw(data2);

        let similarity = emb1.cosine_similarity(&emb2);
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn test_embedding_normalize() {
        let data = vec![3.0, 4.0]
            .into_iter()
            .chain(std::iter::repeat(0.0).take(EMBEDDING_DIM - 2))
            .collect();
        let mut emb = Embedding::from_raw(data);
        emb.normalize();

        let norm: f32 = emb.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
