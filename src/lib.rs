//! Music Assistant Insight Sidecar
//!
//! A lightweight, high-performance inference sidecar for Music Assistant that
//! provides audio and text embeddings using CLAP (Contrastive Language-Audio
//! Pretraining) models.

pub mod config;
pub mod error;
#[cfg(feature = "inference")]
pub mod inference;
pub mod server;
pub mod storage;
pub mod types;

pub use config::{AppConfig, StorageMode};
pub use error::{AppError, Result};

#[cfg(feature = "inference")]
pub use inference::{ClapModel, Embedding};
