//! Music Assistant Insight Sidecar
//!
//! A lightweight, high-performance inference sidecar for Music Assistant that
//! provides audio and text embeddings using CLAP (Contrastive Language-Audio
//! Pretraining) models.

pub mod config;
pub mod error;
#[cfg(feature = "inference")]
pub mod inference;
pub mod mood;
pub mod server;
pub mod storage;
pub mod taste;
pub mod types;
#[cfg(feature = "watcher")]
pub mod watcher;

pub use config::{AppConfig, StorageMode};
pub use error::{AppError, Result};

#[cfg(feature = "inference")]
pub use inference::{ClapModel, Embedding};

pub use mood::{MoodClassification, MoodScore, MoodTier};
