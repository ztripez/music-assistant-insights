//! Music Assistant Insight Sidecar
//!
//! A lightweight, high-performance inference sidecar for Music Assistant that
//! provides audio and text embeddings using CLAP (Contrastive Language-Audio
//! Pretraining) models.

pub mod config;
pub mod error;
pub mod server;
pub mod types;

pub use config::AppConfig;
pub use error::{AppError, Result};
