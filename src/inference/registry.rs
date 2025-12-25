//! Known models registry for the sidecar.
//!
//! Contains a curated list of tested CLAP models that are known to work
//! with the sidecar, along with metadata for display and download.

use serde::Serialize;

/// Information about a known/tested model
/// Note: Only Serialize is derived as this is a static registry used for API output.
/// The static slice fields cannot implement Deserialize.
#[derive(Debug, Clone, Serialize)]
pub struct KnownModel {
    /// Model ID (HuggingFace format: owner/model-name)
    pub model_id: &'static str,
    /// Display name
    pub name: &'static str,
    /// Description
    pub description: &'static str,
    /// Estimated total size in MB (text + audio models)
    pub estimated_size_mb: u32,
    /// Required files
    pub files: &'static [&'static str],
    /// Whether this model requires a tokenizer
    pub needs_tokenizer: bool,
}

/// Registry of known/tested CLAP models
pub const KNOWN_MODELS: &[KnownModel] = &[
    KnownModel {
        model_id: "Xenova/clap-htsat-unfused",
        name: "CLAP HTSAT (Unfused)",
        description: "General-purpose CLAP model from Xenova. Good balance of speed and quality for music similarity.",
        estimated_size_mb: 300,
        files: &["text_model.onnx", "audio_model.onnx", "tokenizer.json"],
        needs_tokenizer: true,
    },
    KnownModel {
        model_id: "laion/larger_clap_music",
        name: "CLAP Music (Large)",
        description: "Larger CLAP model optimized for music. Better quality but slower inference.",
        estimated_size_mb: 600,
        files: &["text_model.onnx", "audio_model.onnx"],
        needs_tokenizer: false,
    },
    KnownModel {
        model_id: "laion/clap-htsat-unfused",
        name: "CLAP HTSAT (LAION)",
        description: "Original LAION CLAP model. Well-tested baseline model.",
        estimated_size_mb: 300,
        files: &["text_model.onnx", "audio_model.onnx"],
        needs_tokenizer: false,
    },
];

impl KnownModel {
    /// Get a known model by ID
    pub fn get(model_id: &str) -> Option<&'static KnownModel> {
        KNOWN_MODELS.iter().find(|m| m.model_id == model_id)
    }

    /// Check if a model ID is in the known models list
    pub fn is_known(model_id: &str) -> bool {
        KNOWN_MODELS.iter().any(|m| m.model_id == model_id)
    }

    /// Get estimated size in bytes
    pub fn estimated_size_bytes(&self) -> u64 {
        (self.estimated_size_mb as u64) * 1024 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_known_model() {
        let model = KnownModel::get("Xenova/clap-htsat-unfused");
        assert!(model.is_some());
        assert_eq!(model.unwrap().name, "CLAP HTSAT (Unfused)");
    }

    #[test]
    fn test_unknown_model() {
        assert!(KnownModel::get("unknown/model").is_none());
        assert!(!KnownModel::is_known("unknown/model"));
    }

    #[test]
    fn test_size_calculation() {
        let model = KnownModel::get("Xenova/clap-htsat-unfused").unwrap();
        assert_eq!(model.estimated_size_bytes(), 300 * 1024 * 1024);
    }
}
