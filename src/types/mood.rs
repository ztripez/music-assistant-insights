//! API types for mood classification operations.
//!
//! This module contains request/response types for classifying track moods
//! and listing available mood definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::mood::{MoodClassification, MoodTier};

/// Request to classify mood of audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodClassifyRequest {
    /// Pre-computed audio embedding (512-dimensional)
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,

    /// Track ID to lookup embedding from storage
    #[serde(default)]
    pub track_id: Option<String>,

    /// Which mood tiers to include in results
    #[serde(default = "default_mood_tiers")]
    pub tiers: Vec<MoodTier>,

    /// Number of top moods to return per tier
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Include valence-arousal coordinates
    #[serde(default = "default_true_bool")]
    pub include_va: bool,
}

fn default_mood_tiers() -> Vec<MoodTier> {
    vec![MoodTier::Primary, MoodTier::Refined]
}

fn default_top_k() -> usize {
    3
}

fn default_true_bool() -> bool {
    true
}

/// Response from mood classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodClassifyResponse {
    /// Classification results
    #[serde(flatten)]
    pub classification: MoodClassification,
}

/// Request to list available moods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListMoodsRequest {
    /// Filter by tier (optional)
    #[serde(default)]
    pub tier: Option<MoodTier>,
}

/// Info about a single mood definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodInfo {
    /// Mood identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Classification tier
    pub tier: MoodTier,
    /// Valence hint (-1 to 1)
    pub valence_hint: f32,
    /// Arousal hint (-1 to 1)
    pub arousal_hint: f32,
}

/// Response listing available moods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListMoodsResponse {
    /// Available moods
    pub moods: Vec<MoodInfo>,
    /// Count by tier
    pub counts: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mood_classify_request_defaults() {
        // Test that defaults work correctly
        let json = r#"{"embedding": [0.1, 0.2, 0.3]}"#;
        let req: MoodClassifyRequest = serde_json::from_str(json).unwrap();

        assert!(req.embedding.is_some());
        assert!(req.track_id.is_none());
        // Check defaults
        assert_eq!(req.tiers.len(), 2); // primary + refined
        assert_eq!(req.top_k, 3);
        assert!(req.include_va);
    }

    #[test]
    fn test_mood_classify_request_with_track_id() {
        let json = r#"{"track_id": "track_123", "tiers": ["primary"], "top_k": 5}"#;
        let req: MoodClassifyRequest = serde_json::from_str(json).unwrap();

        assert!(req.embedding.is_none());
        assert_eq!(req.track_id, Some("track_123".to_string()));
        assert_eq!(req.tiers.len(), 1);
        assert_eq!(req.top_k, 5);
    }

    #[test]
    fn test_mood_info_serialization() {
        let info = MoodInfo {
            id: "happy".to_string(),
            name: "Happy".to_string(),
            tier: MoodTier::Refined,
            valence_hint: 0.8,
            arousal_hint: 0.6,
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("happy"));
        assert!(json.contains("refined"));
        assert!(json.contains("0.8"));

        let decoded: MoodInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "happy");
        assert_eq!(decoded.tier, MoodTier::Refined);
    }

    #[test]
    fn test_list_moods_response() {
        let mut counts = HashMap::new();
        counts.insert("primary".to_string(), 4);
        counts.insert("refined".to_string(), 14);

        let resp = ListMoodsResponse {
            moods: vec![MoodInfo {
                id: "energetic".to_string(),
                name: "Energetic".to_string(),
                tier: MoodTier::Primary,
                valence_hint: 0.5,
                arousal_hint: 0.9,
            }],
            counts,
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("energetic"));
        assert!(json.contains("primary"));

        let decoded: ListMoodsResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.moods.len(), 1);
        assert_eq!(decoded.counts.get("primary"), Some(&4));
    }
}
