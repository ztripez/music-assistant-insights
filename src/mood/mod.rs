//! Mood classification module using zero-shot CLAP embeddings.
//!
//! This module provides mood classification for music tracks by comparing
//! audio embeddings against pre-computed mood prompt embeddings.

pub mod prompts;

pub use prompts::{
    get_all_moods, get_mood_by_id, get_moods_by_tier, MoodPrompt, MoodTier, AROUSAL_HIGH_PROMPT,
    AROUSAL_LOW_PROMPT, CONTEXTUAL_MOODS, PRIMARY_MOODS, REFINED_MOODS, VALENCE_NEGATIVE_PROMPT,
    VALENCE_POSITIVE_PROMPT,
};

use serde::{Deserialize, Serialize};

#[cfg(feature = "inference")]
use std::collections::HashMap;

#[cfg(feature = "inference")]
use crate::inference::ClapModel;

/// Result of mood classification for a single mood
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodScore {
    /// Mood identifier
    pub mood: String,
    /// Display name
    pub name: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Mood tier
    pub tier: MoodTier,
}

/// Full mood classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodClassification {
    /// Top moods with scores
    pub moods: Vec<MoodScore>,
    /// Estimated valence (-1.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valence: Option<f32>,
    /// Estimated arousal (-1.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arousal: Option<f32>,
    /// Primary mood (top tier-1 mood)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_mood: Option<String>,
}

/// Cached mood prompt embeddings for fast classification
#[cfg(feature = "inference")]
pub struct MoodClassifier {
    /// Mood ID -> embedding
    mood_embeddings: HashMap<String, Vec<f32>>,
    /// Valence positive embedding
    valence_positive: Vec<f32>,
    /// Valence negative embedding
    valence_negative: Vec<f32>,
    /// Arousal high embedding
    arousal_high: Vec<f32>,
    /// Arousal low embedding
    arousal_low: Vec<f32>,
}

#[cfg(feature = "inference")]
impl MoodClassifier {
    /// Create a new mood classifier by pre-computing prompt embeddings
    pub fn new(model: &ClapModel) -> Self {
        let mut mood_embeddings = HashMap::new();

        // Compute embeddings for all moods
        for mood in get_all_moods() {
            if let Ok(embedding) = model.text_embedding(mood.prompt) {
                mood_embeddings.insert(mood.id.to_string(), embedding.into_data());
            }
        }

        // Compute valence/arousal embeddings
        let valence_positive = model
            .text_embedding(VALENCE_POSITIVE_PROMPT)
            .map(|e| e.into_data())
            .unwrap_or_else(|_| vec![0.0; 512]);
        let valence_negative = model
            .text_embedding(VALENCE_NEGATIVE_PROMPT)
            .map(|e| e.into_data())
            .unwrap_or_else(|_| vec![0.0; 512]);
        let arousal_high = model
            .text_embedding(AROUSAL_HIGH_PROMPT)
            .map(|e| e.into_data())
            .unwrap_or_else(|_| vec![0.0; 512]);
        let arousal_low = model
            .text_embedding(AROUSAL_LOW_PROMPT)
            .map(|e| e.into_data())
            .unwrap_or_else(|_| vec![0.0; 512]);

        Self {
            mood_embeddings,
            valence_positive,
            valence_negative,
            arousal_high,
            arousal_low,
        }
    }

    /// Classify mood from an audio embedding
    pub fn classify(
        &self,
        audio_embedding: &[f32],
        tiers: &[MoodTier],
        top_k: usize,
        include_va: bool,
    ) -> MoodClassification {
        let mut scores: Vec<MoodScore> = Vec::new();

        // Compute similarity to each requested mood
        for tier in tiers {
            let moods = get_moods_by_tier(*tier);
            let mut tier_scores: Vec<(f32, &MoodPrompt)> = moods
                .iter()
                .filter_map(|mood| {
                    self.mood_embeddings.get(mood.id).map(|emb| {
                        let sim = cosine_similarity(audio_embedding, emb);
                        (sim, mood)
                    })
                })
                .collect();

            // Sort by similarity (descending)
            tier_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Normalize scores within tier using softmax
            let tier_scores_normalized = softmax_normalize(&tier_scores);

            // Take top-k from this tier
            for (confidence, mood) in tier_scores_normalized.into_iter().take(top_k) {
                scores.push(MoodScore {
                    mood: mood.id.to_string(),
                    name: mood.name.to_string(),
                    confidence,
                    tier: mood.tier,
                });
            }
        }

        // Sort all scores by confidence
        scores.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compute valence and arousal if requested
        let (valence, arousal) = if include_va {
            (
                Some(self.compute_valence(audio_embedding)),
                Some(self.compute_arousal(audio_embedding)),
            )
        } else {
            (None, None)
        };

        // Get primary mood (top tier-1 mood)
        let primary_mood = scores
            .iter()
            .find(|s| s.tier == MoodTier::Primary)
            .map(|s| s.mood.clone());

        MoodClassification {
            moods: scores,
            valence,
            arousal,
            primary_mood,
        }
    }

    /// Compute valence (-1 to 1) from audio embedding
    fn compute_valence(&self, audio_embedding: &[f32]) -> f32 {
        let pos_sim = cosine_similarity(audio_embedding, &self.valence_positive);
        let neg_sim = cosine_similarity(audio_embedding, &self.valence_negative);
        // Map to -1..1 range
        (pos_sim - neg_sim).clamp(-1.0, 1.0)
    }

    /// Compute arousal (-1 to 1) from audio embedding
    fn compute_arousal(&self, audio_embedding: &[f32]) -> f32 {
        let high_sim = cosine_similarity(audio_embedding, &self.arousal_high);
        let low_sim = cosine_similarity(audio_embedding, &self.arousal_low);
        // Map to -1..1 range
        (high_sim - low_sim).clamp(-1.0, 1.0)
    }

    /// Get the number of loaded mood embeddings
    pub fn mood_count(&self) -> usize {
        self.mood_embeddings.len()
    }
}

/// Compute cosine similarity between two vectors
#[cfg(any(feature = "inference", test))]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Apply softmax normalization to scores
#[cfg(feature = "inference")]
fn softmax_normalize<'a>(scores: &[(f32, &'a MoodPrompt)]) -> Vec<(f32, &'a MoodPrompt)> {
    if scores.is_empty() {
        return Vec::new();
    }

    // Temperature for softmax (lower = sharper distribution)
    let temperature = 0.1;

    // Find max for numerical stability
    let max_score = scores
        .iter()
        .map(|(s, _)| *s)
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(score/temp - max/temp)
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|(s, _)| ((s - max_score) / temperature).exp())
        .collect();

    let sum: f32 = exp_scores.iter().sum();

    scores
        .iter()
        .zip(exp_scores.iter())
        .map(|((_, mood), exp)| (exp / sum, *mood))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_normalized() {
        let a = vec![0.5, 0.5, 0.5, 0.5];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        // Same direction, should be 1.0
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        // Different lengths should return 0
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_mood_tier_display() {
        assert_eq!(MoodTier::Primary.to_string(), "primary");
        assert_eq!(MoodTier::Refined.to_string(), "refined");
        assert_eq!(MoodTier::Contextual.to_string(), "contextual");
    }

    #[test]
    fn test_mood_score_serialization() {
        let score = MoodScore {
            mood: "energetic".to_string(),
            name: "Energetic".to_string(),
            confidence: 0.85,
            tier: MoodTier::Primary,
        };

        let json = serde_json::to_string(&score).unwrap();
        assert!(json.contains("energetic"));
        assert!(json.contains("0.85"));

        let decoded: MoodScore = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.mood, "energetic");
        assert_eq!(decoded.tier, MoodTier::Primary);
    }

    #[test]
    fn test_mood_classification_serialization() {
        let classification = MoodClassification {
            moods: vec![MoodScore {
                mood: "happy".to_string(),
                name: "Happy".to_string(),
                confidence: 0.9,
                tier: MoodTier::Refined,
            }],
            valence: Some(0.7),
            arousal: Some(0.5),
            primary_mood: Some("energetic".to_string()),
        };

        let json = serde_json::to_string(&classification).unwrap();
        assert!(json.contains("happy"));
        assert!(json.contains("0.7"));

        let decoded: MoodClassification = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.moods.len(), 1);
        assert_eq!(decoded.valence, Some(0.7));
    }

    #[test]
    fn test_mood_classification_optional_fields() {
        let classification = MoodClassification {
            moods: vec![],
            valence: None,
            arousal: None,
            primary_mood: None,
        };

        let json = serde_json::to_string(&classification).unwrap();
        // Optional fields should be skipped when None
        assert!(!json.contains("valence"));
        assert!(!json.contains("arousal"));
        assert!(!json.contains("primary_mood"));
    }
}
