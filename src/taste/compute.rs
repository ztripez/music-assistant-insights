//! Taste vector computation logic.

use std::collections::HashMap;
use std::time::SystemTime;

use crate::types::taste::{AnalyzedInteraction, AnalyzeInteractionsResponse, InteractionSummary};
use crate::types::{SignalType, TasteVector, UserInteraction};

/// Errors that can occur during taste vector computation
#[derive(Debug, thiserror::Error)]
pub enum TasteVectorError {
    #[error("Insufficient data: need at least {0} interactions")]
    InsufficientData(usize),

    #[error("Invalid embedding dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    #[error("No positive signals found")]
    NoPositiveSignals,

    #[error("Missing embedding for track: {0}")]
    MissingEmbedding(String),
}

/// Taste vector computer with configurable parameters
pub struct TasteVectorComputer {
    /// Time decay rate (default: 0.95^days)
    pub time_decay_rate: f32,
    /// Expected embedding dimension
    pub embedding_dim: usize,
}

impl Default for TasteVectorComputer {
    fn default() -> Self {
        Self {
            time_decay_rate: 0.95,
            embedding_dim: 512,
        }
    }
}

impl TasteVectorComputer {
    /// Create a new taste vector computer with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new taste vector computer with custom parameters
    pub fn with_params(time_decay_rate: f32, embedding_dim: usize) -> Self {
        Self {
            time_decay_rate,
            embedding_dim,
        }
    }

    /// Compute a taste vector from user interactions
    ///
    /// # Arguments
    /// * `interactions` - List of user interactions with tracks
    /// * `track_embeddings` - Map of track IDs to embeddings (512-dim vectors)
    /// * `cutoff_days` - How many days of history to consider
    ///
    /// # Returns
    /// A TasteVector with the computed embedding, track count, and confidence
    pub fn compute_taste_vector(
        &self,
        interactions: &[UserInteraction],
        track_embeddings: &HashMap<String, Vec<f32>>,
        cutoff_days: u32,
    ) -> Result<TasteVector, TasteVectorError> {
        if interactions.is_empty() {
            return Err(TasteVectorError::InsufficientData(1));
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Filter interactions by cutoff
        let filtered: Vec<_> = interactions
            .iter()
            .filter(|interaction| {
                let age_days = (now - interaction.timestamp) as f32 / 86400.0;
                age_days <= cutoff_days as f32
            })
            .collect();

        if filtered.is_empty() {
            return Err(TasteVectorError::InsufficientData(1));
        }

        // Initialize accumulators
        let mut weighted_sum = vec![0.0f32; self.embedding_dim];
        let mut total_positive_weight = 0.0f32;
        let mut negative_adjustments = vec![0.0f32; self.embedding_dim];
        let mut total_negative_weight = 0.0f32;
        let mut track_count = 0u32;

        // Compute weighted sum
        for interaction in &filtered {
            let embedding = track_embeddings
                .get(&interaction.track_id)
                .ok_or_else(|| TasteVectorError::MissingEmbedding(interaction.track_id.clone()))?;

            // Validate embedding dimension
            if embedding.len() != self.embedding_dim {
                return Err(TasteVectorError::InvalidDimension {
                    expected: self.embedding_dim,
                    got: embedding.len(),
                });
            }

            let weight = self.calculate_weight(interaction, now);

            if weight > 0.0 {
                // Positive signal: add to taste vector
                for (i, &val) in embedding.iter().enumerate() {
                    weighted_sum[i] += val * weight;
                }
                total_positive_weight += weight;
                track_count += 1;
            } else {
                // Negative signal: record for adjustment
                for (i, &val) in embedding.iter().enumerate() {
                    negative_adjustments[i] += val * weight.abs();
                }
                total_negative_weight += weight.abs();
            }
        }

        if total_positive_weight == 0.0 {
            return Err(TasteVectorError::NoPositiveSignals);
        }

        // Compute average of positive signals
        let mut taste = weighted_sum
            .iter()
            .map(|&x| x / total_positive_weight)
            .collect::<Vec<_>>();

        // Apply negative adjustments (scaled down to 30% impact)
        if total_negative_weight > 0.0 {
            let negative_scale = 0.3; // Negatives have 30% the impact of positives
            for (i, taste_val) in taste.iter_mut().enumerate() {
                let negative_avg = negative_adjustments[i] / total_negative_weight;
                *taste_val -= negative_avg * negative_scale;
            }
        }

        // Normalize to unit vector
        let norm = taste.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            taste.iter_mut().for_each(|x| *x /= norm);
        } else {
            return Err(TasteVectorError::NoPositiveSignals);
        }

        // Calculate confidence based on track count
        let confidence = self.calculate_confidence(track_count);

        Ok(TasteVector {
            embedding: taste,
            track_count,
            confidence,
            total_weight: total_positive_weight,
        })
    }

    /// Calculate weight for an interaction
    ///
    /// Weight = base_score × time_decay × completion_bonus
    fn calculate_weight(&self, interaction: &UserInteraction, now: i64) -> f32 {
        // Base score from signal type
        let base = self.signal_weight(&interaction.signal_type);

        // Time decay: time_decay_rate ^ days_ago
        let age_days = (now - interaction.timestamp) as f32 / 86400.0;
        let decay = self.time_decay_rate.powf(age_days.max(0.0));

        // Play completion bonus (for partial plays)
        let completion = if interaction.duration > 0.0 {
            interaction.seconds_played / interaction.duration
        } else {
            0.0
        };

        let completion_bonus = if matches!(
            interaction.signal_type,
            SignalType::PartialPlay | SignalType::FullPlay
        ) {
            if completion > 0.5 {
                completion * 0.5 // Up to +0.5 bonus for high completion
            } else {
                0.0
            }
        } else {
            0.0
        };

        (base + completion_bonus) * decay
    }

    /// Get base weight for a signal type
    fn signal_weight(&self, signal_type: &SignalType) -> f32 {
        match signal_type {
            SignalType::FullPlay => 1.0,
            SignalType::PartialPlay => 0.7,
            SignalType::Skip => -0.3,
            SignalType::Repeat => 1.5,
            SignalType::Favorite => 2.0,
            SignalType::Dislike => -1.0,
            SignalType::Save => 1.5,
        }
    }

    /// Calculate confidence score based on track count
    ///
    /// Uses sigmoid-like function: x / (x + 20)
    /// - 5 tracks: ~0.20 confidence
    /// - 10 tracks: ~0.33
    /// - 20 tracks: ~0.50
    /// - 50 tracks: ~0.71
    /// - 100 tracks: ~0.83
    fn calculate_confidence(&self, track_count: u32) -> f32 {
        let x = track_count as f32;
        (x / (x + 20.0)).min(1.0)
    }

    /// Analyze interactions without computing embeddings
    ///
    /// Returns detailed weight breakdown for each interaction,
    /// useful for debugging and understanding how the taste profile is computed.
    ///
    /// # Arguments
    /// * `interactions` - List of user interactions to analyze
    /// * `cutoff_days` - How many days of history to consider
    ///
    /// # Returns
    /// Analysis response with per-interaction weights and summary statistics
    pub fn analyze_interactions(
        &self,
        interactions: &[UserInteraction],
        cutoff_days: u32,
    ) -> AnalyzeInteractionsResponse {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let mut analyzed = Vec::with_capacity(interactions.len());
        let mut signal_type_counts: HashMap<String, u32> = HashMap::new();
        let mut total_positive_weight = 0.0f32;
        let mut total_negative_weight = 0.0f32;
        let mut positive_count = 0u32;
        let mut negative_count = 0u32;
        let mut within_cutoff_count = 0u32;
        let mut total_age_days = 0.0f32;

        for interaction in interactions {
            let age_days = (now - interaction.timestamp) as f32 / 86400.0;
            let within_cutoff = age_days <= cutoff_days as f32;

            // Base weight from signal type
            let base_weight = self.signal_weight(&interaction.signal_type);

            // Time decay
            let time_decay = self.time_decay_rate.powf(age_days.max(0.0));

            // Completion bonus
            let completion = if interaction.duration > 0.0 {
                interaction.seconds_played / interaction.duration
            } else {
                0.0
            };

            let completion_bonus = if matches!(
                interaction.signal_type,
                SignalType::PartialPlay | SignalType::FullPlay
            ) && completion > 0.5
            {
                completion * 0.5
            } else {
                0.0
            };

            // Final weight
            let final_weight = (base_weight + completion_bonus) * time_decay;
            let is_positive = final_weight > 0.0;

            // Update counters
            let signal_type_str = format!("{:?}", interaction.signal_type).to_lowercase();
            *signal_type_counts.entry(signal_type_str.clone()).or_insert(0) += 1;
            total_age_days += age_days;

            if within_cutoff {
                within_cutoff_count += 1;
                if is_positive {
                    positive_count += 1;
                    total_positive_weight += final_weight;
                } else {
                    negative_count += 1;
                    total_negative_weight += final_weight.abs();
                }
            }

            analyzed.push(AnalyzedInteraction {
                track_id: interaction.track_id.clone(),
                signal_type: signal_type_str,
                age_days,
                base_weight,
                time_decay,
                completion_bonus,
                final_weight,
                is_positive,
                within_cutoff,
                timestamp: interaction.timestamp,
            });
        }

        // Sort by timestamp (most recent first)
        analyzed.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        let total_count = interactions.len() as u32;
        let average_age_days = if total_count > 0 {
            total_age_days / total_count as f32
        } else {
            0.0
        };

        AnalyzeInteractionsResponse {
            interactions: analyzed,
            summary: InteractionSummary {
                total_count,
                within_cutoff_count,
                positive_count,
                negative_count,
                total_positive_weight,
                total_negative_weight,
                average_age_days,
                signal_type_counts,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SignalType;

    fn create_test_embedding(dim: usize, value: f32) -> Vec<f32> {
        vec![value; dim]
    }

    fn normalize_embedding(embedding: &[f32]) -> Vec<f32> {
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        embedding.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_compute_simple_taste_vector() {
        let computer = TasteVectorComputer::new();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let interactions = vec![
            UserInteraction {
                track_id: "track_1".to_string(),
                timestamp: now - 3600, // 1 hour ago
                signal_type: SignalType::FullPlay,
                seconds_played: 180.0,
                duration: 180.0,
            },
            UserInteraction {
                track_id: "track_2".to_string(),
                timestamp: now - 7200, // 2 hours ago
                signal_type: SignalType::FullPlay,
                seconds_played: 200.0,
                duration: 200.0,
            },
        ];

        let mut embeddings = HashMap::new();
        embeddings.insert("track_1".to_string(), normalize_embedding(&create_test_embedding(512, 1.0)));
        embeddings.insert("track_2".to_string(), normalize_embedding(&create_test_embedding(512, 0.5)));

        let result = computer.compute_taste_vector(&interactions, &embeddings, 30);
        assert!(result.is_ok(), "Result error: {:?}", result.err());

        let taste = result.unwrap();
        assert_eq!(taste.embedding.len(), 512);
        assert_eq!(taste.track_count, 2);
        assert!(taste.confidence > 0.0 && taste.confidence < 1.0);

        // Check normalization
        let norm: f32 = taste.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_negative_signals() {
        let computer = TasteVectorComputer::new();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let interactions = vec![
            UserInteraction {
                track_id: "track_1".to_string(),
                timestamp: now - 3600,
                signal_type: SignalType::FullPlay,
                seconds_played: 180.0,
                duration: 180.0,
            },
            UserInteraction {
                track_id: "track_2".to_string(),
                timestamp: now - 7200,
                signal_type: SignalType::Skip,
                seconds_played: 10.0,
                duration: 200.0,
            },
        ];

        let mut embeddings = HashMap::new();
        embeddings.insert("track_1".to_string(), normalize_embedding(&create_test_embedding(512, 1.0)));
        embeddings.insert("track_2".to_string(), normalize_embedding(&create_test_embedding(512, 0.5)));

        let result = computer.compute_taste_vector(&interactions, &embeddings, 30);
        assert!(result.is_ok(), "Result error: {:?}", result.err());

        let taste = result.unwrap();
        assert_eq!(taste.track_count, 1); // Only positive signals count
    }

    #[test]
    fn test_insufficient_data() {
        let computer = TasteVectorComputer::new();
        let embeddings = HashMap::new();

        let result = computer.compute_taste_vector(&[], &embeddings, 30);
        assert!(matches!(result, Err(TasteVectorError::InsufficientData(_))));
    }

    #[test]
    fn test_confidence_calculation() {
        let computer = TasteVectorComputer::new();

        assert!((computer.calculate_confidence(5) - 0.2).abs() < 0.01);
        assert!((computer.calculate_confidence(20) - 0.5).abs() < 0.01);
        assert!((computer.calculate_confidence(50) - 0.714).abs() < 0.01);
        assert!((computer.calculate_confidence(100) - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_signal_weights() {
        let computer = TasteVectorComputer::new();

        assert_eq!(computer.signal_weight(&SignalType::FullPlay), 1.0);
        assert_eq!(computer.signal_weight(&SignalType::PartialPlay), 0.7);
        assert_eq!(computer.signal_weight(&SignalType::Skip), -0.3);
        assert_eq!(computer.signal_weight(&SignalType::Repeat), 1.5);
        assert_eq!(computer.signal_weight(&SignalType::Favorite), 2.0);
        assert_eq!(computer.signal_weight(&SignalType::Dislike), -1.0);
        assert_eq!(computer.signal_weight(&SignalType::Save), 1.5);
    }
}
