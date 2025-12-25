//! Mood prompt definitions for zero-shot classification.
//!
//! These prompts are used with CLAP text encoder to create mood embeddings.
//! The quality of prompts significantly impacts classification accuracy.

use serde::{Deserialize, Serialize};

/// Mood tier for hierarchical classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoodTier {
    /// Primary moods based on valence-arousal quadrants
    Primary,
    /// More specific refined moods
    Refined,
    /// Context/activity-based moods
    Contextual,
}

impl std::fmt::Display for MoodTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoodTier::Primary => write!(f, "primary"),
            MoodTier::Refined => write!(f, "refined"),
            MoodTier::Contextual => write!(f, "contextual"),
        }
    }
}

/// A mood definition with its classification prompt
#[derive(Debug, Clone)]
pub struct MoodPrompt {
    /// Mood identifier (lowercase, `snake_case`)
    pub id: &'static str,
    /// Display name
    pub name: &'static str,
    /// Classification tier
    pub tier: MoodTier,
    /// Detailed prompt for CLAP text encoder
    pub prompt: &'static str,
    /// Valence hint (-1.0 to 1.0, negative to positive)
    pub valence_hint: f32,
    /// Arousal hint (-1.0 to 1.0, low to high)
    pub arousal_hint: f32,
}

// ============================================================================
// Primary Moods (Valence-Arousal Quadrants)
// ============================================================================

/// High arousal, positive valence
pub const MOOD_ENERGETIC: MoodPrompt = MoodPrompt {
    id: "energetic",
    name: "Energetic",
    tier: MoodTier::Primary,
    prompt: "energetic exciting upbeat powerful music with fast tempo, driving rhythms, \
             strong beats, and dynamic high-energy instrumentation",
    valence_hint: 0.7,
    arousal_hint: 0.8,
};

/// High arousal, negative valence
pub const MOOD_AGGRESSIVE: MoodPrompt = MoodPrompt {
    id: "aggressive",
    name: "Aggressive",
    tier: MoodTier::Primary,
    prompt: "aggressive intense angry heavy music with distorted guitars, pounding drums, \
             harsh vocals, and dark aggressive energy",
    valence_hint: -0.6,
    arousal_hint: 0.9,
};

/// Low arousal, positive valence
pub const MOOD_PEACEFUL: MoodPrompt = MoodPrompt {
    id: "peaceful",
    name: "Peaceful",
    tier: MoodTier::Primary,
    prompt: "peaceful calm serene relaxing ambient music with slow gentle tempo, \
             soft pads, warm tones, and soothing tranquil atmosphere",
    valence_hint: 0.6,
    arousal_hint: -0.7,
};

/// Low arousal, negative valence
pub const MOOD_MELANCHOLIC: MoodPrompt = MoodPrompt {
    id: "melancholic",
    name: "Melancholic",
    tier: MoodTier::Primary,
    prompt: "melancholic sad sorrowful somber music with slow tempo, minor key, \
             emotional strings, and a deeply introspective melancholy atmosphere",
    valence_hint: -0.7,
    arousal_hint: -0.5,
};

/// All primary moods
pub const PRIMARY_MOODS: &[MoodPrompt] = &[
    MOOD_ENERGETIC,
    MOOD_AGGRESSIVE,
    MOOD_PEACEFUL,
    MOOD_MELANCHOLIC,
];

// ============================================================================
// Refined Moods (More specific emotional states)
// ============================================================================

pub const MOOD_HAPPY: MoodPrompt = MoodPrompt {
    id: "happy",
    name: "Happy",
    tier: MoodTier::Refined,
    prompt: "happy joyful cheerful bright music with uplifting melody, major key, \
             bouncy rhythm, and feel-good positive vibes",
    valence_hint: 0.9,
    arousal_hint: 0.5,
};

pub const MOOD_EUPHORIC: MoodPrompt = MoodPrompt {
    id: "euphoric",
    name: "Euphoric",
    tier: MoodTier::Refined,
    prompt: "euphoric ecstatic blissful triumphant music with soaring melodies, \
             build-ups, epic drops, and overwhelming joy and elation",
    valence_hint: 0.95,
    arousal_hint: 0.85,
};

pub const MOOD_POWERFUL: MoodPrompt = MoodPrompt {
    id: "powerful",
    name: "Powerful",
    tier: MoodTier::Refined,
    prompt: "powerful epic majestic anthemic music with strong dramatic orchestration, \
             bold brass, thundering percussion, and heroic grandeur",
    valence_hint: 0.4,
    arousal_hint: 0.8,
};

pub const MOOD_INTENSE: MoodPrompt = MoodPrompt {
    id: "intense",
    name: "Intense",
    tier: MoodTier::Refined,
    prompt: "intense suspenseful urgent thrilling music with building tension, \
             driving pulse, dramatic dynamics, and gripping emotional intensity",
    valence_hint: 0.0,
    arousal_hint: 0.9,
};

pub const MOOD_ANGRY: MoodPrompt = MoodPrompt {
    id: "angry",
    name: "Angry",
    tier: MoodTier::Refined,
    prompt: "angry furious raging hostile music with aggressive distortion, \
             pounding beats, shouted vocals, and raw violent anger",
    valence_hint: -0.8,
    arousal_hint: 0.95,
};

pub const MOOD_DARK: MoodPrompt = MoodPrompt {
    id: "dark",
    name: "Dark",
    tier: MoodTier::Refined,
    prompt: "dark ominous sinister brooding music with minor key, deep bass, \
             eerie textures, and mysterious shadowy atmosphere",
    valence_hint: -0.6,
    arousal_hint: 0.3,
};

pub const MOOD_TENSE: MoodPrompt = MoodPrompt {
    id: "tense",
    name: "Tense",
    tier: MoodTier::Refined,
    prompt: "tense anxious nervous unsettling music with dissonant harmonies, \
             irregular rhythms, building suspense, and uneasy atmosphere",
    valence_hint: -0.5,
    arousal_hint: 0.6,
};

pub const MOOD_CALM: MoodPrompt = MoodPrompt {
    id: "calm",
    name: "Calm",
    tier: MoodTier::Refined,
    prompt: "calm quiet gentle soothing music with soft dynamics, \
             slow tempo, minimal arrangement, and restful stillness",
    valence_hint: 0.4,
    arousal_hint: -0.8,
};

pub const MOOD_DREAMY: MoodPrompt = MoodPrompt {
    id: "dreamy",
    name: "Dreamy",
    tier: MoodTier::Refined,
    prompt: "dreamy ethereal floating ambient music with reverb-drenched textures, \
             hazy synths, soft vocals, and otherworldly dreamscape atmosphere",
    valence_hint: 0.3,
    arousal_hint: -0.4,
};

pub const MOOD_ROMANTIC: MoodPrompt = MoodPrompt {
    id: "romantic",
    name: "Romantic",
    tier: MoodTier::Refined,
    prompt: "romantic loving tender passionate music with lush strings, \
             warm harmonies, intimate vocals, and heartfelt emotional expression",
    valence_hint: 0.7,
    arousal_hint: 0.1,
};

pub const MOOD_TENDER: MoodPrompt = MoodPrompt {
    id: "tender",
    name: "Tender",
    tier: MoodTier::Refined,
    prompt: "tender gentle delicate soft music with acoustic instruments, \
             quiet dynamics, sweet melodies, and touching emotional sensitivity",
    valence_hint: 0.5,
    arousal_hint: -0.3,
};

pub const MOOD_SAD: MoodPrompt = MoodPrompt {
    id: "sad",
    name: "Sad",
    tier: MoodTier::Refined,
    prompt: "sad sorrowful mournful heartbreaking music with minor key, \
             slow tempo, plaintive melodies, and deep emotional sadness",
    valence_hint: -0.8,
    arousal_hint: -0.3,
};

pub const MOOD_NOSTALGIC: MoodPrompt = MoodPrompt {
    id: "nostalgic",
    name: "Nostalgic",
    tier: MoodTier::Refined,
    prompt: "nostalgic wistful bittersweet reminiscent music evoking memories, \
             vintage sounds, emotional longing, and sentimental reflection",
    valence_hint: -0.2,
    arousal_hint: -0.2,
};

pub const MOOD_HOPEFUL: MoodPrompt = MoodPrompt {
    id: "hopeful",
    name: "Hopeful",
    tier: MoodTier::Refined,
    prompt: "hopeful optimistic uplifting inspirational music with ascending melodies, \
             major key progressions, and a sense of promise and possibility",
    valence_hint: 0.7,
    arousal_hint: 0.3,
};

/// All refined moods
pub const REFINED_MOODS: &[MoodPrompt] = &[
    MOOD_HAPPY,
    MOOD_EUPHORIC,
    MOOD_POWERFUL,
    MOOD_INTENSE,
    MOOD_ANGRY,
    MOOD_DARK,
    MOOD_TENSE,
    MOOD_CALM,
    MOOD_DREAMY,
    MOOD_ROMANTIC,
    MOOD_TENDER,
    MOOD_SAD,
    MOOD_NOSTALGIC,
    MOOD_HOPEFUL,
];

// ============================================================================
// Contextual Moods (Activity/situation-based)
// ============================================================================

pub const MOOD_PARTY: MoodPrompt = MoodPrompt {
    id: "party",
    name: "Party",
    tier: MoodTier::Contextual,
    prompt: "party dance club music with heavy beats, catchy hooks, \
             electronic production, and high-energy celebration vibes",
    valence_hint: 0.8,
    arousal_hint: 0.9,
};

pub const MOOD_WORKOUT: MoodPrompt = MoodPrompt {
    id: "workout",
    name: "Workout",
    tier: MoodTier::Contextual,
    prompt: "workout exercise gym music with pumping beats, motivational energy, \
             driving tempo, and adrenaline-fueled intensity for training",
    valence_hint: 0.5,
    arousal_hint: 0.95,
};

pub const MOOD_FOCUS: MoodPrompt = MoodPrompt {
    id: "focus",
    name: "Focus",
    tier: MoodTier::Contextual,
    prompt: "focus concentration study music with minimal distractions, \
             steady ambient textures, lo-fi beats, and productivity atmosphere",
    valence_hint: 0.2,
    arousal_hint: -0.3,
};

pub const MOOD_SLEEP: MoodPrompt = MoodPrompt {
    id: "sleep",
    name: "Sleep",
    tier: MoodTier::Contextual,
    prompt: "sleep relaxation bedtime music with very slow tempo, \
             quiet ambient drones, gentle nature sounds, and deeply calming lullaby quality",
    valence_hint: 0.3,
    arousal_hint: -0.95,
};

pub const MOOD_CHILL: MoodPrompt = MoodPrompt {
    id: "chill",
    name: "Chill",
    tier: MoodTier::Contextual,
    prompt: "chill laid-back relaxed easy-going music with smooth grooves, \
             mellow vibes, cool jazz influences, and effortless casual atmosphere",
    valence_hint: 0.5,
    arousal_hint: -0.4,
};

pub const MOOD_ROADTRIP: MoodPrompt = MoodPrompt {
    id: "roadtrip",
    name: "Road Trip",
    tier: MoodTier::Contextual,
    prompt: "road trip driving music with feel-good energy, sing-along melodies, \
             open road vibes, adventure spirit, and windows-down summer feeling",
    valence_hint: 0.7,
    arousal_hint: 0.5,
};

pub const MOOD_ROMANCE: MoodPrompt = MoodPrompt {
    id: "romance",
    name: "Romance",
    tier: MoodTier::Contextual,
    prompt: "romance date night love music with sensual smooth vocals, \
             intimate atmosphere, slow jams, and seductive romantic mood",
    valence_hint: 0.6,
    arousal_hint: 0.0,
};

pub const MOOD_MEDITATION: MoodPrompt = MoodPrompt {
    id: "meditation",
    name: "Meditation",
    tier: MoodTier::Contextual,
    prompt: "meditation mindfulness zen music with peaceful ambient drones, \
             tibetan bowls, nature sounds, and deeply spiritual tranquility",
    valence_hint: 0.4,
    arousal_hint: -0.9,
};

/// All contextual moods
pub const CONTEXTUAL_MOODS: &[MoodPrompt] = &[
    MOOD_PARTY,
    MOOD_WORKOUT,
    MOOD_FOCUS,
    MOOD_SLEEP,
    MOOD_CHILL,
    MOOD_ROADTRIP,
    MOOD_ROMANCE,
    MOOD_MEDITATION,
];

// ============================================================================
// Utility functions
// ============================================================================

/// Get all moods for a specific tier
pub fn get_moods_by_tier(tier: MoodTier) -> &'static [MoodPrompt] {
    match tier {
        MoodTier::Primary => PRIMARY_MOODS,
        MoodTier::Refined => REFINED_MOODS,
        MoodTier::Contextual => CONTEXTUAL_MOODS,
    }
}

/// Get all moods across all tiers
pub fn get_all_moods() -> Vec<&'static MoodPrompt> {
    let mut moods =
        Vec::with_capacity(PRIMARY_MOODS.len() + REFINED_MOODS.len() + CONTEXTUAL_MOODS.len());
    moods.extend(PRIMARY_MOODS.iter());
    moods.extend(REFINED_MOODS.iter());
    moods.extend(CONTEXTUAL_MOODS.iter());
    moods
}

/// Find a mood by ID
pub fn get_mood_by_id(id: &str) -> Option<&'static MoodPrompt> {
    get_all_moods().into_iter().find(|m| m.id == id)
}

/// Prompts for valence estimation (positive vs negative)
pub const VALENCE_POSITIVE_PROMPT: &str =
    "happy positive uplifting bright cheerful joyful music with major key and optimistic feeling";
pub const VALENCE_NEGATIVE_PROMPT: &str =
    "sad negative dark somber melancholic sorrowful music with minor key and pessimistic feeling";

/// Prompts for arousal estimation (high vs low energy)
pub const AROUSAL_HIGH_PROMPT: &str =
    "high energy intense exciting fast loud powerful aggressive music with strong beats";
pub const AROUSAL_LOW_PROMPT: &str =
    "low energy calm quiet slow soft peaceful relaxing ambient music with gentle sounds";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_moods_by_tier() {
        assert_eq!(get_moods_by_tier(MoodTier::Primary).len(), 4);
        assert_eq!(get_moods_by_tier(MoodTier::Refined).len(), 14);
        assert_eq!(get_moods_by_tier(MoodTier::Contextual).len(), 8);
    }

    #[test]
    fn test_get_all_moods() {
        let all = get_all_moods();
        assert_eq!(all.len(), 4 + 14 + 8);
    }

    #[test]
    fn test_get_mood_by_id() {
        assert!(get_mood_by_id("energetic").is_some());
        assert!(get_mood_by_id("happy").is_some());
        assert!(get_mood_by_id("party").is_some());
        assert!(get_mood_by_id("nonexistent").is_none());
    }

    #[test]
    fn test_valence_arousal_hints_in_range() {
        for mood in get_all_moods() {
            assert!(
                mood.valence_hint >= -1.0 && mood.valence_hint <= 1.0,
                "Mood {} has invalid valence_hint: {}",
                mood.id,
                mood.valence_hint
            );
            assert!(
                mood.arousal_hint >= -1.0 && mood.arousal_hint <= 1.0,
                "Mood {} has invalid arousal_hint: {}",
                mood.id,
                mood.arousal_hint
            );
        }
    }
}
