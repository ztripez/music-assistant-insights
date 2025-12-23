//! Text preprocessing for CLAP model.

/// Track metadata for text embedding generation
#[derive(Debug, Clone)]
pub struct TrackMetadata {
    pub name: String,
    pub artists: Vec<String>,
    pub album: Option<String>,
    pub genres: Vec<String>,
    pub mood: Option<String>,
}

/// Format track metadata into a text string suitable for CLAP embedding
///
/// The format is designed to capture the semantic meaning of the track:
/// "Artist Name - Track Name. Album: Album Name. Genres: rock, indie. Mood: energetic"
pub fn format_track_metadata(metadata: &TrackMetadata) -> String {
    let mut parts = Vec::new();

    // Artist - Track
    let artists = if metadata.artists.is_empty() {
        "Unknown Artist".to_string()
    } else {
        metadata.artists.join(", ")
    };
    parts.push(format!("{} - {}", artists, metadata.name));

    // Album
    if let Some(album) = &metadata.album {
        if !album.is_empty() {
            parts.push(format!("Album: {}", album));
        }
    }

    // Genres
    if !metadata.genres.is_empty() {
        parts.push(format!("Genres: {}", metadata.genres.join(", ")));
    }

    // Mood
    if let Some(mood) = &metadata.mood {
        if !mood.is_empty() {
            parts.push(format!("Mood: {}", mood));
        }
    }

    parts.join(". ")
}

/// Clean and normalize text for embedding
#[allow(dead_code)]
pub fn normalize_text(text: &str) -> String {
    text.trim()
        .to_lowercase()
        // Remove excessive whitespace
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_track_metadata_full() {
        let metadata = TrackMetadata {
            name: "Bohemian Rhapsody".to_string(),
            artists: vec!["Queen".to_string()],
            album: Some("A Night at the Opera".to_string()),
            genres: vec!["Rock".to_string(), "Progressive Rock".to_string()],
            mood: Some("Epic".to_string()),
        };

        let formatted = format_track_metadata(&metadata);
        assert!(formatted.contains("Queen - Bohemian Rhapsody"));
        assert!(formatted.contains("Album: A Night at the Opera"));
        assert!(formatted.contains("Genres: Rock, Progressive Rock"));
        assert!(formatted.contains("Mood: Epic"));
    }

    #[test]
    fn test_format_track_metadata_minimal() {
        let metadata = TrackMetadata {
            name: "Track".to_string(),
            artists: vec![],
            album: None,
            genres: vec![],
            mood: None,
        };

        let formatted = format_track_metadata(&metadata);
        assert_eq!(formatted, "Unknown Artist - Track");
    }

    #[test]
    fn test_format_track_metadata_multiple_artists() {
        let metadata = TrackMetadata {
            name: "Under Pressure".to_string(),
            artists: vec!["Queen".to_string(), "David Bowie".to_string()],
            album: Some("Hot Space".to_string()),
            genres: vec!["Rock".to_string()],
            mood: None,
        };

        let formatted = format_track_metadata(&metadata);
        assert!(formatted.contains("Queen, David Bowie - Under Pressure"));
    }

    #[test]
    fn test_normalize_text() {
        let text = "  Hello   World  ";
        assert_eq!(normalize_text(text), "hello world");
    }
}
