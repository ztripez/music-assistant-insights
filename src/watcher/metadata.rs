//! Audio metadata extraction from ID3/Vorbis tags.
//!
//! Extracts artist, album, title, and genre from audio file tags.

use std::fs::File;
use std::path::Path;

use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, StandardTagKey};
use symphonia::core::probe::Hint;

use super::WatcherError;

/// Extracted metadata from an audio file
#[derive(Debug, Clone, Default)]
pub struct ExtractedMetadata {
    /// Track title
    pub title: Option<String>,
    /// Artists (may be multiple)
    pub artists: Vec<String>,
    /// Album name
    pub album: Option<String>,
    /// Genres (may be multiple)
    pub genres: Vec<String>,
    /// Track number within the album
    pub track_number: Option<u32>,
    /// Release year
    pub year: Option<i32>,
}

impl ExtractedMetadata {
    /// Extract metadata from an audio file
    pub fn from_file(path: &Path) -> Result<Self, WatcherError> {
        let file = File::open(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                WatcherError::FileNotFound(path.to_path_buf())
            } else {
                WatcherError::IoError(e)
            }
        })?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        // Probe with metadata reading enabled
        let mut probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions {
                    limit_metadata_bytes: symphonia::core::meta::Limit::Maximum(1024 * 1024),
                    limit_visual_bytes: symphonia::core::meta::Limit::None,
                },
            )
            .map_err(|e| WatcherError::MetadataError(format!("Failed to probe format: {e}")))?;

        let mut metadata = ExtractedMetadata::default();

        // Extract from format metadata
        if let Some(rev) = probed.format.metadata().current() {
            Self::extract_from_tags(&mut metadata, rev.tags().iter());
        }

        // Extract from container metadata (separate metadata log)
        if let Some(md) = probed.metadata.get() {
            if let Some(rev) = md.current() {
                Self::extract_from_tags(&mut metadata, rev.tags().iter());
            }
        }

        Ok(metadata)
    }

    /// Extract metadata from a tag iterator
    fn extract_from_tags<'a>(
        meta: &mut ExtractedMetadata,
        tags: impl Iterator<Item = &'a symphonia::core::meta::Tag>,
    ) {
        for tag in tags {
            let value = tag.value.to_string();
            if value.is_empty() {
                continue;
            }

            match tag.std_key {
                Some(StandardTagKey::TrackTitle) => {
                    if meta.title.is_none() {
                        meta.title = Some(value);
                    }
                }
                Some(StandardTagKey::Artist)
                | Some(StandardTagKey::Performer)
                | Some(StandardTagKey::AlbumArtist) => {
                    // Handle multiple artists separated by various delimiters
                    for artist in split_values(&value) {
                        if !meta.artists.contains(&artist) {
                            meta.artists.push(artist);
                        }
                    }
                }
                Some(StandardTagKey::Album) => {
                    if meta.album.is_none() {
                        meta.album = Some(value);
                    }
                }
                Some(StandardTagKey::Genre) => {
                    for genre in split_values(&value) {
                        if !meta.genres.contains(&genre) {
                            meta.genres.push(genre);
                        }
                    }
                }
                Some(StandardTagKey::TrackNumber) => {
                    if meta.track_number.is_none() {
                        // Handle "1/12" format
                        let num_str = value.split('/').next().unwrap_or(&value);
                        meta.track_number = num_str.parse().ok();
                    }
                }
                Some(StandardTagKey::Date) | Some(StandardTagKey::OriginalDate) => {
                    if meta.year.is_none() {
                        // Try to extract year from various date formats
                        meta.year = extract_year(&value);
                    }
                }
                _ => {}
            }
        }
    }

    /// Check if the metadata has the essential fields (title and at least one artist)
    pub fn is_complete(&self) -> bool {
        self.title.is_some() && !self.artists.is_empty()
    }
}

/// Split a value by common delimiters used in music tags
fn split_values(value: &str) -> Vec<String> {
    // Common separators: semicolon, forward slash (for artists), comma
    // But be careful with commas as they can appear in names
    let mut results = Vec::new();

    // First try semicolon (most common for multiple values)
    if value.contains(';') {
        for part in value.split(';') {
            let trimmed = part.trim().to_string();
            if !trimmed.is_empty() {
                results.push(trimmed);
            }
        }
        return results;
    }

    // Try forward slash with spaces around it (indicates multiple artists)
    if value.contains(" / ") {
        for part in value.split(" / ") {
            let trimmed = part.trim().to_string();
            if !trimmed.is_empty() {
                results.push(trimmed);
            }
        }
        return results;
    }

    // Otherwise, treat as single value
    results.push(value.to_string());
    results
}

/// Extract year from a date string
fn extract_year(date: &str) -> Option<i32> {
    // Try direct parse (e.g., "2023")
    if let Ok(year) = date.parse::<i32>() {
        if (1900..=2100).contains(&year) {
            return Some(year);
        }
    }

    // Try extracting 4-digit year from beginning (e.g., "2023-01-15")
    if date.len() >= 4 {
        if let Ok(year) = date[..4].parse::<i32>() {
            if (1900..=2100).contains(&year) {
                return Some(year);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_values_semicolon() {
        let result = split_values("Rock; Electronic; Ambient");
        assert_eq!(result, vec!["Rock", "Electronic", "Ambient"]);
    }

    #[test]
    fn test_split_values_slash() {
        let result = split_values("Artist One / Artist Two");
        assert_eq!(result, vec!["Artist One", "Artist Two"]);
    }

    #[test]
    fn test_split_values_single() {
        let result = split_values("Just One Value");
        assert_eq!(result, vec!["Just One Value"]);
    }

    #[test]
    fn test_extract_year_simple() {
        assert_eq!(extract_year("2023"), Some(2023));
        assert_eq!(extract_year("1995"), Some(1995));
    }

    #[test]
    fn test_extract_year_date() {
        assert_eq!(extract_year("2023-01-15"), Some(2023));
        assert_eq!(extract_year("2023/01/15"), Some(2023));
    }

    #[test]
    fn test_extract_year_invalid() {
        assert_eq!(extract_year("not a year"), None);
        assert_eq!(extract_year("99"), None);
    }
}
