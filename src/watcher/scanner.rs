//! Directory scanning for initial/full scans.

use std::path::PathBuf;

use walkdir::WalkDir;

use super::config::WatchedFolder;
use super::state::FileRegistry;
use super::AUDIO_EXTENSIONS;

/// A scanned file with its base path for relative ID generation
#[derive(Debug, Clone)]
pub struct ScannedFile {
    /// Absolute path to the audio file
    pub path: PathBuf,
    /// Base path of the watched folder (for generating relative track IDs)
    pub base_path: PathBuf,
}

/// Folder scanner for discovering audio files
pub struct FolderScanner {
    /// Allowed file extensions (empty = all supported)
    extensions: Vec<String>,
}

impl FolderScanner {
    /// Create a new folder scanner
    pub fn new(extensions: Vec<String>) -> Self {
        Self { extensions }
    }

    /// Scan all configured folders and return files to process
    pub fn scan_all(
        &self,
        folders: &[WatchedFolder],
        registry: &FileRegistry,
        force_rescan: bool,
    ) -> Vec<ScannedFile> {
        let mut files_to_process = Vec::new();

        for folder in folders.iter().filter(|f| f.enabled) {
            let path = std::path::Path::new(&folder.path);
            if !path.exists() {
                tracing::warn!(path = %folder.path, "Watched folder does not exist");
                continue;
            }

            if !path.is_dir() {
                tracing::warn!(path = %folder.path, "Watched path is not a directory");
                continue;
            }

            let files = self.scan_folder(folder, registry, force_rescan);
            files_to_process.extend(files);
        }

        files_to_process
    }

    /// Scan a single folder and return files to process
    pub fn scan_folder(
        &self,
        folder: &WatchedFolder,
        registry: &FileRegistry,
        force_rescan: bool,
    ) -> Vec<ScannedFile> {
        let path = std::path::Path::new(&folder.path);
        let base_path = path.to_path_buf();
        let mut files = Vec::new();

        let walker = if folder.recursive {
            WalkDir::new(path)
        } else {
            WalkDir::new(path).max_depth(1)
        };

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            let entry_path = entry.path();

            if !self.is_supported_file(entry_path) {
                continue;
            }

            let path_buf = entry_path.to_path_buf();

            // Skip unchanged files unless force_rescan
            if !force_rescan && !registry.is_file_changed(&path_buf) {
                continue;
            }

            files.push(ScannedFile {
                path: path_buf,
                base_path: base_path.clone(),
            });
        }

        tracing::info!(
            folder = %folder.path,
            file_count = files.len(),
            "Scanned folder"
        );

        files
    }

    /// Count total audio files in all folders (without filtering by registry)
    pub fn count_files(&self, folders: &[WatchedFolder]) -> usize {
        let mut count = 0;

        for folder in folders.iter().filter(|f| f.enabled) {
            let path = std::path::Path::new(&folder.path);
            if !path.exists() || !path.is_dir() {
                continue;
            }

            let walker = if folder.recursive {
                WalkDir::new(path)
            } else {
                WalkDir::new(path).max_depth(1)
            };

            count += walker
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| self.is_supported_file(e.path()))
                .count();
        }

        count
    }

    /// Check if a file has a supported audio extension and exists
    fn is_supported_file(&self, path: &std::path::Path) -> bool {
        if !path.is_file() {
            return false;
        }
        self.has_audio_extension(path)
    }

    /// Check if a path has a supported audio extension (without checking if file exists)
    pub fn has_audio_extension(&self, path: &std::path::Path) -> bool {
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => return false,
        };

        if self.extensions.is_empty() {
            // Use default audio extensions
            AUDIO_EXTENSIONS.contains(&ext.as_str())
        } else {
            // Use configured extensions
            self.extensions.iter().any(|e| e.to_lowercase() == ext)
        }
    }
}

impl Default for FolderScanner {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported_file_default() {
        let scanner = FolderScanner::new(Vec::new());

        assert!(scanner.has_audio_extension(std::path::Path::new("/music/song.mp3")));
        assert!(scanner.has_audio_extension(std::path::Path::new("/music/song.FLAC")));
        assert!(scanner.has_audio_extension(std::path::Path::new("/music/song.wav")));
        assert!(!scanner.has_audio_extension(std::path::Path::new("/music/doc.pdf")));
    }

    #[test]
    fn test_is_supported_file_custom() {
        let scanner = FolderScanner::new(vec!["flac".to_string(), "wav".to_string()]);

        assert!(!scanner.has_audio_extension(std::path::Path::new("/music/song.mp3")));
        assert!(scanner.has_audio_extension(std::path::Path::new("/music/song.flac")));
        assert!(scanner.has_audio_extension(std::path::Path::new("/music/song.wav")));
    }
}
