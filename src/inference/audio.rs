//! Audio preprocessing for CLAP model.
//!
//! CLAP expects mono audio at 48kHz. This module handles:
//! - Format conversion (PCM s16/s24/f32 to f32)
//! - Stereo to mono conversion
//! - Resampling to 48kHz
//! - Windowing for long audio

use rubato::{FftFixedIn, Resampler};
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::InferenceError;

/// Target sample rate for CLAP models
#[allow(dead_code)]
pub const TARGET_SAMPLE_RATE: u32 = 48_000;

/// Audio format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    /// 32-bit float little-endian
    PcmF32Le,
    /// 16-bit signed integer little-endian
    PcmS16Le,
    /// 24-bit signed integer little-endian (packed)
    PcmS24Le,
}

/// Raw audio data with format information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    /// Audio format
    pub format: AudioFormat,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u8,
    /// Raw PCM bytes
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

impl AudioData {
    /// Convert raw bytes to f32 samples based on format
    pub fn to_f32_samples(&self) -> Result<Vec<f32>, InferenceError> {
        match self.format {
            AudioFormat::PcmF32Le => {
                if self.data.len() % 4 != 0 {
                    return Err(InferenceError::InvalidAudioFormat(
                        "F32 data length not multiple of 4".to_string(),
                    ));
                }
                Ok(self
                    .data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
            AudioFormat::PcmS16Le => {
                if self.data.len() % 2 != 0 {
                    return Err(InferenceError::InvalidAudioFormat(
                        "S16 data length not multiple of 2".to_string(),
                    ));
                }
                Ok(self
                    .data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect())
            }
            AudioFormat::PcmS24Le => {
                if self.data.len() % 3 != 0 {
                    return Err(InferenceError::InvalidAudioFormat(
                        "S24 data length not multiple of 3".to_string(),
                    ));
                }
                Ok(self
                    .data
                    .chunks_exact(3)
                    .map(|chunk| {
                        // Sign-extend 24-bit to 32-bit
                        let sample = if chunk[2] & 0x80 != 0 {
                            i32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0xFF])
                        } else {
                            i32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0x00])
                        };
                        sample as f32 / 8_388_608.0 // 2^23
                    })
                    .collect())
            }
        }
    }
}

/// Audio processor for preparing audio for CLAP inference
pub struct AudioProcessor {
    target_sample_rate: u32,
    window_samples: usize,
    hop_samples: usize,
}

impl AudioProcessor {
    /// Create a new audio processor
    ///
    /// # Arguments
    /// * `target_sample_rate` - Target sample rate (typically 48000 for CLAP)
    /// * `window_size_s` - Window size in seconds
    /// * `hop_size_s` - Hop size in seconds (overlap = window - hop)
    pub fn new(target_sample_rate: u32, window_size_s: f32, hop_size_s: f32) -> Self {
        let window_samples = (target_sample_rate as f32 * window_size_s) as usize;
        let hop_samples = (target_sample_rate as f32 * hop_size_s) as usize;

        Self {
            target_sample_rate,
            window_samples,
            hop_samples,
        }
    }

    /// Process audio data into windows suitable for CLAP inference
    ///
    /// Returns a vector of f32 sample windows, each of length `window_samples`
    pub fn process(&mut self, audio: &AudioData) -> Result<Vec<Vec<f32>>, InferenceError> {
        // Convert to f32
        let samples = audio.to_f32_samples()?;
        let original_len = samples.len();

        // Convert to mono if stereo
        let mono = if audio.channels == 2 {
            to_mono(&samples)
        } else if audio.channels == 1 {
            samples
        } else {
            return Err(InferenceError::InvalidAudioFormat(format!(
                "Unsupported channel count: {}",
                audio.channels
            )));
        };

        // Resample if needed
        let resampled = if audio.sample_rate != self.target_sample_rate {
            resample(&mono, audio.sample_rate, self.target_sample_rate)?
        } else {
            mono
        };

        debug!(
            original_samples = original_len,
            mono_samples = resampled.len(),
            window_samples = self.window_samples,
            hop_samples = self.hop_samples,
            "Audio preprocessed"
        );

        // Extract windows
        let windows = extract_windows(&resampled, self.window_samples, self.hop_samples);

        Ok(windows)
    }
}

/// Convert stereo samples to mono by averaging channels
fn to_mono(samples: &[f32]) -> Vec<f32> {
    samples
        .chunks_exact(2)
        .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
        .collect()
}

/// Resample audio using rubato
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, InferenceError> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let ratio = to_rate as f64 / from_rate as f64;

    // Use FFT-based resampler for quality
    let mut resampler = FftFixedIn::<f32>::new(from_rate as usize, to_rate as usize, 1024, 1, 1)
        .map_err(|e| InferenceError::AudioProcessing(format!("Resampler init failed: {e}")))?;

    let mut output = Vec::new();
    let chunk_size = resampler.input_frames_max();

    // Process in chunks
    for chunk in samples.chunks(chunk_size) {
        let mut input = vec![chunk.to_vec()];

        // Pad last chunk if needed
        if chunk.len() < chunk_size {
            input[0].resize(chunk_size, 0.0);
        }

        let resampled = resampler
            .process(&input, None)
            .map_err(|e| InferenceError::AudioProcessing(format!("Resample failed: {e}")))?;

        if !resampled.is_empty() {
            output.extend_from_slice(&resampled[0]);
        }
    }

    // Trim to expected length
    let expected_len = (samples.len() as f64 * ratio) as usize;
    output.truncate(expected_len);

    Ok(output)
}

/// Extract overlapping windows from audio samples
fn extract_windows(samples: &[f32], window_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
    if samples.len() < window_size {
        // Pad short audio to window size
        let mut padded = samples.to_vec();
        padded.resize(window_size, 0.0);
        return vec![padded];
    }

    let mut windows = Vec::new();
    let mut start = 0;

    while start + window_size <= samples.len() {
        windows.push(samples[start..start + window_size].to_vec());
        start += hop_size;
    }

    // Handle remaining samples if any
    if start < samples.len() && windows.is_empty() {
        let mut last_window = samples[start..].to_vec();
        last_window.resize(window_size, 0.0);
        windows.push(last_window);
    }

    windows
}

/// serde_bytes helper for efficient byte serialization
mod serde_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(bytes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: &[u8] = Deserialize::deserialize(deserializer)?;
        Ok(bytes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_mono() {
        let stereo = vec![1.0, 0.5, 0.8, 0.2, 0.6, 0.4];
        let mono = to_mono(&stereo);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.75).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
        assert!((mono[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_extract_windows_exact() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let windows = extract_windows(&samples, 25, 25);
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0].len(), 25);
    }

    #[test]
    fn test_extract_windows_overlap() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let windows = extract_windows(&samples, 30, 20);
        // Windows: 0-30, 20-50, 40-70, 60-90
        assert_eq!(windows.len(), 4);
    }

    #[test]
    fn test_extract_windows_short() {
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let windows = extract_windows(&samples, 25, 25);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].len(), 25);
        // Should be zero-padded
        assert_eq!(windows[0][10], 0.0);
    }

    #[test]
    fn test_audio_data_s16_to_f32() {
        let audio = AudioData {
            format: AudioFormat::PcmS16Le,
            sample_rate: 44100,
            channels: 1,
            data: vec![0x00, 0x40], // 16384 in little-endian = 0.5
        };
        let samples = audio.to_f32_samples().unwrap();
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_audio_data_f32_to_f32() {
        let value: f32 = 0.75;
        let bytes = value.to_le_bytes();
        let audio = AudioData {
            format: AudioFormat::PcmF32Le,
            sample_rate: 48_000,
            channels: 1,
            data: bytes.to_vec(),
        };
        let samples = audio.to_f32_samples().unwrap();
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.75).abs() < 1e-6);
    }
}
