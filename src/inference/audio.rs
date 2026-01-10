//! Audio preprocessing for CLAP model.
//!
//! CLAP expects mel spectrogram features at 48kHz. This module handles:
//! - Format conversion (PCM s16/s24/f32 to f32)
//! - Stereo to mono conversion
//! - Resampling to 48kHz
//! - Mel spectrogram computation
//! - Windowing for long audio

use mel_spec::mel::mel;
use mel_spec::stft::Spectrogram;
use rubato::{FftFixedIn, Resampler};
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::InferenceError;

/// Target sample rate for CLAP models
pub const TARGET_SAMPLE_RATE: u32 = 48_000;

/// CLAP mel spectrogram parameters (from ClapFeatureExtractor)
pub const CLAP_N_FFT: usize = 1024;
pub const CLAP_HOP_LENGTH: usize = 480;
pub const CLAP_N_MELS: usize = 64;
pub const CLAP_F_MIN: f32 = 0.0;
pub const CLAP_F_MAX: f32 = 14_000.0;
/// CLAP spectrogram size (time dimension) that the HTSAT model expects
pub const CLAP_SPEC_SIZE: usize = 256;

/// Embedding dimension for CLAP models (512-d vectors)
pub const EMBEDDING_DIM: usize = 512;

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

/// Convert raw PCM bytes to f32 samples based on format.
///
/// This is the shared implementation used by both `AudioData::to_f32_samples`
/// and streaming audio processing.
pub fn pcm_to_f32(data: &[u8], format: AudioFormat) -> Result<Vec<f32>, InferenceError> {
    match format {
        AudioFormat::PcmF32Le => {
            if data.len() % 4 != 0 {
                return Err(InferenceError::InvalidAudioFormat(
                    "F32 data length not multiple of 4".to_string(),
                ));
            }
            Ok(data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        AudioFormat::PcmS16Le => {
            if data.len() % 2 != 0 {
                return Err(InferenceError::InvalidAudioFormat(
                    "S16 data length not multiple of 2".to_string(),
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect())
        }
        AudioFormat::PcmS24Le => {
            if data.len() % 3 != 0 {
                return Err(InferenceError::InvalidAudioFormat(
                    "S24 data length not multiple of 3".to_string(),
                ));
            }
            Ok(data
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
        pcm_to_f32(&self.data, self.format)
    }
}

/// Mel spectrogram features for a single audio window
/// Shape: [spec_size (height), n_mels (width)] flattened to 1D in row-major order
/// This matches CLAP's expected input: [batch, 1, height, width] = [1, 1, spec_size, n_mels]
#[derive(Debug, Clone)]
pub struct MelFeatures {
    /// Flattened mel spectrogram data in row-major order [spec_size, n_mels]
    pub data: Vec<f32>,
    /// Spectrogram height (time frames, resized to CLAP_SPEC_SIZE=256)
    pub height: usize,
    /// Spectrogram width (mel bins = 64)
    pub width: usize,
}

/// Audio processor for preparing audio for CLAP inference
pub struct AudioProcessor {
    target_sample_rate: u32,
    window_samples: usize,
    hop_samples: usize,
    mel_filterbank: ndarray::Array2<f32>,
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

        // Create mel filterbank using mel_spec crate
        // mel() returns Array2<f64>, we convert to f32
        let mel_fb_f64 = mel(
            target_sample_rate as f64,
            CLAP_N_FFT,
            CLAP_N_MELS,
            Some(CLAP_F_MIN as f64),
            Some(CLAP_F_MAX as f64),
            false, // htk-style
            true,  // normalize (slaney style)
        );
        let mel_filterbank = mel_fb_f64.mapv(|x| x as f32);

        debug!(
            n_mels = CLAP_N_MELS,
            n_fft = CLAP_N_FFT,
            hop_length = CLAP_HOP_LENGTH,
            filterbank_shape = ?mel_filterbank.shape(),
            "Mel filterbank created"
        );

        Self {
            target_sample_rate,
            window_samples,
            hop_samples,
            mel_filterbank,
        }
    }

    /// Process audio data into mel spectrogram windows suitable for CLAP inference
    ///
    /// Returns a vector of MelFeatures, each representing a time window's mel spectrogram
    pub fn process(&mut self, audio: &AudioData) -> Result<Vec<MelFeatures>, InferenceError> {
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

        // Extract audio windows
        let windows = extract_windows(&resampled, self.window_samples, self.hop_samples);

        // Convert each window to mel spectrogram
        let mut mel_features = Vec::with_capacity(windows.len());
        for window in windows {
            let mel = self.compute_mel(&window)?;
            mel_features.push(mel);
        }

        Ok(mel_features)
    }

    /// Compute mel spectrogram for a single audio window
    fn compute_mel(&self, samples: &[f32]) -> Result<MelFeatures, InferenceError> {
        compute_mel_spectrogram(samples, &self.mel_filterbank)
    }
}

/// Compute mel spectrogram for a single audio window.
///
/// This is the shared implementation used by both `AudioProcessor` and streaming.
///
/// # Arguments
/// * `samples` - Audio samples at 48kHz
/// * `mel_filterbank` - Pre-computed mel filterbank of shape [n_mels, n_fft/2+1]
///
/// # Returns
/// `MelFeatures` with shape [CLAP_SPEC_SIZE, CLAP_N_MELS] ready for CLAP inference
pub fn compute_mel_spectrogram(
    samples: &[f32],
    mel_filterbank: &ndarray::Array2<f32>,
) -> Result<MelFeatures, InferenceError> {
    // Create STFT processor
    let mut stft = Spectrogram::new(CLAP_N_FFT, CLAP_HOP_LENGTH);

    // Collect all STFT frames as power spectra
    let mut power_frames: Vec<Vec<f32>> = Vec::new();

    for chunk in samples.chunks(CLAP_HOP_LENGTH) {
        if let Some(fft_result) = stft.add(chunk) {
            // Compute power spectrum (magnitude squared)
            let power: Vec<f32> = fft_result
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im) as f32)
                .collect();
            power_frames.push(power);
        }
    }

    if power_frames.is_empty() {
        return Err(InferenceError::AudioProcessing(
            "No STFT frames produced".to_string(),
        ));
    }

    // Apply mel filterbank to each power spectrum frame
    let n_freqs = CLAP_N_FFT / 2 + 1;
    let time_frames = power_frames.len();

    // Build mel spectrogram: [n_mels, time_frames]
    // Use checked arithmetic to prevent overflow on extremely long audio
    let mel_spec_size = CLAP_N_MELS
        .checked_mul(time_frames)
        .ok_or_else(|| InferenceError::AudioProcessing("Mel spectrogram size overflow".to_string()))?;
    let mut mel_spec_raw = vec![0.0f32; mel_spec_size];

    for (t, power_frame) in power_frames.iter().enumerate() {
        let frame_len = power_frame.len().min(n_freqs);

        for m in 0..CLAP_N_MELS {
            let mut sum = 0.0f32;
            for f in 0..frame_len {
                sum += mel_filterbank[[m, f]] * power_frame[f];
            }
            // Apply log scaling. Index calculation uses checked arithmetic for safety.
            let idx = m
                .checked_mul(time_frames)
                .and_then(|v| v.checked_add(t))
                .ok_or_else(|| InferenceError::AudioProcessing("Mel spectrogram index overflow".to_string()))?;
            mel_spec_raw[idx] = (sum + 1e-10).ln();
        }
    }

    // Resize to [CLAP_SPEC_SIZE, CLAP_N_MELS] for CLAP model
    let resized = resize_mel_spectrogram(&mel_spec_raw, CLAP_N_MELS, time_frames, CLAP_SPEC_SIZE);

    Ok(MelFeatures {
        data: resized,
        height: CLAP_SPEC_SIZE,
        width: CLAP_N_MELS,
    })
}

/// Resize mel spectrogram from [n_mels, src_time] to [dst_time, n_mels]
/// This transposes and resizes in one pass using linear interpolation along time axis
pub fn resize_mel_spectrogram(
    data: &[f32],
    n_mels: usize,
    src_time: usize,
    dst_time: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; dst_time * n_mels];

    // For each output time frame
    for t_out in 0..dst_time {
        // Map to source time position (linear interpolation)
        let t_src = if dst_time > 1 {
            (t_out as f32) * ((src_time - 1) as f32) / ((dst_time - 1) as f32)
        } else {
            0.0
        };

        let t_low = t_src.floor() as usize;
        let t_high = (t_low + 1).min(src_time - 1);
        let t_frac = t_src - t_low as f32;

        // Interpolate each mel bin for this time frame
        for m in 0..n_mels {
            // Source is [n_mels, src_time]: src[m, t] = data[m * src_time + t]
            let val_low = data[m * src_time + t_low];
            let val_high = data[m * src_time + t_high];
            let interpolated = val_low * (1.0 - t_frac) + val_high * t_frac;

            // Output is [dst_time, n_mels]: result[t, m] = result[t * n_mels + m]
            result[t_out * n_mels + m] = interpolated;
        }
    }

    result
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
