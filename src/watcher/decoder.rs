//! Audio file decoding using symphonia.
//!
//! Decodes audio files to raw PCM samples for embedding generation.

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, SampleBuffer, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::WatcherError;

/// Result of decoding an audio file
#[derive(Debug)]
pub struct DecodedAudio {
    /// Decoded audio samples as f32 (may be stereo interleaved)
    pub samples: Vec<f32>,
    /// Number of channels in the decoded audio
    pub channels: u8,
    /// Original sample rate of the audio
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration_s: f32,
}

impl DecodedAudio {
    /// Convert stereo to mono by averaging channels
    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.samples.clone();
        }

        if self.channels == 2 {
            return self
                .samples
                .chunks_exact(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect();
        }

        // For more than 2 channels, average all
        self.samples
            .chunks_exact(self.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / self.channels as f32)
            .collect()
    }
}

/// Audio file decoder using symphonia
pub struct AudioDecoder;

impl AudioDecoder {
    /// Decode an audio file to f32 samples
    pub fn decode_file(path: &Path) -> Result<DecodedAudio, WatcherError> {
        // Open the file
        let file = File::open(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                WatcherError::FileNotFound(path.to_path_buf())
            } else {
                WatcherError::IoError(e)
            }
        })?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create a hint based on the file extension
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        // Probe the format
        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| WatcherError::DecodeError(format!("Failed to probe format: {e}")))?;

        let mut format = probed.format;

        // Find the first audio track
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or(WatcherError::NoAudioTrack)?;

        let track_id = track.id;

        // Get sample rate
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or(WatcherError::MissingSampleRate)?;

        // Get number of channels
        let channels = track
            .codec_params
            .channels
            .map(|c| c.count() as u8)
            .unwrap_or(2);

        // Create the decoder
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| WatcherError::DecodeError(format!("Failed to create decoder: {e}")))?;

        // Decode all packets
        let mut samples: Vec<f32> = Vec::new();

        loop {
            match format.next_packet() {
                Ok(packet) => {
                    // Skip packets from other tracks
                    if packet.track_id() != track_id {
                        continue;
                    }

                    // Decode the packet
                    match decoder.decode(&packet) {
                        Ok(decoded) => {
                            // Convert to f32 samples
                            Self::append_samples(&decoded, &mut samples);
                        }
                        Err(SymphoniaError::DecodeError(e)) => {
                            tracing::warn!(error = %e, "Decode error, skipping packet");
                            continue;
                        }
                        Err(e) => {
                            return Err(WatcherError::DecodeError(format!("Decode error: {e}")));
                        }
                    }
                }
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    // End of stream
                    break;
                }
                Err(SymphoniaError::ResetRequired) => {
                    // Reset the decoder and continue
                    decoder.reset();
                    continue;
                }
                Err(e) => {
                    // For other errors, log and try to continue
                    tracing::warn!(error = %e, "Error reading packet, attempting to continue");
                    break;
                }
            }
        }

        if samples.is_empty() {
            return Err(WatcherError::DecodeError(
                "No audio samples decoded".to_string(),
            ));
        }

        // Calculate duration
        let num_samples = samples.len() / channels as usize;
        let duration_s = num_samples as f32 / sample_rate as f32;

        Ok(DecodedAudio {
            samples,
            channels,
            sample_rate,
            duration_s,
        })
    }

    /// Append decoded samples to the output buffer
    fn append_samples(decoded: &AudioBufferRef, output: &mut Vec<f32>) {
        match decoded {
            AudioBufferRef::F32(buf) => {
                // Already f32, just copy interleaved
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
            AudioBufferRef::S16(buf) => {
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
            AudioBufferRef::S24(buf) => {
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
            AudioBufferRef::S32(buf) => {
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
            AudioBufferRef::S8(buf) => {
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
            AudioBufferRef::U8(buf) => {
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
            AudioBufferRef::U16(buf) => {
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
            AudioBufferRef::U24(buf) => {
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
            AudioBufferRef::U32(buf) => {
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
            AudioBufferRef::F64(buf) => {
                let mut sample_buf = SampleBuffer::<f32>::new(buf.frames() as u64, *buf.spec());
                sample_buf.copy_interleaved_ref(decoded.clone());
                output.extend_from_slice(sample_buf.samples());
            }
        }
    }
}
