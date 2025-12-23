//! CLAP model wrapper for ONNX Runtime inference.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tracing::{debug, info};

use super::{AudioData, AudioProcessor, Embedding, InferenceError, ModelPaths, EMBEDDING_DIM};

/// Device type for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda => write!(f, "CUDA"),
        }
    }
}

/// CLAP model for generating text and audio embeddings
pub struct ClapModel {
    text_session: Mutex<Session>,
    audio_session: Mutex<Session>,
    audio_processor: Mutex<AudioProcessor>,
    device: Device,
}

impl std::fmt::Debug for ClapModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClapModel")
            .field("device", &self.device)
            .field("embedding_dim", &EMBEDDING_DIM)
            .finish()
    }
}

impl ClapModel {
    /// Load CLAP model from downloaded paths
    pub fn load(paths: &ModelPaths, use_cuda: bool) -> Result<Self, InferenceError> {
        let device = if use_cuda { Device::Cuda } else { Device::Cpu };

        info!(?device, "Loading CLAP model");

        let text_session = Self::create_session(&paths.text_model, use_cuda)?;
        let audio_session = Self::create_session(&paths.audio_model, use_cuda)?;

        // Log model info
        debug!(
            text_inputs = ?text_session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>(),
            text_outputs = ?text_session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>(),
            "Text model loaded"
        );
        debug!(
            audio_inputs = ?audio_session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>(),
            audio_outputs = ?audio_session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>(),
            "Audio model loaded"
        );

        let audio_processor = AudioProcessor::new(48000, 10.0, 10.0);

        Ok(Self {
            text_session: Mutex::new(text_session),
            audio_session: Mutex::new(audio_session),
            audio_processor: Mutex::new(audio_processor),
            device,
        })
    }

    fn create_session(model_path: &Path, use_cuda: bool) -> Result<Session, InferenceError> {
        // Read model bytes from file
        let model_bytes = std::fs::read(model_path)
            .map_err(|e| InferenceError::Onnx(format!("Failed to read model file: {e}")))?;

        let mut builder = Session::builder()
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        builder = builder
            .with_intra_threads(4)
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        if use_cuda {
            #[cfg(feature = "cuda")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                builder = builder
                    .with_execution_providers([CUDAExecutionProvider::default().build()])
                    .map_err(|e| InferenceError::Onnx(e.to_string()))?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA requested but not compiled with cuda feature, using CPU");
            }
        }

        builder
            .commit_from_memory(&model_bytes)
            .map_err(|e| InferenceError::Onnx(format!("Failed to load model: {e}")))
    }

    /// Get the device being used for inference
    pub fn device(&self) -> Device {
        self.device
    }

    /// Generate text embedding from input text
    pub fn text_embedding(&self, text: &str) -> Result<Embedding, InferenceError> {
        let outputs = self.run_text_inference(text)?;
        let mut embedding = Embedding::from_raw(outputs);
        embedding.normalize();
        Ok(embedding)
    }

    fn run_text_inference(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        // CLAP models typically expect tokenized input
        // Create input_ids as i64 array (simple word-based tokenization placeholder)
        let tokens: Vec<i64> = text
            .split_whitespace()
            .take(77) // CLAP max sequence length
            .enumerate()
            .map(|(i, _)| i as i64 + 1)
            .collect();

        let seq_len = tokens.len().max(1);
        let mut padded_tokens = vec![0i64; 77];
        for (i, &t) in tokens.iter().enumerate() {
            padded_tokens[i] = t;
        }

        // Create tensors using Tensor::from_array with shape tuple
        let input_ids = Tensor::from_array(([1usize, 77], padded_tokens.into_boxed_slice()))
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        // Attention mask
        let mut attention_mask_data = vec![0i64; 77];
        for item in attention_mask_data.iter_mut().take(seq_len) {
            *item = 1;
        }
        let attention_mask =
            Tensor::from_array(([1usize, 77], attention_mask_data.into_boxed_slice()))
                .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        // Lock session for inference
        let mut session = self
            .text_session
            .lock()
            .map_err(|e| InferenceError::Onnx(format!("Session lock error: {e}")))?;

        // Get output name before running (to avoid borrow conflicts)
        let output_name = session.outputs[0].name.clone();

        // Run inference
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask
            ])
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        // Get first output (embedding)
        let output = outputs
            .get(output_name.as_str())
            .ok_or_else(|| InferenceError::Onnx(format!("Output '{}' not found", output_name)))?;

        // Extract tensor data - returns (shape, data_slice)
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        debug!(?shape, data_len = data.len(), "Text model output");

        // Take first EMBEDDING_DIM values
        let result: Vec<f32> = data.iter().copied().take(EMBEDDING_DIM).collect();

        if result.len() < EMBEDDING_DIM {
            return Err(InferenceError::Onnx(format!(
                "Output too small: expected {}, got {}",
                EMBEDDING_DIM,
                result.len()
            )));
        }

        Ok(result)
    }

    /// Generate audio embedding from audio data
    pub fn audio_embedding(&self, audio: &AudioData) -> Result<Embedding, InferenceError> {
        let mut processor = self
            .audio_processor
            .lock()
            .map_err(|e| InferenceError::AudioProcessing(e.to_string()))?;

        // Preprocess audio to windows
        let windows = processor.process(audio)?;

        if windows.is_empty() {
            return Err(InferenceError::AudioProcessing(
                "No audio windows extracted".to_string(),
            ));
        }

        // Generate embedding for each window and average
        let mut embeddings: Vec<Vec<f32>> = Vec::new();

        for window in &windows {
            let output = self.run_audio_inference(window)?;
            embeddings.push(output);
        }

        // Average embeddings
        let mut averaged = vec![0.0f32; EMBEDDING_DIM];
        for emb in &embeddings {
            for (i, &v) in emb.iter().enumerate() {
                averaged[i] += v;
            }
        }
        let count = embeddings.len() as f32;
        for v in &mut averaged {
            *v /= count;
        }

        let mut embedding = Embedding::from_raw(averaged);
        embedding.normalize();

        Ok(embedding)
    }

    fn run_audio_inference(&self, samples: &[f32]) -> Result<Vec<f32>, InferenceError> {
        // CLAP audio models expect [batch, samples]
        let input =
            Tensor::from_array(([1usize, samples.len()], samples.to_vec().into_boxed_slice()))
                .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        // Lock session for inference
        let mut session = self
            .audio_session
            .lock()
            .map_err(|e| InferenceError::Onnx(format!("Session lock error: {e}")))?;

        // Get output name before running (to avoid borrow conflicts)
        let output_name = session.outputs[0].name.clone();

        let outputs = session
            .run(ort::inputs![
                "input_values" => input
            ])
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        // Get first output
        let output = outputs
            .get(output_name.as_str())
            .ok_or_else(|| InferenceError::Onnx(format!("Output '{}' not found", output_name)))?;

        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        debug!(?shape, data_len = data.len(), "Audio model output");

        let result: Vec<f32> = data.iter().copied().take(EMBEDDING_DIM).collect();

        if result.len() < EMBEDDING_DIM {
            return Err(InferenceError::Onnx(format!(
                "Audio output too small: expected {}, got {}",
                EMBEDDING_DIM,
                result.len()
            )));
        }

        Ok(result)
    }
}

/// Thread-safe wrapper for ClapModel
#[allow(dead_code)]
pub type SharedClapModel = Arc<ClapModel>;
