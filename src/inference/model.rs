//! CLAP model wrapper for ONNX Runtime inference.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use super::{AudioData, AudioProcessor, Embedding, InferenceError, ModelPaths, EMBEDDING_DIM};

/// Maximum sequence length for CLAP text models
const MAX_SEQ_LENGTH: usize = 77;

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
    tokenizer: Option<Tokenizer>,
    device: Device,
}

impl std::fmt::Debug for ClapModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClapModel")
            .field("device", &self.device)
            .field("embedding_dim", &EMBEDDING_DIM)
            .field("has_tokenizer", &self.tokenizer.is_some())
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

        // Load tokenizer if available
        let tokenizer = if let Some(ref tokenizer_path) = paths.tokenizer_config {
            match Tokenizer::from_file(tokenizer_path) {
                Ok(tok) => {
                    info!(?tokenizer_path, "Tokenizer loaded");
                    Some(tok)
                }
                Err(e) => {
                    warn!(?tokenizer_path, error = %e, "Failed to load tokenizer, using fallback");
                    None
                }
            }
        } else {
            warn!("No tokenizer path provided, using fallback tokenization");
            None
        };

        let audio_processor = AudioProcessor::new(48000, 10.0, 10.0);

        Ok(Self {
            text_session: Mutex::new(text_session),
            audio_session: Mutex::new(audio_session),
            audio_processor: Mutex::new(audio_processor),
            tokenizer,
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

    /// Tokenize text using the loaded tokenizer or fallback
    fn tokenize_text(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>), InferenceError> {
        if let Some(ref tokenizer) = self.tokenizer {
            // Use the real tokenizer
            let encoding = tokenizer
                .encode(text, true)
                .map_err(|e| InferenceError::Onnx(format!("Tokenization failed: {e}")))?;

            let ids = encoding.get_ids();
            let attention = encoding.get_attention_mask();

            // Truncate or pad to MAX_SEQ_LENGTH
            let mut input_ids = vec![0i64; MAX_SEQ_LENGTH];
            let mut attention_mask = vec![0i64; MAX_SEQ_LENGTH];

            let len = ids.len().min(MAX_SEQ_LENGTH);
            for i in 0..len {
                input_ids[i] = ids[i] as i64;
                attention_mask[i] = attention[i] as i64;
            }

            debug!(
                text_len = text.len(),
                token_count = ids.len(),
                truncated_to = len,
                "Text tokenized"
            );

            Ok((input_ids, attention_mask))
        } else {
            // Fallback: simple word-based tokenization (not recommended for production)
            // This produces placeholder IDs that won't give meaningful embeddings
            warn!("Using fallback tokenization - embeddings may not be meaningful");

            let tokens: Vec<i64> = text
                .split_whitespace()
                .take(MAX_SEQ_LENGTH - 2) // Leave room for special tokens
                .enumerate()
                .map(|(i, _)| i as i64 + 1)
                .collect();

            let mut input_ids = vec![0i64; MAX_SEQ_LENGTH];
            let mut attention_mask = vec![0i64; MAX_SEQ_LENGTH];

            // Add CLS token at start (typically ID 101 for BERT-like models)
            input_ids[0] = 101;
            attention_mask[0] = 1;

            // Add word tokens
            for (i, &t) in tokens.iter().enumerate() {
                input_ids[i + 1] = t;
                attention_mask[i + 1] = 1;
            }

            // Add SEP token at end (typically ID 102 for BERT-like models)
            input_ids[tokens.len() + 1] = 102;
            attention_mask[tokens.len() + 1] = 1;

            Ok((input_ids, attention_mask))
        }
    }

    fn run_text_inference(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        // Tokenize the input text
        let (input_ids_data, attention_mask_data) = self.tokenize_text(text)?;

        // Create tensors using Tensor::from_array with shape tuple
        let input_ids =
            Tensor::from_array(([1usize, MAX_SEQ_LENGTH], input_ids_data.into_boxed_slice()))
                .map_err(|e| InferenceError::Onnx(e.to_string()))?;

        let attention_mask =
            Tensor::from_array(([1usize, MAX_SEQ_LENGTH], attention_mask_data.into_boxed_slice()))
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
